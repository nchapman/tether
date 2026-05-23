//! Cursor overlay pass.
//!
//! Renders the host's mouse cursor as an alpha-blended sprite quad on
//! top of the video. Lives behind a shared `Arc<Mutex<CursorState>>`
//! so the client's wire-receive task can update sprite cache, active
//! id, and position from one thread while the renderer reads them on
//! its event-loop thread.
//!
//! ## Why out-of-band
//!
//! With the cursor extracted at the host before encoding, idle frames
//! become byte-identical when only the cursor moved. The capture
//! backend's `SPA_META_VideoDamage` / `SCStreamFrameInfo.status`
//! reports `idle`; `HashDamage::classify` returns `Unchanged`; the host
//! drops the frame. The wire then carries ~80-byte
//! `HostCursorPacket::Position` datagrams instead of full P-frames
//! for cursor-only motion.
//!
//! ## Coordinate frame
//!
//! Host sends cursor position in *captured frame* pixels and sprite
//! size in physical pixels. The client transforms through the same
//! letterbox the blit pass uses — sprite stays at 1:1 capture-pixel
//! scale relative to the video rect, so it doesn't bloat on upscaled
//! windows. Hotspot is per-sprite (cached with the texture), so it's
//! subtracted at draw time, not on the wire.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

/// Maximum sprites cached on the client. A real session uses ~5–10
/// distinct cursors (arrow, i-beam, hand, resize-*, wait). The cap is
/// purely defense against a buggy or malicious host emitting unique
/// ids forever — 32 RGBA8 sprites ≤ 64×64 each is ~512 KiB GPU.
pub(crate) const CURSOR_CACHE_MAX: usize = 32;

/// Sprite + hotspot, kept on the client until evicted or replaced.
struct CachedSprite {
    // Held for RAII: dropping it frees the GPU memory the `view`
    // borrows.
    #[allow(dead_code)]
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    hotspot_x: u32,
    hotspot_y: u32,
    /// Monotonic counter snapshot at last access; smallest = LRU
    /// eviction target. Cheaper than a doubly-linked list for the
    /// expected size.
    last_used: u64,
}

/// Pending shape upload, queued by the wire-receive task. The
/// renderer drains these at the start of each frame because texture
/// uploads need the wgpu device, which only the renderer thread can
/// touch. The queue is bounded indirectly: a misbehaving host can
/// flood it, but the LRU cache evict on insert keeps the GPU side
/// bounded once the renderer drains.
struct PendingShape {
    id: u64,
    width: u32,
    height: u32,
    hotspot_x: u32,
    hotspot_y: u32,
    pixels: Vec<u8>,
}

/// Shared cursor state. Written by the wire-receive task; read by the
/// renderer.
pub struct CursorState {
    cache: HashMap<u64, CachedSprite>,
    /// Pending shapes that arrived on the wire but haven't been
    /// uploaded to a GPU texture yet (only the renderer's thread can
    /// do that). Drained at the start of each render pass.
    pending_shapes: Vec<PendingShape>,
    /// Currently-active sprite id (set by `CursorUseShape` after an
    /// `upload_cursor_shape`). `None` until the host names one.
    active: Option<u64>,
    /// Latest cursor position in *capture-pixel* coordinates. The
    /// renderer applies the letterbox transform at draw time.
    position_x: f32,
    position_y: f32,
    /// `false` whenever the host reports no cursor (text input field
    /// hides it, app set NSCursor::hide, etc.) or `CursorMode::Relative`
    /// is active on the client.
    visible: bool,
    /// Cursor input mode. `Relative` suppresses overlay drawing — the
    /// client is rendering its own locked pointer.
    relative_mode: bool,
    /// Bumped on every cache touch; used as `CachedSprite::last_used`.
    access_counter: u64,
}

impl CursorState {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            pending_shapes: Vec::new(),
            active: None,
            position_x: 0.0,
            position_y: 0.0,
            visible: false,
            relative_mode: false,
            access_counter: 0,
        }
    }

    /// Wire-side entry point: enqueue a new sprite for the renderer
    /// to upload on its next frame. Caller is responsible for
    /// ensuring `pixels.len() == width * height * 4` (RGBA8); the
    /// renderer drops with a warn if not.
    ///
    /// Capped at [`CURSOR_CACHE_MAX`] pending entries — without the
    /// cap, a minimised window (renderer paused → never drains) could
    /// accumulate full sprite buffers indefinitely. The GPU side is
    /// already bounded by the LRU cache; the CPU side needs the same
    /// limit. Drops oldest when over cap.
    pub fn enqueue_shape(
        &mut self,
        id: u64,
        width: u32,
        height: u32,
        hotspot_x: u32,
        hotspot_y: u32,
        pixels: Vec<u8>,
    ) {
        while self.pending_shapes.len() >= CURSOR_CACHE_MAX {
            self.pending_shapes.remove(0);
        }
        self.pending_shapes.push(PendingShape {
            id,
            width,
            height,
            hotspot_x,
            hotspot_y,
            pixels,
        });
    }

    /// Drain pending shape uploads. Called by the renderer at the
    /// start of each cursor pass.
    fn drain_pending(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let pending = std::mem::take(&mut self.pending_shapes);
        for p in pending {
            self.upload_shape_now(device, queue, &p);
        }
    }

    fn upload_shape_now(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        p: &PendingShape,
    ) {
        let (id, width, height, hotspot_x, hotspot_y) =
            (p.id, p.width, p.height, p.hotspot_x, p.hotspot_y);
        if width == 0 || height == 0 {
            return;
        }
        let expected_bytes = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        if p.pixels.len() != expected_bytes {
            tracing::warn!(
                id,
                width,
                height,
                got = p.pixels.len(),
                expected = expected_bytes,
                "cursor shape pixel buffer size mismatch; dropping",
            );
            return;
        }
        // LRU evict before insertion when at cap and id is new.
        if !self.cache.contains_key(&id) && self.cache.len() >= CURSOR_CACHE_MAX {
            if let Some(&oldest_id) = self
                .cache
                .iter()
                .min_by_key(|(_, s)| s.last_used)
                .map(|(k, _)| k)
            {
                self.cache.remove(&oldest_id);
            }
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tether-render cursor sprite"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &p.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.access_counter += 1;
        self.cache.insert(
            id,
            CachedSprite {
                texture,
                view,
                width,
                height,
                hotspot_x,
                hotspot_y,
                last_used: self.access_counter,
            },
        );
    }

    /// Activate a previously-uploaded sprite. Silently no-ops when the
    /// id isn't cached — that's the "host advertised a `UseShape` we
    /// never got the `Shape` for" path (lost reliable control message
    /// reorder, etc.). The next position update will simply skip the
    /// draw.
    pub fn activate(&mut self, id: u64) {
        if let Some(sprite) = self.cache.get_mut(&id) {
            self.access_counter += 1;
            sprite.last_used = self.access_counter;
            self.active = Some(id);
        } else {
            tracing::debug!(id, "host activated cursor shape not in cache");
        }
    }

    /// Update position + visibility. Wire-task entry point for
    /// `HostCursorPacket::Position`.
    pub fn set_position(&mut self, x: f32, y: f32, visible: bool) {
        self.position_x = x;
        self.position_y = y;
        self.visible = visible;
    }

    /// Suppress overlay draw while the local pointer is in
    /// relative-locked mode.
    pub fn set_relative_mode(&mut self, relative: bool) {
        self.relative_mode = relative;
    }

    /// Renderer-side snapshot: returns the active sprite reference +
    /// position uniform inputs if the overlay should draw this frame.
    fn snapshot_for_render(&self) -> Option<RenderSnapshot<'_>> {
        if self.relative_mode || !self.visible {
            return None;
        }
        let id = self.active?;
        let sprite = self.cache.get(&id)?;
        Some(RenderSnapshot {
            view: &sprite.view,
            width: sprite.width,
            height: sprite.height,
            hotspot_x: sprite.hotspot_x,
            hotspot_y: sprite.hotspot_y,
            position_x: self.position_x,
            position_y: self.position_y,
        })
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn cache_len(&self) -> usize {
        self.cache.len()
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn contains(&self, id: u64) -> bool {
        self.cache.contains_key(&id)
    }
}

impl Default for CursorState {
    fn default() -> Self {
        Self::new()
    }
}

/// `Arc<Mutex<CursorState>>` newtype — same shape as `LatestFrame`.
/// Exists so the client can hand one handle to the wire-receive task
/// and another to `tether_render::run`.
#[derive(Clone, Default)]
pub struct CursorChannel(Arc<Mutex<CursorState>>);

impl CursorChannel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the underlying state with a write lock. Used by both
    /// wire-side updaters and the renderer's per-frame snapshot.
    pub fn with<R>(&self, f: impl FnOnce(&mut CursorState) -> R) -> R {
        let mut guard = self.0.lock().expect("CursorChannel mutex poisoned");
        f(&mut *guard)
    }
}

struct RenderSnapshot<'a> {
    view: &'a wgpu::TextureView,
    width: u32,
    height: u32,
    hotspot_x: u32,
    hotspot_y: u32,
    position_x: f32,
    position_y: f32,
}

/// 4 × vec4<f32> packed for the cursor.wgsl uniform. Hand-rolled
/// little-endian bytes to avoid pulling in a `bytemuck` dependency
/// just for this struct.
fn cursor_uniform_bytes(rows: &[[f32; 4]; 4]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            let b = v.to_le_bytes();
            let off = i * 16 + j * 4;
            out[off..off + 4].copy_from_slice(&b);
        }
    }
    out
}

const CURSOR_UNIFORM_BYTES: u64 = 64;

/// GPU-side cursor overlay: pipeline, sampler, bind group layout, and
/// the per-frame uniform buffer the vertex shader consumes.
pub(crate) struct CursorOverlay {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
}

impl CursorOverlay {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tether-render cursor bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(CURSOR_UNIFORM_BYTES),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tether-render cursor pipeline layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tether-render cursor shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("cursor.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tether-render cursor pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    // Straight-alpha sprite over the already-written
                    // video pixels. The blit pass clears to black and
                    // writes opaque video; this pass loads + alpha-
                    // blends on top.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tether-render cursor sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tether-render cursor uniform"),
            size: CURSOR_UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bgl,
            sampler,
            uniform,
        }
    }

    /// Run the cursor pass for one frame. No-ops when the cursor isn't
    /// visible or no sprite is active — the renderer's hot path pays
    /// one mutex lock + one HashMap lookup in those cases, which is
    /// cheap relative to the rest of the frame.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_view: &wgpu::TextureView,
        channel: &CursorChannel,
        video_dims: (u32, u32),
        surface_dims: (u32, u32),
        fit_dims: (u32, u32),
    ) {
        let (vw, vh) = video_dims;
        let (surf_w, surf_h) = surface_dims;
        let (fit_w, fit_h) = fit_dims;
        if surf_w == 0 || surf_h == 0 || vw == 0 || vh == 0 {
            return;
        }
        // Build the bind group *inside* the cursor-state lock so the
        // resulting `wgpu::BindGroup` holds its references to the
        // sprite's texture view before the cache can possibly evict
        // the underlying texture. A `wgpu::TextureView` does not by
        // itself keep its source `Texture` alive — without this
        // ordering, a follow-up `enqueue_shape` that races eviction
        // could drop the texture between snapshot and submit, leaving
        // the bind group dangling.
        let bind_group = channel.with(|state| {
            state.drain_pending(device, queue);
            let snap = state.snapshot_for_render()?;
            // Letterbox rect origin inside the window — same math the
            // blit pass implicitly does via NDC scale, but we need
            // explicit pixel-space anchors here for the sprite quad.
            #[allow(clippy::cast_precision_loss)]
            let rect_x = (surf_w as f32 - fit_w as f32) * 0.5;
            #[allow(clippy::cast_precision_loss)]
            let rect_y = (surf_h as f32 - fit_h as f32) * 0.5;
            #[allow(clippy::cast_precision_loss)]
            let rows = [
                [rect_x, rect_y, fit_w as f32, fit_h as f32],
                [snap.position_x, snap.position_y, snap.hotspot_x as f32, snap.hotspot_y as f32],
                [snap.width as f32, snap.height as f32, vw as f32, vh as f32],
                [surf_w as f32, surf_h as f32, 1.0, 0.0],
            ];
            queue.write_buffer(&self.uniform, 0, &cursor_uniform_bytes(&rows));
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tether-render cursor bind group"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(snap.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            }))
        });
        let Some(bind_group) = bind_group else {
            return;
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tether-render cursor pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    // LoadOp::Load preserves the blit pass's pixels.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..6, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_skipped_when_relative_mode() {
        let mut s = CursorState::new();
        // No sprite, no position — but flip relative mode to confirm
        // the gate fires before we even check the cache.
        s.set_relative_mode(true);
        s.set_position(10.0, 20.0, true);
        assert!(s.snapshot_for_render().is_none());
    }

    #[test]
    fn snapshot_skipped_when_invisible() {
        let mut s = CursorState::new();
        s.set_position(10.0, 20.0, false);
        assert!(s.snapshot_for_render().is_none());
    }

    #[test]
    fn activate_unknown_id_is_noop() {
        let mut s = CursorState::new();
        s.activate(42);
        assert!(s.active.is_none());
    }
}
