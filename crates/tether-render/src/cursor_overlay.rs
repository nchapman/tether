//! Cursor overlay state + shared draw math.
//!
//! Renders the host's mouse cursor as an alpha-blended sprite quad on
//! top of the video. Lives behind a shared `Arc<Mutex<CursorState>>`
//! so the client's wire-receive task can update sprite cache, active
//! id, and position from one thread while the renderer reads them on
//! its event-loop thread.
//!
//! ## Backend split
//!
//! [`CursorState`] is backend-agnostic: it caches each sprite as raw
//! straight-alpha RGBA8 bytes plus its hotspot, and exposes a
//! per-frame [`CursorSnapshot`]. Each renderer owns the GPU side —
//! the wgpu [`CursorOverlay`] (Linux/macOS) and the D3D11
//! `cursor::D3D11CursorOverlay` (Windows) each upload the active
//! sprite into their own API's texture and draw the quad. The
//! coordinate math (sprite placement inside the letterboxed video
//! rect) is shared via [`cursor_uniform_for`] so the two backends
//! can't drift.
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
use std::sync::{Arc, Mutex};

/// Maximum sprites cached on the client. A real session uses ~5–10
/// distinct cursors (arrow, i-beam, hand, resize-*, wait). The cap is
/// purely defense against a buggy or malicious host emitting unique
/// ids forever — 32 RGBA8 sprites ≤ 64×64 each is ~512 KiB.
pub(crate) const CURSOR_CACHE_MAX: usize = 32;

/// Bytes in the packed cursor uniform: 4 × `vec4<f32>`.
pub(crate) const CURSOR_UNIFORM_BYTES: usize = 64;

// D3D11 requires a constant buffer's `ByteWidth` to be a multiple of 16
// (and wgpu's `min_binding_size` is happiest aligned too). Catch a
// future field addition that breaks the alignment at compile time rather
// than as a silent `CreateBuffer` failure on some drivers.
const _: () = assert!(
    CURSOR_UNIFORM_BYTES % 16 == 0,
    "cursor uniform must be a multiple of 16 bytes for D3D11 constant buffers",
);

/// One cached sprite: its straight-alpha RGBA8 pixels plus hotspot.
/// Backend-agnostic — the renderer uploads `pixels` into its own GPU
/// texture when this sprite becomes active.
struct CachedSprite {
    /// `width * height * 4` bytes, straight (non-premultiplied) alpha.
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    hotspot_x: u32,
    hotspot_y: u32,
    /// Monotonic counter snapshot at last access; smallest = LRU
    /// eviction target. Cheaper than a doubly-linked list for the
    /// expected size.
    last_used: u64,
}

/// Shared cursor state. Written by the wire-receive task; read by the
/// renderer.
pub struct CursorState {
    cache: HashMap<u64, CachedSprite>,
    /// Currently-active sprite id (set by `CursorUseShape` after an
    /// `enqueue_shape`). `None` until the host names one.
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
            active: None,
            position_x: 0.0,
            position_y: 0.0,
            visible: false,
            relative_mode: false,
            access_counter: 0,
        }
    }

    /// Wire-side entry point: cache a new sprite by id. Caller supplies
    /// straight-alpha RGBA8 (`pixels.len() == width * height * 4`);
    /// a size mismatch or zero dimension is dropped with a warn rather
    /// than cached, so the renderer never samples a malformed sprite.
    ///
    /// The cache is bounded at [`CURSOR_CACHE_MAX`] via LRU eviction on
    /// insert — without the cap, a buggy or malicious host emitting
    /// unique ids forever would grow unbounded. Re-`enqueue`ing an
    /// existing id replaces its pixels (and refreshes its LRU slot).
    pub fn enqueue_shape(
        &mut self,
        id: u64,
        width: u32,
        height: u32,
        hotspot_x: u32,
        hotspot_y: u32,
        pixels: Vec<u8>,
    ) {
        if width == 0 || height == 0 {
            return;
        }
        let expected_bytes = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        if pixels.len() != expected_bytes {
            tracing::warn!(
                id,
                width,
                height,
                got = pixels.len(),
                expected = expected_bytes,
                "cursor shape pixel buffer size mismatch; dropping",
            );
            return;
        }
        // LRU evict before insertion when at cap and id is new.
        if !self.cache.contains_key(&id) && self.cache.len() >= CURSOR_CACHE_MAX {
            if let Some(oldest_id) = self
                .cache
                .iter()
                .min_by_key(|(_, s)| s.last_used)
                .map(|(k, _)| *k)
            {
                self.cache.remove(&oldest_id);
            }
        }
        self.access_counter += 1;
        self.cache.insert(
            id,
            CachedSprite {
                pixels,
                width,
                height,
                hotspot_x,
                hotspot_y,
                last_used: self.access_counter,
            },
        );
    }

    /// Activate a previously-cached sprite. Silently no-ops when the
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

    /// Renderer-side snapshot: the active sprite's pixels + placement
    /// inputs, or `None` if the overlay should not draw this frame
    /// (relative mode, hidden, no active sprite, or the active sprite
    /// was evicted from the cache). The borrowed `pixels` are valid for
    /// as long as the [`CursorChannel`] lock is held.
    pub(crate) fn snapshot(&self) -> Option<CursorSnapshot<'_>> {
        if self.relative_mode || !self.visible {
            return None;
        }
        let id = self.active?;
        let sprite = self.cache.get(&id)?;
        Some(CursorSnapshot {
            id,
            pixels: &sprite.pixels,
            width: sprite.width,
            height: sprite.height,
            hotspot_x: sprite.hotspot_x,
            hotspot_y: sprite.hotspot_y,
            position_x: self.position_x,
            position_y: self.position_y,
        })
    }

    #[cfg(test)]
    pub(crate) fn cache_len(&self) -> usize {
        self.cache.len()
    }

    #[cfg(test)]
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
        f(&mut guard)
    }
}

/// Renderer-side view of the active sprite for one frame. Backend
/// overlays upload `pixels` into a GPU texture (keyed by `id`, so an
/// unchanged active sprite isn't re-uploaded) and place the quad via
/// [`cursor_uniform_for`].
pub(crate) struct CursorSnapshot<'a> {
    pub id: u64,
    pub pixels: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub hotspot_x: u32,
    pub hotspot_y: u32,
    pub position_x: f32,
    pub position_y: f32,
}

/// Dedup key for the cursor-render-params log line: every input to the
/// shader's placement formula. Logging only when this tuple changes
/// keeps a static cursor from spamming the log at 60 Hz.
pub(crate) type CursorLogDims = (u32, u32, u32, u32, u32, u32, u32, u32);

/// Build the 64-byte packed uniform the cursor shader (`cursor.wgsl` /
/// `cursor.hlsl`) reads, from a snapshot and the frame's letterbox
/// geometry. Shared by both backends so the sprite lands in the same
/// pixel rect regardless of graphics API.
///
/// `last_logged` dedups a diagnostic line (every placement input) so a
/// "cursor too big / wrong spot" complaint is debuggable from logs
/// alone without flooding them each frame.
pub(crate) fn cursor_uniform_for(
    snap: &CursorSnapshot<'_>,
    video_dims: (u32, u32),
    surface_dims: (u32, u32),
    fit_dims: (u32, u32),
    last_logged: &mut Option<CursorLogDims>,
) -> [u8; CURSOR_UNIFORM_BYTES] {
    let (vw, vh) = video_dims;
    let (surf_w, surf_h) = surface_dims;
    let (fit_w, fit_h) = fit_dims;
    // Letterbox rect origin inside the window — same math the blit pass
    // implicitly does via NDC scale, but we need explicit pixel-space
    // anchors here for the sprite quad.
    #[allow(clippy::cast_precision_loss)]
    let rect_x = (surf_w as f32 - fit_w as f32) * 0.5;
    #[allow(clippy::cast_precision_loss)]
    let rect_y = (surf_h as f32 - fit_h as f32) * 0.5;

    let key = (
        snap.width,
        snap.height,
        vw,
        vh,
        surf_w,
        surf_h,
        fit_w,
        fit_h,
    );
    if *last_logged != Some(key) {
        #[allow(clippy::cast_precision_loss)]
        let sprite_px = (
            (snap.width as f32) * (fit_w as f32) / (vw.max(1) as f32),
            (snap.height as f32) * (fit_h as f32) / (vh.max(1) as f32),
        );
        // Diagnostic-only: rounded sprite extents logged as integers.
        #[allow(clippy::cast_possible_truncation)]
        let rendered_sprite_px = (sprite_px.0.round() as i32, sprite_px.1.round() as i32);
        tracing::info!(
            sprite_wh = ?(snap.width, snap.height),
            video_wh = ?(vw, vh),
            surface_wh = ?(surf_w, surf_h),
            fit_wh = ?(fit_w, fit_h),
            position = ?(snap.position_x, snap.position_y),
            hotspot = ?(snap.hotspot_x, snap.hotspot_y),
            rendered_sprite_px = ?rendered_sprite_px,
            "cursor overlay render params changed"
        );
        *last_logged = Some(key);
    }
    #[allow(clippy::cast_precision_loss)]
    let rows = [
        [rect_x, rect_y, fit_w as f32, fit_h as f32],
        [
            snap.position_x,
            snap.position_y,
            snap.hotspot_x as f32,
            snap.hotspot_y as f32,
        ],
        [snap.width as f32, snap.height as f32, vw as f32, vh as f32],
        [surf_w as f32, surf_h as f32, 1.0, 0.0],
    ];
    cursor_uniform_bytes(&rows)
}

/// 4 × vec4<f32> packed little-endian for the cursor uniform. Hand-
/// rolled bytes to avoid pulling in a `bytemuck` dependency just for
/// this struct.
fn cursor_uniform_bytes(rows: &[[f32; 4]; 4]) -> [u8; CURSOR_UNIFORM_BYTES] {
    let mut out = [0u8; CURSOR_UNIFORM_BYTES];
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            let b = v.to_le_bytes();
            let off = i * 16 + j * 4;
            out[off..off + 4].copy_from_slice(&b);
        }
    }
    out
}

// ---------------------------------------------------------------------
// wgpu cursor overlay (Linux/macOS). The Windows D3D11 backend has its
// own overlay in `crate::d3d11::cursor`; both consume the shared
// `CursorState` / `cursor_uniform_for` above.
// ---------------------------------------------------------------------
#[cfg(not(target_os = "windows"))]
pub(crate) use wgpu_overlay::CursorOverlay;

#[cfg(not(target_os = "windows"))]
mod wgpu_overlay {
    use std::num::NonZeroU64;

    use super::{
        cursor_uniform_for, CursorChannel, CursorLogDims, CursorSnapshot, CURSOR_UNIFORM_BYTES,
    };

    /// The active sprite uploaded to a wgpu texture, keyed by id so an
    /// unchanged cursor isn't re-uploaded each frame. Owning the texture
    /// here (rather than borrowing one out of the cache) means a cache
    /// eviction can't drop the texture out from under an in-flight draw.
    struct ActiveSprite {
        id: u64,
        // Held for RAII: dropping it frees the GPU memory `view` borrows.
        _texture: wgpu::Texture,
        view: wgpu::TextureView,
    }

    /// GPU-side cursor overlay: pipeline, sampler, bind group layout, the
    /// per-frame uniform buffer, and the currently-uploaded sprite.
    pub(crate) struct CursorOverlay {
        pipeline: wgpu::RenderPipeline,
        bgl: wgpu::BindGroupLayout,
        sampler: wgpu::Sampler,
        uniform: wgpu::Buffer,
        active: Option<ActiveSprite>,
        /// Dedup key for the cursor-render-params diagnostic — see
        /// [`cursor_uniform_for`].
        last_logged_dims: Option<CursorLogDims>,
    }

    impl CursorOverlay {
        pub(crate) fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tether-render cursor bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(CURSOR_UNIFORM_BYTES as u64),
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
                size: CURSOR_UNIFORM_BYTES as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            Self {
                pipeline,
                bgl,
                sampler,
                uniform,
                active: None,
                last_logged_dims: None,
            }
        }

        /// Upload `snap`'s pixels into a fresh `Rgba8Unorm` texture and
        /// make it the active sprite. Called only when the active id
        /// changes, so a static cursor pays nothing here.
        fn upload_sprite(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            snap: &CursorSnapshot<'_>,
        ) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("tether-render cursor sprite"),
                size: wgpu::Extent3d {
                    width: snap.width,
                    height: snap.height,
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
                snap.pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(snap.width * 4),
                    rows_per_image: Some(snap.height),
                },
                wgpu::Extent3d {
                    width: snap.width,
                    height: snap.height,
                    depth_or_array_layers: 1,
                },
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.active = Some(ActiveSprite {
                id: snap.id,
                _texture: texture,
                view,
            });
        }

        /// Run the cursor pass for one frame. No-ops when the cursor isn't
        /// visible or no sprite is active — the renderer's hot path pays
        /// one mutex lock + one HashMap lookup in those cases, which is
        /// cheap relative to the rest of the frame.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn render(
            &mut self,
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
            if surf_w == 0 || surf_h == 0 || vw == 0 || vh == 0 {
                return;
            }
            let draw = channel.with(|state| {
                let snap = state.snapshot()?;
                if self.active.as_ref().map(|a| a.id) != Some(snap.id) {
                    self.upload_sprite(device, queue, &snap);
                }
                let bytes = cursor_uniform_for(
                    &snap,
                    video_dims,
                    surface_dims,
                    fit_dims,
                    &mut self.last_logged_dims,
                );
                queue.write_buffer(&self.uniform, 0, &bytes);
                Some(())
            });
            if draw.is_none() {
                return;
            }
            // The overlay owns the active texture, so it outlives the
            // cache lock released above — the bind group can't dangle.
            let Some(active) = self.active.as_ref() else {
                return;
            };
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tether-render cursor bind group"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&active.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

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
        assert!(s.snapshot().is_none());
    }

    #[test]
    fn snapshot_skipped_when_invisible() {
        let mut s = CursorState::new();
        s.set_position(10.0, 20.0, false);
        assert!(s.snapshot().is_none());
    }

    #[test]
    fn activate_unknown_id_is_noop() {
        let mut s = CursorState::new();
        s.activate(42);
        assert!(s.active.is_none());
    }

    #[test]
    fn enqueue_then_activate_yields_snapshot() {
        let mut s = CursorState::new();
        // 2×2 RGBA8 = 16 bytes.
        s.enqueue_shape(7, 2, 2, 1, 1, vec![0xAB; 16]);
        s.activate(7);
        s.set_position(40.0, 30.0, true);
        let snap = s.snapshot().expect("active visible sprite");
        assert_eq!(snap.id, 7);
        assert_eq!((snap.width, snap.height), (2, 2));
        assert_eq!((snap.hotspot_x, snap.hotspot_y), (1, 1));
        assert_eq!(snap.pixels.len(), 16);
    }

    #[test]
    fn enqueue_rejects_size_mismatch() {
        let mut s = CursorState::new();
        // Claims 4×4 (64 bytes) but supplies 16 — must be dropped.
        s.enqueue_shape(1, 4, 4, 0, 0, vec![0u8; 16]);
        assert!(!s.contains(1));
    }

    #[test]
    fn enqueue_evicts_lru_at_cap() {
        let mut s = CursorState::new();
        for id in 0..(CURSOR_CACHE_MAX as u64) {
            s.enqueue_shape(id, 1, 1, 0, 0, vec![0u8; 4]);
        }
        assert_eq!(s.cache_len(), CURSOR_CACHE_MAX);
        // id 0 is the LRU; one more insert evicts it, not the newer ids.
        s.enqueue_shape(999, 1, 1, 0, 0, vec![0u8; 4]);
        assert_eq!(s.cache_len(), CURSOR_CACHE_MAX);
        assert!(!s.contains(0), "oldest sprite should be evicted");
        assert!(s.contains(999), "newest sprite should be cached");
    }
}
