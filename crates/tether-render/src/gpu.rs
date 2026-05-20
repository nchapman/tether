use std::sync::Arc;

use tether_codec::{GpuFrameGuard, GpuFrameSource};
use winit::window::Window;

use crate::{CpuFrame, Frame, GpuFrame, RenderError, Result};

/// Y plus interleaved-UV textures and the bind group that points at
/// them. Bundled together so resize-on-frame can swap them atomically.
/// Y is R8Unorm at full resolution; UV is Rg8Unorm at half resolution
/// (each texel holds one U byte in .r and one V byte in .g, matching
/// the NV12 layout the decoder emits).
pub(crate) struct YuvTextures {
    // Both textures own GPU memory and are kept alive solely for the
    // bind group's views; we never reference them by name after the
    // bind group is built. The allow keeps clippy quiet about the
    // unused field reads.
    #[allow(dead_code)]
    y: wgpu::Texture,
    #[allow(dead_code)]
    uv: wgpu::Texture,
    pub(crate) bind_group: wgpu::BindGroup,
    /// Y-plane dimensions (chroma is derived).
    size: (u32, u32),
    /// Backend-side lifetime extender from the decoder (DMA-BUF path)
    /// — typically an `AVFrame` whose Drop releases the VAAPI surface
    /// back to the hwframes pool. `None` for CPU-uploaded textures.
    /// Held in the same struct as the textures so dropping `textures`
    /// (on resize / new frame) releases the surface in the same step.
    _guard: Option<GpuFrameGuard>,
}

pub(crate) struct GpuState {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    yuv_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    textures: YuvTextures,
    /// Previous frame's textures + guard, retired for one extra render
    /// cycle so any in-flight GPU sampling completes before the dma-buf
    /// memory is freed. Without this, the AVFrame guard would drop the
    /// VAAPI surface back to the pool the instant `textures` is
    /// replaced; the decoder could then reuse it for the next frame
    /// while the GPU is still mid-sample, producing torn output on
    /// drivers that don't attach Vulkan read fences to the dma-buf
    /// reservation object. Mesa+Intel does attach them; this slot is
    /// cheap insurance against drivers that don't.
    retired: Option<YuvTextures>,
    scale_buffer: wgpu::Buffer,
    scale_bind_group: wgpu::BindGroup,
    /// True if `VULKAN_EXTERNAL_MEMORY_DMA_BUF` was negotiated on the
    /// adapter. Gates the DMA-BUF import path; without it, a `GpuFrame`
    /// from the decoder gets dropped with a warn — the decoder probe
    /// shouldn't have picked the HW backend in that case, so reaching
    /// this branch indicates a misconfiguration.
    dmabuf_import_supported: bool,
}

impl GpuState {
    pub(crate) async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let (width, height) = (size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window)?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                // Added on wgpu trunk past 29.x; opting out keeps the
                // pre-trunk per-backend limit behaviour. Revisit when we
                // bump to wgpu 30.
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| RenderError::NoAdapter)?;

        // Ask for DMA-BUF import as an *optional* feature: probe the
        // adapter first, request it only if available. On Mesa/Intel
        // (the realistic deployment for our Linux VAAPI path) it lights
        // up; on Vulkan-portability stacks (lavapipe, MoltenVK) it
        // doesn't, and we fall back to the CPU-upload path. The render
        // crate doesn't refuse to start without it — the CPU path is
        // a valid degradation mode and the hard-require lives one layer
        // up in the decoder probe.
        let adapter_features = adapter.features();
        let mut required = wgpu::Features::empty();
        if adapter_features.contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF) {
            required |= wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tether-render device"),
                required_features: required,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await?;
        let dmabuf_import_supported =
            device.features().contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF);
        let info = adapter.get_info();
        tracing::info!(
            adapter = info.name,
            driver = info.driver,
            backend = ?info.backend,
            dmabuf_import_supported,
            "wgpu device initialised"
        );

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        // Per the expert review: Immediate is lowest-latency. Mailbox is
        // smoother on systems that have it. Fifo (vsync) is the universal
        // fallback but costs us a frame of latency.
        let present_mode = if surface_caps
            .present_modes
            .contains(&wgpu::PresentMode::Immediate)
        {
            wgpu::PresentMode::Immediate
        } else if surface_caps
            .present_modes
            .contains(&wgpu::PresentMode::Mailbox)
        {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &surface_config);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tether-render sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Two textures (Y + UV-interleaved) plus a sampler. We share
        // one sampler — chroma upsampling uses the same bilinear
        // filter as luma, and the visual difference vs. a chroma-only
        // nearest sampler is invisible on 4:2:0 desktop content.
        let yuv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tether-render yuv bgl"),
            entries: &[
                bgl_texture_entry(0),
                bgl_texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let scale_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tether-render scale bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let scale_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tether-render scale uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Seed identity so the first draw before any frame upload
        // doesn't read undefined memory off the GPU.
        let identity_scale: [f32; 4] = [1.0, 1.0, 0.0, 0.0];
        queue.write_buffer(&scale_buffer, 0, &bytes_of_f32x4(&identity_scale));
        let scale_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tether-render scale bind group"),
            layout: &scale_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scale_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tether-render shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tether-render pipeline layout"),
            bind_group_layouts: &[Some(&yuv_bgl), Some(&scale_bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tether-render pipeline"),
            layout: Some(&pipeline_layout),
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
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let textures = make_yuv_textures(&device, &yuv_bgl, &sampler, 1, 1);

        Ok(Self {
            surface,
            surface_config,
            device,
            queue,
            pipeline,
            yuv_bgl,
            sampler,
            textures,
            retired: None,
            scale_buffer,
            scale_bind_group,
            dmabuf_import_supported,
        })
    }

    /// Current video texture size and surface size. Returned together
    /// because the cursor-normalisation math in `lib.rs` needs both.
    pub(crate) fn dimensions(&self) -> ((u32, u32), (u32, u32)) {
        (
            self.textures.size,
            (self.surface_config.width, self.surface_config.height),
        )
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        // The video textures are at decoded resolution and don't change
        // on window resize. `render()` recomputes the letterbox scale
        // uniform every frame from `surface_config`, so the next redraw
        // automatically fits the new window without re-uploading or
        // re-importing.
    }

    /// Take ownership of a new frame and either upload it (Cpu) or
    /// import it via DMA-BUF (Gpu). The previous frame's textures
    /// (and, for Gpu, its decoder-side guard) drop here, releasing
    /// any VAAPI surface they kept alive.
    pub(crate) fn apply_frame(&mut self, frame: Frame) -> Result<()> {
        match frame {
            Frame::Cpu(cpu) => self.apply_cpu(cpu),
            Frame::Gpu(gpu) => self.apply_gpu(gpu),
        }
    }

    fn apply_cpu(&mut self, frame: CpuFrame) -> Result<()> {
        if frame.width == 0 || frame.height == 0 {
            return Ok(());
        }
        // Two reasons to rebuild: the previous frame was DMA-BUF-imported
        // (those textures aren't COPY_DST so write_texture would fail
        // — checking `_guard.is_some()` is load-bearing; the size match
        // would silently skip the rebuild otherwise), or the resolution
        // changed.
        if self.textures._guard.is_some() || (frame.width, frame.height) != self.textures.size {
            let fresh = make_yuv_textures(
                &self.device,
                &self.yuv_bgl,
                &self.sampler,
                frame.width,
                frame.height,
            );
            self.retire_textures(fresh);
        }
        let (chroma_w, chroma_h) = frame.chroma_dims();
        // Y is R8 — one byte per texel, so bytes_per_row == width.
        // UV is Rg8 — two bytes per texel, so bytes_per_row == chroma_w * 2.
        write_plane_r8(&self.queue, &self.textures.y, &frame.y, frame.width, frame.height);
        write_plane_rg8(&self.queue, &self.textures.uv, &frame.uv, chroma_w, chroma_h);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn apply_gpu(&mut self, frame: GpuFrame) -> Result<()> {
        if !self.dmabuf_import_supported {
            tracing::warn!(
                "received GpuFrame but adapter lacks VULKAN_EXTERNAL_MEMORY_DMA_BUF; \
                 dropping frame (decoder probe should have picked a SW backend)"
            );
            return Ok(());
        }
        // Use an explicit match (not `let else`) so adding a future
        // GpuFrameSource variant (NVDEC over CUDA-Vulkan interop, a
        // VideoToolbox CVPixelBuffer, etc.) is a compile error here
        // rather than a silent fallthrough.
        let dmabuf = match frame.source {
            GpuFrameSource::DmaBuf(d) => d,
        };
        let fresh = import_dmabuf_textures(
            &self.device,
            &self.yuv_bgl,
            &self.sampler,
            &dmabuf,
            frame.width,
            frame.height,
            frame.guard,
        )?;
        self.retire_textures(fresh);
        Ok(())
    }

    /// Swap in `fresh` as the active textures and move the previous
    /// set into the retired slot. Anything already in the retired
    /// slot is dropped — that frame's GPU submission has had at least
    /// one `render()` cycle's worth of opportunity to complete on the
    /// GPU, so its dma-buf is safe to unmap and its VAAPI surface is
    /// safe to return to the pool.
    fn retire_textures(&mut self, fresh: YuvTextures) {
        let previous = std::mem::replace(&mut self.textures, fresh);
        self.retired = Some(previous);
    }

    #[cfg(not(target_os = "linux"))]
    fn apply_gpu(&mut self, _frame: GpuFrame) -> Result<()> {
        // GpuFrameSource has no variants off-Linux, so a GpuFrame is
        // type-level uninhabitable here; this stub exists to keep the
        // match in `apply_frame` exhaustive across cfgs.
        unreachable!("GpuFrame cannot be constructed on this platform")
    }

    pub(crate) fn render(&mut self) -> std::result::Result<(), String> {
        use wgpu::CurrentSurfaceTexture::*;
        // Handle all seven variants from wgpu 29 deliberately. Defaults
        // matter: Outdated/Lost without reconfigure leaves the window
        // permanently black; Occluded without silencing spams logs while
        // the window is minimised; Suboptimal still has a valid texture
        // that should be presented for this frame.
        let (output, reconfigure_after) = match self.surface.get_current_texture() {
            Success(f) => (f, false),
            Suboptimal(f) => {
                tracing::debug!("wgpu surface suboptimal; reconfigure after present");
                (f, true)
            }
            Outdated => {
                tracing::debug!("wgpu surface outdated; reconfiguring");
                self.surface.configure(&self.device, &self.surface_config);
                return Ok(());
            }
            Lost => {
                tracing::warn!("wgpu surface lost; attempting best-effort reconfigure");
                self.surface.configure(&self.device, &self.surface_config);
                return Ok(());
            }
            Timeout | Occluded => {
                return Ok(());
            }
            Validation => {
                return Err("wgpu surface validation error".into());
            }
        };

        // Update the aspect-correction uniform before kicking off the
        // render pass. Costs a single 16-byte buffer write per frame.
        let (sx, sy) = letterbox_scale(self.textures.size, (
            self.surface_config.width,
            self.surface_config.height,
        ));
        self.queue
            .write_buffer(&self.scale_buffer, 0, &bytes_of_f32x4(&[sx, sy, 0.0, 0.0]));

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tether-render encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tether-render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.textures.bind_group, &[]);
            pass.set_bind_group(1, &self.scale_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Drop the previously-retired textures after the *next* submit
        // completes the previous frame's read. Conservative: this drops
        // one cycle after the texture stopped being bound, not one
        // cycle after the GPU finished reading from it. Tightening to
        // a real signal (queue submission index, fence) is the kind of
        // optimisation worth doing only if profiling shows the extra
        // dma-buf memory held across one frame matters.
        self.retired = None;

        if reconfigure_after {
            self.surface.configure(&self.device, &self.surface_config);
        }
        Ok(())
    }
}

/// Compute the (x, y) NDC scale factors that letterbox / pillarbox the
/// source texture inside the surface while preserving its aspect ratio.
/// Returns (1.0, 1.0) for matching aspect ratios. The unused axis gets
/// the proportional shrink; the dominant axis stays at 1.0.
#[allow(clippy::cast_precision_loss)]
fn letterbox_scale(src: (u32, u32), dst: (u32, u32)) -> (f32, f32) {
    if src.0 == 0 || src.1 == 0 || dst.0 == 0 || dst.1 == 0 {
        return (1.0, 1.0);
    }
    let src_aspect = src.0 as f32 / src.1 as f32;
    let dst_aspect = dst.0 as f32 / dst.1 as f32;
    if (src_aspect - dst_aspect).abs() < f32::EPSILON {
        (1.0, 1.0)
    } else if src_aspect > dst_aspect {
        (1.0, dst_aspect / src_aspect)
    } else {
        (src_aspect / dst_aspect, 1.0)
    }
}

fn bytes_of_f32x4(v: &[f32; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (i, x) in v.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&x.to_le_bytes());
    }
    out
}

fn bgl_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn make_yuv_textures(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    width: u32,
    height: u32,
) -> YuvTextures {
    let chroma_w = width.div_ceil(2);
    let chroma_h = height.div_ceil(2);
    let y = make_plane_texture(
        device,
        "tether-render y plane",
        width,
        height,
        wgpu::TextureFormat::R8Unorm,
    );
    let uv = make_plane_texture(
        device,
        "tether-render uv plane",
        chroma_w,
        chroma_h,
        wgpu::TextureFormat::Rg8Unorm,
    );
    let y_view = y.create_view(&wgpu::TextureViewDescriptor::default());
    let uv_view = uv.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tether-render yuv bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&y_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&uv_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    YuvTextures {
        y,
        uv,
        bind_group,
        size: (width, height),
        _guard: None,
    }
}

/// Import a NV12 VAAPI surface (exported with SEPARATE_LAYERS) as two
/// wgpu textures: Y as `R8Unorm`, UV as `Rg8Unorm`. Each layer of the
/// DMA-BUF descriptor points at an object via `object_index[0]`; we
/// dup the underlying fd for each call because wgpu's hal API takes
/// ownership and the same object can be referenced by both layers
/// (Intel/Mesa typically packs Y and UV into one allocation with
/// different offsets).
#[cfg(target_os = "linux")]
#[allow(clippy::cast_lossless)] // u32 pitch into u64 stride is intentional
pub(crate) fn import_dmabuf_textures(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    dmabuf: &tether_codec::DmaBufFrame,
    width: u32,
    height: u32,
    guard: GpuFrameGuard,
) -> Result<YuvTextures> {
    if dmabuf.layers.len() != 2 {
        return Err(RenderError::DmaBufImport(format!(
            "expected 2 layers (NV12 SEPARATE_LAYERS), got {}",
            dmabuf.layers.len()
        )));
    }
    let chroma_w = width.div_ceil(2);
    let chroma_h = height.div_ceil(2);
    let y = import_one_layer(
        device,
        "tether-render y plane (dmabuf)",
        dmabuf,
        0,
        width,
        height,
        wgpu::TextureFormat::R8Unorm,
    )?;
    let uv = import_one_layer(
        device,
        "tether-render uv plane (dmabuf)",
        dmabuf,
        1,
        chroma_w,
        chroma_h,
        wgpu::TextureFormat::Rg8Unorm,
    )?;
    let y_view = y.create_view(&wgpu::TextureViewDescriptor::default());
    let uv_view = uv.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tether-render yuv bind group (dmabuf)"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&y_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&uv_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    Ok(YuvTextures {
        y,
        uv,
        bind_group,
        size: (width, height),
        _guard: Some(guard),
    })
}

#[cfg(target_os = "linux")]
fn import_one_layer(
    device: &wgpu::Device,
    label: &str,
    dmabuf: &tether_codec::DmaBufFrame,
    layer_idx: usize,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> Result<wgpu::Texture> {
    let layer = &dmabuf.layers[layer_idx];
    // SEPARATE_LAYERS gives one plane per layer; multi-plane within a
    // layer would mean we're looking at a COMPOSED export, which we
    // explicitly didn't ask for and don't import.
    if layer.num_planes != 1 {
        return Err(RenderError::DmaBufImport(format!(
            "layer {layer_idx} has {} planes; expected 1 (SEPARATE_LAYERS)",
            layer.num_planes
        )));
    }
    let obj_idx = layer.object_index[0] as usize;
    let obj = dmabuf.objects.get(obj_idx).ok_or_else(|| {
        RenderError::DmaBufImport(format!(
            "layer {layer_idx} references object {obj_idx} but only {} present",
            dmabuf.objects.len()
        ))
    })?;
    // dup the fd because wgpu takes ownership and the same object may
    // also back another layer. `try_clone` is dup(2) with CLOEXEC.
    let fd = obj
        .fd
        .try_clone()
        .map_err(|e| RenderError::DmaBufImport(format!("dup dma-buf fd: {e}")))?;

    let hal_desc = wgpu::hal::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUses::RESOURCE,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };

    // SAFETY: the device was created from this hal::Api (Vulkan is
    // wgpu's default backend on Linux); `as_hal` returns Some for the
    // matching API. `texture_from_dmabuf_fd` consumes the fd on
    // success and closes it on failure, so we hand over our duped one.
    // `texture_from_raw` (called inside) requires the hal_texture be
    // created from this device, which is exactly what we did.
    let hal_texture = unsafe {
        let hal_dev = device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or_else(|| RenderError::DmaBufImport("device is not Vulkan-backed".into()))?;
        hal_dev
            .texture_from_dmabuf_fd(
                fd,
                &hal_desc,
                obj.drm_format_modifier,
                u64::from(layer.pitch[0]),
                u64::from(layer.offset[0]),
            )
            .map_err(|e| {
                RenderError::DmaBufImport(format!("texture_from_dmabuf_fd: {e:?}"))
            })?
    };

    let wgpu_desc = wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    // SAFETY: hal_texture was just built from this device on the same
    // Vulkan backend. The wgpu_desc must describe the same image as
    // the hal_desc — width/height/format/dimension are identical
    // above; `usage` deliberately differs (`TextureUses::RESOURCE` on
    // the hal side, `TextureUsages::TEXTURE_BINDING` on the wgpu side
    // — they're the equivalent representations in each API's vocabulary
    // and that's the correct pairing).
    let texture = unsafe { device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_texture, &wgpu_desc) };
    Ok(texture)
}

fn make_plane_texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Non-sRGB variants on purpose: YUV values are in a linear
        // encoding space already (the BT.709 limited-range expansion
        // in the fragment shader handles gamma); applying sRGB on
        // sample would double-decode and crush highlights.
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn write_plane_r8(queue: &wgpu::Queue, texture: &wgpu::Texture, bytes: &[u8], width: u32, height: u32) {
    write_plane(queue, texture, bytes, width, height, 1);
}

fn write_plane_rg8(queue: &wgpu::Queue, texture: &wgpu::Texture, bytes: &[u8], width: u32, height: u32) {
    write_plane(queue, texture, bytes, width, height, 2);
}

fn write_plane(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    bytes: &[u8],
    width: u32,
    height: u32,
    bytes_per_texel: u32,
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * bytes_per_texel),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}
