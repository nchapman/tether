use std::sync::Arc;

use tether_codec::{GpuFrameGuard, GpuFrameSource};
use tether_protocol::control::{ChromaSubsampling, ColorTransfer, VideoColorSpec};
use winit::window::Window;

use crate::{CpuFrame, Frame, GpuFrame, RenderError, Result};

#[cfg(target_os = "linux")]
mod import;
#[cfg(target_os = "linux")]
pub(crate) use import::import_dmabuf_textures;

#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
pub(crate) use metal::import_iosurface_textures;

/// One frame's worth of YUV plane textures plus the bind group that
/// points at them. The variant matches the negotiated chroma — NV12
/// shape for 4:2:0, three R8 planes for 4:4:4. Bundled with the bind
/// group so resize-on-frame can swap them atomically.
pub(crate) struct YuvTextures {
    /// Per-chroma plane storage. Different variants in different
    /// modes; bind group already references the correct plane set
    /// via the (chroma-specific) bind-group layout.
    ///
    /// FIELD ORDER IS LOAD-BEARING. Rust drops struct fields in
    /// declaration order, and on macOS the MTLTexture inside each plane
    /// holds the IOSurface alive (via Metal's internal retain), while
    /// `_guard` below holds it alive via the AVFrame's CVPixelBuffer
    /// retain. The texture drop must run first so Metal releases its
    /// retain before the AVFrame drops the CVPixelBuffer; otherwise a
    /// future reorder could leave a dangling MTLTexture pointing at a
    /// released IOSurface. Same ordering serves the Linux side (wgpu
    /// texture releases the imported DMA-BUF before the VAAPI surface
    /// returns to the pool).
    planes: YuvPlanes,
    pub(crate) bind_group: wgpu::BindGroup,
    /// Luma-plane dimensions (chroma is derived from chroma kind).
    size: (u32, u32),
    /// Backend-side lifetime extender from the decoder (DMA-BUF path
    /// on Linux, AVFrame holding the CVPixelBuffer on macOS). `None`
    /// for CPU-uploaded textures. Declared *after* `planes` so the
    /// textures' Drop runs first — see field-order note above.
    _guard: Option<GpuFrameGuard>,
}

/// Chroma-specific plane storage. The texture handles are kept alive
/// solely for the bind group's views; we never reference them by name
/// after the bind group is built.
#[allow(dead_code)]
pub(crate) enum YuvPlanes {
    /// NV12: full-res R8 Y plus half-res Rg8 UV (interleaved chroma).
    Nv12 { y: wgpu::Texture, uv: wgpu::Texture },
    /// XYUV8888 packed 4:4:4: one Rgba8 texture, byte order V/U/Y/X.
    /// See `shader_yuv444.wgsl` for why packed instead of planar.
    Yuv444 { packed: wgpu::Texture },
}

pub(crate) struct GpuState {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    yuv_bgl: wgpu::BindGroupLayout,
    /// Negotiated chroma for this session. Fixed at construction —
    /// a chroma switch needs a full GpuState rebuild, the same way
    /// a resolution change works.
    chroma: ChromaSubsampling,
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
    /// Color-params uniform: holds the EOTF kind the fragment shader
    /// dispatches on (`bt709_eotf` vs `srgb_eotf`). Written once at
    /// pipeline-build time from the negotiated `VideoColorSpec`.
    /// Kept on the struct (rather than dropped after the write) so
    /// a future mid-session resign — when the protocol grows one —
    /// can `queue.write_buffer` into it without rebuilding the pipeline.
    #[allow(dead_code)]
    color_params_buffer: wgpu::Buffer,
    color_params_bind_group: wgpu::BindGroup,
    /// True if `VULKAN_EXTERNAL_MEMORY_DMA_BUF` was negotiated on the
    /// adapter. Gates the DMA-BUF import path; without it, a `GpuFrame`
    /// from the decoder gets dropped with a warn — the decoder probe
    /// shouldn't have picked the HW backend in that case, so reaching
    /// this branch indicates a misconfiguration.
    #[cfg(target_os = "linux")]
    dmabuf_import_supported: bool,
    /// True if the wgpu device exposes a Metal HAL. Required by the
    /// IOSurface→MTLTexture→wgpu import path. Probed at construction so
    /// any backend mismatch (e.g. a forced Vulkan adapter on macOS via
    /// MoltenVK) surfaces here instead of as a first-frame error.
    #[cfg(target_os = "macos")]
    metal_import_supported: bool,
}

/// Renderer plane layout — decoupled from `ChromaSubsampling` because
/// the same chroma can arrive in different GPU shapes depending on
/// backend:
///
/// - **Biplanar** = NV12 / NV24-style: Y in an R8 texture plus
///   interleaved UV in an Rg8 texture. Plane resolutions differ
///   between 4:2:0 (half-res UV) and 4:4:4 (full-res UV), but the
///   bind-group layout and fragment shader are identical — bilinear
///   sampling reads the right value at either resolution. Used by
///   the macOS IOSurface import path for both Yuv420 (`'420v'`) and
///   Yuv444 (`'444v'`), and by the Linux dma-buf path for Yuv420.
/// - **PackedXYUV** = one Rgba8 texture with byte order V/U/Y/X.
///   Only the Linux dma-buf path for Yuv444 — VAAPI's 4:4:4 surfaces
///   are exposed this way and the shader pulls Y from `.z`, U from
///   `.y`, V from `.x`.
///
/// Adding a new `(chroma, backend)` combo means returning the right
/// variant from [`render_layout_for`] and (if it's a third shape)
/// wiring a new fragment shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderLayout {
    Biplanar,
    PackedXYUV,
}

fn render_layout_for(chroma: ChromaSubsampling) -> RenderLayout {
    match chroma {
        ChromaSubsampling::Yuv420 => RenderLayout::Biplanar,
        // macOS gives us biplanar NV24 IOSurfaces for HEVC 4:4:4 via
        // VideoToolbox; Linux gives us packed XYUV via VAAPI. The
        // negotiated chroma is the same; the GPU-side shape is not.
        ChromaSubsampling::Yuv444 => {
            if cfg!(target_os = "macos") {
                RenderLayout::Biplanar
            } else {
                RenderLayout::PackedXYUV
            }
        }
    }
}

/// Numeric tag for the WGSL fragment shader's EOTF dispatch. Keep in
/// sync with the `bt709_eotf` / `srgb_eotf` switch in `shader.wgsl`.
/// Encoding it as a single integer (rather than separate uniforms or
/// pipeline variants) means the renderer never recompiles when a
/// future stream renegotiates the transfer; the shader picks the
/// right path per draw.
const TRANSFER_KIND_BT709: u32 = 0;
const TRANSFER_KIND_SRGB: u32 = 1;

fn transfer_kind_for(spec: VideoColorSpec) -> u32 {
    match spec.transfer {
        ColorTransfer::Bt709 => TRANSFER_KIND_BT709,
        ColorTransfer::Srgb => TRANSFER_KIND_SRGB,
        // PQ / HLG / Linear are reserved on the wire; the shader
        // doesn't implement them yet. Fall back to BT.709 with a
        // warn — the picture will look slightly off but won't be
        // unwatchable. Promoting these to a hard error is what the
        // HDR work should do once the surface format / tone-map
        // chain is in place.
        ColorTransfer::Pq | ColorTransfer::Hlg | ColorTransfer::Linear => {
            tracing::warn!(
                side = "gpu",
                ?spec.transfer,
                "EOTF not yet implemented; falling back to BT.709"
            );
            TRANSFER_KIND_BT709
        }
    }
}

impl GpuState {
    pub(crate) async fn new(
        window: Arc<Window>,
        color_space: VideoColorSpec,
        chroma: ChromaSubsampling,
    ) -> Result<Self> {
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

        // Hard-require DMA-BUF import on Linux. The decoder side
        // (tether-codec's probe) hard-requires VAAPI and the decoded
        // frames arrive as `Frame::Gpu` carrying dma-buf fds; the
        // renderer needs the matching Vulkan extension to consume them.
        // If the adapter doesn't advertise it (lavapipe, very old
        // Mesa, missing VK_EXT_image_drm_format_modifier), fail loudly
        // here rather than silently dropping every frame later.
        let info = adapter.get_info();
        let adapter_features = adapter.features();
        #[cfg(target_os = "linux")]
        if !adapter_features.contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF) {
            return Err(RenderError::DmaBufImport(format!(
                "wgpu adapter '{}' (driver: '{}', backend: {:?}) does not advertise \
                 VULKAN_EXTERNAL_MEMORY_DMA_BUF. Check that the system Vulkan ICD \
                 is a real GPU driver (Mesa 24+ for Intel/AMD) and exposes \
                 VK_EXT_external_memory_dma_buf + VK_EXT_image_drm_format_modifier; \
                 lavapipe and similar software stacks do not. Tether requires \
                 zero-copy decode — there is no CPU-upload fallback.",
                info.name, info.driver, info.backend
            )));
        }
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
        #[cfg(target_os = "linux")]
        let dmabuf_import_supported =
            device.features().contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF);
        // Probe the Metal HAL once at startup. If the wgpu device wasn't
        // built on the Metal backend (could happen if a future config
        // forces Vulkan via MoltenVK on macOS) we want a clean failure
        // here, not a panic deep in `import_iosurface_textures` on the
        // first decoded frame.
        #[cfg(target_os = "macos")]
        let metal_import_supported = unsafe {
            // SAFETY: as_hal is only unsafe because the returned guard
            // must not outlive the device; we drop the result immediately.
            device.as_hal::<wgpu::hal::api::Metal>().is_some()
        };
        #[cfg(target_os = "macos")]
        if !metal_import_supported {
            return Err(RenderError::DmaBufImport(format!(
                "wgpu adapter '{}' (driver: '{}', backend: {:?}) is not Metal-backed; \
                 macOS IOSurface import requires the Metal HAL. Tether requires \
                 zero-copy decode — there is no CPU-upload fallback.",
                info.name, info.driver, info.backend
            )));
        }
        #[cfg(target_os = "linux")]
        tracing::info!(
            adapter = info.name,
            driver = info.driver,
            backend = ?info.backend,
            dmabuf_import_supported,
            "wgpu device initialised"
        );
        #[cfg(target_os = "macos")]
        tracing::info!(
            adapter = info.name,
            driver = info.driver,
            backend = ?info.backend,
            metal_import_supported,
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

        // Bind-group layout depends on the renderer's plane shape, not
        // directly on chroma — see `RenderLayout` for why. Biplanar
        // (NV12 / NV24) has Y + UV textures + sampler at binding 2;
        // PackedXYUV has one Rgba8 texture + sampler at binding 1.
        let layout = render_layout_for(chroma);
        let yuv_bgl = match layout {
            RenderLayout::Biplanar => {
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("tether-render yuv bgl (biplanar)"),
                    entries: &[
                        bgl_texture_entry(0),
                        bgl_texture_entry(1),
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(
                                wgpu::SamplerBindingType::Filtering,
                            ),
                            count: None,
                        },
                    ],
                })
            }
            RenderLayout::PackedXYUV => {
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("tether-render yuv bgl (444 packed)"),
                    entries: &[
                        bgl_texture_entry(0),
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(
                                wgpu::SamplerBindingType::Filtering,
                            ),
                            count: None,
                        },
                    ],
                })
            }
        };

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

        // Color-params bind group: a `vec4<u32>` whose .x is the EOTF
        // dispatch tag the WGSL fragment shader switches on. The
        // other three slots are reserved for future axes (matrix
        // kind, range kind) when those gain shader variants — today
        // only the transfer is variable. (WGSL's std140 layout pads
        // a bare `u32` in a uniform block to 16 bytes anyway;
        // `vec4<u32>` makes that explicit and the reserved slots
        // self-documenting.)
        let color_params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tether-render color-params bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let color_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tether-render color-params uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transfer_kind = transfer_kind_for(color_space);
        let color_params: [u32; 4] = [transfer_kind, 0, 0, 0];
        queue.write_buffer(&color_params_buffer, 0, &bytes_of_u32x4(&color_params));
        tracing::info!(
            matrix = ?color_space.matrix,
            range = ?color_space.range,
            transfer = ?color_space.transfer,
            primaries = ?color_space.primaries,
            transfer_kind,
            "renderer color spec applied"
        );
        let color_params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tether-render color-params bind group"),
            layout: &color_params_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: color_params_buffer.as_entire_binding(),
            }],
        });

        // Shader matches the bind-group layout. Biplanar uses the
        // chroma-resolution-agnostic Y+UV NV12 shader (works for NV12
        // and NV24 alike — bilinear sampling is the same math at any
        // UV resolution). PackedXYUV uses a separate shader that pulls
        // Y/U/V from the channels of one Rgba8 sample.
        let shader_src = match layout {
            RenderLayout::Biplanar => include_str!("../shader.wgsl"),
            RenderLayout::PackedXYUV => include_str!("../shader_yuv444.wgsl"),
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(match layout {
                RenderLayout::Biplanar => "tether-render shader (biplanar)",
                RenderLayout::PackedXYUV => "tether-render shader (444 packed)",
            }),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tether-render pipeline layout"),
            bind_group_layouts: &[Some(&yuv_bgl), Some(&scale_bgl), Some(&color_params_bgl)],
            immediate_size: 0,
        });

        // Both shader files name their fragment entry point `fs`; the
        // module selected above is what determines which fragment body
        // gets compiled into this pipeline.
        let fragment_entry = "fs";
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
                entry_point: Some(fragment_entry),
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

        let textures = make_yuv_textures(&device, &yuv_bgl, &sampler, chroma, 1, 1);

        Ok(Self {
            surface,
            surface_config,
            device,
            queue,
            pipeline,
            yuv_bgl,
            chroma,
            sampler,
            textures,
            retired: None,
            scale_buffer,
            scale_bind_group,
            color_params_buffer,
            color_params_bind_group,
            #[cfg(target_os = "linux")]
            dmabuf_import_supported,
            #[cfg(target_os = "macos")]
            metal_import_supported,
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
        // CPU upload path is NV12-only today — the only CPU producer
        // (sw-decoded fallback in dmabuf_test) emits NV12. A YUV444
        // CPU producer would need a CpuFrame variant change, not a
        // patch here.
        if self.chroma != ChromaSubsampling::Yuv420 {
            return Err(crate::RenderError::DmaBufImport(format!(
                "CPU upload path is NV12-only; session negotiated {:?}",
                self.chroma
            )));
        }
        // Two reasons to rebuild: the previous frame was DMA-BUF-imported
        // (those textures aren't COPY_DST so write_texture would fail —
        // checking `_guard.is_some()` is load-bearing; the size match
        // would silently skip the rebuild otherwise), or the resolution
        // changed.
        if self.textures._guard.is_some() || (frame.width, frame.height) != self.textures.size {
            let fresh = make_yuv_textures(
                &self.device,
                &self.yuv_bgl,
                &self.sampler,
                self.chroma,
                frame.width,
                frame.height,
            );
            self.retire_textures(fresh);
        }
        let (chroma_w, chroma_h) = frame.chroma_dims();
        let YuvPlanes::Nv12 { y, uv } = &self.textures.planes else {
            unreachable!("guarded by chroma check above");
        };
        // Y is R8 — one byte per texel, so bytes_per_row == width.
        // UV is Rg8 — two bytes per texel, so bytes_per_row == chroma_w * 2.
        write_plane_r8(&self.queue, y, &frame.y, frame.width, frame.height);
        write_plane_rg8(&self.queue, uv, &frame.uv, chroma_w, chroma_h);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn apply_gpu(&mut self, frame: GpuFrame) -> Result<()> {
        // `GpuState::new` hard-errors when the dma-buf feature isn't
        // available on Linux, so reaching this branch with the flag
        // false would mean the constructor's check is broken — not
        // a runtime condition the caller can recover from.
        debug_assert!(
            self.dmabuf_import_supported,
            "apply_gpu reached without dma-buf feature; GpuState::new should have failed"
        );
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
            self.chroma,
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

    #[cfg(target_os = "macos")]
    fn apply_gpu(&mut self, frame: GpuFrame) -> Result<()> {
        // `GpuState::new` hard-errors when the Metal HAL isn't
        // available on macOS, so reaching this branch with the flag
        // false would mean the constructor's check is broken — not
        // a runtime condition the caller can recover from. Mirrors
        // the dma-buf invariant on the Linux side.
        debug_assert!(
            self.metal_import_supported,
            "apply_gpu reached without Metal HAL; GpuState::new should have failed"
        );
        // Match shape mirrors the Linux side: an explicit pattern, not
        // `let else`, so adding a future `GpuFrameSource` variant
        // surfaces here at compile time.
        let iosurface = match &frame.source {
            GpuFrameSource::IOSurface(s) => s,
        };
        let fresh = import_iosurface_textures(
            &self.device,
            &self.yuv_bgl,
            &self.sampler,
            self.chroma,
            iosurface,
            frame.guard,
        )?;
        self.retire_textures(fresh);
        Ok(())
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn apply_gpu(&mut self, _frame: GpuFrame) -> Result<()> {
        // GpuFrameSource has no variants off-Linux/macOS, so a GpuFrame
        // is type-level uninhabitable here; this stub exists to keep
        // the match in `apply_frame` exhaustive across cfgs.
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
            pass.set_bind_group(2, &self.color_params_bind_group, &[]);
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

fn bytes_of_u32x4(v: &[u32; 4]) -> [u8; 16] {
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
    chroma: ChromaSubsampling,
    width: u32,
    height: u32,
) -> YuvTextures {
    match render_layout_for(chroma) {
        RenderLayout::Biplanar => {
            let y = make_plane_texture(
                device,
                "tether-render y plane",
                width,
                height,
                wgpu::TextureFormat::R8Unorm,
            );
            let y_view = y.create_view(&wgpu::TextureViewDescriptor::default());
            // UV plane dims: half-res for NV12 (Yuv420), full-res for
            // NV24 (Yuv444). The same Rg8 plane format works for both —
            // bilinear sampling reads the right value at either size.
            let (chroma_w, chroma_h) = match chroma {
                ChromaSubsampling::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
                ChromaSubsampling::Yuv444 => (width, height),
            };
            let uv = make_plane_texture(
                device,
                "tether-render uv plane",
                chroma_w,
                chroma_h,
                wgpu::TextureFormat::Rg8Unorm,
            );
            let uv_view = uv.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tether-render yuv bind group (biplanar)"),
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
                planes: YuvPlanes::Nv12 { y, uv },
                bind_group,
                size: (width, height),
                _guard: None,
            }
        }
        RenderLayout::PackedXYUV => {
            let packed = make_plane_texture(
                device,
                "tether-render xyuv packed (444)",
                width,
                height,
                wgpu::TextureFormat::Rgba8Unorm,
            );
            let packed_view = packed.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tether-render yuv bind group (444 packed)"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&packed_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            YuvTextures {
                planes: YuvPlanes::Yuv444 { packed },
                bind_group,
                size: (width, height),
                _guard: None,
            }
        }
    }
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


#[cfg(test)]
mod tests {
    use super::{transfer_kind_for, TRANSFER_KIND_BT709, TRANSFER_KIND_SRGB};
    use tether_protocol::control::{ColorTransfer, VideoColorSpec};

    /// Pins the Rust→shader integer mapping so a refactor that
    /// renumbers the WGSL constants (or reorders the Rust match
    /// arms) fails LOUDLY rather than silently rendering with the
    /// wrong EOTF. Cross-language constant agreement is otherwise
    /// only documented in comments; this test makes the Rust side
    /// machine-checked. The WGSL side stays unchecked at default
    /// `cargo test` time (a GPU-readback integration test would
    /// close that loop; deferred).
    ///
    /// Note on exhaustiveness: `transfer_kind_for`'s `match`
    /// enforces exhaustiveness at compile time, so a new
    /// `ColorTransfer` variant fails to build until it's wired into
    /// the function. *This test* doesn't add coverage for that
    /// case — the array below is hand-written and would silently
    /// not exercise a future variant. The test pins the *intended*
    /// mapping (sdr_desktop → Srgb, sdr_bt709 → BT.709, all current
    /// unimplemented variants → BT.709 fallback) so an inadvertent
    /// renumbering fails here.
    #[test]
    fn transfer_kind_for_pins_the_mapping() {
        assert_eq!(transfer_kind_for(VideoColorSpec::sdr_desktop()), TRANSFER_KIND_SRGB);
        assert_eq!(transfer_kind_for(VideoColorSpec::sdr_bt709()), TRANSFER_KIND_BT709);

        // Each currently-unimplemented variant should hit the
        // BT.709 fallback. Keep this array in sync with the
        // `Pq | Hlg | Linear` arm in `transfer_kind_for`; a future
        // variant added there should also be appended here.
        for transfer in [ColorTransfer::Pq, ColorTransfer::Hlg, ColorTransfer::Linear] {
            let spec = VideoColorSpec { transfer, ..VideoColorSpec::sdr_desktop() };
            assert_eq!(transfer_kind_for(spec), TRANSFER_KIND_BT709);
        }
    }
}
