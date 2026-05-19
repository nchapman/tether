use std::sync::Arc;

use winit::window::Window;

use crate::{Frame, RenderError, Result};

/// Y plus chroma textures and the bind group that points at them.
/// Bundled together so resize-on-frame can swap them atomically.
struct YuvTextures {
    y: wgpu::Texture,
    u: wgpu::Texture,
    v: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    /// Y-plane dimensions (chroma is derived).
    size: (u32, u32),
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
    scale_buffer: wgpu::Buffer,
    scale_bind_group: wgpu::BindGroup,
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
            })
            .await
            .map_err(|_| RenderError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tether-render device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await?;

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

        // Three single-channel textures (Y, U, V) plus a sampler. We
        // share one sampler — chroma upsampling uses the same bilinear
        // filter as luma, and the visual difference vs. a chroma-only
        // nearest sampler is invisible on 4:2:0 desktop content.
        let yuv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tether-render yuv bgl"),
            entries: &[
                bgl_texture_entry(0),
                bgl_texture_entry(1),
                bgl_texture_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            scale_buffer,
            scale_bind_group,
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
    }

    pub(crate) fn upload(&mut self, frame: &Frame) {
        if frame.width == 0 || frame.height == 0 {
            return;
        }
        if (frame.width, frame.height) != self.textures.size {
            self.textures = make_yuv_textures(
                &self.device,
                &self.yuv_bgl,
                &self.sampler,
                frame.width,
                frame.height,
            );
        }
        let (chroma_w, chroma_h) = frame.chroma_dims();
        write_plane(&self.queue, &self.textures.y, &frame.y, frame.width, frame.height);
        write_plane(&self.queue, &self.textures.u, &frame.u, chroma_w, chroma_h);
        write_plane(&self.queue, &self.textures.v, &frame.v, chroma_w, chroma_h);
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
    let y = make_r8_texture(device, "tether-render y plane", width, height);
    let u = make_r8_texture(device, "tether-render u plane", chroma_w, chroma_h);
    let v = make_r8_texture(device, "tether-render v plane", chroma_w, chroma_h);
    let y_view = y.create_view(&wgpu::TextureViewDescriptor::default());
    let u_view = u.create_view(&wgpu::TextureViewDescriptor::default());
    let v_view = v.create_view(&wgpu::TextureViewDescriptor::default());
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
                resource: wgpu::BindingResource::TextureView(&u_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&v_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    YuvTextures {
        y,
        u,
        v,
        bind_group,
        size: (width, height),
    }
}

fn make_r8_texture(device: &wgpu::Device, label: &str, width: u32, height: u32) -> wgpu::Texture {
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
        // R8Unorm rather than R8UnormSrgb: YUV values are in a linear
        // encoding space already (the BT.709 limited-range expansion
        // in the fragment shader handles gamma); applying sRGB on
        // sample would double-decode and crush highlights.
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn write_plane(queue: &wgpu::Queue, texture: &wgpu::Texture, bytes: &[u8], width: u32, height: u32) {
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
            bytes_per_row: Some(width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}
