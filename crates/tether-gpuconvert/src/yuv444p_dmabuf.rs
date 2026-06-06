//! BGRA wgpu texture -> planar YUV444P DMA-BUF planes for NVENC.

use std::os::fd::OwnedFd;

use crate::{
    dmabuf_export::{export_yuv444p_shared_dmabuf, ExportError, SharedYuv444pExport},
    pipeline::build_yuv444p_pipeline,
};

pub struct Yuv444pDmaBufFrame {
    pub width: u32,
    pub height: u32,
    pub fd: OwnedFd,
    pub size: u64,
    pub modifier: u64,
    pub y_offset: u64,
    pub y_stride: u64,
    pub u_offset: u64,
    pub u_stride: u64,
    pub v_offset: u64,
    pub v_stride: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum Yuv444pDmaBufError {
    #[error("no wgpu adapter")]
    NoAdapter,
    #[error("wgpu request_device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error(
        "adapter doesn't advertise the features required for zero-copy planar YUV444P \
         (VULKAN_EXTERNAL_MEMORY_DMA_BUF + TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)"
    )]
    FeatureUnsupported,
    #[error("dma-buf export: {0}")]
    Export(#[from] ExportError),
    #[error("input texture format must be Bgra8Unorm or Rgba8Unorm, got {0:?}")]
    InputFormat(wgpu::TextureFormat),
    #[error("input texture dimensions {input_w}x{input_h} don't match converter {w}x{h}")]
    DimMismatch {
        input_w: u32,
        input_h: u32,
        w: u32,
        h: u32,
    },
    #[error("input BGRA byte buffer is {got} bytes; converter at {w}x{h} expects {expected}")]
    ByteLenMismatch {
        got: usize,
        expected: usize,
        w: u32,
        h: u32,
    },
    #[error("wgpu poll: {0}")]
    Poll(String),
    #[error("dup fd: {0}")]
    DupFd(std::io::Error),
}

pub type Result<T> = std::result::Result<T, Yuv444pDmaBufError>;

pub struct Yuv444pDmaBuf {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
    yuv: SharedYuv444pExport,
    y_view: wgpu::TextureView,
    u_view: wgpu::TextureView,
    v_view: wgpu::TextureView,
}

impl Yuv444pDmaBuf {
    pub async fn new(width: u32, height: u32) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| Yuv444pDmaBufError::NoAdapter)?;

        let required_features = wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        if !adapter.features().contains(required_features) {
            return Err(Yuv444pDmaBufError::FeatureUnsupported);
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tether-gpuconvert yuv444p-dmabuf device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await?;

        let (pipeline, bgl) = build_yuv444p_pipeline(&device);
        let yuv = export_yuv444p_shared_dmabuf(
            &device,
            width,
            height,
            wgpu::TextureUsages::STORAGE_BINDING,
        )?;

        let y_view = yuv.y_texture.create_view(&Default::default());
        let u_view = yuv.u_texture.create_view(&Default::default());
        let v_view = yuv.v_texture.create_view(&Default::default());

        Ok(Self {
            device,
            queue,
            pipeline,
            bgl,
            width,
            height,
            yuv,
            y_view,
            u_view,
            v_view,
        })
    }

    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn import_bgra_dmabuf(
        &self,
        fd: OwnedFd,
        modifier: u64,
        stride: u64,
        offset: u64,
        width: u32,
        height: u32,
    ) -> Result<wgpu::Texture> {
        let hal_desc = wgpu::hal::TextureDescriptor {
            label: Some("imported bgra"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUses::RESOURCE,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        let hal_tex = unsafe {
            self.device
                .as_hal::<wgpu::hal::api::Vulkan>()
                .ok_or_else(|| {
                    Yuv444pDmaBufError::Poll(
                        "device is not Vulkan-backed; DMA-BUF import unavailable".into(),
                    )
                })?
                .texture_from_dmabuf_fd(fd, &hal_desc, modifier, stride, offset)
                .map_err(|e| Yuv444pDmaBufError::Poll(format!("texture_from_dmabuf_fd: {e:?}")))?
        };
        let tex = unsafe {
            self.device
                .create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                    hal_tex,
                    &wgpu::TextureDescriptor {
                        label: Some("imported bgra"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Bgra8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                )
        };
        Ok(tex)
    }

    pub fn convert(&self, src_bgra: &wgpu::Texture) -> Result<Yuv444pDmaBufFrame> {
        if !matches!(
            src_bgra.format(),
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
        ) {
            return Err(Yuv444pDmaBufError::InputFormat(src_bgra.format()));
        }
        if src_bgra.width() != self.width || src_bgra.height() != self.height {
            return Err(Yuv444pDmaBufError::DimMismatch {
                input_w: src_bgra.width(),
                input_h: src_bgra.height(),
                w: self.width,
                h: self.height,
            });
        }

        let src_view = src_bgra.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yuv444p-dmabuf bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.u_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.v_view),
                },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("yuv444p-dmabuf enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yuv444p-dmabuf pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        self.queue.submit(Some(enc.finish()));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Yuv444pDmaBufError::Poll(format!("{e:?}")))?;

        Ok(Yuv444pDmaBufFrame {
            width: self.width,
            height: self.height,
            fd: self.yuv.fd.try_clone().map_err(Yuv444pDmaBufError::DupFd)?,
            size: self.yuv.size,
            modifier: self.yuv.modifier,
            y_offset: self.yuv.y_offset,
            y_stride: self.yuv.y_pitch,
            u_offset: self.yuv.u_offset,
            u_stride: self.yuv.u_pitch,
            v_offset: self.yuv.v_offset,
            v_stride: self.yuv.v_pitch,
        })
    }

    pub fn convert_bgra_bytes(&self, bgra: &[u8]) -> Result<Yuv444pDmaBufFrame> {
        let expected = (self.width as usize) * (self.height as usize) * 4;
        if bgra.len() != expected {
            return Err(Yuv444pDmaBufError::ByteLenMismatch {
                got: bgra.len(),
                expected,
                w: self.width,
                h: self.height,
            });
        }
        let src = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("yuv444p bgra scratch"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bgra,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.convert(&src)
    }
}
