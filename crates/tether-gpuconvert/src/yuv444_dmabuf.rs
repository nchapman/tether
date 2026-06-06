//! BGRA wgpu texture → packed XYUV 4:4:4 DMA-BUF, via compute shader
//! + single-plane shared export.
//!
//! Mirror of [`crate::nv12_dmabuf::Nv12DmaBuf`]. Same shape — open a
//! wgpu device, build the compute pipeline once, allocate the shared
//! DMA-BUF target once, run one compute pass per frame and hand back
//! dup'd fds. The differences are all in the output: 4:4:4 has three
//! full-resolution R8 planes instead of NV12's R8 + Rg8 at half-chroma,
//! and the consumer side (VAAPI HEVC Main444) expects DRM_FORMAT_XYUV8888
//! `XYUV` as the layer fourcc rather than `NV12`.
//!
//! This is the path that feeds [`tether_codec::VaapiEncoder`] when the
//! handshake negotiates HEVC Main444. NVIDIA's planar `YU24` path lives
//! in `yuv444p_dmabuf`.

use std::os::fd::OwnedFd;

use crate::{
    dmabuf_export::{export_yuv444_shared_dmabuf, ExportError, SharedYuv444Export},
    pipeline::build_yuv444_pipeline,
};

/// One frame's worth of packed XYUV — a single dma-buf fd carrying one
/// full-resolution 32-bpp plane.
///
/// `fd` is a dup'd copy of the bridge's persistent export. The bridge
/// keeps the underlying memory alive via its own owned export for as
/// long as the bridge exists; this frame is safe to hand off and let
/// the consumer drop independently.
///
/// Field shape matches what `tether_codec::DmaBufFrame` needs for
/// `av_hwframe_map(DRM_PRIME → VAAPI Main444)`: one DRM object, one
/// layer (`DRM_FORMAT_XYUV8888`), one plane (packed 4 bytes/pixel,
/// memory order V,U,Y,X — see shared_yuv444.rs for why packed not
/// planar).
pub struct Yuv444DmaBufFrame {
    pub width: u32,
    pub height: u32,
    pub fd: OwnedFd,
    pub size: u64,
    pub modifier: u64,
    pub offset: u64,
    pub stride: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum Yuv444DmaBufError {
    #[error("no wgpu adapter")]
    NoAdapter,
    #[error("wgpu request_device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error(
        "adapter doesn't advertise the features required for zero-copy YUV 4:4:4 \
         (VULKAN_EXTERNAL_MEMORY_DMA_BUF + TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES). \
         The Vulkan ICD must support VK_EXT_external_memory_dma_buf, \
         VK_EXT_image_drm_format_modifier, and VK_KHR_external_memory_fd."
    )]
    FeatureUnsupported,
    #[error("dma-buf export: {0}")]
    Export(#[from] ExportError),
    /// Source view must be Bgra8Unorm (capture path) or Rgba8Unorm
    /// (tether-scaler output). Both present as shader-visible RGBA so
    /// the chroma shader's `.rgb` swizzle is format-agnostic.
    #[error("input texture format must be Bgra8Unorm or Rgba8Unorm, got {0:?}")]
    InputFormat(wgpu::TextureFormat),
    #[error("input texture dimensions {input_w}x{input_h} don't match converter {w}x{h}")]
    DimMismatch {
        input_w: u32,
        input_h: u32,
        w: u32,
        h: u32,
    },
    #[error("wgpu poll: {0}")]
    Poll(String),
    #[error("dup fd: {0}")]
    DupFd(std::io::Error),
}

pub type Result<T> = std::result::Result<T, Yuv444DmaBufError>;

/// Persistent BGRA → YUV 4:4:4 converter writing into DMA-BUF-exported
/// Y/U/V textures. Built once per resolution; the per-frame call takes
/// an imported BGRA wgpu texture (from PipeWire) and returns dup'd fds
/// the encoder consumes via `VaapiEncoder::submit_dmabuf`.
pub struct Yuv444DmaBuf {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
    yuv: SharedYuv444Export,
    packed_view: wgpu::TextureView,
}

impl Yuv444DmaBuf {
    /// Open a wgpu device with the required features and allocate the
    /// 3-plane shared dma-buf export sized for `width`×`height`.
    pub async fn new(width: u32, height: u32) -> Result<Self> {
        let instance = crate::headless_wgpu_instance();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| Yuv444DmaBufError::NoAdapter)?;

        let required_features = wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        if !adapter.features().contains(required_features) {
            return Err(Yuv444DmaBufError::FeatureUnsupported);
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tether-gpuconvert yuv444-dmabuf device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await?;

        let (pipeline, bgl) = build_yuv444_pipeline(&device);

        let yuv = export_yuv444_shared_dmabuf(
            &device,
            width,
            height,
            wgpu::TextureUsages::STORAGE_BINDING,
        )?;

        let packed_view = yuv.packed_texture.create_view(&Default::default());

        Ok(Self {
            device,
            queue,
            pipeline,
            bgl,
            width,
            height,
            yuv,
            packed_view,
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

    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Import a PipeWire-supplied BGRA DMA-BUF as a wgpu texture on
    /// this bridge's device. Same contract as
    /// [`crate::nv12_dmabuf::Nv12DmaBuf::import_bgra_dmabuf`].
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
        // SAFETY: caller asserts fd is a valid DMA-BUF fd we own and
        // that modifier/stride/offset describe the same image. wgpu-hal
        // validates Vulkan-side; failure returns Err.
        let hal_tex = unsafe {
            self.device
                .as_hal::<wgpu::hal::api::Vulkan>()
                .ok_or_else(|| {
                    Yuv444DmaBufError::Poll(
                        "device is not Vulkan-backed; DMA-BUF import unavailable".into(),
                    )
                })?
                .texture_from_dmabuf_fd(fd, &hal_desc, modifier, stride, offset)
                .map_err(|e| Yuv444DmaBufError::Poll(format!("texture_from_dmabuf_fd: {e:?}")))?
        };
        let wgpu_desc = wgpu::TextureDescriptor {
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
        };
        // SAFETY: hal_tex just built from this device, descs match.
        let tex = unsafe {
            self.device
                .create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_tex, &wgpu_desc)
        };
        Ok(tex)
    }

    /// Convert one imported BGRA frame into the bridge's YUV 4:4:4
    /// DMA-BUF targets and return dup'd fds the caller wraps in a
    /// codec `DmaBufFrame` for `VaapiEncoder::submit_dmabuf`.
    ///
    /// Blocks on GPU completion before returning — see the
    /// [`crate::nv12_dmabuf`] module comment for why we don't yet use
    /// an explicit sync_file fence.
    pub fn convert(&self, src_bgra: &wgpu::Texture) -> Result<Yuv444DmaBufFrame> {
        if !matches!(
            src_bgra.format(),
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
        ) {
            return Err(Yuv444DmaBufError::InputFormat(src_bgra.format()));
        }
        if src_bgra.width() != self.width || src_bgra.height() != self.height {
            return Err(Yuv444DmaBufError::DimMismatch {
                input_w: src_bgra.width(),
                input_h: src_bgra.height(),
                w: self.width,
                h: self.height,
            });
        }

        let src_view = src_bgra.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yuv444-dmabuf bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.packed_view),
                },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("yuv444-dmabuf enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yuv444-dmabuf pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // 8x8 workgroup, one invocation per output pixel — no
            // chroma collapse in 4:4:4.
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        self.queue.submit(Some(enc.finish()));

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Yuv444DmaBufError::Poll(format!("{e:?}")))?;

        Ok(Yuv444DmaBufFrame {
            width: self.width,
            height: self.height,
            fd: self.yuv.fd.try_clone().map_err(Yuv444DmaBufError::DupFd)?,
            size: self.yuv.size,
            modifier: self.yuv.modifier,
            offset: self.yuv.offset,
            stride: self.yuv.pitch,
        })
    }
}

#[cfg(test)]
mod tests {
    // Readback pixel-math casts (u64 offset → usize) are intentional.
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

    use super::*;

    /// Read one R8 plane back from the shared dma-buf by re-importing
    /// it at the given offset/stride, copying to a mappable buffer,
    /// and returning a tightly-packed `Vec<u8>`. Used by the round-trip
    /// tests to verify Y / U / V (extracted from the packed plane).
    fn read_packed(
        bridge: &Yuv444DmaBuf,
        fd: OwnedFd,
        modifier: u64,
        stride: u64,
        offset: u64,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let import_desc = wgpu::hal::TextureDescriptor {
            label: Some("xyuv reimport"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUses::COPY_SRC,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        // SAFETY: caller provides authoritative fd / modifier / stride / offset
        // for the packed plane just exported by this bridge.
        let hal_tex = unsafe {
            bridge
                .device()
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("vulkan backend")
                .texture_from_dmabuf_fd(fd, &import_desc, modifier, stride, offset)
                .expect("packed texture_from_dmabuf_fd")
        };
        let import_tex = unsafe {
            bridge
                .device()
                .create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                    hal_tex,
                    &wgpu::TextureDescriptor {
                        label: Some("xyuv reimport"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    },
                )
        };

        let bytes_per_row = width * 4;
        let padded_row = u64::from(bytes_per_row).div_ceil(256) * 256;
        let readback = bridge.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("xyuv readback"),
            size: padded_row * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = bridge
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("xyuv readback enc"),
            });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &import_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(u32::try_from(padded_row).unwrap()),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        bridge.queue().submit(Some(enc.finish()));
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        bridge
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll packed readback");
        rx.recv().expect("cb").expect("map");
        let mapped = slice.get_mapped_range().expect("range");

        let mut out = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height {
            let start = (u64::from(row) * padded_row) as usize;
            out.extend_from_slice(&mapped[start..start + bytes_per_row as usize]);
        }
        out
    }

    /// End-to-end without VAAPI: build the bridge, push solid-colour
    /// BGRA through it, re-import the packed XYUV plane, read it back,
    /// and assert each per-pixel quad has the expected byte order
    /// [V, U, Y, X] matching DRM_FORMAT_XYUV8888 + BT.709 limited
    /// range. Catches both colour-math and channel-order regressions
    /// — a shader that wrote (Y, U, V, 1.0) into RGBA would still get
    /// the right Y for solid white but the U/V channels would land in
    /// the wrong memory positions.
    #[test]
    #[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF"]
    fn convert_solid_white_roundtrip_packed_xyuv() {
        let width = 64u32;
        let height = 32u32;

        let bridge = match pollster::block_on(Yuv444DmaBuf::new(width, height)) {
            Ok(b) => b,
            Err(Yuv444DmaBufError::NoAdapter | Yuv444DmaBufError::FeatureUnsupported) => {
                eprintln!("SKIP: no wgpu adapter with DMA-BUF export feature");
                return;
            }
            Err(e) => panic!("Yuv444DmaBuf::new: {e}"),
        };

        let src = bridge.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("test bgra white"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let n = (width * height) as usize;
        let bgra = vec![255u8; n * 4];
        bridge.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bgra,
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

        let out = bridge.convert(&src).expect("convert");
        assert_eq!(out.offset, bridge.yuv.offset);
        assert_eq!(out.stride, bridge.yuv.pitch);

        let packed = read_packed(
            &bridge,
            out.fd.try_clone().expect("dup fd"),
            out.modifier,
            out.stride,
            out.offset,
            width,
            height,
        );

        // BT.709 limited-range for pure white: Y=235, U=V=128.
        // DRM_FORMAT_XYUV8888 byte order: [V, U, Y, X].
        let assert_near = |label: &str, got: u8, expected: u8| {
            let diff = i32::from(got) - i32::from(expected);
            assert!(
                diff.abs() <= 2,
                "{label} = {got}, expected ~{expected} (diff {diff})"
            );
        };
        for &(x, y) in &[
            (0u32, 0u32),
            (width / 2, height / 2),
            (width - 1, height - 1),
        ] {
            let idx = ((y * width + x) * 4) as usize;
            assert_near(&format!("V[{x},{y}]"), packed[idx], 128);
            assert_near(&format!("U[{x},{y}]"), packed[idx + 1], 128);
            assert_near(&format!("Y[{x},{y}]"), packed[idx + 2], 235);
            // X is don't-care under XYUV8888; shader writes 0xFF
            // (the alpha channel) but a consumer must not rely on it.
        }
    }

    /// Chroma-distinguishing sibling of the white cell. White has
    /// U=V=128, so it cannot catch a Cb↔Cr swap or a wrong chroma matrix
    /// row — only the byte *position* (V vs U) and luma. Pure red has
    /// very distinct Cb vs Cr (102 vs 240), so this cell pins the actual
    /// chroma math and a U↔V swap. Asserts against the BT.709 limited-
    /// range spec directly (the compliant reference), not a round-trip.
    #[test]
    #[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF"]
    fn convert_solid_red_roundtrip_packed_xyuv() {
        let width = 64u32;
        let height = 32u32;

        let bridge = match pollster::block_on(Yuv444DmaBuf::new(width, height)) {
            Ok(b) => b,
            Err(Yuv444DmaBufError::NoAdapter | Yuv444DmaBufError::FeatureUnsupported) => {
                eprintln!("SKIP: no wgpu adapter with DMA-BUF export feature");
                return;
            }
            Err(e) => panic!("Yuv444DmaBuf::new: {e}"),
        };

        let src = bridge.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("test bgra red"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let n = (width * height) as usize;
        // BGRA layout: B, G, R, A. Pure red = (0, 0, 255, 255).
        let mut bgra = Vec::with_capacity(n * 4);
        for _ in 0..n {
            bgra.extend_from_slice(&[0, 0, 255, 255]);
        }
        bridge.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bgra,
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

        let out = bridge.convert(&src).expect("convert");

        let packed = read_packed(
            &bridge,
            out.fd.try_clone().expect("dup fd"),
            out.modifier,
            out.stride,
            out.offset,
            width,
            height,
        );

        // BT.709 limited-range 8-bit, pure red (R=1, G=0, B=0):
        //   Y  = round(0.2126 * 219 + 16)           = 63
        //   Cb = round(-0.11457 * 224 + 128)        = 102
        //   Cr = round( 0.50000 * 224 + 128)        = 240
        // DRM_FORMAT_XYUV8888 byte order [V, U, Y, X]: Cr/Cb are far
        // enough apart (240 vs 102) that a U↔V swap fails loudly.
        let assert_near = |label: &str, got: u8, expected: u8| {
            let diff = i32::from(got) - i32::from(expected);
            assert!(
                diff.abs() <= 2,
                "{label} = {got}, expected ~{expected} (diff {diff})"
            );
        };
        for &(x, y) in &[
            (0u32, 0u32),
            (width / 2, height / 2),
            (width - 1, height - 1),
        ] {
            let idx = ((y * width + x) * 4) as usize;
            assert_near(&format!("V[{x},{y}]"), packed[idx], 240);
            assert_near(&format!("U[{x},{y}]"), packed[idx + 1], 102);
            assert_near(&format!("Y[{x},{y}]"), packed[idx + 2], 63);
        }
    }
}
