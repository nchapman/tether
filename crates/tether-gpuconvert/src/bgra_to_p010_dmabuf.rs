//! BGRA wgpu texture → P010 DMA-BUF planes, via 10-bit compute shader +
//! shared export.
//!
//! Mirror of [`crate::nv12_dmabuf::Nv12DmaBuf`] for the HEVC Main10
//! (4:2:0 10-bit) production path. Same shape: open a wgpu device,
//! build the compute pipeline once, allocate the shared DMA-BUF target
//! once, run one compute pass per frame and hand back dup'd fds the
//! VAAPI encoder consumes.
//!
//! Differences from the NV12 path are all in the output plane formats:
//! `R16Unorm` Y + `Rg16Unorm` UV (10-bit data MSB-aligned in 16-bit
//! cells; see `bgra_to_p010.wgsl` for the storage convention) and DRM
//! fourcc family R16 / GR32 instead of R8 / GR88. The encoder side
//! consumes this as `AV_PIX_FMT_P010LE` via `av_hwframe_map(DRM_PRIME →
//! VAAPI)` — the only 10-bit 4:2:0 entry in ffmpeg's
//! `vaapi_drm_format_map`.
//!
//! Capability gate: the constructor runs [`crate::storable_dmabuf_modifiers`]
//! against R16 + GR32 before allocating the export. Some drivers
//! advertise 16-bit unorm as sampleable but not storage-writable; on
//! those, `create_compute_pipeline` would fail mid-session with a
//! validation error. Probing storage support up front lets the host
//! refuse construction loudly (matching the bridge-fatal-init contract
//! in `CLAUDE.md`) so the 10-bit profile gets filtered out of the
//! negotiation set before a session starts.

use std::os::fd::OwnedFd;

use crate::{
    dmabuf_export::{
        export_p010_shared_dmabuf, ExportError, SharedP010Export, DRM_FORMAT_MOD_LINEAR,
    },
    modifier_query::ModifierQueryError,
    pipeline::build_p010_pipeline,
};

/// One frame's worth of P010 — a single DMA-BUF fd carrying both planes
/// at distinct offsets within one shared `VkDeviceMemory` allocation.
/// Same shape as [`crate::nv12_dmabuf::Nv12DmaBufFrame`] — the encoder
/// side keys on per-plane offsets/strides and doesn't care about the
/// 8 vs 16 bit cell size.
pub struct P010DmaBufFrame {
    pub width: u32,
    pub height: u32,
    pub fd: OwnedFd,
    pub size: u64,
    pub modifier: u64,
    pub y_offset: u64,
    pub y_stride: u64,
    pub uv_offset: u64,
    pub uv_stride: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum P010DmaBufError {
    #[error("no wgpu adapter")]
    NoAdapter,
    #[error("wgpu request_device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error(
        "adapter doesn't advertise the features required for zero-copy P010 \
         (VULKAN_EXTERNAL_MEMORY_DMA_BUF + TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES \
         + TEXTURE_FORMAT_16BIT_NORM). The Vulkan ICD must support \
         VK_EXT_external_memory_dma_buf, VK_EXT_image_drm_format_modifier, \
         and VK_KHR_external_memory_fd."
    )]
    FeatureUnsupported,
    #[error(
        "driver doesn't advertise STORAGE_IMAGE on {plane} (DRM fourcc {fourcc}) \
         for DRM_FORMAT_MOD_LINEAR — the 10-bit compute shader needs to write \
         R16Unorm / Rg16Unorm storage textures. The 10-bit encode profile \
         should be filtered out before session negotiation; this error means \
         the upstream filter missed."
    )]
    StorageUnsupported {
        plane: &'static str,
        fourcc: &'static str,
    },
    #[error("storage-modifier probe: {0}")]
    ProbeFailed(#[from] ModifierQueryError),
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

pub type Result<T> = std::result::Result<T, P010DmaBufError>;

/// DRM fourccs for the 16-bit Y / UV planes of P010. The probe and the
/// downstream encoder both key on these.
const DRM_FOURCC_R16: u32 = u32::from_le_bytes(*b"R16 ");
const DRM_FOURCC_GR32: u32 = u32::from_le_bytes(*b"GR32");

/// Persistent BGRA→P010 converter writing into DMA-BUF-exported Y/UV
/// textures. Built once per resolution; the per-frame call takes an
/// imported BGRA wgpu texture (from PipeWire) and returns dup'd fds the
/// encoder consumes via `VaapiEncoder::submit_dmabuf`.
pub struct Bgra2P010DmaBuf {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
    p010: SharedP010Export,
    y_view: wgpu::TextureView,
    uv_view: wgpu::TextureView,
}

impl Bgra2P010DmaBuf {
    /// Open a wgpu device with the required features and allocate the
    /// shared P010 export sized for `width`×`height`. Probes
    /// `STORAGE_IMAGE` modifier support on R16 + GR32 up front — see
    /// the module comment for why.
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
            .map_err(|_| P010DmaBufError::NoAdapter)?;

        // TEXTURE_FORMAT_16BIT_NORM gates the wgpu-side ability to bind
        // R16Unorm / Rg16Unorm at all (storage *or* sampled). Same
        // feature the renderer's 10-bit import path opts in to —
        // commit 942ba53 added it on the client.
        let required_features = wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
        if !adapter.features().contains(required_features) {
            return Err(P010DmaBufError::FeatureUnsupported);
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tether-gpuconvert p010-dmabuf device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await?;

        // STORAGE_IMAGE_BIT gate on both planes. Without this, the
        // compute pipeline build may pass on adapter capability but
        // fail validation when the storage write actually issues. The
        // probe opens a transient device under the hood — non-trivial
        // cost, but it runs once per bridge construction (per session).
        //
        // Multi-GPU caveat: the probe selects its own adapter via the
        // same `HighPerformance + no surface` constraint we use above,
        // which is deterministic on a single-GPU box. On a system with
        // multiple Vulkan ICDs (e.g. mesa + nvidia, or hybrid laptops),
        // the probe and the bridge could pick different physical
        // devices. Acceptable for v1 because every shipping Linux GPU
        // either supports R16/Rg16 storage everywhere or nowhere; if a
        // hybrid path shows up, the probe needs to thread the adapter
        // through instead.
        let y_mods = crate::modifier_query::storable_dmabuf_modifiers(DRM_FOURCC_R16).await?;
        if !y_mods.contains(&DRM_FORMAT_MOD_LINEAR) {
            return Err(P010DmaBufError::StorageUnsupported {
                plane: "Y",
                fourcc: "R16",
            });
        }
        let uv_mods = crate::modifier_query::storable_dmabuf_modifiers(DRM_FOURCC_GR32).await?;
        if !uv_mods.contains(&DRM_FORMAT_MOD_LINEAR) {
            return Err(P010DmaBufError::StorageUnsupported {
                plane: "UV",
                fourcc: "GR32",
            });
        }

        let (pipeline, bgl) = build_p010_pipeline(&device);

        let p010 = export_p010_shared_dmabuf(
            &device,
            width,
            height,
            wgpu::TextureUsages::STORAGE_BINDING,
        )?;

        let y_view = p010.y_texture.create_view(&Default::default());
        let uv_view = p010.uv_texture.create_view(&Default::default());

        Ok(Self {
            device,
            queue,
            pipeline,
            bgl,
            width,
            height,
            p010,
            y_view,
            uv_view,
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
        // validates Vulkan-side; failure returns Err rather than UB.
        let hal_tex = unsafe {
            self.device
                .as_hal::<wgpu::hal::api::Vulkan>()
                .ok_or_else(|| {
                    P010DmaBufError::Poll(
                        "device is not Vulkan-backed; DMA-BUF import unavailable".into(),
                    )
                })?
                .texture_from_dmabuf_fd(fd, &hal_desc, modifier, stride, offset)
                .map_err(|e| P010DmaBufError::Poll(format!("texture_from_dmabuf_fd: {e:?}")))?
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
        // SAFETY: hal_tex was built from this device's Vulkan backend
        // moments ago and descs match in shape.
        let tex = unsafe {
            self.device
                .create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_tex, &wgpu_desc)
        };
        Ok(tex)
    }

    /// Convenience wrapper around [`Self::convert`] that takes a
    /// tightly-packed BGRA byte slice (`width * height * 4` bytes).
    /// Allocates a transient source texture on the bridge's device,
    /// uploads the bytes via `write_texture`, and runs the compute
    /// pass. Returns the resulting P010 dma-buf frame.
    ///
    /// For callers that already have a wgpu BGRA texture (production
    /// PipeWire path: imported via `import_bgra_dmabuf`), use
    /// [`Self::convert`] directly to skip the upload. This helper
    /// exists for test + probe contexts that have CPU-resident bytes
    /// and don't want a wgpu dep in their crate just to allocate a
    /// scratch source texture.
    pub fn convert_bgra_bytes(&self, bgra: &[u8]) -> Result<P010DmaBufFrame> {
        let expected = (self.width * self.height * 4) as usize;
        if bgra.len() != expected {
            return Err(P010DmaBufError::ByteLenMismatch {
                got: bgra.len(),
                expected,
                w: self.width,
                h: self.height,
            });
        }
        let src = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("p010 bgra scratch"),
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

    /// Convert one imported BGRA frame into the bridge's P010 DMA-BUF
    /// targets and return dup'd fds the caller wraps in a codec
    /// `DmaBufFrame` for `VaapiEncoder::submit_dmabuf`.
    pub fn convert(&self, src_bgra: &wgpu::Texture) -> Result<P010DmaBufFrame> {
        if !matches!(
            src_bgra.format(),
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
        ) {
            return Err(P010DmaBufError::InputFormat(src_bgra.format()));
        }
        if src_bgra.width() != self.width || src_bgra.height() != self.height {
            return Err(P010DmaBufError::DimMismatch {
                input_w: src_bgra.width(),
                input_h: src_bgra.height(),
                w: self.width,
                h: self.height,
            });
        }

        let src_view = src_bgra.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("p010-dmabuf bg"),
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
                    resource: wgpu::BindingResource::TextureView(&self.uv_view),
                },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("p010-dmabuf enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("p010-dmabuf pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let chroma_w = self.width.div_ceil(2);
            let chroma_h = self.height.div_ceil(2);
            pass.dispatch_workgroups(chroma_w.div_ceil(8), chroma_h.div_ceil(8), 1);
        }
        self.queue.submit(Some(enc.finish()));

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| P010DmaBufError::Poll(format!("{e:?}")))?;

        Ok(P010DmaBufFrame {
            width: self.width,
            height: self.height,
            fd: self.p010.fd.try_clone().map_err(P010DmaBufError::DupFd)?,
            size: self.p010.size,
            modifier: self.p010.modifier,
            y_offset: self.p010.y_offset,
            y_stride: self.p010.y_pitch,
            uv_offset: self.p010.uv_offset,
            uv_stride: self.p010.uv_pitch,
        })
    }
}

#[cfg(test)]
mod tests {
    // Readback pixel-math casts (u64 offset → usize) are intentional.
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

    use super::*;

    /// End-to-end without VAAPI: build the bridge, push solid-colour
    /// BGRA through it, re-import the Y plane via `texture_from_dmabuf_fd`
    /// with `R16Unorm` format, read it back as u16 values, and verify
    /// the MSB-aligned 10-bit Y bytes match the BT.709 limited-range
    /// 10-bit prediction for that colour.
    ///
    /// Solid red was chosen for parity with the NV12 sibling test
    /// (`convert_solid_red_roundtrip`). The shader's clamp ceiling
    /// (`Y_STORAGE_CEILING = 60160/65535`) doesn't activate at this
    /// luma — pure red Y is well inside the limited-range envelope.
    #[test]
    #[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF + storage-writable R16/Rg16"]
    fn convert_solid_red_roundtrip_p010() {
        let width = 64u32;
        let height = 32u32;

        let bridge = match pollster::block_on(Bgra2P010DmaBuf::new(width, height)) {
            Ok(b) => b,
            Err(
                P010DmaBufError::NoAdapter
                | P010DmaBufError::FeatureUnsupported
                | P010DmaBufError::StorageUnsupported { .. },
            ) => {
                eprintln!("SKIP: driver doesn't host the 10-bit gpuconvert path");
                return;
            }
            Err(e) => panic!("Bgra2P010DmaBuf::new: {e}"),
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
        let mut bgra = Vec::with_capacity(n * 4);
        for _ in 0..n {
            bgra.extend_from_slice(&[0, 0, 255, 255]); // red
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

        let out = bridge.convert(&src).expect("bridge convert");

        // Re-import Y plane as R16Unorm.
        let import_desc = wgpu::hal::TextureDescriptor {
            label: Some("y reimport"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Unorm,
            usage: wgpu::TextureUses::COPY_SRC,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        // SAFETY: out.fd / modifier / stride / offset were just produced
        // by the bridge's export path so the values are authoritative.
        let hal_tex = unsafe {
            bridge
                .device()
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("vulkan backend")
                .texture_from_dmabuf_fd(
                    out.fd
                        .try_clone()
                        .expect("dup shared P010 fd for Y reimport"),
                    &import_desc,
                    out.modifier,
                    out.y_stride,
                    out.y_offset,
                )
                .expect("y texture_from_dmabuf_fd")
        };
        let import_tex = unsafe {
            bridge
                .device()
                .create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                    hal_tex,
                    &wgpu::TextureDescriptor {
                        label: Some("y reimport"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::R16Unorm,
                        usage: wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    },
                )
        };

        // R16Unorm is 2 bytes/texel. wgpu still requires 256-byte row
        // alignment on copy_texture_to_buffer.
        let row_bytes = u64::from(width) * 2;
        let padded_row = row_bytes.div_ceil(256) * 256;
        let readback = bridge.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("y readback"),
            size: padded_row * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = bridge
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("y readback enc"),
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
            .expect("poll readback");
        rx.recv().expect("cb").expect("map");
        let mapped = slice.get_mapped_range().expect("range");

        // BT.709 limited-range Y for pure red in 10-bit:
        //   Y' = 0.2126 (full-range linear coeff)
        //   limited 10-bit = round(Y' * 876 + 64) = round(186.2 + 64) = 250
        //
        // We check the top 10 bits (cell >> 6) — that's the actual
        // 10-bit Y value that P010 consumers read. The shader's
        // float-math + R16Unorm round-to-nearest path is not bit-exact
        // for the bottom 6 bits (those are sub-LSB residue in 10-bit
        // space; the spec-mandated MSB-align contract only applies to
        // the clamp ceiling, not interior values), so asserting them
        // would be over-specifying the storage convention.
        let expected_y_10bit: u16 = 250;

        for &(x, y) in &[
            (0u32, 0u32),
            (width / 2, height / 2),
            (width - 1, height - 1),
        ] {
            let off = (u64::from(y) * padded_row + u64::from(x) * 2) as usize;
            let lo = mapped[off];
            let hi = mapped[off + 1];
            let cell = u16::from_le_bytes([lo, hi]);
            let y_10bit = cell >> 6;
            let diff = i32::from(y_10bit) - i32::from(expected_y_10bit);
            assert!(
                diff.abs() <= 1, // ±1 LSB in 10-bit space
                "Y[{x},{y}] = {y_10bit} (cell 0x{cell:04x}), expected ~{expected_y_10bit}",
            );
        }
        // Drop the Y mapping/buffer before re-using the readback path
        // for UV, so the second slice.map_async doesn't trip on the
        // first range still being held.
        drop(mapped);
        readback.unmap();

        // UV plane verification — without this, a 2×2 chroma block
        // off-by-one or a wrong UV_SCALE/UV_OFFSET would slip past the
        // Y-only check. Solid red gives a well-defined (Cb, Cr) pair
        // that triggers the V clamp at 960 (UV_STORAGE_CEILING) so the
        // test also exercises the saturation path.
        //
        // BT.709 limited-range Cb/Cr for pure red (R=1, G=0, B=0):
        //   Cb_centered = U_R * 1 = -0.11457
        //     stored 10-bit = round(-0.11457 * 448 + 512) = round(409.7) = 409
        //   Cr_centered = V_R * 1 = 0.5
        //     stored 10-bit = round(0.5 * 448 + 512) = 736 (raw spec)
        //     -- but the shader saturates at UV_STORAGE_CEILING=960 — wait,
        //     this is wrong. Recompute against the shader's actual math:
        //   shader stores (UV_centered * 57344 + 32768) / 65535 (normalised),
        //   textureStore on Rg16Unorm rounds × 65535; top 10 bits read back.
        //   For Cr = 0.5: cell = round(0.5 * 57344 + 32768) = round(61440) = 61440.
        //                 top 10 bits = 61440 >> 6 = 960 (this IS the
        //                 UV_STORAGE_CEILING clamp engaging).
        //   For Cb = -0.11457: cell = round(-0.11457 * 57344 + 32768)
        //                          = round(-6569.2 + 32768) = 26199.
        //                       top 10 bits = 26199 >> 6 = 409.
        let chroma_w = width.div_ceil(2);
        let chroma_h = height.div_ceil(2);
        let uv_import_desc = wgpu::hal::TextureDescriptor {
            label: Some("uv reimport"),
            size: wgpu::Extent3d {
                width: chroma_w,
                height: chroma_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Unorm,
            usage: wgpu::TextureUses::COPY_SRC,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        // SAFETY: out.fd / modifier / uv_stride / uv_offset come from
        // the bridge's own export; they're the authoritative descriptor
        // for the UV plane within the shared allocation.
        let uv_hal_tex = unsafe {
            bridge
                .device()
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("vulkan backend")
                .texture_from_dmabuf_fd(
                    out.fd
                        .try_clone()
                        .expect("dup shared P010 fd for UV reimport"),
                    &uv_import_desc,
                    out.modifier,
                    out.uv_stride,
                    out.uv_offset,
                )
                .expect("uv texture_from_dmabuf_fd")
        };
        let uv_import_tex = unsafe {
            bridge
                .device()
                .create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                    uv_hal_tex,
                    &wgpu::TextureDescriptor {
                        label: Some("uv reimport"),
                        size: wgpu::Extent3d {
                            width: chroma_w,
                            height: chroma_h,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rg16Unorm,
                        usage: wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    },
                )
        };

        // Rg16Unorm is 4 bytes/texel (two 16-bit channels). 256-byte
        // row align still applies for copy_texture_to_buffer.
        let uv_row_bytes = u64::from(chroma_w) * 4;
        let uv_padded_row = uv_row_bytes.div_ceil(256) * 256;
        let uv_readback = bridge.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("uv readback"),
            size: uv_padded_row * u64::from(chroma_h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut uv_enc = bridge
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("uv readback enc"),
            });
        uv_enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &uv_import_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &uv_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(u32::try_from(uv_padded_row).unwrap()),
                    rows_per_image: Some(chroma_h),
                },
            },
            wgpu::Extent3d {
                width: chroma_w,
                height: chroma_h,
                depth_or_array_layers: 1,
            },
        );
        bridge.queue().submit(Some(uv_enc.finish()));
        let uv_slice = uv_readback.slice(..);
        let (uv_tx, uv_rx) = std::sync::mpsc::channel();
        uv_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = uv_tx.send(r);
        });
        bridge
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll uv readback");
        uv_rx.recv().expect("cb").expect("map");
        let uv_mapped = uv_slice.get_mapped_range().expect("range");

        let expected_cb_10bit: u16 = 409;
        let expected_cr_10bit: u16 = 960;
        for &(x, y) in &[
            (0u32, 0u32),
            (chroma_w / 2, chroma_h / 2),
            (chroma_w - 1, chroma_h - 1),
        ] {
            let off = (u64::from(y) * uv_padded_row + u64::from(x) * 4) as usize;
            let cb_cell = u16::from_le_bytes([uv_mapped[off], uv_mapped[off + 1]]);
            let cr_cell = u16::from_le_bytes([uv_mapped[off + 2], uv_mapped[off + 3]]);
            let cb = cb_cell >> 6;
            let cr = cr_cell >> 6;
            let cb_diff = i32::from(cb) - i32::from(expected_cb_10bit);
            let cr_diff = i32::from(cr) - i32::from(expected_cr_10bit);
            assert!(
                cb_diff.abs() <= 1,
                "Cb[{x},{y}] = {cb} (cell 0x{cb_cell:04x}), expected ~{expected_cb_10bit}",
            );
            assert!(
                cr_diff.abs() <= 1,
                "Cr[{x},{y}] = {cr} (cell 0x{cr_cell:04x}), expected ~{expected_cr_10bit}",
            );
        }
    }
}
