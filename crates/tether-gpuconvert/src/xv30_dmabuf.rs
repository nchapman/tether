//! BGRA wgpu texture → packed XV30 DMA-BUF, via 10-bit compute shader
//! + single-plane shared export.
//!
//! Mirror of [`crate::yuv444_dmabuf::Yuv444DmaBuf`] for the HEVC Main
//! 4:4:4 10-bit (XV30) production path. Same single-plane shape — the
//! consumer (VAAPI encoder) takes one DRM object with one layer and one
//! plane; only the bit depth and Vulkan format differ.
//!
//! Why packed and not biplanar P410: ffmpeg's `vaapi_drm_format_map`
//! carries no P410 entry; `av_hwframe_map(DRM_PRIME → VAAPI)` rejects
//! biplanar 10-bit 4:4:4 descriptors. DRM_FORMAT_XV30 is the only
//! 4:4:4 10-bit format VAAPI accepts as encode input on Linux.
//!
//! Capability gate at construction: probe `STORAGE_IMAGE` modifier
//! support on `DRM_FORMAT_XV30` against `DRM_FORMAT_MOD_LINEAR`. Some
//! drivers advertise `A2B10G10R10` as sampleable but not
//! storage-writable; on those, `create_compute_pipeline` would fail
//! mid-session. Probing up front lets the host refuse construction
//! before negotiation (the `Goodbye(InternalError)` contract in
//! `CLAUDE.md`).

use std::os::fd::OwnedFd;

use crate::{
    dmabuf_export::{
        export_xv30_shared_dmabuf, ExportError, SharedXv30Export, DRM_FORMAT_MOD_LINEAR,
    },
    modifier_query::ModifierQueryError,
    pipeline::build_xv30_pipeline,
};

/// One frame's worth of packed XV30 — a single dma-buf fd carrying one
/// `R=U, G=Y, B=V, A=X` packed 10:10:10:2 plane (XV30LE "X:Cr:Y:Cb" lane
/// order: bits[9:0]=Cb, bits[19:10]=Y, bits[29:20]=Cr).
///
/// Field shape matches [`crate::yuv444_dmabuf::Yuv444DmaBufFrame`]; the
/// downstream encoder doesn't distinguish 8-bit from 10-bit at this
/// layer — only the fourcc (`b"XV30"` here) and `VaapiEncoder`'s
/// pix-fmt table do.
pub struct Xv30DmaBufFrame {
    pub width: u32,
    pub height: u32,
    pub fd: OwnedFd,
    pub size: u64,
    pub modifier: u64,
    pub offset: u64,
    pub stride: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum Xv30DmaBufError {
    #[error("no wgpu adapter")]
    NoAdapter,
    #[error("wgpu request_device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error(
        "adapter doesn't advertise the features required for zero-copy XV30 \
         (VULKAN_EXTERNAL_MEMORY_DMA_BUF + TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES). \
         The Vulkan ICD must support VK_EXT_external_memory_dma_buf, \
         VK_EXT_image_drm_format_modifier, and VK_KHR_external_memory_fd."
    )]
    FeatureUnsupported,
    #[error(
        "driver doesn't advertise STORAGE_IMAGE on DRM_FORMAT_XV30 \
         (VK_FORMAT_A2B10G10R10_UNORM_PACK32) for DRM_FORMAT_MOD_LINEAR — the \
         BGRA→XV30 compute shader needs to write an Rgb10a2Unorm storage \
         texture. The 4:4:4 10-bit encode profile should be filtered out \
         before session negotiation; this error means the upstream filter \
         missed."
    )]
    StorageUnsupported,
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

pub type Result<T> = std::result::Result<T, Xv30DmaBufError>;

const DRM_FOURCC_XV30: u32 = u32::from_le_bytes(*b"XV30");

/// Persistent BGRA → XV30 converter writing into a DMA-BUF-exported
/// packed 10:10:10:2 texture. Built once per resolution; the per-frame
/// call takes an imported BGRA wgpu texture (from PipeWire) and returns
/// dup'd fds the encoder consumes via `VaapiEncoder::submit_dmabuf`.
pub struct Bgra2Xv30DmaBuf {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
    xv30: SharedXv30Export,
    packed_view: wgpu::TextureView,
}

impl Bgra2Xv30DmaBuf {
    /// Open a wgpu device with the required features and allocate the
    /// single-plane XV30 export sized for `width`×`height`. Probes
    /// `STORAGE_IMAGE` modifier support on XV30 up front — see the
    /// module comment for why.
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
            .map_err(|_| Xv30DmaBufError::NoAdapter)?;

        let required_features = wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        if !adapter.features().contains(required_features) {
            return Err(Xv30DmaBufError::FeatureUnsupported);
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tether-gpuconvert xv30-dmabuf device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await?;

        // STORAGE_IMAGE_BIT gate. Multi-GPU caveat is the same as the
        // P010 sibling — see `bgra_to_p010_dmabuf.rs`'s comment on the
        // hybrid-adapter case.
        let mods = crate::modifier_query::storable_dmabuf_modifiers(DRM_FOURCC_XV30).await?;
        if !mods.contains(&DRM_FORMAT_MOD_LINEAR) {
            return Err(Xv30DmaBufError::StorageUnsupported);
        }

        let (pipeline, bgl) = build_xv30_pipeline(&device);

        let xv30 = export_xv30_shared_dmabuf(
            &device,
            width,
            height,
            wgpu::TextureUsages::STORAGE_BINDING,
        )?;

        let packed_view = xv30.packed_texture.create_view(&Default::default());

        Ok(Self {
            device,
            queue,
            pipeline,
            bgl,
            width,
            height,
            xv30,
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
                    Xv30DmaBufError::Poll(
                        "device is not Vulkan-backed; DMA-BUF import unavailable".into(),
                    )
                })?
                .texture_from_dmabuf_fd(fd, &hal_desc, modifier, stride, offset)
                .map_err(|e| Xv30DmaBufError::Poll(format!("texture_from_dmabuf_fd: {e:?}")))?
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

    /// Convenience wrapper around [`Self::convert`] that takes a
    /// tightly-packed BGRA byte slice. Allocates a transient source
    /// texture, uploads the bytes, runs the compute pass. Used by the
    /// probe + unit tests; production callers (PipeWire imports) go
    /// through [`Self::convert`] directly to skip the upload.
    pub fn convert_bgra_bytes(&self, bgra: &[u8]) -> Result<Xv30DmaBufFrame> {
        let expected = (self.width * self.height * 4) as usize;
        if bgra.len() != expected {
            return Err(Xv30DmaBufError::ByteLenMismatch {
                got: bgra.len(),
                expected,
                w: self.width,
                h: self.height,
            });
        }
        let src = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("xv30 bgra scratch"),
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

    /// Convert one imported BGRA frame into the bridge's XV30 dma-buf
    /// target and return a dup'd fd the caller wraps in a codec
    /// `DmaBufFrame` for `VaapiEncoder::submit_dmabuf`.
    pub fn convert(&self, src_bgra: &wgpu::Texture) -> Result<Xv30DmaBufFrame> {
        if !matches!(
            src_bgra.format(),
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
        ) {
            return Err(Xv30DmaBufError::InputFormat(src_bgra.format()));
        }
        if src_bgra.width() != self.width || src_bgra.height() != self.height {
            return Err(Xv30DmaBufError::DimMismatch {
                input_w: src_bgra.width(),
                input_h: src_bgra.height(),
                w: self.width,
                h: self.height,
            });
        }

        let src_view = src_bgra.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xv30-dmabuf bg"),
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
                label: Some("xv30-dmabuf enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("xv30-dmabuf pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // 8×8 workgroup, one invocation per output pixel — no
            // chroma collapse in 4:4:4.
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        self.queue.submit(Some(enc.finish()));

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Xv30DmaBufError::Poll(format!("{e:?}")))?;

        Ok(Xv30DmaBufFrame {
            width: self.width,
            height: self.height,
            fd: self.xv30.fd.try_clone().map_err(Xv30DmaBufError::DupFd)?,
            size: self.xv30.size,
            modifier: self.xv30.modifier,
            offset: self.xv30.offset,
            stride: self.xv30.pitch,
        })
    }
}

#[cfg(test)]
mod tests {
    // Readback pixel-math casts (u64 offset → usize) are intentional.
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

    use super::*;

    /// Re-import one packed XV30 plane as Rgb10a2Unorm, copy it to a
    /// mappable buffer, and return raw 32-bit words. Used by the
    /// round-trip test to inspect the packed bits directly rather than
    /// going through any sampling-side normalisation.
    fn read_packed_u32(
        bridge: &Bgra2Xv30DmaBuf,
        fd: OwnedFd,
        modifier: u64,
        stride: u64,
        offset: u64,
        width: u32,
        height: u32,
    ) -> Vec<u32> {
        let import_desc = wgpu::hal::TextureDescriptor {
            label: Some("xv30 reimport"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgb10a2Unorm,
            usage: wgpu::TextureUses::COPY_SRC,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        // SAFETY: caller passes the bridge's authoritative
        // fd / modifier / stride / offset.
        let hal_tex = unsafe {
            bridge
                .device()
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("vulkan backend")
                .texture_from_dmabuf_fd(fd, &import_desc, modifier, stride, offset)
                .expect("xv30 texture_from_dmabuf_fd")
        };
        let import_tex = unsafe {
            bridge
                .device()
                .create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                    hal_tex,
                    &wgpu::TextureDescriptor {
                        label: Some("xv30 reimport"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgb10a2Unorm,
                        usage: wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    },
                )
        };

        let bytes_per_row = width * 4;
        let padded_row = u64::from(bytes_per_row).div_ceil(256) * 256;
        let readback = bridge.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("xv30 readback"),
            size: padded_row * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = bridge
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("xv30 readback enc"),
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
            .expect("poll xv30 readback");
        rx.recv().expect("cb").expect("map");
        let mapped = slice.get_mapped_range().expect("range");

        let mut out = Vec::with_capacity((width * height) as usize);
        for row in 0..height {
            let row_start = (u64::from(row) * padded_row) as usize;
            for x in 0..width {
                let off = row_start + (x as usize) * 4;
                let word = u32::from_le_bytes([
                    mapped[off],
                    mapped[off + 1],
                    mapped[off + 2],
                    mapped[off + 3],
                ]);
                out.push(word);
            }
        }
        out
    }

    /// End-to-end without VAAPI: build the bridge, push solid-colour
    /// BGRA through it, re-import the packed XV30 plane, read it back
    /// as 32-bit words, and assert each pixel has the expected 10-bit
    /// Y / U / V code values for BT.709 limited-range. Catches both
    /// the colour math and the R=U / G=Y / B=V / A=X channel mapping
    /// (XV30LE "X:Cr:Y:Cb") — the red cell additionally catches a U↔V
    /// swap, since pure red has very distinct Cb vs Cr codes.
    #[test]
    #[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF and Rgb10a2Unorm storage; run with: cargo test -p tether-gpuconvert -- --ignored"]
    fn convert_solid_white_roundtrip_packed_xv30() {
        let width = 64u32;
        let height = 32u32;

        let bridge = match pollster::block_on(Bgra2Xv30DmaBuf::new(width, height)) {
            Ok(b) => b,
            Err(
                Xv30DmaBufError::NoAdapter
                | Xv30DmaBufError::FeatureUnsupported
                | Xv30DmaBufError::StorageUnsupported,
            ) => {
                eprintln!("SKIP: no wgpu adapter with DMA-BUF export + Rgb10a2Unorm storage");
                return;
            }
            Err(e) => panic!("Bgra2Xv30DmaBuf::new: {e}"),
        };

        let n = (width * height) as usize;
        let bgra = vec![255u8; n * 4];
        let out = bridge.convert_bgra_bytes(&bgra).expect("convert");
        assert_eq!(out.offset, bridge.xv30.offset);
        assert_eq!(out.stride, bridge.xv30.pitch);

        let packed = read_packed_u32(
            &bridge,
            out.fd.try_clone().expect("dup fd"),
            out.modifier,
            out.stride,
            out.offset,
            width,
            height,
        );

        // BT.709 limited-range 10-bit for pure white:
        //   Y = 940, U = V = 512.
        // DRM_FORMAT_XV30 (Y410 family, "[31:0] X:Cr:Y:Cb"):
        //   [31:30]X | [29:20]Cr | [19:10]Y | [9:0]Cb. (Was extracted with
        //   Y↔Cb swapped, which is why this passed against a swapped pack
        //   while macOS VideoToolbox rendered the live stream purple.)
        let extract_u = |w: u32| w & 0x3FF;
        let extract_y = |w: u32| (w >> 10) & 0x3FF;
        let extract_v = |w: u32| (w >> 20) & 0x3FF;
        let assert_near = |label: &str, got: u32, expected: u32| {
            let diff = i64::from(got) - i64::from(expected);
            assert!(
                diff.abs() <= 2,
                "{label} = {got}, expected ~{expected} (diff {diff})",
            );
        };
        for &(x, y) in &[
            (0u32, 0u32),
            (width / 2, height / 2),
            (width - 1, height - 1),
        ] {
            let word = packed[(y * width + x) as usize];
            assert_near(&format!("Y[{x},{y}]"), extract_y(word), 940);
            assert_near(&format!("U[{x},{y}]"), extract_u(word), 512);
            assert_near(&format!("V[{x},{y}]"), extract_v(word), 512);
        }
    }

    /// Same shape but for solid red — the chroma-channel test:
    /// pure red has very distinct V vs U values, so a U↔V swap would
    /// show up as wildly wrong codes (V≈806, U≈400 in 10-bit).
    #[test]
    #[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF and Rgb10a2Unorm storage; run with: cargo test -p tether-gpuconvert -- --ignored"]
    fn convert_solid_red_roundtrip_packed_xv30() {
        let width = 64u32;
        let height = 32u32;

        let bridge = match pollster::block_on(Bgra2Xv30DmaBuf::new(width, height)) {
            Ok(b) => b,
            Err(
                Xv30DmaBufError::NoAdapter
                | Xv30DmaBufError::FeatureUnsupported
                | Xv30DmaBufError::StorageUnsupported,
            ) => {
                eprintln!("SKIP: no wgpu adapter with DMA-BUF export + Rgb10a2Unorm storage");
                return;
            }
            Err(e) => panic!("Bgra2Xv30DmaBuf::new: {e}"),
        };

        let n = (width * height) as usize;
        // BGRA layout: B, G, R, A. Pure red = (0, 0, 255, 255).
        let mut bgra = Vec::with_capacity(n * 4);
        for _ in 0..n {
            bgra.extend_from_slice(&[0, 0, 255, 255]);
        }
        let out = bridge.convert_bgra_bytes(&bgra).expect("convert");

        let packed = read_packed_u32(
            &bridge,
            out.fd.try_clone().expect("dup fd"),
            out.modifier,
            out.stride,
            out.offset,
            width,
            height,
        );

        // BT.709 limited-range 10-bit, pure red (R=1, G=0, B=0):
        //   Y' = 0.2126                  → Y  = 64 + 876 * 0.2126 ≈ 250
        //   U  = -0.11457                → Cb = 512 + 896 * -0.11457 ≈ 409
        //   V  =  0.50000                → Cr = 512 + 896 *  0.5     = 960
        // XV30 "[31:0] X:Cr:Y:Cb": Cb in bits[9:0], Y in bits[19:10],
        // Cr in bits[29:20].
        let extract_u = |w: u32| w & 0x3FF;
        let extract_y = |w: u32| (w >> 10) & 0x3FF;
        let extract_v = |w: u32| (w >> 20) & 0x3FF;
        let assert_near = |label: &str, got: u32, expected: u32| {
            let diff = i64::from(got) - i64::from(expected);
            assert!(
                diff.abs() <= 3,
                "{label} = {got}, expected ~{expected} (diff {diff})",
            );
        };
        for &(x, y) in &[
            (0u32, 0u32),
            (width / 2, height / 2),
            (width - 1, height - 1),
        ] {
            let word = packed[(y * width + x) as usize];
            assert_near(&format!("Y[{x},{y}]"), extract_y(word), 250);
            assert_near(&format!("U[{x},{y}]"), extract_u(word), 409);
            assert_near(&format!("V[{x},{y}]"), extract_v(word), 960);
        }
    }
}
