//! BGRA wgpu texture → NV12 DMA-BUF planes, via compute shader + export.
//!
//! Ties the BGRA→NV12 compute pipeline (shared with [`crate::Bgra2Nv12`])
//! to a pair of DMA-BUF-exported storage textures. The exported textures
//! are allocated once at construction and reused every frame; the caller
//! gets back dup'd fds per call so the consumer (VAAPI here, IOSurface
//! later on macOS, D3D11 shared handle on Windows — same shape via
//! different export modules) can own its own lifetime.
//!
//! Synchronisation: we block on GPU completion (`device.poll`) before
//! returning the descriptor. Without an explicit Vulkan→VAAPI fence the
//! consumer has no way to know the compute write has retired, and not
//! every libva backend honours the dma-buf reservation-object fence.
//! Cost is one synchronous poll per frame; the eventual answer is to
//! plumb a `sync_file` fd alongside the descriptor — see the comment
//! near [`tether_codec::DmaBufFrame`].
//!
//! Cross-platform shape:
//! - this module is Linux-only because DMA-BUF is the Linux GPU-IPC
//!   primitive. The macOS sibling will be `nv12_iosurface.rs` (compute
//!   pipeline identical, export goes through `IOSurface_create` +
//!   `MTLDevice.newTextureWithDescriptor`). The Windows sibling is
//!   D3D11 shared handle. The bridge struct's *shape* (own a device,
//!   own pre-allocated Y/UV targets, take an imported BGRA texture,
//!   return per-frame descriptors) is what's shared.

use std::os::fd::OwnedFd;

use crate::{
    dmabuf_export::{export_nv12_shared_dmabuf, ExportError, SharedNv12Export},
    pipeline::build_pipeline,
};

/// One frame's worth of NV12 — a single DMA-BUF fd carrying both planes
/// at distinct offsets within one shared `VkDeviceMemory` allocation.
///
/// `fd` is a *dup'd copy* of the bridge's persistent export — drop this
/// struct (or hand it off and let the consumer drop it) once the
/// downstream importer has refcounted what it cares about. The
/// underlying GPU memory stays alive via the bridge's own export for as
/// long as the bridge exists.
///
/// Field shape matches what `tether_codec::DmaBufFrame` needs for
/// ffmpeg's `av_hwframe_map(DRM_PRIME → VAAPI)`: one object with two
/// layers (Y as R8, UV as GR88) both pointing at `object_index=0` with
/// per-plane offsets/strides. This is the only shape ffmpeg's VAAPI
/// hwframe_map accepts ("VAAPI can only map frames made from a single
/// DRM object").
pub struct Nv12DmaBufFrame {
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
pub enum Nv12DmaBufError {
    #[error("no wgpu adapter")]
    NoAdapter,
    #[error("wgpu request_device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error(
        "adapter doesn't advertise the features required for zero-copy NV12 \
         (VULKAN_EXTERNAL_MEMORY_DMA_BUF + TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES). \
         The Vulkan ICD must support VK_EXT_external_memory_dma_buf, \
         VK_EXT_image_drm_format_modifier, and VK_KHR_external_memory_fd."
    )]
    FeatureUnsupported,
    #[error("dma-buf export: {0}")]
    Export(#[from] ExportError),
    /// Source view must be Bgra8Unorm (the capture-side DMA-BUF
    /// import) or Rgba8Unorm (the tether-scaler output). wgpu
    /// presents both as shader-visible RGBA so the chroma shader's
    /// `.rgb` swizzle works against either; the runtime check exists
    /// only to catch genuinely wrong formats (sRGB views, fp16, etc.)
    /// before the bind-group binding fails opaquely.
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

pub type Result<T> = std::result::Result<T, Nv12DmaBufError>;

/// Persistent BGRA→NV12 converter that writes its output into
/// DMA-BUF-exported Y/UV textures. Built once per resolution; the
/// per-frame call takes an imported BGRA wgpu texture (typically from
/// PipeWire's DMA-BUF buffer-type) and returns dup'd fds to the same
/// underlying GPU memory the consumer wants to read.
pub struct Nv12DmaBuf {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
    /// Shared NV12 surface — one VkDeviceMemory backing both plane
    /// textures, exported as a single dma-buf fd. Held for re-use
    /// across frames; dropping it drops both plane textures (which in
    /// turn destroy the VkImages) and frees the memory.
    nv12: SharedNv12Export,
    /// Cached views into the export textures so we don't rebuild them
    /// per frame; bind groups themselves still rebuild because the
    /// source texture is per-frame.
    y_view: wgpu::TextureView,
    uv_view: wgpu::TextureView,
}

impl Nv12DmaBuf {
    /// Build the bridge: open a wgpu device with the required features,
    /// allocate the shared compute pipeline, and allocate the Y + UV
    /// DMA-BUF exports sized for `width`×`height`.
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
            .map_err(|_| Nv12DmaBufError::NoAdapter)?;

        let required_features = wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        if !adapter.features().contains(required_features) {
            return Err(Nv12DmaBufError::FeatureUnsupported);
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tether-gpuconvert nv12-dmabuf device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await?;

        let (pipeline, bgl) = build_pipeline(&device);

        let nv12 = export_nv12_shared_dmabuf(
            &device,
            width,
            height,
            wgpu::TextureUsages::STORAGE_BINDING,
        )?;

        let y_view = nv12.y_texture.create_view(&Default::default());
        let uv_view = nv12.uv_texture.create_view(&Default::default());

        Ok(Self {
            device,
            queue,
            pipeline,
            bgl,
            width,
            height,
            nv12,
            y_view,
            uv_view,
        })
    }

    /// The wgpu device the bridge owns. Callers importing PipeWire
    /// DMA-BUFs as wgpu textures must use this device so the imported
    /// texture lives on the same Vulkan instance as the compute
    /// pipeline that reads it.
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The wgpu queue paired with [`Self::device`]. Exposed so capture
    /// imports can be submitted alongside the bridge's own work.
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
    /// this bridge's device. Format is fixed to `Bgra8Unorm` (matches
    /// what compositors emit for BGRx/BGRA video formats); usage is
    /// `TEXTURE_BINDING` because the shader reads via `textureLoad`.
    ///
    /// The bridge takes ownership of `fd`; on import success the
    /// returned `wgpu::Texture` holds the corresponding `VkImage` and
    /// `VkDeviceMemory`, and dropping the texture closes the fd
    /// indirectly via that allocation. The caller's PipeWire buffer
    /// release guard should be held alongside the texture for as long
    /// as the GPU is reading.
    ///
    /// `modifier`, `stride`, and `offset` come from PipeWire's
    /// `SPA_DATA_DmaBuf` chunk metadata (queried via the buffer's
    /// chunk + the negotiated format's modifier param).
    ///
    /// # Errors
    /// Returns [`Nv12DmaBufError::Poll`] if wgpu-hal rejects the
    /// import — typically a stride/modifier mismatch with what the
    /// driver expects, or an unsupported DMA-BUF source.
    /// Import an externally-allocated BGRA dma-buf as a wgpu texture.
    ///
    /// `width` and `height` describe the *source's* dimensions, which
    /// may differ from the bridge's own width/height when a scaler
    /// sits between the import and `convert()`. For the direct
    /// (no-scaler) path, callers pass the bridge's `width()` /
    /// `height()` here.
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
        // SAFETY: caller asserts fd is a valid DMA-BUF fd owned by us
        // (we took the `OwnedFd`), and that modifier/stride/offset
        // describe the same image. wgpu-hal does the Vulkan-side
        // validation; failure returns Err rather than UB.
        let hal_tex = unsafe {
            self.device
                .as_hal::<wgpu::hal::api::Vulkan>()
                .ok_or_else(|| {
                    Nv12DmaBufError::Poll(
                        "device is not Vulkan-backed; DMA-BUF import unavailable".into(),
                    )
                })?
                .texture_from_dmabuf_fd(fd, &hal_desc, modifier, stride, offset)
                .map_err(|e| Nv12DmaBufError::Poll(format!("texture_from_dmabuf_fd: {e:?}")))?
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
        // SAFETY: hal_tex was just built from this device on the same
        // Vulkan backend; descs match in shape.
        let tex = unsafe {
            self.device
                .create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_tex, &wgpu_desc)
        };
        Ok(tex)
    }

    /// Convert one imported BGRA frame into the bridge's NV12 DMA-BUF
    /// targets and return dup'd fds the caller can wrap in a codec
    /// `DmaBufFrame` for `VaapiEncoder::submit_dmabuf`.
    ///
    /// Blocks on GPU completion so VAAPI's subsequent read sees the
    /// finished compute write. See module-level comment for the
    /// rationale and the path to an explicit-fence-based version.
    pub fn convert(&self, src_bgra: &wgpu::Texture) -> Result<Nv12DmaBufFrame> {
        if !matches!(
            src_bgra.format(),
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
        ) {
            return Err(Nv12DmaBufError::InputFormat(src_bgra.format()));
        }
        if src_bgra.width() != self.width || src_bgra.height() != self.height {
            return Err(Nv12DmaBufError::DimMismatch {
                input_w: src_bgra.width(),
                input_h: src_bgra.height(),
                w: self.width,
                h: self.height,
            });
        }

        let src_view = src_bgra.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nv12-dmabuf bg"),
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
                label: Some("nv12-dmabuf enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nv12-dmabuf pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Dispatch covers the **visible** chroma region (derived
            // from `self.width`), not the texture's actual chroma
            // dimensions. Since `shared_nv12.rs` allocates the Y/UV
            // images at `align_up(width, 64)` for Intel iHD VAAPI's
            // sake, the wgpu textures are wider than the visible
            // region; the shader's `textureDimensions(uv_dst)` guard
            // returns the aligned dims, but we deliberately dispatch
            // fewer workgroups so the right-edge padding stays
            // undefined and VAAPI crops it via the declared frame
            // width. Aligning the dispatch to `textureDimensions`
            // would write padding columns and confuse the encoder.
            let chroma_w = self.width.div_ceil(2);
            let chroma_h = self.height.div_ceil(2);
            pass.dispatch_workgroups(chroma_w.div_ceil(8), chroma_h.div_ceil(8), 1);
        }
        self.queue.submit(Some(enc.finish()));

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Nv12DmaBufError::Poll(format!("{e:?}")))?;

        Ok(Nv12DmaBufFrame {
            width: self.width,
            height: self.height,
            fd: self.nv12.fd.try_clone().map_err(Nv12DmaBufError::DupFd)?,
            size: self.nv12.size,
            modifier: self.nv12.modifier,
            y_offset: self.nv12.y_offset,
            y_stride: self.nv12.y_pitch,
            uv_offset: self.nv12.uv_offset,
            uv_stride: self.nv12.uv_pitch,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dmabuf_export::export_texture_as_dmabuf;

    /// Cross-table fourcc invariant: every `Nv12DmaBufFrame` this
    /// crate produces must carry the fourcc the VAAPI encoder
    /// expects for an 8-bit 4:2:0 profile, as reported by
    /// `tether_codec::vaapi::expected_dmabuf_fourcc`. The
    /// macOS-side equivalent (`nv12_iosurface::nv12_fourccs_round_trip_across_tables`)
    /// catches the same family of drift on its host-scaler bridge;
    /// the Linux side gets a smaller invariant because the
    /// gpuconvert bridges hardcode their output fourcc rather than
    /// having a runtime accept set.
    ///
    /// Cheap unit test: doesn't construct a bridge or touch GPU
    /// state; just round-trips the constant through both crates'
    /// tables.
    #[test]
    fn nv12_dmabuf_fourcc_matches_encoder_expectation() {
        use tether_codec::vaapi::expected_dmabuf_fourcc;
        use tether_protocol::control::ChromaSubsampling;
        // The bridge's `nv12_dmabuf_to_codec_frame` (in tether-host)
        // and the producer-side `DmaBufObject` here both encode
        // NV12 via `u32::from_le_bytes(*b"NV12")`. The encoder
        // looks up the same constant via `expected_dmabuf_fourcc`.
        let bridge_fourcc = u32::from_le_bytes(*b"NV12");
        let encoder_fourcc = expected_dmabuf_fourcc(ChromaSubsampling::Yuv420, 8)
            .expect("VAAPI encoder must report an expected fourcc for 8-bit 4:2:0");
        assert_eq!(
            bridge_fourcc, encoder_fourcc,
            "Nv12DmaBuf bridge fourcc 0x{bridge_fourcc:08x} drift vs encoder \
             expected 0x{encoder_fourcc:08x} — a mismatch would crash \
             av_hwframe_map(DRM_PRIME → VAAPI) at submit-frame time."
        );
    }

    /// End-to-end without VAAPI: load a solid-colour BGRA texture, run
    /// the bridge, re-import the returned Y plane via wgpu's existing
    /// `texture_from_dmabuf_fd`, read it back, assert the bytes match
    /// what BT.709 limited-range gives for that colour.
    ///
    /// This is the smallest test that proves: shader runs against
    /// exported storage textures (not just normal ones), exported
    /// memory is the same memory the importer sees, the dispatcher
    /// covers the full frame. Catches binding-order regressions in
    /// [`crate::pipeline::build_pipeline`] too.
    #[test]
    #[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF"]
    fn convert_solid_red_roundtrip() {
        let width = 64u32;
        let height = 32u32;

        let bridge = match pollster::block_on(Nv12DmaBuf::new(width, height)) {
            Ok(b) => b,
            Err(Nv12DmaBufError::NoAdapter | Nv12DmaBufError::FeatureUnsupported) => {
                eprintln!("SKIP: no wgpu adapter with DMA-BUF export feature");
                return;
            }
            Err(e) => panic!("Nv12DmaBuf::new: {e}"),
        };

        // Build a non-exported source texture filled with solid red
        // (BGRA: B=0, G=0, R=255, A=255). Uses the bridge's own device
        // so the texture is on the same Vulkan instance the compute
        // pipeline will read from.
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

        let out = bridge.convert(&src).expect("bridge convert");

        // Re-import the Y plane via the existing DMA-BUF import path,
        // copy to a readback buffer, and verify a handful of pixels.
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
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUses::COPY_SRC,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        // SAFETY: out.y_fd was just produced by the bridge's export
        // path so the modifier/stride/offset values are authoritative.
        let hal_tex = unsafe {
            bridge
                .device()
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("vulkan backend")
                .texture_from_dmabuf_fd(
                    out.fd
                        .try_clone()
                        .expect("dup shared NV12 fd for Y reimport"),
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
                        format: wgpu::TextureFormat::R8Unorm,
                        usage: wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    },
                )
        };

        let padded_row = u64::from(width).div_ceil(256) * 256;
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

        // BT.709 limited-range Y for pure red (R=1, G=0, B=0):
        //   Y' = 0.2126; scaled to limited = 0.2126*219/255 + 16/255
        //      ≈ 0.2455 → byte ≈ 63.
        let expected_y: u8 = ((0.2126_f32 * 219.0 / 255.0 + 16.0 / 255.0) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8;

        for &(x, y) in &[
            (0u32, 0u32),
            (width / 2, height / 2),
            (width - 1, height - 1),
        ] {
            let got = mapped[(u64::from(y) * padded_row + u64::from(x)) as usize];
            let diff = i32::from(got) - i32::from(expected_y);
            assert!(
                diff.abs() <= 2,
                "Y[{x},{y}] = {got}, expected ~{expected_y} (diff {diff})",
            );
        }
    }

    /// Same conversion math as the test above, but the BGRA source
    /// arrives via [`Nv12DmaBuf::import_bgra_dmabuf`] rather than as a
    /// native wgpu texture — the production shape. Uses
    /// `export_texture_as_dmabuf` to stand in for PipeWire as the
    /// producer.
    #[test]
    #[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF"]
    fn convert_via_imported_bgra() {
        let width = 32u32;
        let height = 32u32;

        let bridge = match pollster::block_on(Nv12DmaBuf::new(width, height)) {
            Ok(b) => b,
            Err(Nv12DmaBufError::NoAdapter | Nv12DmaBufError::FeatureUnsupported) => {
                eprintln!("SKIP: no wgpu adapter with DMA-BUF export feature");
                return;
            }
            Err(e) => panic!("Nv12DmaBuf::new: {e}"),
        };

        // Stand in for PipeWire: export a BGRA DMA-BUF on the same
        // device, fill it with solid blue (B=255, G=0, R=0), then
        // hand the fd to the bridge's import helper. In production the
        // BGRA DMA-BUF comes from the compositor through PipeWire.
        let src_export = export_texture_as_dmabuf(
            bridge.device(),
            width,
            height,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            "bgra source",
        )
        .expect("export bgra source");
        let n = (width * height) as usize;
        let mut bgra = Vec::with_capacity(n * 4);
        for _ in 0..n {
            bgra.extend_from_slice(&[255, 0, 0, 255]); // blue
        }
        bridge.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src_export.texture,
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
        bridge.queue().submit(std::iter::empty());
        bridge
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll write");

        let dup_fd = src_export.fd.try_clone().expect("dup src fd");
        let imported = bridge
            .import_bgra_dmabuf(
                dup_fd,
                src_export.drm_format_modifier,
                src_export.stride,
                src_export.offset,
                width,
                height,
            )
            .expect("import_bgra_dmabuf");

        let out = bridge.convert(&imported).expect("convert");

        // BT.709 Y for pure blue: Y' = 0.0722; limited byte ≈
        // (0.0722*219/255 + 16/255) * 255 ≈ 32.
        let expected_y: u8 = ((0.0722_f32 * 219.0 / 255.0 + 16.0 / 255.0) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8;

        // Read back Y plane via the export's own re-import path.
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
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUses::COPY_SRC,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        // SAFETY: out.y_fd was produced by the bridge's export path.
        let hal_tex = unsafe {
            bridge
                .device()
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("vulkan backend")
                .texture_from_dmabuf_fd(
                    out.fd
                        .try_clone()
                        .expect("dup shared NV12 fd for Y reimport"),
                    &import_desc,
                    out.modifier,
                    out.y_stride,
                    out.y_offset,
                )
                .expect("texture_from_dmabuf_fd y")
        };
        let y_tex = unsafe {
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
                        format: wgpu::TextureFormat::R8Unorm,
                        usage: wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    },
                )
        };
        let padded_row = u64::from(width).div_ceil(256) * 256;
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
                texture: &y_tex,
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

        for &(x, y) in &[(0u32, 0u32), (width / 2, height / 2)] {
            let got = mapped[(u64::from(y) * padded_row + u64::from(x)) as usize];
            let diff = i32::from(got) - i32::from(expected_y);
            assert!(
                diff.abs() <= 2,
                "Y[{x},{y}] (blue via import) = {got}, expected ~{expected_y} (diff {diff})",
            );
        }
    }

    /// Structural regression for the iHD-VAAPI alignment fix in
    /// `shared_nv12.rs`: at unaligned widths, the bridge must report
    /// a Y-plane row pitch that's a multiple of 64 bytes (Intel iHD's
    /// hardcoded NV12 import expectation). Pre-fix, the Vulkan driver
    /// picked tight `row_pitch == width` for LINEAR R8 images at e.g.
    /// 2160-px width, which VAAPI then mis-read as 64-aligned and
    /// produced left-edge row aliasing in every encoded frame. The
    /// fix forces the underlying image's allocated width to
    /// `align_up(visible_width, 64)`, so `row_pitch` falls out at the
    /// aligned value by construction. A reintroduction of the tight-
    /// stride behaviour fails this assertion before any
    /// encoder/decoder/renderer runs.
    ///
    /// Picking 2160×1440: 2160 is the smallest non-64-aligned width
    /// production hits on HiDPI 2× displays and it's the exact width
    /// the end-to-end roundtrip-harness `repro_shape` cell uses.
    #[test]
    #[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF"]
    fn convert_reports_64_aligned_y_stride_at_unaligned_width() {
        let width = 2160u32;
        let height = 1440u32;
        debug_assert_ne!(
            width % 64,
            0,
            "the test point must be a non-64-aligned width to exercise the alignment fix",
        );

        let bridge = match pollster::block_on(Nv12DmaBuf::new(width, height)) {
            Ok(b) => b,
            Err(Nv12DmaBufError::NoAdapter | Nv12DmaBufError::FeatureUnsupported) => {
                eprintln!("SKIP: no wgpu adapter with DMA-BUF export feature");
                return;
            }
            Err(e) => panic!("Nv12DmaBuf::new: {e}"),
        };
        let src = bridge.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("alignment-test bgra src"),
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
        // Content doesn't matter for the alignment check — we only
        // inspect the returned metadata. Still write something so the
        // bridge actually exercises its dispatch path at this width.
        let bgra = vec![0u8; (width * height * 4) as usize];
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
        assert!(
            out.y_stride >= u64::from(width),
            "y_stride {} must be at least the visible width {width}",
            out.y_stride
        );
        assert_eq!(
            out.y_stride % 64,
            0,
            "y_stride {} is not 64-aligned at width {width} — Intel iHD VAAPI will mis-read the dma-buf and corrupt the left edge",
            out.y_stride
        );
        // UV pitch is the same byte count as Y at full-resolution
        // (each chroma sample is 2 bytes, chroma width is half luma,
        // so 2 * (aligned_luma/2) == aligned_luma bytes). Catches a
        // future change that aligns Y but forgets UV.
        assert_eq!(
            out.uv_stride % 64,
            0,
            "uv_stride {} must be 64-aligned for the same reason as y_stride",
            out.uv_stride,
        );
    }
}
