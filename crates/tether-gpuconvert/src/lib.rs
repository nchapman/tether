//! Host-side GPU colour-space conversion.
//!
//! v1 scope: BGRA → NV12 via a wgpu compute shader. The eventual
//! consumer is the zero-copy capture pipeline:
//!
//!   PipeWire DMA-BUF (BGRx, compositor format)
//!     -> wgpu texture import (VK_EXT_external_memory_dma_buf)
//!     -> [this crate] compute pass writes Y + UV storage textures
//!     -> wgpu texture export as DMA-BUFs
//!     -> VAAPI surface import (already wired via VaapiEncoder::submit_dmabuf)
//!     -> h264_vaapi encode
//!
//! This file currently exposes only the CPU↔CPU path
//! ([`Bgra2Nv12::convert`]) — copy BGRA in, get NV12 planes out. That's
//! the form the conversion math can be validated against known-colour
//! inputs without involving PipeWire or VAAPI; once it's correct, the
//! DMA-BUF import/export plumbing wraps the same compute pipeline
//! without changing the shader. Keeping the API split this way means
//! the eventual zero-copy path is purely glue.
//!
//! Why a separate crate (not under `tether-render` or `tether-codec`):
//! - `tether-render` is the client's display path; conflating it with
//!   host-side conversion would have the same crate doing both ends of
//!   the wire. Bad coupling.
//! - `tether-codec` is intentionally wgpu-free — encoders take buffers
//!   or VA surfaces, not GPU contexts. Pulling wgpu in there would
//!   leak a heavy dep across the workspace.
//! - This crate's surface is one wgpu device + one compute pipeline.
//!   Small, focused, easy to test in isolation.

// Host-side GPU conversion is a Linux + macOS concern: the Windows host
// converts BGRA→NV12/P010 with `ID3D11VideoProcessor` inside
// `tether-codec` instead and never links this crate (its `wgpu`
// dependency is declared only for Linux/macOS targets). The crate is a
// workspace member, so `cargo build --workspace` on Windows still
// compiles it — gate the whole crate to Linux/macOS so it resolves to an
// empty rlib on Windows rather than failing on the unconditional `wgpu`
// references in the conversion path below.
#![cfg(any(target_os = "linux", target_os = "macos"))]

#[cfg(target_os = "linux")]
pub mod bgra_to_p010_dmabuf;
#[cfg(target_os = "linux")]
pub mod dmabuf_export;
/// GPU device identity for multi-GPU pinning: read a producer device's
/// Vulkan `deviceUUID` so the codec side binds CUDA/EGL to the same GPU.
#[cfg(target_os = "linux")]
pub mod gpu_select;
#[cfg(target_os = "linux")]
pub mod modifier_query;
#[cfg(target_os = "linux")]
pub mod nv12_dmabuf;
#[cfg(target_os = "linux")]
pub mod xv30_dmabuf;
#[cfg(target_os = "linux")]
pub mod yuv444_dmabuf;

/// macOS NV12 IOSurface scaler bridge — the host-side equivalent of
/// `nv12_dmabuf` (Linux DMA-BUF NV12 bridge), driving the
/// `tether-scaler` YUV-plane paths to produce a downscaled
/// IOSurface ready for VideoToolbox.
#[cfg(target_os = "macos")]
pub mod nv12_iosurface;

mod pipeline;

#[cfg(target_os = "linux")]
pub use bgra_to_p010_dmabuf::{Bgra2P010DmaBuf, P010DmaBufError, P010DmaBufFrame};
#[cfg(target_os = "linux")]
pub use dmabuf_export::{
    export_texture_as_dmabuf, DmaBufExport, ExportError, DRM_FORMAT_MOD_LINEAR,
};
#[cfg(target_os = "linux")]
pub use modifier_query::{
    importable_dmabuf_modifiers, storable_dmabuf_modifiers, ModifierQueryError,
};
#[cfg(target_os = "linux")]
pub use nv12_dmabuf::{Nv12DmaBuf, Nv12DmaBufError, Nv12DmaBufFrame};
#[cfg(target_os = "linux")]
pub use xv30_dmabuf::{Bgra2Xv30DmaBuf, Xv30DmaBufError, Xv30DmaBufFrame};
#[cfg(target_os = "linux")]
pub use yuv444_dmabuf::{Yuv444DmaBuf, Yuv444DmaBufError, Yuv444DmaBufFrame};

#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("no wgpu adapter")]
    NoAdapter,
    #[error(
        "wgpu adapter doesn't advertise TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES; \
         R8Unorm / Rg8Unorm storage writes (required for NV12 output) are \
         unavailable. Check that the system Vulkan ICD is a real GPU driver."
    )]
    FeatureUnsupported,
    #[error("wgpu request_device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("wgpu create_surface: {0}")]
    CreateSurface(#[from] wgpu::CreateSurfaceError),
    #[error("input size {got} doesn't match width*height*4 = {expected}")]
    InputSize { got: usize, expected: usize },
    #[error("invalid dimensions: {width}x{height} (must be > 0)")]
    InvalidDims { width: u32, height: u32 },
    #[error("buffer mapping failed: {0:?}")]
    Map(wgpu::BufferAsyncError),
    #[error("wgpu poll / readback: {0}")]
    Poll(String),
}

pub type Result<T> = std::result::Result<T, ConvertError>;

/// One frame's worth of NV12 planes packed tightly (no per-row padding).
///
/// Layout matches the [`tether_codec::DecodedFrame`] / `DmaBufFrame`
/// shape so a downstream encoder consumer can pass these straight in
/// without copying. `y` is `width * height` bytes; `uv` is
/// `chroma_width * chroma_height * 2` bytes with U-then-V interleaved
/// per chroma sample (standard NV12).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nv12Planes {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub uv: Vec<u8>,
}

impl Nv12Planes {
    /// Chroma plane dimensions in chroma samples. The `uv` buffer is
    /// twice this wide in bytes (U,V interleaved).
    #[must_use]
    pub fn chroma_dims(&self) -> (u32, u32) {
        (self.width.div_ceil(2), self.height.div_ceil(2))
    }
}

/// BGRA → NV12 converter. Owns the wgpu device, queue, compute
/// pipeline, and a pair of cached intermediate textures sized for the
/// current input. Re-allocates the textures on dimension change.
///
/// One converter handles a single input resolution at a time. Multi-
/// resolution callers should rebuild via [`Self::new`] on resize;
/// resize is rare (display reconfig only) and the cost is one device
/// query + a few small allocations.
pub struct Bgra2Nv12 {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    width: u32,
    height: u32,
    src_tex: wgpu::Texture,
    y_tex: wgpu::Texture,
    uv_tex: wgpu::Texture,
    /// Cached bind group + views — same input/output textures every
    /// frame, so we pay the cost once at construction instead of per
    /// `convert()` call. Matters at 60+ fps.
    bind_group: wgpu::BindGroup,
    /// Padded readback buffer for the Y plane (wgpu requires a
    /// 256-byte row alignment on buffer-image transfers).
    y_readback: wgpu::Buffer,
    /// Same, for the UV plane.
    uv_readback: wgpu::Buffer,
    y_padded_row: u32,
    uv_padded_row: u32,
}

impl Bgra2Nv12 {
    /// Construct a converter for `width`×`height` BGRA input.
    ///
    /// Allocates a single wgpu device — callers should build one
    /// converter per host and reuse it. The async-ness comes from
    /// wgpu's `request_adapter` / `request_device`; wrap in
    /// `pollster::block_on` for synchronous contexts.
    pub async fn new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(ConvertError::InvalidDims { width, height });
        }

        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                // No surface — this is a headless compute device.
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| ConvertError::NoAdapter)?;

        // R8Unorm / Rg8Unorm aren't in WebGPU's portable storage-texture
        // set; we need the adapter-specific extension to write to them.
        // Every modern desktop GPU (Intel/AMD/NVIDIA via Vulkan) supports
        // both as storage images natively, so this is "ask for what the
        // hardware already has" rather than a niche extension. If a
        // future adapter doesn't, request_device fails up-front with a
        // clear "feature not supported" rather than a runtime validation
        // error on the bind-group-layout build.
        let required_features = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        if !adapter.features().contains(required_features) {
            return Err(ConvertError::FeatureUnsupported);
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tether-gpuconvert device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await?;

        let (pipeline, bind_group_layout) = pipeline::build_pipeline(&device);

        let (src_tex, y_tex, uv_tex, y_readback, uv_readback, y_padded_row, uv_padded_row) =
            allocate_textures(&device, width, height);

        let src_view = src_tex.create_view(&Default::default());
        let y_view = y_tex.create_view(&Default::default());
        let uv_view = uv_tex.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bgra_to_nv12 bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&uv_view),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            width,
            height,
            src_tex,
            y_tex,
            uv_tex,
            bind_group,
            y_readback,
            uv_readback,
            y_padded_row,
            uv_padded_row,
        })
    }

    /// Convert one BGRA frame. `bgra` must be `width * height * 4`
    /// bytes, tightly packed (no per-row stride padding).
    ///
    /// Blocks on GPU completion + buffer readback. For the future
    /// zero-copy DMA-BUF path this method goes away — replaced by one
    /// that takes imported wgpu textures as input/output, no CPU touch.
    pub fn convert(&self, bgra: &[u8]) -> Result<Nv12Planes> {
        let expected = (self.width as usize) * (self.height as usize) * 4;
        if bgra.len() != expected {
            return Err(ConvertError::InputSize {
                got: bgra.len(),
                expected,
            });
        }

        // Upload BGRA into the source texture.
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.src_tex,
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

        let (chroma_w, chroma_h) = (self.width.div_ceil(2), self.height.div_ceil(2));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bgra_to_nv12 enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bgra_to_nv12 pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            // Dispatch one workgroup per 8x8 chroma block. ceil-div so
            // odd chroma dimensions still cover the last partial group;
            // the shader bounds-checks against the texture size.
            let groups_x = chroma_w.div_ceil(8);
            let groups_y = chroma_h.div_ceil(8);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }

        // Copy Y + UV planes into mappable buffers. Per-row padding
        // (256-byte alignment) gets stripped on the CPU after readback.
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.y_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.y_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.y_padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.uv_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.uv_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.uv_padded_row),
                    rows_per_image: Some(chroma_h),
                },
            },
            wgpu::Extent3d {
                width: chroma_w,
                height: chroma_h,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        // Map both buffers and wait. For the synchronous CPU↔CPU API
        // we block on completion; the future DMA-BUF path skips this
        // entirely (the textures themselves are exported).
        let y_bytes = read_buffer(
            &self.device,
            &self.y_readback,
            self.width as usize,
            self.height as usize,
            self.y_padded_row as usize,
        )?;
        let uv_bytes = read_buffer(
            &self.device,
            &self.uv_readback,
            (chroma_w as usize) * 2,
            chroma_h as usize,
            self.uv_padded_row as usize,
        )?;

        Ok(Nv12Planes {
            width: self.width,
            height: self.height,
            y: y_bytes,
            uv: uv_bytes,
        })
    }

    /// Convenience: build a converter and run one frame. For one-shot
    /// callers (mostly tests).
    pub fn convert_oneshot(bgra: &[u8], width: u32, height: u32) -> Result<Nv12Planes> {
        let converter = pollster::block_on(Self::new(width, height))?;
        converter.convert(bgra)
    }
}

fn allocate_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (
    wgpu::Texture,
    wgpu::Texture,
    wgpu::Texture,
    wgpu::Buffer,
    wgpu::Buffer,
    u32,
    u32,
) {
    let chroma_w = width.div_ceil(2);
    let chroma_h = height.div_ceil(2);

    let src_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bgra src"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // PipeWire BGRx maps to Bgra8Unorm in wgpu (the alpha byte is
        // ignored in our shader's textureLoad path; using a 4-byte
        // format keeps the upload size simple).
        // Non-sRGB on purpose. PipeWire/compositor output is the
        // already-gamma-encoded framebuffer bytes; the shader treats
        // them as-is (matches what swscale + every other BGRA→YUV
        // path does). Switching to Bgra8UnormSrgb would apply an
        // unwanted sRGB→linear decode on textureLoad and break the
        // colour matrix.
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let y_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("nv12 y"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let uv_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("nv12 uv"),
        size: wgpu::Extent3d {
            width: chroma_w,
            height: chroma_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    // wgpu requires bytes_per_row to be a multiple of 256 on the
    // copy_texture_to_buffer side.
    let y_padded_row = padded_row(width); // Y plane: 1 byte per texel
    let uv_padded_row = padded_row(chroma_w * 2); // UV: 2 bytes per texel (R + G)

    let y_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("y readback"),
        size: u64::from(y_padded_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let uv_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uv readback"),
        size: u64::from(uv_padded_row) * u64::from(chroma_h),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    (
        src_tex,
        y_tex,
        uv_tex,
        y_readback,
        uv_readback,
        y_padded_row,
        uv_padded_row,
    )
}

/// Round `tight_row` up to wgpu's 256-byte alignment requirement.
fn padded_row(tight_row: u32) -> u32 {
    const ALIGN: u32 = 256;
    tight_row.div_ceil(ALIGN) * ALIGN
}

fn read_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    tight_row: usize,
    rows: usize,
    padded_row: usize,
) -> Result<Vec<u8>> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    // Scope guard so any error / panic from here on still unmaps the
    // buffer. Without it, a device loss / driver crash would panic
    // (via `?` returning, or a future panic in caller) with the
    // mapping still held — wgpu's debug builds then assert-fail on
    // the next operation, masking the real error.
    struct Unmapper<'a>(&'a wgpu::Buffer);
    impl<'a> Drop for Unmapper<'a> {
        fn drop(&mut self) {
            self.0.unmap();
        }
    }
    let _guard = Unmapper(buffer);

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| ConvertError::Poll(format!("{e:?}")))?;
    rx.recv()
        .map_err(|_| ConvertError::Poll("map_async callback never fired".into()))?
        .map_err(ConvertError::Map)?;

    let mapped = slice
        .get_mapped_range()
        .map_err(|e| ConvertError::Poll(format!("get_mapped_range: {e:?}")))?;
    let mut out = Vec::with_capacity(tight_row * rows);
    for row in 0..rows {
        let start = row * padded_row;
        out.extend_from_slice(&mapped[start..start + tight_row]);
    }
    drop(mapped);
    Ok(out)
}

#[cfg(test)]
mod tests {
    // BT.709 reference pixel-math casts (clamped fp → u8) are intentional.
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

    use super::*;

    /// Fill `bgra` with a solid colour. BGRA layout: B, G, R, A.
    fn solid(width: u32, height: u32, b: u8, g: u8, r: u8) -> Vec<u8> {
        let n = (width as usize) * (height as usize);
        let mut v = Vec::with_capacity(n * 4);
        for _ in 0..n {
            v.extend_from_slice(&[b, g, r, 255]);
        }
        v
    }

    /// Expected NV12 values for a solid BGRA colour under BT.709
    /// limited range. Computed in f32 then rounded the same way an
    /// R8Unorm storage write does (round-to-nearest, ties-to-even on
    /// most GPUs; tolerance below absorbs the ±1 difference).
    fn expected(b: u8, g: u8, r: u8) -> (u8, u8, u8) {
        let rf = f32::from(r) / 255.0;
        let gf = f32::from(g) / 255.0;
        let bf = f32::from(b) / 255.0;
        let y = 0.2126 * rf + 0.7152 * gf + 0.0722 * bf;
        let u = -0.11457 * rf - 0.38543 * gf + 0.5 * bf;
        let v = 0.5 * rf - 0.45415 * gf - 0.04585 * bf;
        let y_byte = (y * (219.0 / 255.0) + 16.0 / 255.0) * 255.0;
        let u_byte = (u * (224.0 / 255.0) + 128.0 / 255.0) * 255.0;
        let v_byte = (v * (224.0 / 255.0) + 128.0 / 255.0) * 255.0;
        (
            y_byte.round().clamp(0.0, 255.0) as u8,
            u_byte.round().clamp(0.0, 255.0) as u8,
            v_byte.round().clamp(0.0, 255.0) as u8,
        )
    }

    fn assert_near(label: &str, got: u8, expected: u8, tol: i32) {
        let diff = i32::from(got) - i32::from(expected);
        assert!(
            diff.abs() <= tol,
            "{label}: got {got}, expected {expected} (±{tol}), diff {diff}",
        );
    }

    /// Solid-colour conversion smoke. We sample a handful of pixels
    /// from each plane rather than asserting on every byte — uniform
    /// input means every output pixel must match, so a sparse check is
    /// enough to catch sign/coefficient/range bugs.
    fn check_solid(width: u32, height: u32, b: u8, g: u8, r: u8) {
        let bgra = solid(width, height, b, g, r);
        let nv12 = Bgra2Nv12::convert_oneshot(&bgra, width, height).expect("convert");
        let (ey, eu, ev) = expected(b, g, r);

        assert_eq!(nv12.width, width);
        assert_eq!(nv12.height, height);
        assert_eq!(nv12.y.len(), (width * height) as usize);
        let (cw, ch) = nv12.chroma_dims();
        assert_eq!(nv12.uv.len(), (cw * ch * 2) as usize);

        // Sample 9 evenly-spaced Y pixels (corners + edges + center).
        for &(x, y) in &[
            (0, 0),
            (width / 2, 0),
            (width - 1, 0),
            (0, height / 2),
            (width / 2, height / 2),
            (width - 1, height / 2),
            (0, height - 1),
            (width / 2, height - 1),
            (width - 1, height - 1),
        ] {
            let idx = (y * width + x) as usize;
            assert_near(&format!("Y[{x},{y}]"), nv12.y[idx], ey, 2);
        }
        // Sample 4 UV pairs at corners + center.
        for &(x, y) in &[(0, 0), (cw - 1, 0), (0, ch - 1), (cw / 2, ch / 2)] {
            let idx = ((y * cw + x) * 2) as usize;
            assert_near(&format!("U[{x},{y}]"), nv12.uv[idx], eu, 2);
            assert_near(&format!("V[{x},{y}]"), nv12.uv[idx + 1], ev, 2);
        }
    }

    #[test]
    #[ignore = "requires a working wgpu adapter (run on a machine with a real GPU)"]
    fn convert_solid_black() {
        check_solid(64, 64, 0, 0, 0);
    }

    #[test]
    #[ignore = "requires a working wgpu adapter"]
    fn convert_solid_white() {
        check_solid(64, 64, 255, 255, 255);
    }

    #[test]
    #[ignore = "requires a working wgpu adapter"]
    fn convert_solid_red() {
        check_solid(64, 64, 0, 0, 255);
    }

    #[test]
    #[ignore = "requires a working wgpu adapter"]
    fn convert_solid_green() {
        check_solid(64, 64, 0, 255, 0);
    }

    #[test]
    #[ignore = "requires a working wgpu adapter"]
    fn convert_solid_blue() {
        check_solid(64, 64, 255, 0, 0);
    }

    /// Non-uniform input: left half red, right half blue. Validates
    /// that spatial layout survives the conversion + 2×2 chroma
    /// averaging (the boundary column averages to magenta).
    #[test]
    #[ignore = "requires a working wgpu adapter"]
    fn convert_split_red_blue() {
        let w = 64u32;
        let h = 32u32;
        let mut bgra = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..h {
            for x in 0..w {
                if x < w / 2 {
                    bgra.extend_from_slice(&[0, 0, 255, 255]); // red
                } else {
                    bgra.extend_from_slice(&[255, 0, 0, 255]); // blue
                }
            }
        }
        let nv12 = Bgra2Nv12::convert_oneshot(&bgra, w, h).expect("convert");

        // Y plane: well-inside left half should match red Y; right
        // half should match blue Y. The actual boundary pixel is a
        // direct sample, not averaged — Y is per-pixel, not subsampled.
        let (ery, _, _) = expected(0, 0, 255);
        let (eby, _, _) = expected(255, 0, 0);
        assert_near("Y[8,16] (left/red)", nv12.y[(16 * w + 8) as usize], ery, 2);
        assert_near(
            "Y[56,16] (right/blue)",
            nv12.y[(16 * w + 56) as usize],
            eby,
            2,
        );

        // UV plane: corners are pure-colour averages. Sample at chroma
        // coords (4, 8) for red half and (28, 8) for blue half.
        let (cw, _) = nv12.chroma_dims();
        let (_, eru, erv) = expected(0, 0, 255);
        let (_, ebu, ebv) = expected(255, 0, 0);
        let red_uv = ((8 * cw + 4) * 2) as usize;
        let blue_uv = ((8 * cw + 28) * 2) as usize;
        assert_near("U[4,8] (red)", nv12.uv[red_uv], eru, 2);
        assert_near("V[4,8] (red)", nv12.uv[red_uv + 1], erv, 2);
        assert_near("U[28,8] (blue)", nv12.uv[blue_uv], ebu, 2);
        assert_near("V[28,8] (blue)", nv12.uv[blue_uv + 1], ebv, 2);

        // Boundary: w/2 = 32, so the seam in luma falls between cols
        // 31 (red) and 32 (blue). At chroma resolution the seam is
        // between chroma cols 15 and 16: chroma col 15 averages
        // luma cols 30,31 (both red), so it should still read pure
        // red; chroma col 16 averages luma cols 32,33 (both blue),
        // pure blue. This catches off-by-one in the shader's `lx`
        // computation, which would smear the seam by half a chroma
        // sample either direction.
        let last_red_uv = ((8 * cw + (cw / 2 - 1)) * 2) as usize;
        let first_blue_uv = ((8 * cw + cw / 2) * 2) as usize;
        assert_near(
            "U[cw/2-1,8] (last red chroma)",
            nv12.uv[last_red_uv],
            eru,
            2,
        );
        assert_near(
            "V[cw/2-1,8] (last red chroma)",
            nv12.uv[last_red_uv + 1],
            erv,
            2,
        );
        assert_near(
            "U[cw/2,8] (first blue chroma)",
            nv12.uv[first_blue_uv],
            ebu,
            2,
        );
        assert_near(
            "V[cw/2,8] (first blue chroma)",
            nv12.uv[first_blue_uv + 1],
            ebv,
            2,
        );
    }
}
