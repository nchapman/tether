//! Shared macOS IOSurface ↔ Metal ↔ wgpu interop primitives.
//!
//! Both the client renderer (`tether-render::gpu::metal`) and the host
//! NV12 scaler bridge (`tether-gpuconvert::nv12_iosurface`) need to:
//!
//! 1. Reinterpret an opaque IOSurface pointer carried in an
//!    [`IOSurfaceFrame`] as a typed `&IOSurfaceRef`.
//! 2. Wrap an individual plane of that surface as an `MTLTexture` and
//!    then promote the MTLTexture into a `wgpu::Texture` via the Metal
//!    HAL escape hatch, with caller-specified usage / storage mode (the
//!    renderer needs read-only `ShaderRead | Private`; the host bridge
//!    needs read+write `ShaderRead | ShaderWrite | PixelFormatView |
//!    Shared` for its scaler destinations).
//! 3. Agree on which `(chroma, bit_depth, fourcc)` triples are
//!    importable, so the cross-table consistency test (encoder accept
//!    set ↔ probe expected set ↔ renderer accept set) has exactly one
//!    place to add a new family.
//!
//! Lives in `tether-codec` (and not `tether-render` or
//! `tether-gpuconvert`) because both downstream crates already depend
//! on `tether-codec` — adding this helper here avoids creating a new
//! `tether-render` ↔ `tether-gpuconvert` dependency edge or pulling
//! the renderer into the host's GPU pipeline.

use std::ffi::c_void;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_io_surface::IOSurfaceRef;
use objc2_metal::{
    MTLDevice, MTLPixelFormat, MTLStorageMode, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};
use tether_protocol::control::ChromaSubsampling;

use crate::IOSurfaceFrame;

/// `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange` — `'420v'`, the
/// canonical NV12 video-range fourcc Apple's frameworks use.
pub const NV12_VIDEO_RANGE_FOURCC: u32 = u32::from_be_bytes(*b"420v");
/// `kCVPixelFormatType_420YpCbCr8BiPlanarFullRange` — `'420f'`. Same
/// plane shape as `'420v'`; range is signalled through `VideoColorSpec`,
/// not the fourcc.
pub const NV12_FULL_RANGE_FOURCC: u32 = u32::from_be_bytes(*b"420f");
/// `kCVPixelFormatType_444YpCbCr8BiPlanarVideoRange` — `'444v'`,
/// NV24-style biplanar 4:4:4 video-range. M-series VT HEVC decoder
/// emits this for HEVC Main 4:4:4 input.
pub const NV24_VIDEO_RANGE_FOURCC: u32 = u32::from_be_bytes(*b"444v");
/// `kCVPixelFormatType_444YpCbCr8BiPlanarFullRange` — `'444f'`.
pub const NV24_FULL_RANGE_FOURCC: u32 = u32::from_be_bytes(*b"444f");
/// `kCVPixelFormatType_444YpCbCr10BiPlanarVideoRange` / `'x444'` —
/// what VT decode lands in for HEVC 4:4:4 10-bit **video-range**
/// bitstreams, which is the only kind tether hosts emit (VAAPI /
/// Windows / macOS all tag limited range). Same biplanar 16-bit
/// MSB-aligned plane shape as `'xf44'`; differs only in declared
/// range, exactly like `'x420'` vs `'xf20'` for the 4:2:0 family.
/// A Linux host's stream decodes to this; the macOS↔macOS loopback
/// only ever produced full-range `'xf44'`, which is why this label
/// was missing from the accept set and cross-platform decode broke.
pub const X444_FOURCC: u32 = u32::from_be_bytes(*b"x444");
/// `'xf44'` — biplanar 10-bit 4:4:4 full-range. 16-bit MSB-aligned
/// planes.
pub const XF44_FOURCC: u32 = u32::from_be_bytes(*b"xf44");
/// `'P410'` — conventional fourcc for the same biplanar 10-bit 4:4:4
/// shape as `'xf44'`. VT decoder configurations may emit either label.
pub const P410_FOURCC: u32 = u32::from_be_bytes(*b"P410");
/// `'P010'` — biplanar 10-bit 4:2:0 full-range. 16-bit Y + interleaved
/// 16-bit UV at half-resolution, 10 bits MSB-aligned.
pub const P010_FOURCC: u32 = u32::from_be_bytes(*b"P010");
/// `kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange` / `'x420'` —
/// what VT decode lands in for HEVC Main10 video-range. Same shape
/// as `'P010'`; differs only in declared range.
pub const X420_FOURCC: u32 = u32::from_be_bytes(*b"x420");
/// `kCVPixelFormatType_420YpCbCr10BiPlanarFullRange` / `'xf20'`.
pub const XF20_FOURCC: u32 = u32::from_be_bytes(*b"xf20");

/// Whether the IOSurface import path will accept a given
/// `(chroma, bit_depth, fourcc)` triple. Cross-table consistency tests
/// compare this set against `videotoolbox::encoder::iosurface_fourcc_matches`
/// and `videotoolbox::expected_iosurface_fourccs` — drift between the
/// three is the family of bug that bit us when the renderer used to
/// reject `'x420'`.
///
/// **Range policy:** the renderer shader is hardcoded BT.709 limited
/// (see `tether-render::shader.wgsl`), so the surface a real decode
/// lands in is the **video-range** fourcc of each family — every
/// tether host (VAAPI / Windows / macOS) tags its bitstream
/// limited/video-range, so VT decode of a cross-platform stream
/// always emits the video-range label. Accordingly:
/// - 4:2:0 8-bit → `'420v'`; 4:4:4 8-bit → `'444v'`.
/// - 4:2:0 10-bit → `'x420'` (video range) plus the canonical
///   MSB-aligned `'P010'` label.
/// - 4:4:4 10-bit → `'x444'` (video range) plus `'xf44'` / `'P410'`.
///
/// `'xf44'` / `'P410'` are full-range labels retained because this
/// function plays a dual role: besides the renderer's decode-import
/// gate it is also the host-scaler destination gate
/// (`tether-gpuconvert::nv12_iosurface`), and the macOS encode /
/// SCK-capture path for 4:4:4 10-bit produces `'xf44'`. That encode
/// path is probe-gated, so the full-range label never actually
/// reaches the renderer in a live session; keeping it here preserves
/// the four-table lockstep with
/// `videotoolbox::encoder::iosurface_fourcc_matches`.
///
/// Full-range siblings that no path produces (`'xf20'`, `'444f'`)
/// stay rejected: accepting one would only mask a real "host sent a
/// range we can't display" problem rather than fix it.
#[must_use]
pub fn accepts_iosurface_fourcc(chroma: ChromaSubsampling, bit_depth: u8, fourcc: u32) -> bool {
    matches!(
        (chroma, bit_depth, fourcc),
        (ChromaSubsampling::Yuv420, 8, NV12_VIDEO_RANGE_FOURCC)
            | (ChromaSubsampling::Yuv444, 8, NV24_VIDEO_RANGE_FOURCC)
            | (ChromaSubsampling::Yuv420, 10, X420_FOURCC | P010_FOURCC)
            | (
                ChromaSubsampling::Yuv444,
                10,
                X444_FOURCC | XF44_FOURCC | P410_FOURCC
            )
    )
}

/// Human-readable label of the fourcc family expected for this
/// profile. Used to compose error messages when an unexpected surface
/// arrives; living alongside [`accepts_iosurface_fourcc`] keeps a new
/// family's accept entry and error label in lockstep.
#[must_use]
pub fn iosurface_fourcc_expected_label(chroma: ChromaSubsampling, bit_depth: u8) -> &'static str {
    match (chroma, bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => "'420v' (NV12 biplanar 8-bit)",
        (ChromaSubsampling::Yuv444, 8) => "'444v' (NV24 biplanar 8-bit)",
        (ChromaSubsampling::Yuv420, 10) => "'x420' or 'P010' (biplanar 4:2:0 10-bit)",
        (ChromaSubsampling::Yuv444, 10) => "'x444', 'xf44' or 'P410' (biplanar 4:4:4 10-bit)",
        _ => {
            "an IOSurface fourcc supported by this build (8-bit 4:2:0/4:4:4 or 10-bit 4:2:0/4:4:4)"
        }
    }
}

/// Errors the import helpers can produce. Distinct from
/// `tether-render::RenderError` so both downstream crates can map into
/// their own error type without one depending on the other.
#[derive(Debug, thiserror::Error)]
pub enum IOSurfaceImportError {
    #[error("IOSurface pointer is null")]
    NullSurface,
    #[error("device is not Metal-backed (wgpu::Device::as_hal::<Metal> returned None)")]
    NotMetalBacked,
    #[error("newTextureWithDescriptor:iosurface:plane: returned nil for plane {plane}")]
    MtlImportReturnedNil { plane: usize },
    #[error("{0}")]
    Other(String),
}

/// Caller-controlled knobs for a single-plane import. The renderer
/// passes `READ_ONLY_*` defaults (`ShaderRead | Private`,
/// `TEXTURE_BINDING`); the host bridge passes `READ_WRITE_*` defaults
/// (`ShaderRead | ShaderWrite | PixelFormatView | Shared`,
/// `TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC`) so its compute
/// scaler can write into the same MTLTexture that wraps a
/// host-allocated destination IOSurface.
#[derive(Debug, Clone, Copy)]
pub struct ImportPlaneOptions<'a> {
    /// Debug label propagated to the wgpu texture. Borrowed (not
    /// `'static`) so callers that compose labels at runtime — e.g.
    /// `"nv12-iosurface dst y plane (pool slot 2)"` — can pass a
    /// `&String`.
    pub label: &'a str,
    pub plane_index: usize,
    pub width: u32,
    pub height: u32,
    pub metal_format: MTLPixelFormat,
    pub wgpu_format: wgpu::TextureFormat,
    pub mtl_usage: MTLTextureUsage,
    pub mtl_storage: MTLStorageMode,
    pub wgpu_usage: wgpu::TextureUsages,
}

/// MTLTextureUsage flags for read-only renderer-side import (the
/// renderer samples the IOSurface in the YUV→RGB fragment shader and
/// nothing else writes to it).
pub const READ_ONLY_MTL_USAGE: MTLTextureUsage = MTLTextureUsage::ShaderRead;

/// MTLTextureUsage flags for host-side scaler I/O. `ShaderWrite` is
/// needed for compute-shader writes; `PixelFormatView` is needed
/// because wgpu may create internal views with a different format
/// (e.g. an sRGB-typed view of a linear-typed texture).
pub const READ_WRITE_MTL_USAGE: MTLTextureUsage = MTLTextureUsage(
    MTLTextureUsage::ShaderRead.0
        | MTLTextureUsage::ShaderWrite.0
        | MTLTextureUsage::PixelFormatView.0,
);

/// Reinterpret the [`IOSurfaceFrame`]'s opaque pointer as the typed
/// `&IOSurfaceRef`. The pointer came from `CVPixelBufferGetIOSurface`
/// (decoder path) or directly from SCK/`IOSurfaceCreate` (host path);
/// either way the caller must hold a retaining guard for as long as
/// the returned reference is in scope.
///
/// SAFETY: `IOSurfaceRef` is a `repr(C)` opaque type whose body is
/// `UnsafeCell<PhantomData<…>>` — a Rust-side marker that the C
/// IOSurface API may mutate state. No Rust code in this workspace
/// mutates through `&IOSurfaceRef`, and the C API is thread-safe on
/// a refcounted surface. Caller's responsibility is keeping the
/// surface alive (via [`crate::GpuFrameGuard`] on decode, the capture
/// `release_guard` on encode, or a pool's owning slot on the host
/// scaler output path).
pub fn iosurface_as_ref(iosurface: &IOSurfaceFrame) -> Result<&IOSurfaceRef, IOSurfaceImportError> {
    if iosurface.surface.is_null() {
        return Err(IOSurfaceImportError::NullSurface);
    }
    // SAFETY: see function-level doc.
    Ok(unsafe { &*(iosurface.surface as *const c_void as *const IOSurfaceRef) })
}

/// Wrap a single plane of an IOSurface as a `wgpu::Texture` with the
/// caller-specified Metal + wgpu usage flags and storage mode.
///
/// SAFETY (invariants the caller must uphold):
/// - `device` is Metal-backed (built on macOS via wgpu's metal backend).
/// - `surface_ref` is kept alive for as long as the returned texture
///   is in use. The MTLTexture retains the IOSurface internally
///   (Core Foundation refcount += 1), so once this call returns the
///   MTLTexture holds its own reference; releasing the caller's
///   refcount (e.g. pool slot recycle, decoder-side guard drop) is
///   safe — the underlying memory is not freed until the last
///   MTLTexture drops. What is NOT safe is releasing the surface
///   *more times than it was retained* (e.g. an unbalanced
///   `IOSurfaceDecrementUseCount`).
/// - `width`/`height` match the plane dims reported by
///   `IOSurfaceGetWidthOfPlane`/`HeightOfPlane`. Mismatched dims
///   cause Metal to either reject the texture or silently read past
///   the plane.
pub fn import_iosurface_plane(
    device: &wgpu::Device,
    surface_ref: &IOSurfaceRef,
    opts: ImportPlaneOptions<'_>,
) -> Result<wgpu::Texture, IOSurfaceImportError> {
    if opts.width == 0 || opts.height == 0 {
        return Err(IOSurfaceImportError::Other(format!(
            "plane dims must be non-zero (got {}x{})",
            opts.width, opts.height
        )));
    }

    // Pull the MTLDevice out of wgpu via the Metal HAL escape hatch.
    // Clone it so the borrow on `as_hal` doesn't last across the
    // MTLTexture construction below.
    // SAFETY: the wgpu device is built on the Metal backend by the
    // caller (precondition); `as_hal::<Metal>` returns Some.
    let mtl_device: Retained<ProtocolObject<dyn MTLDevice>> = unsafe {
        device
            .as_hal::<wgpu::hal::api::Metal>()
            .ok_or(IOSurfaceImportError::NotMetalBacked)?
            .raw_device()
            .clone()
    };

    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(MTLTextureType::Type2D);
    descriptor.setPixelFormat(opts.metal_format);
    // SAFETY: setWidth/setHeight/setMipmapLevelCount/setSampleCount/
    // setArrayLength are unsafe in objc2-metal because Apple's docs
    // allow rejection of pathological values. We pass 1 for the
    // scalar fields and non-zero plane dims (checked above) for
    // width/height.
    unsafe {
        descriptor.setWidth(opts.width as usize);
        descriptor.setHeight(opts.height as usize);
        descriptor.setMipmapLevelCount(1);
        descriptor.setSampleCount(1);
        descriptor.setArrayLength(1);
    }
    descriptor.setUsage(opts.mtl_usage);
    descriptor.setStorageMode(opts.mtl_storage);

    let mtl_texture = mtl_device
        .newTextureWithDescriptor_iosurface_plane(&descriptor, surface_ref, opts.plane_index)
        .ok_or(IOSurfaceImportError::MtlImportReturnedNil {
            plane: opts.plane_index,
        })?;

    let copy_size = wgpu::hal::CopyExtent {
        width: opts.width,
        height: opts.height,
        depth: 1,
    };
    // SAFETY: `texture_from_raw` requires the MTLTexture come from a
    // device matching the wgpu device; we just pulled the MTLDevice
    // out of `device` above, so by construction they match.
    let hal_texture = unsafe {
        wgpu::hal::metal::Device::texture_from_raw(
            mtl_texture,
            opts.wgpu_format,
            MTLTextureType::Type2D,
            1, // array_layers
            1, // mip_levels
            copy_size,
        )
    };

    let wgpu_desc = wgpu::TextureDescriptor {
        label: Some(opts.label),
        size: wgpu::Extent3d {
            width: opts.width,
            height: opts.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: opts.wgpu_format,
        usage: opts.wgpu_usage,
        view_formats: &[],
    };
    // SAFETY: `hal_texture` was built from this device's MTLDevice
    // above; `wgpu_desc` faithfully describes the same image (width,
    // height, format match the MTLTextureDescriptor and the CopyExtent).
    let texture =
        unsafe { device.create_texture_from_hal::<wgpu::hal::api::Metal>(hal_texture, &wgpu_desc) };
    Ok(texture)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decode-output fourcc the renderer MUST accept for each
    /// tether-negotiated profile, named explicitly so a regression
    /// shows up as a failed assert rather than a black screen. Every
    /// tether host encodes video/limited range, so VT decode of a
    /// cross-platform stream always lands in the **video-range**
    /// label of the family — that is the one the limited-range
    /// renderer shader displays correctly.
    ///
    /// `'x444'` is the entry this test exists to pin: it was missing
    /// for months because the macOS↔macOS loopback only ever produced
    /// full-range `'xf44'`, so cross-platform (Linux→macOS) 4:4:4
    /// 10-bit decode was rejected at import with the surface intact.
    #[test]
    fn renderer_accepts_video_range_decode_fourcc_for_every_negotiated_family() {
        let cases = [
            (ChromaSubsampling::Yuv420, 8, NV12_VIDEO_RANGE_FOURCC),
            (ChromaSubsampling::Yuv444, 8, NV24_VIDEO_RANGE_FOURCC),
            (ChromaSubsampling::Yuv420, 10, X420_FOURCC),
            (ChromaSubsampling::Yuv444, 10, X444_FOURCC),
        ];
        for (chroma, bit_depth, fourcc) in cases {
            assert!(
                accepts_iosurface_fourcc(chroma, bit_depth, fourcc),
                "renderer must accept the video-range decode fourcc 0x{fourcc:08x} \
                 for {chroma:?} {bit_depth}-bit — a real cross-platform decode lands here"
            );
        }
    }

    /// The MSB-aligned (`'P010'`/`'P410'`) and probe-gated full-range
    /// (`'xf44'`) labels the dual-role accept set also tolerates — see
    /// the function doc for why these stay in.
    #[test]
    fn renderer_accepts_alias_and_encode_path_labels() {
        for (chroma, bit_depth, fourcc) in [
            (ChromaSubsampling::Yuv420, 10, P010_FOURCC),
            (ChromaSubsampling::Yuv444, 10, XF44_FOURCC),
            (ChromaSubsampling::Yuv444, 10, P410_FOURCC),
        ] {
            assert!(
                accepts_iosurface_fourcc(chroma, bit_depth, fourcc),
                "accept set must keep 0x{fourcc:08x} for {chroma:?} {bit_depth}-bit"
            );
        }
    }

    /// Full-range siblings that no tether path produces are rejected:
    /// the renderer can't display them (shader is BT.709 limited), and
    /// silently accepting one would mask a "host sent a range we can't
    /// display" bug. `'xf44'` is deliberately absent here — it has the
    /// encode-path dual role covered above.
    #[test]
    fn renderer_rejects_unreachable_full_range_fourccs() {
        let nv12_full = u32::from_be_bytes(*b"420f");
        let nv24_full = u32::from_be_bytes(*b"444f");
        for (chroma, bit_depth, fourcc) in [
            (ChromaSubsampling::Yuv420, 8, nv12_full),
            (ChromaSubsampling::Yuv444, 8, nv24_full),
            (ChromaSubsampling::Yuv420, 10, XF20_FOURCC),
        ] {
            assert!(
                !accepts_iosurface_fourcc(chroma, bit_depth, fourcc),
                "renderer must reject unreachable full-range fourcc 0x{fourcc:08x} \
                 for {chroma:?} {bit_depth}-bit"
            );
        }
    }

    /// A fourcc from the wrong chroma/bit-depth family is rejected —
    /// e.g. a 4:2:0 surface arriving on a 4:4:4-negotiated session
    /// (the silent-downsample failure mode) must not import.
    #[test]
    fn renderer_rejects_cross_family_fourcc() {
        assert!(
            !accepts_iosurface_fourcc(ChromaSubsampling::Yuv444, 10, X420_FOURCC),
            "4:2:0 10-bit fourcc must not import on a 4:4:4 10-bit session"
        );
        assert!(
            !accepts_iosurface_fourcc(ChromaSubsampling::Yuv420, 10, X444_FOURCC),
            "4:4:4 10-bit fourcc must not import on a 4:2:0 10-bit session"
        );
        assert!(
            !accepts_iosurface_fourcc(ChromaSubsampling::Yuv444, 8, X444_FOURCC),
            "10-bit fourcc must not import on an 8-bit session"
        );
    }
}
