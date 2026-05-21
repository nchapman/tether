//! macOS IOSurface import path: VideoToolbox-decoded IOSurface →
//! `wgpu::Texture` plane(s) the renderer can sample. Mirrors the Linux
//! DMA-BUF import path in `import.rs`, just with Metal HAL on the
//! receiving end.
//!
//! The flow per frame:
//!
//! 1. Pull the underlying `MTLDevice` out of wgpu via the Metal HAL
//!    escape hatch (`as_hal::<Metal>()`).
//! 2. For each plane of the IOSurface, call
//!    `MTLDevice::newTextureWithDescriptor:iosurface:plane:` to wrap
//!    that plane in an `MTLTexture`. The texture borrows storage from
//!    the IOSurface — no allocation, no copy.
//! 3. Hand the raw `MTLTexture` to wgpu via
//!    `wgpu::hal::api::Metal::Device::texture_from_raw` and then
//!    `device.create_texture_from_hal`, the Metal-side equivalent of
//!    the Vulkan `texture_from_dmabuf_fd` path.
//!
//! Lifetime: the renderer holds the decoder's `GpuFrameGuard` (the
//! `AVFrame` whose Drop releases the `CVPixelBufferRef`, which in turn
//! releases the IOSurface) for as long as the `YuvTextures` are
//! retained, just like the Linux side keeps the VAAPI surface alive
//! across the GPU read. The `MTLTexture` retains the IOSurface
//! internally as long as it exists, so there is no double-free risk if
//! the guard drops before the MTLTexture; the typical order is the
//! opposite — `YuvTextures` drops, which drops both the MTLTextures
//! and the guard.

use std::ffi::c_void;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_io_surface::IOSurfaceRef;
use objc2_metal::{
    MTLDevice, MTLPixelFormat, MTLStorageMode, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};
use tether_codec::{GpuFrameGuard, IOSurfaceFrame};
use tether_protocol::control::ChromaSubsampling;

use crate::{RenderError, Result};

use super::{YuvPlanes, YuvTextures};

/// `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange` — `'420v'`, the
/// canonical NV12 video-range fourcc Apple's frameworks use. The host
/// side (ScreenCaptureKit) is configured to deliver this; VideoToolbox
/// decoders typically emit it too.
const NV12_VIDEO_RANGE_FOURCC: u32 = u32::from_be_bytes(*b"420v");
/// `kCVPixelFormatType_420YpCbCr8BiPlanarFullRange` — `'420f'`. Apple's
/// hardware can decode full-range bitstreams into this layout. The
/// plane shape is identical to `'420v'`; only the YUV→RGB matrix +
/// range expansion in the shader differ, and that's driven by the
/// per-session `VideoColorSpec`, not the IOSurface fourcc.
const NV12_FULL_RANGE_FOURCC: u32 = u32::from_be_bytes(*b"420f");
/// `kCVPixelFormatType_444YpCbCr8BiPlanarVideoRange` — `'444v'`,
/// NV24-style biplanar 4:4:4 video-range. M-series Apple Silicon's
/// VideoToolbox HEVC decoder emits this for HEVC Main 4:4:4 input.
/// Plane shape is identical to NV12 except the UV plane is full-res
/// instead of half-res; same R8 + Rg8 textures, same fragment shader,
/// no special handling needed in the import path.
const NV24_VIDEO_RANGE_FOURCC: u32 = u32::from_be_bytes(*b"444v");
/// `kCVPixelFormatType_444YpCbCr8BiPlanarFullRange` — `'444f'`,
/// full-range counterpart of `'444v'`. Same plane shape; range is
/// signalled through `VideoColorSpec`, not the fourcc.
const NV24_FULL_RANGE_FOURCC: u32 = u32::from_be_bytes(*b"444f");
/// `kCVPixelFormatType_4444AYpCbCr10` style biplanar 4:4:4 10-bit
/// full-range — Apple's `'xf44'`. SCK exposes this as its documented
/// 10-bit 4:4:4 capture format and VT decoders can land here for
/// 10-bit Main 4:4:4 input. Plane shape is biplanar with 16-bit Y +
/// 16-bit interleaved UV cells, 10 bits MSB-aligned in each.
const XF44_FOURCC: u32 = u32::from_be_bytes(*b"xf44");
/// `kCVPixelFormatType_4444YpCbCrA10` / `'P410'` — biplanar 10-bit
/// 4:4:4 full-range, the conventional fourcc for the same shape as
/// `'xf44'`. Some VT decoder configurations emit this label instead;
/// both have identical 16-bit MSB-aligned plane layouts so they
/// import through the same code path.
const P410_FOURCC: u32 = u32::from_be_bytes(*b"P410");
/// `'P010'` — biplanar 10-bit 4:2:0 full-range. VT's Main10 decode
/// path lands here. Shape: 16-bit Y plane + interleaved 16-bit UV
/// plane at half resolution, 10 bits MSB-aligned in each cell.
const P010_FOURCC: u32 = u32::from_be_bytes(*b"P010");

pub(crate) fn import_iosurface_textures(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    chroma: ChromaSubsampling,
    bit_depth: u8,
    iosurface: &IOSurfaceFrame,
    guard: GpuFrameGuard,
) -> Result<YuvTextures> {
    let (fourcc_ok, expected) = match (chroma, bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => (
            matches!(
                iosurface.pixel_format,
                NV12_VIDEO_RANGE_FOURCC | NV12_FULL_RANGE_FOURCC
            ),
            "'420v' or '420f' (NV12 biplanar 8-bit)",
        ),
        (ChromaSubsampling::Yuv444, 8) => (
            matches!(
                iosurface.pixel_format,
                NV24_VIDEO_RANGE_FOURCC | NV24_FULL_RANGE_FOURCC
            ),
            "'444v' or '444f' (NV24 biplanar 8-bit)",
        ),
        // 10-bit 4:2:0: VT's Main10 decode lands in a 'P010' IOSurface
        // with biplanar R16/Rg16 plane formats (10 bits MSB-aligned).
        // Same fourcc family as the Linux DRM P010.
        (ChromaSubsampling::Yuv420, 10) => (
            iosurface.pixel_format == P010_FOURCC,
            "'P010' (biplanar 4:2:0 10-bit)",
        ),
        (ChromaSubsampling::Yuv444, 10) => (
            matches!(iosurface.pixel_format, XF44_FOURCC | P410_FOURCC),
            "'xf44' or 'P410' (biplanar 4:4:4 10-bit)",
        ),
        // Other (chroma, bit_depth) combos haven't been wired; surface
        // an actionable error rather than mapping them silently.
        _ => (
            false,
            "an IOSurface fourcc supported by this build (8-bit 4:2:0/4:4:4 or 10-bit 4:2:0/4:4:4)",
        ),
    };
    if !fourcc_ok {
        return Err(RenderError::DmaBufImport(format!(
            "IOSurface pixel format 0x{:08x} doesn't match negotiated profile \
             ({chroma:?} {bit_depth}-bit); expected {expected}",
            iosurface.pixel_format
        )));
    }
    // Pick per-plane Metal + wgpu formats from bit depth. Plane *count*
    // (always 2 for the biplanar fourccs we accept) and per-plane dims
    // come from the IOSurface — Apple reports the correct half-res or
    // full-res UV per the source fourcc.
    let (y_mtl, y_wgpu, uv_mtl, uv_wgpu, hbd) = match bit_depth {
        8 => (
            MTLPixelFormat::R8Unorm,
            wgpu::TextureFormat::R8Unorm,
            MTLPixelFormat::RG8Unorm,
            wgpu::TextureFormat::Rg8Unorm,
            false,
        ),
        10 => (
            MTLPixelFormat::R16Unorm,
            wgpu::TextureFormat::R16Unorm,
            MTLPixelFormat::RG16Unorm,
            wgpu::TextureFormat::Rg16Unorm,
            true,
        ),
        other => {
            return Err(RenderError::DmaBufImport(format!(
                "no IOSurface plane format wired for {other}-bit input"
            )))
        }
    };
    import_biplanar(
        device, layout, sampler, iosurface, guard, y_mtl, y_wgpu, uv_mtl, uv_wgpu, hbd,
    )
}

#[allow(clippy::too_many_arguments)]
fn import_biplanar(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    iosurface: &IOSurfaceFrame,
    guard: GpuFrameGuard,
    y_mtl: MTLPixelFormat,
    y_wgpu: wgpu::TextureFormat,
    uv_mtl: MTLPixelFormat,
    uv_wgpu: wgpu::TextureFormat,
    high_bit_depth: bool,
) -> Result<YuvTextures> {
    let surface_ref = iosurface_as_ref(iosurface)?;
    // Plane dims via IOSurface itself. For odd resolutions the
    // per-plane dims can differ from `width.div_ceil(2)` (e.g. a
    // 1919×1079 source yields a 960×540 UV plane on some drivers and
    // 959×539 on others); the IOSurface is the ground truth.
    if surface_ref.plane_count() < 2 {
        return Err(RenderError::DmaBufImport(format!(
            "biplanar IOSurface should have 2 planes; got {}",
            surface_ref.plane_count()
        )));
    }
    let (y_w, y_h) = (
        surface_ref.width_of_plane(0) as u32,
        surface_ref.height_of_plane(0) as u32,
    );
    let (uv_w, uv_h) = (
        surface_ref.width_of_plane(1) as u32,
        surface_ref.height_of_plane(1) as u32,
    );
    // Reject zero-dimension planes before they reach
    // `MTLTextureDescriptor::setWidth/setHeight`, which is documented as
    // UB for zero values. A pathological bitstream (or some Apple
    // hardware that has historically produced 0-height UV planes for
    // 1-row luma) could otherwise abort the renderer thread.
    if y_w == 0 || y_h == 0 || uv_w == 0 || uv_h == 0 {
        return Err(RenderError::DmaBufImport(format!(
            "IOSurface plane has zero dimension: Y={y_w}x{y_h} UV={uv_w}x{uv_h}"
        )));
    }

    let y = import_plane(
        device,
        "tether-render y plane (iosurface)",
        surface_ref,
        0,
        y_w,
        y_h,
        y_mtl,
        y_wgpu,
    )?;
    let uv = import_plane(
        device,
        "tether-render uv plane (iosurface)",
        surface_ref,
        1,
        uv_w,
        uv_h,
        uv_mtl,
        uv_wgpu,
    )?;
    let y_view = y.create_view(&wgpu::TextureViewDescriptor::default());
    let uv_view = uv.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(if high_bit_depth {
            "tether-render yuv bind group (iosurface biplanar 16)"
        } else {
            "tether-render yuv bind group (iosurface biplanar 8)"
        }),
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
    // Public size tag is the luma-plane dims — the renderer uses this
    // for letterboxing and for the cursor-normalisation math in lib.rs.
    let planes = if high_bit_depth {
        YuvPlanes::Biplanar16 { y, uv }
    } else {
        YuvPlanes::Biplanar8 { y, uv }
    };
    Ok(YuvTextures {
        planes,
        bind_group,
        size: (y_w, y_h),
        _guard: Some(guard),
    })
}

/// Reinterpret the renderer-side `IOSurfaceFrame` (which carries a raw
/// `*mut c_void` pointer) as the typed objc2 `&IOSurfaceRef`. The
/// pointer came from `CVPixelBufferGetIOSurface` on the decoder side
/// and is kept alive by the `GpuFrameGuard` (the underlying AVFrame
/// whose Drop releases the CVPixelBuffer ref).
///
/// SAFETY: `IOSurfaceRef` is a `repr(C)` opaque type with a zero-size
/// inline body and a single `UnsafeCell<PhantomData<…>>` marker field
/// (see `objc2-io-surface::IOSurfaceRef`). The `UnsafeCell` is a
/// Rust-side opaque marker that the binding uses to express "may be
/// mutated through &IOSurfaceRef by the C API" — no Rust code in this
/// crate mutates through it, and the C IOSurface API the marker
/// describes is thread-safe on a refcounted surface. The struct also
/// carries `PhantomPinned`, but that only constrains `Pin<&mut T>`
/// ergonomics and is unaffected by a shared `&IOSurfaceRef` derived
/// from a raw pointer. The caller must hold the guard for as long as
/// the returned reference is used; that's the contract of this module.
fn iosurface_as_ref(iosurface: &IOSurfaceFrame) -> Result<&IOSurfaceRef> {
    if iosurface.surface.is_null() {
        return Err(RenderError::DmaBufImport(
            "IOSurface pointer is null".into(),
        ));
    }
    // SAFETY: see function-level doc.
    Ok(unsafe { &*(iosurface.surface as *const c_void as *const IOSurfaceRef) })
}

#[allow(clippy::too_many_arguments)]
fn import_plane(
    device: &wgpu::Device,
    label: &str,
    surface_ref: &IOSurfaceRef,
    plane: usize,
    width: u32,
    height: u32,
    metal_format: MTLPixelFormat,
    wgpu_format: wgpu::TextureFormat,
) -> Result<wgpu::Texture> {
    // Pull the underlying MTLDevice out of wgpu via the Metal HAL
    // escape hatch. Then hold it as a separate clone so the inner
    // borrow on `device.as_hal` doesn't last across the MTLTexture
    // construction (and so we can release the hal handle promptly).
    // SAFETY: the wgpu device was created on the Metal backend; on
    // macOS that's the default and `as_hal::<Metal>` returns Some.
    let mtl_device: Retained<ProtocolObject<dyn MTLDevice>> = unsafe {
        device
            .as_hal::<wgpu::hal::api::Metal>()
            .ok_or_else(|| RenderError::DmaBufImport("device is not Metal-backed".into()))?
            .raw_device()
            .clone()
    };

    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(MTLTextureType::Type2D);
    descriptor.setPixelFormat(metal_format);
    // SAFETY: setWidth/setHeight/setMipmapLevelCount/setSampleCount/
    // setArrayLength are marked unsafe by objc2-metal because Apple's
    // docs reserve the right to reject pathological values (zero,
    // huge). We pass 1 for the scalar fields and the IOSurface's own
    // dims for width/height (which the decoder side already produced
    // from `CVPixelBufferGetWidth/Height`, so they're non-zero and
    // within MTL's hard limits).
    unsafe {
        descriptor.setWidth(width as usize);
        descriptor.setHeight(height as usize);
        descriptor.setMipmapLevelCount(1);
        descriptor.setSampleCount(1);
        descriptor.setArrayLength(1);
    }
    descriptor.setUsage(MTLTextureUsage::ShaderRead);
    // Storage mode for IOSurface-backed textures is dictated by the
    // IOSurface itself, so Metal ignores this field in practice. Apple's
    // sample code and CVMetalTextureCache both pass Private, which also
    // avoids a Metal validation-layer warning on discrete-GPU Macs
    // (where Shared is reserved for unified-memory devices). Match that
    // pattern.
    descriptor.setStorageMode(MTLStorageMode::Private);

    let mtl_texture = mtl_device
        .newTextureWithDescriptor_iosurface_plane(&descriptor, surface_ref, plane)
        .ok_or_else(|| {
            RenderError::DmaBufImport(format!(
                "newTextureWithDescriptor:iosurface:plane: returned nil for plane {plane}",
            ))
        })?;

    // Hand the MTLTexture to wgpu. `texture_from_raw` builds a
    // `hal::metal::Texture`; `create_texture_from_hal` wraps it as a
    // first-class `wgpu::Texture`. Same pairing the Linux Vulkan path
    // uses with `texture_from_dmabuf_fd` + `create_texture_from_hal`.
    let copy_size = wgpu::hal::CopyExtent {
        width,
        height,
        depth: 1,
    };
    // SAFETY: `texture_from_raw` is unsafe because it requires the
    // caller to ensure the MTLTexture was created from a device
    // compatible with the wgpu device. We just pulled the MTLDevice
    // out of the wgpu device above, so by construction they match.
    let hal_texture = unsafe {
        wgpu::hal::metal::Device::texture_from_raw(
            mtl_texture,
            wgpu_format,
            objc2_metal::MTLTextureType::Type2D,
            1, // array_layers
            1, // mip_levels
            copy_size,
        )
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
        format: wgpu_format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    // SAFETY: hal_texture was built from this device's underlying
    // MTLDevice above; wgpu_desc faithfully describes the same image
    // (width/height/format match the MTLTextureDescriptor and the
    // hal CopyExtent passed to texture_from_raw).
    let texture = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Metal>(hal_texture, &wgpu_desc)
    };
    Ok(texture)
}
