//! macOS IOSurface import path: VideoToolbox-decoded IOSurface →
//! `wgpu::Texture` plane(s) the renderer can sample.
//!
//! The IOSurface→MTLTexture→wgpu primitive lives in
//! `tether_codec::macos_interop` so the host scaler bridge
//! (`tether_gpuconvert::nv12_iosurface`) can share it. This module is
//! the renderer-specific layer: it consumes biplanar Y+UV surfaces and
//! produces a [`YuvTextures`] bind group ready for the YUV→RGB shader.
//!
//! Lifetime: the renderer holds the decoder's `GpuFrameGuard` (the
//! `AVFrame` whose Drop releases the `CVPixelBufferRef`, which in turn
//! releases the IOSurface) for as long as the `YuvTextures` are
//! retained. The `MTLTexture` retains the IOSurface internally as long
//! as it exists, so there is no double-free risk if the guard drops
//! before the MTLTexture; the typical order is the opposite —
//! `YuvTextures` drops, which drops both the MTLTextures and the guard.

use objc2_metal::{MTLPixelFormat, MTLStorageMode};
use tether_codec::macos_interop::{
    accepts_iosurface_fourcc, import_iosurface_plane, iosurface_as_ref,
    iosurface_fourcc_expected_label, ImportPlaneOptions, READ_ONLY_MTL_USAGE,
};
use tether_codec::{GpuFrameGuard, IOSurfaceFrame};
use tether_protocol::control::ChromaSubsampling;

use crate::{RenderError, Result};

use super::{YuvPlanes, YuvTextures};

fn map_import_err(err: tether_codec::macos_interop::IOSurfaceImportError) -> RenderError {
    RenderError::DmaBufImport(err.to_string())
}

pub(crate) fn import_iosurface_textures(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    chroma: ChromaSubsampling,
    bit_depth: u8,
    iosurface: &IOSurfaceFrame,
    guard: GpuFrameGuard,
) -> Result<YuvTextures> {
    if !accepts_iosurface_fourcc(chroma, bit_depth, iosurface.pixel_format) {
        return Err(RenderError::DmaBufImport(format!(
            "IOSurface pixel format 0x{:08x} doesn't match negotiated profile \
             ({chroma:?} {bit_depth}-bit); expected {}",
            iosurface.pixel_format,
            iosurface_fourcc_expected_label(chroma, bit_depth)
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
    let surface_ref = iosurface_as_ref(iosurface).map_err(map_import_err)?;
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
    if y_w == 0 || y_h == 0 || uv_w == 0 || uv_h == 0 {
        return Err(RenderError::DmaBufImport(format!(
            "IOSurface plane has zero dimension: Y={y_w}x{y_h} UV={uv_w}x{uv_h}"
        )));
    }

    // Renderer-side import: read-only sampled access, Private storage
    // mode (Apple's sample code + CVMetalTextureCache pattern; avoids
    // a validation-layer warning on discrete-GPU Macs).
    let y = import_iosurface_plane(
        device,
        surface_ref,
        ImportPlaneOptions {
            label: "tether-render y plane (iosurface)",
            plane_index: 0,
            width: y_w,
            height: y_h,
            metal_format: y_mtl,
            wgpu_format: y_wgpu,
            mtl_usage: READ_ONLY_MTL_USAGE,
            mtl_storage: MTLStorageMode::Private,
            wgpu_usage: wgpu::TextureUsages::TEXTURE_BINDING,
        },
    )
    .map_err(map_import_err)?;
    let uv = import_iosurface_plane(
        device,
        surface_ref,
        ImportPlaneOptions {
            label: "tether-render uv plane (iosurface)",
            plane_index: 1,
            width: uv_w,
            height: uv_h,
            metal_format: uv_mtl,
            wgpu_format: uv_wgpu,
            mtl_usage: READ_ONLY_MTL_USAGE,
            mtl_storage: MTLStorageMode::Private,
            wgpu_usage: wgpu::TextureUsages::TEXTURE_BINDING,
        },
    )
    .map_err(map_import_err)?;

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
