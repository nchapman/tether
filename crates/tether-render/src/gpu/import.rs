//! Linux DMA-BUF import path: VAAPI-decoded surface → `wgpu::Texture`
//! plane(s) the renderer can sample. Dispatches on the negotiated
//! chroma — NV12 gets 2 planes (Y as R8, UV as Rg8); YUV444 gets 3
//! planes (Y/U/V each as R8). Pure Linux module — macOS will land its
//! IOSurface equivalent here as a sibling file.

use tether_codec::{DmaBufFrame, GpuFrameGuard};
use tether_protocol::control::ChromaSubsampling;

use crate::{RenderError, Result};

use super::{YuvPlanes, YuvTextures};

#[allow(clippy::cast_lossless)] // u32 pitch into u64 stride is intentional
pub(crate) fn import_dmabuf_textures(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    chroma: ChromaSubsampling,
    dmabuf: &DmaBufFrame,
    width: u32,
    height: u32,
    guard: GpuFrameGuard,
) -> Result<YuvTextures> {
    // Log the shape libva handed us at debug so a driver that emits an
    // unexpected layer count (e.g. AMD VA exporting multi-plane YUV444P
    // as a single layer rather than three) is self-diagnosing instead
    // of producing an opaque "expected N layers, got M" error with no
    // hint about what to change.
    tracing::debug!(
        chroma = ?chroma,
        fourcc = format_args!("0x{:08x}", dmabuf.fourcc),
        layers = dmabuf.layers.len(),
        planes_per_layer = ?dmabuf
            .layers
            .iter()
            .map(|l| l.num_planes)
            .collect::<Vec<_>>(),
        objects = dmabuf.objects.len(),
        "import_dmabuf_textures dispatch"
    );
    match chroma {
        ChromaSubsampling::Yuv420 => {
            import_nv12(device, layout, sampler, dmabuf, width, height, guard)
        }
        ChromaSubsampling::Yuv444 => {
            import_yuv444(device, layout, sampler, dmabuf, width, height, guard)
        }
    }
}

fn import_nv12(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    dmabuf: &DmaBufFrame,
    width: u32,
    height: u32,
    guard: GpuFrameGuard,
) -> Result<YuvTextures> {
    if dmabuf.layers.len() != 2 {
        return Err(RenderError::DmaBufImport(format!(
            "expected 2 layers (NV12 SEPARATE_LAYERS), got {}",
            dmabuf.layers.len()
        )));
    }
    let chroma_w = width.div_ceil(2);
    let chroma_h = height.div_ceil(2);
    let y = import_one_layer(
        device,
        "tether-render y plane (dmabuf)",
        dmabuf,
        0,
        width,
        height,
        wgpu::TextureFormat::R8Unorm,
    )?;
    let uv = import_one_layer(
        device,
        "tether-render uv plane (dmabuf)",
        dmabuf,
        1,
        chroma_w,
        chroma_h,
        wgpu::TextureFormat::Rg8Unorm,
    )?;
    let y_view = y.create_view(&wgpu::TextureViewDescriptor::default());
    let uv_view = uv.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tether-render yuv bind group (dmabuf nv12)"),
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
    Ok(YuvTextures {
        planes: YuvPlanes::Nv12 { y, uv },
        bind_group,
        size: (width, height),
        _guard: Some(guard),
    })
}

/// VAAPI exports a Main444 surface as a single packed XYUV layer:
/// one DRM object, one layer with fourcc `XYUV` (DRM_FORMAT_XYUV8888),
/// one plane (32 bpp, byte order V/U/Y/X). Confirmed on Intel
/// media-driver via the `hevc_main444_dmabuf_roundtrip` hardware
/// test in `tether-codec/src/vaapi/tests.rs`.
///
/// Planar YUV444P is *not* a possible shape on this codepath:
/// ffmpeg's `vaapi_drm_format_map` has no DRM_PRIME entry for
/// planar 4:4:4 8-bit, so libavcodec returns the packed XYUV form
/// regardless of how the driver allocated the surface internally.
/// (See `bgra_to_yuv444.wgsl` for the matching encoder-side rationale.)
fn import_yuv444(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    dmabuf: &DmaBufFrame,
    width: u32,
    height: u32,
    guard: GpuFrameGuard,
) -> Result<YuvTextures> {
    if dmabuf.layers.len() != 1 {
        return Err(RenderError::DmaBufImport(format!(
            "YUV444 dma-buf has {} layers; expected 1 (packed XYUV)",
            dmabuf.layers.len()
        )));
    }
    let layer = &dmabuf.layers[0];
    if layer.num_planes != 1 {
        return Err(RenderError::DmaBufImport(format!(
            "YUV444 packed layer should have 1 plane, got {}",
            layer.num_planes
        )));
    }
    let expected_fourcc = u32::from_le_bytes(*b"XYUV");
    if layer.drm_format != expected_fourcc {
        return Err(RenderError::DmaBufImport(format!(
            "YUV444 layer fourcc 0x{:08x} != expected XYUV (0x{:08x})",
            layer.drm_format, expected_fourcc
        )));
    }
    let packed = import_one_layer(
        device,
        "tether-render xyuv packed (dmabuf 444)",
        dmabuf,
        0,
        width,
        height,
        wgpu::TextureFormat::Rgba8Unorm,
    )?;
    let packed_view = packed.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tether-render yuv bind group (dmabuf 444 packed)"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&packed_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    Ok(YuvTextures {
        planes: YuvPlanes::Yuv444 { packed },
        bind_group,
        size: (width, height),
        _guard: Some(guard),
    })
}

fn import_one_layer(
    device: &wgpu::Device,
    label: &str,
    dmabuf: &DmaBufFrame,
    layer_idx: usize,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> Result<wgpu::Texture> {
    let layer = &dmabuf.layers[layer_idx];
    // SEPARATE_LAYERS gives one plane per layer; multi-plane within a
    // layer would mean we're looking at a COMPOSED export, which the
    // YUV444 path imports via `import_one_plane` instead.
    if layer.num_planes != 1 {
        return Err(RenderError::DmaBufImport(format!(
            "layer {layer_idx} has {} planes; expected 1 (SEPARATE_LAYERS)",
            layer.num_planes
        )));
    }
    import_one_plane(device, label, dmabuf, layer_idx, 0, width, height, format)
}

/// Import one plane of a (possibly multi-plane) layer. Generalises
/// `import_one_layer` for the COMPOSED YUV444 case, where a single
/// `DmaBufLayer` carries three plane offsets within one DRM object.
/// For the SEPARATE_LAYERS case (`plane_idx=0`), this collapses to
/// the same code path.
#[allow(clippy::too_many_arguments)]
fn import_one_plane(
    device: &wgpu::Device,
    label: &str,
    dmabuf: &DmaBufFrame,
    layer_idx: usize,
    plane_idx: usize,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> Result<wgpu::Texture> {
    let layer = &dmabuf.layers[layer_idx];
    if plane_idx >= layer.num_planes as usize {
        return Err(RenderError::DmaBufImport(format!(
            "layer {layer_idx} plane {plane_idx} out of range ({} planes)",
            layer.num_planes
        )));
    }
    let obj_idx = layer.object_index[plane_idx] as usize;
    let obj = dmabuf.objects.get(obj_idx).ok_or_else(|| {
        RenderError::DmaBufImport(format!(
            "layer {layer_idx} plane {plane_idx} references object {obj_idx} but \
             only {} present",
            dmabuf.objects.len()
        ))
    })?;
    // dup the fd because wgpu takes ownership and the same object may
    // also back another plane. `try_clone` is dup(2) with CLOEXEC.
    let fd = obj
        .fd
        .try_clone()
        .map_err(|e| RenderError::DmaBufImport(format!("dup dma-buf fd: {e}")))?;

    let hal_desc = wgpu::hal::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUses::RESOURCE,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };

    // SAFETY: the device was created from this hal::Api (Vulkan is
    // wgpu's default backend on Linux); `as_hal` returns Some for the
    // matching API. `texture_from_dmabuf_fd` consumes the fd on
    // success and closes it on failure, so we hand over our duped one.
    // `texture_from_raw` (called inside) requires the hal_texture be
    // created from this device, which is exactly what we did.
    let hal_texture = unsafe {
        let hal_dev = device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or_else(|| RenderError::DmaBufImport("device is not Vulkan-backed".into()))?;
        hal_dev
            .texture_from_dmabuf_fd(
                fd,
                &hal_desc,
                obj.drm_format_modifier,
                u64::from(layer.pitch[plane_idx]),
                u64::from(layer.offset[plane_idx]),
            )
            .map_err(|e| {
                RenderError::DmaBufImport(format!("texture_from_dmabuf_fd: {e:?}"))
            })?
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
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    // SAFETY: hal_texture was just built from this device on the same
    // Vulkan backend. The wgpu_desc must describe the same image as
    // the hal_desc — width/height/format/dimension are identical
    // above; `usage` deliberately differs (`TextureUses::RESOURCE` on
    // the hal side, `TextureUsages::TEXTURE_BINDING` on the wgpu side
    // — they're the equivalent representations in each API's vocabulary
    // and that's the correct pairing).
    let texture = unsafe { device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_texture, &wgpu_desc) };
    Ok(texture)
}
