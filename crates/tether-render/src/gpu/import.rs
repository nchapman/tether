//! Linux DMA-BUF import path: VAAPI-decoded surface → `wgpu::Texture`
//! plane(s) the renderer can sample. Dispatches on the negotiated
//! `(chroma, bit_depth)` — 8-bit 4:2:0 gets biplanar NV12 (R8 Y +
//! Rg8 UV), 8-bit 4:4:4 gets packed XYUV in a single Rgba8 texture,
//! 10-bit gets biplanar P010/P410 (R16 Y + Rg16 UV). Pure Linux
//! module — macOS lands its IOSurface equivalent in `metal.rs`.

use tether_codec::{DmaBufFrame, GpuFrameGuard};
use tether_protocol::control::ChromaSubsampling;

use crate::{RenderError, Result};

use super::{YuvPlanes, YuvTextures};

#[allow(clippy::cast_lossless)] // u32 pitch into u64 stride is intentional
#[allow(clippy::too_many_arguments)]
pub(crate) fn import_dmabuf_textures(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    chroma: ChromaSubsampling,
    bit_depth: u8,
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
        bit_depth,
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
    match (chroma, bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => import_biplanar(
            device, layout, sampler, chroma, dmabuf, width, height, guard, false,
        ),
        (ChromaSubsampling::Yuv444, 8) => {
            import_yuv444_packed(device, layout, sampler, dmabuf, width, height, guard)
        }
        (ChromaSubsampling::Yuv420 | ChromaSubsampling::Yuv444, 10) => {
            // 10-bit 4:2:0 (P010) and 4:4:4 (P410) both land here:
            // biplanar 16-bit cells, MSB-aligned 10-bit data. UV plane
            // dims come from `chroma` (half-res for 4:2:0, full-res
            // for 4:4:4) — passing it through means we never have to
            // sniff the DRM fourcc for layout decisions, which would
            // be wrong for any driver that emits a non-P010/P410
            // fourcc for the same (chroma, bit_depth) shape.
            //
            // **Unverified for 4:4:4 10-bit on Linux**: the encoder
            // takes packed XV30 input, but VAAPI's *decode* output
            // format for HEVC Main 4:4:4 10-bit is driver-dependent —
            // RADV has emitted both packed XV30 (1 layer) and
            // biplanar P410-style 16-bit (2 layers) across Mesa
            // versions. If `import_biplanar` errors below with
            // "expected 2 layers, got 1" on a 4:4:4 10-bit frame,
            // the driver emitted packed XV30 and the renderer needs
            // a new `RenderLayout::Packed1010102` import variant
            // analogous to `PackedXYUV`. The hardware test
            // `dmabuf_test::roundtrip_hevc_main444_10bit_identity`
            // is the gate that surfaces this on RADV/NVK.
            import_biplanar(
                device, layout, sampler, chroma, dmabuf, width, height, guard, true,
            )
        }
        _ => Err(RenderError::DmaBufImport(format!(
            "no Linux dma-buf import path wired for {chroma:?} {bit_depth}-bit"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn import_biplanar(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    chroma: ChromaSubsampling,
    dmabuf: &DmaBufFrame,
    width: u32,
    height: u32,
    guard: GpuFrameGuard,
    high_bit_depth: bool,
) -> Result<YuvTextures> {
    if dmabuf.layers.len() != 2 {
        return Err(RenderError::DmaBufImport(format!(
            "expected 2 layers (biplanar SEPARATE_LAYERS), got {}",
            dmabuf.layers.len()
        )));
    }
    // Chroma sub-sampling drives the UV plane dimensions directly;
    // the bit depth picks the per-plane formats further down. We
    // never look at the DRM fourcc for layout decisions — the
    // negotiated `chroma` is the source of truth.
    let (chroma_w, chroma_h) = match chroma {
        ChromaSubsampling::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
        ChromaSubsampling::Yuv444 => (width, height),
    };
    let (y_fmt, uv_fmt) = if high_bit_depth {
        (wgpu::TextureFormat::R16Unorm, wgpu::TextureFormat::Rg16Unorm)
    } else {
        (wgpu::TextureFormat::R8Unorm, wgpu::TextureFormat::Rg8Unorm)
    };
    let y = import_one_layer(
        device,
        if high_bit_depth {
            "tether-render y plane (dmabuf 16-bit)"
        } else {
            "tether-render y plane (dmabuf 8-bit)"
        },
        dmabuf,
        0,
        width,
        height,
        y_fmt,
    )?;
    let uv = import_one_layer(
        device,
        if high_bit_depth {
            "tether-render uv plane (dmabuf 16-bit)"
        } else {
            "tether-render uv plane (dmabuf 8-bit)"
        },
        dmabuf,
        1,
        chroma_w,
        chroma_h,
        uv_fmt,
    )?;
    let y_view = y.create_view(&wgpu::TextureViewDescriptor::default());
    let uv_view = uv.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(if high_bit_depth {
            "tether-render yuv bind group (dmabuf biplanar 16)"
        } else {
            "tether-render yuv bind group (dmabuf biplanar 8)"
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
/// (See `tether-gpuconvert/src/bgra_to_yuv444.wgsl` for the matching
/// encoder-side rationale.)
fn import_yuv444_packed(
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
        planes: YuvPlanes::Yuv444Packed { packed },
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
