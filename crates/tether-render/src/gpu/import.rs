//! Linux DMA-BUF import path: VAAPI-decoded surface → `wgpu::Texture`
//! plane(s) the renderer can sample. Dispatches on the negotiated
//! `(chroma, bit_depth)` — 8-bit 4:2:0 gets biplanar NV12 (R8 Y +
//! Rg8 UV), 8-bit 4:4:4 gets packed XYUV in a single Rgba8 texture,
//! 10-bit 4:2:0 gets biplanar P010 (R16 Y + Rg16 UV), 10-bit 4:4:4 gets
//! packed Y410 in a single Rgb10a2 texture. Pure Linux module — macOS
//! lands its IOSurface equivalent in `metal.rs`.

use tether_codec::vaapi_interop::{
    accepts_dmabuf_fourcc, dmabuf_fourcc_expected_label, YU24_FOURCC,
};
use tether_codec::{DmaBufFrame, GpuFrameGuard};
use tether_protocol::control::ChromaSubsampling;

use crate::{RenderError, Result};

use super::{YuvPlanes, YuvTextures};

#[cfg(target_os = "linux")]
use ash::vk;
#[cfg(target_os = "linux")]
use std::os::fd::IntoRawFd;

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
            if dmabuf.fourcc == YU24_FOURCC {
                import_yuv444_planar(device, layout, sampler, dmabuf, width, height, guard)
            } else {
                import_yuv444_packed(device, layout, sampler, dmabuf, width, height, guard)
            }
        }
        (ChromaSubsampling::Yuv420, 10) => {
            // P010: biplanar 16-bit cells, MSB-aligned 10-bit data,
            // half-res UV. We never sniff the DRM fourcc for layout —
            // the negotiated `chroma` is the source of truth.
            import_biplanar(
                device, layout, sampler, chroma, dmabuf, width, height, guard, true,
            )
        }
        (ChromaSubsampling::Yuv444, 10) => {
            // HEVC Main 4:4:4 10-bit. Intel's VAAPI decoder exports this
            // as a single packed Y410 layer (DRM_FORMAT_Y410, 10:10:10:2).
            // `render_layout_for` returns `PackedY410` for this cell on
            // Linux, so the bind-group layout here is the single-texture
            // packed one. (A driver that instead emits biplanar P410 or
            // XV30 would need a separate cell; not observed on Intel
            // media-driver, which is our reference for 4:4:4.)
            import_yuv444_packed_y410(device, layout, sampler, dmabuf, width, height, guard)
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
    // negotiated `chroma` is the source of truth. NOTE: the decode
    // probe's L2 gate (`tether_probe::host::vaapi`) relies on this
    // path staying ungated — it checks the surface-level `NV12`/`P010`
    // fourcc, not the per-layer `R8`/`R16`. If a per-layer fourcc gate
    // is ever added here, update `tether_codec::vaapi_interop` and that
    // probe together.
    let (chroma_w, chroma_h) = match chroma {
        ChromaSubsampling::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
        ChromaSubsampling::Yuv444 => (width, height),
    };
    let (y_fmt, uv_fmt) = if high_bit_depth {
        (
            wgpu::TextureFormat::R16Unorm,
            wgpu::TextureFormat::Rg16Unorm,
        )
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
    // Single source of truth for the accept set (shared with the decode
    // probe in tether-probe) — see `tether_codec::vaapi_interop`.
    if !accepts_dmabuf_fourcc(ChromaSubsampling::Yuv444, 8, layer.drm_format) {
        return Err(RenderError::DmaBufImport(format!(
            "YUV444 layer fourcc 0x{:08x} not render-accepted; expected {}",
            layer.drm_format,
            dmabuf_fourcc_expected_label(ChromaSubsampling::Yuv444, 8),
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

fn import_yuv444_planar(
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
            "YUV444P dma-buf has {} layers; expected 1 (planar YU24)",
            dmabuf.layers.len()
        )));
    }
    let layer = &dmabuf.layers[0];
    if layer.num_planes != 3 {
        return Err(RenderError::DmaBufImport(format!(
            "YUV444P layer should have 3 planes, got {}",
            layer.num_planes
        )));
    }
    if !accepts_dmabuf_fourcc(ChromaSubsampling::Yuv444, 8, layer.drm_format) {
        return Err(RenderError::DmaBufImport(format!(
            "YUV444P layer fourcc 0x{:08x} not render-accepted; expected {}",
            layer.drm_format,
            dmabuf_fourcc_expected_label(ChromaSubsampling::Yuv444, 8),
        )));
    }

    let y = import_one_plane(
        device,
        "tether-render y plane (dmabuf 444 planar)",
        dmabuf,
        0,
        0,
        width,
        height,
        wgpu::TextureFormat::R8Unorm,
    )?;
    let u = import_one_plane(
        device,
        "tether-render u plane (dmabuf 444 planar)",
        dmabuf,
        0,
        1,
        width,
        height,
        wgpu::TextureFormat::R8Unorm,
    )?;
    let v = import_one_plane(
        device,
        "tether-render v plane (dmabuf 444 planar)",
        dmabuf,
        0,
        2,
        width,
        height,
        wgpu::TextureFormat::R8Unorm,
    )?;
    let y_view = y.create_view(&wgpu::TextureViewDescriptor::default());
    let u_view = u.create_view(&wgpu::TextureViewDescriptor::default());
    let v_view = v.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tether-render yuv bind group (dmabuf 444 planar)"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&y_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&u_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&v_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    Ok(YuvTextures {
        planes: YuvPlanes::Yuv444Planar { y, u, v },
        bind_group,
        size: (width, height),
        _guard: Some(guard),
    })
}

/// VAAPI decodes a Main444 10-bit surface to a single packed Y410 layer:
/// one DRM object, one layer, fourcc `Y410` (DRM_FORMAT_Y410, "[31:0]
/// X:Cr:Y:Cb"), one plane (32 bpp, 10:10:10:2 little-endian). Imported as
/// a single `Rgb10a2Unorm` texture — native 10-bit, so no MSB-align — and
/// sampled by `shader_yuv444_10bit.wgsl` in DRM-spec lane order
/// (bits[9:0]=Cb→R, bits[19:10]=Y→G, bits[29:20]=Cr→B).
///
/// NOTE the decode-output fourcc (Y410) differs from the encoder-input
/// fourcc (XV30 that gpuconvert's BGRA→XV30 pass writes), but both are the
/// same "X:Cr:Y:Cb" lane order — the pack and unpack are conformant, so a
/// conformant decoder (macOS VideoToolbox) and ours agree. A driver that
/// emits a genuinely different layout (e.g. biplanar P410) needs its own
/// cell.
fn import_yuv444_packed_y410(
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
            "YUV444 10-bit dma-buf has {} layers; expected 1 (packed Y410)",
            dmabuf.layers.len()
        )));
    }
    let layer = &dmabuf.layers[0];
    if layer.num_planes != 1 {
        return Err(RenderError::DmaBufImport(format!(
            "YUV444 10-bit packed layer should have 1 plane, got {}",
            layer.num_planes
        )));
    }
    // Single source of truth for the accept set (shared with the decode
    // probe in tether-probe) — see `tether_codec::vaapi_interop`.
    if !accepts_dmabuf_fourcc(ChromaSubsampling::Yuv444, 10, layer.drm_format) {
        return Err(RenderError::DmaBufImport(format!(
            "YUV444 10-bit layer fourcc 0x{:08x} not render-accepted; expected {}",
            layer.drm_format,
            dmabuf_fourcc_expected_label(ChromaSubsampling::Yuv444, 10),
        )));
    }
    let packed = import_one_layer(
        device,
        "tether-render y410 packed (dmabuf 444 10-bit)",
        dmabuf,
        0,
        width,
        height,
        wgpu::TextureFormat::Rgb10a2Unorm,
    )?;
    let packed_view = packed.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tether-render yuv bind group (dmabuf 444 packed 10-bit)"),
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
    // matching API. `import_dmabuf_layer_texture` consumes this duped fd
    // on successful Vulkan memory import and closes it on failures before
    // ownership transfer. `texture_from_raw` (called inside) requires the
    // hal_texture be created from this device, which is exactly what we do.
    let hal_texture = unsafe {
        let hal_dev = device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or_else(|| RenderError::DmaBufImport("device is not Vulkan-backed".into()))?;
        import_dmabuf_layer_texture(
            &hal_dev,
            fd,
            &hal_desc,
            obj.size,
            obj.drm_format_modifier,
            u64::from(layer.pitch[plane_idx]),
            u64::from(layer.offset[plane_idx]),
        )?
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
    let texture = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_texture, &wgpu_desc)
    };
    Ok(texture)
}

#[cfg(target_os = "linux")]
unsafe fn import_dmabuf_layer_texture(
    hal_dev: &wgpu::hal::vulkan::Device,
    fd: std::os::fd::OwnedFd,
    desc: &wgpu::hal::TextureDescriptor<'_>,
    object_size: u64,
    drm_modifier: u64,
    stride: u64,
    offset: u64,
) -> Result<wgpu::hal::vulkan::Texture> {
    let raw_device = hal_dev.raw_device();
    let raw_instance = hal_dev.shared_instance().raw_instance();
    let raw_physical = hal_dev.raw_physical_device();
    let ext_mem_fd = ash::khr::external_memory_fd::Device::new(raw_instance, raw_device);

    let mut external_memory_image_info = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let plane_layout = vk::SubresourceLayout {
        offset: 0,
        row_pitch: stride,
        size: 0,
        array_pitch: 0,
        depth_pitch: 0,
    };
    let mut drm_modifier_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(drm_modifier)
        .plane_layouts(std::slice::from_ref(&plane_layout));
    let vk_format = wgpu_format_to_vk(desc.format)?;
    let vk_usage = texture_uses_to_vk(desc.usage);
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk_format)
        .extent(vk::Extent3D {
            width: desc.size.width,
            height: desc.size.height,
            depth: desc.size.depth_or_array_layers,
        })
        .mip_levels(desc.mip_level_count)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(vk_usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external_memory_image_info)
        .push_next(&mut drm_modifier_info);

    let image = unsafe { raw_device.create_image(&image_info, None) }
        .map_err(|e| RenderError::DmaBufImport(format!("vkCreateImage dmabuf import: {e:?}")))?;
    let requirements = unsafe { raw_device.get_image_memory_requirements(image) };
    let fd_raw = fd.into_raw_fd();
    let imported = unsafe {
        import_dmabuf_memory(
            raw_instance,
            raw_physical,
            raw_device,
            &ext_mem_fd,
            fd_raw,
            object_size,
            requirements.memory_type_bits,
        )
    };
    let memory = match imported {
        Ok(memory) => memory,
        Err(e) => {
            unsafe { raw_device.destroy_image(image, None) };
            return Err(e);
        }
    };

    // The exporter created each plane as its own VkImage bound into a shared
    // allocation at this DMA-BUF offset. Recreate that shape on import:
    // single-plane image layout starts at 0, image memory binding starts at
    // the reported plane offset.
    if let Err(e) = unsafe { raw_device.bind_image_memory(image, memory, offset) } {
        unsafe {
            raw_device.free_memory(memory, None);
            raw_device.destroy_image(image, None);
        }
        return Err(RenderError::DmaBufImport(format!(
            "vkBindImageMemory dmabuf import: {e:?}"
        )));
    }

    Ok(unsafe {
        hal_dev.texture_from_raw(
            image,
            desc,
            None,
            wgpu::hal::vulkan::TextureMemory::Dedicated(memory),
        )
    })
}

#[cfg(target_os = "linux")]
unsafe fn import_dmabuf_memory(
    raw_instance: &ash::Instance,
    raw_physical: vk::PhysicalDevice,
    raw_device: &ash::Device,
    ext_mem_fd: &ash::khr::external_memory_fd::Device,
    fd_raw: i32,
    object_size: u64,
    memory_type_bits: u32,
) -> Result<vk::DeviceMemory> {
    let mut fd_props = vk::MemoryFdPropertiesKHR::default();
    if let Err(e) = unsafe {
        ext_mem_fd.get_memory_fd_properties(
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            fd_raw,
            &mut fd_props,
        )
    } {
        unsafe { libc::close(fd_raw) };
        return Err(RenderError::DmaBufImport(format!(
            "vkGetMemoryFdPropertiesKHR: {e:?}"
        )));
    }

    let mem_props = unsafe { raw_instance.get_physical_device_memory_properties(raw_physical) };
    let mem_type_index = find_memory_type(&mem_props, memory_type_bits & fd_props.memory_type_bits)
        .ok_or_else(|| {
            unsafe { libc::close(fd_raw) };
            RenderError::DmaBufImport("no memory type for imported dma-buf".into())
        })?;
    let mut import_memory_info = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        .fd(fd_raw);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(object_size)
        .memory_type_index(mem_type_index)
        .push_next(&mut import_memory_info);

    match unsafe { raw_device.allocate_memory(&alloc_info, None) } {
        Ok(memory) => Ok(memory),
        Err(e) => {
            unsafe { libc::close(fd_raw) };
            Err(RenderError::DmaBufImport(format!(
                "vkAllocateMemory dmabuf import: {e:?}"
            )))
        }
    }
}

#[cfg(target_os = "linux")]
fn find_memory_type(mem_props: &vk::PhysicalDeviceMemoryProperties, type_bits: u32) -> Option<u32> {
    for (i, _mem_ty) in mem_props.memory_types_as_slice().iter().enumerate() {
        let bit = 1u32 << i;
        if type_bits & bit != 0 {
            #[allow(clippy::cast_possible_truncation)]
            return Some(i as u32);
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn wgpu_format_to_vk(format: wgpu::TextureFormat) -> Result<vk::Format> {
    match format {
        wgpu::TextureFormat::R8Unorm => Ok(vk::Format::R8_UNORM),
        wgpu::TextureFormat::Rg8Unorm => Ok(vk::Format::R8G8_UNORM),
        wgpu::TextureFormat::R16Unorm => Ok(vk::Format::R16_UNORM),
        wgpu::TextureFormat::Rg16Unorm => Ok(vk::Format::R16G16_UNORM),
        wgpu::TextureFormat::Rgba8Unorm => Ok(vk::Format::R8G8B8A8_UNORM),
        wgpu::TextureFormat::Rgb10a2Unorm => Ok(vk::Format::A2B10G10R10_UNORM_PACK32),
        _ => Err(RenderError::DmaBufImport(format!(
            "unsupported dma-buf import texture format {format:?}"
        ))),
    }
}

#[cfg(target_os = "linux")]
fn texture_uses_to_vk(uses: wgpu::TextureUses) -> vk::ImageUsageFlags {
    let mut flags = vk::ImageUsageFlags::empty();
    if uses.contains(wgpu::TextureUses::RESOURCE) {
        flags |= vk::ImageUsageFlags::SAMPLED;
    }
    if uses.contains(wgpu::TextureUses::COPY_SRC) {
        flags |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if uses.contains(wgpu::TextureUses::COPY_DST) {
        flags |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    if uses.contains(wgpu::TextureUses::STORAGE_READ_WRITE) {
        flags |= vk::ImageUsageFlags::STORAGE;
    }
    flags
}
