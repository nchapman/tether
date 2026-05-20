//! Linux DMA-BUF import path: VAAPI-decoded NV12 surface → two
//! `wgpu::Texture`s the renderer can sample. Pure Linux module —
//! macOS will land its IOSurface equivalent here as a sibling file.

use tether_codec::{DmaBufFrame, GpuFrameGuard};

use crate::{RenderError, Result};

use super::YuvTextures;

#[allow(clippy::cast_lossless)] // u32 pitch into u64 stride is intentional
pub(crate) fn import_dmabuf_textures(
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
        label: Some("tether-render yuv bind group (dmabuf)"),
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
        y,
        uv,
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
    // layer would mean we're looking at a COMPOSED export, which we
    // explicitly didn't ask for and don't import.
    if layer.num_planes != 1 {
        return Err(RenderError::DmaBufImport(format!(
            "layer {layer_idx} has {} planes; expected 1 (SEPARATE_LAYERS)",
            layer.num_planes
        )));
    }
    let obj_idx = layer.object_index[0] as usize;
    let obj = dmabuf.objects.get(obj_idx).ok_or_else(|| {
        RenderError::DmaBufImport(format!(
            "layer {layer_idx} references object {obj_idx} but only {} present",
            dmabuf.objects.len()
        ))
    })?;
    // dup the fd because wgpu takes ownership and the same object may
    // also back another layer. `try_clone` is dup(2) with CLOEXEC.
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
                u64::from(layer.pitch[0]),
                u64::from(layer.offset[0]),
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
