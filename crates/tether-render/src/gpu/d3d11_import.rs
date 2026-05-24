//! Windows D3D11 texture import into wgpu via Vulkan external memory.
//!
//! The decoder exports an NV12 texture as a shared DXGI handle.
//! We import it into wgpu's Vulkan backend via
//! `VK_KHR_external_memory_win32` (`texture_from_d3d11_shared_handle`),
//! then create R8 (Y plane) and Rg8 (UV plane) views for the YUV
//! shader.

use tether_codec::GpuFrameGuard;
use tether_protocol::control::ChromaSubsampling;

use crate::{RenderError, Result};
use super::YuvTextures;

/// Import a D3D11 shared-handle NV12 texture into wgpu and build the
/// YUV plane textures + bind group for rendering.
pub(crate) fn import_d3d11_textures(
    device: &wgpu::Device,
    yuv_bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    chroma: ChromaSubsampling,
    bit_depth: u8,
    shared_handle: *mut std::ffi::c_void,
    width: u32,
    height: u32,
    guard: GpuFrameGuard,
) -> Result<YuvTextures> {
    if shared_handle.is_null() {
        return Err(RenderError::DmaBufImport("null D3D11 shared handle".into()));
    }

    let chroma_w = match chroma {
        ChromaSubsampling::Yuv420 => (width + 1) / 2,
        ChromaSubsampling::Yuv444 => width,
    };
    let chroma_h = match chroma {
        ChromaSubsampling::Yuv420 => (height + 1) / 2,
        ChromaSubsampling::Yuv444 => height,
    };

    let (y_format, uv_format) = match bit_depth {
        10 => (wgpu::TextureFormat::R16Unorm, wgpu::TextureFormat::Rg16Unorm),
        _ => (wgpu::TextureFormat::R8Unorm, wgpu::TextureFormat::Rg8Unorm),
    };

    let y_texture = import_plane(
        device,
        shared_handle,
        width,
        height,
        y_format,
        "d3d11_y",
    )?;

    let uv_texture = import_plane(
        device,
        shared_handle,
        chroma_w,
        chroma_h,
        uv_format,
        "d3d11_uv",
    )?;

    let y_view = y_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let uv_view = uv_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("d3d11_yuv_bg"),
        layout: yuv_bgl,
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

    let planes = match bit_depth {
        10 => super::YuvPlanes::Biplanar16 { y: y_texture, uv: uv_texture },
        _ => super::YuvPlanes::Biplanar8 { y: y_texture, uv: uv_texture },
    };
    Ok(YuvTextures {
        planes,
        bind_group,
        size: (width, height),
        _guard: Some(guard),
    })
}

fn import_plane(
    device: &wgpu::Device,
    shared_handle: *mut std::ffi::c_void,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> Result<wgpu::Texture> {
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

    let hal_texture = unsafe {
        let hal_dev = device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or_else(|| {
                RenderError::DmaBufImport("device is not Vulkan-backed".into())
            })?;
        // HANDLE is repr(transparent) over *mut c_void in the windows
        // crate. Transmute to match wgpu's pinned version.
        hal_dev
            .texture_from_d3d11_shared_handle(
                std::mem::transmute(shared_handle),
                &hal_desc,
            )
            .map_err(|e| {
                RenderError::DmaBufImport(format!(
                    "texture_from_d3d11_shared_handle: {e:?}"
                ))
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

    let texture = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_texture, &wgpu_desc)
    };
    Ok(texture)
}
