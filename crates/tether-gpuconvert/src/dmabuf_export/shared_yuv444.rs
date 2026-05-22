//! Packed YUV 4:4:4 (DRM_FORMAT_XYUV8888) backed by a single
//! `VkDeviceMemory`, exported as one DMA-BUF.
//!
//! Two reasons this is single-plane rather than three (despite the
//! 4:4:4 label):
//!
//! 1. ffmpeg 8.x's `vaapi_drm_format_map` has no entry for planar
//!    YUV444P over DRM_PRIME. `av_hwframe_map(DRM_PRIME → VAAPI)`
//!    rejects any descriptor whose layer format is `YU24` (the
//!    aggregate fourcc) or whose layers are three separate `R8`
//!    planes. The only 4:4:4 8-bit format ffmpeg can import via
//!    DRM_PRIME is `DRM_FORMAT_XYUV8888` packed → `VA_FOURCC_XYUV`
//!    (`AV_PIX_FMT_VUYX`). See `MAP(XYUV, YUV444, VUYX, 0)` in
//!    `libavutil/hwcontext_vaapi.c`.
//!
//! 2. The packed layout is structurally simpler — one VkImage, one
//!    bind, no plane-offset walking. The 3-plane shared-memory code
//!    this replaces was correct but unused (the encoder couldn't
//!    consume what it produced).
//!
//! Per-byte layout from `<drm/drm_fourcc.h>`:
//!   DRM_FORMAT_XYUV8888 = [31:0] X:Y:U:V 8:8:8:8 little endian
//!   in-memory byte order: V, U, Y, X
//!
//! Maps to `vk::Format::R8G8B8A8_UNORM` on the Vulkan side; the
//! compute shader writes `vec4<f32>(V, U, Y, 1.0)` into an
//! `Rgba8Unorm` storage texture which lands as the above bytes.

use std::os::fd::{FromRawFd, OwnedFd};

use ash::{ext, khr, vk};

use super::{
    find_memory_type, wgpu_usage_to_hal_uses, wgpu_usage_to_vk, ExportError, Result,
    DRM_FORMAT_MOD_LINEAR,
};

/// One packed XYUV surface backed by a single dma-buf.
///
/// Kept named `SharedYuv444Export` for callers' sake even though the
/// underlying memory is one plane — the user-facing chroma identity
/// is still 4:4:4. The dma-buf descriptor produced from this carries
/// `DRM_FORMAT_XYUV8888` as the layer fourcc; VAAPI's importer maps
/// that to `VA_FOURCC_XYUV` and the encoder takes it as a Main 4:4:4
/// input source.
pub struct SharedYuv444Export {
    pub packed_texture: wgpu::Texture,
    pub fd: OwnedFd,
    pub size: u64,
    pub modifier: u64,
    pub offset: u64,
    pub pitch: u64,
}

pub fn export_yuv444_shared_dmabuf(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsages,
) -> Result<SharedYuv444Export> {
    if !device
        .features()
        .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
    {
        return Err(ExportError::FeatureUnsupported);
    }
    // Same Intel iHD VAAPI 64-byte row-pitch alignment requirement as
    // the NV12 sibling. XYUV is 4 bytes/luma-pixel, so 64-byte rows
    // mean luma-pixel width aligned to 16. Common widths (1920, 2160,
    // 2560) are already 16-aligned; this defends against the rarer
    // non-16-aligned widths (e.g. 2180) that would otherwise produce
    // the same left-edge row-aliasing corruption.
    const VAAPI_LUMA_STRIDE_ALIGN_XYUV: u32 = 16;
    let aligned_w = width.next_multiple_of(VAAPI_LUMA_STRIDE_ALIGN_XYUV);
    let vk_usage = wgpu_usage_to_vk(usage);

    // SAFETY: hal escape hatch — Vulkan backend verified below; raw
    // handles valid for the lifetime of the hal::Device borrow.
    unsafe {
        let hal_dev = device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or(ExportError::NotVulkan)?;
        let raw_device = hal_dev.raw_device();
        let raw_instance = hal_dev.shared_instance().raw_instance();
        let raw_physical = hal_dev.raw_physical_device();

        let ext_mem_fd = khr::external_memory_fd::Device::new(raw_instance, raw_device);
        let modifier_ext =
            ext::image_drm_format_modifier::Device::new(raw_instance, raw_device);

        let modifiers = [DRM_FORMAT_MOD_LINEAR];
        let mut ext_mem_create = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut modifier_info = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
            .drm_format_modifiers(&modifiers);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D { width: aligned_w, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut ext_mem_create)
            .push_next(&mut modifier_info);
        let image = raw_device
            .create_image(&info, None)
            .map_err(|e| ExportError::Vk(e, "vkCreateImage (XYUV)"))?;

        let result = (|| -> Result<SharedYuv444Export> {
            let mem_req = raw_device.get_image_memory_requirements(image);
            let mem_props = raw_instance.get_physical_device_memory_properties(raw_physical);
            let mem_type_index = find_memory_type(
                &mem_props,
                mem_req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .or_else(|| {
                find_memory_type(
                    &mem_props,
                    mem_req.memory_type_bits,
                    vk::MemoryPropertyFlags::empty(),
                )
            })
            .ok_or(ExportError::NoMemoryType)?;

            let mut export_alloc_info = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_req.size)
                .memory_type_index(mem_type_index)
                .push_next(&mut export_alloc_info);
            let memory = raw_device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| ExportError::Vk(e, "vkAllocateMemory (XYUV)"))?;

            let bind_and_export = (|| -> Result<SharedYuv444Export> {
                raw_device
                    .bind_image_memory(image, memory, 0)
                    .map_err(|e| ExportError::Vk(e, "vkBindImageMemory (XYUV)"))?;

                let fd_info = vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                let raw_fd = ext_mem_fd
                    .get_memory_fd(&fd_info)
                    .map_err(|e| ExportError::Vk(e, "vkGetMemoryFdKHR (XYUV)"))?;
                // SAFETY: raw_fd is a freshly-returned open fd we own.
                let fd = OwnedFd::from_raw_fd(raw_fd);

                let plane0_subres = vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT)
                    .mip_level(0)
                    .array_layer(0);
                let layout = raw_device.get_image_subresource_layout(image, plane0_subres);

                let mut mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
                modifier_ext
                    .get_image_drm_format_modifier_properties(image, &mut mod_props)
                    .map_err(|e| {
                        ExportError::Vk(e, "vkGetImageDrmFormatModifierPropertiesEXT (XYUV)")
                    })?;

                let device_for_drop = raw_device.clone();
                let memory_for_drop = memory;
                let drop_cb: wgpu::hal::DropCallback = Box::new(move || {
                    // SAFETY (extends the outer `unsafe` block): image
                    // was created on `device_for_drop`; no one else
                    // holds this vk::Image handle; the memory is
                    // exclusively bound to this image (no shared
                    // allocation here, unlike the NV12 sibling).
                    device_for_drop.destroy_image(image, None);
                    device_for_drop.free_memory(memory_for_drop, None);
                });

                let hal_desc = wgpu::hal::TextureDescriptor {
                    label: Some("yuv444-xyuv shared"),
                    size: wgpu::Extent3d { width: aligned_w, height, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu_usage_to_hal_uses(usage),
                    memory_flags: wgpu::hal::MemoryFlags::empty(),
                    view_formats: vec![],
                };

                // SAFETY: device is Vulkan-backed (verified above);
                // `image` was created from this device's raw_device
                // and bound to `memory` at offset 0; drop_cb takes
                // ownership of both for teardown.
                let hal_tex = hal_dev.texture_from_raw(
                    image,
                    &hal_desc,
                    Some(drop_cb),
                    wgpu::hal::vulkan::TextureMemory::External,
                );

                let wgpu_desc = wgpu::TextureDescriptor {
                    label: Some("yuv444-xyuv shared"),
                    size: wgpu::Extent3d { width: aligned_w, height, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage,
                    view_formats: &[],
                };
                let packed_texture = device
                    .create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_tex, &wgpu_desc);

                Ok(SharedYuv444Export {
                    packed_texture,
                    fd,
                    size: mem_req.size,
                    modifier: mod_props.drm_format_modifier,
                    offset: layout.offset,
                    pitch: layout.row_pitch,
                })
            })();

            match bind_and_export {
                Ok(export) => Ok(export),
                Err(e) => {
                    // Memory/image still owned by us — the drop_cb
                    // hadn't been registered yet.
                    raw_device.free_memory(memory, None);
                    Err(e)
                }
            }
        })();

        match result {
            Ok(export) => Ok(export),
            Err(e) => {
                raw_device.destroy_image(image, None);
                Err(e)
            }
        }
    }
}
