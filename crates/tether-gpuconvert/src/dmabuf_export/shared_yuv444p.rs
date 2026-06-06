//! Planar YUV444P (`DRM_FORMAT_YUV444` / `YU24`) backed by one shared
//! `VkDeviceMemory`, exported as a single DMA-BUF.
//!
//! This is the NVIDIA NVENC input shape for HEVC Main 4:4:4 8-bit:
//! three full-resolution R8 planes (Y, U, V) at distinct offsets in one
//! allocation. The VAAPI path uses the packed sibling in `shared_yuv444.rs`;
//! keep the two shapes separate because their consumers accept different
//! DRM formats.

use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;

use ash::{ext, khr, vk};

use super::{
    align_up, find_memory_type, wgpu_usage_to_hal_uses, wgpu_usage_to_vk, ExportError, Result,
    DRM_FORMAT_MOD_LINEAR,
};

pub struct SharedYuv444pExport {
    pub y_texture: wgpu::Texture,
    pub u_texture: wgpu::Texture,
    pub v_texture: wgpu::Texture,
    pub fd: OwnedFd,
    pub size: u64,
    pub modifier: u64,
    pub y_offset: u64,
    pub y_pitch: u64,
    pub u_offset: u64,
    pub u_pitch: u64,
    pub v_offset: u64,
    pub v_pitch: u64,
}

pub fn export_yuv444p_shared_dmabuf(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsages,
) -> Result<SharedYuv444pExport> {
    if !device
        .features()
        .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
    {
        return Err(ExportError::FeatureUnsupported);
    }

    const LUMA_STRIDE_ALIGN: u32 = 64;
    const HEIGHT_ALIGN: u32 = 16;
    let aligned_w = width.next_multiple_of(LUMA_STRIDE_ALIGN);
    let aligned_h = height.next_multiple_of(HEIGHT_ALIGN);
    let vk_usage = wgpu_usage_to_vk(usage);

    unsafe {
        let hal_dev = device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or(ExportError::NotVulkan)?;
        let raw_device = hal_dev.raw_device();
        let raw_instance = hal_dev.shared_instance().raw_instance();
        let raw_physical = hal_dev.raw_physical_device();

        let ext_mem_fd = khr::external_memory_fd::Device::new(raw_instance, raw_device);
        let modifier_ext = ext::image_drm_format_modifier::Device::new(raw_instance, raw_device);

        let modifiers = [DRM_FORMAT_MOD_LINEAR];
        let create_plane = |label: &'static str| -> Result<vk::Image> {
            let mut ext_mem_create = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let mut modifier_info = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
                .drm_format_modifiers(&modifiers);
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8_UNORM)
                .extent(vk::Extent3D {
                    width: aligned_w,
                    height: aligned_h,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(vk_usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut ext_mem_create)
                .push_next(&mut modifier_info);
            raw_device
                .create_image(&info, None)
                .map_err(|e| ExportError::Vk(e, label))
        };

        let y_image = create_plane("vkCreateImage (YUV444P Y)")?;
        let u_image = match create_plane("vkCreateImage (YUV444P U)") {
            Ok(i) => i,
            Err(e) => {
                raw_device.destroy_image(y_image, None);
                return Err(e);
            }
        };
        let v_image = match create_plane("vkCreateImage (YUV444P V)") {
            Ok(i) => i,
            Err(e) => {
                raw_device.destroy_image(u_image, None);
                raw_device.destroy_image(y_image, None);
                return Err(e);
            }
        };

        let result = (|| -> Result<SharedYuv444pExport> {
            let y_req = raw_device.get_image_memory_requirements(y_image);
            let u_req = raw_device.get_image_memory_requirements(u_image);
            let v_req = raw_device.get_image_memory_requirements(v_image);

            let combined_type_bits =
                y_req.memory_type_bits & u_req.memory_type_bits & v_req.memory_type_bits;
            if combined_type_bits == 0 {
                return Err(ExportError::NoMemoryType);
            }
            let mem_props = raw_instance.get_physical_device_memory_properties(raw_physical);
            let mem_type_index = find_memory_type(
                &mem_props,
                combined_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .or_else(|| {
                find_memory_type(
                    &mem_props,
                    combined_type_bits,
                    vk::MemoryPropertyFlags::empty(),
                )
            })
            .ok_or(ExportError::NoMemoryType)?;

            let u_align = y_req.alignment.max(u_req.alignment);
            let v_align = u_align.max(v_req.alignment);
            let u_bind_offset = align_up(y_req.size, u_align);
            let v_bind_offset = align_up(u_bind_offset + u_req.size, v_align);
            let total_size = v_bind_offset + v_req.size;

            let mut export_alloc_info = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(total_size)
                .memory_type_index(mem_type_index)
                .push_next(&mut export_alloc_info);
            let memory = raw_device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| ExportError::Vk(e, "vkAllocateMemory (YUV444P shared)"))?;

            let bind_and_export = (|| -> Result<SharedYuv444pExport> {
                raw_device
                    .bind_image_memory(y_image, memory, 0)
                    .map_err(|e| ExportError::Vk(e, "vkBindImageMemory (YUV444P Y)"))?;
                raw_device
                    .bind_image_memory(u_image, memory, u_bind_offset)
                    .map_err(|e| ExportError::Vk(e, "vkBindImageMemory (YUV444P U)"))?;
                raw_device
                    .bind_image_memory(v_image, memory, v_bind_offset)
                    .map_err(|e| ExportError::Vk(e, "vkBindImageMemory (YUV444P V)"))?;

                let fd_info = vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                let raw_fd = ext_mem_fd
                    .get_memory_fd(&fd_info)
                    .map_err(|e| ExportError::Vk(e, "vkGetMemoryFdKHR (YUV444P shared)"))?;
                let fd = OwnedFd::from_raw_fd(raw_fd);

                let mut y_mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
                let mut u_mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
                let mut v_mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
                modifier_ext
                    .get_image_drm_format_modifier_properties(y_image, &mut y_mod_props)
                    .map_err(|e| {
                        ExportError::Vk(e, "vkGetImageDrmFormatModifierPropertiesEXT (YUV444P Y)")
                    })?;
                modifier_ext
                    .get_image_drm_format_modifier_properties(u_image, &mut u_mod_props)
                    .map_err(|e| {
                        ExportError::Vk(e, "vkGetImageDrmFormatModifierPropertiesEXT (YUV444P U)")
                    })?;
                modifier_ext
                    .get_image_drm_format_modifier_properties(v_image, &mut v_mod_props)
                    .map_err(|e| {
                        ExportError::Vk(e, "vkGetImageDrmFormatModifierPropertiesEXT (YUV444P V)")
                    })?;
                debug_assert_eq!(y_mod_props.drm_format_modifier, DRM_FORMAT_MOD_LINEAR);
                debug_assert_eq!(u_mod_props.drm_format_modifier, DRM_FORMAT_MOD_LINEAR);
                debug_assert_eq!(v_mod_props.drm_format_modifier, DRM_FORMAT_MOD_LINEAR);

                let plane0_subres = vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT)
                    .mip_level(0)
                    .array_layer(0);
                let y_layout = raw_device.get_image_subresource_layout(y_image, plane0_subres);
                let u_layout = raw_device.get_image_subresource_layout(u_image, plane0_subres);
                let v_layout = raw_device.get_image_subresource_layout(v_image, plane0_subres);
                debug_assert_eq!(y_layout.offset, 0);
                debug_assert_eq!(u_layout.offset, 0);
                debug_assert_eq!(v_layout.offset, 0);

                let mem_arc = Arc::new(SharedDeviceMemory {
                    device: raw_device.clone(),
                    memory,
                });
                let y_texture = import_shared_image_into_wgpu(
                    device,
                    raw_device,
                    y_image,
                    aligned_w,
                    aligned_h,
                    usage,
                    "yuv444p-shared y",
                    mem_arc.clone(),
                );
                let u_texture = import_shared_image_into_wgpu(
                    device,
                    raw_device,
                    u_image,
                    aligned_w,
                    aligned_h,
                    usage,
                    "yuv444p-shared u",
                    mem_arc.clone(),
                );
                let v_texture = import_shared_image_into_wgpu(
                    device,
                    raw_device,
                    v_image,
                    aligned_w,
                    aligned_h,
                    usage,
                    "yuv444p-shared v",
                    mem_arc,
                );

                Ok(SharedYuv444pExport {
                    y_texture,
                    u_texture,
                    v_texture,
                    fd,
                    size: total_size,
                    modifier: y_mod_props.drm_format_modifier,
                    y_offset: y_layout.offset,
                    y_pitch: y_layout.row_pitch,
                    u_offset: u_bind_offset + u_layout.offset,
                    u_pitch: u_layout.row_pitch,
                    v_offset: v_bind_offset + v_layout.offset,
                    v_pitch: v_layout.row_pitch,
                })
            })();

            match bind_and_export {
                Ok(export) => Ok(export),
                Err(e) => {
                    raw_device.free_memory(memory, None);
                    Err(e)
                }
            }
        })();

        match result {
            Ok(export) => Ok(export),
            Err(e) => {
                raw_device.destroy_image(v_image, None);
                raw_device.destroy_image(u_image, None);
                raw_device.destroy_image(y_image, None);
                Err(e)
            }
        }
    }
}

struct SharedDeviceMemory {
    device: ash::Device,
    memory: vk::DeviceMemory,
}

impl Drop for SharedDeviceMemory {
    fn drop(&mut self) {
        unsafe { self.device.free_memory(self.memory, None) };
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn import_shared_image_into_wgpu(
    device: &wgpu::Device,
    raw_device: &ash::Device,
    image: vk::Image,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsages,
    label: &'static str,
    mem_arc: Arc<SharedDeviceMemory>,
) -> wgpu::Texture {
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
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu_usage_to_hal_uses(usage),
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };

    let device_for_drop = raw_device.clone();
    let drop_cb: wgpu::hal::DropCallback = Box::new(move || {
        unsafe { device_for_drop.destroy_image(image, None) };
        drop(mem_arc);
    });

    unsafe {
        let hal_dev = device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .expect("device must still be Vulkan-backed");
        let hal_tex = hal_dev.texture_from_raw(
            image,
            &hal_desc,
            Some(drop_cb),
            wgpu::hal::vulkan::TextureMemory::External,
        );

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
            format: wgpu::TextureFormat::R8Unorm,
            usage,
            view_formats: &[],
        };
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_tex, &wgpu_desc)
    }
}
