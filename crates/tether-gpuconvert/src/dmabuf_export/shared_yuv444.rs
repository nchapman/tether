//! YUV 4:4:4 planar (Y + U + V) backed by one shared `VkDeviceMemory`,
//! exported as a single DMA-BUF.
//!
//! Mirrors [`super::shared_nv12::SharedNv12Export`] but with three
//! full-resolution R8 planes instead of NV12's R8 + Rg8. This is the
//! shape `av_hwframe_map(DRM_PRIME → VAAPI Main444)` requires: one
//! DRM object with three offsets, plane format `R8` per plane, layer
//! fourcc `DRM_FORMAT_YUV444` (`'YU24'`).

use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;

use ash::{ext, khr, vk};

use super::{
    align_up, find_memory_type, wgpu_usage_to_hal_uses, wgpu_usage_to_vk, ExportError, Result,
    DRM_FORMAT_MOD_LINEAR,
};

/// One YUV 4:4:4 surface backed by a single shared DMA-BUF.
///
/// Layout:
/// ```text
///   [Y (R8, w × h)] [pad] [U (R8, w × h)] [pad] [V (R8, w × h)]
///   ^offset 0             ^u_offset             ^v_offset
/// ```
/// All three planes are full resolution (no chroma subsampling — that's
/// the whole point of 4:4:4). Each plane is a real `wgpu::Texture` the
/// compute pass can bind as an R8 storage image; the underlying memory
/// is one `VkDeviceMemory` exported as one dma-buf fd.
pub struct SharedYuv444Export {
    pub y_texture: wgpu::Texture,
    pub u_texture: wgpu::Texture,
    pub v_texture: wgpu::Texture,
    pub fd: OwnedFd,
    pub size: u64,
    /// LINEAR. See the comment on [`super::shared_nv12::SharedNv12Export`]
    /// for why we don't try tiled modifiers here.
    pub modifier: u64,
    pub y_offset: u64,
    pub y_pitch: u64,
    pub u_offset: u64,
    pub u_pitch: u64,
    pub v_offset: u64,
    pub v_pitch: u64,
}

/// Allocate Y, U, V plane images in one shared `VkDeviceMemory` and
/// export the whole thing as a single dma-buf fd. Same structural
/// approach as [`super::shared_nv12::export_nv12_shared_dmabuf`]:
/// memory-type intersection across all plane reqs, place each plane at
/// the strictest required alignment, single allocation.
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
        let create_image_with = |format: vk::Format| -> Result<vk::Image> {
            let mut ext_mem_create = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let mut modifier_info = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
                .drm_format_modifiers(&modifiers);
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width, height, depth: 1 })
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
                .map_err(|e| ExportError::Vk(e, "vkCreateImage (YUV444 shared)"))
        };

        let y_image = create_image_with(vk::Format::R8_UNORM)?;
        let u_image = match create_image_with(vk::Format::R8_UNORM) {
            Ok(i) => i,
            Err(e) => {
                raw_device.destroy_image(y_image, None);
                return Err(e);
            }
        };
        let v_image = match create_image_with(vk::Format::R8_UNORM) {
            Ok(i) => i,
            Err(e) => {
                raw_device.destroy_image(u_image, None);
                raw_device.destroy_image(y_image, None);
                return Err(e);
            }
        };

        // From here on, any error must destroy all three images (and
        // free memory if allocated). Manual goto-fail.
        let result = (|| -> Result<SharedYuv444Export> {
            let y_req = raw_device.get_image_memory_requirements(y_image);
            let u_req = raw_device.get_image_memory_requirements(u_image);
            let v_req = raw_device.get_image_memory_requirements(v_image);

            // Memory-type intersection across all three planes. In
            // practice all three are R8_UNORM with the same usage and
            // tiling, so the bitsets are identical on every driver
            // we've tested — but the intersection is the principled
            // computation.
            let combined_type_bits =
                y_req.memory_type_bits & u_req.memory_type_bits & v_req.memory_type_bits;
            if combined_type_bits == 0 {
                raw_device.destroy_image(v_image, None);
                raw_device.destroy_image(u_image, None);
                raw_device.destroy_image(y_image, None);
                return Err(ExportError::NoMemoryType);
            }
            let mem_props = raw_instance.get_physical_device_memory_properties(raw_physical);
            let mem_type_index = find_memory_type(
                &mem_props,
                combined_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .or_else(|| {
                find_memory_type(&mem_props, combined_type_bits, vk::MemoryPropertyFlags::empty())
            })
            .ok_or(ExportError::NoMemoryType)?;

            // Walking offsets. Each plane's bind offset must satisfy
            // its alignment AND span the preceding plane(s). Use the
            // strictest alignment seen so the same offset works for
            // any plane.
            let strict_align = y_req.alignment.max(u_req.alignment).max(v_req.alignment);
            let u_bind_offset = align_up(y_req.size, strict_align);
            let v_bind_offset = align_up(u_bind_offset + u_req.size, strict_align);
            let total_size = v_bind_offset + v_req.size;

            let mut export_alloc_info = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(total_size)
                .memory_type_index(mem_type_index)
                .push_next(&mut export_alloc_info);
            let memory = raw_device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| ExportError::Vk(e, "vkAllocateMemory (YUV444 shared)"))?;

            let bind_and_export = (|| -> Result<SharedYuv444Export> {
                raw_device
                    .bind_image_memory(y_image, memory, 0)
                    .map_err(|e| ExportError::Vk(e, "vkBindImageMemory (Y)"))?;
                raw_device
                    .bind_image_memory(u_image, memory, u_bind_offset)
                    .map_err(|e| ExportError::Vk(e, "vkBindImageMemory (U)"))?;
                raw_device
                    .bind_image_memory(v_image, memory, v_bind_offset)
                    .map_err(|e| ExportError::Vk(e, "vkBindImageMemory (V)"))?;

                let fd_info = vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                let raw_fd = ext_mem_fd
                    .get_memory_fd(&fd_info)
                    .map_err(|e| ExportError::Vk(e, "vkGetMemoryFdKHR (YUV444 shared)"))?;
                // SAFETY: raw_fd is a freshly-returned open fd we own.
                let fd = OwnedFd::from_raw_fd(raw_fd);

                let plane0_subres = vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT)
                    .mip_level(0)
                    .array_layer(0);
                let y_layout = raw_device.get_image_subresource_layout(y_image, plane0_subres);
                let u_layout = raw_device.get_image_subresource_layout(u_image, plane0_subres);
                let v_layout = raw_device.get_image_subresource_layout(v_image, plane0_subres);
                // Y is bound at VkDeviceMemory offset 0; the spec doesn't
                // guarantee the subresource layout's offset matches, but
                // every LINEAR R8 image we've tested returns 0. Fail
                // loudly in debug if a future driver surprises us — a
                // silent non-zero would make Y advertise the wrong
                // start address and corrupt every frame.
                debug_assert_eq!(
                    y_layout.offset, 0,
                    "LINEAR R8 image bound at offset 0 reported non-zero subresource offset"
                );

                // Modifier must agree across all three planes — they
                // share the same VkDeviceMemory and a divergent modifier
                // on any plane would make VAAPI's importer reject the
                // surface. Three R8 LINEAR images with identical usage
                // always agree on every driver we test, but verify the
                // assumption.
                let mut y_mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
                modifier_ext
                    .get_image_drm_format_modifier_properties(y_image, &mut y_mod_props)
                    .map_err(|e| {
                        ExportError::Vk(e, "vkGetImageDrmFormatModifierPropertiesEXT (Y)")
                    })?;
                let mut u_mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
                modifier_ext
                    .get_image_drm_format_modifier_properties(u_image, &mut u_mod_props)
                    .map_err(|e| {
                        ExportError::Vk(e, "vkGetImageDrmFormatModifierPropertiesEXT (U)")
                    })?;
                let mut v_mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
                modifier_ext
                    .get_image_drm_format_modifier_properties(v_image, &mut v_mod_props)
                    .map_err(|e| {
                        ExportError::Vk(e, "vkGetImageDrmFormatModifierPropertiesEXT (V)")
                    })?;
                assert_eq!(
                    y_mod_props.drm_format_modifier,
                    u_mod_props.drm_format_modifier,
                    "Y and U planes disagree on DRM modifier — driver bug?"
                );
                assert_eq!(
                    y_mod_props.drm_format_modifier,
                    v_mod_props.drm_format_modifier,
                    "Y and V planes disagree on DRM modifier — driver bug?"
                );

                // Lifetime invariant from this point on: once `mem_arc`
                // is created below and handed to import_shared_image_into_wgpu,
                // the texture's drop_callback owns the image teardown.
                // The outer error handler must NOT destroy the same
                // image — a future fallible step inserted between
                // mem_arc creation and the texture imports would need
                // to track which images have been transferred and only
                // destroy the rest. Today the imports are infallible,
                // so this invariant holds trivially.

                let mem_arc = Arc::new(SharedDeviceMemory {
                    device: raw_device.clone(),
                    memory,
                });

                let y_texture = import_shared_image_into_wgpu(
                    device,
                    raw_device,
                    y_image,
                    width,
                    height,
                    usage,
                    "yuv444-shared y",
                    mem_arc.clone(),
                );
                let u_texture = import_shared_image_into_wgpu(
                    device,
                    raw_device,
                    u_image,
                    width,
                    height,
                    usage,
                    "yuv444-shared u",
                    mem_arc.clone(),
                );
                let v_texture = import_shared_image_into_wgpu(
                    device,
                    raw_device,
                    v_image,
                    width,
                    height,
                    usage,
                    "yuv444-shared v",
                    mem_arc,
                );

                Ok(SharedYuv444Export {
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
                    raw_device.destroy_image(v_image, None);
                    raw_device.destroy_image(u_image, None);
                    raw_device.destroy_image(y_image, None);
                    Err(e)
                }
            }
        })();

        result
    }
}

/// Shared `VkDeviceMemory` ownership for the three-image YUV444 export.
/// Same pattern as the NV12 sibling: every plane texture holds an Arc
/// clone via its hal drop_callback; the last clone to drop frees the
/// memory.
struct SharedDeviceMemory {
    device: ash::Device,
    memory: vk::DeviceMemory,
}

impl Drop for SharedDeviceMemory {
    fn drop(&mut self) {
        // SAFETY: memory was allocated from `device`; we own it
        // exclusively here (last Arc) and won't touch it again.
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
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
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
        // SAFETY: image was created on `device_for_drop`; no one else
        // holds this vk::Image handle; backing memory stays alive via
        // mem_arc until after we destroy the image.
        unsafe { device_for_drop.destroy_image(image, None) };
        drop(mem_arc);
    });

    // SAFETY: caller asserts device is Vulkan-backed; `image` was
    // created from this device's raw_device and bound to mem_arc's
    // memory at the correct offset; drop_cb takes ownership of both.
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
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
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
