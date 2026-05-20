//! Single-plane wgpu texture → DMA-BUF export.
//!
//! The shared-NV12 path in [`super::shared_nv12`] uses the same Vulkan
//! sequence (create image with DRM_FORMAT_MODIFIER + EXTERNAL_MEMORY,
//! allocate memory with EXPORT, get fd, hand to wgpu) but spreads it
//! across two images bound to one allocation.

use std::os::fd::{FromRawFd, OwnedFd};

use ash::{ext, khr, vk};

use super::{
    find_memory_type, wgpu_format_to_vk, wgpu_usage_to_hal_uses, wgpu_usage_to_vk, DmaBufExport,
    ExportError, Result, DRM_FORMAT_MOD_LINEAR,
};

/// Export a freshly-allocated wgpu texture of the given shape as a
/// DMA-BUF. The texture comes back already wrapped in a `wgpu::Texture`
/// — you can bind it as a render target / storage texture / sampler
/// source per the requested `usage`, and the bound DMA-BUF stays
/// coherent as long as you respect the usual Vulkan synchronisation
/// (queue submits before consumer reads, etc.).
///
/// Currently only DRM_FORMAT_MOD_LINEAR is requested; the call still
/// queries the chosen modifier back from the driver, so consumers
/// don't need to assume.
///
/// # Safety
/// The wgpu `device` must outlive the returned `DmaBufExport.texture`
/// (it does by construction — `wgpu::Texture` holds a `Device` ref).
/// All Vulkan resource lifetime invariants are upheld internally.
pub fn export_texture_as_dmabuf(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    label: &str,
) -> Result<DmaBufExport> {
    if !device
        .features()
        .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
    {
        return Err(ExportError::FeatureUnsupported);
    }

    let vk_format = wgpu_format_to_vk(format)?;
    let vk_usage = wgpu_usage_to_vk(usage);

    // SAFETY: we check `as_hal` returns `Some` (verifies Vulkan
    // backend); raw_device / raw_instance / raw_physical_device are
    // valid for the lifetime of the hal::Device borrow, which spans
    // this entire function.
    unsafe {
        let hal_dev = device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or(ExportError::NotVulkan)?;
        let raw_device = hal_dev.raw_device();
        let raw_instance = hal_dev.shared_instance().raw_instance();
        let raw_physical = hal_dev.raw_physical_device();

        // Per-call extension function loaders. Lightweight — they're
        // just function-pointer tables. wgpu's own internal copies
        // aren't reachable from outside the crate, so we instantiate
        // our own; both reference the same underlying VkDevice so the
        // function pointers resolve identically.
        let ext_mem_fd = khr::external_memory_fd::Device::new(raw_instance, raw_device);
        let modifier_ext =
            ext::image_drm_format_modifier::Device::new(raw_instance, raw_device);

        let mut ext_mem_create = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        // Modifier-list path (vs explicit): we offer one modifier
        // (LINEAR) and let the driver "pick" it. Behaviour is identical
        // to the explicit single-modifier form for our one-item list,
        // but uses ImageDrmFormatModifierListCreateInfoEXT which is the
        // simpler variant — the explicit one requires us to also
        // supply per-plane subresource layouts up-front, which is the
        // wrong shape for "let the driver allocate, then tell me what
        // it picked".
        let modifiers = [DRM_FORMAT_MOD_LINEAR];
        let mut modifier_info = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
            .drm_format_modifiers(&modifiers);

        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            // DRM_FORMAT_MODIFIER_EXT instead of LINEAR — the spec
            // requires the modifier extension's tiling enum when any
            // DRM modifier info is in the pNext chain, even for the
            // linear modifier.
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut ext_mem_create)
            .push_next(&mut modifier_info);

        let image = raw_device
            .create_image(&image_create_info, None)
            .map_err(|e| ExportError::Vk(e, "vkCreateImage"))?;

        // Manual goto-fail cleanup: if any subsequent step errors we
        // must destroy `image` before returning. Wrap each fallible
        // call so we can centralise the cleanup; no scopeguard
        // dependency.
        let result = (|| -> Result<DmaBufExport> {
            let mem_req = raw_device.get_image_memory_requirements(image);

            // Look for a memory type that's:
            //   - Compatible with the image's requirements
            //   - DEVICE_LOCAL (we want GPU-native memory; DMA-BUF
            //     export from system memory works but is slower)
            // We accept any memory type if no DEVICE_LOCAL match (some
            // iGPUs report unified memory without DEVICE_LOCAL on the
            // dma-buf-exportable types).
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

            // Allocate memory tagged for export + dedicated to this
            // image. Dedicated allocation is recommended (and on many
            // drivers required) for external memory.
            let mut export_alloc_info = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let mut dedicated_info =
                vk::MemoryDedicatedAllocateInfo::default().image(image);

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_req.size)
                .memory_type_index(mem_type_index)
                .push_next(&mut export_alloc_info)
                .push_next(&mut dedicated_info);

            let memory = raw_device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| ExportError::Vk(e, "vkAllocateMemory"))?;

            // Inner cleanup scope: free memory on later failure.
            let inner = (|| -> Result<(vk::DeviceMemory, OwnedFd, u64, vk::SubresourceLayout)> {
                raw_device
                    .bind_image_memory(image, memory, 0)
                    .map_err(|e| ExportError::Vk(e, "vkBindImageMemory"))?;

                let fd_info = vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                let raw_fd = ext_mem_fd
                    .get_memory_fd(&fd_info)
                    .map_err(|e| ExportError::Vk(e, "vkGetMemoryFdKHR"))?;
                // SAFETY: vkGetMemoryFdKHR returned an open fd we own
                // and must close exactly once. OwnedFd handles that.
                let fd = OwnedFd::from_raw_fd(raw_fd);

                let mut mod_props =
                    vk::ImageDrmFormatModifierPropertiesEXT::default();
                modifier_ext
                    .get_image_drm_format_modifier_properties(image, &mut mod_props)
                    .map_err(|e| {
                        ExportError::Vk(e, "vkGetImageDrmFormatModifierPropertiesEXT")
                    })?;

                // Plane-0 layout. For DRM-modifier-tiled images you
                // MUST use `MEMORY_PLANE_*_EXT` aspect masks, not the
                // colour/depth ones the non-modifier path uses (spec:
                // VUID-VkImageSubresource-aspectMask-01672).
                let subres = vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT)
                    .mip_level(0)
                    .array_layer(0);
                let layout = raw_device.get_image_subresource_layout(image, subres);

                Ok((memory, fd, mod_props.drm_format_modifier, layout))
            })();

            let (memory, fd, drm_format_modifier, layout) = match inner {
                Ok(v) => v,
                Err(e) => {
                    raw_device.free_memory(memory, None);
                    return Err(e);
                }
            };

            // Hand off vk::Image + memory ownership to wgpu so a
            // future drop of the wgpu::Texture frees them. From here
            // on the goto-fail guard for `image` is no longer needed
            // — but we structured this so an Err return below
            // doesn't happen.
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
                usage: wgpu_usage_to_hal_uses(usage),
                memory_flags: wgpu::hal::MemoryFlags::empty(),
                view_formats: vec![],
            };

            // SAFETY: image was created from raw_device (this hal
            // device); memory was allocated from the same device and
            // bound to image; we pass `None` drop_callback so wgpu
            // takes ownership of both. (If texture_from_raw ever
            // panics on a debug-assert we'd leak image + memory —
            // panic = process death by our convention so this is
            // acceptable, but worth knowing.)
            let hal_tex = hal_dev.texture_from_raw(
                image,
                &hal_desc,
                None,
                wgpu::hal::vulkan::TextureMemory::Dedicated(memory),
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
                format,
                usage,
                view_formats: &[],
            };
            // SAFETY: hal_tex was built from this device on the same
            // Vulkan backend; wgpu_desc matches hal_desc except for
            // the usage representation difference (TextureUsages vs
            // TextureUses), which is the correct pairing.
            let texture = device
                .create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_tex, &wgpu_desc);

            Ok(DmaBufExport {
                texture,
                fd,
                drm_format_modifier,
                stride: layout.row_pitch,
                offset: layout.offset,
                size: mem_req.size,
            })
        })();

        match result {
            Ok(export) => Ok(export),
            Err(e) => {
                // Image wasn't transferred to wgpu (we error'd
                // before texture_from_raw); destroy it.
                raw_device.destroy_image(image, None);
                Err(e)
            }
        }
    }
}

