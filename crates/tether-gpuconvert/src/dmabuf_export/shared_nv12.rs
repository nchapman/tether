//! NV12 Y+UV planes backed by one shared `VkDeviceMemory`, exported
//! as a single DMA-BUF.
//!
//! This is the shape `av_hwframe_map(DRM_PRIME → VAAPI)` requires —
//! both planes living in the same DRM object at distinct offsets.

use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;

use ash::{ext, khr, vk};

use super::{
    align_up, find_memory_type, wgpu_usage_to_hal_uses, wgpu_usage_to_vk, ExportError, Result,
    DRM_FORMAT_MOD_LINEAR,
};

/// One NV12 surface backed by a single shared DMA-BUF, suitable for
/// hand-off to `av_hwframe_map(DRM_PRIME → VAAPI)` which requires both
/// planes to live in the *same* DRM object at different offsets.
///
/// Layout:
/// ```text
///   [   Y plane (R8, width × height)   ] [pad to UV align] [ UV plane (Rg8, w/2 × h/2) ]
///   ^offset 0                                              ^uv_offset
/// ```
/// Each plane is a real `wgpu::Texture` the compute pass can bind as a
/// storage image; the underlying memory is one `VkDeviceMemory` exported
/// once as a single dma-buf fd.
pub struct SharedNv12Export {
    pub y_texture: wgpu::Texture,
    pub uv_texture: wgpu::Texture,
    /// Single dma-buf fd referencing the shared `VkDeviceMemory`. Both
    /// plane descriptors point at `object_index=0` for ffmpeg /
    /// `av_hwframe_map`'s sake.
    pub fd: OwnedFd,
    /// Total allocation size — what `AVDRMObjectDescriptor.size` wants.
    pub size: u64,
    /// DRM modifier the driver picked. Always LINEAR for this path —
    /// tiled modifiers don't have a portable "plane-offset within
    /// shared allocation" contract that VAAPI's DRM_PRIME importer
    /// honours; LINEAR is the only safe choice for ffmpeg interop here.
    pub modifier: u64,
    /// Y plane offset / row pitch in the shared allocation.
    pub y_offset: u64,
    pub y_pitch: u64,
    /// UV plane offset / row pitch in the shared allocation.
    pub uv_offset: u64,
    pub uv_pitch: u64,
}

/// Allocate the Y and UV plane images of an NV12 surface in **one**
/// shared `VkDeviceMemory` and export a single dma-buf fd referencing
/// the whole allocation. The two `wgpu::Texture`s are independent
/// resources (different bindings, different formats) but the underlying
/// memory is one object — which is what ffmpeg's hwframe_map for VAAPI
/// requires.
///
/// `usage` is applied to both plane textures (typically
/// `STORAGE_BINDING` for the compute pass + `COPY_SRC` for the readback
/// tests). Modifier is hard-pinned to LINEAR; tiled modifiers don't have
/// a portable shared-allocation contract for VAAPI consumers.
pub fn export_nv12_shared_dmabuf(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsages,
) -> Result<SharedNv12Export> {
    if !device
        .features()
        .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
    {
        return Err(ExportError::FeatureUnsupported);
    }
    // Intel iHD's VAAPI NV12 import expects the Y-plane row pitch to
    // be aligned to 64 bytes (luma pixels — R8 is 1 byte/pixel).
    // When the Vulkan driver picks a tight `row_pitch == width` for
    // LINEAR-modifier R8 images at non-64-aligned widths (e.g. 2160),
    // VAAPI reads as if the pitch were `align_up(width, 64)` — each
    // row's leftmost 16 luma columns alias into the previous row's
    // padding, producing visible left-edge corruption in every
    // encoded frame. Allocating the underlying images at the aligned
    // width forces the driver to report a 64-aligned `row_pitch`;
    // the bridge's caller still sees the original `width` and the
    // shader only writes to the visible region, leaving the right-
    // edge padding columns undefined (the encoder crops to the
    // declared frame width and ignores them). UV is half-resolution
    // so `aligned_width / 2` is automatically aligned too.
    const VAAPI_LUMA_STRIDE_ALIGN: u32 = 64;
    let aligned_w = width.next_multiple_of(VAAPI_LUMA_STRIDE_ALIGN);
    let chroma_w = aligned_w.div_ceil(2);
    let chroma_h = height.div_ceil(2);
    let vk_usage = wgpu_usage_to_vk(usage);

    // SAFETY: hal escape hatch — verified Vulkan backend below; raw
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

        // Same image-create template as the single-plane path; called
        // twice with different format + extent. The modifier list lives
        // outside the closure so the &-borrow into ImageCreateInfo's
        // push_next stays valid for both calls.
        let modifiers = [DRM_FORMAT_MOD_LINEAR];
        let create_image_with = |format: vk::Format, w: u32, h: u32| -> Result<vk::Image> {
            let mut ext_mem_create = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let mut modifier_info = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
                .drm_format_modifiers(&modifiers);
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width: w, height: h, depth: 1 })
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
                .map_err(|e| ExportError::Vk(e, "vkCreateImage (NV12 shared)"))
        };

        let y_image = create_image_with(vk::Format::R8_UNORM, aligned_w, height)?;
        let uv_image = match create_image_with(vk::Format::R8G8_UNORM, chroma_w, chroma_h) {
            Ok(i) => i,
            Err(e) => {
                raw_device.destroy_image(y_image, None);
                return Err(e);
            }
        };

        // From here on, any error must destroy both images (and free
        // memory if allocated). Manual goto-fail.
        let result = (|| -> Result<SharedNv12Export> {
            let y_req = raw_device.get_image_memory_requirements(y_image);
            let uv_req = raw_device.get_image_memory_requirements(uv_image);

            // Memory-type intersection: both images must accept the
            // same memory type for one allocation to back both. In
            // practice memory_type_bits is identical for two LINEAR
            // images on a given physical device + same handle type;
            // intersection-empty is exotic and we surface it cleanly.
            let combined_type_bits = y_req.memory_type_bits & uv_req.memory_type_bits;
            if combined_type_bits == 0 {
                raw_device.destroy_image(uv_image, None);
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

            // UV bind offset must satisfy both the UV image's alignment
            // and span the Y image's full size. Use the stricter of the
            // two alignments to be safe across drivers.
            let uv_align = uv_req.alignment.max(y_req.alignment);
            let uv_bind_offset = align_up(y_req.size, uv_align);
            let total_size = uv_bind_offset + uv_req.size;

            // External + bind-image-memory dance. Dedicated allocation
            // is NOT used here — by definition the memory backs two
            // images, so it can't be dedicated to either.
            let mut export_alloc_info = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(total_size)
                .memory_type_index(mem_type_index)
                .push_next(&mut export_alloc_info);
            let memory = raw_device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| ExportError::Vk(e, "vkAllocateMemory (NV12 shared)"))?;

            let bind_and_export = (|| -> Result<SharedNv12Export> {
                raw_device
                    .bind_image_memory(y_image, memory, 0)
                    .map_err(|e| ExportError::Vk(e, "vkBindImageMemory (Y)"))?;
                raw_device
                    .bind_image_memory(uv_image, memory, uv_bind_offset)
                    .map_err(|e| ExportError::Vk(e, "vkBindImageMemory (UV)"))?;

                let fd_info = vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                let raw_fd = ext_mem_fd
                    .get_memory_fd(&fd_info)
                    .map_err(|e| ExportError::Vk(e, "vkGetMemoryFdKHR (NV12 shared)"))?;
                // SAFETY: raw_fd is a freshly-returned open fd we own.
                let fd = OwnedFd::from_raw_fd(raw_fd);

                let mut y_mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
                modifier_ext
                    .get_image_drm_format_modifier_properties(y_image, &mut y_mod_props)
                    .map_err(|e| {
                        ExportError::Vk(e, "vkGetImageDrmFormatModifierPropertiesEXT (Y)")
                    })?;

                let y_subres = vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT)
                    .mip_level(0)
                    .array_layer(0);
                let y_layout = raw_device.get_image_subresource_layout(y_image, y_subres);
                let uv_subres = vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT)
                    .mip_level(0)
                    .array_layer(0);
                let uv_layout = raw_device.get_image_subresource_layout(uv_image, uv_subres);

                // Shared lifetime for the single VkDeviceMemory: both
                // plane textures hand a clone of `mem_arc` to their hal
                // drop_callback, plus their own VkImage. Last clone to
                // drop frees the memory.
                let mem_arc = Arc::new(SharedDeviceMemory {
                    device: raw_device.clone(),
                    memory,
                });

                // The wgpu textures wrap the Vulkan images, so their
                // descriptor dimensions must match the image's
                // (aligned) dimensions, not the visible ones. The
                // bridge's compute dispatch (in `nv12_dmabuf::convert`)
                // is keyed off the bridge's stored visible width/height
                // so it only writes to `[0, width) × [0, height)` — the
                // right-edge padding columns stay undefined and the
                // encoder crops them off via the declared frame width.
                let y_texture = import_shared_image_into_wgpu(
                    device,
                    raw_device,
                    y_image,
                    wgpu::TextureFormat::R8Unorm,
                    aligned_w,
                    height,
                    usage,
                    "nv12-shared y",
                    mem_arc.clone(),
                );
                let uv_texture = import_shared_image_into_wgpu(
                    device,
                    raw_device,
                    uv_image,
                    wgpu::TextureFormat::Rg8Unorm,
                    chroma_w,
                    chroma_h,
                    usage,
                    "nv12-shared uv",
                    mem_arc,
                );

                Ok(SharedNv12Export {
                    y_texture,
                    uv_texture,
                    fd,
                    size: total_size,
                    modifier: y_mod_props.drm_format_modifier,
                    y_offset: y_layout.offset,
                    y_pitch: y_layout.row_pitch,
                    uv_offset: uv_bind_offset + uv_layout.offset,
                    uv_pitch: uv_layout.row_pitch,
                })
            })();

            match bind_and_export {
                Ok(export) => Ok(export),
                Err(e) => {
                    // Images and memory weren't transferred (we error'd
                    // before texture import). Destroy in reverse order.
                    raw_device.free_memory(memory, None);
                    raw_device.destroy_image(uv_image, None);
                    raw_device.destroy_image(y_image, None);
                    Err(e)
                }
            }
        })();

        result
    }
}

/// Shared `VkDeviceMemory` ownership for the two-image NV12 export. Both
/// plane textures hold an `Arc<Self>` (handed to their wgpu-hal
/// drop_callback); the last clone to drop frees the memory.
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

/// Wrap a Vulkan image bound to externally-managed (`Arc<SharedDeviceMemory>`)
/// memory as a `wgpu::Texture`. The drop_callback destroys the image
/// and drops the Arc clone — the last Arc frees the memory.
#[allow(clippy::too_many_arguments)]
unsafe fn import_shared_image_into_wgpu(
    device: &wgpu::Device,
    raw_device: &ash::Device,
    image: vk::Image,
    format: wgpu::TextureFormat,
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
        format,
        usage: wgpu_usage_to_hal_uses(usage),
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };

    // drop_callback fires once when the wgpu::Texture is dropped:
    // destroy the image, then drop our Arc clone (which may free
    // memory if this was the last clone).
    let device_for_drop = raw_device.clone();
    let drop_cb: wgpu::hal::DropCallback = Box::new(move || {
        // SAFETY: image was created on `device_for_drop`; no one else
        // holds a vk::Image handle equal to this one (we minted it
        // exclusively for this texture); memory backing was bound to
        // it and is still alive via mem_arc, which we drop after.
        unsafe { device_for_drop.destroy_image(image, None) };
        drop(mem_arc);
    });

    // SAFETY: caller asserts device is Vulkan-backed; `image` was
    // created from this device's raw_device and bound to mem_arc's
    // memory at the appropriate offset; drop_cb takes ownership of
    // both `image` and the mem_arc clone.
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
            format,
            usage,
            view_formats: &[],
        };
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_tex, &wgpu_desc)
    }
}
