//! P010 Y+UV planes backed by one shared `VkDeviceMemory`, exported
//! as a single DMA-BUF.
//!
//! Same shape as [`super::shared_nv12`] — both planes living in one DRM
//! object at distinct offsets so `av_hwframe_map(DRM_PRIME → VAAPI)`
//! accepts it. The differences are per-plane format (`R16_UNORM` Y,
//! `R16G16_UNORM` UV) and per-cell value convention: P010 stores its
//! 10-bit data MSB-aligned in 16-bit cells (bits [15:6]; bits [5:0]
//! must read zero for spec-conforming consumers).
//!
//! The bytes-on-the-wire DRM fourccs are `R16` (Y) and `GR32` (UV) —
//! same family as P010 layer descriptors. FFmpeg's `vaapi_drm_format_map`
//! recognises this layer pair as `AV_PIX_FMT_P010LE` for both VAAPI
//! encoder *input* and decoder *output*.

use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;

use ash::{ext, khr, vk};

use super::{
    align_up, find_memory_type, wgpu_usage_to_hal_uses, wgpu_usage_to_vk, ExportError, Result,
    DRM_FORMAT_MOD_LINEAR,
};

/// One P010 surface (4:2:0 10-bit biplanar) backed by a single shared
/// dma-buf. Y plane is `R16Unorm` at full resolution; UV plane is
/// `Rg16Unorm` at half resolution.
pub struct SharedP010Export {
    pub y_texture: wgpu::Texture,
    pub uv_texture: wgpu::Texture,
    /// Single dma-buf fd referencing the shared `VkDeviceMemory`. Both
    /// plane descriptors point at `object_index=0` for ffmpeg /
    /// `av_hwframe_map`'s sake.
    pub fd: OwnedFd,
    /// Total allocation size for `AVDRMObjectDescriptor.size`.
    pub size: u64,
    /// DRM modifier the driver picked. Always LINEAR — see the
    /// rationale on [`super::shared_nv12::SharedNv12Export::modifier`];
    /// applies verbatim here.
    pub modifier: u64,
    pub y_offset: u64,
    pub y_pitch: u64,
    pub uv_offset: u64,
    pub uv_pitch: u64,
}

/// Allocate the Y and UV plane images of a P010 surface in **one**
/// shared `VkDeviceMemory` and export a single dma-buf fd referencing
/// the whole allocation.
///
/// `usage` is applied to both plane textures. Caller must ensure the
/// device's adapter advertises `STORAGE_IMAGE` on R16/Rg16 for the
/// chosen modifier — see [`crate::storable_dmabuf_modifiers`]. Without
/// that gate, downstream `create_compute_pipeline` on the 10-bit
/// pipeline can fail at runtime.
pub fn export_p010_shared_dmabuf(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsages,
) -> Result<SharedP010Export> {
    if !device
        .features()
        .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
    {
        return Err(ExportError::FeatureUnsupported);
    }
    // Same Intel iHD VAAPI 64-byte row-pitch alignment requirement as
    // the NV12 sibling (`shared_nv12.rs`). P010's Y plane is R16Unorm
    // = 2 bytes/pixel, so the 64-byte rule means luma-pixel width
    // aligned to 32. Non-32-aligned widths (e.g. 2180, 2340) would
    // produce the same left-edge row-aliasing corruption HEVC Main10
    // sessions saw in the NV12 path. See the NV12 fix's comment for
    // the permanence rationale and the AMD-VCN height alignment.
    const VAAPI_LUMA_STRIDE_ALIGN_P010: u32 = 32;
    const VAAPI_HEIGHT_ALIGN: u32 = 16;
    let aligned_w = width.next_multiple_of(VAAPI_LUMA_STRIDE_ALIGN_P010);
    let aligned_h = height.next_multiple_of(VAAPI_HEIGHT_ALIGN);
    let chroma_w = aligned_w.div_ceil(2);
    let chroma_h = aligned_h.div_ceil(2);
    let vk_usage = wgpu_usage_to_vk(usage);

    // SAFETY: hal escape hatch — Vulkan backend verified below; raw
    // handles valid for the lifetime of the hal::Device borrow, which
    // spans the entire unsafe block.
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

        // Same image-create template as the NV12 sibling; called twice
        // with R16 / Rg16 + per-plane extents. The modifier list lives
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
                .map_err(|e| ExportError::Vk(e, "vkCreateImage (P010 shared)"))
        };

        let y_image = create_image_with(vk::Format::R16_UNORM, aligned_w, aligned_h)?;
        let uv_image = match create_image_with(vk::Format::R16G16_UNORM, chroma_w, chroma_h) {
            Ok(i) => i,
            Err(e) => {
                raw_device.destroy_image(y_image, None);
                return Err(e);
            }
        };

        // From here on, any error must destroy both images (and free
        // memory if allocated). Goto-fail.
        let result = (|| -> Result<SharedP010Export> {
            let y_req = raw_device.get_image_memory_requirements(y_image);
            let uv_req = raw_device.get_image_memory_requirements(uv_image);

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

            let uv_align = uv_req.alignment.max(y_req.alignment);
            let uv_bind_offset = align_up(y_req.size, uv_align);
            let total_size = uv_bind_offset + uv_req.size;

            let mut export_alloc_info = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(total_size)
                .memory_type_index(mem_type_index)
                .push_next(&mut export_alloc_info);
            let memory = raw_device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| ExportError::Vk(e, "vkAllocateMemory (P010 shared)"))?;

            let bind_and_export = (|| -> Result<SharedP010Export> {
                raw_device
                    .bind_image_memory(y_image, memory, 0)
                    .map_err(|e| ExportError::Vk(e, "vkBindImageMemory (P010 Y)"))?;
                raw_device
                    .bind_image_memory(uv_image, memory, uv_bind_offset)
                    .map_err(|e| ExportError::Vk(e, "vkBindImageMemory (P010 UV)"))?;

                let fd_info = vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                let raw_fd = ext_mem_fd
                    .get_memory_fd(&fd_info)
                    .map_err(|e| ExportError::Vk(e, "vkGetMemoryFdKHR (P010 shared)"))?;
                // SAFETY: raw_fd is a freshly-returned open fd we own.
                let fd = OwnedFd::from_raw_fd(raw_fd);

                let mut y_mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
                modifier_ext
                    .get_image_drm_format_modifier_properties(y_image, &mut y_mod_props)
                    .map_err(|e| {
                        ExportError::Vk(e, "vkGetImageDrmFormatModifierPropertiesEXT (P010 Y)")
                    })?;
                debug_assert_eq!(
                    y_mod_props.drm_format_modifier, DRM_FORMAT_MOD_LINEAR,
                    "driver picked a non-LINEAR modifier despite single-element LINEAR list",
                );

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
                // For DRM_FORMAT_MOD_LINEAR, the per-image plane-0 layout
                // offset is always 0 (the bind offset captures the
                // in-allocation position). Future modifier additions or a
                // driver bug returning non-zero here would silently
                // double-add against `uv_bind_offset` and the encoder
                // would read garbage. Catch it in debug builds; release
                // ships with the same shape as the NV12 sibling.
                debug_assert_eq!(
                    y_layout.offset, 0,
                    "DRM_FORMAT_MOD_LINEAR Y layout.offset should be 0"
                );
                debug_assert_eq!(
                    uv_layout.offset, 0,
                    "DRM_FORMAT_MOD_LINEAR UV layout.offset should be 0"
                );
                // `uv_bind_offset` is already aligned to the driver-
                // reported `max(y_req.alignment, uv_req.alignment)`
                // upstream. The 4 KiB dma-buf page rule holds in
                // practice at production sizes but doesn't hold for
                // small allocations, so no explicit assertion here.

                let mem_arc = Arc::new(SharedDeviceMemory {
                    device: raw_device.clone(),
                    memory,
                });

                // wgpu texture wraps the aligned Vulkan image (see the
                // NV12 sibling for the dispatch-vs-textureDimensions
                // mismatch this introduces — the compute pass uses the
                // bridge's visible width for dispatch so padding stays
                // undefined and the encoder crops it).
                let y_texture = import_shared_image_into_wgpu(
                    device,
                    raw_device,
                    y_image,
                    wgpu::TextureFormat::R16Unorm,
                    aligned_w,
                    aligned_h,
                    usage,
                    "p010-shared y",
                    mem_arc.clone(),
                );
                let uv_texture = import_shared_image_into_wgpu(
                    device,
                    raw_device,
                    uv_image,
                    wgpu::TextureFormat::Rg16Unorm,
                    chroma_w,
                    chroma_h,
                    usage,
                    "p010-shared uv",
                    mem_arc,
                );

                Ok(SharedP010Export {
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

/// Shared `VkDeviceMemory` ownership for the two-image P010 export.
///
/// Identical shape to the NV12 sibling's `SharedDeviceMemory`. Two
/// independent types keep the per-format drop_callback closures from
/// having to know about each other's image-format details; the cost is
/// one extra struct definition.
struct SharedDeviceMemory {
    device: ash::Device,
    memory: vk::DeviceMemory,
}

impl Drop for SharedDeviceMemory {
    fn drop(&mut self) {
        // SAFETY: memory was allocated from `device`; we own it
        // exclusively here (last Arc clone) and won't touch it again.
        unsafe { self.device.free_memory(self.memory, None) };
    }
}

/// Wrap a Vulkan image bound to externally-managed (`Arc<SharedDeviceMemory>`)
/// memory as a `wgpu::Texture`. Mirror of the NV12 sibling helper —
/// kept separate so the per-export struct shape (and thus the closure
/// type captured by `drop_cb`) stays scoped to its own module.
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

    let device_for_drop = raw_device.clone();
    let drop_cb: wgpu::hal::DropCallback = Box::new(move || {
        // SAFETY: image was created on `device_for_drop`; no one else
        // holds a vk::Image handle equal to this one (minted exclusively
        // for this texture); memory backing is bound to it and stays
        // alive via mem_arc, which we drop after destroying the image.
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
