//! Packed YUV 4:4:4 10-bit (DRM_FORMAT_XV30) backed by a single
//! `VkDeviceMemory`, exported as one DMA-BUF.
//!
//! Sister to [`shared_yuv444`] (the 8-bit packed bridge) — same
//! single-plane shape, just one step up in bit depth. DRM_FORMAT_XV30
//! is the only 4:4:4 10-bit format ffmpeg's `vaapi_drm_format_map`
//! knows how to import (`MAP(XV30, YUV444_10, XV30, 0)` in
//! `libavutil/hwcontext_vaapi.c`); biplanar P410 has no entry, so the
//! packed layout isn't a stylistic choice, it's the only viable
//! input for HEVC Main 4:4:4 10-bit encode via VAAPI.
//!
//! Per-pixel layout from `<drm/drm_fourcc.h>`:
//!   DRM_FORMAT_XV30 = [31:0] X:V:U:Y 2:10:10:10 little endian
//!
//! In Vulkan's `A2B10G10R10_UNORM_PACK32` mapping (R→[9:0],
//! G→[19:10], B→[29:20], A→[31:30]) and wgpu's `Rgb10a2Unorm` storage
//! format, that lands as `R=Y, G=U, B=V, A=X`. The compute shader
//! (`bgra_to_xv30.wgsl`) writes `vec4<f32>(Y, U, V, 1.0)` to match.

use std::os::fd::{FromRawFd, OwnedFd};

use ash::{ext, khr, vk};

use super::{
    find_memory_type, wgpu_usage_to_hal_uses, wgpu_usage_to_vk, ExportError, Result,
    DRM_FORMAT_MOD_LINEAR,
};

/// One packed XV30 surface backed by a single dma-buf. Field shape
/// matches [`super::SharedYuv444Export`] — the consumer doesn't need
/// to know the bit depth, only the offset and pitch.
pub struct SharedXv30Export {
    pub packed_texture: wgpu::Texture,
    pub fd: OwnedFd,
    pub size: u64,
    pub modifier: u64,
    pub offset: u64,
    pub pitch: u64,
}

pub fn export_xv30_shared_dmabuf(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsages,
) -> Result<SharedXv30Export> {
    if !device
        .features()
        .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
    {
        return Err(ExportError::FeatureUnsupported);
    }
    // Intel HEVC 4:4:4 10-bit encode reads in 32×32 CTUs and (per the
    // same hardware-side row-pitch alignment quirk documented for NV12
    // in `nv12_dmabuf.rs`) wants pitch aligned to 128 bytes — i.e. 32
    // luma pixels at 4 B/px for the packed XV30 cell. The NV12-luma
    // bug was 64-byte alignment; at 4× the bytes-per-pixel the
    // analogous boundary lands at 128 bytes. Common widths (1920,
    // 2160, 2560, 3840) are already 32-pixel-aligned; this defends
    // against the rarer non-aligned widths.
    //
    // Height alignment matches `shared_yuv444.rs` — 16 rows for AMD
    // VCN read-past-visible-height defence — bumped to 32 here for
    // Intel's 32×32 CTU constraint at 4:4:4 10-bit. Cheap insurance;
    // overshoots by at most one CTU row.
    const VAAPI_LUMA_STRIDE_ALIGN_XV30: u32 = 32;
    const VAAPI_HEIGHT_ALIGN: u32 = 32;
    let aligned_w = width.next_multiple_of(VAAPI_LUMA_STRIDE_ALIGN_XV30);
    let aligned_h = height.next_multiple_of(VAAPI_HEIGHT_ALIGN);
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
            .format(vk::Format::A2B10G10R10_UNORM_PACK32)
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
        let image = raw_device
            .create_image(&info, None)
            .map_err(|e| ExportError::Vk(e, "vkCreateImage (XV30)"))?;

        let result = (|| -> Result<SharedXv30Export> {
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
                .map_err(|e| ExportError::Vk(e, "vkAllocateMemory (XV30)"))?;

            let bind_and_export = (|| -> Result<SharedXv30Export> {
                raw_device
                    .bind_image_memory(image, memory, 0)
                    .map_err(|e| ExportError::Vk(e, "vkBindImageMemory (XV30)"))?;

                let fd_info = vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                let raw_fd = ext_mem_fd
                    .get_memory_fd(&fd_info)
                    .map_err(|e| ExportError::Vk(e, "vkGetMemoryFdKHR (XV30)"))?;
                // SAFETY: raw_fd is a freshly-returned open fd we own.
                let fd = OwnedFd::from_raw_fd(raw_fd);

                let plane0_subres = vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT)
                    .mip_level(0)
                    .array_layer(0);
                let layout = raw_device.get_image_subresource_layout(image, plane0_subres);
                debug_assert_eq!(
                    layout.offset, 0,
                    "plane-0 subresource offset must be 0 for LINEAR-modifier single-plane image",
                );

                let mut mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
                modifier_ext
                    .get_image_drm_format_modifier_properties(image, &mut mod_props)
                    .map_err(|e| {
                        ExportError::Vk(e, "vkGetImageDrmFormatModifierPropertiesEXT (XV30)")
                    })?;
                debug_assert_eq!(
                    mod_props.drm_format_modifier, DRM_FORMAT_MOD_LINEAR,
                    "driver picked a non-LINEAR modifier despite single-element LINEAR list",
                );

                let device_for_drop = raw_device.clone();
                let memory_for_drop = memory;
                let drop_cb: wgpu::hal::DropCallback = Box::new(move || {
                    // SAFETY (extends the outer `unsafe` block): image
                    // was created on `device_for_drop`; no one else
                    // holds this vk::Image handle; the memory is
                    // exclusively bound to this image.
                    device_for_drop.destroy_image(image, None);
                    device_for_drop.free_memory(memory_for_drop, None);
                });

                let hal_desc = wgpu::hal::TextureDescriptor {
                    label: Some("xv30 shared"),
                    size: wgpu::Extent3d {
                        width: aligned_w,
                        height: aligned_h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgb10a2Unorm,
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
                    label: Some("xv30 shared"),
                    size: wgpu::Extent3d {
                        width: aligned_w,
                        height: aligned_h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgb10a2Unorm,
                    usage,
                    view_formats: &[],
                };
                let packed_texture = device
                    .create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_tex, &wgpu_desc);

                Ok(SharedXv30Export {
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
