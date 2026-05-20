//! Export a wgpu texture as a DMA-BUF file descriptor.
//!
//! This is the mirror image of wgpu's existing `texture_from_dmabuf_fd`
//! — same Vulkan extensions (`VK_EXT_external_memory_dma_buf`,
//! `VK_EXT_image_drm_format_modifier`, `VK_KHR_external_memory_fd`),
//! just the export direction. Lets us hand wgpu-compute output
//! (NV12 Y + UV plane textures) to another GPU API (VAAPI here, the
//! same shape on macOS would be IOSurface and on Windows D3D11 shared
//! handles) without a CPU round-trip.
//!
//! v1 only handles `DRM_FORMAT_MOD_LINEAR`. Linear is universally
//! importable across compositors, drivers, and GPU APIs; tiled
//! modifiers buy 5-20% bandwidth on streaming-sized textures at the
//! cost of complex per-vendor capability negotiation. We'll add a
//! proper modifier query path when the bandwidth wins become
//! measurable in real workloads.
//!
//! Eventually this belongs upstream in wgpu as the obvious counterpart
//! to `texture_from_dmabuf_fd`. Keeping the API shape clean (raw
//! `wgpu::TextureDescriptor` input, opaque return) so the migration
//! when wgpu lands the equivalent is a delete-this-file change.

use std::os::fd::{FromRawFd, OwnedFd};

use ash::{ext, khr, vk};

/// DRM format modifier for linear-tiled memory. From
/// `<drm/drm_fourcc.h>` — `DRM_FORMAT_MOD_LINEAR == 0`. Universally
/// supported by every VAAPI/Vulkan/EGL importer.
pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// One exported wgpu texture along with the DMA-BUF descriptor a
/// consumer (libva, IOSurface, etc.) needs to import it.
///
/// Field shapes match what `VASurfaceAttribExternalBuffers` /
/// `VADRMPRIMESurfaceDescriptor` consume. `tether_codec`'s
/// `DmaBufFrame` is constructed from these.
pub struct DmaBufExport {
    /// The wgpu texture. Owns the underlying `VkImage` and bound
    /// `VkDeviceMemory`; their lifetime is tied to this `Texture`'s
    /// `Drop`. Safe to render into / sample as usual.
    pub texture: wgpu::Texture,
    /// DMA-BUF file descriptor referencing the same `VkDeviceMemory`.
    /// `OwnedFd` so close-exactly-once is type-enforced. The consumer
    /// (e.g. libva's PRIME_2 import) typically `dup`s on its side, so
    /// dropping this `fd` after the consumer reffed the dma-buf is
    /// safe — the memory itself stays bound to `texture`.
    pub fd: OwnedFd,
    /// DRM modifier the Vulkan driver actually picked. We asked for
    /// `LINEAR`; we query back to verify and so the consumer's
    /// import call has the authoritative value to pass in.
    pub drm_format_modifier: u64,
    /// Plane-0 row pitch in bytes. From
    /// `vkGetImageSubresourceLayout(MEMORY_PLANE_0_EXT)` — the
    /// authoritative number, not derived from the format.
    pub stride: u64,
    /// Plane-0 byte offset into the DMA-BUF. Usually 0 for
    /// single-plane formats but the spec allows non-zero so we plumb
    /// it through.
    pub offset: u64,
    /// Total memory allocation size for this image. Some importers
    /// (libva's `VASurfaceAttribExternalBuffers.data_size`) need it.
    pub size: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("device is not Vulkan-backed; DMA-BUF export only works on the wgpu Vulkan backend")]
    NotVulkan,
    #[error(
        "wgpu adapter doesn't advertise VULKAN_EXTERNAL_MEMORY_DMA_BUF; \
         enable it on device creation and check the Vulkan ICD supports \
         VK_EXT_external_memory_dma_buf + VK_EXT_image_drm_format_modifier"
    )]
    FeatureUnsupported,
    #[error("vulkan call failed: {0:?} ({1})")]
    Vk(vk::Result, &'static str),
    #[error("no Vulkan memory type supports the required heap properties + DMA-BUF export")]
    NoMemoryType,
    #[error("unsupported wgpu texture format for DMA-BUF export: {0:?}")]
    UnsupportedFormat(wgpu::TextureFormat),
}

pub type Result<T> = std::result::Result<T, ExportError>;

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

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits_req: u32,
    flags_req: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for (i, mem_ty) in mem_props.memory_types_as_slice().iter().enumerate() {
        let types_bits = 1u32 << i;
        let is_required_memory_type = type_bits_req & types_bits != 0;
        let has_required_properties = mem_ty.property_flags & flags_req == flags_req;
        if is_required_memory_type && has_required_properties {
            return Some(i as u32);
        }
    }
    None
}

/// Translate the wgpu texture formats we export to their Vulkan
/// equivalents. Small fixed set — we only export single-plane Y/UV
/// for NV12, plus BGRA8 for round-trip tests. New formats add here.
fn wgpu_format_to_vk(f: wgpu::TextureFormat) -> Result<vk::Format> {
    match f {
        wgpu::TextureFormat::R8Unorm => Ok(vk::Format::R8_UNORM),
        wgpu::TextureFormat::Rg8Unorm => Ok(vk::Format::R8G8_UNORM),
        wgpu::TextureFormat::Bgra8Unorm => Ok(vk::Format::B8G8R8A8_UNORM),
        _ => Err(ExportError::UnsupportedFormat(f)),
    }
}

/// Map wgpu's user-facing `TextureUsages` to Vulkan's `ImageUsageFlags`.
/// Mirrors what wgpu-hal's `conv::map_texture_usage` does internally
/// for the bits we care about.
///
/// **Must stay in sync with [`wgpu_usage_to_hal_uses`] below**: the
/// VkImage's actual `VkImageUsageFlags` (this function) and what we
/// tell wgpu's barrier tracker (the other) describe the same image
/// and must not disagree on what operations it can sink. Both
/// functions are private to this module and called once each per
/// export — if you add a new TextureUsages bit, add it to *both*.
fn wgpu_usage_to_vk(u: wgpu::TextureUsages) -> vk::ImageUsageFlags {
    let mut out = vk::ImageUsageFlags::empty();
    if u.contains(wgpu::TextureUsages::COPY_SRC) {
        out |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if u.contains(wgpu::TextureUsages::COPY_DST) {
        out |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    if u.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
        out |= vk::ImageUsageFlags::SAMPLED;
    }
    if u.contains(wgpu::TextureUsages::STORAGE_BINDING) {
        out |= vk::ImageUsageFlags::STORAGE;
    }
    if u.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
        out |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    out
}

/// Same translation in the wgpu-hal `TextureUses` vocabulary, for the
/// `hal::TextureDescriptor` we hand to `texture_from_raw`. The two
/// enums describe the same concept in different layer-appropriate
/// terms; this conversion mirrors the one wgpu does internally when
/// a public `TextureDescriptor` is lowered to hal.
fn wgpu_usage_to_hal_uses(u: wgpu::TextureUsages) -> wgpu::TextureUses {
    let mut out = wgpu::TextureUses::empty();
    if u.contains(wgpu::TextureUsages::COPY_SRC) {
        out |= wgpu::TextureUses::COPY_SRC;
    }
    if u.contains(wgpu::TextureUsages::COPY_DST) {
        out |= wgpu::TextureUses::COPY_DST;
    }
    if u.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
        out |= wgpu::TextureUses::RESOURCE;
    }
    if u.contains(wgpu::TextureUsages::STORAGE_BINDING) {
        // wgpu's hal layer splits STORAGE_BINDING into READ_ONLY /
        // WRITE_ONLY / READ_WRITE / ATOMIC bits — chosen per bind
        // group at use time. The export-side image declares the
        // most-permissive (READ_WRITE) so wgpu's barrier tracker
        // emits a barrier compatible with any subsequent storage
        // access. Cost is one possibly-unnecessary barrier when the
        // texture is only read; never produces a missing-barrier bug.
        out |= wgpu::TextureUses::STORAGE_READ_WRITE;
    }
    if u.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
        out |= wgpu::TextureUses::COLOR_TARGET;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;

    async fn make_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .ok()?;
        if !adapter
            .features()
            .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
        {
            return None;
        }
        let required_features = wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("dmabuf export test"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .ok()
    }

    /// Plain export sanity. Doesn't try to use the DMA-BUF — just
    /// validates the call returns a plausible descriptor (fd > 0,
    /// stride >= row size, modifier as requested).
    #[test]
    #[ignore = "requires VK_EXT_external_memory_dma_buf (run on a real Linux GPU adapter)"]
    fn export_r8unorm_smoke() {
        let Some((device, _queue)) = pollster::block_on(make_device()) else {
            eprintln!("SKIP: no wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF");
            return;
        };

        let width = 1920u32;
        let height = 1080u32;
        let export = export_texture_as_dmabuf(
            &device,
            width,
            height,
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            "test r8 y plane",
        )
        .expect("export r8");

        assert!(export.fd.as_raw_fd() >= 0, "fd must be valid");
        assert_eq!(export.drm_format_modifier, DRM_FORMAT_MOD_LINEAR);
        assert!(
            export.stride >= u64::from(width),
            "stride {} must be >= row width {}",
            export.stride,
            width
        );
        // Allocation must hold at least height * stride bytes.
        assert!(
            export.size >= u64::from(height) * export.stride,
            "size {} must be >= height({}) * stride({})",
            export.size,
            height,
            export.stride
        );
        // Offset is typically 0 for single-plane formats but the spec
        // allows non-zero. Just bound it.
        assert!(export.offset < export.size);
    }

    /// Round-trip: export → write a known pattern via wgpu →
    /// re-import the same DMA-BUF as a separate texture →
    /// copy_texture_to_buffer + readback → verify bytes match.
    ///
    /// Proves the exported memory is *the same memory* the importer
    /// sees (i.e. the fd actually references the right
    /// VkDeviceMemory), and that the stride/offset/modifier values we
    /// report let the importer access the data correctly. This is
    /// the smallest test that validates the end-to-end export
    /// contract without requiring VAAPI.
    #[test]
    #[ignore = "requires VK_EXT_external_memory_dma_buf"]
    fn export_then_reimport_roundtrip() {
        let Some((device, queue)) = pollster::block_on(make_device()) else {
            eprintln!("SKIP: no wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF");
            return;
        };

        let width = 64u32;
        let height = 32u32;

        // Export with COPY_DST so we can write via queue.write_texture,
        // plus COPY_SRC isn't needed on the export side — the importer
        // is the one reading. STORAGE_BINDING is irrelevant for this
        // test but present in the production usage path.
        let export = export_texture_as_dmabuf(
            &device,
            width,
            height,
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING,
            "roundtrip export",
        )
        .expect("export");

        // Write a recognisable pattern. With LINEAR tiling, byte
        // (x, y) at offset `y * stride + x` should be `(x ^ y) as u8`
        // — distinguishable even at small sizes.
        let mut bytes = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                bytes[(y * width + x) as usize] = ((x ^ y) & 0xff) as u8;
            }
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &export.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Submit + wait so the write lands in memory before the
        // import reads it. wgpu's write_texture is queued; we need
        // the queue to drain.
        queue.submit(std::iter::empty());
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll write");

        // Re-import via wgpu's existing import path. `try_clone` dups
        // the fd because texture_from_dmabuf_fd takes ownership.
        let import_fd = export
            .fd
            .try_clone()
            .expect("dup fd for re-import");
        let import_desc = wgpu::hal::TextureDescriptor {
            label: Some("roundtrip import"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUses::COPY_SRC,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        // SAFETY: import_fd was just produced by us via the export
        // path, so the modifier/stride/offset we pass match.
        let import_hal = unsafe {
            device
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("vulkan backend")
                .texture_from_dmabuf_fd(
                    import_fd,
                    &import_desc,
                    export.drm_format_modifier,
                    export.stride,
                    export.offset,
                )
                .expect("texture_from_dmabuf_fd")
        };
        let import_wgpu_desc = wgpu::TextureDescriptor {
            label: Some("roundtrip import"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        // SAFETY: hal texture built from this device on the same
        // backend; descs match shape.
        let import_tex = unsafe {
            device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                import_hal,
                &import_wgpu_desc,
            )
        };

        // Copy the imported texture into a readback buffer and verify.
        let padded_row = ((width + 255) / 256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(padded_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback enc") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &import_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll readback");
        rx.recv().expect("map callback").expect("map");
        let mapped = slice.get_mapped_range().expect("get_mapped_range");

        for y in 0..height {
            for x in 0..width {
                let got = mapped[(y * padded_row + x) as usize];
                let want = ((x ^ y) & 0xff) as u8;
                assert_eq!(
                    got, want,
                    "byte ({x},{y}): exported pattern didn't survive re-import"
                );
            }
        }
    }

    /// Same round-trip but at Rg8Unorm (NV12 UV plane format) and
    /// with an *odd* width to exercise stride padding. The export
    /// path's `vkGetImageSubresourceLayout` should report a stride
    /// that the importer's per-row math respects.
    #[test]
    #[ignore = "requires VK_EXT_external_memory_dma_buf"]
    fn export_rg8unorm_odd_width_roundtrip() {
        let Some((device, queue)) = pollster::block_on(make_device()) else {
            eprintln!("SKIP: no wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF");
            return;
        };

        let width = 63u32; // odd — forces driver to pad the row
        let height = 32u32;

        let export = export_texture_as_dmabuf(
            &device,
            width,
            height,
            wgpu::TextureFormat::Rg8Unorm,
            wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING,
            "rg8 roundtrip",
        )
        .expect("export rg8");

        // Each texel is 2 bytes (R + G). Write a per-texel pattern:
        // R = x, G = y. Tight rows on the CPU side; wgpu's
        // write_texture handles any per-row padding internally.
        let row_bytes = (width * 2) as usize;
        let mut bytes = vec![0u8; row_bytes * height as usize];
        for y in 0..height {
            for x in 0..width {
                let i = (y as usize) * row_bytes + (x as usize) * 2;
                bytes[i] = (x & 0xff) as u8;
                bytes[i + 1] = (y & 0xff) as u8;
            }
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &export.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 2),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::empty());
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll write");

        // Re-import using the export's reported stride/offset/modifier.
        // Dup the fd then drop the export's own fd *before* the
        // import reads, to prove the fd handle's lifetime is
        // independent of the memory's lifetime (memory stays alive
        // via the export's wgpu::Texture even after the fd is gone).
        let import_fd = export.fd.try_clone().expect("dup");
        let original_fd_value = {
            use std::os::fd::AsRawFd;
            export.fd.as_raw_fd()
        };
        drop(export.fd);
        let _ = original_fd_value; // future-proofing if we ever assert on it

        let import_desc = wgpu::hal::TextureDescriptor {
            label: Some("rg8 import"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUses::COPY_SRC,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        // SAFETY: import_fd was just produced by the export path so
        // modifier/stride/offset are authoritative.
        let import_hal = unsafe {
            device
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("vulkan backend")
                .texture_from_dmabuf_fd(
                    import_fd,
                    &import_desc,
                    export.drm_format_modifier,
                    export.stride,
                    export.offset,
                )
                .expect("texture_from_dmabuf_fd Rg8")
        };
        let import_wgpu_desc = wgpu::TextureDescriptor {
            label: Some("rg8 import"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let import_tex = unsafe {
            device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                import_hal,
                &import_wgpu_desc,
            )
        };

        // Read back. Rg8 is 2 bytes per texel.
        let row_pad = ((width * 2 + 255) / 256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rg8 readback"),
            size: u64::from(row_pad) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rg8 enc") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &import_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_pad),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        rx.recv().expect("cb").expect("map");
        let mapped = slice.get_mapped_range().expect("range");

        for y in 0..height {
            for x in 0..width {
                let row_off = y * row_pad;
                let i = (row_off + x * 2) as usize;
                assert_eq!(
                    mapped[i],
                    (x & 0xff) as u8,
                    "R[{x},{y}] mismatched (stride={})",
                    export.stride
                );
                assert_eq!(
                    mapped[i + 1],
                    (y & 0xff) as u8,
                    "G[{x},{y}] mismatched"
                );
            }
        }
    }
}
