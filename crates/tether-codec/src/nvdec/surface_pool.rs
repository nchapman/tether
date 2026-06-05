//! Fixed pool of NV12 dma-buf surfaces the zero-copy NVDEC decoder fills
//! and hands to the renderer (GH #16, D2).
//!
//! NVDEC decodes into CUDA surfaces; the renderer imports an NV12 dma-buf
//! (`GpuFrameSource::DmaBuf`, 2 layers R8 + GR88) directly into wgpu — the
//! same shape the VAAPI decoder exports. So the decoder needs a dma-buf it
//! can write the decoded pixels into. This pool allocates a small fixed set
//! of NV12 surfaces and lends them out one at a time. A surface stays
//! reserved (its slot marked in-use) until the renderer drops the frame, so
//! the decoder can't overwrite a surface the renderer is still reading;
//! [`PooledSurface`]'s slot guard returns the slot on drop.
//!
//! **Why raw Vulkan (`ash`), not `tether-gpuconvert`'s exporter.** The
//! obvious move is to reuse `tether_gpuconvert::export_nv12_shared_dmabuf`,
//! but tether-codec depending on tether-gpuconvert in production forms a
//! normal-dependency cycle (tether-gpuconvert already depends on tether-codec
//! on macOS for IOSurface interop; cargo's resolver-2 evaluates the graph
//! cfg-merged and rejects the cross-platform mutual prod dependency). So this
//! module reimplements the minimal NV12 two-plane shared-allocation export
//! directly on `ash`. It mirrors `shared_nv12.rs`'s layout and — load-bearing
//! — its 64-byte Y-row-pitch + 16-row-height alignment (Intel iHD's importer
//! mis-reads non-aligned widths; harmless on NVIDIA/RADV but kept identical so
//! the descriptor is universally importable; see the gpuconvert comment).
//!
//! Single-GPU assumption (known follow-up, not solved here): the Vulkan
//! physical device is the first NVIDIA (`0x10de`) device, which must be the
//! same physical GPU as the decoder's CUDA context (device 0) for the EGL→CUDA
//! import of the surface dma-buf to succeed. On the target box (RTX 3090 Ti)
//! that's the only NVIDIA GPU. A multi-GPU host would need an explicit
//! device-match (matched by PCI/UUID), deferred until a user needs it.

use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ash::vk;

use crate::{CodecError, Result};

/// Number of NV12 surfaces in the pool. NVDEC reorder depth plus a couple
/// in flight to the renderer; matches the encoder's small-pool sizing
/// rationale (a handful is plenty when the renderer consumes promptly).
const POOL_SIZE: usize = 4;

/// NVIDIA PCI vendor id (matches `crate::nvenc`'s sysfs check). The Vulkan
/// physical device we allocate surfaces on must be this vendor so the dma-buf
/// is importable by the CUDA-device-0 EGL display.
const VENDOR_NVIDIA: u32 = 0x10de;

/// `DRM_FORMAT_MOD_LINEAR` — universally importable; the only modifier with a
/// portable plane-offset-within-shared-allocation contract.
const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// Load-bearing alignment (identical to `tether_gpuconvert::shared_nv12`):
/// Intel iHD's DRM_PRIME importer mis-reads non-64-aligned luma row pitches;
/// 16-row height defends against readers running past visible_height. No-op on
/// NVIDIA/RADV (they already report ≥128-byte pitch) but kept so the exported
/// descriptor is universally importable.
const LUMA_STRIDE_ALIGN: u32 = 64;
const HEIGHT_ALIGN: u32 = 16;

/// The dma-buf descriptor fields for one NV12 surface, as produced by the
/// shared-allocation export. Mirrors `tether_gpuconvert::SharedNv12Export`'s
/// descriptor fields (minus the wgpu textures, which the decoder doesn't need
/// — it writes via CUDA, not wgpu).
struct SurfaceExport {
    /// Single dma-buf fd referencing the shared `VkDeviceMemory` (both planes).
    fd: OwnedFd,
    size: u64,
    modifier: u64,
    y_offset: u64,
    y_pitch: u64,
    uv_offset: u64,
    uv_pitch: u64,
}

/// One pool slot: the surface's dma-buf export plus a free flag. The export
/// keeps the backing `VkImage`s / `VkDeviceMemory` alive (so the fd stays
/// valid); `free` is `true` when the slot can be acquired.
struct Slot {
    export: SurfaceExport,
    free: Arc<AtomicBool>,
}

/// A fixed pool of NV12 dma-buf surfaces for the zero-copy NVDEC decoder.
///
/// Owns the Vulkan device + the `POOL_SIZE` surfaces' images/memory.
/// [`Self::acquire`] hands out a free surface; the returned [`PooledSurface`]
/// releases the slot on drop.
pub struct NvdecSurfacePool {
    /// Owns the Vulkan device/instance and all surface images + memory. Drops
    /// last (declaration order) so the images/memory free before the device.
    vk: Arc<VulkanDevice>,
    /// Per-slot `VkImage`s + `VkDeviceMemory`, freed on pool drop.
    images: Vec<SurfaceImages>,
    slots: Vec<Slot>,
    width: u32,
    height: u32,
}

// SAFETY: the Vulkan handles are only ever touched under `&mut self` /
// pool-internal serialisation (the decoder is single-threaded); the
// `Arc<AtomicBool>` free flags are `Send + Sync`; `OwnedFd` is `Send + Sync`.
// The explicit impls document the move-between-threads contract the decoder's
// own `unsafe impl Send` relies on (ash handles are `!Send` by default).
unsafe impl Send for NvdecSurfacePool {}
unsafe impl Sync for NvdecSurfacePool {}

impl NvdecSurfacePool {
    /// Build the pool: create a Vulkan device on the NVIDIA GPU with the
    /// dma-buf external-memory extensions and allocate `POOL_SIZE` NV12
    /// surfaces sized `width`×`height`.
    ///
    /// Returns `NoHardwareCodec` if no NVIDIA Vulkan device with the required
    /// extensions exists or an allocation fails — the decoder maps that to a
    /// clean failure rather than silently corrupting frames.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let vk = Arc::new(VulkanDevice::new_nvidia()?);

        let mut images = Vec::with_capacity(POOL_SIZE);
        let mut slots = Vec::with_capacity(POOL_SIZE);
        for i in 0..POOL_SIZE {
            let (surface_images, export) = vk.export_nv12_surface(width, height).map_err(|e| {
                CodecError::NoHardwareCodec(format!(
                    "NVDEC surface pool: NV12 surface {i} allocation failed: {e}"
                ))
            })?;
            images.push(surface_images);
            slots.push(Slot {
                export,
                free: Arc::new(AtomicBool::new(true)),
            });
        }

        Ok(Self {
            vk,
            images,
            slots,
            width,
            height,
        })
    }

    /// The dimensions the pool's surfaces were allocated for. The decoder
    /// rebuilds the pool when the decoded frame's dims change.
    pub fn dims(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Acquire a free surface, marking its slot in-use. Returns `None` when
    /// every surface is still held by the renderer — the caller reports a
    /// clean error rather than overwriting a surface in use.
    ///
    /// The returned [`PooledSurface`] exposes the dma-buf fields (a dup'd fd,
    /// offsets, strides) and frees the slot on drop.
    pub fn acquire(&self) -> Option<PooledSurface> {
        for slot in &self.slots {
            // Claim the slot atomically: only the thread that flips true→false
            // owns it. (The decoder is single-threaded, but the
            // compare-exchange keeps the free flag the single source of truth
            // shared with the renderer-side release guard.)
            if slot
                .free
                .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let export = &slot.export;
                // Dup the fd so the surface keeps its own copy (still owned by
                // the export, kept alive by the pool) while the caller hands a
                // duplicate to the EGL import and another to the output frame.
                let fd = match export.fd.try_clone() {
                    Ok(fd) => fd,
                    Err(_) => {
                        slot.free.store(true, Ordering::Release);
                        return None;
                    }
                };
                return Some(PooledSurface {
                    fd,
                    size: export.size,
                    modifier: export.modifier,
                    y_offset: export.y_offset,
                    y_pitch: export.y_pitch,
                    uv_offset: export.uv_offset,
                    uv_pitch: export.uv_pitch,
                    width: self.width,
                    height: self.height,
                    release: SlotGuard {
                        free: slot.free.clone(),
                    },
                });
            }
        }
        None
    }
}

impl Drop for NvdecSurfacePool {
    fn drop(&mut self) {
        // Free every surface's images + memory before the device drops (the
        // `Arc<VulkanDevice>` Drop destroys instance/device last).
        // SAFETY: each handle was created on `self.vk.device`, owned uniquely
        // by this pool, and is destroyed exactly once here.
        unsafe {
            let device = &self.vk.device;
            for img in &self.images {
                device.destroy_image(img.uv_image, None);
                device.destroy_image(img.y_image, None);
                device.free_memory(img.memory, None);
            }
        }
    }
}

/// Per-surface Vulkan resources kept alive for the pool's lifetime.
struct SurfaceImages {
    y_image: vk::Image,
    uv_image: vk::Image,
    memory: vk::DeviceMemory,
}

/// Releases a pool slot back to the free list on drop. Held by the decoder's
/// `GpuFrameGuard` so the slot stays reserved until the renderer drops the
/// frame.
pub struct SlotGuard {
    free: Arc<AtomicBool>,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        self.free.store(true, Ordering::Release);
    }
}

/// A surface checked out of the pool. Carries the dma-buf descriptor fields
/// the decoder needs to (a) EGL-import the surface into CUDA and (b) build
/// the output [`crate::DmaBufFrame`] for the renderer. The `release` guard
/// frees the slot when this — or the `GpuFrameGuard` it's moved into — drops.
pub struct PooledSurface {
    /// A dup of the surface's dma-buf fd. The decoder uses it to import the
    /// surface into CUDA; the output frame gets its own fresh dup.
    pub fd: OwnedFd,
    pub size: u64,
    pub modifier: u64,
    pub y_offset: u64,
    pub y_pitch: u64,
    pub uv_offset: u64,
    pub uv_pitch: u64,
    pub width: u32,
    pub height: u32,
    /// Returns the slot to the pool's free list on drop. Moved into the
    /// decoder's `GpuFrameGuard` so the slot stays reserved while the renderer
    /// reads the surface.
    pub release: SlotGuard,
}

/// Minimal Vulkan instance + device on the NVIDIA GPU, with the extensions
/// the NV12 dma-buf export needs. Owns the `ash::Instance` / `ash::Device`
/// and is dropped last so all surface resources free first.
struct VulkanDevice {
    // `Entry` keeps the loaded Vulkan loader alive for the device's lifetime.
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    ext_mem_fd: ash::khr::external_memory_fd::Device,
    modifier_ext: ash::ext::image_drm_format_modifier::Device,
}

impl VulkanDevice {
    /// Create a headless Vulkan device on the first NVIDIA physical device
    /// that supports the dma-buf external-memory + DRM-format-modifier
    /// extensions. No surface, no swapchain — the pool only allocates
    /// exportable images.
    fn new_nvidia() -> Result<Self> {
        // SAFETY: `ash::Entry::load` dlopens the system Vulkan loader; the
        // handles it returns stay valid while `_entry` is held.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| {
            CodecError::NoHardwareCodec(format!("NVDEC surface pool: Vulkan loader: {e}"))
        })?;

        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        // SAFETY: `create_info` is a well-formed instance descriptor with no
        // enabled layers/extensions; the returned instance is destroyed in Drop.
        let instance = unsafe { entry.create_instance(&create_info, None) }.map_err(|e| {
            CodecError::NoHardwareCodec(format!("NVDEC surface pool: vkCreateInstance: {e}"))
        })?;

        // Pick the NVIDIA physical device. SAFETY: `instance` is live.
        let physical_device = unsafe {
            let devices = instance.enumerate_physical_devices().map_err(|e| {
                instance.destroy_instance(None);
                CodecError::NoHardwareCodec(format!(
                    "NVDEC surface pool: enumerate_physical_devices: {e}"
                ))
            })?;
            let chosen = devices.into_iter().find(|&pd| {
                instance.get_physical_device_properties(pd).vendor_id == VENDOR_NVIDIA
            });
            match chosen {
                Some(pd) => pd,
                None => {
                    instance.destroy_instance(None);
                    return Err(CodecError::NoHardwareCodec(
                        "NVDEC surface pool: no NVIDIA Vulkan physical device found \
                         (the surface pool must allocate on the same GPU as CUDA device 0)"
                            .to_string(),
                    ));
                }
            }
        };

        // Require the extensions the dma-buf export needs: external-memory fd,
        // dma-buf handle type, and DRM-format-modifier images. Their other
        // dependencies (`sampler_ycbcr_conversion`, `image_format_list`,
        // `bind_memory2`, `get_physical_device_properties2`, base
        // `external_memory`) are all promoted to Vulkan 1.1 core, which the
        // instance requests — listing them here as device extensions would make
        // vkCreateDevice fail EXTENSION_NOT_PRESENT on a 1.1 driver.
        let device_exts = [
            ash::khr::external_memory_fd::NAME.as_ptr(),
            ash::ext::external_memory_dma_buf::NAME.as_ptr(),
            ash::ext::image_drm_format_modifier::NAME.as_ptr(),
        ];
        // One queue (graphics/transfer family 0) — required to create a device,
        // unused otherwise (the pool allocates memory, runs no commands).
        let queue_priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(0)
            .queue_priorities(&queue_priorities)];
        let device_create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_exts);

        // SAFETY: `physical_device` is from `instance`; the extension list is
        // valid and required for the export below. On failure we tear down the
        // instance to avoid a leak.
        let device = match unsafe {
            instance.create_device(physical_device, &device_create, None)
        } {
            Ok(d) => d,
            Err(e) => {
                // SAFETY: instance is live and otherwise unreferenced.
                unsafe { instance.destroy_instance(None) };
                return Err(CodecError::NoHardwareCodec(format!(
                    "NVDEC surface pool: vkCreateDevice (NVIDIA, dma-buf exts): {e}. \
                     The Vulkan driver may lack VK_EXT_external_memory_dma_buf / \
                     VK_EXT_image_drm_format_modifier."
                )));
            }
        };

        let ext_mem_fd = ash::khr::external_memory_fd::Device::new(&instance, &device);
        let modifier_ext = ash::ext::image_drm_format_modifier::Device::new(&instance, &device);

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            ext_mem_fd,
            modifier_ext,
        })
    }

    /// Allocate one NV12 surface (Y as R8 + UV as R8G8) in a single shared
    /// `VkDeviceMemory` and export it as one dma-buf fd. The Y and UV images
    /// live at distinct offsets in the same allocation — the layout the
    /// renderer's `build_nv12_dmabuf_frame` import and EGL→CUDA registration
    /// both expect.
    ///
    /// Ported from `tether_gpuconvert::shared_nv12::export_nv12_shared_dmabuf`,
    /// keeping its load-bearing 64-byte luma-pitch / 16-row alignment. The
    /// caller owns the returned images + memory (freed on pool drop) and the
    /// dma-buf fd (freed when the last dup drops).
    //
    // Vulkan image dims / offsets are u32/u64 at the ABI; the only casts are
    // the aligned-dimension `next_multiple_of` results, which fit by construction.
    fn export_nv12_surface(
        &self,
        width: u32,
        height: u32,
    ) -> std::result::Result<(SurfaceImages, SurfaceExport), String> {
        let aligned_w = width.next_multiple_of(LUMA_STRIDE_ALIGN);
        let aligned_h = height.next_multiple_of(HEIGHT_ALIGN);
        let chroma_w = aligned_w.div_ceil(2);
        let chroma_h = aligned_h.div_ceil(2);

        let device = &self.device;
        let modifiers = [DRM_FORMAT_MOD_LINEAR];

        let create_image = |format: vk::Format, w: u32, h: u32| -> std::result::Result<vk::Image, String> {
            let mut ext_mem_create = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let mut modifier_info = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
                .drm_format_modifiers(&modifiers);
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                // The decoder writes via CUDA (not Vulkan); TRANSFER_DST is the
                // honest usage for an externally-filled image.
                .usage(vk::ImageUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut ext_mem_create)
                .push_next(&mut modifier_info);
            // SAFETY: `info` is a well-formed external/modifier image descriptor;
            // the returned image is destroyed by the caller on error / pool drop.
            unsafe { device.create_image(&info, None) }
                .map_err(|e| format!("vkCreateImage (NV12 shared): {e}"))
        };

        let y_image = create_image(vk::Format::R8_UNORM, aligned_w, aligned_h)?;
        let uv_image = match create_image(vk::Format::R8G8_UNORM, chroma_w, chroma_h) {
            Ok(i) => i,
            Err(e) => {
                // SAFETY: y_image is live, not yet exported; destroy it.
                unsafe { device.destroy_image(y_image, None) };
                return Err(e);
            }
        };

        // From here, any error must destroy both images (+ memory if allocated).
        let built = (|| -> std::result::Result<(SurfaceImages, SurfaceExport), String> {
            // SAFETY: both images are live, created on `device`.
            let (y_req, uv_req) = unsafe {
                (
                    device.get_image_memory_requirements(y_image),
                    device.get_image_memory_requirements(uv_image),
                )
            };
            let combined_bits = y_req.memory_type_bits & uv_req.memory_type_bits;
            if combined_bits == 0 {
                return Err("no common memory type for Y+UV images".to_string());
            }
            // SAFETY: `physical_device` is from `self.instance`.
            let mem_props = unsafe {
                self.instance
                    .get_physical_device_memory_properties(self.physical_device)
            };
            let mem_type_index = find_memory_type(
                &mem_props,
                combined_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .or_else(|| find_memory_type(&mem_props, combined_bits, vk::MemoryPropertyFlags::empty()))
            .ok_or_else(|| "no suitable memory type".to_string())?;

            // UV bind offset spans the Y image and satisfies both alignments.
            let uv_align = uv_req.alignment.max(y_req.alignment);
            let uv_bind_offset = align_up(y_req.size, uv_align);
            let total_size = uv_bind_offset + uv_req.size;

            let mut export_alloc = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(total_size)
                .memory_type_index(mem_type_index)
                .push_next(&mut export_alloc);
            // SAFETY: exportable allocation of `total_size`; freed by the caller.
            let memory = unsafe { device.allocate_memory(&alloc_info, None) }
                .map_err(|e| format!("vkAllocateMemory (NV12 shared): {e}"))?;

            let finish = (|| -> std::result::Result<SurfaceExport, String> {
                // SAFETY: both images + `memory` are live; offsets satisfy the
                // images' alignments (Y at 0, UV at `uv_bind_offset`).
                unsafe {
                    device
                        .bind_image_memory(y_image, memory, 0)
                        .map_err(|e| format!("vkBindImageMemory (Y): {e}"))?;
                    device
                        .bind_image_memory(uv_image, memory, uv_bind_offset)
                        .map_err(|e| format!("vkBindImageMemory (UV): {e}"))?;
                }

                let fd_info = vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                // SAFETY: `memory` was allocated exportable as DMA_BUF_EXT.
                let raw_fd = unsafe { self.ext_mem_fd.get_memory_fd(&fd_info) }
                    .map_err(|e| format!("vkGetMemoryFdKHR (NV12 shared): {e}"))?;
                // SAFETY: raw_fd is a freshly-returned open fd we own.
                let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

                // Confirm the driver picked LINEAR (we asked via a 1-element list).
                let mut y_mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
                // SAFETY: y_image is a live DRM-modifier image on `device`.
                unsafe {
                    self.modifier_ext
                        .get_image_drm_format_modifier_properties(y_image, &mut y_mod_props)
                }
                .map_err(|e| format!("vkGetImageDrmFormatModifierPropertiesEXT (Y): {e}"))?;

                // Per-plane subresource layouts give the row pitches. For a
                // LINEAR single-plane image plane-0 sits at offset 0 of its
                // bound memory, so the surface offsets are the bind offsets.
                let y_subres = vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT);
                let uv_subres = vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT);
                // SAFETY: both images are live LINEAR-modifier images.
                let (y_layout, uv_layout) = unsafe {
                    (
                        device.get_image_subresource_layout(y_image, y_subres),
                        device.get_image_subresource_layout(uv_image, uv_subres),
                    )
                };

                Ok(SurfaceExport {
                    fd,
                    size: total_size,
                    modifier: y_mod_props.drm_format_modifier,
                    y_offset: y_layout.offset,
                    y_pitch: y_layout.row_pitch,
                    uv_offset: uv_bind_offset + uv_layout.offset,
                    uv_pitch: uv_layout.row_pitch,
                })
            })();

            match finish {
                Ok(export) => Ok((
                    SurfaceImages {
                        y_image,
                        uv_image,
                        memory,
                    },
                    export,
                )),
                Err(e) => {
                    // SAFETY: memory + images live, not yet handed out; free them.
                    unsafe { device.free_memory(memory, None) };
                    Err(e)
                }
            }
        })();

        if built.is_err() {
            // SAFETY: both images live; memory (if it was allocated) was freed
            // in the inner closure's error arm before returning Err.
            unsafe {
                device.destroy_image(uv_image, None);
                device.destroy_image(y_image, None);
            }
        }
        built
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // SAFETY: all surface images/memory are freed by the pool's Drop
        // (which runs first — the pool owns the `Arc<VulkanDevice>` and drops
        // its `images`/`slots` before this Arc's last ref). Destroy device then
        // instance, each exactly once.
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// First memory type satisfying `type_bits` and `flags`. Ported from
/// `tether_gpuconvert::dmabuf_export::find_memory_type`.
fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for (i, mem_ty) in mem_props.memory_types_as_slice().iter().enumerate() {
        let bit = 1u32 << i;
        if type_bits & bit != 0 && mem_ty.property_flags & flags == flags {
            // i < memory_type_count <= VK_MAX_MEMORY_TYPES (32), fits u32.
            #[allow(clippy::cast_possible_truncation)]
            return Some(i as u32);
        }
    }
    None
}

/// Round `value` up to a multiple of `align` (a power of two). Ported from
/// `tether_gpuconvert::dmabuf_export::align_up`.
fn align_up(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two(), "align must be power of two");
    (value + align - 1) & !(align - 1)
}
