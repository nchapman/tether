//! Fixed pool of dma-buf surfaces the zero-copy NVDEC decoder fills and hands
//! to the renderer (GH #16, D2).
//!
//! NVDEC decodes into CUDA surfaces; the renderer imports a dma-buf
//! (`GpuFrameSource::DmaBuf`) directly into wgpu — NV12 for 8-bit 4:2:0, P010
//! for 10-bit 4:2:0, and diagnostic YUV444P for 8-bit 4:4:4. So the decoder
//! needs a dma-buf it can write the decoded pixels into. This pool allocates a
//! small fixed set of surfaces and lends them out one at a time. A surface
//! stays reserved (its slot marked in-use) until the renderer drops the frame,
//! so the decoder can't overwrite a surface the renderer is still reading;
//! [`PooledSurface`]'s slot guard returns the slot on drop.
//!
//! **Why raw Vulkan (`ash`), not `tether-gpuconvert`'s exporter.** The
//! obvious move is to reuse tether-gpuconvert's shared dma-buf exporters, but
//! tether-codec depending on tether-gpuconvert in production forms a
//! normal-dependency cycle (tether-gpuconvert already depends on tether-codec
//! on macOS for IOSurface interop; cargo's resolver-2 evaluates the graph
//! cfg-merged and rejects the cross-platform mutual prod dependency). So this
//! module reimplements the minimal shared-allocation export directly on `ash`.
//! It mirrors `shared_nv12.rs` / `shared_p010.rs` / `shared_yuv444p.rs` and —
//! load-bearing — keeps the 64-byte Y-row-pitch + 16-row-height alignment
//! (Intel iHD's importer mis-reads non-aligned widths; harmless on NVIDIA/RADV
//! but kept identical so the descriptor is universally importable; see the
//! gpuconvert comment).
//!
//! Multi-GPU pinning: when the host has pinned a producer GPU, the pool matches
//! that Vulkan `deviceUUID` before allocating surfaces. Unpinned keeps the old
//! single-GPU/default-driver behavior by taking the first NVIDIA Vulkan device.

use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ash::vk;

use crate::{CodecError, Result};

/// Number of surfaces in the pool. NVDEC reorder depth plus a couple in flight
/// to the renderer; matches the encoder's small-pool sizing rationale (a
/// handful is plenty when the renderer consumes promptly).
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

/// On-device pixel layout of the pooled surfaces. NVDEC decodes 8-bit 4:2:0
/// into NV12 and 10-bit 4:2:0 into P010; the pool allocates the matching
/// two-plane dma-buf (same shared-allocation layout, different per-sample
/// width) so the decoded planes copy in and the renderer imports zero-copy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NvdecSurfaceFormat {
    /// 8-bit 4:2:0 — Y as `R8`, UV as `R8G8` (fourcc `NV12`).
    Nv12,
    /// 10-bit 4:2:0 — Y as `R16`, UV as `R16G16` (fourcc `P010`).
    P010,
    /// 8-bit 4:4:4 — three full-resolution R8 planes (fourcc `YU24`).
    Yuv444p,
}

impl NvdecSurfaceFormat {
    /// Vulkan format for the Y (luma) plane image.
    fn y_vk_format(self) -> vk::Format {
        match self {
            Self::Nv12 => vk::Format::R8_UNORM,
            Self::P010 => vk::Format::R16_UNORM,
            Self::Yuv444p => vk::Format::R8_UNORM,
        }
    }

    /// Vulkan format for the interleaved UV (chroma) plane image.
    fn uv_vk_format(self) -> vk::Format {
        match self {
            Self::Nv12 => vk::Format::R8G8_UNORM,
            Self::P010 => vk::Format::R16G16_UNORM,
            Self::Yuv444p => vk::Format::R8_UNORM,
        }
    }

    fn v_vk_format(self) -> Option<vk::Format> {
        match self {
            Self::Yuv444p => Some(vk::Format::R8_UNORM),
            Self::Nv12 | Self::P010 => None,
        }
    }

    /// Bytes per luma sample — the per-row byte multiplier for the device copy.
    /// (UV rows span the same byte width: `w/2` CbCr pairs × 2 components ×
    /// `bytes_per_sample` = `w × bytes_per_sample`, equal to the Y row.)
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::Nv12 | Self::Yuv444p => 1,
            Self::P010 => 2,
        }
    }

    pub fn plane_count(self) -> usize {
        match self {
            Self::Nv12 | Self::P010 => 2,
            Self::Yuv444p => 3,
        }
    }
}

/// The dma-buf descriptor fields for one pooled surface, as produced by the
/// shared-allocation export. Mirrors tether-gpuconvert's shared-export
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
    v_offset: u64,
    v_pitch: u64,
}

/// One pool slot: the surface's dma-buf export, a free flag, and the shared
/// `Arc<SurfaceBacking>` that keeps the surface's `VkImage`s / `VkDeviceMemory`
/// alive (so the fd stays valid). `free` is `true` when the slot can be
/// acquired.
struct Slot {
    export: SurfaceExport,
    free: Arc<AtomicBool>,
    /// The slot's Vulkan images/memory. Shared (via `Arc`) with any
    /// `PooledSurface` handed out from this slot so the backing — and the
    /// exported dma-buf fd that aliases it — outlives a pool rebuild that drops
    /// the old pool while the renderer still holds the surface.
    backing: Arc<SurfaceBacking>,
}

/// A fixed pool of dma-buf surfaces for the zero-copy NVDEC decoder.
///
/// Owns the Vulkan device + the `POOL_SIZE` surfaces' images/memory.
/// [`Self::acquire`] hands out a free surface; the returned [`PooledSurface`]
/// releases the slot on drop.
pub struct NvdecSurfacePool {
    /// Each slot's `SurfaceBacking` holds an `Arc<VulkanDevice>` clone, so the
    /// device stays alive as long as any slot exists — and, because a
    /// handed-out surface holds the same `Arc`, also as long as any surface
    /// outlives a pool rebuild. The pool needs no separate device handle.
    slots: Vec<Slot>,
    format: NvdecSurfaceFormat,
    width: u32,
    height: u32,
}

// SAFETY: the pool is owned by `NvdecDecoder` (itself only `Send`) and its
// Vulkan handles are only ever touched under `&mut self` on the single decode
// thread; the `Arc<AtomicBool>` free flags and `OwnedFd` are already `Send`.
// Only `Send` is asserted — moving the pool between threads is the decoder's
// contract (ash handles are `!Send` by default). `Sync` is deliberately NOT
// implemented: nothing shares `&NvdecSurfacePool` across threads, so asserting
// it would only widen the unsafe surface for no caller.
unsafe impl Send for NvdecSurfacePool {}

impl NvdecSurfacePool {
    /// Build the pool: create a Vulkan device on the NVIDIA GPU with the
    /// dma-buf external-memory extensions and allocate `POOL_SIZE` surfaces of
    /// `format` sized `width`×`height`.
    ///
    /// Returns `NoHardwareCodec` if no NVIDIA Vulkan device with the required
    /// extensions exists or an allocation fails — the decoder maps that to a
    /// clean failure rather than silently corrupting frames.
    pub fn new(format: NvdecSurfaceFormat, width: u32, height: u32) -> Result<Self> {
        let vk = Arc::new(VulkanDevice::new_nvidia()?);

        let mut slots = Vec::with_capacity(POOL_SIZE);
        for i in 0..POOL_SIZE {
            let (surface_images, export) =
                vk.export_surface(format, width, height).map_err(|e| {
                    CodecError::NoHardwareCodec(format!(
                        "NVDEC surface pool: {format:?} surface {i} allocation failed: {e}"
                    ))
                })?;
            slots.push(Slot {
                export,
                free: Arc::new(AtomicBool::new(true)),
                backing: Arc::new(SurfaceBacking {
                    vk: vk.clone(),
                    images: surface_images,
                }),
            });
        }

        Ok(Self {
            slots,
            format,
            width,
            height,
        })
    }

    /// The dimensions the pool's surfaces were allocated for. The decoder
    /// rebuilds the pool when the decoded frame's dims change.
    pub fn dims(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The pixel format the pool's surfaces were allocated for. The decoder
    /// rebuilds the pool when the decoded frame's layout changes (8↔10-bit).
    pub fn format(&self) -> NvdecSurfaceFormat {
        self.format
    }

    /// Test-only: the GPU the pool allocated its surfaces on, by Vulkan
    /// `deviceUUID`. All slots share one `VulkanDevice`, so any slot answers.
    /// Used to assert pinning routed the pool to the pinned GPU.
    #[cfg(test)]
    pub(crate) fn vulkan_device_uuid(&self) -> [u8; 16] {
        self.slots[0].backing.vk.device_uuid()
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
                    format: self.format,
                    size: export.size,
                    modifier: export.modifier,
                    y_offset: export.y_offset,
                    y_pitch: export.y_pitch,
                    uv_offset: export.uv_offset,
                    uv_pitch: export.uv_pitch,
                    v_offset: export.v_offset,
                    v_pitch: export.v_pitch,
                    width: self.width,
                    height: self.height,
                    release: SlotGuard {
                        free: slot.free.clone(),
                        // Hold a backing ref so the images/memory (and the
                        // exported fd this surface dup'd) outlive a pool rebuild
                        // that drops the pool while this surface is still out.
                        _backing: slot.backing.clone(),
                    },
                });
            }
        }
        None
    }
}

// No explicit `Drop` for the pool: each slot's `Arc<SurfaceBacking>` frees its
// own images/memory when the last reference (the slot or a handed-out surface)
// drops, and the `Arc<VulkanDevice>` inside each backing keeps the device alive
// until then.

/// Per-surface Vulkan resources, shared via `Arc` between the pool's slot and
/// any handed-out [`PooledSurface`]. Freeing is deferred to the last `Arc`
/// drop, so a pool rebuilt mid-frame can't free memory (or invalidate the
/// exported dma-buf fd) out from under a surface the renderer still holds.
struct SurfaceBacking {
    /// Keeps the device alive until this backing's images/memory are freed.
    vk: Arc<VulkanDevice>,
    images: SurfaceImages,
}

impl Drop for SurfaceBacking {
    fn drop(&mut self) {
        // SAFETY: each handle was created on `self.vk.device` and is owned
        // uniquely by this backing (the `Arc` guarantees exactly one `Drop`),
        // so it is destroyed exactly once, before the device.
        unsafe {
            let device = &self.vk.device;
            if let Some(v_image) = self.images.v_image {
                device.destroy_image(v_image, None);
            }
            device.destroy_image(self.images.uv_image, None);
            device.destroy_image(self.images.y_image, None);
            device.free_memory(self.images.memory, None);
        }
    }
}

/// Per-surface Vulkan handle triple owned by a [`SurfaceBacking`].
struct SurfaceImages {
    y_image: vk::Image,
    uv_image: vk::Image,
    v_image: Option<vk::Image>,
    memory: vk::DeviceMemory,
}

/// Releases a pool slot back to the free list on drop. Held by the decoder's
/// `GpuFrameGuard` so the slot stays reserved until the renderer drops the
/// frame.
pub struct SlotGuard {
    free: Arc<AtomicBool>,
    /// Keeps the surface's Vulkan images/memory (and the exported dma-buf fd
    /// aliasing them) alive while the renderer holds this surface, even if the
    /// pool was rebuilt and dropped in the meantime. Never read — its `Drop`
    /// (via the shared `Arc`) does the work.
    _backing: Arc<SurfaceBacking>,
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
    /// The surface's pixel layout — selects the dma-buf fourcc and the
    /// per-sample byte width of the device copy.
    pub format: NvdecSurfaceFormat,
    pub size: u64,
    pub modifier: u64,
    pub y_offset: u64,
    pub y_pitch: u64,
    pub uv_offset: u64,
    pub uv_pitch: u64,
    pub v_offset: u64,
    pub v_pitch: u64,
    pub width: u32,
    pub height: u32,
    /// Returns the slot to the pool's free list on drop. Moved into the
    /// decoder's `GpuFrameGuard` so the slot stays reserved while the renderer
    /// reads the surface.
    pub release: SlotGuard,
}

impl PooledSurface {
    /// Build the `DmaBufFrame` describing this surface from a (dup'd) fd.
    /// One source of truth for both the EGL import descriptor and the renderer
    /// output frame, so they can't disagree on fourcc/plane formats.
    pub fn build_dmabuf_frame(&self, fd: OwnedFd) -> crate::DmaBufFrame {
        match self.format {
            NvdecSurfaceFormat::Nv12 => crate::build_nv12_dmabuf_frame(
                fd,
                self.size,
                self.modifier,
                self.y_offset,
                self.y_pitch,
                self.uv_offset,
                self.uv_pitch,
            ),
            NvdecSurfaceFormat::P010 => crate::build_p010_dmabuf_frame(
                fd,
                self.size,
                self.modifier,
                self.y_offset,
                self.y_pitch,
                self.uv_offset,
                self.uv_pitch,
            ),
            NvdecSurfaceFormat::Yuv444p => crate::build_yuv444p_dmabuf_frame(
                fd,
                self.size,
                self.modifier,
                self.y_offset,
                self.y_pitch,
                self.uv_offset,
                self.uv_pitch,
                self.v_offset,
                self.v_pitch,
            ),
        }
    }
}

/// Minimal Vulkan instance + device on the NVIDIA GPU, with the extensions
/// the dma-buf export needs. Owns the `ash::Instance` / `ash::Device`
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

/// Read a physical device's `VkPhysicalDeviceIDProperties::deviceUUID` (core
/// Vulkan 1.1). On NVIDIA this equals the CUDA `cuDeviceGetUuid` value and the
/// wgpu producer's `deviceUUID` for the same GPU, so it correlates the surface
/// pool's allocation device with the host's pinned GPU.
///
/// # Safety
/// `instance` must be live and `physical` one of the devices it enumerated.
unsafe fn physical_device_uuid(instance: &ash::Instance, physical: vk::PhysicalDevice) -> [u8; 16] {
    let mut id_props = vk::PhysicalDeviceIDProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id_props);
    // SAFETY: `physical` belongs to `instance`; `props2` is a well-formed
    // properties2 chain with the ID-properties struct linked via push_next.
    unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
    id_props.device_uuid
}

impl VulkanDevice {
    /// Test-only: the GPU this device was created on, by Vulkan `deviceUUID`.
    #[cfg(test)]
    fn device_uuid(&self) -> [u8; 16] {
        // SAFETY: `instance` + `physical_device` are live for `self`'s lifetime.
        unsafe { physical_device_uuid(&self.instance, self.physical_device) }
    }

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

        // Pick the NVIDIA physical device the decoder's CUDA context lives on.
        // When the host pinned a GPU (`nvenc::pin_gpu_uuid`) we must allocate
        // surfaces on that exact GPU by matching the Vulkan `deviceUUID` to the
        // pin — the decoded planes are copied into this surface inside the
        // decoder's CUDA context, then EGL-imported back, so a surface on the
        // wrong GPU would fault. Unpinned: the first NVIDIA device (single-GPU
        // behavior). SAFETY: `instance` is live.
        let pinned = crate::nvenc::gpu_pin::pinned_uuid();
        let physical_device = unsafe {
            let devices = instance.enumerate_physical_devices().map_err(|e| {
                instance.destroy_instance(None);
                CodecError::NoHardwareCodec(format!(
                    "NVDEC surface pool: enumerate_physical_devices: {e}"
                ))
            })?;
            let chosen = devices.into_iter().find(|&pd| {
                if instance.get_physical_device_properties(pd).vendor_id != VENDOR_NVIDIA {
                    return false;
                }
                match pinned {
                    Some(target) => physical_device_uuid(&instance, pd) == target,
                    None => true,
                }
            });
            match chosen {
                Some(pd) => pd,
                None => {
                    instance.destroy_instance(None);
                    let detail = match pinned {
                        Some(uuid) => format!(
                            "no NVIDIA Vulkan device matches the pinned GPU UUID {uuid:02x?} \
                             (the surface pool must allocate on the decoder's CUDA GPU)"
                        ),
                        None => "no NVIDIA Vulkan physical device found".to_string(),
                    };
                    return Err(CodecError::NoHardwareCodec(format!(
                        "NVDEC surface pool: {detail}"
                    )));
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
        // vkCreateDevice requires one queue, but the pool runs no commands (it
        // only allocates + exports memory), so any valid family works. Pick the
        // first non-empty one rather than assuming family 0 exists — on NVIDIA
        // family 0 is the universal graphics/transfer family, but the spec
        // doesn't guarantee its index.
        // SAFETY: `physical_device` is from `instance`.
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family_index = queue_families
            .iter()
            .position(|f| f.queue_count > 0)
            .and_then(|i| u32::try_from(i).ok())
            .ok_or_else(|| {
                // SAFETY: instance is live and otherwise unreferenced.
                unsafe { instance.destroy_instance(None) };
                CodecError::NoHardwareCodec(
                    "NVDEC surface pool: NVIDIA Vulkan device exposes no queue families"
                        .to_string(),
                )
            })?;
        let queue_priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities)];
        let device_create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_exts);

        // SAFETY: `physical_device` is from `instance`; the extension list is
        // valid and required for the export below. On failure we tear down the
        // instance to avoid a leak.
        let device = match unsafe { instance.create_device(physical_device, &device_create, None) }
        {
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

    /// Allocate one surface of `format` in a single shared `VkDeviceMemory`
    /// and export it as one dma-buf fd. NV12/P010 use Y + interleaved UV
    /// images; YUV444P adds a full-resolution V image. The images live at
    /// distinct offsets in the same allocation — the layout the renderer's
    /// `build_*_dmabuf_frame` import and EGL→CUDA registration both expect.
    ///
    /// Ported from `tether_gpuconvert::shared_nv12::export_nv12_shared_dmabuf`,
    /// keeping its load-bearing 64-byte luma-pitch / 16-row alignment. The
    /// caller owns the returned images + memory (freed on pool drop) and the
    /// dma-buf fd (freed when the last dup drops).
    //
    // Vulkan image dims / offsets are u32/u64 at the ABI; the only casts are
    // the aligned-dimension `next_multiple_of` results, which fit by construction.
    fn export_surface(
        &self,
        format: NvdecSurfaceFormat,
        width: u32,
        height: u32,
    ) -> std::result::Result<(SurfaceImages, SurfaceExport), String> {
        let aligned_w = width.next_multiple_of(LUMA_STRIDE_ALIGN);
        let aligned_h = height.next_multiple_of(HEIGHT_ALIGN);
        let surface_label = match format {
            NvdecSurfaceFormat::Nv12 => "NV12",
            NvdecSurfaceFormat::P010 => "P010",
            NvdecSurfaceFormat::Yuv444p => "YUV444P",
        };
        let (chroma_w, chroma_h) = match format {
            NvdecSurfaceFormat::Nv12 | NvdecSurfaceFormat::P010 => {
                (aligned_w.div_ceil(2), aligned_h.div_ceil(2))
            }
            NvdecSurfaceFormat::Yuv444p => (aligned_w, aligned_h),
        };

        let device = &self.device;
        let modifiers = [DRM_FORMAT_MOD_LINEAR];

        let create_image =
            |format: vk::Format, w: u32, h: u32| -> std::result::Result<vk::Image, String> {
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
                    .map_err(|e| format!("vkCreateImage ({surface_label} shared): {e}"))
            };

        let y_image = create_image(format.y_vk_format(), aligned_w, aligned_h)?;
        let uv_image = match create_image(format.uv_vk_format(), chroma_w, chroma_h) {
            Ok(i) => i,
            Err(e) => {
                // SAFETY: y_image is live, not yet exported; destroy it.
                unsafe { device.destroy_image(y_image, None) };
                return Err(e);
            }
        };
        let v_image = match format.v_vk_format() {
            Some(v_format) => match create_image(v_format, aligned_w, aligned_h) {
                Ok(i) => Some(i),
                Err(e) => {
                    unsafe {
                        device.destroy_image(uv_image, None);
                        device.destroy_image(y_image, None);
                    }
                    return Err(e);
                }
            },
            None => None,
        };

        // From here, any error must destroy every image (+ memory if allocated).
        let built = (|| -> std::result::Result<(SurfaceImages, SurfaceExport), String> {
            // SAFETY: images are live, created on `device`.
            let (y_req, uv_req, v_req) = unsafe {
                (
                    device.get_image_memory_requirements(y_image),
                    device.get_image_memory_requirements(uv_image),
                    v_image.map(|image| device.get_image_memory_requirements(image)),
                )
            };
            let combined_bits = match v_req {
                Some(v_req) => {
                    y_req.memory_type_bits & uv_req.memory_type_bits & v_req.memory_type_bits
                }
                None => y_req.memory_type_bits & uv_req.memory_type_bits,
            };
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
            .or_else(|| {
                find_memory_type(&mem_props, combined_bits, vk::MemoryPropertyFlags::empty())
            })
            .ok_or_else(|| "no suitable memory type".to_string())?;

            // UV bind offset spans the Y image and satisfies both alignments.
            let uv_align = uv_req.alignment.max(y_req.alignment);
            let uv_bind_offset = align_up(y_req.size, uv_align);
            let (v_bind_offset, total_size) = match v_req {
                Some(v_req) => {
                    let v_align = uv_align.max(v_req.alignment);
                    let offset = align_up(uv_bind_offset.saturating_add(uv_req.size), v_align);
                    (offset, offset.saturating_add(v_req.size))
                }
                None => (0, uv_bind_offset.saturating_add(uv_req.size)),
            };
            // Saturating, consistent with `align_up`: a driver-reported size near
            // u64::MAX must not wrap to a small `total_size`, which would
            // under-allocate the memory and bind the UV plane past its end.
            // (`MAX_DECODE_DIM` keeps real inputs far from this, but the
            // arithmetic stays honest.)

            let mut export_alloc = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(total_size)
                .memory_type_index(mem_type_index)
                .push_next(&mut export_alloc);
            // SAFETY: exportable allocation of `total_size`; freed by the caller.
            let memory = unsafe { device.allocate_memory(&alloc_info, None) }
                .map_err(|e| format!("vkAllocateMemory ({surface_label} shared): {e}"))?;

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
                    if let Some(v_image) = v_image {
                        device
                            .bind_image_memory(v_image, memory, v_bind_offset)
                            .map_err(|e| format!("vkBindImageMemory (V): {e}"))?;
                    }
                }

                let fd_info = vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                // SAFETY: `memory` was allocated exportable as DMA_BUF_EXT.
                let raw_fd = unsafe { self.ext_mem_fd.get_memory_fd(&fd_info) }
                    .map_err(|e| format!("vkGetMemoryFdKHR ({surface_label} shared): {e}"))?;
                // SAFETY: raw_fd is a freshly-returned open fd we own.
                let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

                // Confirm the driver picked LINEAR (we asked via a 1-element
                // list) for BOTH images. The single SurfaceExport modifier and
                // the EGL import assume Y and UV share one modifier; a driver
                // that diverged would otherwise fail later inside eglCreateImage
                // with an opaque error.
                let mut y_mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
                let mut uv_mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
                let mut v_mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
                // SAFETY: both are live DRM-modifier images on `device`.
                unsafe {
                    self.modifier_ext
                        .get_image_drm_format_modifier_properties(y_image, &mut y_mod_props)
                        .map_err(|e| {
                            format!("vkGetImageDrmFormatModifierPropertiesEXT (Y): {e}")
                        })?;
                    self.modifier_ext
                        .get_image_drm_format_modifier_properties(uv_image, &mut uv_mod_props)
                        .map_err(|e| {
                            format!("vkGetImageDrmFormatModifierPropertiesEXT (UV): {e}")
                        })?;
                    if let Some(v_image) = v_image {
                        self.modifier_ext
                            .get_image_drm_format_modifier_properties(v_image, &mut v_mod_props)
                            .map_err(|e| {
                                format!("vkGetImageDrmFormatModifierPropertiesEXT (V): {e}")
                            })?;
                    }
                }
                if y_mod_props.drm_format_modifier != uv_mod_props.drm_format_modifier {
                    return Err(format!(
                        "surface Y/UV modifiers differ (Y 0x{:x}, UV 0x{:x}); \
                         single-modifier EGL import would fail",
                        y_mod_props.drm_format_modifier, uv_mod_props.drm_format_modifier
                    ));
                }
                if v_image.is_some()
                    && y_mod_props.drm_format_modifier != v_mod_props.drm_format_modifier
                {
                    return Err(format!(
                        "surface Y/V modifiers differ (Y 0x{:x}, V 0x{:x}); \
                         single-modifier EGL import would fail",
                        y_mod_props.drm_format_modifier, v_mod_props.drm_format_modifier
                    ));
                }

                // Per-plane subresource layouts give the row pitches. For a
                // LINEAR single-plane image plane-0 sits at offset 0 of its
                // bound memory, so the surface offsets are the bind offsets.
                let y_subres = vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT);
                let uv_subres = vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT);
                // SAFETY: both images are live LINEAR-modifier images.
                let (y_layout, uv_layout, v_layout) = unsafe {
                    (
                        device.get_image_subresource_layout(y_image, y_subres),
                        device.get_image_subresource_layout(uv_image, uv_subres),
                        v_image.map(|image| device.get_image_subresource_layout(image, uv_subres)),
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
                    v_offset: v_layout.map_or(0, |layout| v_bind_offset + layout.offset),
                    v_pitch: v_layout.map_or(0, |layout| layout.row_pitch),
                })
            })();

            match finish {
                Ok(export) => Ok((
                    SurfaceImages {
                        y_image,
                        uv_image,
                        v_image,
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
                if let Some(v_image) = v_image {
                    device.destroy_image(v_image, None);
                }
                device.destroy_image(uv_image, None);
                device.destroy_image(y_image, None);
            }
        }
        built
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // SAFETY: every surface's images/memory live in a `SurfaceBacking` that
        // holds its own `Arc<VulkanDevice>` clone, so this `Drop` (the last Arc
        // ref) runs only after all backings have freed their images/memory.
        // Destroy device then instance, each exactly once.
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
    // saturating, not `value + align - 1`: a driver-reported size/alignment near
    // u64::MAX would otherwise wrap in release. Capped dimensions keep real
    // inputs far from this, but the saturating add removes the foot-gun.
    value.saturating_add(align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn require_nvidia_gpu(test_name: &str) -> bool {
        if crate::nvenc::nvidia_gpu_present() {
            return true;
        }
        eprintln!("SKIP {test_name}: no NVIDIA GPU present");
        false
    }

    #[test]
    fn align_up_is_saturating_and_correct() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
        // Near-u64::MAX inputs saturate instead of wrapping to a small value:
        // u64::MAX rounded down to a 64-multiple (the low 6 bits cleared).
        assert_eq!(align_up(u64::MAX, 64), 0xFFFF_FFFF_FFFF_FFC0);
    }

    /// Exercise the pool's free-list contract on real hardware: every slot is
    /// handed out exactly once, exhaustion returns `None`, and releasing a
    /// surface (drop) returns its slot for reuse. Also sanity-checks the NV12
    /// surface descriptor (LINEAR modifier, plane pitches ≥ width). `#[ignore]`
    /// — needs an NVIDIA GPU + a Vulkan dma-buf device.
    #[test]
    #[ignore = "requires NVIDIA GPU + Vulkan dma-buf; run with: cargo test -p tether-codec -- --ignored"]
    fn acquire_exhausts_then_release_reuses_slots() {
        if !require_nvidia_gpu("acquire_exhausts_then_release_reuses_slots") {
            return;
        }

        let (w, h) = (256u32, 256u32);
        let pool =
            NvdecSurfacePool::new(NvdecSurfaceFormat::Nv12, w, h).expect("NV12 surface pool");
        assert_eq!(pool.dims(), (w, h));
        assert_eq!(pool.format(), NvdecSurfaceFormat::Nv12);

        // Drain every slot; each surface must be a well-formed NV12 descriptor.
        let mut held: Vec<PooledSurface> = Vec::new();
        for _ in 0..POOL_SIZE {
            let s = pool
                .acquire()
                .expect("a free slot while the pool is not exhausted");
            assert_eq!((s.width, s.height), (w, h));
            assert!(s.size > 0, "surface allocation has non-zero size");
            // We asked for a 1-element LINEAR modifier list, so the driver must
            // have picked LINEAR (0); a non-LINEAR modifier means the export
            // assumptions are broken.
            assert_eq!(s.modifier, 0, "NV12 surface must be DRM_FORMAT_MOD_LINEAR");
            // Y row spans w bytes; UV row spans w bytes (w/2 CbCr pairs × 2).
            // Alignment may make either larger, never smaller.
            assert!(
                s.y_pitch >= u64::from(w),
                "Y pitch {} < width {w}",
                s.y_pitch
            );
            assert!(
                s.uv_pitch >= u64::from(w),
                "UV pitch {} < width {w}",
                s.uv_pitch
            );
            assert!(s.uv_offset >= s.y_offset + s.y_pitch * u64::from(h));
            held.push(s);
        }

        // Pool is now exhausted.
        assert!(
            pool.acquire().is_none(),
            "acquire must return None when every slot is in use"
        );

        // Release one surface; its slot must become acquirable again.
        held.pop();
        assert!(
            pool.acquire().is_some(),
            "dropping a surface must return its slot to the free list"
        );
    }

    /// When the host pins a GPU, the surface pool must allocate on *that* GPU,
    /// not just "the first NVIDIA device". Pin to a real GPU and assert the
    /// pool's Vulkan device UUID matches. This is the decode-side twin of the
    /// EGL/CUDA pinning proofs in `nvenc::tests`. Sets the process GPU pin, so
    /// run it in isolation. `#[ignore]` — needs an NVIDIA GPU + Vulkan.
    #[test]
    #[ignore = "requires NVIDIA GPU + Vulkan; sets the process GPU pin (run in isolation); run with: cargo test -p tether-codec -- --ignored"]
    fn surface_pool_allocates_on_the_pinned_gpu() {
        use crate::nvenc::gpu_pin;

        if !require_nvidia_gpu("surface_pool_allocates_on_the_pinned_gpu") {
            return;
        }

        // Use whatever's already pinned (robust if a prior test pinned), else
        // pin to the first CUDA device so the test is self-contained.
        let target = gpu_pin::pinned_uuid().unwrap_or_else(|| {
            let first = crate::nvenc::cuda_device_uuids()
                .first()
                .map(|&(_, uuid)| uuid)
                .expect("at least one CUDA device on an NVIDIA host");
            crate::nvenc::pin_gpu_uuid(first);
            gpu_pin::pinned_uuid().expect("pin just set")
        });

        let pool = NvdecSurfacePool::new(NvdecSurfaceFormat::Nv12, 256, 256)
            .expect("build NVDEC surface pool on the pinned GPU");
        assert_eq!(
            pool.vulkan_device_uuid(),
            target,
            "surface pool allocated on a different GPU than the pinned one"
        );
    }
}
