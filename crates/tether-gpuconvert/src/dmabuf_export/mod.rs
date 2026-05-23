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

use std::os::fd::OwnedFd;

use ash::vk;

mod shared_nv12;
mod shared_p010;
mod shared_xv30;
mod shared_yuv444;
mod single;

#[cfg(test)]
mod tests;

pub use shared_nv12::{export_nv12_shared_dmabuf, SharedNv12Export};
pub use shared_p010::{export_p010_shared_dmabuf, SharedP010Export};
pub use shared_xv30::{export_xv30_shared_dmabuf, SharedXv30Export};
pub use shared_yuv444::{export_yuv444_shared_dmabuf, SharedYuv444Export};
pub use single::export_texture_as_dmabuf;

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

// ---------------------------------------------------------------------
// Shared Vulkan helpers (used by both single-plane and shared-NV12
// exporters). Kept here so both call sites pull from the same
// translation tables — see the warning on `wgpu_usage_to_vk`.

/// Align `value` up to the next multiple of `align`. `align` must be a
/// power of two (Vulkan requires it for memory alignment).
pub(super) fn align_up(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two(), "align must be power of two");
    (value + align - 1) & !(align - 1)
}

pub(super) fn find_memory_type(
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
/// equivalents. Small fixed set — single-plane Y/UV in 8 and 10 bit
/// flavours (NV12 / P010) plus BGRA8 for round-trip tests. New formats
/// add here.
pub(super) fn wgpu_format_to_vk(f: wgpu::TextureFormat) -> Result<vk::Format> {
    match f {
        wgpu::TextureFormat::R8Unorm => Ok(vk::Format::R8_UNORM),
        wgpu::TextureFormat::Rg8Unorm => Ok(vk::Format::R8G8_UNORM),
        wgpu::TextureFormat::R16Unorm => Ok(vk::Format::R16_UNORM),
        wgpu::TextureFormat::Rg16Unorm => Ok(vk::Format::R16G16_UNORM),
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
pub(super) fn wgpu_usage_to_vk(u: wgpu::TextureUsages) -> vk::ImageUsageFlags {
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
pub(super) fn wgpu_usage_to_hal_uses(u: wgpu::TextureUsages) -> wgpu::TextureUses {
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
