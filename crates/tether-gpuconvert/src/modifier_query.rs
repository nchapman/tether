//! Query which DRM format modifiers the wgpu/Vulkan side can import.
//!
//! PipeWire's DMA-BUF negotiation requires the capture side to advertise
//! exactly the modifiers it can consume — offering more produces frames
//! we can't import; offering fewer (e.g. LINEAR-only) means tiled
//! compositor allocations get rejected and the stream silently falls
//! through to SHM on every real Wayland session.
//!
//! Implementation reaches through wgpu's Vulkan hal escape hatch to call
//! `vkGetPhysicalDeviceFormatProperties2` with
//! `VkDrmFormatModifierPropertiesListEXT` in the pNext chain. Filters by
//! `SAMPLED_IMAGE` (the BGRA→NV12 shader's `textureLoad` access) so we
//! only return modifiers usable by the downstream compute pass.
//!
//! Currently called once at host startup. Selects the same adapter the
//! [`crate::Nv12DmaBuf`] bridge will later pick (Vulkan, high-perf, no
//! surface) so the modifier list is authoritative for the device the
//! bridge actually opens.

use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum ModifierQueryError {
    #[error("no wgpu adapter available for modifier query")]
    NoAdapter,
    #[error("wgpu request_device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error(
        "adapter doesn't advertise VULKAN_EXTERNAL_MEMORY_DMA_BUF; the Vulkan ICD must \
         support VK_EXT_external_memory_dma_buf + VK_EXT_image_drm_format_modifier"
    )]
    FeatureUnsupported,
    #[error("wgpu device is not Vulkan-backed; modifier query only works on Vulkan")]
    NotVulkan,
    #[error("unsupported DRM fourcc for modifier query: 0x{0:08x}")]
    UnsupportedFourcc(u32),
}

pub type Result<T> = std::result::Result<T, ModifierQueryError>;

/// Open a temporary wgpu Vulkan device and return the DRM modifiers the
/// driver advertises as importable for `drm_fourcc`, filtered to those
/// that support `SAMPLED_IMAGE` (the BGRA→NV12 shader's access mode).
///
/// The returned list is the authoritative set to advertise to the
/// compositor. An empty list means the format can be imported only as
/// LINEAR (which is always in the list when supported) — if even LINEAR
/// is absent the GPU can't import this format at all and the caller
/// should skip the DMA-BUF path entirely for it.
pub async fn importable_dmabuf_modifiers(drm_fourcc: u32) -> Result<Vec<u64>> {
    let vk_format = drm_fourcc_to_vk_format(drm_fourcc)?;

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|_| ModifierQueryError::NoAdapter)?;

    let required_features = wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF
        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    if !adapter.features().contains(required_features) {
        return Err(ModifierQueryError::FeatureUnsupported);
    }
    let (device, _queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("tether-gpuconvert modifier-query device"),
            required_features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await?;

    // SAFETY: as_hal returns Some only on the Vulkan backend; raw_instance
    // and raw_physical_device are valid for the lifetime of the hal::Device
    // borrow, which spans this entire unsafe block.
    let modifiers = unsafe {
        let hal_dev = device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or(ModifierQueryError::NotVulkan)?;
        let raw_instance = hal_dev.shared_instance().raw_instance();
        let raw_physical = hal_dev.raw_physical_device();

        // Two-call idiom: first call discovers count, second fills the
        // array. p_drm_format_modifier_properties must be null on the
        // counting call (spec) so we don't preallocate.
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut props2 = vk::FormatProperties2::default().push_next(&mut list);
        raw_instance.get_physical_device_format_properties2(
            raw_physical,
            vk_format,
            &mut props2,
        );

        let count = list.drm_format_modifier_count as usize;
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut storage: Vec<vk::DrmFormatModifierPropertiesEXT> =
            vec![vk::DrmFormatModifierPropertiesEXT::default(); count];
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
            .drm_format_modifier_properties(&mut storage);
        let mut props2 = vk::FormatProperties2::default().push_next(&mut list);
        raw_instance.get_physical_device_format_properties2(
            raw_physical,
            vk_format,
            &mut props2,
        );

        storage
            .into_iter()
            .filter(|p| {
                p.drm_format_modifier_tiling_features
                    .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
            })
            .map(|p| p.drm_format_modifier)
            .collect::<Vec<_>>()
    };

    Ok(modifiers)
}

/// Map the DRM fourcc codes PipeWire emits for our 4-byte BGR-family
/// formats to the Vulkan format the importer expects. AR24/XR24 both
/// land on `B8G8R8A8_UNORM` — Vulkan has no separate "X" form, and the
/// shader ignores the alpha channel for BGRx, so the alias is safe.
fn drm_fourcc_to_vk_format(drm_fourcc: u32) -> Result<vk::Format> {
    // Fourcc constants from <drm/drm_fourcc.h> — little-endian 4-char.
    const AR24: u32 = u32::from_le_bytes(*b"AR24"); // DRM_FORMAT_ARGB8888
    const XR24: u32 = u32::from_le_bytes(*b"XR24"); // DRM_FORMAT_XRGB8888
    match drm_fourcc {
        AR24 | XR24 => Ok(vk::Format::B8G8R8A8_UNORM),
        other => Err(ModifierQueryError::UnsupportedFourcc(other)),
    }
}
