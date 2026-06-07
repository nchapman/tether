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
//! `VkDrmFormatModifierPropertiesListEXT` in the pNext chain. The
//! [`importable_dmabuf_modifiers`] entry point filters by `SAMPLED_IMAGE`
//! (the BGRA→NV12 shader's `textureLoad` access). [`storable_dmabuf_modifiers`]
//! filters by `STORAGE_IMAGE` instead, gating the 10-bit compute
//! pipelines (P010/P410/XV30) whose outputs are storage textures rather
//! than sampled reads.
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
    query_dmabuf_modifiers(drm_fourcc, vk::FormatFeatureFlags::SAMPLED_IMAGE).await
}

/// Sibling of [`importable_dmabuf_modifiers`] that filters on
/// `STORAGE_IMAGE` instead of `SAMPLED_IMAGE`.
///
/// Required gate for the 10-bit gpuconvert compute pipelines: a BGRA→P010
/// or BGRA→XV30 shader writes into an R16/Rg16 (P010) or
/// Rgb10a2Unorm (XV30) storage texture,
/// which the importer must support as `VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT`,
/// not just sampled-image. Some drivers expose 16-bit unorm as sampleable
/// (so [`importable_dmabuf_modifiers`] returns a non-empty list) but not
/// storage-writable — in that case `create_compute_pipeline` would fail
/// at runtime with a validation error. Probing storage support up front
/// lets the host filter the 10-bit profiles out of the encode-capability
/// set before negotiation.
///
/// Same return semantics as [`importable_dmabuf_modifiers`]: empty list
/// means no modifier supports storage writes on this device, including
/// LINEAR.
pub async fn storable_dmabuf_modifiers(drm_fourcc: u32) -> Result<Vec<u64>> {
    query_dmabuf_modifiers(drm_fourcc, vk::FormatFeatureFlags::STORAGE_IMAGE).await
}

/// Shared Vulkan path behind [`importable_dmabuf_modifiers`] and
/// [`storable_dmabuf_modifiers`]. The only difference between the two
/// is which `VkFormatFeatureFlagBits` the modifier must advertise.
async fn query_dmabuf_modifiers(
    drm_fourcc: u32,
    required_feature: vk::FormatFeatureFlags,
) -> Result<Vec<u64>> {
    let vk_format = drm_fourcc_to_vk_format(drm_fourcc)?;

    let instance = crate::headless_wgpu_instance();
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
        raw_instance.get_physical_device_format_properties2(raw_physical, vk_format, &mut props2);

        let count = list.drm_format_modifier_count as usize;
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut storage: Vec<vk::DrmFormatModifierPropertiesEXT> =
            vec![vk::DrmFormatModifierPropertiesEXT::default(); count];
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
            .drm_format_modifier_properties(&mut storage);
        let mut props2 = vk::FormatProperties2::default().push_next(&mut list);
        raw_instance.get_physical_device_format_properties2(raw_physical, vk_format, &mut props2);

        storage
            .into_iter()
            .filter(|p| {
                p.drm_format_modifier_tiling_features
                    .contains(required_feature)
            })
            .map(|p| p.drm_format_modifier)
            .collect::<Vec<_>>()
    };

    Ok(modifiers)
}

/// Map DRM fourcc codes to the Vulkan format the importer would use.
///
/// **What this answers, precisely**: "If we asked Vulkan to bind a
/// DMA-BUF carrying `drm_fourcc` to an image, which `VkFormat` would we
/// use?" — and consequently, when paired with either
/// [`importable_dmabuf_modifiers`] or [`storable_dmabuf_modifiers`],
/// "can the renderer / gpuconvert sample-read or storage-write this
/// format at all on this device?" The `SAMPLED_IMAGE` /
/// `STORAGE_IMAGE` filter in the caller is the gate.
///
/// **What this does *not* answer**: whether the VAAPI encoder accepts
/// the format as input via `av_hwframe_map(DRM_PRIME → VAAPI)`. That
/// path goes through `vaapi_drm_format_map` in libavcodec, not through
/// Vulkan modifier tables — it has its own probe (see
/// `tether-codec::vaapi::probe`). A 10-bit encode profile may only be
/// advertised when *all three* probes return supported: gpuconvert can
/// storage-write the producer format (this function via
/// [`storable_dmabuf_modifiers`]), the renderer can sample-read the
/// decoded format (this function via [`importable_dmabuf_modifiers`]),
/// AND the VAAPI encoder accepts the input format.
///
/// Fourcc families covered:
/// - Capture input (BGRA family): `AR24`, `XR24` → `B8G8R8A8_UNORM`.
///   Vulkan has no separate "X" form; the shader ignores alpha for
///   BGRx, so the alias is safe.
/// - Encoder-output 8-bit planes: `R8`, `GR88` for NV12/NV24 Y/UV.
/// - Encoder-output 10-bit planes: `R16`, `GR32` for P010/P410 Y/UV
///   (10 bits MSB-aligned in 16-bit cells).
/// - Encoder-output packed 4:4:4 8-bit: `XYUV` → `R8G8B8A8_UNORM` per
///   the `VK_EXT_image_drm_format_modifier` spec appendix's table of
///   DRM-fourcc-to-VkFormat compatibility. Memory layout on LE is
///   `[V][U][Y][X]` which the renderer samples as `.r/.g/.b/.a` =
///   `V/U/Y/X`.
fn drm_fourcc_to_vk_format(drm_fourcc: u32) -> Result<vk::Format> {
    // Fourcc constants from <drm/drm_fourcc.h> — little-endian 4-char.
    const AR24: u32 = u32::from_le_bytes(*b"AR24"); // DRM_FORMAT_ARGB8888
    const XR24: u32 = u32::from_le_bytes(*b"XR24"); // DRM_FORMAT_XRGB8888
                                                    // Y plane of NV12 / NV24 — single-channel 8-bit, DRM_FORMAT_R8.
    const R8: u32 = u32::from_le_bytes(*b"R8  ");
    // UV plane of NV12 / NV24 — two-channel 8-bit, DRM_FORMAT_GR88.
    const GR88: u32 = u32::from_le_bytes(*b"GR88");
    // Y plane of P010 / P410 — single-channel 16-bit, DRM_FORMAT_R16.
    // MSB-aligned 10-bit data lives in bits [15:6] of each cell.
    const R16: u32 = u32::from_le_bytes(*b"R16 ");
    // UV plane of P010 / P410 — two-channel 16-bit,
    // DRM_FORMAT_GR1616 (`fourcc_code('G','R','3','2')`; the "32" in
    // the fourcc means 32 total bits across both channels, not 32
    // bits per channel). Same MSB-align convention as R16.
    const GR32: u32 = u32::from_le_bytes(*b"GR32");
    // Packed XYUV (DRM_FORMAT_XYUV8888) for HEVC Main 4:4:4 8-bit; the
    // VAAPI encoder accepts this as input. The
    // VK_EXT_image_drm_format_modifier spec lists XYUV8888 as
    // compatible with R8G8B8A8_UNORM — the alias is spec-mandated,
    // not a convenience choice (changing it to B8G8R8A8_UNORM would
    // silently query the wrong modifier table).
    const XYUV: u32 = u32::from_le_bytes(*b"XYUV");
    // Packed XV30 (DRM_FORMAT_XV30) for HEVC Main 4:4:4 10-bit. Each
    // pixel is one 10:10:10:2 little-endian word X:V:U:Y; the matching
    // Vulkan format is A2B10G10R10_UNORM_PACK32 per the
    // VK_EXT_image_drm_format_modifier spec appendix.
    const XV30: u32 = u32::from_le_bytes(*b"XV30");
    match drm_fourcc {
        AR24 | XR24 => Ok(vk::Format::B8G8R8A8_UNORM),
        R8 => Ok(vk::Format::R8_UNORM),
        GR88 => Ok(vk::Format::R8G8_UNORM),
        R16 => Ok(vk::Format::R16_UNORM),
        GR32 => Ok(vk::Format::R16G16_UNORM),
        XYUV => Ok(vk::Format::R8G8B8A8_UNORM),
        XV30 => Ok(vk::Format::A2B10G10R10_UNORM_PACK32),
        other => Err(ModifierQueryError::UnsupportedFourcc(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the fourcc → Vulkan format table so a future addition can't
    /// silently break P010/P410 advertisement. Comparing by Vulkan
    /// format discriminant keeps the test stable against ash version
    /// bumps that might wrap `vk::Format` in a newtype.
    #[test]
    fn drm_fourcc_table_covers_10_bit_biplanar_planes() {
        let r16 = u32::from_le_bytes(*b"R16 ");
        let gr32 = u32::from_le_bytes(*b"GR32");
        assert_eq!(drm_fourcc_to_vk_format(r16).unwrap(), vk::Format::R16_UNORM);
        assert_eq!(
            drm_fourcc_to_vk_format(gr32).unwrap(),
            vk::Format::R16G16_UNORM
        );
    }

    /// XYUV and the NV12 plane fourccs are new entries; pin them so
    /// the spec-mandated aliasing (esp. XYUV → R8G8B8A8_UNORM per the
    /// VK_EXT_image_drm_format_modifier appendix) doesn't drift.
    #[test]
    fn drm_fourcc_table_covers_xyuv_and_nv12_planes() {
        let xyuv = u32::from_le_bytes(*b"XYUV");
        let r8 = u32::from_le_bytes(*b"R8  ");
        let gr88 = u32::from_le_bytes(*b"GR88");
        assert_eq!(
            drm_fourcc_to_vk_format(xyuv).unwrap(),
            vk::Format::R8G8B8A8_UNORM
        );
        assert_eq!(drm_fourcc_to_vk_format(r8).unwrap(), vk::Format::R8_UNORM);
        assert_eq!(
            drm_fourcc_to_vk_format(gr88).unwrap(),
            vk::Format::R8G8_UNORM
        );
    }

    /// Pin DRM_FORMAT_XV30 → A2B10G10R10_UNORM_PACK32 — the
    /// VK_EXT_image_drm_format_modifier-mandated alias for HEVC Main
    /// 4:4:4 10-bit encode input. Drift here silently breaks the
    /// XV30 storage-modifier probe.
    #[test]
    fn drm_fourcc_table_covers_xv30() {
        let xv30 = u32::from_le_bytes(*b"XV30");
        assert_eq!(
            drm_fourcc_to_vk_format(xv30).unwrap(),
            vk::Format::A2B10G10R10_UNORM_PACK32,
        );
    }

    #[test]
    fn drm_fourcc_table_still_handles_existing_8_bit_inputs() {
        // Regression pin: the 10-bit addition must not break the
        // BGRA capture-input paths the live host uses today.
        let ar24 = u32::from_le_bytes(*b"AR24");
        let xr24 = u32::from_le_bytes(*b"XR24");
        assert_eq!(
            drm_fourcc_to_vk_format(ar24).unwrap(),
            vk::Format::B8G8R8A8_UNORM
        );
        assert_eq!(
            drm_fourcc_to_vk_format(xr24).unwrap(),
            vk::Format::B8G8R8A8_UNORM
        );
    }

    #[test]
    fn drm_fourcc_table_rejects_unknown_codes() {
        let unknown = 0xDEADBEEFu32;
        assert!(matches!(
            drm_fourcc_to_vk_format(unknown),
            Err(ModifierQueryError::UnsupportedFourcc(0xDEADBEEF))
        ));
    }

    /// Storage-image probe sanity check on a real Vulkan device.
    ///
    /// On any Mesa or recent Intel/AMD driver, R16/Rg16 must support
    /// `STORAGE_IMAGE` for `DRM_FORMAT_MOD_LINEAR` — that's the path the
    /// 10-bit BGRA→P010 compute shader requires. A failure here means
    /// the box can't host 10-bit encode regardless of what the VAAPI
    /// encoder probe says; the bridge constructor in
    /// [`crate::Bgra2P010DmaBuf`] uses the same query to refuse
    /// construction loudly rather than crash at
    /// `create_compute_pipeline`.
    #[test]
    #[ignore = "requires a working Vulkan adapter advertising VK_EXT_image_drm_format_modifier; run with: cargo test -p tether-gpuconvert -- --ignored"]
    fn storable_probe_returns_linear_for_r16_and_gr32() {
        let r16 = u32::from_le_bytes(*b"R16 ");
        let gr32 = u32::from_le_bytes(*b"GR32");
        let r16_mods =
            pollster::block_on(storable_dmabuf_modifiers(r16)).expect("storage probe for R16");
        let gr32_mods =
            pollster::block_on(storable_dmabuf_modifiers(gr32)).expect("storage probe for GR32");
        // LINEAR is the only modifier the encoder DMA-BUF export uses;
        // anything else returned is bonus. If LINEAR is missing the
        // driver can't host storage writes to 16-bit unorm at all.
        let linear = crate::dmabuf_export::DRM_FORMAT_MOD_LINEAR;
        assert!(
            r16_mods.contains(&linear),
            "expected DRM_FORMAT_MOD_LINEAR in R16 storage modifiers, got {r16_mods:?}",
        );
        assert!(
            gr32_mods.contains(&linear),
            "expected DRM_FORMAT_MOD_LINEAR in GR32 storage modifiers, got {gr32_mods:?}",
        );
    }

    /// XV30 sibling of the R16/GR32 storage probe — confirms LINEAR
    /// `STORAGE_IMAGE` support on `DRM_FORMAT_XV30`
    /// (`VK_FORMAT_A2B10G10R10_UNORM_PACK32`) on whatever hardware
    /// runs the test. Catches a driver regression that drops storage
    /// writability on the packed 10-bit format before it manifests as
    /// a `Bgra2Xv30DmaBuf::new` mid-session failure. May SKIP on
    /// Intel iHD per the same driver-gap CLAUDE.md documents for the
    /// other 10-bit paths.
    #[test]
    #[ignore = "requires a working Vulkan adapter advertising VK_EXT_image_drm_format_modifier; may SKIP on Intel iHD; run with: cargo test -p tether-gpuconvert -- --ignored"]
    fn storable_probe_returns_linear_for_xv30() {
        let xv30 = u32::from_le_bytes(*b"XV30");
        let xv30_mods =
            pollster::block_on(storable_dmabuf_modifiers(xv30)).expect("storage probe for XV30");
        let linear = crate::dmabuf_export::DRM_FORMAT_MOD_LINEAR;
        assert!(
            xv30_mods.contains(&linear),
            "expected DRM_FORMAT_MOD_LINEAR in XV30 storage modifiers, got {xv30_mods:?}",
        );
    }
}
