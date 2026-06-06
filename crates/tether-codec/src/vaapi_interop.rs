//! Shared Linux VAAPI dma-buf ↔ renderer format contract.
//!
//! Both the client renderer (`tether-render::gpu::import`) and the host
//! decode probe (`tether-probe::host::vaapi`) need to agree on which
//! `(chroma, bit_depth, drm_fourcc)` surfaces the renderer can import,
//! so the cross-table consistency test (decoder emit set ↔ renderer
//! accept set) has exactly one place to add a new family.
//!
//! Lives in `tether-codec` (and not `tether-render`) for the same
//! reason [`crate::macos_interop`] does: both `tether-render` and
//! `tether-probe` already depend on `tether-codec`, so putting the
//! predicate here lets the probe consult the renderer's accept set
//! WITHOUT a new `tether-probe → tether-render` dependency edge (which
//! would be a cycle — `tether-render` is a client-only crate the probe
//! must not pull in).
//!
//! **Fourcc convention:** DRM fourccs are little-endian here, matching
//! `tether-render::gpu::import` (which builds them with
//! `u32::from_le_bytes`) and the `DmaBufLayer.drm_format` value libva
//! hands back. This is the opposite of [`crate::macos_interop`], where
//! IOSurface fourccs are big-endian — do not cross the conventions.

use tether_protocol::control::{ChromaSubsampling, VideoProfile};

/// `DRM_FORMAT_NV12` — biplanar 4:2:0 8-bit (Y plane + interleaved UV
/// at half-resolution). The surface-level fourcc VAAPI reports for an
/// NV12 decode; the per-layer DRM formats under SEPARATE_LAYERS export
/// are `R8`/`GR88`, but the surface fourcc is `NV12`.
pub const NV12_FOURCC: u32 = u32::from_le_bytes(*b"NV12");
/// `DRM_FORMAT_P010` — biplanar 4:2:0 10-bit, 16-bit cells with 10
/// bits MSB-aligned, half-res UV.
pub const P010_FOURCC: u32 = u32::from_le_bytes(*b"P010");
/// `DRM_FORMAT_XYUV8888` — packed 4:4:4 8-bit, 32 bpp (V,U,Y,X bytes
/// little-endian). The only 4:4:4 8-bit format in FFmpeg's
/// `vaapi_drm_format_map`, so libavcodec returns this regardless of
/// the driver's internal tiling.
pub const XYUV_FOURCC: u32 = u32::from_le_bytes(*b"XYUV");
/// `DRM_FORMAT_YUV444` — planar 4:4:4 8-bit, three full-resolution
/// R8 planes. NVIDIA NVDEC exports this for HEVC Main 4:4:4 8-bit and
/// the renderer imports it through the `Planar444` layout.
pub const YU24_FOURCC: u32 = u32::from_le_bytes(*b"YU24");
/// `DRM_FORMAT_Y410` — packed 4:4:4 10-bit, 10:10:10:2
/// (A:Cr:Y:Cb little-endian). What the Intel media-driver decoder —
/// our 4:4:4 reference — exports for HEVC Main 4:4:4 10-bit. A
/// different driver may emit `XV30` or biplanar `P410`; the decode
/// probe ([`crate`]-side L2 in `tether-probe`) rejects those so the
/// profile degrades at negotiation instead of black-screening, until
/// those layouts get their own renderer cell.
pub const Y410_FOURCC: u32 = u32::from_le_bytes(*b"Y410");

/// Whether the Linux dma-buf import path
/// (`tether-render::gpu::import::import_dmabuf_textures`) will accept a
/// surface with this `(chroma, bit_depth, surface_fourcc)`.
///
/// The renderer's packed 4:4:4 paths hard-gate on the layer's
/// `drm_format` (`XYUV` for 8-bit, `Y410` for 10-bit); the biplanar
/// 4:2:0 paths trust the negotiated `chroma` and don't sniff the
/// fourcc at all. This predicate captures the *surface-level* fourcc
/// VAAPI reports for each family (`NV12`/`P010`/`XYUV`/`Y410`), which
/// equals the per-layer `drm_format` for the single-layer packed
/// surfaces the renderer gates on, and is the reliable family code for
/// the biplanar surfaces it doesn't gate on. One source of truth for
/// "is a decoded surface of this fourcc renderable for this profile".
///
/// Mirrors [`crate::macos_interop::accepts_iosurface_fourcc`].
#[must_use]
pub fn accepts_dmabuf_fourcc(chroma: ChromaSubsampling, bit_depth: u8, fourcc: u32) -> bool {
    matches!(
        (chroma, bit_depth, fourcc),
        (ChromaSubsampling::Yuv420, 8, NV12_FOURCC)
            | (ChromaSubsampling::Yuv420, 10, P010_FOURCC)
            | (ChromaSubsampling::Yuv444, 8, XYUV_FOURCC | YU24_FOURCC)
            | (ChromaSubsampling::Yuv444, 10, Y410_FOURCC)
    )
}

/// Human-readable label of the DRM fourcc expected for this profile,
/// for error messages. Sibling of
/// [`crate::macos_interop::iosurface_fourcc_expected_label`].
#[must_use]
pub fn dmabuf_fourcc_expected_label(chroma: ChromaSubsampling, bit_depth: u8) -> &'static str {
    match (chroma, bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => "'NV12' (biplanar 4:2:0 8-bit)",
        (ChromaSubsampling::Yuv420, 10) => "'P010' (biplanar 4:2:0 10-bit)",
        (ChromaSubsampling::Yuv444, 8) => {
            "'XYUV' (packed 4:4:4 8-bit) or 'YU24' (planar 4:4:4 8-bit)"
        }
        (ChromaSubsampling::Yuv444, 10) => "'Y410' (packed 4:4:4 10-bit)",
        _ => "a DRM fourcc supported by this build (8/10-bit 4:2:0 or 4:4:4)",
    }
}

/// The surface-level DRM fourcc(s) a VAAPI decode may emit for
/// `profile` — the **decode-output** currency.
///
/// Deliberately distinct from
/// [`crate::vaapi::encoder::expected_dmabuf_fourcc`], which is the
/// **encoder-input** currency and lists `XV30` for 4:4:4 10-bit (the
/// fourcc gpuconvert's BGRA→4:4:4-10 pass would write into the encoder
/// surface). The decode output on our Intel reference is `Y410`, not
/// `XV30` — the bit layout matches but the fourcc differs, and
/// conflating the two is exactly the kind of cross-table drift this
/// module exists to prevent. Only **verified** fourccs are listed; a
/// non-Intel driver emitting something else is caught by the decode
/// probe (L2) rather than silently declared renderable here.
///
/// Sibling of [`crate::videotoolbox::expected_iosurface_fourccs`].
#[must_use]
pub fn expected_dmabuf_decode_fourcc(profile: VideoProfile) -> &'static [u32] {
    match (profile.chroma, profile.bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => &[NV12_FOURCC],
        (ChromaSubsampling::Yuv420, 10) => &[P010_FOURCC],
        (ChromaSubsampling::Yuv444, 8) => &[XYUV_FOURCC, YU24_FOURCC],
        (ChromaSubsampling::Yuv444, 10) => &[Y410_FOURCC],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decode-output fourcc the renderer MUST accept for each
    /// chroma/bit-depth family tether negotiates on Linux. Pins the
    /// exact fourccs so a regression shows as a failed assert rather
    /// than a black screen — the Linux analog of the macOS `'x444'`
    /// regression.
    #[test]
    fn renderer_accepts_decode_fourcc_for_every_family() {
        let cases = [
            (ChromaSubsampling::Yuv420, 8, NV12_FOURCC),
            (ChromaSubsampling::Yuv420, 10, P010_FOURCC),
            (ChromaSubsampling::Yuv444, 8, XYUV_FOURCC),
            (ChromaSubsampling::Yuv444, 8, YU24_FOURCC),
            (ChromaSubsampling::Yuv444, 10, Y410_FOURCC),
        ];
        for (chroma, bit_depth, fourcc) in cases {
            assert!(
                accepts_dmabuf_fourcc(chroma, bit_depth, fourcc),
                "renderer must accept decode fourcc 0x{fourcc:08x} for {chroma:?} {bit_depth}-bit"
            );
        }
    }

    /// A surface fourcc from the wrong family must not import — e.g. a
    /// 4:2:0 surface arriving on a 4:4:4 session (silent-downsample), or
    /// the encoder-input `XV30` (which is NOT the decode output) showing
    /// up where `Y410` is expected.
    #[test]
    fn renderer_rejects_cross_family_and_encoder_input_fourccs() {
        assert!(
            !accepts_dmabuf_fourcc(ChromaSubsampling::Yuv444, 10, NV12_FOURCC),
            "4:2:0 fourcc must not import on a 4:4:4 10-bit session"
        );
        assert!(
            !accepts_dmabuf_fourcc(ChromaSubsampling::Yuv420, 8, XYUV_FOURCC),
            "4:4:4 fourcc must not import on a 4:2:0 8-bit session"
        );
        let xv30 = u32::from_le_bytes(*b"XV30");
        assert!(
            !accepts_dmabuf_fourcc(ChromaSubsampling::Yuv444, 10, xv30),
            "encoder-input XV30 is not a declared decode output — must not be accepted \
             until a driver is verified to emit it and a renderer cell exists"
        );
    }

    /// `expected_dmabuf_decode_fourcc` ⊆ `accepts_dmabuf_fourcc` for
    /// every family — the consistency invariant, restricted to the
    /// cells this module knows. The `PROFILE_PREFERENCE`-driven version
    /// lives in `tether-probe` (which has the negotiable profile set).
    #[test]
    fn declared_decode_emit_is_renderer_accepted() {
        for (chroma, bit_depth) in [
            (ChromaSubsampling::Yuv420, 8),
            (ChromaSubsampling::Yuv420, 10),
            (ChromaSubsampling::Yuv444, 8),
            (ChromaSubsampling::Yuv444, 10),
        ] {
            let profile = VideoProfile {
                codec: tether_protocol::control::CodecKind::Hevc,
                chroma,
                bit_depth,
            };
            for &fourcc in expected_dmabuf_decode_fourcc(profile) {
                assert!(
                    accepts_dmabuf_fourcc(chroma, bit_depth, fourcc),
                    "{chroma:?} {bit_depth}-bit: decoder emits 0x{fourcc:08x} that import rejects"
                );
            }
        }
    }
}
