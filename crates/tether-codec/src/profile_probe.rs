//! Cross-platform profile capability probe.
//!
//! `decoder.open()` and `encoder.open()` are profile-blind in libavcodec:
//! they take a codec ID, not a profile/chroma/bit-depth triple. That's
//! why we used to over-advertise — e.g. `VaapiDecoder::new(Hevc)` would
//! succeed on a driver that only handles 4:2:0 decode, then fail at
//! first-frame time on a 4:4:4 bitstream; similarly the macOS client
//! had a blanket "Yuv420 only" gate that erred on the conservative side
//! but couldn't *prove* anything about the hardware.
//!
//! The fix is a real round-trip probe: actually try to encode + decode
//! at the target profile and observe what the hardware does. Encode
//! probe runs the backend's own encoder at 128×128. Decode probe loads
//! a small checked-in IDR fixture for the profile (`fixtures/probe/`)
//! and feeds it to the backend's decoder, demanding back a hardware
//! frame (no software fallback).
//!
//! The [`ProfileProbe`] trait expresses the shape every backend must
//! satisfy. We pick the active backend at compile time via cfg in
//! [`crate::probe`]; the trait makes the contract discoverable when a
//! future backend lands (NVENC, QSV, AMF, Media Foundation) so adding
//! it doesn't require finding every cfg-island scattered through the
//! crate.

use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

use crate::Result;

/// Backend-specific probe contract. One impl per hardware codec
/// pipeline (VAAPI, VideoToolbox, ...). The orchestration in
/// [`crate::probe::supported_profiles`] dispatches to the active
/// platform's impl and rolls the per-profile results up into the
/// session-wide capability list.
///
/// Methods are associated functions (no `&self`) because every
/// backend's probe is stateless — the relevant state lives in the OS
/// driver, not in the probe object. A trait (rather than two
/// platform-specific free functions) keeps the contract explicit when
/// adding a new backend.
pub(crate) trait ProfileProbe {
    /// Construct an encoder at `profile` (128×128 floor — satisfies
    /// HEVC's minimum-block constraint on Intel hardware) and verify
    /// it actually produces a non-empty IDR packet. Returns `Err` if
    /// any step fails; the caller treats either outcome as a single
    /// "encode supported" bit.
    fn probe_encode(profile: VideoProfile) -> Result<()>;

    /// Submit `fixture` (an IDR bitstream pre-generated at this
    /// profile; see `fixtures/probe/`) to a fresh decoder and verify
    /// a hardware frame comes back. Software fallback inside ffmpeg's
    /// hwaccel wrapper counts as failure — that's exactly what we're
    /// trying to detect (e.g. macOS VT silently falling back to the
    /// native HEVC decoder for 4:4:4 input).
    fn probe_decode(profile: VideoProfile, fixture: &[u8]) -> Result<()>;
}

/// Decode-side probe fixture for `profile`, or `None` if we don't
/// ship one (advertised profiles without a fixture report
/// `decode=false`).
///
/// Fixtures are tiny single-IDR bitstreams generated offline at
/// 128×128 grey:
///   - `h264_yuv420_8bit.idr` — libx264 Constrained Baseline, ~700 B
///   - `hevc_yuv420_8bit.idr` — libx265 Main, ~2.4 KB
///   - `hevc_yuv444_8bit.idr` — libx265 Rext (Main 4:4:4 8-bit), ~2.4 KB
///
/// Regenerate via the commands documented in `fixtures/probe/README.md`.
pub(crate) fn fixture_for(profile: VideoProfile) -> Option<&'static [u8]> {
    match (profile.codec, profile.chroma, profile.bit_depth) {
        (CodecKind::H264, ChromaSubsampling::Yuv420, 8) => Some(include_bytes!(
            "../fixtures/probe/h264_yuv420_8bit.idr"
        )),
        (CodecKind::Hevc, ChromaSubsampling::Yuv420, 8) => Some(include_bytes!(
            "../fixtures/probe/hevc_yuv420_8bit.idr"
        )),
        (CodecKind::Hevc, ChromaSubsampling::Yuv444, 8) => Some(include_bytes!(
            "../fixtures/probe/hevc_yuv444_8bit.idr"
        )),
        // H.264 4:4:4, AV1, 10-bit profiles, etc. — no fixture means
        // we conservatively report decode=false until one is added.
        _ => None,
    }
}

/// Per-profile capability bundle. Both halves are independent: a
/// platform can encode a profile but not decode it (rare but real on
/// some VAAPI drivers), or decode but not encode (macOS for HEVC 4:4:4
/// on hardware that supports the Rext decode path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileCapability {
    pub profile: VideoProfile,
    pub encode: bool,
    pub decode: bool,
}
