//! Cross-platform profile capability probe.
//!
//! `decoder.open()` and `encoder.open()` are profile-blind in libavcodec:
//! they take a codec ID, not a profile/chroma/bit-depth triple. That's
//! why we used to over-advertise — `VaapiDecoder::new(Hevc)` would
//! succeed on a driver that only handles 4:2:0 decode, then fail at
//! first-frame time on a 4:4:4 bitstream; similarly the macOS client
//! had a blanket "Yuv420 only" gate that erred on the conservative side
//! but couldn't *prove* anything about the hardware.
//!
//! The fix is a real round-trip probe: actually try to encode + decode
//! at the target profile and observe what the hardware does. Encode
//! probe runs the backend's own encoder at 128×128 *plus*, on Linux,
//! the production gpuconvert bridge (step 3 of the migration will wire
//! this — for now Linux 10-bit is construction-only). Decode probe
//! loads a small checked-in IDR fixture for the profile and feeds it
//! to the backend's decoder, demanding back a hardware frame (no
//! software fallback).
//!
//! The [`ProfileProbe`] trait expresses the shape every backend must
//! satisfy. We pick the active backend at compile time via cfg in
//! [`crate`]'s `probe_host` / `probe_client` orchestration; the trait
//! makes the contract discoverable when a future backend lands (NVENC,
//! QSV, AMF, Media Foundation) so adding it doesn't require finding
//! every cfg-island scattered through the crate.

use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

use crate::PipelineStage;

/// Backend probe error: a per-stage tag + a human-readable reason.
/// The trait returns this instead of `tether_codec::CodecError` so
/// each backend can name *which* stage rejected the probe (capture
/// vs encoder construct vs encoder submit vs decode round trip).
/// The orchestration in [`crate::host_supported_profiles`] /
/// [`crate::client_supported_profiles`] surfaces the tag verbatim in
/// the [`crate::SupportStatus::Unsupported`] verdict.
#[derive(Debug)]
pub(crate) struct ProbeError {
    pub stage: PipelineStage,
    pub reason: String,
}

impl ProbeError {
    pub(crate) fn new(stage: PipelineStage, reason: impl Into<String>) -> Self {
        Self {
            stage,
            reason: reason.into(),
        }
    }

    /// Adapter for backend code that already produces a
    /// `tether_codec::CodecError`. Tags it with `stage` and formats
    /// the message into a string for the SupportStatus.
    pub(crate) fn from_codec(stage: PipelineStage, e: tether_codec::CodecError) -> Self {
        Self {
            stage,
            reason: format!("{e}"),
        }
    }
}

pub(crate) type Result<T> = std::result::Result<T, ProbeError>;

/// Backend-specific probe contract. One impl per hardware codec
/// pipeline (VAAPI, VideoToolbox, …). The orchestration in
/// [`crate::host_supported_profiles`] / [`crate::client_supported_profiles`]
/// dispatches to the active platform's impl and rolls the per-profile
/// results up into the session-wide capability list.
///
/// Methods are associated functions (no `&self`) because every
/// backend's probe is stateless — the relevant state lives in the OS
/// driver, not in the probe object. A trait (rather than two
/// platform-specific free functions) keeps the contract explicit when
/// adding a new backend.
#[cfg_attr(not(any(target_os = "linux", target_os = "macos")), allow(dead_code))]
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
/// 128×128 grey. Regeneration commands live in `fixtures/probe/README.md`.
#[cfg_attr(not(any(target_os = "linux", target_os = "macos")), allow(dead_code))]
pub(crate) fn fixture_for(profile: VideoProfile) -> Option<&'static [u8]> {
    match (profile.codec, profile.chroma, profile.bit_depth) {
        (CodecKind::H264, ChromaSubsampling::Yuv420, 8) => {
            Some(include_bytes!("../fixtures/probe/h264_yuv420_8bit.idr"))
        }
        (CodecKind::Hevc, ChromaSubsampling::Yuv420, 8) => {
            Some(include_bytes!("../fixtures/probe/hevc_yuv420_8bit.idr"))
        }
        (CodecKind::Hevc, ChromaSubsampling::Yuv420, 10) => {
            Some(include_bytes!("../fixtures/probe/hevc_yuv420_10bit.idr"))
        }
        (CodecKind::Hevc, ChromaSubsampling::Yuv444, 8) => {
            Some(include_bytes!("../fixtures/probe/hevc_yuv444_8bit.idr"))
        }
        (CodecKind::Hevc, ChromaSubsampling::Yuv444, 10) => {
            Some(include_bytes!("../fixtures/probe/hevc_yuv444_10bit.idr"))
        }
        // 4:2:2 fixtures are checked in but not wired up until
        // ChromaSubsampling::Yuv422 lands on the wire. H.264 4:4:4,
        // AV1, etc. — no fixture means we conservatively report
        // decode=false until one is added.
        _ => None,
    }
}
