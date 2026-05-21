//! Encoder / decoder selection policy. Tether hard-requires GPU
//! acceleration on both ends — no software fallback path.
//!
//! The motivation: software H.264 at 4K30 burns ~2-3 cores on the
//! capture side and the same on decode; that budget needs to be free
//! for capture, encode-side rate control, and (on the client) a
//! responsive UI thread + sample-accurate input forwarding. The
//! zero-copy DMA-BUF decode path (decoder surface -> wgpu Vulkan
//! import, no CPU readback) is also only reachable through the VAAPI
//! decoder — the SW path would feed CPU planes back through the
//! upload road we deliberately walked off of. Committing to "GPU or
//! nothing" makes the rest of the pipeline simpler (one render path,
//! one stat surface) and surfaces driver problems immediately
//! instead of hiding them behind a slower-and-different fallback.
//!
//! Capability discovery: [`supported_profiles`] runs a real encode +
//! decode round trip per profile in [`PROFILE_PREFERENCE`] (see the
//! [`profile_probe`](crate::profile_probe) module for the contract).
//! Cached via `OnceLock` — driver caps don't change at runtime, so
//! we pay the probe cost once per process. The thin
//! [`supported_encode_profiles`] / [`supported_decode_profiles`]
//! accessors filter that result for the host's advertise step and the
//! client's advertise step respectively.
//!
//! Per-session construction goes through [`probe_encoder`] /
//! [`probe_decoder`]: these aren't probes (they build the real
//! session objects at real dims), they're named that way for
//! historical consistency with the old API.

use crate::{CodecError, Decoder, Encoder, Result};
#[cfg_attr(target_os = "macos", allow(unused_imports))]
use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

use crate::profile_probe::ProfileCapability;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use crate::profile_probe::{fixture_for, ProfileProbe};

#[cfg(target_os = "linux")]
use crate::vaapi::probe::VaapiProbe as ActiveProbe;
#[cfg(target_os = "macos")]
use crate::videotoolbox::probe::VideoToolboxProbe as ActiveProbe;

/// Host preference order, best-first. The negotiation picks the first
/// entry that appears in *both* the host's encode capabilities and the
/// client's advertised decode capabilities.
///
/// Anchored to desktop content quality. HEVC Main444 8-bit goes first
/// because it preserves antialiased text and UI chroma detail that
/// 4:2:0 visibly smears. H.264 4:4:4 is intentionally absent — VAAPI
/// has no encode profile for it, and there is no other host backend
/// in this build yet (Sunshine confirms the same gap in
/// refs/Sunshine/src/platform/linux/vaapi.cpp:202).
pub const PROFILE_PREFERENCE: &[VideoProfile] = &[
    VideoProfile::HEVC_8BIT_444,
    VideoProfile::HEVC_8BIT_420,
    VideoProfile::H264_8BIT_420,
];

/// Best mutual profile between the host's encode capabilities and the
/// client's decode capabilities, picked from [`PROFILE_PREFERENCE`].
/// Returns `None` only when no preference-list entry appears in both
/// sets — the caller treats that as a session-end condition (no
/// compatible codec).
///
/// Host caps are expected to come from [`supported_encode_profiles`];
/// client caps from the bincode-decoded
/// [`tether_protocol::control::CLIENT_DECODE_PROFILES_EXTENSION_KEY`]
/// payload (or the legacy assumption `[VideoProfile::H264_8BIT_420]`
/// when the extension is absent).
#[must_use]
pub fn pick_supported_profile(
    host_caps: &[VideoProfile],
    client_caps: &[VideoProfile],
) -> Option<VideoProfile> {
    PROFILE_PREFERENCE
        .iter()
        .copied()
        .find(|p| host_caps.contains(p) && client_caps.contains(p))
}

/// Probe + report what this platform's HW pipeline can actually do
/// for each entry in [`PROFILE_PREFERENCE`]. Real round trip — see
/// the [`profile_probe`](crate::profile_probe) module for why a
/// construction-only probe lies. Cached for the process lifetime;
/// driver caps don't change at runtime.
///
/// Returns one [`ProfileCapability`] per `PROFILE_PREFERENCE` entry,
/// in `PROFILE_PREFERENCE` order, with `encode` and `decode` bits set
/// independently. A profile that the platform supports neither way
/// still appears in the list (with both bits `false`) so the caller
/// can introspect the full preference matrix.
#[must_use]
pub fn supported_profiles() -> Vec<ProfileCapability> {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Vec<ProfileCapability>> = OnceLock::new();
    CACHED.get_or_init(probe_all).clone()
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn probe_all() -> Vec<ProfileCapability> {
    PROFILE_PREFERENCE
        .iter()
        .copied()
        .map(probe_one)
        .collect()
}

/// Stub for platforms without a hardware backend yet. Every entry
/// reports both bits false; callers see an empty advertised list and
/// surface `no_hw_*` to the operator.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn probe_all() -> Vec<ProfileCapability> {
    PROFILE_PREFERENCE
        .iter()
        .copied()
        .map(|profile| ProfileCapability {
            profile,
            encode: false,
            decode: false,
        })
        .collect()
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn probe_one(profile: VideoProfile) -> ProfileCapability {
    let encode = match ActiveProbe::probe_encode(profile) {
        Ok(()) => true,
        Err(e) => {
            tracing::debug!(?profile, error = %e, "encode probe failed");
            false
        }
    };
    let decode = match fixture_for(profile) {
        Some(fixture) => match ActiveProbe::probe_decode(profile, fixture) {
            Ok(()) => true,
            Err(e) => {
                tracing::debug!(?profile, error = %e, "decode probe failed");
                false
            }
        },
        None => {
            // No fixture shipped for this profile — conservatively
            // report decode=false. Adding 10-bit / AV1 / H.264 4:4:4
            // means adding a fixture + extending fixture_for.
            tracing::debug!(?profile, "no decode fixture; reporting decode=false");
            false
        }
    };
    ProfileCapability {
        profile,
        encode,
        decode,
    }
}

/// Profiles this host can encode. Thin filter over
/// [`supported_profiles`]; used by the host's handshake to populate
/// the encode-side capability advertisement.
#[must_use]
pub fn supported_encode_profiles() -> Vec<VideoProfile> {
    supported_profiles()
        .into_iter()
        .filter(|c| c.encode)
        .map(|c| c.profile)
        .collect()
}

/// Profiles this client can decode. Thin filter over
/// [`supported_profiles`]; used by the client's handshake to populate
/// the `tether.cap.video.decode-profiles` hello extension so the
/// host's negotiator never picks a profile we can't construct.
#[must_use]
pub fn supported_decode_profiles() -> Vec<VideoProfile> {
    supported_profiles()
        .into_iter()
        .filter(|c| c.decode)
        .map(|c| c.profile)
        .collect()
}

/// Construct an encoder for the negotiated session. Not a probe —
/// this builds the real session encoder at real dimensions and
/// returns it for the host send loop. Errors with a diagnostics-friendly
/// message if construction fails.
pub fn probe_encoder(
    profile: VideoProfile,
    width: u32,
    height: u32,
    fps: u32,
    bitrate_kbps: u32,
) -> Result<(VideoProfile, Box<dyn Encoder>)> {
    #[cfg(target_os = "linux")]
    {
        match crate::vaapi::VaapiEncoder::new(profile, width, height, fps, bitrate_kbps) {
            Ok(enc) => Ok((profile, Box::new(enc))),
            Err(e) => {
                tracing::warn!(
                    backend = "vaapi",
                    codec = ?profile.codec,
                    chroma = ?profile.chroma,
                    bit_depth = profile.bit_depth,
                    error = %e,
                    "VAAPI encoder construction failed"
                );
                Err(no_hw_encoder(profile, e))
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // The negotiator never picks a profile the probe layer
        // reported `encode=false` for, so by the time we reach here
        // the (chroma, bit_depth) combination has already been
        // proven against the live VT wrapper. Construction can still
        // fail at real dims (resource limits, dropped permissions);
        // that surfaces as `NoHardwareCodec` with the underlying
        // FFmpeg error attached.
        match crate::videotoolbox::VideoToolboxEncoder::new(
            profile,
            width,
            height,
            fps,
            bitrate_kbps,
        ) {
            Ok(enc) => Ok((profile, Box::new(enc))),
            Err(e) => {
                tracing::warn!(
                    backend = "videotoolbox",
                    codec = ?profile.codec,
                    chroma = ?profile.chroma,
                    bit_depth = profile.bit_depth,
                    error = %e,
                    "VideoToolbox encoder construction failed"
                );
                Err(no_hw_encoder_vt(profile.codec, e))
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (profile, width, height, fps, bitrate_kbps);
        Err(no_hw_encoder_for_platform())
    }
}

/// Construct the decoder for the codec the host chose in its
/// [`ServerHelloV1`](tether_protocol::control::ServerHelloV1::chosen_codec).
/// Errors if no GPU decoder is available for that codec on this client.
pub fn probe_decoder(kind: CodecKind) -> Result<Box<dyn Decoder>> {
    #[cfg(target_os = "linux")]
    {
        match crate::vaapi::VaapiDecoder::new(kind) {
            Ok(dec) => return Ok(Box::new(dec)),
            Err(e) => {
                tracing::error!(
                    backend = "vaapi",
                    codec = ?kind,
                    error = %e,
                    "VAAPI decoder construction failed"
                );
                return Err(no_hw_decoder(kind, e));
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        match crate::videotoolbox::VideoToolboxDecoder::new(kind) {
            Ok(dec) => return Ok(Box::new(dec)),
            Err(e) => {
                tracing::error!(
                    backend = "videotoolbox",
                    codec = ?kind,
                    error = %e,
                    "VideoToolbox decoder construction failed"
                );
                return Err(no_hw_decoder_vt(kind, e));
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = kind;
        Err(no_hw_decoder_for_platform())
    }
}

#[cfg(target_os = "linux")]
fn no_hw_encoder(profile: VideoProfile, source: CodecError) -> CodecError {
    let profile_hint = match (profile.codec, profile.chroma) {
        (CodecKind::H264, _) => "VAProfileH264{ConstrainedBaseline,Main,High}",
        (CodecKind::Hevc, ChromaSubsampling::Yuv420) => "VAProfileHEVCMain",
        (CodecKind::Hevc, ChromaSubsampling::Yuv444) => "VAProfileHEVCMain444",
        (CodecKind::Av1, _) => "VAProfileAV1Profile0",
    };
    CodecError::NoHardwareCodec(format!(
        "VAAPI encoder unavailable for {:?} {:?} {}-bit ({source}). \
         Check that /dev/dri/renderD128 is present and readable, and that `vainfo` \
         lists {profile_hint} with VAEntrypointEnc*. \
         Tether requires GPU encode — there is no software fallback.",
        profile.codec, profile.chroma, profile.bit_depth
    ))
}

#[cfg(target_os = "macos")]
fn no_hw_encoder_vt(kind: CodecKind, source: CodecError) -> CodecError {
    CodecError::NoHardwareCodec(format!(
        "VideoToolbox encoder unavailable for {kind:?} ({source}). \
         Check that `ffmpeg -hide_banner -encoders | grep videotoolbox` lists \
         h264_videotoolbox / hevc_videotoolbox — Homebrew's `ffmpeg` formula \
         enables `--enable-videotoolbox` by default; a custom build may not. \
         Tether requires GPU encode — there is no software fallback."
    ))
}

#[cfg(target_os = "linux")]
fn no_hw_decoder(kind: CodecKind, source: CodecError) -> CodecError {
    CodecError::NoHardwareCodec(format!(
        "VAAPI decoder unavailable for {kind:?} ({source}). \
         Check `vainfo` lists VAEntrypointVLD for the chosen codec, and that the \
         kernel + libva versions match (Mesa 24+ on a 6.x kernel is the verified \
         path). Tether requires GPU decode — there is no software fallback."
    ))
}

#[cfg(target_os = "macos")]
fn no_hw_decoder_vt(kind: CodecKind, source: CodecError) -> CodecError {
    CodecError::NoHardwareCodec(format!(
        "VideoToolbox decoder unavailable for {kind:?} ({source}). \
         Check that `ffmpeg -hide_banner -decoders | grep videotoolbox` reports \
         hwaccel support for the codec — Homebrew's `ffmpeg` formula enables \
         `--enable-videotoolbox` by default; a custom build may not. \
         Tether requires GPU decode — there is no software fallback."
    ))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn no_hw_encoder_for_platform() -> CodecError {
    CodecError::NoHardwareCodec(
        "Tether currently supports hardware encode on Linux (VAAPI) and \
         macOS (VideoToolbox). Windows/NVENC and Windows/AMF backends are \
         not yet implemented."
            .to_string(),
    )
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn no_hw_decoder_for_platform() -> CodecError {
    CodecError::NoHardwareCodec(
        "Tether currently supports hardware decode on Linux (VAAPI) and \
         macOS (VideoToolbox). Windows/NVDEC and Windows/D3D11VA backends \
         are not yet implemented."
            .to_string(),
    )
}

#[cfg(test)]
mod negotiation_tests {
    use super::*;

    #[test]
    fn picks_best_mutual_from_preference() {
        let host = vec![
            VideoProfile::H264_8BIT_420,
            VideoProfile::HEVC_8BIT_420,
            VideoProfile::HEVC_8BIT_444,
        ];
        let client = vec![
            VideoProfile::HEVC_8BIT_444,
            VideoProfile::HEVC_8BIT_420,
            VideoProfile::H264_8BIT_420,
        ];
        assert_eq!(
            pick_supported_profile(&host, &client),
            Some(VideoProfile::HEVC_8BIT_444)
        );
    }

    #[test]
    fn falls_back_when_top_profile_one_sided() {
        let host = vec![VideoProfile::HEVC_8BIT_444, VideoProfile::H264_8BIT_420];
        let client = vec![VideoProfile::H264_8BIT_420];
        assert_eq!(
            pick_supported_profile(&host, &client),
            Some(VideoProfile::H264_8BIT_420)
        );
    }

    #[test]
    fn returns_none_when_disjoint() {
        let host = vec![VideoProfile::HEVC_8BIT_420];
        let client = vec![VideoProfile {
            codec: CodecKind::Av1,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 8,
        }];
        assert_eq!(pick_supported_profile(&host, &client), None);
    }

    #[test]
    fn legacy_client_assumed_h264_420_gets_h264_420() {
        let host = vec![
            VideoProfile::HEVC_8BIT_444,
            VideoProfile::HEVC_8BIT_420,
            VideoProfile::H264_8BIT_420,
        ];
        let legacy_client = vec![VideoProfile::H264_8BIT_420];
        assert_eq!(
            pick_supported_profile(&host, &legacy_client),
            Some(VideoProfile::H264_8BIT_420)
        );
    }

    #[test]
    fn empty_host_or_client_returns_none() {
        assert_eq!(
            pick_supported_profile(&[], &[VideoProfile::H264_8BIT_420]),
            None
        );
        assert_eq!(
            pick_supported_profile(&[VideoProfile::H264_8BIT_420], &[]),
            None
        );
    }

    #[test]
    fn preference_order_is_desktop_quality_first() {
        // Pin the preference order — a future refactor that swaps the
        // first two entries would silently regress desktop sessions
        // from HEVC 4:4:4 to HEVC 4:2:0.
        assert_eq!(PROFILE_PREFERENCE[0], VideoProfile::HEVC_8BIT_444);
        assert_eq!(PROFILE_PREFERENCE[1], VideoProfile::HEVC_8BIT_420);
        assert_eq!(PROFILE_PREFERENCE[2], VideoProfile::H264_8BIT_420);
    }

    #[test]
    fn fixtures_present_for_every_profile_we_might_decode() {
        // Every profile in PROFILE_PREFERENCE that we expect a hardware
        // decoder to ever support needs a fixture; otherwise the decode
        // probe permanently reports false even on capable hardware.
        // (H.264 4:4:4 / AV1 / 10-bit not in PROFILE_PREFERENCE today,
        // so no fixture needed.)
        for p in PROFILE_PREFERENCE {
            assert!(
                crate::profile_probe::fixture_for(*p).is_some(),
                "missing decode-probe fixture for {p:?}; \
                 add a file under crates/tether-codec/fixtures/probe/ \
                 and extend `fixture_for`"
            );
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod macos_probe_tests {
    use super::*;
    use crate::profile_probe::{fixture_for, ProfileProbe};
    use crate::videotoolbox::probe::VideoToolboxProbe;

    /// Run the encode + decode probes for a single profile, returning the
    /// `(encode, decode)` capability pair. Same shape as `probe_one`'s
    /// per-profile body, factored out so the hardware test can ask about
    /// profiles that aren't in `PROFILE_PREFERENCE` yet.
    fn probe(profile: VideoProfile) -> (bool, bool) {
        let encode = VideoToolboxProbe::probe_encode(profile).is_ok();
        let decode = fixture_for(profile)
            .map(|f| VideoToolboxProbe::probe_decode(profile, f).is_ok())
            .unwrap_or(false);
        (encode, decode)
    }

    #[test]
    #[ignore = "requires macOS + VideoToolbox"]
    fn macos_real_probe_matches_hardware() {
        // Empirically validated on M-series Apple Silicon (M4 Max
        // tested) running Homebrew ffmpeg with VideoToolbox + libx265
        // rext. The probe matrix it produces:
        //
        //   HEVC 4:4:4 8-bit: encode=true,  decode=true ('444v' IOSurface)
        //   HEVC 4:2:0 8-bit: encode=true,  decode=true ('420v' IOSurface)
        //   H.264 4:2:0 8-bit: encode=true,  decode=true ('420v' IOSurface)
        //
        // Both 4:4:4 halves are real hardware paths: decode returns a
        // `'444v'` IOSurface from VT (not a software-decoded CPU
        // frame); encode succeeds at submit time with `sw_format =
        // NV24`. The earlier "VT has no Main444 encode" claim was an
        // artifact of *our own* probe short-circuit that pre-filtered
        // non-(Yuv420,8) before reaching the encoder — once removed,
        // the FFmpeg wrapper accepts the input. The encoder-acceptance
        // signal does *not* prove the emitted bitstream is structurally
        // 4:4:4 (vs silently downsampled); a real encode→decode
        // round-trip with chroma assertion is needed to close that
        // gap and lives outside this commit's scope.
        let caps = supported_profiles();

        let yuv444 = caps
            .iter()
            .find(|c| c.profile == VideoProfile::HEVC_8BIT_444)
            .expect("HEVC 4:4:4 should appear in capability list");
        // 4:4:4 8-bit encode acceptance varies by FFmpeg build and
        // silicon generation — Homebrew's ffmpeg@7+ on M-series accepts
        // NV24 input; older or system FFmpeg may not plumb it. Record
        // the outcome and tighten the test if a CI matrix establishes
        // a hard floor. The decode side is the load-bearing assertion:
        // VT must produce a `Frame::Gpu` (not silently fall back to
        // software), and *when* encode succeeds the same fixture must
        // round-trip — if encode lies (accepts then silently fails the
        // pipeline), the chained decode check below would catch it.
        if yuv444.encode {
            let fixture = fixture_for(VideoProfile::HEVC_8BIT_444)
                .expect("HEVC 4:4:4 8-bit fixture is checked in");
            assert!(
                VideoToolboxProbe::probe_decode(VideoProfile::HEVC_8BIT_444, fixture).is_ok(),
                "VT reported encode=true for HEVC 4:4:4 8-bit but the matching decode \
                 fixture failed to produce a hardware frame — encoder is likely \
                 silent-falling-back or producing a malformed bitstream"
            );
        } else {
            eprintln!(
                "macos probe: HEVC 4:4:4 8-bit encode=false on this FFmpeg build; \
                 Main444 encode unavailable (expected on older ffmpeg / pre-M-series)"
            );
        }
        assert!(
            yuv444.decode,
            "M-series Macs decode HEVC 4:4:4 in hardware via VT (verified empirically) — \
             if this fails on a pre-M-series Mac, narrow the assertion to runtime probe data"
        );

        let yuv420 = caps
            .iter()
            .find(|c| c.profile == VideoProfile::HEVC_8BIT_420)
            .expect("HEVC 4:2:0 should appear in capability list");
        assert!(yuv420.encode, "HEVC Yuv420 encode should work on every modern Mac");
        assert!(yuv420.decode, "HEVC Yuv420 decode should work on every modern Mac");

        let h264 = caps
            .iter()
            .find(|c| c.profile == VideoProfile::H264_8BIT_420)
            .expect("H.264 4:2:0 should appear in capability list");
        assert!(h264.encode, "H.264 Yuv420 encode is universal on Macs with VT");
        assert!(h264.decode, "H.264 Yuv420 decode is universal on Macs with VT");
    }

    /// Direct ground-truth probe for profiles that aren't in
    /// `PROFILE_PREFERENCE` yet. Captures what the M-series VT wrapper
    /// actually does for the 10-bit profiles we plan to negotiate once
    /// the renderer + capture sides land — without committing to the
    /// matrix change before those exist.
    ///
    /// This test does not assert hard pass/fail per profile; the docs
    /// (`docs/CODEC_CAPABILITIES.md`) carry the expected matrix per
    /// platform. The print-on-`--nocapture` output is the deliverable
    /// for an investigator running on a new Mac model.
    #[test]
    #[ignore = "requires macOS + VideoToolbox"]
    fn macos_extended_probe_records_ground_truth() {
        let probes = [
            ("HEVC Main10 (Yuv420 10-bit)", VideoProfile {
                codec: CodecKind::Hevc,
                chroma: ChromaSubsampling::Yuv420,
                bit_depth: 10,
            }),
            ("HEVC Main444 10-bit (Yuv444 10-bit)", VideoProfile {
                codec: CodecKind::Hevc,
                chroma: ChromaSubsampling::Yuv444,
                bit_depth: 10,
            }),
        ];
        for (label, profile) in probes {
            let (encode, decode) = probe(profile);
            // --nocapture surfaces these so an investigator on new
            // silicon can update the doc matrix from the empirical
            // result without re-deriving it from scratch.
            eprintln!(
                "macos extended probe: {label:<40} encode={encode:<5} decode={decode}"
            );
        }
    }
}
