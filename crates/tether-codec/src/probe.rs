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
//! Resolution changes recreate the encoder via this same function, so
//! probe cost is paid per resize, not per frame.
//!
//! Codec negotiation: [`probe_encoder`] walks the client's
//! preferred-codec list and returns the first `(CodecKind, encoder)`
//! pair we can actually construct. The shape is registry-of-backends:
//! today only VAAPI on Linux, but VideoToolbox / Media Foundation slot
//! into the same iteration when their backends land. Trait-based
//! abstraction is deferred until a second concrete backend exists to
//! shape it against.

use crate::{CodecError, Decoder, Encoder, Result};
use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

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

/// Enumerate the video profiles this host can actually encode today.
///
/// Implemented as a real construction probe (same approach as
/// [`probe_encoder_kind`]) per `(codec, chroma, bit_depth)` triple —
/// VAAPI's runtime capability set depends on driver, kernel, and
/// hardware generation, none of which FFmpeg's build-time codec list
/// reflects accurately.
///
/// Today the constructor doesn't yet take a `VideoProfile`, so we only
/// probe `(_, Yuv420, 8)` triples. The Yuv444 probes light up in phase B
/// when [`crate::vaapi::VaapiEncoder::new`] grows the chroma parameter.
/// Listing the to-be-supported profiles here would lie to the
/// negotiator and cause a downstream encoder construction failure
/// mid-session.
#[must_use]
pub fn supported_encode_profiles() -> Vec<VideoProfile> {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Vec<VideoProfile>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            let mut out = Vec::new();
            for codec in [CodecKind::H264, CodecKind::Hevc] {
                let profile_420 = VideoProfile {
                    codec,
                    chroma: ChromaSubsampling::Yuv420,
                    bit_depth: 8,
                };
                if probe_encoder_kind(codec) {
                    out.push(profile_420);
                }
            }
            out
        })
        .clone()
}

/// Probe + construct an encoder for the first codec in `preferred`
/// that this host can actually build. Returns the chosen
/// [`CodecKind`] alongside the constructed encoder so the caller
/// (host send loop) can echo it back to the client through
/// [`tether_protocol::control::ServerHelloV1::chosen_codec`].
///
/// `fps` sets the encoder's time_base. `bitrate_kbps` is a soft VBR
/// target.
///
/// Errors with a diagnostics-friendly message if no codec in
/// `preferred` constructs successfully. An empty preference list is
/// also an error — the caller is expected to have validated this
/// upstream (the client's `preferred_codecs` defaults to a non-empty
/// list).
pub fn probe_encoder(
    preferred: &[CodecKind],
    width: u32,
    height: u32,
    fps: u32,
    bitrate_kbps: u32,
) -> Result<(CodecKind, Box<dyn Encoder>)> {
    if preferred.is_empty() {
        return Err(CodecError::NoHardwareCodec(
            "client preferred_codecs list was empty".to_string(),
        ));
    }

    #[cfg(target_os = "linux")]
    {
        let mut last_err: Option<(CodecKind, CodecError)> = None;
        for kind in preferred {
            match crate::vaapi::VaapiEncoder::new(*kind, width, height, fps, bitrate_kbps) {
                Ok(enc) => return Ok((*kind, Box::new(enc))),
                Err(e) => {
                    tracing::warn!(
                        backend = "vaapi",
                        codec = ?kind,
                        error = %e,
                        "VAAPI encoder construction failed for codec; trying next"
                    );
                    last_err = Some((*kind, e));
                }
            }
        }
        let (kind, src) = last_err.expect("loop entered with non-empty preferred");
        return Err(no_hw_encoder(kind, src));
    }

    #[cfg(target_os = "macos")]
    {
        let mut last_err: Option<(CodecKind, CodecError)> = None;
        for kind in preferred {
            match crate::videotoolbox::VideoToolboxEncoder::new(
                *kind,
                width,
                height,
                fps,
                bitrate_kbps,
            ) {
                Ok(enc) => return Ok((*kind, Box::new(enc))),
                Err(e) => {
                    tracing::warn!(
                        backend = "videotoolbox",
                        codec = ?kind,
                        error = %e,
                        "VideoToolbox encoder construction failed for codec; trying next"
                    );
                    last_err = Some((*kind, e));
                }
            }
        }
        let (kind, src) = last_err.expect("loop entered with non-empty preferred");
        return Err(no_hw_encoder_vt(kind, src));
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (preferred, width, height, fps, bitrate_kbps);
        Err(no_hw_encoder_for_platform())
    }
}

/// Lightweight handshake-time capability check: is `kind` a codec we
/// can *currently* build on this host? Implemented as a tiny
/// construction probe at 128×128 — captures real driver state (not
/// just FFmpeg build-time support), at the cost of a one-time VAAPI
/// device open. Caller is expected to invoke this once per session
/// during handshake, not per frame.
///
/// The 128×128 floor satisfies HEVC's minimum-block constraint on
/// Intel hardware (which rejects anything under 128×128 with EINVAL).
/// H.264 accepts smaller, but using the same dims keeps the probe a
/// single config across codecs.
///
/// Driver-portability caveat: rsmpeg's `AVCodecContext` exposes a
/// known failure mode where `encoder.open()` returning an error
/// leaves the context partially-initialized, and the subsequent Drop
/// segfaults (see the comment in `vaapi/encoder.rs` about the LP
/// entrypoint). We've validated this probe at 128×128 against H.264
/// and HEVC on Intel Arc (Meteor Lake). AMD and NVIDIA-via-VAAPI may
/// have different minimum block sizes for HEVC; if the probe ever
/// SIGSEGVs on a new driver, the principled fix is to add the
/// `vaQueryConfigProfiles` libva probe before construction. Today
/// we accept the risk because we don't have the test hardware.
pub fn probe_encoder_kind(kind: CodecKind) -> bool {
    #[cfg(target_os = "linux")]
    {
        crate::vaapi::VaapiEncoder::new(kind, 128, 128, 30, 1_000).is_ok()
    }
    #[cfg(target_os = "macos")]
    {
        // Apple Silicon h264/hevc_videotoolbox accept down to 128×128,
        // matching the VAAPI floor. Intel Macs (pre-M1) have been
        // observed to reject HEVC below ~144×144 — bumping to 256×144
        // gives headroom across both arches and still costs the probe
        // only a single one-shot encode. If the probe ever passes here
        // and `new()` then fails at a real resolution on an Intel mac,
        // raise the floor further; we don't have an Intel mac in CI to
        // validate against today.
        crate::videotoolbox::VideoToolboxEncoder::new(kind, 256, 144, 30, 1_000).is_ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = kind;
        false
    }
}

/// Probe + construct the decoder for the codec the host chose in its
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
        // VideoToolbox decoder is a follow-up plan. This is a
        // missing-feature, not a misconfiguration — make that
        // unambiguous in the error so operators don't chase a
        // permissions / driver problem that doesn't exist.
        let _ = kind;
        return Err(CodecError::NoHardwareCodec(
            "VideoToolbox decoder is not yet implemented in this build — \
             macOS client support is planned but not available. \
             Run tether-client on Linux (VAAPI) to receive from a macOS host."
                .to_string(),
        ));
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = kind;
        Err(no_hw_decoder_for_platform())
    }
}

#[cfg(target_os = "linux")]
fn no_hw_encoder(kind: CodecKind, source: CodecError) -> CodecError {
    let profile_hint = match kind {
        CodecKind::H264 => "VAProfileH264{ConstrainedBaseline,Main,High}",
        CodecKind::Hevc => "VAProfileHEVCMain",
        CodecKind::Av1 => "VAProfileAV1Profile0",
    };
    CodecError::NoHardwareCodec(format!(
        "VAAPI encoder unavailable for {kind:?} ({source}). \
         Check that /dev/dri/renderD128 is present and readable, and that `vainfo` \
         lists {profile_hint} with VAEntrypointEnc*. \
         Tether requires GPU encode — there is no software fallback."
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
        "Tether currently supports hardware decode on Linux (VAAPI). \
         macOS/VideoToolbox (client), Windows/NVDEC, and Windows/D3D11VA \
         backends are not yet implemented."
            .to_string(),
    )
}

#[cfg(test)]
mod negotiation_tests {
    use super::*;

    #[test]
    fn picks_best_mutual_from_preference() {
        // Both sides support all three; pref order wins -> HEVC 444.
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
        // Host has HEVC 4:4:4 but client only decodes H.264 4:2:0.
        // Negotiation must walk down to the floor rather than failing.
        let host = vec![
            VideoProfile::HEVC_8BIT_444,
            VideoProfile::H264_8BIT_420,
        ];
        let client = vec![VideoProfile::H264_8BIT_420];
        assert_eq!(
            pick_supported_profile(&host, &client),
            Some(VideoProfile::H264_8BIT_420)
        );
    }

    #[test]
    fn returns_none_when_disjoint() {
        // No mutual entry in the preference list — host advertises
        // a profile we know about (HEVC 4:2:0), client advertises a
        // profile not in PROFILE_PREFERENCE at all (a hypothetical AV1
        // 4:2:0). Caller treats None as a session-end signal.
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
        // Protocol contract: when the client doesn't send the
        // decode-profiles extension, the host treats them as a legacy
        // client supporting only the universal floor. The negotiation
        // must still pick that floor on a host that has richer caps —
        // otherwise pre-phase-A clients are silently broken when
        // connecting to a phase-A+ host.
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
}
