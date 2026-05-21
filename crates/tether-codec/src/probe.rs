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
/// Probes every entry in [`PROFILE_PREFERENCE`] (skipping any that the
/// driver rejects — Main444 on Tiger Lake- Intel, for example) plus
/// the 4:2:0 H.264 floor, in deterministic order. Output order is
/// preserved so the negotiator's `contains` check sees the same list
/// shape regardless of which driver / build is running.
#[must_use]
pub fn supported_encode_profiles() -> Vec<VideoProfile> {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Vec<VideoProfile>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            // Walk PROFILE_PREFERENCE so we never advertise a profile
            // that isn't in the negotiator's preference list. The
            // (HEVC, Yuv444) probe is the load-bearing addition here:
            // it lights up the desktop-quality rung on hosts that
            // can actually do it and stays absent elsewhere.
            PROFILE_PREFERENCE
                .iter()
                .copied()
                .filter(|p| probe_encoder_profile(*p))
                .collect()
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
        // VideoToolbox doesn't yet take VideoProfile (4:4:4 / Main444
        // path on Apple Silicon is separate work). The macOS arm of
        // `probe_encoder_profile` already returns false for non-Yuv420
        // profiles, so the negotiator never picks one — but this
        // function is the next layer of defense, in case the gate
        // ever moves. Fail loudly rather than silently mis-encoding.
        assert_eq!(
            profile.chroma,
            ChromaSubsampling::Yuv420,
            "VideoToolbox encoder only handles Yuv420 today; \
             probe_encoder_profile should have rejected {:?}",
            profile.chroma
        );
        assert_eq!(
            profile.bit_depth, 8,
            "VideoToolbox encoder only handles 8-bit today; got {}-bit",
            profile.bit_depth
        );
        match crate::videotoolbox::VideoToolboxEncoder::new(
            profile.codec,
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
    probe_encoder_profile(VideoProfile {
        codec: kind,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 8,
    })
}

/// Profile-level capability probe: construct + tear down an encoder
/// at the given profile to learn whether this driver actually supports
/// it. Same 128×128 floor as [`probe_encoder_kind`]; same Drop-on-failure
/// caveat.
///
/// 4:4:4 probes go through here. `(Hevc, Yuv444, 8)` builds successfully
/// on Intel Tiger Lake+ and AMD VCN3+; the open() returns an error
/// elsewhere and we filter the profile out of [`supported_encode_profiles`].
#[must_use]
pub fn probe_encoder_profile(profile: VideoProfile) -> bool {
    #[cfg(target_os = "linux")]
    {
        crate::vaapi::VaapiEncoder::new(profile, 128, 128, 30, 1_000).is_ok()
    }
    #[cfg(target_os = "macos")]
    {
        // VideoToolbox 4:4:4 isn't wired yet — refuse to advertise it
        // so the negotiator doesn't pick a profile we can't actually
        // construct end-to-end. Yuv420 8-bit is the only macOS profile
        // we support today.
        if profile.chroma != ChromaSubsampling::Yuv420 || profile.bit_depth != 8 {
            return false;
        }
        crate::videotoolbox::VideoToolboxEncoder::new(profile.codec, 256, 144, 30, 1_000).is_ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = profile;
        false
    }
}

/// Enumerate the video profiles this client can actually decode
/// today. Used to populate the `tether.cap.video.decode-profiles`
/// hello extension so the host's negotiator never picks a profile we
/// can't construct.
///
/// On Linux this is every entry in [`PROFILE_PREFERENCE`] — VAAPI
/// covers H.264 4:2:0, HEVC 4:2:0, and HEVC 4:4:4 8-bit on Tiger
/// Lake+ Intel / VCN3+ AMD. On macOS this is Yuv420 only: FFmpeg's
/// VideoToolbox HEVC wrapper does not expose Main444 / Rext decode
/// even on Apple Silicon Media Engines that can technically do it,
/// and a 4:4:4 advert here would only set us up to fail at decoder
/// construction time. Filtered through the same probe-and-tear-down
/// pattern as [`supported_encode_profiles`] so a runtime gap (missing
/// FFmpeg hwaccel, codec not compiled in) is caught at startup
/// rather than at first-frame time.
#[must_use]
pub fn supported_decode_profiles() -> Vec<VideoProfile> {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Vec<VideoProfile>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            PROFILE_PREFERENCE
                .iter()
                .copied()
                .filter(|p| probe_decoder_profile(*p))
                .collect()
        })
        .clone()
}

/// Profile-level decode probe. Mirrors [`probe_encoder_profile`]:
/// real `Decoder::new` construction so driver-level gaps surface here
/// rather than at first-frame time. On macOS the 4:4:4 path is
/// gated off up front (see [`supported_decode_profiles`] for the
/// rationale).
///
/// Linux caveat: `VaapiDecoder::new` is codec-keyed, not profile-keyed,
/// so this probe returns `true` for HEVC 4:4:4 as long as a basic HEVC
/// decoder constructs — the chroma is not actually verified at probe
/// time. In practice, consumer VAAPI drivers always expose decode for
/// the same chromas they expose encode for (decode being the cheaper
/// path), and HEVC 4:4:4 decode failures would surface as `Err` from
/// the first `receive_frame` rather than as silent corruption. A
/// follow-up that constructs a tiny 4:4:4 test bitstream and runs it
/// through the decoder would close this gap. The macOS arm has no such
/// hole because it gates on chroma directly before constructing.
#[must_use]
pub fn probe_decoder_profile(profile: VideoProfile) -> bool {
    #[cfg(target_os = "linux")]
    {
        crate::vaapi::VaapiDecoder::new(profile.codec).is_ok()
    }
    #[cfg(target_os = "macos")]
    {
        if profile.chroma != ChromaSubsampling::Yuv420 || profile.bit_depth != 8 {
            return false;
        }
        crate::videotoolbox::VideoToolboxDecoder::new(profile.codec).is_ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = profile;
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

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires macOS + VideoToolbox"]
    fn macos_advertises_yuv420_only() {
        // VideoToolbox has no HEVC Main444 hardware path (and never has,
        // up through M4 generation as of this writing). The negotiator
        // must never see a 4:4:4 profile from a macOS host, or it will
        // pick something we can't construct end-to-end. This nails the
        // gate down so a refactor of `probe_encoder_profile` that
        // accidentally lifts the macOS check gets caught at test time.
        let advertised = supported_encode_profiles();
        for p in &advertised {
            assert_eq!(
                p.chroma,
                ChromaSubsampling::Yuv420,
                "macOS host advertised non-Yuv420 profile {p:?}"
            );
            assert_eq!(p.bit_depth, 8, "macOS host advertised >8-bit profile {p:?}");
        }
        // Hevc-420 should be available on any modern Apple Silicon
        // device. H.264-420 on every Mac that ships VideoToolbox at
        // all. If neither appears the linked FFmpeg isn't built with
        // VideoToolbox — fail loudly rather than silently passing.
        assert!(
            advertised.contains(&VideoProfile::HEVC_8BIT_420)
                || advertised.contains(&VideoProfile::H264_8BIT_420),
            "neither HEVC nor H.264 Yuv420 probed successfully on macOS \
             (FFmpeg build missing VideoToolbox?): got {advertised:?}"
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
