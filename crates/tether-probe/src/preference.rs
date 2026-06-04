//! Negotiation preference order + selector.
//!
//! Lifted here from `tether-codec::probe` because preference order and
//! negotiation are probe-domain concepts: they encode opinions about
//! *which capabilities to prefer when both ends support multiple*, not
//! anything intrinsic to a codec library. Keeping them next to the
//! capability data they consume avoids cross-crate hops.
//!
//! Pinned by `preference_order_is_desktop_quality_first` —
//! reorderings are a behavioural change, not a refactor.

use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

/// Dev/test override for forcing one negotiated profile when both
/// sides actually support it. Values are lowercase
/// `<codec>-<bit-depth>-<chroma>`, e.g. `av1-10-420`,
/// `hevc-10-444`, or `h264-8-420`.
pub const FORCE_VIDEO_PROFILE_ENV: &str = "TETHER_FORCE_VIDEO_PROFILE";

/// Host preference order, best-first. The negotiator picks the first
/// entry that appears in *both* the host's encode capabilities and
/// the client's advertised decode capabilities.
///
/// Anchored to desktop content quality. 4:4:4 goes ahead of 4:2:0
/// because it preserves antialiased text and UI chroma detail that
/// 4:2:0 visibly smears; 10-bit goes ahead of 8-bit at each chroma
/// rung because the extra precision suppresses gradient banding on
/// desktop content where flat colour fields are common. The probe
/// layer filters out entries this hardware can't deliver, so the
/// preference list can stay aspirational without breaking sessions
/// on lower-capability hardware.
///
/// AV1 sits between HEVC 4:4:4 and HEVC 4:2:0: it has no 4:4:4 rung
/// in this build (encoder support is sparse), so it can't displace
/// HEVC 4:4:4 for desktop-text fidelity, but at the 4:2:0 rung AV1
/// delivers the same quality as HEVC at ~30% less bitrate — worth
/// preferring whenever both ends advertise it.
///
/// H.264 4:4:4 is intentionally absent — VAAPI has no encode profile
/// for it, and no other host backend in this build supports it yet.
/// Yuv422 is likewise absent until the wire-side `ChromaSubsampling`
/// enum grows the variant.
pub const PROFILE_PREFERENCE: &[VideoProfile] = &[
    VideoProfile::HEVC_10BIT_444,
    VideoProfile::HEVC_8BIT_444,
    VideoProfile::AV1_10BIT_420,
    VideoProfile::AV1_8BIT_420,
    VideoProfile::HEVC_10BIT_420,
    VideoProfile::HEVC_8BIT_420,
    VideoProfile::H264_8BIT_420,
];

/// Best mutual profile between the host's encode capabilities and the
/// client's decode capabilities, picked from [`PROFILE_PREFERENCE`].
/// Returns `None` only when no preference-list entry appears in both
/// sets — the caller treats that as a session-end condition (no
/// compatible codec).
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

/// Parse [`FORCE_VIDEO_PROFILE_ENV`] if present.
///
/// This is deliberately small and explicit: it is a test/development
/// knob, not a user-facing codec policy language.
pub fn forced_video_profile_from_env() -> Result<Option<VideoProfile>, String> {
    match std::env::var(FORCE_VIDEO_PROFILE_ENV) {
        Ok(raw) if raw.trim().is_empty() => Ok(None),
        Ok(raw) => parse_forced_video_profile(&raw).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(format!("{FORCE_VIDEO_PROFILE_ENV} is not valid UTF-8"))
        }
    }
}

/// Pick `forced` when provided and mutually supported; otherwise use
/// normal preference order.
#[must_use]
pub fn pick_supported_profile_with_force(
    host_caps: &[VideoProfile],
    client_caps: &[VideoProfile],
    forced: Option<VideoProfile>,
) -> Option<VideoProfile> {
    if let Some(profile) = forced {
        return host_caps
            .contains(&profile)
            .then_some(profile)
            .filter(|p| client_caps.contains(p));
    }
    pick_supported_profile(host_caps, client_caps)
}

/// Parse compact values accepted by [`FORCE_VIDEO_PROFILE_ENV`].
pub fn parse_forced_video_profile(raw: &str) -> Result<VideoProfile, String> {
    let normalized = raw.trim().to_ascii_lowercase().replace('_', "-");
    let mut parts = normalized.split('-');
    let codec = match parts.next() {
        Some("h264" | "h.264" | "avc") => CodecKind::H264,
        Some("hevc" | "h265" | "h.265") => CodecKind::Hevc,
        Some("av1") => CodecKind::Av1,
        _ => {
            return Err(format!(
                "unsupported codec in {FORCE_VIDEO_PROFILE_ENV}={raw:?}"
            ))
        }
    };
    let bit_depth = match parts.next().and_then(|p| p.parse::<u8>().ok()) {
        Some(depth @ (8 | 10)) => depth,
        _ => Err(format!(
            "expected bit depth 8 or 10 in {FORCE_VIDEO_PROFILE_ENV}={raw:?}"
        ))?,
    };
    let chroma = match parts.next() {
        Some("420" | "yuv420") => ChromaSubsampling::Yuv420,
        Some("444" | "yuv444") => ChromaSubsampling::Yuv444,
        _ => {
            return Err(format!(
                "expected chroma 420 or 444 in {FORCE_VIDEO_PROFILE_ENV}={raw:?}"
            ))
        }
    };
    if parts.next().is_some() {
        return Err(format!(
            "expected <codec>-<bit-depth>-<chroma> in {FORCE_VIDEO_PROFILE_ENV}={raw:?}"
        ));
    }

    match (codec, chroma, bit_depth) {
        (CodecKind::H264, ChromaSubsampling::Yuv420, 8) => Ok(VideoProfile::H264_8BIT_420),
        (CodecKind::Hevc, ChromaSubsampling::Yuv420, 8) => Ok(VideoProfile::HEVC_8BIT_420),
        (CodecKind::Hevc, ChromaSubsampling::Yuv420, 10) => Ok(VideoProfile::HEVC_10BIT_420),
        (CodecKind::Hevc, ChromaSubsampling::Yuv444, 8) => Ok(VideoProfile::HEVC_8BIT_444),
        (CodecKind::Hevc, ChromaSubsampling::Yuv444, 10) => Ok(VideoProfile::HEVC_10BIT_444),
        (CodecKind::Av1, ChromaSubsampling::Yuv420, 8) => Ok(VideoProfile::AV1_8BIT_420),
        (CodecKind::Av1, ChromaSubsampling::Yuv420, 10) => Ok(VideoProfile::AV1_10BIT_420),
        _ => Err(format!(
            "{FORCE_VIDEO_PROFILE_ENV}={raw:?} is not in PROFILE_PREFERENCE"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tether_protocol::control::{ChromaSubsampling, CodecKind};

    #[test]
    fn preference_order_is_desktop_quality_first() {
        assert_eq!(PROFILE_PREFERENCE[0], VideoProfile::HEVC_10BIT_444);
        assert_eq!(PROFILE_PREFERENCE[1], VideoProfile::HEVC_8BIT_444);
        assert_eq!(PROFILE_PREFERENCE[2], VideoProfile::AV1_10BIT_420);
        assert_eq!(PROFILE_PREFERENCE[3], VideoProfile::AV1_8BIT_420);
        assert_eq!(PROFILE_PREFERENCE[4], VideoProfile::HEVC_10BIT_420);
        assert_eq!(PROFILE_PREFERENCE[5], VideoProfile::HEVC_8BIT_420);
        assert_eq!(PROFILE_PREFERENCE[6], VideoProfile::H264_8BIT_420);
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
    fn future_bit_depth_does_not_match_known_one() {
        let host = vec![VideoProfile::HEVC_8BIT_420];
        let future_client = vec![VideoProfile {
            codec: CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 12,
        }];
        assert_eq!(pick_supported_profile(&host, &future_client), None);
    }

    #[test]
    fn forced_profile_parser_accepts_compact_names() {
        assert_eq!(
            parse_forced_video_profile("av1-10-420").unwrap(),
            VideoProfile::AV1_10BIT_420
        );
        assert_eq!(
            parse_forced_video_profile("hevc_8_444").unwrap(),
            VideoProfile::HEVC_8BIT_444
        );
        assert_eq!(
            parse_forced_video_profile("h264-8-yuv420").unwrap(),
            VideoProfile::H264_8BIT_420
        );
    }

    #[test]
    fn forced_profile_parser_rejects_unpreferred_profiles() {
        assert!(parse_forced_video_profile("av1-8-444").is_err());
        assert!(parse_forced_video_profile("h264-10-420").is_err());
        assert!(parse_forced_video_profile("hevc-12-420").is_err());
    }

    #[test]
    fn force_selector_requires_both_sides() {
        let host = vec![VideoProfile::HEVC_10BIT_444, VideoProfile::AV1_10BIT_420];
        let client = vec![VideoProfile::AV1_10BIT_420, VideoProfile::H264_8BIT_420];
        assert_eq!(
            pick_supported_profile_with_force(&host, &client, Some(VideoProfile::AV1_10BIT_420)),
            Some(VideoProfile::AV1_10BIT_420)
        );
        assert_eq!(
            pick_supported_profile_with_force(&host, &client, Some(VideoProfile::HEVC_10BIT_444)),
            None
        );
        assert_eq!(
            pick_supported_profile_with_force(&host, &client, None),
            Some(VideoProfile::AV1_10BIT_420)
        );
    }
}
