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

use tether_protocol::control::VideoProfile;

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
}
