//! Client-side application handshake + negotiated session state.
//!
//! [`ClientSession::connect`] owns the post-QUIC handshake that used to
//! live inline in `apps/tether-client/src/main.rs`: building a
//! `ClientHello` with the structured decode-profile extension, awaiting
//! the `ServerHello`, resolving the negotiated [`VideoProfile`] from
//! either the structured echo or the legacy inline fields, cross-checking
//! the host's advertised pixel format against the negotiated chroma +
//! bit-depth (warn-only — the structured profile is authoritative), and
//! refusing the session if the host picked a profile this client never
//! advertised.
//!
//! The initial `ControlMessage::ForceIdr` is also sent here: the host's
//! encoder always emits IDR on its first frame, but capture can be
//! gated on a portal prompt or screen-recording permission dialog, so
//! we want the *next* frame to be a keyframe rather than dropping into
//! the host's P-frame chain mid-GOP if the capture stream was already
//! running.

use std::collections::BTreeMap;
use std::sync::Arc;

use tether_protocol::control::{
    is_known_bit_depth, ChromaSubsampling, ClientHello, ClientHelloV1, ClockSync, CodecKind,
    ControlMessage, PixelFormat, ServerHello, ServerHelloV1, VideoProfile,
    CLIENT_DECODE_PROFILES_EXTENSION_KEY, PIXEL_FORMAT_EXTENSION_KEY,
    SERVER_ENCODE_PROFILE_EXTENSION_KEY,
};
use tether_protocol::MonoNanos;
use tether_transport::{ControlChannel, TransportError};
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct ClientSessionConfig {
    /// Sent to the host as `ClientHelloV1::client_name`.
    pub client_name: String,
    /// What this client can decode. Becomes the
    /// `tether.cap.video.decode-profiles` extension payload and is
    /// retained on the returned session for the post-handshake
    /// validation against whatever profile the host picked.
    pub client_decode_profiles: Vec<VideoProfile>,
}

/// Per-connection state produced by a successful [`ClientSession::connect`].
///
/// Public fields by design — see the rationale on `HostSession`.
pub struct ClientSession {
    pub channel: Arc<dyn ControlChannel>,
    pub negotiated: VideoProfile,
    pub server_hello: ServerHelloV1,
    pub clock_sync: ClockSync,
    pub client_decode_profiles: Vec<VideoProfile>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    #[error("transport: {0}")]
    Transport(#[from] TransportError),

    /// The host's `tether.cap.video.encode-profile` extension was
    /// present but didn't decode as a [`VideoProfile`]. We refuse to
    /// fall back to the inline 8-bit assumption — a future 10-bit
    /// host with an unexpected payload shape would silently downgrade.
    #[error(
        "host encode-profile extension failed to decode \
         ({source}, payload_len={payload_len}); refusing to fall back to \
         the inline 8-bit assumption"
    )]
    InvalidEncodeProfile {
        source: tether_protocol::CodecError,
        payload_len: usize,
    },

    /// The host picked a profile this client never advertised. Either
    /// a buggy host or a hostile peer; either way we end the session
    /// rather than try to render with the wrong pipeline.
    #[error(
        "host chose profile {chosen:?} which this client did not advertise \
         ({} entries in client_decode_profiles)", .advertised.len()
    )]
    ProfileNotAdvertised {
        chosen: VideoProfile,
        advertised: Vec<VideoProfile>,
    },

    /// The host chose a profile with a `bit_depth` this build doesn't
    /// know how to render. Could be a future host advertising 12-bit,
    /// or a malformed peer. Either way we can't build the pipeline.
    #[error("host chose profile with unknown bit_depth {0}; expected one of {1:?}")]
    UnknownBitDepth(u8, &'static [u8]),
}

impl ClientSession {
    /// Send the [`ClientHello`], await the [`ServerHello`], resolve the
    /// negotiated profile, validate it against this client's advert,
    /// and prime the host with a `ForceIdr` so the next frame is a
    /// keyframe.
    ///
    /// The connection is assumed already established (QUIC + TLS done).
    /// On any error from this function the connection is left to the
    /// caller to close.
    pub async fn connect(
        channel: Arc<dyn ControlChannel>,
        cfg: ClientSessionConfig,
    ) -> Result<Self, ConnectError> {
        let mut extensions = BTreeMap::new();
        extensions.insert(
            CLIENT_DECODE_PROFILES_EXTENSION_KEY.to_string(),
            tether_protocol::encode(&cfg.client_decode_profiles)
                .expect("VideoProfile list encodes; types under our control"),
        );
        info!(
            client_decode_profiles = ?cfg.client_decode_profiles,
            "advertising video decode capabilities to host"
        );

        let hello = ClientHello::V1(ClientHelloV1 {
            client_name: cfg.client_name,
            // Legacy inline advert for hosts predating the structured
            // extension. The structured one is authoritative on any
            // post-phase-A host. HEVC first because legacy hosts pick
            // the first they can build.
            preferred_codecs: vec![CodecKind::Hevc, CodecKind::H264],
            max_resolution: None,
            // Stamped by `client_handshake` immediately before the
            // send; the value here is overwritten.
            clock_probe_t0: MonoNanos::ZERO,
            extensions,
            resume_token: None,
        });

        let (server_hello, clock_sync) = channel.client_handshake(hello).await?;
        // Newer hosts may send a future ServerHello variant we don't
        // know; bincode's enum-variant decode catches that as an error
        // upstream. Reaching here means the variant decoded; match
        // exhaustively so a future V2 forces a compile-time update.
        let ServerHello::V1(server_body) = server_hello;

        let negotiated = resolve_negotiated_profile(
            server_body
                .extensions
                .get(SERVER_ENCODE_PROFILE_EXTENSION_KEY)
                .map(Vec::as_slice),
            server_body.chosen_codec,
            server_body.chosen_chroma,
        )?;
        if !is_known_bit_depth(negotiated.bit_depth) {
            return Err(ConnectError::UnknownBitDepth(
                negotiated.bit_depth,
                tether_protocol::control::KNOWN_BIT_DEPTHS,
            ));
        }
        info!(
            server = %server_body.server_name,
            negotiated_codec = ?negotiated.codec,
            negotiated_chroma = ?negotiated.chroma,
            negotiated_bit_depth = negotiated.bit_depth,
            resolution = ?server_body.resolution,
            rtt_us = clock_sync.rtt_nanos / 1_000,
            clock_offset_us = clock_sync.offset_nanos / 1_000,
            "video profile negotiated; handshake complete"
        );

        cross_check_pixel_format(&server_body, negotiated);

        // Validate the host's pick before signalling any session
        // intent. Sending `ForceIdr` first would tell a misbehaving
        // host to start producing keyframes for a session we're about
        // to refuse — wasted work and a minor protocol-shape lie
        // ("we accepted this session") in the same breath as our
        // refusal.
        if !cfg.client_decode_profiles.contains(&negotiated) {
            return Err(ConnectError::ProfileNotAdvertised {
                chosen: negotiated,
                advertised: cfg.client_decode_profiles,
            });
        }

        // First IDR request goes out immediately after handshake so
        // capture-gated hosts (portal prompt, TCC dialog) still produce
        // a keyframe on first delivered frame instead of dropping the
        // client into a partial GOP.
        if let Err(e) = channel.send_control(&ControlMessage::ForceIdr).await {
            warn!(error = ?e, "initial ForceIdr send failed; continuing anyway");
        }

        Ok(Self {
            channel,
            negotiated,
            server_hello: server_body,
            clock_sync,
            client_decode_profiles: cfg.client_decode_profiles,
        })
    }
}

/// Decode the structured `tether.cap.video.encode-profile` extension or
/// — for legacy hosts that don't emit it — synthesize an 8-bit profile
/// from the inline `chosen_codec` / `chosen_chroma` fields. An
/// undecodable extension is a hard error rather than a silent fall
/// back to 8-bit, because a future 10-bit host whose payload shape we
/// don't understand would otherwise downgrade itself invisibly.
fn resolve_negotiated_profile(
    extension_bytes: Option<&[u8]>,
    chosen_codec: CodecKind,
    chosen_chroma: ChromaSubsampling,
) -> Result<VideoProfile, ConnectError> {
    match extension_bytes {
        Some(bytes) => {
            tether_protocol::decode::<VideoProfile>(bytes).map_err(|e| {
                ConnectError::InvalidEncodeProfile {
                    source: e,
                    payload_len: bytes.len(),
                }
            })
        }
        None => Ok(VideoProfile {
            codec: chosen_codec,
            chroma: chosen_chroma,
            bit_depth: 8,
        }),
    }
}

/// Warn-only check that the host's pixel-format advert (an extension)
/// agrees with the negotiated chroma + bit-depth. A mismatch doesn't
/// gate the session — the structured profile extension is the
/// authoritative source — but it does indicate a host whose two
/// advertisements disagree, which is a real correctness signal.
fn cross_check_pixel_format(server_body: &ServerHelloV1, negotiated: VideoProfile) {
    let Some(pf_bytes) = server_body.extensions.get(PIXEL_FORMAT_EXTENSION_KEY) else {
        return;
    };
    let pf = match tether_protocol::decode::<PixelFormat>(pf_bytes) {
        Ok(pf) => pf,
        Err(e) => {
            warn!(error = %e, "host advertised an unparseable pixel format extension");
            return;
        }
    };
    let expected = match (negotiated.chroma, negotiated.bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => Some(PixelFormat::Nv12),
        (ChromaSubsampling::Yuv420, 10) => Some(PixelFormat::P010),
        (ChromaSubsampling::Yuv444, 8) => Some(PixelFormat::Yuv444p),
        (ChromaSubsampling::Yuv444, 10) => Some(PixelFormat::P410),
        _ => None,
    };
    match expected {
        Some(exp) if pf == exp => {
            tracing::debug!(?pf, "host pixel-format extension matches negotiated chroma + bit_depth");
        }
        Some(exp) => {
            warn!(
                advertised = ?pf,
                expected = ?exp,
                negotiated_chroma = ?negotiated.chroma,
                negotiated_bit_depth = negotiated.bit_depth,
                "host pixel-format extension disagrees with negotiated chroma + bit_depth; \
                 trusting the negotiated profile"
            );
        }
        None => {
            tracing::debug!(
                ?pf,
                negotiated_chroma = ?negotiated.chroma,
                negotiated_bit_depth = negotiated.bit_depth,
                "negotiated chroma/bit_depth combo not in pixel-format cross-check table; skipping"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_present_and_decodes_is_authoritative() {
        let profile = VideoProfile::HEVC_10BIT_444;
        let bytes = tether_protocol::encode(&profile).unwrap();
        // Inline fields deliberately disagree — the extension wins.
        let resolved = resolve_negotiated_profile(
            Some(&bytes),
            CodecKind::H264,
            ChromaSubsampling::Yuv420,
        )
        .unwrap();
        assert_eq!(resolved, profile);
    }

    #[test]
    fn extension_absent_synthesizes_8bit_legacy_profile() {
        let resolved =
            resolve_negotiated_profile(None, CodecKind::Hevc, ChromaSubsampling::Yuv420).unwrap();
        assert_eq!(resolved.codec, CodecKind::Hevc);
        assert_eq!(resolved.chroma, ChromaSubsampling::Yuv420);
        assert_eq!(resolved.bit_depth, 8);
    }

    #[test]
    fn unknown_bit_depth_is_rejected_via_helper() {
        // is_known_bit_depth is the boundary check; assert it rejects
        // values outside {8, 10}. The full session-level integration
        // (host picks a 12-bit profile, client refuses) is exercised
        // indirectly through ConnectError::UnknownBitDepth — which
        // requires a real handshake to surface — so the unit test
        // here pins the predicate.
        assert!(tether_protocol::control::is_known_bit_depth(8));
        assert!(tether_protocol::control::is_known_bit_depth(10));
        assert!(!tether_protocol::control::is_known_bit_depth(0));
        assert!(!tether_protocol::control::is_known_bit_depth(9));
        assert!(!tether_protocol::control::is_known_bit_depth(12));
        assert!(!tether_protocol::control::is_known_bit_depth(255));
    }

    #[test]
    fn extension_undecodable_returns_error_not_fallback() {
        let garbage = [0xFFu8; 3];
        let err = resolve_negotiated_profile(
            Some(&garbage),
            CodecKind::Hevc,
            ChromaSubsampling::Yuv420,
        )
        .expect_err("garbage payload should error, not 8-bit fallback");
        assert!(matches!(err, ConnectError::InvalidEncodeProfile { .. }));
    }
}
