//! Host-side application handshake + per-session lifecycle state.
//!
//! [`HostSession::accept`] owns the parts of the post-QUIC handshake that
//! used to live inline in `apps/tether-host/src/main.rs`: parsing the
//! client's structured decode-profile advert, picking the mutually
//! supported [`VideoProfile`] via the caller-supplied selector, building
//! the [`ServerHello`] extension map (pixel format + echo of the chosen
//! profile), and — when the two sides have no codec in common — sending
//! a clean [`ControlMessage::Goodbye`] before returning.
//!
//! The selector is a callback rather than a baked-in call into
//! `tether-probe` so this crate doesn't pull capture/codec/GPU
//! dependencies in transitively. Typical caller writes:
//!
//! ```ignore
//! HostSession::accept(conn, cfg, |client_caps| {
//!     tether_probe::pick_supported_profile(&host_encode_profiles, client_caps)
//! }).await?
//! ```
//!
//! The session also constructs the per-connection [`IdrSignal`] and
//! `stream_ready` `AtomicBool` so platform-specific orchestration code
//! that follows (capture, encode, recv tasks) can clone them without
//! re-deriving the lifecycle.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use tether_protocol::control::{
    is_known_bit_depth, ChromaSubsampling, ClientHello, ClientHelloV1, CodecKind, ControlMessage,
    GoodbyeCode, PixelFormat, ServerHello, ServerHelloV1, VideoColorSpec, VideoProfile,
    CLIENT_DECODE_PROFILES_EXTENSION_KEY, PIXEL_FORMAT_EXTENSION_KEY,
    SERVER_ENCODE_PROFILE_EXTENSION_KEY,
};
use tether_protocol::MonoNanos;
use tether_transport::{Connection, TransportError};
use tracing::{info, warn};

use crate::idr::IdrSignal;

/// Upper bound on the decoded length of the client's
/// `tether.cap.video.decode-profiles` extension payload. A legitimate
/// list is a handful of entries (per-`VideoProfile` bincode overhead is
/// ~5–7 bytes); 256 bytes accommodates dozens with headroom while
/// preventing a hostile peer from forcing a multi-KiB `Vec` allocation
/// just by setting a large length prefix.
const MAX_DECODE_PROFILES_BYTES: usize = 256;

/// Universal floor advertised by clients predating the structured
/// decode-profiles extension. Same value as the lowest entry of
/// `tether_probe::PROFILE_PREFERENCE`.
const LEGACY_DECODE_PROFILE_FLOOR: VideoProfile = VideoProfile::H264_8BIT_420;

#[derive(Debug, Clone)]
pub struct HostSessionConfig {
    /// Sent back to the client as `ServerHelloV1::server_name`.
    pub server_name: String,
}

/// Per-connection state produced by a successful [`HostSession::accept`].
///
/// Field-level public access is intentional — this seam is internal to
/// the workspace and accessors would just be noise. Each field maps 1:1
/// to a piece of state the host orchestration code already needed:
///
/// - `conn`: same `Arc<Connection>` the caller handed in. Cheap to clone
///   into the spawned recv/send tasks.
/// - `negotiated`: the chosen [`VideoProfile`]. Drives encoder build,
///   gpuconvert bridge selection, and per-frame metadata.
/// - `client_hello` / `client_decode_profiles`: the structured client
///   advert + parsed decode caps; held so the caller can log them or
///   feed them into downstream policy.
/// - `idr_signal`: shared coalesced "force keyframe" flag. The control
///   recv task raises it on `ControlMessage::ForceIdr`; the encode loop
///   takes it once per frame.
/// - `stream_ready`: client signalled it has built its decoders. The
///   send loop drops captured frames until this flips true so the
///   first ~hundreds of ms of frames don't race the client's decoder
///   construction.
pub struct HostSession {
    pub conn: Arc<Connection>,
    pub negotiated: VideoProfile,
    pub client_hello: ClientHelloV1,
    pub client_decode_profiles: Vec<VideoProfile>,
    pub idr_signal: IdrSignal,
    pub stream_ready: Arc<AtomicBool>,
}

#[derive(Debug, thiserror::Error)]
pub enum AcceptError {
    #[error("transport: {0}")]
    Transport(#[from] TransportError),

    /// The selector returned `None` — host encode capabilities and the
    /// client's advertised decode capabilities have no entry in common.
    /// A `Goodbye(InternalError)` has already been sent to the client
    /// before this error is returned; the caller's only remaining job
    /// is to drop the connection and (in a reconnect loop) accept the
    /// next client. The host's own list isn't carried — the caller
    /// owns it (it built the selector) and can log it alongside.
    #[error(
        "no video profile intersects host encode + client decode capabilities \
         (client: {client:?})"
    )]
    NoProfileIntersection { client: Vec<VideoProfile> },
}

impl HostSession {
    /// Run the application-layer handshake and produce a [`HostSession`].
    ///
    /// The `selector` closure is called inside the handshake's `build`
    /// step with the client's advertised decode profiles and must return
    /// the host's chosen profile (or `None` for no match). Keep it
    /// **fast** (sub-millisecond) — the `host_handshake` performance
    /// contract documents how slow work in `build` biases the
    /// `ClockSync` offset. `pick_supported_profile` is fine; anything
    /// that touches hardware is not.
    ///
    /// On `Err(NoProfileIntersection)` this function has already sent a
    /// clean `Goodbye(InternalError)` to the client. On
    /// `Err(Transport(_))` no goodbye was sent (the connection is the
    /// problem). On `Ok`, the QUIC connection is alive, the handshake
    /// is complete, and the caller can wire up capture / encode / recv
    /// tasks against the returned session.
    pub async fn accept<S>(
        conn: Arc<Connection>,
        cfg: HostSessionConfig,
        selector: S,
    ) -> Result<Self, AcceptError>
    where
        S: FnOnce(&[VideoProfile]) -> Option<VideoProfile>,
    {
        // The handshake closure isn't async, so we stash its outputs
        // through `Option`s captured by `&mut`. The closure runs exactly
        // once (`host_handshake` takes `FnOnce`), so the `Option::take`
        // pattern here can't drop work on the floor.
        let mut chosen_profile_out: Option<VideoProfile> = None;
        let mut client_decode_profiles_out: Option<Vec<VideoProfile>> = None;
        let mut selector_slot = Some(selector);

        let client_hello = conn
            .host_handshake(|hello| {
                let ClientHello::V1(body) = hello;
                let client_caps = parse_client_decode_profiles(body);
                info!(
                    client_decode_profiles = ?client_caps,
                    legacy_preferred_codecs = ?body.preferred_codecs,
                    "client video decode capabilities (parsed from hello)"
                );

                // Hand the caller's selector exactly the inputs it needs.
                // Take the closure out of the slot so we don't violate
                // FnOnce; `host_handshake` itself only invokes `build`
                // once, so the slot is guaranteed populated here.
                let selector = selector_slot
                    .take()
                    .expect("host_handshake invokes build exactly once");
                let chosen = selector(&client_caps);
                chosen_profile_out = chosen;
                client_decode_profiles_out = Some(client_caps);

                ServerHello::V1(build_server_hello(&cfg, chosen))
            })
            .await?;

        let ClientHello::V1(client_body) = client_hello;
        let client_decode_profiles = client_decode_profiles_out
            .expect("host_handshake build closure populates client_decode_profiles_out");

        let chosen_profile = match chosen_profile_out {
            Some(p) => p,
            None => {
                warn!(
                    client_decode_profiles = ?client_decode_profiles,
                    "no video profile intersects host encode + client decode capabilities; \
                     sending Goodbye(InternalError) and ending session"
                );
                // Best-effort goodbye. If this fails the connection is
                // already gone — either way we're returning an error
                // and the caller drops `conn`.
                let _ = conn
                    .send_control(&ControlMessage::Goodbye {
                        reason: "host and client video capabilities do not intersect".to_string(),
                        code: GoodbyeCode::InternalError,
                    })
                    .await;
                return Err(AcceptError::NoProfileIntersection {
                    client: client_decode_profiles,
                });
            }
        };

        info!(
            client = %client_body.client_name,
            chosen_codec = ?chosen_profile.codec,
            chosen_chroma = ?chosen_profile.chroma,
            chosen_bit_depth = chosen_profile.bit_depth,
            max_resolution = ?client_body.max_resolution,
            "video profile negotiated; handshake complete"
        );

        Ok(Self {
            conn,
            negotiated: chosen_profile,
            client_hello: client_body,
            client_decode_profiles,
            idr_signal: IdrSignal::new(),
            stream_ready: Arc::new(AtomicBool::new(false)),
        })
    }
}

/// Parse the client's structured decode-profile advert out of the hello
/// extension map. Falls back to the universal floor on absence, decode
/// failure, or over-size payload — none of which should drop the
/// session (a legacy client that hadn't yet learned this extension is
/// expected, and a malformed extension shouldn't trump the rest of the
/// handshake either).
fn parse_client_decode_profiles(body: &ClientHelloV1) -> Vec<VideoProfile> {
    let Some(bytes) = body.extensions.get(CLIENT_DECODE_PROFILES_EXTENSION_KEY) else {
        return vec![LEGACY_DECODE_PROFILE_FLOOR];
    };
    if bytes.len() > MAX_DECODE_PROFILES_BYTES {
        warn!(
            payload_len = bytes.len(),
            cap = MAX_DECODE_PROFILES_BYTES,
            "client decode-profiles extension exceeds size cap; \
             treating as legacy H.264 4:2:0 client"
        );
        return vec![LEGACY_DECODE_PROFILE_FLOOR];
    }
    match tether_protocol::decode::<Vec<VideoProfile>>(bytes) {
        Ok(v) => {
            // Forward-compat: a future client may advertise a profile
            // with a bit_depth this host doesn't know how to encode
            // (12-bit, say). Filter it out rather than reject the whole
            // list — the rest of the advertised profiles may still
            // contain something we can match.
            let total = v.len();
            let filtered: Vec<VideoProfile> =
                v.into_iter().filter(|p| is_known_bit_depth(p.bit_depth)).collect();
            if filtered.len() != total {
                warn!(
                    dropped = total - filtered.len(),
                    kept = filtered.len(),
                    "client advertised profiles with unknown bit_depth values; dropped them"
                );
            }
            if filtered.is_empty() && total > 0 {
                // A non-empty advert that filtered down to nothing is
                // a real diagnostic — the no-match Goodbye that follows
                // would otherwise read as "client and host have no
                // codec in common" when the real cause is "every
                // profile the client offered had a bit_depth we don't
                // know about."
                warn!(
                    advertised_count = total,
                    "client advertised only unknown bit_depths; \
                     no profiles survived the boundary filter"
                );
            }
            filtered
        }
        Err(e) => {
            warn!(
                error = %e,
                payload_len = bytes.len(),
                "client decode-profiles extension failed to decode; \
                 treating as legacy H.264 4:2:0 client"
            );
            vec![LEGACY_DECODE_PROFILE_FLOOR]
        }
    }
}

/// Build the body of the `ServerHello` we'll send back. `chosen` is
/// `None` iff there's no mutual profile; we still need a syntactically
/// valid hello so the wire response decodes cleanly on the client side
/// before our follow-up `Goodbye` arrives.
fn build_server_hello(cfg: &HostSessionConfig, chosen: Option<VideoProfile>) -> ServerHelloV1 {
    let mut extensions = BTreeMap::new();

    // Pixel-format advert: lets the client's decoder import path
    // (VAAPI / VT / MF) pick its format without waiting on the first
    // SPS. Value matches (chroma, bit_depth) — NV12 for 4:2:0 8-bit,
    // P010 for 4:2:0 10-bit, Yuv444p for 4:4:4 8-bit, P410 for 4:4:4
    // 10-bit. On no-match we still advertise NV12 (the floor) so the
    // wire shape is valid; the follow-up Goodbye is what ends the
    // session.
    let pixel_format = match chosen.map(|p| (p.chroma, p.bit_depth)) {
        Some((ChromaSubsampling::Yuv444, 10)) => PixelFormat::P410,
        Some((ChromaSubsampling::Yuv444, _)) => PixelFormat::Yuv444p,
        Some((ChromaSubsampling::Yuv420, 10)) => PixelFormat::P010,
        _ => PixelFormat::Nv12,
    };
    extensions.insert(
        PIXEL_FORMAT_EXTENSION_KEY.to_string(),
        tether_protocol::encode(&pixel_format)
            .expect("PixelFormat encodes; types under our control"),
    );

    if let Some(p) = chosen {
        extensions.insert(
            SERVER_ENCODE_PROFILE_EXTENSION_KEY.to_string(),
            tether_protocol::encode(&p).expect("VideoProfile encodes; types under our control"),
        );
    }

    ServerHelloV1 {
        server_name: cfg.server_name.clone(),
        // On no-match the response carries H.264 / Yuv420 as a placeholder
        // — the immediately-following Goodbye is what the client acts on.
        // We use the universal floor (not a future variant) so a buggy
        // client that ignores the Goodbye doesn't trip on an unknown
        // codec before it processes the goodbye.
        chosen_codec: chosen.map_or(CodecKind::H264, |p| p.codec),
        chosen_chroma: chosen.map_or(ChromaSubsampling::Yuv420, |p| p.chroma),
        // sRGB transfer, BT.709 matrix / primaries / limited range — the
        // honest spec for every host backend we ship today.
        color_space: VideoColorSpec::sdr_desktop(),
        // Encoded source dims aren't known yet (lazy encoder init happens
        // on first frame); per-frame `VideoFrameMeta::dimensions` is the
        // source of truth.
        resolution: (0, 0),
        clock_probe_t0_echo: MonoNanos::ZERO,
        t1_server_recv: MonoNanos::ZERO,
        t2_server_send: MonoNanos::ZERO,
        extensions,
        resume_token: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tether_protocol::control::ClientHelloV1;

    fn empty_hello() -> ClientHelloV1 {
        ClientHelloV1 {
            client_name: "test".to_string(),
            preferred_codecs: vec![],
            max_resolution: None,
            clock_probe_t0: MonoNanos::ZERO,
            extensions: BTreeMap::new(),
            resume_token: None,
        }
    }

    #[test]
    fn missing_extension_falls_back_to_floor() {
        let parsed = parse_client_decode_profiles(&empty_hello());
        assert_eq!(parsed, vec![LEGACY_DECODE_PROFILE_FLOOR]);
    }

    #[test]
    fn oversize_extension_falls_back_to_floor() {
        let mut hello = empty_hello();
        hello.extensions.insert(
            CLIENT_DECODE_PROFILES_EXTENSION_KEY.to_string(),
            vec![0xAA; MAX_DECODE_PROFILES_BYTES + 1],
        );
        let parsed = parse_client_decode_profiles(&hello);
        assert_eq!(parsed, vec![LEGACY_DECODE_PROFILE_FLOOR]);
    }

    #[test]
    fn malformed_extension_falls_back_to_floor() {
        let mut hello = empty_hello();
        hello.extensions.insert(
            CLIENT_DECODE_PROFILES_EXTENSION_KEY.to_string(),
            vec![0xFF, 0xFF, 0xFF, 0xFF],
        );
        let parsed = parse_client_decode_profiles(&hello);
        assert_eq!(parsed, vec![LEGACY_DECODE_PROFILE_FLOOR]);
    }

    #[test]
    fn well_formed_extension_round_trips() {
        let advertised = vec![VideoProfile::HEVC_10BIT_444, VideoProfile::HEVC_8BIT_420];
        let mut hello = empty_hello();
        hello.extensions.insert(
            CLIENT_DECODE_PROFILES_EXTENSION_KEY.to_string(),
            tether_protocol::encode(&advertised).unwrap(),
        );
        let parsed = parse_client_decode_profiles(&hello);
        assert_eq!(parsed, advertised);
    }

    #[test]
    fn server_hello_with_no_match_uses_floor_placeholder() {
        let cfg = HostSessionConfig {
            server_name: "test".to_string(),
        };
        let hello = build_server_hello(&cfg, None);
        assert_eq!(hello.chosen_codec, CodecKind::H264);
        assert_eq!(hello.chosen_chroma, ChromaSubsampling::Yuv420);
        // Pixel format advert still present (Nv12 floor).
        let pf_bytes = hello.extensions.get(PIXEL_FORMAT_EXTENSION_KEY).unwrap();
        let pf: PixelFormat = tether_protocol::decode(pf_bytes).unwrap();
        assert_eq!(pf, PixelFormat::Nv12);
        // Encode-profile echo is *not* present (no match to echo).
        assert!(!hello
            .extensions
            .contains_key(SERVER_ENCODE_PROFILE_EXTENSION_KEY));
    }

    #[test]
    fn server_hello_echoes_chosen_profile_and_pixel_format() {
        let cfg = HostSessionConfig {
            server_name: "test".to_string(),
        };
        let hello = build_server_hello(&cfg, Some(VideoProfile::HEVC_10BIT_444));
        let pf_bytes = hello.extensions.get(PIXEL_FORMAT_EXTENSION_KEY).unwrap();
        let pf: PixelFormat = tether_protocol::decode(pf_bytes).unwrap();
        assert_eq!(pf, PixelFormat::P410);

        let p_bytes = hello
            .extensions
            .get(SERVER_ENCODE_PROFILE_EXTENSION_KEY)
            .unwrap();
        let p: VideoProfile = tether_protocol::decode(p_bytes).unwrap();
        assert_eq!(p, VideoProfile::HEVC_10BIT_444);
    }
}
