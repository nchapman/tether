//! Wire protocol for Tether.
//!
//! Five logical channels, carried by `tether-transport`:
//! - **Control** (reliable, bidirectional) — handshake, clock sync,
//!   IDR requests, stream lifecycle, cursor shape, display topology,
//!   extension escape hatch.
//! - **Video datagrams** (unreliable) — fragmented encoded frames.
//! - **Audio datagrams** (unreliable) — Opus packets, host→client.
//!   Capture/encode/decode/output is wired end-to-end and negotiated by
//!   `ServerHello::audio`; see [`audio`].
//! - **Cursor datagrams** (unreliable, high priority) — pointer
//!   position. Sprite payloads ride the reliable control stream.
//! - **Input stream** (reliable, client→host) — keyboard + mouse events.

pub mod audio;
pub mod control;
pub mod cursor;
pub mod guard;
pub mod input;
pub mod pairing;
pub mod video;

pub use guard::GpuResourceGuard;

use serde::{Deserialize, Serialize};

pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/tether.v1.rs"));
}

/// Conservative QUIC datagram budget — the upper bound on encoded
/// [`video::VideoPacket`] size we target. The actual `max_datagram_size`
/// reported by quinn at runtime may be larger; we slice frames to this size
/// so packetization is FEC-friendly from day one.
///
/// The transport layer is responsible for measuring the runtime overhead of
/// a packet header (which varies with varint encoding of `frame_seq`,
/// `fragment_index`, and the `InputEchoBatch` size) and choosing an actual
/// payload chunk size that keeps the encoded packet under this budget.
pub const MAX_DATAGRAM_PAYLOAD: usize = 1200;

/// Hard ceiling on the bytes a single *received* datagram decode may allocate.
///
/// Distinct from [`MAX_DATAGRAM_PAYLOAD`], the soft per-datagram size the video
/// fragmenter *aims* for. This ceiling exists purely to stop a forged length
/// prefix on an untrusted datagram from driving a giant pre-allocation; quinn
/// already caps the real wire size to the path MTU.
///
/// Two channels share the datagram path, and the ceiling must clear the
/// largest legitimate packet on either:
/// - **Video:** the fragmenter sizes every shard so the encoded `First` /
///   `Parity` datagram stays `<= MAX_DATAGRAM_PAYLOAD` (it derives the shard
///   size from the connection's real `max_datagram_size` minus the encoded
///   meta + header — see `video::FrameFragmenter`). Video therefore never
///   approaches this ceiling.
/// - **Audio:** an `AudioPacket::Opus` payload can be up to
///   `tether_audio::MAX_PACKET_BYTES` (4000, libopus's recommended max) plus
///   ~20 bytes of enum/varint framing. A high-bitrate / long-frame config
///   (e.g. 510 kbps @ 60 ms) lands near that ceiling; the v1 default
///   (~80 B/packet at 5 ms) is comfortably under, as is its RED tail.
///
/// Set to clear `MAX_PACKET_BYTES` plus framing so a non-default audio config is
/// not silently dropped, while still bounding a hostile allocation to ~4 KB.
/// `tether-audio` carries a static assertion that this stays at or above
/// `MAX_PACKET_BYTES` plus framing.
///
/// Nested collections (e.g. `Vec<Bytes>` in `AudioPacket::Opus::redundant`) are
/// equally bounded — the decoder counts bytes across the whole decoded
/// structure, not per field, so a forged outer length claiming many inner copies
/// can't drive a giant pre-allocation either.
pub const MAX_DATAGRAM_DECODE_BYTES: usize = 4100;

/// Wire-protocol version string. Bumped on any breaking change to the
/// control/handshake/pairing wire contract.
///
/// Load-bearing for the pairing layer: it is folded byte-for-byte into the
/// SPAKE2 key-confirmation transcript (`tether-pairing`), so a peer running a
/// different version cannot complete pairing — closing a downgrade seam where
/// an attacker forces an older, weaker pairing exchange. Changing this literal
/// is a breaking change; keep it stable within a release line.
pub const PROTOCOL_VERSION: &str = "tether/1";

/// Monotonic nanoseconds since an arbitrary local epoch (the first call to
/// [`MonoNanos::now`] in this process).
///
/// Values are only comparable within a single machine's clock domain. To
/// convert a peer's timestamp into the local clock, apply the offset
/// computed from a [`control::ClockProbe`] exchange.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct MonoNanos(pub u64);

impl MonoNanos {
    pub const ZERO: Self = Self(0);

    pub fn now() -> Self {
        use std::sync::OnceLock;
        use std::time::Instant;
        static PROCESS_START: OnceLock<Instant> = OnceLock::new();
        let start = PROCESS_START.get_or_init(Instant::now);
        let nanos = Instant::now().duration_since(*start).as_nanos();
        // `Duration::as_nanos` returns u128 to accommodate ~584-year
        // intervals; we saturate to u64 (still ~584 years from process
        // start) rather than panic. Not an `unwrap` site — clamping is
        // the intended behavior, just not the form `try_from` ships in.
        Self(u64::try_from(nanos).unwrap_or(u64::MAX))
    }

    pub fn saturating_sub(self, other: Self) -> u64 {
        self.0.saturating_sub(other.0)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("encode: {0}")]
    Encode(#[from] bincode::error::EncodeError),
    #[error("decode: {0}")]
    Decode(#[from] bincode::error::DecodeError),
    #[error("prost decode: {0}")]
    ProstDecode(#[from] prost::DecodeError),
    #[error("wire: {0}")]
    Wire(&'static str),
}

// Compact serde/bincode remains the codec for MTU-sensitive media datagrams and
// the pre-session pairing stream. Reliable session control/input use prost via
// `ReliableMessage`.
fn bincode_config() -> impl bincode::config::Config {
    bincode::config::standard()
}

/// Like [`bincode_config`] but caps the total bytes a single decode may claim
/// at [`MAX_DATAGRAM_DECODE_BYTES`]. Datagrams arrive from an untrusted peer, and
/// a forged length prefix on a byte-sequence field (e.g.
/// `AudioPacket::Opus.payload`, a `Bytes`, or a nested `Vec<Bytes>` tail) would
/// otherwise drive bincode to pre-allocate gigabytes. bincode claims a
/// container's bytes against this limit *before* allocating, so an over-long
/// field is rejected up front. The ceiling is [`MAX_DATAGRAM_DECODE_BYTES`], not
/// the soft [`MAX_DATAGRAM_PAYLOAD`] target — a legitimate `First`/`Parity`
/// packet with an input-echo batch encodes past 1200 yet is delivered fine by
/// quinn, so bounding at the soft target would reject real traffic.
fn datagram_config() -> impl bincode::config::Config {
    bincode::config::standard().with_limit::<MAX_DATAGRAM_DECODE_BYTES>()
}

pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, CodecError> {
    Ok(bincode::serde::encode_to_vec(value, bincode_config())?)
}

pub fn decode<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, CodecError> {
    decode_with_config(bytes, bincode_config())
}

/// Decode a value read off the untrusted QUIC datagram path, bounding any
/// allocation to [`MAX_DATAGRAM_DECODE_BYTES`] (see [`datagram_config`]). Use this —
/// not [`decode`] — for anything deserialized straight from a received
/// datagram, so a forged length prefix can't trigger an out-of-memory abort.
pub fn decode_datagram<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, CodecError> {
    decode_with_config(bytes, datagram_config())
}

pub trait ReliableMessage: Sized {
    fn encode_reliable(&self) -> Vec<u8>;
    fn decode_reliable(bytes: &[u8]) -> Result<Self, CodecError>;
}

pub fn encode_reliable<T: ReliableMessage>(value: &T) -> Result<Vec<u8>, CodecError> {
    Ok(value.encode_reliable())
}

pub fn decode_reliable<T: ReliableMessage>(bytes: &[u8]) -> Result<T, CodecError> {
    T::decode_reliable(bytes)
}

fn decode_with_config<T: for<'de> Deserialize<'de>>(
    bytes: &[u8],
    config: impl bincode::config::Config,
) -> Result<T, CodecError> {
    let (value, consumed) = bincode::serde::decode_from_slice(bytes, config)?;
    // Strict-decode compact messages: every datagram/pairing frame must be
    // fully consumed by its declared type. The reliable session protocol gets
    // field-level evolution from prost; bincode payloads do not.
    if consumed != bytes.len() {
        return Err(CodecError::Decode(bincode::error::DecodeError::Other(
            "decoded message had trailing bytes; sender may be using a \
             schema-incompatible protocol revision",
        )));
    }
    Ok(value)
}

#[cfg(test)]
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::control::*;
    use crate::cursor::*;
    use crate::input::*;
    use crate::video::{
        fec_layout, FrameFragmenter, FrameReassembler, HostFrameTiming, HostFrameTimingBuilder,
        InputEchoBatch, VideoFrameMeta, VideoFrameMetaEnvelope, VideoPacket,
        DATAGRAM_WRAPPER_BYTES, FEC_MAX_PRIMARY_SHARDS,
    };
    use prost::Message as _;

    /// The datagram decoder bounds allocation: a payload larger than
    /// `MAX_DATAGRAM_DECODE_BYTES` is rejected before allocation (where the
    /// unbounded `decode` would accept it). This is the guard against a forged
    /// length prefix on an untrusted datagram (e.g. an Opus payload) driving an
    /// OOM. The ceiling is the decode bound, not the soft `MAX_DATAGRAM_PAYLOAD`
    /// fragmentation target (see `MAX_DATAGRAM_DECODE_BYTES`).
    #[test]
    fn decode_datagram_rejects_oversize_payload() {
        let big = vec![0u8; MAX_DATAGRAM_DECODE_BYTES + 64];
        let bytes = encode(&big).unwrap();
        assert!(
            decode::<Vec<u8>>(&bytes).is_ok(),
            "unbounded decode accepts the oversize payload"
        );
        assert!(
            decode_datagram::<Vec<u8>>(&bytes).is_err(),
            "bounded datagram decode rejects it before allocating"
        );
    }

    /// A normal (within-limit) payload still round-trips through the bounded
    /// datagram decoder.
    #[test]
    fn decode_datagram_round_trips_a_normal_payload() {
        let payload = vec![7u8; 900];
        let bytes = encode(&payload).unwrap();
        let back: Vec<u8> = decode_datagram(&bytes).unwrap();
        assert_eq!(back, payload);
    }

    /// #37: the fragmenter sizes shards from the datagram budget *minus* the
    /// encoded meta envelope, so even a `First`/`Parity` packet carrying a
    /// fat `input_echo` batch stays within the budget once wrapped in
    /// `Datagram::Video`. The pre-fix fixed shard size ignored the meta and
    /// pushed those packets past a low-MTU path's datagram size (dropped at
    /// send under active input). Asserts every emitted datagram fits, across a
    /// range of input-echo sizes and budgets.
    #[test]
    fn every_fragment_fits_the_datagram_budget_under_input_echo() {
        for budget in [1200usize, 1280, 1100] {
            for echo in [0usize, 8, 16, 64, 256] {
                let mut frag = FrameFragmenter::new_with_fec(0u8, 20);
                let meta = VideoFrameMeta {
                    timing: HostFrameTiming {
                        t_capture_kernel: MonoNanos(u64::MAX / 2),
                        t_capture_userspace: MonoNanos(u64::MAX / 2),
                        t_encode_submit: MonoNanos(u64::MAX / 2),
                        t_encode_done: MonoNanos(u64::MAX / 2),
                        t_send: MonoNanos(u64::MAX / 2),
                    },
                    keyframe: true,
                    input_echo: InputEchoBatch {
                        event_ids: vec![u64::MAX; echo],
                    },
                    dimensions: (3840, 2160),
                };
                // A multi-shard, multi-block-eligible body.
                let body = bytes::Bytes::from(vec![0xABu8; 8000]);
                let packets = frag.fragment(meta, body, budget);
                for p in &packets {
                    // +1 for the outer Datagram::Video discriminant — the real
                    // on-wire size the path MTU bounds.
                    let wire = p.wire_size() + DATAGRAM_WRAPPER_BYTES;
                    assert!(
                        wire <= budget,
                        "packet wire size {wire} exceeds budget {budget} \
                         (echo={echo}): {p:?}"
                    );
                    // And every datagram still decodes through the guarded path.
                    let bytes = encode(p).unwrap();
                    decode_datagram::<VideoPacket>(&bytes)
                        .expect("a budget-sized datagram must decode");
                }
            }
        }
    }

    #[test]
    fn pairing_messages_round_trip() {
        use crate::pairing::*;

        let cases_client = [
            PairingClientMsg::Pair {
                spake2: vec![1, 2, 3, 4],
            },
            PairingClientMsg::Resume,
        ];
        for m in &cases_client {
            let bytes = encode(m).unwrap();
            let back: PairingClientMsg = decode(&bytes).unwrap();
            assert_eq!(*m, back);
        }

        let cases_server = [
            PairingServerMsg::PairChallenge {
                spake2: vec![9, 8, 7],
            },
            PairingServerMsg::Authorized,
            PairingServerMsg::Rejected {
                reason: "no pairing window open".into(),
            },
        ];
        for m in &cases_server {
            let bytes = encode(m).unwrap();
            let back: PairingServerMsg = decode(&bytes).unwrap();
            assert_eq!(*m, back);
        }

        let confirm = PairingConfirm {
            mac: vec![0xAB; 32],
        };
        assert_eq!(confirm, decode(&encode(&confirm).unwrap()).unwrap());

        let cases_result = [
            PairingResult::Confirmed {
                mac: vec![0xCD; 32],
            },
            PairingResult::Rejected {
                reason: "confirmation failed".into(),
            },
        ];
        for m in &cases_result {
            let bytes = encode(m).unwrap();
            let back: PairingResult = decode(&bytes).unwrap();
            assert_eq!(*m, back);
        }
    }

    #[test]
    fn unknown_pairing_client_variant_fails_decode() {
        // Forward-compat probe (mirrors unknown_client_hello_variant_fails_decode):
        // a future PairingClientMsg variant must fail decode on an older peer,
        // not silently misparse. Variants 0 (Pair) and 1 (Resume) are assigned;
        // a hand-crafted discriminator of 5 is unassigned.
        let bytes = [5u8];
        let result = decode::<crate::pairing::PairingClientMsg>(&bytes);
        assert!(
            result.is_err(),
            "unknown PairingClientMsg variant must fail decode, not silently succeed"
        );
    }

    #[test]
    fn unknown_pairing_server_and_result_variants_fail_decode() {
        // Same forward-compat guard for the host→client enums: an unassigned
        // discriminator must fail decode on an older peer.
        let bytes = [9u8];
        assert!(
            decode::<crate::pairing::PairingServerMsg>(&bytes).is_err(),
            "unknown PairingServerMsg variant must fail decode"
        );
        assert!(
            decode::<crate::pairing::PairingResult>(&bytes).is_err(),
            "unknown PairingResult variant must fail decode"
        );
    }

    #[test]
    fn protocol_version_literal_is_pinned() {
        // The pairing transcript binds this byte-for-byte; a silent change
        // would break pairing across builds and weaken the downgrade defense.
        // Update deliberately, in lockstep with a wire-contract bump.
        assert_eq!(PROTOCOL_VERSION, "tether/1");
    }

    #[test]
    fn host_frame_timing_builder_finishes() {
        let mut b = HostFrameTimingBuilder::captured(MonoNanos(100), MonoNanos(200));
        b.encode_submit();
        b.encode_done();
        assert!(b.encode_delta_ns() < 1_000_000); // sub-ms for two consecutive now()
        let t = b.finish();
        assert_eq!(t.t_capture_kernel, MonoNanos(100));
        assert_eq!(t.t_capture_userspace, MonoNanos(200));
        assert!(t.t_encode_submit <= t.t_encode_done);
        assert!(t.t_encode_done <= t.t_send);
    }

    #[test]
    #[should_panic(expected = "encode_submit not called")]
    fn host_frame_timing_builder_panics_if_submit_skipped() {
        let mut b = HostFrameTimingBuilder::captured(MonoNanos(0), MonoNanos(0));
        b.encode_done();
        let _ = b.finish();
    }

    #[test]
    #[should_panic(expected = "encode_done not called")]
    fn host_frame_timing_builder_panics_if_done_skipped() {
        let mut b = HostFrameTimingBuilder::captured(MonoNanos(0), MonoNanos(0));
        b.encode_submit();
        let _ = b.finish();
    }

    #[test]
    fn mono_nanos_monotonic() {
        let a = MonoNanos::now();
        let b = MonoNanos::now();
        assert!(b >= a);
    }

    #[test]
    fn round_trip_client_hello() {
        let hello = ClientHello {
            client_name: "tether-client/0.0.1".into(),
            decode_profiles: vec![VideoProfile::H264_8BIT_420, VideoProfile::HEVC_8BIT_420],
            initial_viewport: Some(crate::control::Viewport::new(1280, 720)),
            input_capabilities: InputCapabilities::default(),
            requested_features: vec![],
        };
        let bytes = encode_reliable(&hello).unwrap();
        let decoded: ClientHello = decode_reliable(&bytes).unwrap();
        assert_eq!(decoded, hello);
    }

    #[test]
    fn client_hello_feature_adverts_round_trip() {
        let features = vec![
            FeatureAdvert {
                key: "tether.clipboard".to_string(),
                min_version: 1,
                max_version: 2,
                payload: vec![1],
            },
            FeatureAdvert {
                key: "tether.gamepad".to_string(),
                min_version: 1,
                max_version: 1,
                payload: vec![0, 64],
            },
        ];
        let hello = ClientHello {
            client_name: "x".into(),
            decode_profiles: vec![VideoProfile::H264_8BIT_420],
            initial_viewport: None,
            input_capabilities: InputCapabilities::default(),
            requested_features: features.clone(),
        };
        let decoded: ClientHello = decode_reliable(&encode_reliable(&hello).unwrap()).unwrap();
        assert_eq!(decoded.requested_features, features);
    }

    #[test]
    fn round_trip_server_hello_hevc() {
        use crate::control::{ServerHello, VideoColorSpec};
        let mode = DisplayMode::new(1920, 1080, 60_000);
        let hello = ServerHello {
            server_name: "tether-host".into(),
            video: NegotiatedVideo {
                stream_id: VideoStreamId(0),
                display_id: DisplayId(0),
                profile: VideoProfile::HEVC_8BIT_420,
                pixel_format: PixelFormat::Nv12,
                color_space: VideoColorSpec::sdr_desktop(),
            },
            audio: Some(crate::audio::AudioConfig {
                sample_rate_hz: 48_000,
                channels: 2,
                streams: 1,
                coupled_streams: 1,
                channel_mapping: vec![0, 1],
            }),
            displays: vec![DisplayDescriptor {
                id: DisplayId(0),
                name: "DP-1".into(),
                scale_num: 1,
                scale_den: 1,
                primary: true,
                position: (0, 0),
                current_mode: mode,
                available_modes: vec![mode],
                can_set_mode: true,
            }],
            accepted_features: vec![FeatureAccept {
                key: "tether.clipboard".to_string(),
                version: 1,
                payload: vec![1],
            }],
        };
        let decoded: ServerHello = decode_reliable(&encode_reliable(&hello).unwrap()).unwrap();
        assert_eq!(decoded, hello);
    }

    #[test]
    fn round_trip_server_handshake_rejection() {
        let rejection = ServerHandshake::Rejected(HandshakeFailure {
            code: GoodbyeCode::InternalError,
            reason: "no mutual profile".into(),
        });
        let decoded: ServerHandshake =
            decode_reliable(&encode_reliable(&rejection).unwrap()).unwrap();
        assert_eq!(decoded, rejection);
    }

    #[test]
    fn prost_unknown_client_hello_field_is_skipped() {
        let hello = ClientHello {
            client_name: "x".into(),
            decode_profiles: vec![VideoProfile::H264_8BIT_420],
            initial_viewport: None,
            input_capabilities: InputCapabilities::default(),
            requested_features: vec![],
        };
        let mut bytes = encode_reliable(&hello).unwrap();
        // field 99, wire type 2 (length-delimited), length 3, payload "new".
        bytes.extend_from_slice(&[0x9A, 0x06, 0x03, b'n', b'e', b'w']);
        let decoded: ClientHello = decode_reliable(&bytes).unwrap();
        assert_eq!(decoded, hello);
    }

    #[test]
    fn client_hello_skips_unknown_advertised_profiles() {
        use prost::Message as _;
        let wire = pb::ClientHello {
            client_name: "future-client".into(),
            decode_profiles: vec![
                pb::VideoProfile {
                    codec: 99,
                    chroma: 1,
                    bit_depth: 8,
                },
                pb::VideoProfile {
                    codec: 1,
                    chroma: 77,
                    bit_depth: 8,
                },
                pb::VideoProfile {
                    codec: 2,
                    chroma: 1,
                    bit_depth: 10,
                },
            ],
            initial_viewport: None,
            input_capabilities: Some(pb::InputCapabilities {
                keyboard: true,
                mouse: true,
                relative_mouse: true,
                text: true,
            }),
            requested_features: vec![],
        }
        .encode_to_vec();
        let decoded: ClientHello = decode_reliable(&wire).unwrap();
        assert_eq!(decoded.decode_profiles, vec![VideoProfile::HEVC_10BIT_420]);
    }

    #[test]
    fn round_trip_set_viewport_hint() {
        use crate::control::{ControlMessage, Viewport};
        let msg = ControlMessage::SetViewportHint {
            stream_id: VideoStreamId(2),
            viewport: Viewport::new(1280, 800),
        };
        let bytes = encode_reliable(&msg).unwrap();
        let msg2: ControlMessage = decode_reliable(&bytes).unwrap();
        match msg2 {
            ControlMessage::SetViewportHint {
                stream_id,
                viewport: v,
            } => {
                assert_eq!(stream_id, VideoStreamId(2));
                assert_eq!(v.width, 1280);
                assert_eq!(v.height, 800);
            }
            other => panic!("expected SetViewportHint, got {other:?}"),
        }
    }

    #[test]
    fn viewport_is_valid_rejects_zero_dims() {
        use crate::control::Viewport;
        assert!(Viewport::new(1, 1).is_valid());
        assert!(!Viewport::new(0, 1).is_valid());
        assert!(!Viewport::new(1, 0).is_valid());
        assert!(!Viewport::new(0, 0).is_valid());
    }

    #[test]
    fn round_trip_cursor_shape_control() {
        // CursorShape rides the reliable control stream (sprite payloads
        // are too large for the 1200-byte cursor datagram budget).
        let msg = ControlMessage::CursorShape {
            id: 42,
            hotspot: (4, 7),
            width: 16,
            height: 16,
            format: CursorPixelFormat::Rgba8,
            pixels: vec![0xABu8; 16 * 16 * 4],
        };
        let bytes = encode_reliable(&msg).unwrap();
        let msg2: ControlMessage = decode_reliable(&bytes).unwrap();
        match msg2 {
            ControlMessage::CursorShape {
                id,
                hotspot,
                width,
                height,
                format,
                pixels,
            } => {
                assert_eq!(id, 42);
                assert_eq!(hotspot, (4, 7));
                assert_eq!(width, 16);
                assert_eq!(height, 16);
                assert_eq!(format, CursorPixelFormat::Rgba8);
                assert_eq!(pixels.len(), 16 * 16 * 4);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn round_trip_cursor_use_shape_control() {
        let msg = ControlMessage::CursorUseShape { id: 7 };
        let bytes = encode_reliable(&msg).unwrap();
        let msg2: ControlMessage = decode_reliable(&bytes).unwrap();
        match msg2 {
            ControlMessage::CursorUseShape { id } => assert_eq!(id, 7),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn round_trip_display_list_multi() {
        // Multi-entry exercises the Vec<DisplayDescriptor> serde shape
        // even though the host today always emits a one-element list.
        use crate::control::DisplayDescriptor;
        let primary_mode = DisplayMode::new(3840, 2160, 60_000);
        let secondary_mode = DisplayMode::new(1920, 1080, 59_940);
        let displays = vec![
            DisplayDescriptor {
                id: DisplayId(0),
                name: "DP-1".into(),
                scale_num: 2,
                scale_den: 1,
                primary: true,
                position: (0, 0),
                current_mode: primary_mode,
                available_modes: vec![primary_mode],
                can_set_mode: true,
            },
            DisplayDescriptor {
                id: DisplayId(1),
                name: "HDMI-A-2".into(),
                scale_num: 1,
                scale_den: 1,
                primary: false,
                position: (3840, 0),
                current_mode: secondary_mode,
                available_modes: vec![secondary_mode],
                can_set_mode: false,
            },
        ];
        let msg = ControlMessage::DisplayList {
            displays: displays.clone(),
        };
        let bytes = encode_reliable(&msg).unwrap();
        let msg2: ControlMessage = decode_reliable(&bytes).unwrap();
        match msg2 {
            ControlMessage::DisplayList { displays: d2 } => assert_eq!(d2, displays),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn round_trip_set_active_displays() {
        let msg = ControlMessage::SetActiveDisplays {
            displays: vec![DisplayId(0), DisplayId(2), DisplayId(5)],
        };
        let bytes = encode_reliable(&msg).unwrap();
        let msg2: ControlMessage = decode_reliable(&bytes).unwrap();
        match msg2 {
            ControlMessage::SetActiveDisplays { displays } => {
                assert_eq!(displays, vec![DisplayId(0), DisplayId(2), DisplayId(5)]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn round_trip_display_mode_request_and_result() {
        let mode = DisplayMode::new(2560, 1440, 144_000);
        let request = ControlMessage::SetDisplayMode {
            request_id: RequestId(42),
            display_id: DisplayId(7),
            mode,
            restore_on_disconnect: true,
        };
        let decoded: ControlMessage = decode_reliable(&encode_reliable(&request).unwrap()).unwrap();
        assert_eq!(decoded, request);

        let result = ControlMessage::DisplayModeResult {
            request_id: RequestId(42),
            display_id: DisplayId(7),
            status: DisplayModeStatus::Unsupported,
            actual_mode: None,
        };
        let decoded: ControlMessage = decode_reliable(&encode_reliable(&result).unwrap()).unwrap();
        assert_eq!(decoded, result);
    }

    #[test]
    fn pixel_format_round_trips_as_typed_value() {
        let pf = PixelFormat::Nv12;
        let bytes = encode(&pf).unwrap();
        let pf2: PixelFormat = decode(&bytes).unwrap();
        assert_eq!(pf, pf2);
    }

    /// Pin the serde/bincode representation used by compact paths and debug
    /// tooling: `VideoProfile { codec, chroma, bit_depth }` must round-trip 10
    /// cleanly. Reliable hello negotiation uses protobuf and has its own tests.
    #[test]
    fn ten_bit_video_profile_round_trips() {
        use crate::control::VideoProfile;
        for profile in [VideoProfile::HEVC_10BIT_420, VideoProfile::HEVC_10BIT_444] {
            let bytes = encode(&profile).unwrap();
            let decoded: VideoProfile = decode(&bytes).unwrap();
            assert_eq!(decoded, profile);
            assert_eq!(decoded.bit_depth, 10);
        }
    }

    /// Pin the serde/bincode representation used by compact paths and debug
    /// tooling: the new `CodecKind::Av1` variant must round-trip at both 8- and
    /// 10-bit depths cleanly. Reliable hello negotiation uses protobuf and has
    /// its own tests.
    #[test]
    fn av1_video_profile_round_trips() {
        use crate::control::{CodecKind, VideoProfile};
        for profile in [VideoProfile::AV1_8BIT_420, VideoProfile::AV1_10BIT_420] {
            let bytes = encode(&profile).unwrap();
            let decoded: VideoProfile = decode(&bytes).unwrap();
            assert_eq!(decoded, profile);
            assert_eq!(decoded.codec, CodecKind::Av1);
        }
    }

    /// Regression guard: `bit_depth` must stay a plain `u8` on the
    /// wire. If this field is ever changed to a closed enum (e.g.
    /// `enum BitDepth { Eight, Ten }`), this test stops compiling,
    /// surfacing the wire-compat break before it ships. The
    /// behavioural forward-compat property — that older negotiators
    /// gracefully ignore an unrecognised depth instead of panicking —
    /// is covered by `returns_none_when_disjoint` in
    /// `tether-codec::probe`.
    #[test]
    fn future_bit_depth_decodes_as_raw_u8() {
        use crate::control::{ChromaSubsampling, CodecKind, VideoProfile};
        let future = VideoProfile {
            codec: CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 12,
        };
        let bytes = encode(&future).unwrap();
        let decoded: VideoProfile = decode(&bytes).unwrap();
        assert_eq!(decoded.bit_depth, 12);
        // The codec/chroma fields stay readable — the negotiator can
        // skip the profile cleanly even when the bit_depth is one it
        // doesn't recognise, instead of getting tripped by a decode
        // error mid-handshake.
        assert_eq!(decoded.codec, CodecKind::Hevc);
        assert_eq!(decoded.chroma, ChromaSubsampling::Yuv420);
    }

    #[test]
    fn unknown_video_profile_codec_fails_decode() {
        // Forward-compat: a future host that advertises an unknown
        // codec discriminator should fail decode cleanly on an older
        // client rather than silently misinterpret the byte. We let
        // the serializer dictate the field layout (so the test stays
        // valid if VideoProfile field order ever changes) and only
        // corrupt the CodecKind discriminator byte at position 0.
        let mut bytes = encode(&crate::control::VideoProfile::H264_8BIT_420).unwrap();
        bytes[0] = 99; // past any known CodecKind variant
        let result = decode::<crate::control::VideoProfile>(&bytes);
        assert!(
            result.is_err(),
            "unknown CodecKind discriminator must fail decode"
        );
    }

    #[test]
    fn video_color_spec_round_trips() {
        use crate::control::{
            ColorMatrix, ColorPrimaries, ColorRange, ColorTransfer, VideoColorSpec,
        };
        // Hand-build a non-default spec so every field is exercised
        // (a `default()` round-trip would still pass if we
        // accidentally dropped one of the four axes from the struct
        // on either end).
        let spec = VideoColorSpec {
            matrix: ColorMatrix::Bt2020Ncl,
            range: ColorRange::Full,
            transfer: ColorTransfer::Pq,
            primaries: ColorPrimaries::Bt2020,
        };
        let bytes = encode(&spec).unwrap();
        let decoded: VideoColorSpec = decode(&bytes).unwrap();
        assert_eq!(decoded, spec);
    }

    #[test]
    fn color_spec_named_constructors_match_intent() {
        use crate::control::{
            ColorMatrix, ColorPrimaries, ColorRange, ColorTransfer, VideoColorSpec,
        };
        // `sdr_desktop` is what every current host backend advertises:
        // sRGB transfer (compositor framebuffer reality) with BT.709
        // matrix / primaries / limited range. Pin the four axes so a
        // future refactor that swaps an axis fails loudly.
        let desktop = VideoColorSpec::sdr_desktop();
        assert_eq!(desktop.matrix, ColorMatrix::Bt709);
        assert_eq!(desktop.range, ColorRange::Limited);
        assert_eq!(desktop.transfer, ColorTransfer::Srgb);
        assert_eq!(desktop.primaries, ColorPrimaries::Bt709);
        assert_eq!(VideoColorSpec::default(), desktop);

        // `sdr_bt709` is the broadcast spec (BT.709 transfer instead
        // of sRGB). Same matrix / primaries / range as desktop;
        // distinguished only by the transfer curve.
        let bt709 = VideoColorSpec::sdr_bt709();
        assert_eq!(bt709.matrix, ColorMatrix::Bt709);
        assert_eq!(bt709.range, ColorRange::Limited);
        assert_eq!(bt709.transfer, ColorTransfer::Bt709);
        assert_eq!(bt709.primaries, ColorPrimaries::Bt709);
        assert_ne!(bt709, desktop);
    }

    #[test]
    fn round_trip_audio_packet_opus() {
        use crate::audio::{AudioConfig, AudioPacket};
        use bytes::Bytes;
        let p = AudioPacket::Opus {
            stream_epoch: 1,
            frame_seq: 1234,
            t_capture: MonoNanos(98765),
            payload: Bytes::from(vec![0xAB; 64]),
            // RED tail: two previous frames, newest-first.
            redundant: vec![Bytes::from(vec![0xCD; 48]), Bytes::from(vec![0xEF; 32])],
        };
        let bytes = encode(&p).unwrap();
        let p2: AudioPacket = decode(&bytes).unwrap();
        assert_eq!(p, p2);

        // The empty-redundancy case (redundancy off / stream start) also
        // round-trips — a client that never populates the tail is unaffected.
        let p_no_red = AudioPacket::Opus {
            stream_epoch: 1,
            frame_seq: 1235,
            t_capture: MonoNanos(98766),
            payload: Bytes::from(vec![0x11; 16]),
            redundant: vec![],
        };
        assert_eq!(p_no_red, decode(&encode(&p_no_red).unwrap()).unwrap());

        // Typed hello audio config round-trips identically too.
        let cfg = AudioConfig {
            sample_rate_hz: 48_000,
            channels: 2,
            streams: 1,
            coupled_streams: 1,
            channel_mapping: vec![0, 1],
        };
        let cfg_bytes = encode(&cfg).unwrap();
        let cfg2: AudioConfig = decode(&cfg_bytes).unwrap();
        assert_eq!(cfg, cfg2);
    }

    /// A near-maximum Opus payload (a high-bitrate / long-frame config can
    /// approach `tether_audio::MAX_PACKET_BYTES` = 4000) must survive the
    /// untrusted-decode allocation guard in [`decode_datagram`]. Regression
    /// for the ceiling sitting below the largest legitimate audio packet,
    /// which silently dropped such datagrams (audio dropouts, no log).
    #[test]
    fn near_max_audio_datagram_decodes() {
        use crate::audio::AudioPacket;
        use bytes::Bytes;
        // 4000-byte payload (tether_audio::MAX_PACKET_BYTES), above the prior
        // 2048 decode ceiling. The `Datagram::Audio` wrapper (in
        // tether-transport) adds ~1 byte; decode the AudioPacket directly here
        // since the payload is what the allocation guard bounds.
        let payload = vec![0x5Au8; 4000];
        assert!(payload.len() > 2048, "test must exceed the old ceiling");
        let packet = AudioPacket::Opus {
            stream_epoch: 7,
            frame_seq: 999,
            t_capture: MonoNanos(424242),
            payload: Bytes::from(payload.clone()),
            redundant: vec![],
        };
        let bytes = encode(&packet).unwrap();
        assert!(
            bytes.len() <= MAX_DATAGRAM_DECODE_BYTES,
            "encoded {} must fit the decode ceiling {}",
            bytes.len(),
            MAX_DATAGRAM_DECODE_BYTES
        );
        let decoded: AudioPacket =
            decode_datagram(&bytes).expect("near-max audio datagram must decode");
        match decoded {
            AudioPacket::Opus { payload: p, .. } => assert_eq!(&p[..], &payload[..]),
        }
    }

    /// A realistically-sized primary payload plus a depth-1 RED tail (the
    /// production shape) round-trips through the bounded datagram decoder. Pins
    /// that a normal RED-carrying datagram stays under the ceiling, so a future
    /// ceiling change that would silently drop RED packets trips this.
    #[test]
    fn typical_audio_datagram_with_red_decodes() {
        use crate::audio::AudioPacket;
        use bytes::Bytes;
        // 160 B ~ a 10 ms / 128 kbps frame (the default 5 ms frame is ~80 B);
        // depth-1 RED adds one prior copy. A larger-than-default vector here
        // exercises the ceiling with margin.
        let payload = vec![0x33u8; 160];
        let packet = AudioPacket::Opus {
            stream_epoch: 0,
            frame_seq: 42,
            t_capture: MonoNanos(1000),
            payload: Bytes::from(payload.clone()),
            redundant: vec![Bytes::from(vec![0x44u8; 160])],
        };
        let bytes = encode(&packet).unwrap();
        assert!(
            bytes.len() <= MAX_DATAGRAM_DECODE_BYTES,
            "a typical primary + depth-1 RED datagram ({} B) must fit the ceiling {}",
            bytes.len(),
            MAX_DATAGRAM_DECODE_BYTES
        );
        let decoded: AudioPacket =
            decode_datagram(&bytes).expect("typical RED datagram must decode");
        match decoded {
            AudioPacket::Opus {
                payload: p,
                redundant: r,
                ..
            } => {
                assert_eq!(&p[..], &payload[..]);
                assert_eq!(r, vec![Bytes::from(vec![0x44u8; 160])]);
            }
        }
    }

    /// The bounded datagram decoder caps total allocation across the whole
    /// `AudioPacket` — including the RED tail — so a forged `redundant` vec of
    /// many large payloads can't drive a huge pre-allocation. A datagram whose
    /// combined payload + redundancy exceeds the ceiling is rejected, not
    /// allocated.
    #[test]
    fn oversize_red_tail_is_rejected_by_the_decode_guard() {
        use crate::audio::AudioPacket;
        use bytes::Bytes;
        // Several max-size payloads in the tail blow well past the ceiling.
        let packet = AudioPacket::Opus {
            stream_epoch: 0,
            frame_seq: 0,
            t_capture: MonoNanos(0),
            payload: Bytes::from(vec![0u8; 2000]),
            redundant: vec![Bytes::from(vec![0u8; 2000]), Bytes::from(vec![0u8; 2000])],
        };
        let bytes = encode(&packet).unwrap();
        assert!(
            bytes.len() > MAX_DATAGRAM_DECODE_BYTES,
            "test must exceed the ceiling to exercise the guard"
        );
        assert!(
            decode_datagram::<AudioPacket>(&bytes).is_err(),
            "an over-ceiling RED datagram must be rejected before allocation"
        );
    }

    #[test]
    fn round_trip_client_stats() {
        let msg = ControlMessage::ClientStats {
            window_ms: 1000,
            frames_received: 60,
            incomplete_frames: 2,
            fragment_loss_events: 4,
            rtt_us: 9_500,
            fec_recovered_frames: 1,
            fec_recovered_fragments: 3,
        };
        let bytes = encode_reliable(&msg).unwrap();
        let msg2: ControlMessage = decode_reliable(&bytes).unwrap();
        assert_eq!(msg, msg2);
    }

    #[test]
    fn client_stats_maps_named_fields_without_cross_wire_swap() {
        use crate::pb::control_message::Kind;
        use prost::Message as _;

        let msg = ControlMessage::ClientStats {
            window_ms: 1001,
            frames_received: 62,
            incomplete_frames: 3,
            fragment_loss_events: 5,
            rtt_us: 7000,
            fec_recovered_frames: 7,
            fec_recovered_fragments: 11,
        };
        let bytes = encode_reliable(&msg).unwrap();
        let wire = pb::ControlMessage::decode(bytes.as_slice()).unwrap();
        match wire.kind.expect("kind") {
            Kind::ClientStats(stats) => {
                assert_eq!(stats.window_ms, 1001);
                assert_eq!(stats.frames_received, 62);
                assert_eq!(stats.incomplete_frames, 3);
                assert_eq!(stats.fragment_loss_events, 5);
                assert_eq!(stats.rtt_us, 7000);
                assert_eq!(stats.fec_recovered_frames, 7);
                assert_eq!(stats.fec_recovered_fragments, 11);
            }
            other => panic!("expected ClientStats, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_stream_lifecycle() {
        // The three lifecycle variants gate host frame emission. All
        // three need to survive the wire identically.
        for msg in [
            ControlMessage::StreamReady {
                video: true,
                audio: false,
            },
            ControlMessage::StreamPause {
                stream_id: VideoStreamId(3),
            },
            ControlMessage::StreamResume {
                stream_id: VideoStreamId(3),
            },
        ] {
            let bytes = encode_reliable(&msg).unwrap();
            let msg2: ControlMessage = decode_reliable(&bytes).unwrap();
            assert_eq!(msg, msg2);
        }
    }

    #[test]
    fn round_trip_control_extension() {
        // The Extension escape unblocks future control features
        // without forcing a ClientHelloV2. Confirm it survives the
        // wire identically.
        let msg = ControlMessage::Extension(ExtensionMessage {
            key: "tether.cap.test".into(),
            version: 1,
            request_id: RequestId(9),
            reply_to: RequestId(0),
            payload: vec![1, 2, 3, 0xFF],
        });
        let bytes = encode_reliable(&msg).unwrap();
        let msg2: ControlMessage = decode_reliable(&bytes).unwrap();
        match msg2 {
            ControlMessage::Extension(msg) => {
                assert_eq!(msg.key, "tether.cap.test");
                assert_eq!(msg.version, 1);
                assert_eq!(msg.request_id, RequestId(9));
                assert_eq!(msg.reply_to, RequestId(0));
                assert_eq!(msg.payload, vec![1, 2, 3, 0xFF]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn oversized_extension_payload_fails_decode() {
        let msg = crate::pb::ControlMessage {
            kind: Some(crate::pb::control_message::Kind::Extension(
                crate::pb::ExtensionMessage {
                    key: "tether.exp.too-big".into(),
                    version: 1,
                    request_id: 1,
                    reply_to: 0,
                    payload: vec![0; MAX_EXTENSION_PAYLOAD_BYTES + 1],
                },
            )),
        };
        let bytes = msg.encode_to_vec();
        let err = decode_reliable::<ControlMessage>(&bytes).unwrap_err();
        assert!(matches!(
            err,
            CodecError::Wire("extension payload too large")
        ));
    }

    #[test]
    fn unknown_protobuf_control_enums_decode_to_unknown_variants() {
        let goodbye = crate::pb::ControlMessage {
            kind: Some(crate::pb::control_message::Kind::Goodbye(
                crate::pb::Goodbye {
                    reason: "future".into(),
                    code: 99,
                    final_stats: None,
                },
            )),
        };
        match decode_reliable::<ControlMessage>(&goodbye.encode_to_vec()).unwrap() {
            ControlMessage::Goodbye { code, .. } => assert_eq!(code, GoodbyeCode::Unknown(99)),
            other => panic!("expected Goodbye, got {other:?}"),
        }

        let mode = crate::pb::ControlMessage {
            kind: Some(crate::pb::control_message::Kind::SetCursorMode(
                crate::pb::SetCursorMode { mode: 77 },
            )),
        };
        match decode_reliable::<ControlMessage>(&mode.encode_to_vec()).unwrap() {
            ControlMessage::SetCursorMode { mode } => assert_eq!(mode, CursorMode::Unknown(77)),
            other => panic!("expected SetCursorMode, got {other:?}"),
        }

        let result = crate::pb::ControlMessage {
            kind: Some(crate::pb::control_message::Kind::DisplayModeResult(
                crate::pb::DisplayModeResult {
                    request_id: 1,
                    display_id: 2,
                    status: 88,
                    actual_mode: None,
                },
            )),
        };
        match decode_reliable::<ControlMessage>(&result.encode_to_vec()).unwrap() {
            ControlMessage::DisplayModeResult { status, .. } => {
                assert_eq!(status, DisplayModeStatus::Unknown(88));
            }
            other => panic!("expected DisplayModeResult, got {other:?}"),
        }
    }

    #[test]
    fn cursor_shape_payload_must_match_rgba_dimensions() {
        let msg = crate::pb::ControlMessage {
            kind: Some(crate::pb::control_message::Kind::CursorShape(
                crate::pb::CursorShape {
                    id: 42,
                    hotspot_x: 0,
                    hotspot_y: 0,
                    width: 16,
                    height: 16,
                    format: 1,
                    pixels: vec![0; (16 * 16 * 4) - 1],
                },
            )),
        };
        let err = decode_reliable::<ControlMessage>(&msg.encode_to_vec()).unwrap_err();
        assert!(matches!(
            err,
            CodecError::Wire("cursor shape payload length mismatch")
        ));
    }

    #[test]
    fn oversized_cursor_shape_payload_fails_decode() {
        let msg = crate::pb::ControlMessage {
            kind: Some(crate::pb::control_message::Kind::CursorShape(
                crate::pb::CursorShape {
                    id: 42,
                    hotspot_x: 0,
                    hotspot_y: 0,
                    width: 129,
                    height: 128,
                    format: 1,
                    pixels: vec![0; (129 * 128 * 4) as usize],
                },
            )),
        };
        let err = decode_reliable::<ControlMessage>(&msg.encode_to_vec()).unwrap_err();
        assert!(matches!(
            err,
            CodecError::Wire("cursor shape payload too large")
        ));
    }

    #[test]
    fn goodbye_carries_machine_readable_code() {
        let g = ControlMessage::Goodbye {
            reason: "user quit".into(),
            code: GoodbyeCode::Clean,
            final_stats: None,
        };
        let bytes = encode_reliable(&g).unwrap();
        let g2: ControlMessage = decode_reliable(&bytes).unwrap();
        match g2 {
            ControlMessage::Goodbye {
                reason,
                code,
                final_stats,
            } => {
                assert_eq!(reason, "user quit");
                assert_eq!(code, GoodbyeCode::Clean);
                assert!(final_stats.is_none());
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn goodbye_carries_final_session_summary() {
        let summary = SessionSummary {
            role: "client".into(),
            duration_ms: 1_234,
            codec: "Hevc".into(),
            chroma: "Yuv420".into(),
            bit_depth: 10,
            video: VideoSessionStats {
                frames_sent: 0,
                frames_received: 99,
                keyframes: 2,
                bytes_sent: 0,
                bytes_received: 4_096,
                incomplete_frames: 1,
                fragment_loss_events: 3,
                decode_errors: 4,
                render_drop_frames: 5,
                idr_requests: 6,
                decode_queue_drop_frames: 7,
                transient_send_drop_frames: 0,
                fec_recovered_frames: 8,
                fec_recovered_fragments: 9,
                datagrams_sent: 10,
                parity_datagrams_sent: 11,
                max_datagrams_per_frame: 12,
                max_frame_bytes: 13,
                max_keyframe_bytes: 14,
                forced_idr_misses: 15,
                decode_stale_epoch_drop_frames: 16,
                decode_epoch_throttle_drop_frames: 17,
            },
            audio: Some(AudioSessionStats {
                packets_sent: 0,
                packets_received: 88,
                capture_frames: 0,
                underruns: 1,
                dropped_samples: 2,
                recovered_frames: 3,
                concealed_frames: 4,
                dropout_frames: 5,
                dropouts: 6,
                stale_packets: 7,
                decode_errors: 8,
                decode_queue_drop_packets: 9,
            }),
        };
        let msg = ControlMessage::Goodbye {
            reason: "done".into(),
            code: GoodbyeCode::Clean,
            final_stats: Some(Box::new(summary.clone())),
        };

        let bytes = encode_reliable(&msg).unwrap();
        let decoded: ControlMessage = decode_reliable(&bytes).unwrap();

        match decoded {
            ControlMessage::Goodbye {
                reason,
                code,
                final_stats,
            } => {
                assert_eq!(reason, "done");
                assert_eq!(code, GoodbyeCode::Clean);
                assert_eq!(final_stats.as_deref(), Some(&summary));
            }
            other => panic!("expected Goodbye, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_video_packet_first() {
        let p = VideoPacket::First {
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 42,
            fragment_count: 3,
            fec_pct: 20,
            shard_size: 1100,
            total_body_len: 3000,
            meta: VideoFrameMetaEnvelope::V1(VideoFrameMeta {
                timing: HostFrameTiming {
                    t_capture_kernel: MonoNanos(1_000),
                    t_capture_userspace: MonoNanos(1_100),
                    t_encode_submit: MonoNanos(1_200),
                    t_encode_done: MonoNanos(2_500),
                    t_send: MonoNanos(2_600),
                },
                keyframe: true,
                input_echo: InputEchoBatch {
                    event_ids: vec![1, 2, 3],
                },
                dimensions: (320, 240),
            }),
            payload: bytes::Bytes::from(vec![0xAA; 1100]),
        };
        let bytes = encode(&p).unwrap();
        let p2: VideoPacket = decode(&bytes).unwrap();
        match p2 {
            VideoPacket::First {
                stream_id,
                stream_epoch,
                frame_seq,
                fragment_count,
                fec_pct,
                shard_size,
                total_body_len,
                meta,
                payload,
            } => {
                assert_eq!(stream_id, VideoStreamId(0));
                assert_eq!(stream_epoch, 0);
                assert_eq!(frame_seq, 42);
                assert_eq!(fragment_count, 3);
                assert_eq!(fec_pct, 20);
                assert_eq!(shard_size, 1100);
                assert_eq!(total_body_len, 3000);
                let meta = meta.into_meta();
                assert!(meta.keyframe);
                assert_eq!(meta.input_echo.event_ids, vec![1, 2, 3]);
                assert_eq!(payload.len(), 1100);
            }
            VideoPacket::Continuation { .. } | VideoPacket::Parity { .. } => {
                panic!("wrong variant")
            }
        }
    }

    #[test]
    fn relative_mouse_move_input_event_round_trips() {
        let e = InputEvent {
            event_id: 7,
            t_client: MonoNanos(42),
            device_id: 0,
            kind: InputEventKind::RelativeMouseMove {
                dx: -120,
                dy: 250,
                modifiers: Modifiers::default(),
            },
        };
        let bytes = encode(&e).unwrap();
        let e2: InputEvent = decode(&bytes).unwrap();
        match e2.kind {
            InputEventKind::RelativeMouseMove { dx, dy, .. } => {
                assert_eq!(dx, -120);
                assert_eq!(dy, 250);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn set_cursor_mode_round_trips_and_defaults_to_absolute() {
        assert_eq!(CursorMode::default(), CursorMode::Absolute);
        let m = ControlMessage::SetCursorMode {
            mode: CursorMode::Relative,
        };
        let bytes = encode(&m).unwrap();
        let m2: ControlMessage = decode(&bytes).unwrap();
        match m2 {
            ControlMessage::SetCursorMode { mode } => assert_eq!(mode, CursorMode::Relative),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn video_packet_parity_round_trips_all_fields() {
        // Every field on VideoPacket::Parity must round-trip: recovery depends
        // on fragment_count, fec_pct, shard_size, block_index, and parity_index
        // agreeing wire-to-receiver, and total_body_len sizes the reassembler's
        // BytesMut. Bincode positional encoding makes any field reordering a
        // silent data corruption — assert each value individually.
        let p = VideoPacket::Parity {
            stream_id: VideoStreamId(3),
            stream_epoch: 42,
            frame_seq: 1001,
            fragment_count: 8,
            fec_pct: 25,
            shard_size: 1100,
            total_body_len: 8500,
            block_index: 1,
            parity_index: 1,
            meta: VideoFrameMetaEnvelope::V1(default_meta()),
            payload: bytes::Bytes::from(vec![0xee; 128]),
        };
        let bytes = encode(&p).unwrap();
        let p2: VideoPacket = decode(&bytes).unwrap();
        match p2 {
            VideoPacket::Parity {
                stream_id,
                stream_epoch,
                frame_seq,
                fragment_count,
                fec_pct,
                shard_size,
                total_body_len,
                block_index,
                parity_index,
                meta,
                payload,
            } => {
                assert_eq!(stream_id, VideoStreamId(3));
                assert_eq!(stream_epoch, 42);
                assert_eq!(frame_seq, 1001);
                assert_eq!(fragment_count, 8);
                assert_eq!(fec_pct, 25);
                assert_eq!(shard_size, 1100);
                assert_eq!(total_body_len, 8500);
                assert_eq!(block_index, 1);
                assert_eq!(parity_index, 1);
                assert_eq!(meta.into_meta().dimensions, (640, 480));
                assert_eq!(payload.len(), 128);
                assert!(payload.iter().all(|&b| b == 0xee));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn request_recovery_control_message_round_trips() {
        let m = ControlMessage::RequestRecovery {
            last_reassembled_frame_id: 12345,
        };
        let bytes = encode(&m).unwrap();
        let m2: ControlMessage = decode(&bytes).unwrap();
        match m2 {
            ControlMessage::RequestRecovery {
                last_reassembled_frame_id,
            } => assert_eq!(last_reassembled_frame_id, 12345),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn videoframe_meta_envelope_round_trip() {
        // The envelope discriminator is the load-bearing addition:
        // future per-frame metadata (HDR, ROI, QP) lands as new
        // variants without breaking the wire. Confirm V1 unwraps
        // back to the original meta.
        let original = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: true,
            input_echo: InputEchoBatch::default(),
            dimensions: (1280, 720),
        };
        let env = VideoFrameMetaEnvelope::V1(original.clone());
        let bytes = encode(&env).unwrap();
        let env2: VideoFrameMetaEnvelope = decode(&bytes).unwrap();
        let unwrapped = env2.into_meta();
        assert_eq!(unwrapped, original);
    }

    #[test]
    fn round_trip_video_packet_stream_epoch_above_u16() {
        // stream_epoch is u32: a long-lived host that restarts the
        // encoder past u16::MAX (65_535) must round-trip cleanly.
        // 70_000 is just past that ceiling.
        let epoch = 70_000u32;
        let p = VideoPacket::First {
            stream_id: VideoStreamId(0),
            stream_epoch: epoch,
            frame_seq: 0,
            fragment_count: 1,
            fec_pct: 0,
            shard_size: 256,
            total_body_len: 0,
            meta: VideoFrameMetaEnvelope::V1(VideoFrameMeta {
                timing: HostFrameTiming::default(),
                keyframe: true,
                input_echo: InputEchoBatch::default(),
                dimensions: (1, 1),
            }),
            payload: bytes::Bytes::new(),
        };
        let bytes = encode(&p).unwrap();
        let p2: VideoPacket = decode(&bytes).unwrap();
        match p2 {
            VideoPacket::First { stream_epoch, .. } => assert_eq!(stream_epoch, epoch),
            VideoPacket::Continuation { .. } | VideoPacket::Parity { .. } => {
                panic!("wrong variant")
            }
        }
    }

    #[test]
    fn round_trip_cursor_position() {
        let c = HostCursorPacket::Position {
            t_capture: MonoNanos(999),
            x: 100,
            y: -50,
            visible: true,
        };
        let bytes = encode(&c).unwrap();
        let c2: HostCursorPacket = decode(&bytes).unwrap();
        let HostCursorPacket::Position {
            t_capture,
            x,
            y,
            visible,
        } = c2;
        assert_eq!(t_capture, MonoNanos(999));
        assert_eq!(x, 100);
        assert_eq!(y, -50);
        assert!(visible);
    }

    #[test]
    fn round_trip_input_event() {
        let e = InputEvent {
            event_id: 12345,
            t_client: MonoNanos(54321),
            device_id: 1, // non-zero to confirm varint shape is fine
            kind: InputEventKind::KeyDown {
                key: HidUsage(0x0007_0004), // HID page 7 (kbd), usage 4 ('a')
                modifiers: Modifiers {
                    shift: true,
                    ..Default::default()
                },
            },
        };
        let bytes = encode(&e).unwrap();
        let e2: InputEvent = decode(&bytes).unwrap();
        assert_eq!(e.event_id, e2.event_id);
        assert_eq!(e.t_client, e2.t_client);
        assert_eq!(e2.device_id, 1);
        match e2.kind {
            InputEventKind::KeyDown { key, modifiers } => {
                assert_eq!(key, HidUsage(0x0007_0004));
                assert!(modifiers.shift);
                assert!(!modifiers.ctrl);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn first_video_packet_fits_in_datagram() {
        // Stress test: max-valued numeric fields + a realistic input echo +
        // payload sized so the whole packet stays under the datagram budget.
        let p = VideoPacket::First {
            stream_id: VideoStreamId(u32::from(u8::MAX)),
            stream_epoch: u32::MAX,
            frame_seq: u32::MAX,
            fragment_count: u16::MAX,
            fec_pct: u8::MAX,
            shard_size: u32::MAX,
            total_body_len: u32::MAX,
            meta: VideoFrameMetaEnvelope::V1(VideoFrameMeta {
                timing: HostFrameTiming {
                    t_capture_kernel: MonoNanos(u64::MAX / 2),
                    t_capture_userspace: MonoNanos(u64::MAX / 2),
                    t_encode_submit: MonoNanos(u64::MAX / 2),
                    t_encode_done: MonoNanos(u64::MAX / 2),
                    t_send: MonoNanos(u64::MAX / 2),
                },
                keyframe: true,
                input_echo: InputEchoBatch {
                    event_ids: vec![u64::MAX; 4],
                },
                dimensions: (u32::MAX, u32::MAX),
            }),
            payload: bytes::Bytes::from(vec![0u8; 1040]),
        };
        let bytes = encode(&p).unwrap();
        assert!(
            bytes.len() <= MAX_DATAGRAM_PAYLOAD,
            "first-packet encoded size {} exceeds {}",
            bytes.len(),
            MAX_DATAGRAM_PAYLOAD
        );
    }

    #[test]
    fn fragmenter_with_fec_zero_is_wire_identical_to_no_fec() {
        // fec_percentage=0 emits no Parity packets; the two constructors are
        // equivalent, producing byte-identical primary fragmentation.
        let body: bytes::Bytes = vec![0xab; 5000].into();
        let mut a = FrameFragmenter::new(0u8);
        let mut b = FrameFragmenter::new_with_fec(0u8, 0);
        let pa = a.fragment(default_meta(), body.clone(), MAX_DATAGRAM_PAYLOAD);
        let pb = b.fragment(default_meta(), body, MAX_DATAGRAM_PAYLOAD);
        assert_eq!(pa.len(), pb.len());
        for (x, y) in pa.iter().zip(pb.iter()) {
            assert_eq!(encode(x).unwrap(), encode(y).unwrap());
        }
        // And no parity packets in either.
        assert!(pa.iter().all(|p| !matches!(p, VideoPacket::Parity { .. })));
    }

    fn default_meta() -> VideoFrameMeta {
        VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (640, 480),
        }
    }

    #[test]
    fn fragmenter_with_fec_emits_parity_proportional_to_percentage() {
        let body: bytes::Bytes = vec![0u8; 11_000].into();
        let mut frag = FrameFragmenter::new_with_fec(0u8, 20);
        let pkts = frag.fragment(default_meta(), body, MAX_DATAGRAM_PAYLOAD);
        let primary = pkts
            .iter()
            .filter(|p| {
                matches!(
                    p,
                    VideoPacket::First { .. } | VideoPacket::Continuation { .. }
                )
            })
            .count();
        let parity = pkts
            .iter()
            .filter(|p| matches!(p, VideoPacket::Parity { .. }))
            .count();
        // Shard size is derived from the datagram budget, so the exact primary
        // count depends on the meta size — assert the FEC relationship instead:
        // a single block (this body is well under the per-block ceiling) gets
        // ceil(primary * 20 / 100) parity shards.
        assert!(primary > 1, "multi-shard body expected, got {primary}");
        assert_eq!(parity, primary.div_ceil(5), "20% parity, single block");
    }

    #[test]
    fn fec_recovers_from_losing_up_to_parity_count_primaries() {
        // Drop up to K of the N primaries; reassembler must
        // reconstruct from the K parity shards.
        let body: bytes::Bytes = (0..5000u32)
            .map(|i| (i & 0xff) as u8)
            .collect::<Vec<u8>>()
            .into();
        let mut frag = FrameFragmenter::new_with_fec(0u8, 25); // 25% parity
        let pkts = frag.fragment(default_meta(), body.clone(), MAX_DATAGRAM_PAYLOAD);
        let parity_count = pkts
            .iter()
            .filter(|p| matches!(p, VideoPacket::Parity { .. }))
            .count();
        // 5000 / 1100 = 5 primaries → 25% = 2 parity (rounded up).
        assert!(parity_count >= 1);

        // Drop the first `parity_count` primaries; keep all parity.
        let kept: Vec<_> = pkts
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i >= parity_count) // drop primaries 0..parity_count
            .map(|(_, p)| p)
            .collect();
        let mut r = FrameReassembler::new();
        let recovery_before = r.recovery_counters();
        let mut got = None;
        for p in kept {
            if let Some(f) = r.handle(p) {
                got = Some(f);
            }
        }
        let f = got.expect("reassembled via FEC");
        assert_eq!(f.body.as_ref(), body.as_ref());
        let recovery_after = r.recovery_counters();
        assert_eq!(recovery_after.0, recovery_before.0 + 1);
        assert_eq!(
            recovery_after.1,
            recovery_before.1 + u64::try_from(parity_count).unwrap()
        );
    }

    #[test]
    fn fec_fails_when_loss_exceeds_parity() {
        // Losing more primaries than we have parity for must NOT
        // produce a reconstructed frame.
        let body: bytes::Bytes = vec![0xff; 5000].into();
        let mut frag = FrameFragmenter::new_with_fec(0u8, 20); // 1 parity for 5 primaries
        let pkts = frag.fragment(default_meta(), body, MAX_DATAGRAM_PAYLOAD);
        let parity_count = pkts
            .iter()
            .filter(|p| matches!(p, VideoPacket::Parity { .. }))
            .count();
        // Drop parity_count + 1 primaries.
        let kept: Vec<_> = pkts
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i > parity_count)
            .map(|(_, p)| p)
            .collect();
        let mut r = FrameReassembler::new();
        let mut got = None;
        for p in kept {
            if let Some(f) = r.handle(p) {
                got = Some(f);
            }
        }
        assert!(got.is_none(), "loss above parity must not reconstruct");
        assert_eq!(
            r.recovery_counters(),
            (0, 0),
            "failed partial recovery must not count as useful repair"
        );
    }

    #[test]
    fn large_frame_splits_into_multiple_fec_blocks() {
        // #36: a frame whose primary count exceeds a single block's per-pct
        // ceiling is split into multiple independent RS blocks and stays
        // loss-protected — no more no-FEC fallback for big IDRs.
        let body: bytes::Bytes = vec![0x5au8; 300_000].into();
        let mut frag = FrameFragmenter::new_with_fec(0u8, 20);
        let pkts = frag.fragment(default_meta(), body, MAX_DATAGRAM_PAYLOAD);
        let primary = pkts
            .iter()
            .filter(|p| {
                matches!(
                    p,
                    VideoPacket::First { .. } | VideoPacket::Continuation { .. }
                )
            })
            .count();
        assert!(
            primary > FEC_MAX_PRIMARY_SHARDS,
            "body should need more than one block's worth of primaries, got {primary}"
        );
        let max_block = pkts
            .iter()
            .filter_map(|p| match p {
                VideoPacket::Parity { block_index, .. } => Some(*block_index),
                _ => None,
            })
            .max()
            .expect("multi-block frame must carry parity");
        assert!(
            max_block >= 1,
            "expected >1 FEC block, max block_index {max_block}"
        );
        // The block layout the receiver derives from (K, fec_pct) matches the
        // number of blocks the sender actually emitted parity for.
        assert_eq!(fec_layout(primary, 20).len() as u16, max_block + 1);
    }

    #[test]
    fn multi_block_frame_recovers_lost_primaries() {
        // End-to-end multi-block recovery: drop several primaries from the
        // first block (within its parity budget) and confirm the reassembler
        // rebuilds the exact body. Losing the First also exercises descriptor
        // + meta recovery from a parity packet.
        let body: bytes::Bytes = (0..300_000u32)
            .map(|i| (i & 0xff) as u8)
            .collect::<Vec<u8>>()
            .into();
        let mut frag = FrameFragmenter::new_with_fec(0u8, 20);
        let pkts = frag.fragment(default_meta(), body.clone(), MAX_DATAGRAM_PAYLOAD);
        // Packets are ordered First, Continuations…, then Parity. Drop the
        // first 3 primaries (all in block 0); keep the rest + all parity.
        let kept: Vec<_> = pkts
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i >= 3)
            .map(|(_, p)| p)
            .collect();
        let mut r = FrameReassembler::new();
        let mut got = None;
        for p in kept {
            if let Some(f) = r.handle(p) {
                got = Some(f);
            }
        }
        let f = got.expect("multi-block frame reassembled via per-block FEC");
        assert_eq!(f.body.as_ref(), body.as_ref());
    }

    #[test]
    fn parity_packet_round_trips() {
        let p = VideoPacket::Parity {
            stream_id: VideoStreamId(7),
            stream_epoch: 42,
            frame_seq: 100,
            fragment_count: 5,
            fec_pct: 20,
            shard_size: 1100,
            total_body_len: 5000,
            block_index: 0,
            parity_index: 0,
            meta: VideoFrameMetaEnvelope::V1(default_meta()),
            payload: bytes::Bytes::from(vec![0x42; 1100]),
        };
        let bytes = encode(&p).unwrap();
        assert!(
            bytes.len() <= MAX_DATAGRAM_PAYLOAD,
            "parity packet must fit in a datagram"
        );
        let p2: VideoPacket = decode(&bytes).unwrap();
        match p2 {
            VideoPacket::Parity {
                stream_id,
                stream_epoch,
                frame_seq,
                fragment_count,
                fec_pct,
                shard_size,
                total_body_len,
                block_index,
                parity_index,
                meta,
                payload,
            } => {
                assert_eq!(stream_id, VideoStreamId(7));
                assert_eq!(stream_epoch, 42);
                assert_eq!(frame_seq, 100);
                assert_eq!(fragment_count, 5);
                assert_eq!(fec_pct, 20);
                assert_eq!(shard_size, 1100);
                assert_eq!(total_body_len, 5000);
                assert_eq!(block_index, 0);
                assert_eq!(parity_index, 0);
                let m = meta.into_meta();
                assert_eq!(m.dimensions, (640, 480));
                assert_eq!(payload.len(), 1100);
            }
            _ => panic!("expected Parity"),
        }
    }

    #[test]
    fn wire_size_matches_serialized_length() {
        // The pacer relies on `wire_size()` for byte accounting.
        // Must equal the exact `encode().len()` for every packet
        // shape the fragmenter produces — including Parity.
        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (1920, 1080),
        };
        let body: bytes::Bytes = vec![0u8; 8 * 1024].into();
        let mut frag = FrameFragmenter::new(0u8);
        let packets = frag.fragment(meta.clone(), body.clone(), MAX_DATAGRAM_PAYLOAD);
        for p in &packets {
            assert_eq!(
                p.wire_size(),
                encode(p).unwrap().len(),
                "wire_size must equal actual serialized length"
            );
        }

        // FEC path: include Parity in the coverage so a future
        // change to Parity's fields can't silently desync wire_size
        // from the actual encoded length.
        let mut frag_fec = FrameFragmenter::new_with_fec(0u8, 20);
        let packets_fec = frag_fec.fragment(meta, body, MAX_DATAGRAM_PAYLOAD);
        assert!(
            packets_fec
                .iter()
                .any(|p| matches!(p, VideoPacket::Parity { .. })),
            "FEC fragmenter must emit Parity for this body size"
        );
        for p in &packets_fec {
            assert_eq!(
                p.wire_size(),
                encode(p).unwrap().len(),
                "wire_size must equal actual serialized length for {p:?}"
            );
        }
    }

    #[test]
    fn fragment_then_reassemble_round_trips() {
        let body: Vec<u8> = (0..10_000u32).map(|i| (i & 0xFF) as u8).collect();
        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: true,
            input_echo: InputEchoBatch::default(),
            dimensions: (320, 240),
        };

        let mut fragmenter = FrameFragmenter::new(0u8);
        let packets = fragmenter.fragment(
            meta.clone(),
            bytes::Bytes::from(body.clone()),
            MAX_DATAGRAM_PAYLOAD,
        );
        assert!(packets.len() > 1, "10 KB body must span multiple shards");

        // Encode every packet to wire and back to simulate the network
        // boundary, then reassemble.
        let mut reassembler = FrameReassembler::new();
        let mut got = None;
        for p in packets {
            let bytes = encode(&p).unwrap();
            assert!(bytes.len() <= MAX_DATAGRAM_PAYLOAD);
            let p2: VideoPacket = decode(&bytes).unwrap();
            if let Some(frame) = reassembler.handle(p2) {
                got = Some(frame);
            }
        }
        let frame = got.expect("reassembled frame");
        assert_eq!(frame.body.as_ref(), body.as_slice());
        assert_eq!(frame.frame_seq, 0);
        assert_eq!(frame.stream_id, VideoStreamId(0));
        assert!(frame.meta.keyframe);
    }

    #[test]
    fn reassembler_handles_out_of_order_fragments() {
        let body: Vec<u8> = (0..5_000u32).map(|i| (i & 0xFF) as u8).collect();
        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (320, 240),
        };
        let mut fragmenter = FrameFragmenter::new(2u8);
        let mut packets =
            fragmenter.fragment(meta, bytes::Bytes::from(body.clone()), MAX_DATAGRAM_PAYLOAD);
        // Reverse the order — reassembler should still produce the frame.
        packets.reverse();

        let mut reassembler = FrameReassembler::new();
        let mut got = None;
        for p in packets {
            if let Some(frame) = reassembler.handle(p) {
                got = Some(frame);
            }
        }
        let frame = got.expect("reassembled out-of-order frame");
        assert_eq!(frame.body.as_ref(), body.as_slice());
        assert_eq!(frame.stream_id, VideoStreamId(2));
    }

    #[test]
    fn reassembler_drops_stale_fragments() {
        let mut fragmenter = FrameFragmenter::new(0u8);
        let mut reassembler = FrameReassembler::new().with_max_age(1);

        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (320, 240),
        };

        // Advance latest_seq on the reassembler to 5 by feeding it 6
        // fully-assembled tiny frames (seqs 0..=5).
        for _ in 0..6 {
            for p in fragmenter.fragment(
                meta.clone(),
                bytes::Bytes::from_static(&[0u8; 100]),
                MAX_DATAGRAM_PAYLOAD,
            ) {
                reassembler.handle(p);
            }
        }

        // Now inject a stale Continuation claiming to belong to seq 0 —
        // 5 frames behind latest, max_age=1, so the reassembler should
        // drop it silently.
        let stale = VideoPacket::Continuation {
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 0,
            fragment_index: 1,
            payload: bytes::Bytes::from_static(&[0u8; 10]),
        };
        assert!(reassembler.handle(stale).is_none());
    }

    #[test]
    fn reassembler_evicts_pending_past_wall_clock_timeout() {
        // Quiet-stream case: a frame goes incomplete and no newer
        // frames arrive to advance `latest_seq` past `max_age`. The
        // wall-clock timeout is the only thing standing between us
        // and a stuck pending entry that holds memory indefinitely.
        let mut reassembler =
            FrameReassembler::new().with_max_pending_age(std::time::Duration::from_millis(20));

        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (320, 240),
        };

        // Half-deliver frame 0 — First arrives but Continuation
        // doesn't.
        let first = VideoPacket::First {
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 0,
            fragment_count: 2,
            fec_pct: 0,
            shard_size: 100,
            total_body_len: 200,
            meta: VideoFrameMetaEnvelope::V1(meta.clone()),
            payload: bytes::Bytes::from_static(&[0u8; 100]),
        };
        assert!(reassembler.handle(first).is_none());
        let (dropped_before, _) = reassembler.loss_counters();
        assert_eq!(dropped_before, 0);

        std::thread::sleep(std::time::Duration::from_millis(40));

        // Feeding any other fragment triggers prune_old. The stuck
        // frame_seq=0 entry should be evicted by the wall-clock check.
        let unrelated = VideoPacket::First {
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 1,
            fragment_count: 1,
            fec_pct: 0,
            shard_size: 10,
            total_body_len: 10,
            meta: VideoFrameMetaEnvelope::V1(meta),
            payload: bytes::Bytes::from_static(&[0u8; 10]),
        };
        let _ = reassembler.handle(unrelated);
        let (dropped_after, _) = reassembler.loss_counters();
        assert_eq!(
            dropped_after, 1,
            "wall-clock timeout did not evict stuck pending frame"
        );
    }

    #[test]
    fn continuation_video_packet_fits_in_datagram() {
        // Even with max-valued numeric fields (worst case for varint
        // expansion), a continuation packet must fit in the datagram budget.
        // ~15 bytes of header overhead in the worst case for this variant.
        let p = VideoPacket::Continuation {
            stream_id: VideoStreamId(u32::from(u8::MAX)),
            stream_epoch: u32::MAX,
            frame_seq: u32::MAX,
            fragment_index: u16::MAX,
            payload: bytes::Bytes::from(vec![0u8; 1180]),
        };
        let bytes = encode(&p).unwrap();
        assert!(
            bytes.len() <= MAX_DATAGRAM_PAYLOAD,
            "continuation-packet encoded size {} exceeds {}",
            bytes.len(),
            MAX_DATAGRAM_PAYLOAD
        );
    }

    // Forward-compat probes. Each tagged enum we wire-serialise needs a
    // test that pins "older receiver rejects a future variant cleanly"
    // — otherwise a future V2 byte sequence could silently decode into
    // the current variant's body shape. The probe hand-crafts a
    // discriminator byte one past the highest known variant index;
    // bincode's varint enum encoding makes that the right byte
    // regardless of payload size.

    #[test]
    fn unknown_server_hello_variant_fails_decode() {
        // V1 = discriminator 0. Variant 1 is hypothetical V2.
        let bytes = [1u8, 0, 0, 0, 0];
        assert!(decode::<crate::control::ServerHello>(&bytes).is_err());
    }

    #[test]
    fn unknown_video_packet_variant_fails_decode() {
        // First = 0, Continuation = 1, Parity = 2. Variant 3 is the
        // hypothetical next addition.
        let bytes = [3u8, 0, 0, 0, 0];
        assert!(decode::<crate::video::VideoPacket>(&bytes).is_err());
    }

    #[test]
    fn unknown_host_cursor_packet_variant_fails_decode() {
        // Position = 0 (only defined variant). Variant 1 is hypothetical.
        let bytes = [1u8, 0, 0, 0, 0];
        assert!(decode::<crate::cursor::HostCursorPacket>(&bytes).is_err());
    }

    #[test]
    fn unknown_goodbye_code_variant_fails_decode() {
        // Clean = 0, ProtocolError = 1, UnsupportedVersion = 2,
        // InternalError = 3. Variant 4 is hypothetical. Pinning this
        // is important because `GoodbyeCode` drives reconnect
        // behaviour on the peer side; a future variant decoded as a
        // current one would silently misclassify the shutdown reason.
        let bytes = [4u8];
        assert!(decode::<crate::control::GoodbyeCode>(&bytes).is_err());
    }

    #[test]
    fn unknown_video_frame_meta_envelope_variant_fails_decode() {
        // V1 = 0. Variant 1 is hypothetical V2.
        let bytes = [1u8, 0, 0, 0, 0];
        assert!(decode::<crate::video::VideoFrameMetaEnvelope>(&bytes).is_err());
    }

    #[test]
    fn unknown_control_message_variant_fails_decode() {
        // ControlMessage has many variants. Pick a discriminator well
        // beyond the current count; bincode rejects unknown
        // discriminators cleanly.
        let bytes = [200u8, 0, 0, 0, 0];
        assert!(decode::<ControlMessage>(&bytes).is_err());
    }

    #[test]
    fn unknown_audio_packet_variant_fails_decode() {
        // Opus is discriminator 0 (the only defined variant).
        // Discriminator 1 is unknown.
        let bytes = [1u8, 0, 0, 0, 0];
        assert!(decode::<crate::audio::AudioPacket>(&bytes).is_err());
    }

    #[test]
    fn reassembler_drops_cross_epoch_fragments() {
        // ARCHITECTURE.md invariant: "stream_epoch bumped on encoder
        // restart; defragmenter drops cross-epoch fragments." Without
        // this, fragments from a pre-restart encoder state could fuse
        // with post-restart fragments and produce a corrupt frame.
        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (320, 240),
        };

        let mut reassembler = FrameReassembler::new();

        // Deliver a complete frame under epoch 0.
        let mut fragmenter_epoch0 = FrameFragmenter::new(0u8);
        let packets_e0 = fragmenter_epoch0.fragment(
            meta.clone(),
            bytes::Bytes::from_static(&[1u8; 200]),
            MAX_DATAGRAM_PAYLOAD,
        );
        let mut out0 = None;
        for p in packets_e0 {
            if let Some(f) = reassembler.handle(p) {
                out0 = Some(f);
            }
        }
        assert_eq!(
            out0.expect("epoch 0 frame should reassemble").stream_epoch,
            0
        );

        // Switch epochs (simulating encoder restart) and deliver an
        // independent frame. The two streams share `(display=0)` but
        // not `(display, epoch)`, so latest_seq for the old epoch
        // stays parked. A First arrives under the new epoch with the
        // same frame_seq=0 — must reassemble independently.
        let mut fragmenter_epoch1 = FrameFragmenter::new(0u8);
        fragmenter_epoch1.bump_epoch();
        assert_eq!(fragmenter_epoch1.stream_epoch(), 1);
        let packets_e1 = fragmenter_epoch1.fragment(
            meta,
            bytes::Bytes::from_static(&[2u8; 200]),
            MAX_DATAGRAM_PAYLOAD,
        );
        let mut out1 = None;
        for p in packets_e1 {
            if let Some(f) = reassembler.handle(p) {
                out1 = Some(f);
            }
        }
        let frame1 = out1.expect("epoch 1 frame should reassemble");
        assert_eq!(frame1.stream_epoch, 1);
        assert_eq!(
            frame1.body[0], 2,
            "epoch 1 body must not be fused with epoch 0"
        );
    }

    #[test]
    fn keyframe_sized_frame_roundtrips_through_unified_datagram_path() {
        // #36: IDRs no longer ride a separate reliable stream — a large
        // keyframe-sized body goes through the same fragment() + FEC datagram
        // path as every P-frame and reassembles to the exact body. Loss-free
        // delivery here is the happy path; FEC recovery is covered elsewhere.
        let mut fragmenter = FrameFragmenter::new_with_fec(3u8, 20);
        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: true,
            input_echo: InputEchoBatch::default(),
            dimensions: (1920, 1080),
        };
        let body: Vec<u8> = (0..50_000).map(|i| (i & 0xff) as u8).collect();
        let packets = fragmenter.fragment(
            meta.clone(),
            bytes::Bytes::from(body.clone()),
            MAX_DATAGRAM_PAYLOAD,
        );
        assert!(packets.len() > 1, "a 50 KB IDR must span many shards");

        // The seq counter advances so subsequent P-frames slot in sequentially.
        let next = fragmenter.fragment(
            meta,
            bytes::Bytes::from_static(&[0u8; 100]),
            MAX_DATAGRAM_PAYLOAD,
        );
        match &next[0] {
            VideoPacket::First { frame_seq, .. } => assert_eq!(*frame_seq, 1),
            other => panic!("expected First, got {other:?}"),
        }

        let mut reassembler = FrameReassembler::new();
        let mut got = None;
        for p in packets {
            if let Some(f) = reassembler.handle(p) {
                got = Some(f);
            }
        }
        let frame = got.expect("keyframe must reassemble from its datagrams");
        assert_eq!(frame.body.as_ref(), body.as_slice());
        assert_eq!(frame.stream_id, VideoStreamId(3));
        assert!(frame.meta.keyframe);
    }

    #[test]
    fn reassembler_handles_duplicate_fragments_idempotently() {
        // Reliable streams shouldn't duplicate, datagrams shouldn't
        // either, but quinn does retransmit at the QUIC layer and a
        // future protocol bug could deliver the same fragment twice.
        // Reassembler must not double-count or corrupt the frame.
        let mut fragmenter = FrameFragmenter::new(0u8);
        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (320, 240),
        };
        let body = vec![0xabu8; 2_500];
        let packets =
            fragmenter.fragment(meta, bytes::Bytes::from(body.clone()), MAX_DATAGRAM_PAYLOAD);
        assert!(packets.len() >= 2, "test needs a multi-fragment frame");

        let mut reassembler = FrameReassembler::new();
        // Deliver the first fragment twice.
        reassembler.handle(packets[0].clone());
        let after_dup = reassembler.handle(packets[0].clone());
        assert!(
            after_dup.is_none(),
            "duplicate First must not complete the frame on its own"
        );

        // Now deliver the rest; the frame should still complete with
        // the correct body, not a corrupted concatenation.
        let mut out = None;
        for p in packets.iter().skip(1) {
            if let Some(f) = reassembler.handle(p.clone()) {
                out = Some(f);
            }
        }
        let frame = out.expect("frame should reassemble after duplicates");
        assert_eq!(frame.body.as_ref(), body.as_slice());
    }

    #[test]
    fn reassembler_continuation_before_first_holds_then_completes() {
        // A Continuation can legitimately arrive before its First on
        // the wire (UDP reorder). Reassembler should buffer the
        // Continuation, then complete the frame when First lands.
        let mut fragmenter = FrameFragmenter::new(0u8);
        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (320, 240),
        };
        let body = vec![0x5au8; 3_000];
        let packets =
            fragmenter.fragment(meta, bytes::Bytes::from(body.clone()), MAX_DATAGRAM_PAYLOAD);
        assert!(packets.len() >= 2);

        let mut reassembler = FrameReassembler::new();
        // Deliver continuations first.
        for p in packets.iter().skip(1) {
            assert!(
                reassembler.handle(p.clone()).is_none(),
                "frame cannot complete without First (no meta)"
            );
        }
        // Now First.
        let frame = reassembler
            .handle(packets[0].clone())
            .expect("frame should complete after First arrives");
        assert_eq!(frame.body.as_ref(), body.as_slice());
    }
}
