//! Wire protocol for Tether.
//!
//! Five logical channels, carried by `tether-transport`:
//! - **Control** (reliable, bidirectional) — handshake, clock sync,
//!   IDR requests, stream lifecycle, cursor shape, display topology,
//!   extension escape hatch.
//! - **Video datagrams** (unreliable) — fragmented encoded frames.
//! - **Audio datagrams** (unreliable) — Opus packets, host→client.
//!   Wire shape defined; pipeline (capture/encode/decode/output) is
//!   future work — see [`audio`].
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
}

// bincode (over postcard / rkyv / prost) because: we control both endpoints
// and ship them together, so schema evolution via protobuf-style tags isn't
// worth the verbosity; varint encoding keeps the per-frame header compact;
// and the serde adapter lets us keep #[derive(Serialize, Deserialize)] on
// every protocol type so the same definitions can flow into telemetry JSON
// or debug printing without a parallel set of derives.
fn bincode_config() -> impl bincode::config::Config {
    bincode::config::standard()
}

pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, CodecError> {
    Ok(bincode::serde::encode_to_vec(value, bincode_config())?)
}

// TODO(security): decode of untrusted input can allocate large Vec<u8> from
// a forged length prefix. Video datagrams are guarded by
// `VideoPacket::validate_packet_sizing` against MAX_FRAGMENTS_PER_FRAME /
// MAX_FRAME_BODY_BYTES before any allocation, so the video path is safe.
// What remains: `ControlMessage::CursorShape::pixels` and
// `ControlMessage::Extension::payload` decoded over the reliable control
// stream — both `Vec<u8>` with no length cap today. Wire bincode's Limit
// config or per-field caps once we measure realistic sizes.
pub fn decode<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, CodecError> {
    let (value, consumed) = bincode::serde::decode_from_slice(bytes, bincode_config())?;
    // Strict-decode: every framed message must be fully consumed by its
    // declared type. The forward-compat policy (see control.rs) says
    // appending fields to a body is forbidden — only new enum variants
    // are wire-additive. Catching trailing bytes here is what enforces
    // that promise: a buggy future encoder that ignored the rule would
    // trip this error in old receivers rather than having its extra
    // bytes silently misattributed.
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
        FrameFragmenter, FrameReassembler, HostFrameTiming, HostFrameTimingBuilder, InputEchoBatch,
        VideoFrameMeta, VideoFrameMetaEnvelope, VideoPacket, CONTINUATION_PAYLOAD_BUDGET,
        FEC_MAX_PRIMARY_SHARDS, FEC_SHARD_SIZE, FIRST_PAYLOAD_BUDGET,
    };

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
        let body = ClientHelloV1 {
            client_name: "tether-client/0.0.1".into(),
            preferred_codecs: vec![CodecKind::H264, CodecKind::Hevc],
            max_resolution: Some((3840, 2160)),
            viewport: Some(crate::control::Viewport::new(1280, 720)),
            clock_probe_t0: MonoNanos(123_456_789),
            extensions: Default::default(),
            resume_token: None,
        };
        let h = ClientHello::V1(body.clone());
        let bytes = encode(&h).unwrap();
        let h2: ClientHello = decode(&bytes).unwrap();
        let ClientHello::V1(body2) = h2;
        assert_eq!(body.client_name, body2.client_name);
        assert_eq!(body.preferred_codecs, body2.preferred_codecs);
        assert_eq!(body.max_resolution, body2.max_resolution);
        assert_eq!(body.viewport, body2.viewport);
        assert_eq!(body.clock_probe_t0, body2.clock_probe_t0);
        assert!(body2.extensions.is_empty());
        assert!(body2.resume_token.is_none());
    }

    #[test]
    fn client_hello_extensions_round_trip() {
        // Extensions populated round-trip identically. This is the
        // forward-compat probe: a future feature opt-in lands here.
        let mut extensions = std::collections::BTreeMap::new();
        extensions.insert("av1-preferred".to_string(), vec![1u8]);
        extensions.insert("adaptive-bitrate-hint".to_string(), vec![0u8, 0, 64, 0]);
        let body = ClientHelloV1 {
            client_name: "x".into(),
            preferred_codecs: vec![CodecKind::H264],
            max_resolution: None,
            viewport: None,
            clock_probe_t0: MonoNanos(1),
            extensions: extensions.clone(),
            resume_token: Some(vec![0xde, 0xad, 0xbe, 0xef]),
        };
        let bytes = encode(&ClientHello::V1(body)).unwrap();
        let ClientHello::V1(body2) = decode::<ClientHello>(&bytes).unwrap();
        assert_eq!(body2.extensions, extensions);
        assert_eq!(body2.resume_token, Some(vec![0xde, 0xad, 0xbe, 0xef]));
    }

    #[test]
    fn round_trip_server_hello_hevc() {
        // Codec negotiation lands HEVC: client advertised [Hevc, H264],
        // host probed and picked Hevc, echoes back in chosen_codec.
        // Round-tripping confirms the host's selection survives the wire.
        use crate::control::{ChromaSubsampling, ServerHello, ServerHelloV1, VideoColorSpec};
        let body = ServerHelloV1 {
            server_name: "tether-host".into(),
            chosen_codec: CodecKind::Hevc,
            chosen_chroma: ChromaSubsampling::Yuv420,
            color_space: VideoColorSpec::sdr_desktop(),
            resolution: (1920, 1080),
            clock_probe_t0_echo: MonoNanos(42),
            t1_server_recv: MonoNanos(43),
            t2_server_send: MonoNanos(44),
            extensions: Default::default(),
            resume_token: None,
        };
        let h = ServerHello::V1(body.clone());
        let bytes = encode(&h).unwrap();
        let h2: ServerHello = decode(&bytes).unwrap();
        let ServerHello::V1(body2) = h2;
        assert_eq!(body2.chosen_codec, CodecKind::Hevc);
        assert_eq!(body2.server_name, body.server_name);
        assert_eq!(body2.resolution, body.resolution);
    }

    #[test]
    fn trailing_bytes_fail_decode() {
        // Strict-decode is the forcing function for the "no appended
        // fields" forward-compat policy: a future encoder that appends
        // a field to ClientHelloV1 (forbidden — see control.rs module
        // doc) would produce a wire payload that decodes one extra
        // byte past where this build's schema thinks the message ends.
        // We surface that as a decode error rather than silently
        // truncating.
        let body = ClientHelloV1 {
            client_name: "x".into(),
            preferred_codecs: vec![CodecKind::H264],
            max_resolution: None,
            viewport: None,
            clock_probe_t0: MonoNanos(0),
            extensions: Default::default(),
            resume_token: None,
        };
        let mut bytes = encode(&ClientHello::V1(body)).unwrap();
        bytes.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        let result = decode::<ClientHello>(&bytes);
        assert!(
            result.is_err(),
            "trailing bytes after a valid message must fail decode"
        );
    }

    #[test]
    fn round_trip_set_client_viewport() {
        use crate::control::{ControlMessage, Viewport};
        let msg = ControlMessage::SetClientViewport(Viewport::new(1280, 800));
        let bytes = encode(&msg).unwrap();
        let msg2: ControlMessage = decode(&bytes).unwrap();
        match msg2 {
            ControlMessage::SetClientViewport(v) => {
                assert_eq!(v.width, 1280);
                assert_eq!(v.height, 800);
            }
            other => panic!("expected SetClientViewport, got {other:?}"),
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
    fn unknown_client_hello_variant_fails_decode() {
        // Hand-craft bytes for a hypothetical V2: discriminator byte
        // claiming variant 1 (V1 is 0). This pins the forward-compat
        // story: an older receiver decoding a newer variant must error
        // cleanly, not silently misinterpret the body bytes.
        let bytes = [1u8, 0, 0, 0, 0];
        let result = decode::<ClientHello>(&bytes);
        assert!(
            result.is_err(),
            "unknown ClientHello variant must fail decode, not silently succeed"
        );
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
        let bytes = encode(&msg).unwrap();
        let msg2: ControlMessage = decode(&bytes).unwrap();
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
        let bytes = encode(&msg).unwrap();
        let msg2: ControlMessage = decode(&bytes).unwrap();
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
        let displays = vec![
            DisplayDescriptor {
                id: 0,
                name: "DP-1".into(),
                width: 3840,
                height: 2160,
                refresh_mhz: 60_000,
                scale_num: 2,
                scale_den: 1,
                primary: true,
                position: (0, 0),
            },
            DisplayDescriptor {
                id: 1,
                name: "HDMI-A-2".into(),
                width: 1920,
                height: 1080,
                refresh_mhz: 59_940,
                scale_num: 1,
                scale_den: 1,
                primary: false,
                position: (3840, 0),
            },
        ];
        let msg = ControlMessage::DisplayList {
            displays: displays.clone(),
        };
        let bytes = encode(&msg).unwrap();
        let msg2: ControlMessage = decode(&bytes).unwrap();
        match msg2 {
            ControlMessage::DisplayList { displays: d2 } => assert_eq!(d2, displays),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn round_trip_set_active_displays() {
        let msg = ControlMessage::SetActiveDisplays {
            displays: vec![0, 2, 5],
        };
        let bytes = encode(&msg).unwrap();
        let msg2: ControlMessage = decode(&bytes).unwrap();
        match msg2 {
            ControlMessage::SetActiveDisplays { displays } => {
                assert_eq!(displays, vec![0, 2, 5]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn pixel_format_extension_round_trips() {
        use crate::control::{PixelFormat, PIXEL_FORMAT_EXTENSION_KEY};
        let pf = PixelFormat::Nv12;
        let bytes = encode(&pf).unwrap();
        let pf2: PixelFormat = decode(&bytes).unwrap();
        assert_eq!(pf, pf2);

        // Round-trip via a hello extension map to confirm the integration
        // shape: server encodes into BTreeMap value, client decodes back.
        let mut ext = std::collections::BTreeMap::<String, Vec<u8>>::new();
        ext.insert(PIXEL_FORMAT_EXTENSION_KEY.to_string(), bytes);
        let read = ext.get(PIXEL_FORMAT_EXTENSION_KEY).expect("present");
        let decoded: PixelFormat = decode(read).unwrap();
        assert_eq!(decoded, PixelFormat::Nv12);
        assert_eq!(PIXEL_FORMAT_EXTENSION_KEY, "tether.pixel-format");
    }

    #[test]
    fn video_profile_round_trips_via_extension() {
        use crate::control::{
            VideoProfile, CLIENT_DECODE_PROFILES_EXTENSION_KEY, SERVER_ENCODE_PROFILE_EXTENSION_KEY,
        };
        // Client packs its full decode set into one extension value.
        let client_caps = vec![
            VideoProfile::HEVC_8BIT_444,
            VideoProfile::HEVC_8BIT_420,
            VideoProfile::H264_8BIT_420,
        ];
        let payload = encode(&client_caps).unwrap();
        let decoded: Vec<VideoProfile> = decode(&payload).unwrap();
        assert_eq!(decoded, client_caps);

        // Host echoes a single chosen profile.
        let chosen = VideoProfile::HEVC_8BIT_444;
        let payload = encode(&chosen).unwrap();
        let decoded: VideoProfile = decode(&payload).unwrap();
        assert_eq!(decoded, chosen);

        // Key strings are part of the wire contract — pin them so a
        // typo in a future refactor breaks the test, not the network.
        assert_eq!(
            CLIENT_DECODE_PROFILES_EXTENSION_KEY,
            "tether.cap.video.decode-profiles"
        );
        assert_eq!(
            SERVER_ENCODE_PROFILE_EXTENSION_KEY,
            "tether.cap.video.encode-profile"
        );
    }

    /// Pin the 10-bit profile constants on the wire: the bincode shape
    /// for `VideoProfile { codec, chroma, bit_depth }` must round-trip
    /// 10 cleanly so a session that negotiates HEVC Main10 / Main 4:4:4
    /// 10-bit doesn't get the bit_depth byte silently corrupted. Pinned
    /// alongside the 8-bit cases (no shared body) so the bit_depth axis
    /// gets independent coverage from the codec+chroma axes — a
    /// future refactor that drops or reorders the field is caught here.
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

    /// Pin the AV1 profile constants on the wire: the bincode shape
    /// must round-trip the new `CodecKind::Av1` variant at both 8- and
    /// 10-bit depths cleanly. AV1 is a brand-new codec axis (no
    /// `CodecKind::Av1` discriminant existed in shipped builds before
    /// this), so the bincode enum tag is what older peers will treat
    /// as "unknown" — the round-trip here is the contract that current
    /// peers won't corrupt the new variant.
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
        use crate::audio::{AudioConfig, AudioPacket, AUDIO_CONFIG_EXTENSION_KEY};
        let p = AudioPacket::Opus {
            stream_epoch: 1,
            frame_seq: 1234,
            t_capture: MonoNanos(98765),
            payload: vec![0xAB; 64],
        };
        let bytes = encode(&p).unwrap();
        let p2: AudioPacket = decode(&bytes).unwrap();
        assert_eq!(p, p2);

        // Hello-extension config round-trips identically too.
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
        assert_eq!(AUDIO_CONFIG_EXTENSION_KEY, "tether.audio");
    }

    #[test]
    fn round_trip_client_stats() {
        let msg = ControlMessage::ClientStats {
            interval_ms: 1000,
            frames_received: 60,
            frames_dropped: 2,
            fragments_lost: 4,
            rtt_ewma_us: 9_500,
        };
        let bytes = encode(&msg).unwrap();
        let msg2: ControlMessage = decode(&bytes).unwrap();
        assert_eq!(msg, msg2);
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
            ControlMessage::StreamPause { display: 3 },
            ControlMessage::StreamResume { display: 3 },
        ] {
            let bytes = encode(&msg).unwrap();
            let msg2: ControlMessage = decode(&bytes).unwrap();
            assert_eq!(msg, msg2);
        }
    }

    #[test]
    fn round_trip_control_extension() {
        // The Extension escape unblocks future control features
        // without forcing a ClientHelloV2. Confirm it survives the
        // wire identically.
        let msg = ControlMessage::Extension {
            key: "tether.cap.test".into(),
            payload: vec![1, 2, 3, 0xFF],
        };
        let bytes = encode(&msg).unwrap();
        let msg2: ControlMessage = decode(&bytes).unwrap();
        match msg2 {
            ControlMessage::Extension { key, payload } => {
                assert_eq!(key, "tether.cap.test");
                assert_eq!(payload, vec![1, 2, 3, 0xFF]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn goodbye_carries_machine_readable_code() {
        let g = ControlMessage::Goodbye {
            reason: "user quit".into(),
            code: GoodbyeCode::Clean,
        };
        let bytes = encode(&g).unwrap();
        let g2: ControlMessage = decode(&bytes).unwrap();
        match g2 {
            ControlMessage::Goodbye { reason, code } => {
                assert_eq!(reason, "user quit");
                assert_eq!(code, GoodbyeCode::Clean);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn round_trip_video_packet_first() {
        let p = VideoPacket::First {
            display: 0,
            stream_epoch: 0,
            frame_seq: 42,
            fragment_count: 3,
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
                display,
                stream_epoch,
                frame_seq,
                fragment_count,
                meta,
                payload,
            } => {
                assert_eq!(display, 0);
                assert_eq!(stream_epoch, 0);
                assert_eq!(frame_seq, 42);
                assert_eq!(fragment_count, 3);
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
        // Every field on VideoPacket::Parity must round-trip: the
        // shard math depends on data_shards, parity_shards, and
        // shard_index agreeing wire-to-receiver, and total_body_len
        // sizes the reassembler's BytesMut. Bincode positional
        // encoding makes any field reordering a silent data
        // corruption — assert each value individually.
        let p = VideoPacket::Parity {
            display: 3,
            stream_epoch: 42,
            frame_seq: 1001,
            data_shards: 8,
            parity_shards: 2,
            shard_index: 1,
            total_body_len: 9000,
            meta: VideoFrameMetaEnvelope::V1(default_meta()),
            payload: bytes::Bytes::from(vec![0xee; 128]),
        };
        let bytes = encode(&p).unwrap();
        let p2: VideoPacket = decode(&bytes).unwrap();
        match p2 {
            VideoPacket::Parity {
                display,
                stream_epoch,
                frame_seq,
                data_shards,
                parity_shards,
                shard_index,
                total_body_len,
                meta,
                payload,
            } => {
                assert_eq!(display, 3);
                assert_eq!(stream_epoch, 42);
                assert_eq!(frame_seq, 1001);
                assert_eq!(data_shards, 8);
                assert_eq!(parity_shards, 2);
                assert_eq!(shard_index, 1);
                assert_eq!(total_body_len, 9000);
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
            last_known_good_frame_id: 12345,
        };
        let bytes = encode(&m).unwrap();
        let m2: ControlMessage = decode(&bytes).unwrap();
        match m2 {
            ControlMessage::RequestRecovery {
                last_known_good_frame_id,
            } => assert_eq!(last_known_good_frame_id, 12345),
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
            display: 0,
            stream_epoch: epoch,
            frame_seq: 0,
            fragment_count: 1,
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
            display: u8::MAX,
            stream_epoch: u32::MAX,
            frame_seq: u32::MAX,
            fragment_count: u16::MAX,
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
            payload: bytes::Bytes::from(vec![0u8; 1080]),
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
        // Default fec_percentage=0 means no Parity packets emitted
        // and primary fragmentation stays in the original
        // 1100/1180 mixed-budget shape — i.e. wire-compatible with
        // pre-FEC reassemblers.
        let body: bytes::Bytes = vec![0xab; 5000].into();
        let mut a = FrameFragmenter::new(0);
        let mut b = FrameFragmenter::new_with_fec(0, 0);
        let pa = a.fragment(default_meta(), body.clone());
        let pb = b.fragment(default_meta(), body);
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
        let body: bytes::Bytes = vec![0u8; 11_000].into(); // ~10 primary shards
        let mut frag = FrameFragmenter::new_with_fec(0, 20);
        let pkts = frag.fragment(default_meta(), body);
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
        // 11_000 / 1100 = 10 primaries; 20% parity = 2.
        assert_eq!(primary, 10);
        assert_eq!(parity, 2);
    }

    #[test]
    fn fec_recovers_from_losing_up_to_parity_count_primaries() {
        // Drop up to K of the N primaries; reassembler must
        // reconstruct from the K parity shards.
        let body: bytes::Bytes = (0..5000u32)
            .map(|i| (i & 0xff) as u8)
            .collect::<Vec<u8>>()
            .into();
        let mut frag = FrameFragmenter::new_with_fec(0, 25); // 25% parity
        let pkts = frag.fragment(default_meta(), body.clone());
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
        let mut got = None;
        for p in kept {
            if let Some(f) = r.handle(p) {
                got = Some(f);
            }
        }
        let f = got.expect("reassembled via FEC");
        assert_eq!(f.body.as_ref(), body.as_ref());
    }

    #[test]
    fn fec_fails_when_loss_exceeds_parity() {
        // Losing more primaries than we have parity for must NOT
        // produce a reconstructed frame.
        let body: bytes::Bytes = vec![0xff; 5000].into();
        let mut frag = FrameFragmenter::new_with_fec(0, 20); // 1 parity for 5 primaries
        let pkts = frag.fragment(default_meta(), body);
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
    }

    #[test]
    fn fec_falls_back_to_legacy_when_frame_exceeds_max_primary_shards() {
        // A frame above FEC_MAX_PRIMARY_SHARDS × FEC_SHARD_SIZE
        // can't fit in a single FEC block — fragmenter falls back
        // to the pre-FEC mixed-budget shape with zero parity. Test
        // that fec stays a no-op for this case (no Parity packets)
        // and that the fallback counter bumps so the host can
        // surface this in stats.
        let oversize = (FEC_MAX_PRIMARY_SHARDS + 5) * FEC_SHARD_SIZE;
        let body: bytes::Bytes = vec![0u8; oversize].into();
        let mut frag = FrameFragmenter::new_with_fec(0, 20);
        assert_eq!(frag.no_fec_fallback_count(), 0);
        let pkts = frag.fragment(default_meta(), body);
        assert!(
            pkts.iter()
                .all(|p| !matches!(p, VideoPacket::Parity { .. })),
            "oversize frames must fall back to no-FEC fragmentation"
        );
        assert_eq!(
            frag.no_fec_fallback_count(),
            1,
            "oversize FEC frame must bump the fallback counter"
        );
    }

    #[test]
    fn fec_falls_back_when_primary_count_exceeds_per_pct_ceiling() {
        // At fec_percentage = 25, the GF(2^8) ceiling on total
        // shards gives (255 * 100) / 125 = 204 primaries max — well
        // under FEC_MAX_PRIMARY_SHARDS (212). A frame that needs
        // 205-212 primaries used to silently fall through to
        // ReedSolomon::new with total=257 > 255 and emit a misleading
        // "construction failed" warning; the fix routes it cleanly
        // to the no-FEC legacy path.
        use crate::video::max_primary_shards_for_pct;
        let ceiling_25 = max_primary_shards_for_pct(25);
        assert!(
            ceiling_25 < FEC_MAX_PRIMARY_SHARDS,
            "25% parity should narrow the per-pct ceiling below the hard cap"
        );
        // One shard above the per-pct ceiling but inside the hard
        // cap exercises the new gating without exceeding it.
        let primaries = ceiling_25 + 1;
        let body: bytes::Bytes = vec![0u8; primaries * FEC_SHARD_SIZE].into();
        let mut frag = FrameFragmenter::new_with_fec(0, 25);
        let pkts = frag.fragment(default_meta(), body);
        assert!(
            pkts.iter()
                .all(|p| !matches!(p, VideoPacket::Parity { .. })),
            "frame above per-pct ceiling must fall back to no-FEC"
        );
        assert_eq!(frag.no_fec_fallback_count(), 1);
    }

    #[test]
    fn fec_at_25_pct_inside_ceiling_still_emits_parity() {
        // Sanity check the other direction of the per-pct guard:
        // at fec_percentage = 25 with a frame at the ceiling, FEC
        // must still engage. Catches regressions that swap the
        // comparison.
        use crate::video::max_primary_shards_for_pct;
        let primaries = max_primary_shards_for_pct(25);
        let body: bytes::Bytes = vec![0u8; primaries * FEC_SHARD_SIZE].into();
        let mut frag = FrameFragmenter::new_with_fec(0, 25);
        let pkts = frag.fragment(default_meta(), body);
        assert!(
            pkts.iter().any(|p| matches!(p, VideoPacket::Parity { .. })),
            "frame exactly at the per-pct ceiling must still emit parity"
        );
        assert_eq!(frag.no_fec_fallback_count(), 0);
    }

    #[test]
    fn parity_packet_round_trips() {
        let p = VideoPacket::Parity {
            display: 7,
            stream_epoch: 42,
            frame_seq: 100,
            data_shards: 5,
            parity_shards: 1,
            shard_index: 0,
            total_body_len: 5000,
            meta: VideoFrameMetaEnvelope::V1(default_meta()),
            payload: bytes::Bytes::from(vec![0x42; FEC_SHARD_SIZE]),
        };
        let bytes = encode(&p).unwrap();
        assert!(
            bytes.len() <= MAX_DATAGRAM_PAYLOAD,
            "parity packet must fit in a datagram"
        );
        let p2: VideoPacket = decode(&bytes).unwrap();
        match p2 {
            VideoPacket::Parity {
                display,
                stream_epoch,
                frame_seq,
                data_shards,
                parity_shards,
                shard_index,
                total_body_len,
                meta,
                payload,
            } => {
                assert_eq!(display, 7);
                assert_eq!(stream_epoch, 42);
                assert_eq!(frame_seq, 100);
                assert_eq!(data_shards, 5);
                assert_eq!(parity_shards, 1);
                assert_eq!(shard_index, 0);
                assert_eq!(total_body_len, 5000);
                let m = meta.into_meta();
                assert_eq!(m.dimensions, (640, 480));
                assert_eq!(payload.len(), FEC_SHARD_SIZE);
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
        let mut frag = FrameFragmenter::new(0);
        let packets = frag.fragment(meta.clone(), body.clone());
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
        let mut frag_fec = FrameFragmenter::new_with_fec(0, 20);
        let packets_fec = frag_fec.fragment(meta, body);
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

        let mut fragmenter = FrameFragmenter::new(0);
        let packets = fragmenter.fragment(meta.clone(), bytes::Bytes::from(body.clone()));

        // First fragment carries FIRST_PAYLOAD_BUDGET, remainder split into
        // CONTINUATION_PAYLOAD_BUDGET chunks.
        let expected_count = 1 + body
            .len()
            .saturating_sub(FIRST_PAYLOAD_BUDGET)
            .div_ceil(CONTINUATION_PAYLOAD_BUDGET);
        assert_eq!(packets.len(), expected_count);

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
        assert_eq!(frame.display, 0);
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
        let mut fragmenter = FrameFragmenter::new(2);
        let mut packets = fragmenter.fragment(meta, bytes::Bytes::from(body.clone()));
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
        assert_eq!(frame.display, 2);
    }

    #[test]
    fn reassembler_drops_stale_fragments() {
        let mut fragmenter = FrameFragmenter::new(0);
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
            for p in fragmenter.fragment(meta.clone(), bytes::Bytes::from_static(&[0u8; 100])) {
                reassembler.handle(p);
            }
        }

        // Now inject a stale Continuation claiming to belong to seq 0 —
        // 5 frames behind latest, max_age=1, so the reassembler should
        // drop it silently.
        let stale = VideoPacket::Continuation {
            display: 0,
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
            display: 0,
            stream_epoch: 0,
            frame_seq: 0,
            fragment_count: 2,
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
            display: 0,
            stream_epoch: 0,
            frame_seq: 1,
            fragment_count: 1,
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
            display: u8::MAX,
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
        let mut fragmenter_epoch0 = FrameFragmenter::new(0);
        let packets_e0 =
            fragmenter_epoch0.fragment(meta.clone(), bytes::Bytes::from_static(&[1u8; 200]));
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
        let mut fragmenter_epoch1 = FrameFragmenter::new(0);
        fragmenter_epoch1.bump_epoch();
        assert_eq!(fragmenter_epoch1.stream_epoch(), 1);
        let packets_e1 = fragmenter_epoch1.fragment(meta, bytes::Bytes::from_static(&[2u8; 200]));
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
    fn fragmenter_single_packet_roundtrips_through_reassembler() {
        // The reliable-keyframe path uses single_packet instead of
        // fragment() — produces one VideoPacket::First with
        // fragment_count=1 carrying the whole body. Receiver must
        // accept this directly and complete the frame on the first
        // (and only) fragment.
        let mut fragmenter = FrameFragmenter::new(3);
        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: true,
            input_echo: InputEchoBatch::default(),
            dimensions: (1920, 1080),
        };
        let body: Vec<u8> = (0..50_000).map(|i| (i & 0xff) as u8).collect();
        let packet = fragmenter.single_packet(meta.clone(), bytes::Bytes::from(body.clone()));
        match &packet {
            VideoPacket::First {
                fragment_count,
                payload,
                ..
            } => {
                assert_eq!(
                    *fragment_count, 1,
                    "single_packet must produce fragment_count=1"
                );
                assert_eq!(payload.len(), body.len());
            }
            other => panic!("expected First, got {other:?}"),
        }

        // Also advances the seq counter so subsequent P-frames slot in.
        let next = fragmenter.fragment(meta, bytes::Bytes::from_static(&[0u8; 100]));
        match &next[0] {
            VideoPacket::First { frame_seq, .. } => {
                assert_eq!(
                    *frame_seq, 1,
                    "next frame_seq must be 1 after single_packet"
                );
            }
            other => panic!("expected First, got {other:?}"),
        }

        let mut reassembler = FrameReassembler::new();
        let frame = reassembler
            .handle(packet)
            .expect("single_packet must complete reassembly on first fragment");
        assert_eq!(frame.body.as_ref(), body.as_slice());
        assert_eq!(frame.display, 3);
    }

    #[test]
    fn reassembler_handles_duplicate_fragments_idempotently() {
        // Reliable streams shouldn't duplicate, datagrams shouldn't
        // either, but quinn does retransmit at the QUIC layer and a
        // future protocol bug could deliver the same fragment twice.
        // Reassembler must not double-count or corrupt the frame.
        let mut fragmenter = FrameFragmenter::new(0);
        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (320, 240),
        };
        let body = vec![0xabu8; 2_500];
        let packets = fragmenter.fragment(meta, bytes::Bytes::from(body.clone()));
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
        let mut fragmenter = FrameFragmenter::new(0);
        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (320, 240),
        };
        let body = vec![0x5au8; 3_000];
        let packets = fragmenter.fragment(meta, bytes::Bytes::from(body.clone()));
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
