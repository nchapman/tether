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
// a forged length prefix. Transport caps incoming datagrams at the network
// boundary, which mitigates this, but defense in depth says the decoder
// should also refuse oversize payloads. Wire bincode's Limit config or wrap
// here once the transport is in place and we can measure realistic sizes.
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
mod tests {
    use super::*;
    use crate::control::*;
    use crate::cursor::*;
    use crate::input::*;
    use crate::video::{
        FrameFragmenter, FrameReassembler, HostFrameTiming, HostFrameTimingBuilder,
        InputEchoBatch, VideoFrameMeta, VideoFrameMetaEnvelope, VideoPacket,
        CONTINUATION_PAYLOAD_BUDGET, FIRST_PAYLOAD_BUDGET,
    };

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
                id, hotspot, width, height, format, pixels,
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
        let read = ext
            .get(PIXEL_FORMAT_EXTENSION_KEY)
            .expect("present");
        let decoded: PixelFormat = decode(read).unwrap();
        assert_eq!(decoded, PixelFormat::Nv12);
        assert_eq!(PIXEL_FORMAT_EXTENSION_KEY, "tether.pixel-format");
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
            payload: vec![0xAA; 1100],
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
            VideoPacket::Continuation { .. } => panic!("wrong variant"),
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
            payload: vec![],
        };
        let bytes = encode(&p).unwrap();
        let p2: VideoPacket = decode(&bytes).unwrap();
        match p2 {
            VideoPacket::First { stream_epoch, .. } => assert_eq!(stream_epoch, epoch),
            VideoPacket::Continuation { .. } => panic!("wrong variant"),
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
            payload: vec![0; 1080],
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
    fn fragment_then_reassemble_round_trips() {
        let body: Vec<u8> = (0..10_000u32).map(|i| (i & 0xFF) as u8).collect();
        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: true,
            input_echo: InputEchoBatch::default(),
            dimensions: (320, 240),
        };

        let mut fragmenter = FrameFragmenter::new(0);
        let packets = fragmenter.fragment(meta.clone(), &body);

        // First fragment carries FIRST_PAYLOAD_BUDGET, remainder split into
        // CONTINUATION_PAYLOAD_BUDGET chunks.
        let expected_count = 1
            + body
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
        assert_eq!(frame.body, body);
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
        let mut packets = fragmenter.fragment(meta, &body);
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
        assert_eq!(frame.body, body);
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
            for p in fragmenter.fragment(meta.clone(), &[0u8; 100]) {
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
            payload: vec![0u8; 10],
        };
        assert!(reassembler.handle(stale).is_none());
    }

    #[test]
    fn reassembler_evicts_pending_past_wall_clock_timeout() {
        // Quiet-stream case: a frame goes incomplete and no newer
        // frames arrive to advance `latest_seq` past `max_age`. The
        // wall-clock timeout is the only thing standing between us
        // and a stuck pending entry that holds memory indefinitely.
        let mut reassembler = FrameReassembler::new()
            .with_max_pending_age(std::time::Duration::from_millis(20));

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
            payload: vec![0u8; 100],
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
            payload: vec![0u8; 10],
        };
        let _ = reassembler.handle(unrelated);
        let (dropped_after, _) = reassembler.loss_counters();
        assert_eq!(dropped_after, 1, "wall-clock timeout did not evict stuck pending frame");
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
            payload: vec![0; 1180],
        };
        let bytes = encode(&p).unwrap();
        assert!(
            bytes.len() <= MAX_DATAGRAM_PAYLOAD,
            "continuation-packet encoded size {} exceeds {}",
            bytes.len(),
            MAX_DATAGRAM_PAYLOAD
        );
    }
}
