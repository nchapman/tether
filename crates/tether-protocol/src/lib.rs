//! Wire protocol for Tether.
//!
//! Four logical channels, carried by `tether-transport`:
//! - **Control** (reliable, bidirectional) — handshake, clock sync, IDR requests.
//! - **Video datagrams** (unreliable) — fragmented encoded frames.
//! - **Cursor datagrams** (unreliable, high priority) — position + shape.
//! - **Input stream** (reliable, client→host) — keyboard + mouse events.

pub mod control;
pub mod cursor;
pub mod input;
pub mod video;

use serde::{Deserialize, Serialize};

/// Wire-format version. Bumped on any breaking change to the on-wire
/// representation of any message defined in this crate.
pub const PROTOCOL_VERSION: u32 = 1;

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
    let (value, _) = bincode::serde::decode_from_slice(bytes, bincode_config())?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::*;
    use crate::cursor::*;
    use crate::input::*;
    use crate::video::{
        FrameFragmenter, FrameReassembler, HostFrameTiming, InputEchoBatch, VideoFrameMeta,
        VideoPacket, CONTINUATION_PAYLOAD_BUDGET, FIRST_PAYLOAD_BUDGET,
    };

    #[test]
    fn mono_nanos_monotonic() {
        let a = MonoNanos::now();
        let b = MonoNanos::now();
        assert!(b >= a);
    }

    #[test]
    fn round_trip_client_hello() {
        let h = ClientHello {
            protocol_version: PROTOCOL_VERSION,
            client_name: "tether-client/0.0.1".into(),
            preferred_codecs: vec![CodecKind::H264, CodecKind::Hevc],
            max_resolution: Some((3840, 2160)),
            clock_probe_t0: MonoNanos(123_456_789),
        };
        let bytes = encode(&h).unwrap();
        let h2: ClientHello = decode(&bytes).unwrap();
        assert_eq!(h.protocol_version, h2.protocol_version);
        assert_eq!(h.client_name, h2.client_name);
        assert_eq!(h.preferred_codecs, h2.preferred_codecs);
        assert_eq!(h.max_resolution, h2.max_resolution);
        assert_eq!(h.clock_probe_t0, h2.clock_probe_t0);
    }

    #[test]
    fn round_trip_video_packet_first() {
        let p = VideoPacket::First {
            display: 0,
            stream_epoch: 0,
            frame_seq: 42,
            fragment_count: 3,
            meta: VideoFrameMeta {
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
            },
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
                assert!(meta.keyframe);
                assert_eq!(meta.input_echo.event_ids, vec![1, 2, 3]);
                assert_eq!(payload.len(), 1100);
            }
            VideoPacket::Continuation { .. } => panic!("wrong variant"),
        }
    }

    #[test]
    fn round_trip_cursor_position() {
        let c = CursorPacket::Position {
            t_capture: MonoNanos(999),
            x: 100,
            y: -50,
            visible: true,
        };
        let bytes = encode(&c).unwrap();
        let c2: CursorPacket = decode(&bytes).unwrap();
        match c2 {
            CursorPacket::Position {
                t_capture,
                x,
                y,
                visible,
            } => {
                assert_eq!(t_capture, MonoNanos(999));
                assert_eq!(x, 100);
                assert_eq!(y, -50);
                assert!(visible);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn round_trip_input_event() {
        let e = InputEvent {
            event_id: 12345,
            t_client: MonoNanos(54321),
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
            stream_epoch: u16::MAX,
            frame_seq: u32::MAX,
            fragment_count: u16::MAX,
            meta: VideoFrameMeta {
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
            },
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
    fn continuation_video_packet_fits_in_datagram() {
        // Even with max-valued numeric fields (worst case for varint
        // expansion), a continuation packet must fit in the datagram budget.
        // ~15 bytes of header overhead in the worst case for this variant.
        let p = VideoPacket::Continuation {
            display: u8::MAX,
            stream_epoch: u16::MAX,
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
