//! Video datagram format + fragmentation / reassembly helpers.

use std::collections::HashMap;

use crate::MonoNanos;
use serde::{Deserialize, Serialize};

/// Timing fields populated by the host as a frame moves through its pipeline.
/// All times in **host-local** [`MonoNanos`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostFrameTiming {
    pub t_capture_kernel: MonoNanos,
    pub t_capture_userspace: MonoNanos,
    pub t_encode_submit: MonoNanos,
    pub t_encode_done: MonoNanos,
    pub t_send: MonoNanos,
}

/// Echo of input events the host injected since the previous frame. Lets the
/// client compute true motion-to-photon latency (not just network RTT).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputEchoBatch {
    pub event_ids: Vec<u64>,
}

/// Per-frame metadata. Sent in fragment 0 of a video frame, not repeated
/// across continuation packets.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoFrameMeta {
    pub timing: HostFrameTiming,
    pub keyframe: bool,
    pub input_echo: InputEchoBatch,
    /// Frame width × height in pixels. Always populated so a raw or
    /// keyframe-only consumer can size its render target without parsing
    /// codec-specific SPS / sequence headers. Costs ~10 bytes per frame
    /// header (varint-encoded u32 pair).
    pub dimensions: (u32, u32),
}

/// A single video datagram. Frames larger than the transport's max datagram
/// size are sliced into multiple packets sharing
/// `(display, stream_epoch, frame_seq)`.
///
/// `display` identifies which host display the frame came from. v0 is
/// single-monitor and always uses `display = 0`, but the field is present
/// so multi-monitor support can land later as a pure additive change.
/// Each display gets its own encoder thread and therefore its own
/// `stream_epoch` + `frame_seq` counters (cribbed from RustDesk's
/// `video_threads: HashMap<usize, _>` pattern).
///
/// `stream_epoch` is `u16` (varint-encoded as 1 byte for typical values) so
/// a long-lived session that restarts the encoder thousands of times can't
/// collide with prior epochs. The host bumps `stream_epoch` whenever the
/// encoder is restarted (resize, codec switch, HW context loss). Clients
/// drop all packets from prior epochs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoPacket {
    First {
        display: u8,
        stream_epoch: u16,
        frame_seq: u32,
        fragment_count: u16,
        meta: VideoFrameMeta,
        payload: Vec<u8>,
    },
    Continuation {
        display: u8,
        stream_epoch: u16,
        frame_seq: u32,
        fragment_index: u16,
        payload: Vec<u8>,
    },
}

/// Conservative payload budget per [`VideoPacket::First`] packet. Leaves
/// ~100 bytes of header headroom inside [`crate::MAX_DATAGRAM_PAYLOAD`].
pub const FIRST_PAYLOAD_BUDGET: usize = 1100;

/// Conservative payload budget per [`VideoPacket::Continuation`] packet.
/// Leaves ~20 bytes of header headroom.
pub const CONTINUATION_PAYLOAD_BUDGET: usize = 1180;

/// Splits a video frame body into a sequence of `VideoPacket`s sized to
/// fit inside the QUIC datagram budget. Owns the per-stream `frame_seq`
/// counter; bump `stream_epoch` via [`Self::bump_epoch`] whenever the
/// underlying encoder is restarted.
pub struct FrameFragmenter {
    display: u8,
    stream_epoch: u16,
    next_frame_seq: u32,
}

impl FrameFragmenter {
    pub fn new(display: u8) -> Self {
        Self {
            display,
            stream_epoch: 0,
            next_frame_seq: 0,
        }
    }

    pub fn display(&self) -> u8 {
        self.display
    }

    pub fn stream_epoch(&self) -> u16 {
        self.stream_epoch
    }

    pub fn bump_epoch(&mut self) {
        self.stream_epoch = self.stream_epoch.wrapping_add(1);
        self.next_frame_seq = 0;
    }

    /// Fragment a frame body into one or more packets. `meta` rides in
    /// fragment 0 only. An empty body still produces a single
    /// [`VideoPacket::First`] with `fragment_count = 1`.
    pub fn fragment(&mut self, meta: VideoFrameMeta, body: &[u8]) -> Vec<VideoPacket> {
        let frame_seq = self.next_frame_seq;
        self.next_frame_seq = self.next_frame_seq.wrapping_add(1);

        let first_len = body.len().min(FIRST_PAYLOAD_BUDGET);
        let tail_len = body.len() - first_len;
        let cont_count = tail_len.div_ceil(CONTINUATION_PAYLOAD_BUDGET);
        let fragment_count = u16::try_from(1 + cont_count)
            .expect("frame exceeds u16::MAX fragments");

        let mut packets = Vec::with_capacity(1 + cont_count);
        packets.push(VideoPacket::First {
            display: self.display,
            stream_epoch: self.stream_epoch,
            frame_seq,
            fragment_count,
            meta,
            payload: body[..first_len].to_vec(),
        });

        let mut offset = first_len;
        let mut idx: u16 = 1;
        while offset < body.len() {
            let end = (offset + CONTINUATION_PAYLOAD_BUDGET).min(body.len());
            packets.push(VideoPacket::Continuation {
                display: self.display,
                stream_epoch: self.stream_epoch,
                frame_seq,
                fragment_index: idx,
                payload: body[offset..end].to_vec(),
            });
            offset = end;
            idx += 1;
        }

        packets
    }
}

/// Reassembled frame produced by [`FrameReassembler::handle`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReassembledFrame {
    pub display: u8,
    pub stream_epoch: u16,
    pub frame_seq: u32,
    pub meta: VideoFrameMeta,
    pub body: Vec<u8>,
}

/// Buffers in-flight fragments by `(display, stream_epoch, frame_seq)`
/// and emits a [`ReassembledFrame`] when all fragments for a key have
/// arrived. Drops fragments belonging to frames more than `max_age`
/// frames behind the latest seen on that stream, so a permanent loss
/// can't leak memory.
pub struct FrameReassembler {
    pending: HashMap<FrameKey, Pending>,
    latest_seq: HashMap<StreamKey, u32>,
    max_age: u32,
}

type FrameKey = (u8, u16, u32);
type StreamKey = (u8, u16);

struct Pending {
    fragment_count: u16,
    received_count: u16,
    fragments: Vec<Option<Vec<u8>>>,
    meta: Option<VideoFrameMeta>,
}

impl Default for FrameReassembler {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameReassembler {
    pub fn new() -> Self {
        Self {
            pending: HashMap::new(),
            latest_seq: HashMap::new(),
            max_age: 4,
        }
    }

    /// Configure how many frames behind the latest a fragment can arrive
    /// before being dropped. Default is 4.
    pub fn with_max_age(mut self, max_age: u32) -> Self {
        self.max_age = max_age;
        self
    }

    pub fn handle(&mut self, packet: VideoPacket) -> Option<ReassembledFrame> {
        let (display, stream_epoch, frame_seq) = match &packet {
            VideoPacket::First {
                display,
                stream_epoch,
                frame_seq,
                ..
            }
            | VideoPacket::Continuation {
                display,
                stream_epoch,
                frame_seq,
                ..
            } => (*display, *stream_epoch, *frame_seq),
        };

        let stream_key = (display, stream_epoch);
        let latest = *self
            .latest_seq
            .entry(stream_key)
            .and_modify(|s| *s = (*s).max(frame_seq))
            .or_insert(frame_seq);

        if latest.saturating_sub(frame_seq) > self.max_age {
            // tracing macros shadow `display` with `tracing::field::display`
            // inside the macro body, so rebind to a non-colliding name first.
            let display_id = display;
            tracing::trace!(
                "dropping stale fragment: display={} epoch={} seq={} latest={}",
                display_id,
                stream_epoch,
                frame_seq,
                latest
            );
            return None;
        }

        if latest == frame_seq {
            self.prune_old();
        }

        let key = (display, stream_epoch, frame_seq);
        let entry = self.pending.entry(key).or_insert_with(|| Pending {
            fragment_count: 0,
            received_count: 0,
            fragments: Vec::new(),
            meta: None,
        });

        match packet {
            VideoPacket::First {
                fragment_count,
                meta,
                payload,
                ..
            } => {
                ensure_capacity(&mut entry.fragments, fragment_count as usize);
                if entry.fragment_count == 0 {
                    entry.fragment_count = fragment_count;
                }
                if entry.fragments[0].is_none() {
                    entry.fragments[0] = Some(payload);
                    entry.received_count += 1;
                }
                entry.meta = Some(meta);
            }
            VideoPacket::Continuation {
                fragment_index,
                payload,
                ..
            } => {
                let idx = fragment_index as usize;
                ensure_capacity(&mut entry.fragments, idx + 1);
                if entry.fragments[idx].is_none() {
                    entry.fragments[idx] = Some(payload);
                    entry.received_count += 1;
                }
            }
        }

        // Complete if we know the total and have received that many,
        // and the First (with meta) has arrived.
        if entry.fragment_count > 0
            && entry.received_count == entry.fragment_count
            && entry.meta.is_some()
        {
            let pending = self
                .pending
                .remove(&key)
                .expect("entry exists, we just inserted into it");
            let meta = pending.meta.expect("checked Some above");
            let body: Vec<u8> = pending
                .fragments
                .into_iter()
                .flatten()
                .flatten()
                .collect();
            return Some(ReassembledFrame {
                display,
                stream_epoch,
                frame_seq,
                meta,
                body,
            });
        }
        None
    }

    fn prune_old(&mut self) {
        let max_age = self.max_age;
        let latest = self.latest_seq.clone();
        self.pending.retain(|(d, e, seq), _| {
            latest
                .get(&(*d, *e))
                .is_none_or(|l| l.saturating_sub(*seq) <= max_age)
        });
    }
}

fn ensure_capacity(v: &mut Vec<Option<Vec<u8>>>, len: usize) {
    if v.len() < len {
        v.resize_with(len, || None);
    }
}
