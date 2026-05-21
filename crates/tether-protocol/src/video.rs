//! Video datagram format + fragmentation / reassembly helpers.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::MonoNanos;
use serde::{Deserialize, Serialize};

/// Timing fields populated by the host as a frame moves through its pipeline.
/// All times in **host-local** [`MonoNanos`].
///
/// Construct via [`HostFrameTimingBuilder`] rather than this struct's
/// public fields — the builder enforces that every stamp is set before
/// the wire `HostFrameTiming` is materialized, so a future instrumentation
/// site that forgets to stamp will panic in dev instead of silently
/// shipping zeroes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostFrameTiming {
    pub t_capture_kernel: MonoNanos,
    pub t_capture_userspace: MonoNanos,
    pub t_encode_submit: MonoNanos,
    pub t_encode_done: MonoNanos,
    pub t_send: MonoNanos,
}

/// Assembles a [`HostFrameTiming`] one stamp at a time as a frame moves
/// through the host pipeline. The capture backend constructs it with the
/// two capture stamps; the host's encode loop calls [`Self::encode_submit`]
/// / [`Self::encode_done`] around the encoder invocation; the send loop
/// calls [`Self::finish`] right before handing the body to the
/// [`FrameFragmenter`].
///
/// `finish` panics if any stamp is missing — the intent is that a future
/// site that adds a new stage (e.g. a CPU-side scale pass) and forgets
/// to call the new setter trips a debug-time panic rather than ships
/// zero timestamps to the wire. (Forgetting `.finish()` itself is
/// already caught by the type system: [`VideoFrameMeta::timing`] is
/// `HostFrameTiming`, not `HostFrameTimingBuilder`, so the call site
/// can't compile without finalizing.)
pub struct HostFrameTimingBuilder {
    t_capture_kernel: MonoNanos,
    t_capture_userspace: MonoNanos,
    t_encode_submit: Option<MonoNanos>,
    t_encode_done: Option<MonoNanos>,
}

impl HostFrameTimingBuilder {
    /// Start a builder with both capture stamps populated. These come
    /// from the capture backend (`CapturedFrame::timestamps()`).
    pub fn captured(t_capture_kernel: MonoNanos, t_capture_userspace: MonoNanos) -> Self {
        Self {
            t_capture_kernel,
            t_capture_userspace,
            t_encode_submit: None,
            t_encode_done: None,
        }
    }

    /// Stamp `t_encode_submit = MonoNanos::now()`. Call immediately
    /// before invoking the encoder.
    pub fn encode_submit(&mut self) {
        self.t_encode_submit = Some(MonoNanos::now());
    }

    /// Stamp `t_encode_done = MonoNanos::now()`. Call immediately after
    /// the encoder returns.
    pub fn encode_done(&mut self) {
        self.t_encode_done = Some(MonoNanos::now());
    }

    /// Useful for the host's encode-latency rolling-average log line.
    /// Returns the delta in nanoseconds; 0 if either stamp is unset —
    /// in practice this means the encoder errored or returned no
    /// packets between [`Self::encode_submit`] and a `continue` that
    /// skipped [`Self::encode_done`].
    #[must_use]
    pub fn encode_delta_ns(&self) -> u64 {
        match (self.t_encode_submit, self.t_encode_done) {
            (Some(s), Some(d)) => d.saturating_sub(s),
            _ => 0,
        }
    }

    /// Stamp `t_send = MonoNanos::now()` and finalize. Panics if the
    /// encode stamps were not set — see the struct doc for rationale.
    pub fn finish(self) -> HostFrameTiming {
        HostFrameTiming {
            t_capture_kernel: self.t_capture_kernel,
            t_capture_userspace: self.t_capture_userspace,
            t_encode_submit: self
                .t_encode_submit
                .expect("HostFrameTimingBuilder::encode_submit not called"),
            t_encode_done: self
                .t_encode_done
                .expect("HostFrameTimingBuilder::encode_done not called"),
            t_send: MonoNanos::now(),
        }
    }
}

/// Echo of input events the host injected since the previous frame. Lets the
/// client compute true motion-to-photon latency (not just network RTT).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputEchoBatch {
    pub event_ids: Vec<u64>,
}

/// Per-frame metadata. Sent in fragment 0 of a video frame, not repeated
/// across continuation packets.
///
/// **Wire-additive policy:** `VideoFrameMeta` is a closed struct — adding a
/// field would be a wire break. New per-frame metadata (HDR mastering
/// display info, ROI hints, encoder QP feedback, etc.) must land as a new
/// variant of [`VideoFrameMetaEnvelope`], which is the type that actually
/// rides on the wire in [`VideoPacket::First`]. The receive side unwraps
/// to a `VideoFrameMeta` for downstream consumers; new variants update
/// that conversion.
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

/// Versioned envelope around [`VideoFrameMeta`]. Costs one discriminator
/// byte per keyframe-bearing packet; in return, future per-frame metadata
/// (HDR, ROI, QP) is purely additive — a new envelope variant rather than
/// a struct-field append (forbidden by the bincode positional-encoding
/// forward-compat rules).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoFrameMetaEnvelope {
    V1(VideoFrameMeta),
}

impl VideoFrameMetaEnvelope {
    /// Convenience for receivers: collapse any current envelope variant
    /// into the legacy `VideoFrameMeta` shape. Future variants update
    /// this method to project their richer payload back onto the
    /// `VideoFrameMeta` fields downstream code already reads.
    #[must_use]
    pub fn into_meta(self) -> VideoFrameMeta {
        match self {
            Self::V1(m) => m,
        }
    }
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
/// `stream_epoch` is `u32` (varint-encoded as 1 byte for typical values) so
/// a long-lived session that restarts the encoder cannot wrap and reuse a
/// prior epoch (which would let the client misattribute fragments at the
/// wrong resolution / codec / hw context). The host bumps `stream_epoch`
/// whenever the encoder is restarted (resize, codec switch, HW context
/// loss). Clients drop all packets from prior epochs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoPacket {
    First {
        display: u8,
        stream_epoch: u32,
        frame_seq: u32,
        fragment_count: u16,
        meta: VideoFrameMetaEnvelope,
        payload: Vec<u8>,
    },
    Continuation {
        display: u8,
        stream_epoch: u32,
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
    stream_epoch: u32,
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

    pub fn stream_epoch(&self) -> u32 {
        self.stream_epoch
    }

    pub fn bump_epoch(&mut self) {
        self.stream_epoch = self.stream_epoch.wrapping_add(1);
        self.next_frame_seq = 0;
    }

    /// Build a single-fragment `VideoPacket::First` carrying the whole
    /// body. For use on the reliable keyframe-stream path, where QUIC
    /// handles segmentation so we don't need to chunk into datagram-
    /// sized pieces. Advances `next_frame_seq` so subsequent P-frame
    /// fragments still reference correct sequence numbers.
    pub fn single_packet(&mut self, meta: VideoFrameMeta, body: Vec<u8>) -> VideoPacket {
        let frame_seq = self.next_frame_seq;
        self.next_frame_seq = self.next_frame_seq.wrapping_add(1);
        VideoPacket::First {
            display: self.display,
            stream_epoch: self.stream_epoch,
            frame_seq,
            fragment_count: 1,
            meta: VideoFrameMetaEnvelope::V1(meta),
            payload: body,
        }
    }

    /// Fragment a frame body into one or more packets. `meta` rides in
    /// fragment 0 only, wrapped in the current `VideoFrameMetaEnvelope`
    /// variant. An empty body still produces a single
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
            meta: VideoFrameMetaEnvelope::V1(meta),
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
    pub stream_epoch: u32,
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
    /// Wall-clock cap on how long a pending (incomplete) frame stays
    /// in the buffer before being evicted. Belt-and-braces alongside
    /// `max_age`: a near-quiet stream that suddenly stops can leave a
    /// half-reassembled frame sitting around because no newer
    /// fragments arrive to advance `latest_seq` past the eviction
    /// threshold. The timeout fires on the next fragment of any
    /// frame, so the cost is one `Instant::now()` per packet.
    max_pending_age: Duration,
    /// Cumulative count of frames the reassembler started but pruned
    /// (timed out past `max_age`) before completing — i.e. frames the
    /// client never got to render.
    frames_dropped: u64,
    /// Cumulative count of fragments rejected as stale (older than
    /// `max_age` behind the latest seq seen on their stream). This is
    /// a lower bound on lost-then-arrived-late fragments; truly lost
    /// fragments never show up at all and are inferred from
    /// `frames_dropped` instead.
    fragments_lost: u64,
}

type FrameKey = (u8, u32, u32);
type StreamKey = (u8, u32);

struct Pending {
    fragment_count: u16,
    received_count: u16,
    fragments: Vec<Option<Vec<u8>>>,
    meta: Option<VideoFrameMeta>,
    /// When the first fragment for this frame arrived. Used by the
    /// wall-clock timeout in `prune_old`.
    first_seen: Instant,
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
            max_pending_age: Duration::from_millis(500),
            frames_dropped: 0,
            fragments_lost: 0,
        }
    }

    /// Configure how many frames behind the latest a fragment can arrive
    /// before being dropped. Default is 4.
    pub fn with_max_age(mut self, max_age: u32) -> Self {
        self.max_age = max_age;
        self
    }

    /// Configure the wall-clock timeout for incomplete pending frames.
    /// Default is 500 ms — chosen as roughly the "user will notice a
    /// freeze" threshold; lower values evict faster but risk pruning a
    /// frame whose final fragments are still in flight on a high-RTT
    /// link.
    pub fn with_max_pending_age(mut self, max_pending_age: Duration) -> Self {
        self.max_pending_age = max_pending_age;
        self
    }

    /// Returns `(frames_dropped, fragments_lost)`. Counters are
    /// cumulative over the reassembler's lifetime; callers diff
    /// successive reads to compute per-interval rates for
    /// `ControlMessage::ClientStats`.
    #[must_use]
    pub fn loss_counters(&self) -> (u64, u64) {
        (self.frames_dropped, self.fragments_lost)
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
            self.fragments_lost = self.fragments_lost.saturating_add(1);
            return None;
        }

        // Run the wall-clock + frame_seq prune on every fragment so a
        // stream that suddenly goes silent (e.g. encoder restart with
        // half-delivered final frame) doesn't leak the orphan past
        // `max_pending_age`. Frame_seq-distance pruning happens here
        // too for parity with the previous "only on latest" trigger.
        self.prune_old();

        let key = (display, stream_epoch, frame_seq);
        let entry = self.pending.entry(key).or_insert_with(|| Pending {
            fragment_count: 0,
            received_count: 0,
            fragments: Vec::new(),
            meta: None,
            first_seen: Instant::now(),
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
                entry.meta = Some(meta.into_meta());
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
        let max_pending_age = self.max_pending_age;
        let now = Instant::now();
        let latest = self.latest_seq.clone();
        let before = self.pending.len();
        self.pending.retain(|(d, e, seq), pending| {
            // Wall-clock first: a frame that's been incomplete for
            // longer than `max_pending_age` is evicted even if no
            // newer frames have arrived on the stream to advance
            // `latest_seq`. Catches the "encoder went silent
            // mid-frame" case.
            if now.duration_since(pending.first_seen) > max_pending_age {
                return false;
            }
            latest
                .get(&(*d, *e))
                .is_none_or(|l| l.saturating_sub(*seq) <= max_age)
        });
        let pruned = before.saturating_sub(self.pending.len());
        // Each pending entry that got evicted is a frame the receiver
        // started reassembling but never finished — i.e. a frame the
        // renderer never sees. Count it as a drop so ClientStats can
        // report the loss.
        self.frames_dropped = self.frames_dropped.saturating_add(pruned as u64);
    }
}

fn ensure_capacity(v: &mut Vec<Option<Vec<u8>>>, len: usize) {
    if v.len() < len {
        v.resize_with(len, || None);
    }
}
