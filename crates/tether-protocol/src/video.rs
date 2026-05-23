//! Video datagram format + fragmentation / reassembly helpers.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::MonoNanos;
use bytes::{Bytes, BytesMut};
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
    /// V2 adds `reference_id`: a monotonically-increasing counter the
    /// host stamps each emitted frame with. The client tracks the
    /// most recent successfully-decoded `reference_id` and quotes it
    /// in [`crate::control::ControlMessage::RequestRecovery`] when
    /// the reassembler observes a stale-dropped fragment. The host
    /// uses the id to invalidate newer references and re-predict
    /// from an LTR (when LTR plumbing lands) or fall back to IDR.
    V2 {
        meta: VideoFrameMeta,
        reference_id: u32,
    },
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
            Self::V2 { meta, .. } => meta,
        }
    }

    /// `reference_id` if the envelope carries one. `V1` returns
    /// `None`; `V2`+ returns `Some` with the host-stamped id. The
    /// client uses this for LTR-style recovery requests; older
    /// clients ignore it without harm.
    #[must_use]
    pub fn reference_id(&self) -> Option<u32> {
        match self {
            Self::V1(_) => None,
            Self::V2 { reference_id, .. } => Some(*reference_id),
        }
    }
}

/// A single video datagram. Frames larger than the transport's max datagram
/// size are sliced into multiple packets sharing
/// `(display, stream_epoch, frame_seq)`.
///
/// `display` identifies which host display the frame came from. The
/// current host always uses `display = 0` (single-monitor capture), but
/// the field is present so multi-monitor support can land later as a
/// pure additive change.
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
        /// Encoded payload slice. `Bytes` (refcounted) rather than
        /// `Vec<u8>` so the fragmenter can produce per-fragment
        /// payloads via `Bytes::slice` (refcount bump, no copy) and
        /// the host can pass the encoder's output straight through to
        /// the QUIC `send_datagram` path without a per-fragment
        /// `to_vec()`. Wire shape under bincode is identical to the
        /// previous `Vec<u8>` encoding (length-prefixed bytes).
        payload: Bytes,
    },
    Continuation {
        display: u8,
        stream_epoch: u32,
        frame_seq: u32,
        fragment_index: u16,
        payload: Bytes,
    },
    /// Reed-Solomon parity shard for a frame. Only emitted when the
    /// fragmenter is constructed with `fec_percentage > 0`. The
    /// receiver runs an RS-decode when at least `data_shards` of the
    /// frame's `data_shards + parity_shards` total shards have
    /// arrived (any mix of primary and parity).
    ///
    /// `data_shards` mirrors `First.fragment_count` for sanity-check
    /// (and so the receiver can size the RS context before First
    /// arrives). `parity_shards` is the total parity count for the
    /// frame; `shard_index` is `0..parity_shards`.
    ///
    /// Wire size: each parity shard payload is exactly
    /// [`FEC_SHARD_SIZE`] bytes. Sub-shard-size primaries (the last
    /// one) are zero-padded for the RS computation but transmitted
    /// at their actual length in the First/Continuation packets.
    Parity {
        display: u8,
        stream_epoch: u32,
        frame_seq: u32,
        data_shards: u16,
        parity_shards: u16,
        shard_index: u16,
        /// Total bytes in the original frame body. The receiver
        /// needs this to trim zero-padding from a reconstructed
        /// last shard.
        total_body_len: u32,
        /// Frame metadata, replicated from `First`. Carried on every
        /// parity packet so reconstruction can still succeed when
        /// First itself is one of the lost primaries. The ~50-byte
        /// duplication is the cost of FEC actually being useful in
        /// the lost-First case; without it, losing First makes the
        /// frame unrecoverable even with sufficient parity.
        meta: VideoFrameMetaEnvelope,
        payload: Bytes,
    },
}

impl VideoPacket {
    /// Exact serialized wire size, in bytes. Used by the host's
    /// packet pacer to spread datagrams across the frame interval.
    /// Uses `bincode::serialized_size` so the answer tracks the
    /// real wire shape automatically as the protocol evolves
    /// (envelope variants, new input-echo fields, etc.) — no
    /// manual header-constant maintenance.
    ///
    /// Roughly 100 ns per call at typical packet sizes. At 60 fps ×
    /// ~30 packets/frame that's ~180 µs/sec — well under the
    /// per-second budget of any of the encode or send steps.
    /// Returns `0` if serialization fails (only possible with a
    /// programmer error like a non-finite f32, which our wire
    /// schema doesn't expose).
    ///
    /// Note: on the wire each packet is wrapped in
    /// `crate::Datagram::Video(packet)`, which adds one byte for
    /// the outer enum discriminant. The pacer's byte accounting
    /// is therefore off by 1 byte/packet — sub-0.1% error at
    /// typical fragmenter output, well below pacing precision.
    #[must_use]
    pub fn wire_size(&self) -> usize {
        crate::encode(self).map(|b| b.len()).unwrap_or(0)
    }

    /// `(display, stream_epoch, frame_seq)` accessor common to all
    /// variants. Used by the reassembler to route packets without
    /// repeating the match in five places.
    #[must_use]
    pub fn route_key(&self) -> (u8, u32, u32) {
        match self {
            Self::First {
                display,
                stream_epoch,
                frame_seq,
                ..
            }
            | Self::Continuation {
                display,
                stream_epoch,
                frame_seq,
                ..
            }
            | Self::Parity {
                display,
                stream_epoch,
                frame_seq,
                ..
            } => (*display, *stream_epoch, *frame_seq),
        }
    }
}

/// Conservative payload budget per [`VideoPacket::First`] packet. Leaves
/// ~100 bytes of header headroom inside [`crate::MAX_DATAGRAM_PAYLOAD`].
pub const FIRST_PAYLOAD_BUDGET: usize = 1100;

/// Conservative payload budget per [`VideoPacket::Continuation`] packet.
/// Leaves ~20 bytes of header headroom.
pub const CONTINUATION_PAYLOAD_BUDGET: usize = 1180;

/// Uniform shard size used when FEC is enabled.
///
/// Reed-Solomon requires all shards (primaries and parity) to be the
/// same length. Bounded by [`FIRST_PAYLOAD_BUDGET`] because the
/// First packet's meta envelope eats more header headroom than the
/// Continuation/Parity variants do — making this any larger would
/// push the First packet past [`crate::MAX_DATAGRAM_PAYLOAD`]. The
/// 7% bandwidth penalty on Continuations (vs their 1180 B budget)
/// is the cost of the uniform-shard requirement.
pub const FEC_SHARD_SIZE: usize = FIRST_PAYLOAD_BUDGET;

/// Maximum primary shards per FEC block. Single-block FEC today (no
/// multi-block split), so this is also the per-frame primary cap when
/// FEC is on. Caps at the GF(2^8) Reed-Solomon ceiling of 255 total
/// shards (primary + parity); the formula
/// `(255 * 100) / (100 + fec_pct)` floors as `fec_pct` rises.
///
/// At the default 20% parity, this lands at 212 primaries — ~233 KB
/// of frame body, comfortably above any P-frame at 25 Mbps / 60 fps
/// and large enough for typical IDRs too. Frames that exceed it fall
/// back to no-FEC fragmentation; multi-block split is a future
/// addition for sustained >100 Mbps streams.
pub const FEC_MAX_PRIMARY_SHARDS: usize = 212;

/// Hard ceiling on the per-frame fragment count enforced by
/// [`FrameReassembler`]. The legitimate sender produces at most
/// [`FEC_MAX_PRIMARY_SHARDS`] (212) for FEC-protected frames and
/// roughly `body_len / CONTINUATION_PAYLOAD_BUDGET` for non-FEC P-frames
/// — at the project's 25 Mbps / 60 fps budget, that's well under 100
/// fragments per frame. 1024 leaves comfortable headroom for the worst
/// realistic P-frame while bounding the receive-side allocation that
/// a forged `fragment_count` could request to ~1.2 MB per crafted
/// packet (vs. the unbounded GB-scale request the wire format alone
/// would permit). Above this ceiling the receiver drops the packet
/// rather than allocating the requested space.
pub const MAX_FRAGMENTS_PER_FRAME: usize = 1024;

/// Hard ceiling on `VideoPacket::Parity::total_body_len`, in bytes.
/// Mirrors [`MAX_FRAGMENTS_PER_FRAME`] times the per-fragment payload
/// budget — the largest body any legitimate sender could possibly
/// fragment under those rules.
pub const MAX_FRAME_BODY_BYTES: usize = MAX_FRAGMENTS_PER_FRAME * CONTINUATION_PAYLOAD_BUDGET;

/// Splits a video frame body into a sequence of `VideoPacket`s sized to
/// fit inside the QUIC datagram budget. Owns the per-stream `frame_seq`
/// counter; bump `stream_epoch` via [`Self::bump_epoch`] whenever the
/// underlying encoder is restarted.
pub struct FrameFragmenter {
    display: u8,
    stream_epoch: u32,
    next_frame_seq: u32,
    /// Parity ratio as a percentage of primary shards. `0` disables
    /// FEC entirely (no `Parity` packets emitted; First/Continuation
    /// fragments stay byte-identical to the pre-FEC output for the
    /// same body — wire-compat with older clients).
    fec_percentage: u8,
}

impl FrameFragmenter {
    pub fn new(display: u8) -> Self {
        Self::new_with_fec(display, 0)
    }

    /// Construct a fragmenter with the given parity ratio. `0` keeps
    /// behavior bit-identical to [`Self::new`]; positive values
    /// emit additional `VideoPacket::Parity` packets after each
    /// `fragment()` call. The fragmenter still emits the original
    /// `First`/`Continuation` packets; FEC is purely additive.
    pub fn new_with_fec(display: u8, fec_percentage: u8) -> Self {
        Self {
            display,
            stream_epoch: 0,
            next_frame_seq: 0,
            fec_percentage,
        }
    }

    /// Current parity ratio.
    #[must_use]
    pub fn fec_percentage(&self) -> u8 {
        self.fec_percentage
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
    pub fn single_packet(&mut self, meta: VideoFrameMeta, body: Bytes) -> VideoPacket {
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
    ///
    /// Per-fragment payloads are produced via `Bytes::slice` — a
    /// refcount bump on the underlying buffer, not a copy. The whole
    /// frame stays in a single allocation owned by `body`.
    pub fn fragment(&mut self, meta: VideoFrameMeta, body: Bytes) -> Vec<VideoPacket> {
        let frame_seq = self.next_frame_seq;
        self.next_frame_seq = self.next_frame_seq.wrapping_add(1);

        // FEC path uses uniform shard sizes so Reed-Solomon's
        // same-length-shard requirement is satisfied without
        // per-shard padding bookkeeping. Falls back to the
        // pre-FEC mixed-budget shape when fec_percentage = 0 OR
        // the body would need more than FEC_MAX_PRIMARY_SHARDS
        // shards (multi-block FEC is a future addition).
        let fec_on = self.fec_percentage > 0;
        let body_len = body.len();
        let primary_shards_needed = if body_len == 0 {
            1
        } else {
            body_len.div_ceil(FEC_SHARD_SIZE)
        };
        let use_fec = fec_on && primary_shards_needed <= FEC_MAX_PRIMARY_SHARDS;

        if !use_fec {
            return self.fragment_legacy(meta, body, frame_seq);
        }

        // Uniform-shard primary fragmentation. First.payload sits at
        // shard[0] (≤ FEC_SHARD_SIZE bytes); each Continuation
        // carries one full shard. The last shard may be shorter than
        // FEC_SHARD_SIZE on the wire, but its conceptual length for
        // RS math is the full FEC_SHARD_SIZE (zero-padded).
        let fragment_count = u16::try_from(primary_shards_needed)
            .expect("primary count fits in u16; capped at FEC_MAX_PRIMARY_SHARDS");
        let parity_count = compute_parity_count(primary_shards_needed, self.fec_percentage);

        let mut packets =
            Vec::with_capacity(primary_shards_needed + parity_count);

        // First shard. Wrap meta in the envelope; clone so we can
        // also replicate it across every parity packet for the
        // lost-First recovery case.
        let envelope = VideoFrameMetaEnvelope::V1(meta);
        let first_end = FEC_SHARD_SIZE.min(body_len);
        packets.push(VideoPacket::First {
            display: self.display,
            stream_epoch: self.stream_epoch,
            frame_seq,
            fragment_count,
            meta: envelope.clone(),
            payload: body.slice(..first_end),
        });

        let mut offset = first_end;
        let mut idx: u16 = 1;
        while offset < body_len {
            let end = (offset + FEC_SHARD_SIZE).min(body_len);
            packets.push(VideoPacket::Continuation {
                display: self.display,
                stream_epoch: self.stream_epoch,
                frame_seq,
                fragment_index: idx,
                payload: body.slice(offset..end),
            });
            offset = end;
            idx += 1;
        }

        // Compute parity. Reed-Solomon requires uniform-length
        // shards; assemble a contiguous zero-padded buffer of size
        // (primary + parity) * FEC_SHARD_SIZE, fill the primary
        // region from the body, leave the parity region zero, then
        // encode in place.
        if parity_count > 0 {
            if let Some(parity_packets) = encode_parity(
                self.display,
                self.stream_epoch,
                frame_seq,
                &body,
                primary_shards_needed,
                parity_count,
                envelope,
            ) {
                packets.extend(parity_packets);
            } else {
                // RS construction failure is unreachable for the
                // (data, parity) sizes we permit, but the crate
                // returns Result so log and ship without parity if
                // we ever hit it.
                tracing::warn!(
                    primary = primary_shards_needed,
                    parity = parity_count,
                    "reed-solomon construction failed; sending without FEC"
                );
            }
        }

        packets
    }

    /// Pre-FEC fragmentation shape: First gets FIRST_PAYLOAD_BUDGET,
    /// continuations get CONTINUATION_PAYLOAD_BUDGET. Wire-identical
    /// to the original `fragment` implementation; used when FEC is
    /// disabled or the frame is too large for single-block FEC.
    fn fragment_legacy(
        &self,
        meta: VideoFrameMeta,
        body: Bytes,
        frame_seq: u32,
    ) -> Vec<VideoPacket> {
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
            payload: body.slice(..first_len),
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
                payload: body.slice(offset..end),
            });
            offset = end;
            idx += 1;
        }

        packets
    }
}

/// `ceil(primary * pct / 100)`, but never zero when `pct > 0`. A
/// "fec_percentage = 1" + primary = 50 frame would otherwise produce
/// 0 parity shards (round-down), defeating the configuration.
fn compute_parity_count(primary: usize, fec_percentage: u8) -> usize {
    if fec_percentage == 0 {
        return 0;
    }
    let raw = primary.saturating_mul(fec_percentage as usize).div_ceil(100);
    raw.max(1)
}

/// Build the parity packets for a frame. Returns `None` if the
/// reed-solomon crate failed to construct an encoder (impossible for
/// the data + parity counts we permit, but the crate returns Result).
fn encode_parity(
    display: u8,
    stream_epoch: u32,
    frame_seq: u32,
    body: &Bytes,
    primary: usize,
    parity: usize,
    meta: VideoFrameMetaEnvelope,
) -> Option<Vec<VideoPacket>> {
    use reed_solomon_erasure::galois_8::ReedSolomon;

    let rs = ReedSolomon::new(primary, parity).ok()?;
    let total_body_len = u32::try_from(body.len()).ok()?;

    // Assemble shards: contiguous (primary + parity) * FEC_SHARD_SIZE
    // buffer. Primary region copied from body (zero-padded last
    // shard), parity region left zero. Then encode in place.
    let mut shards: Vec<Vec<u8>> = (0..(primary + parity))
        .map(|i| {
            if i < primary {
                let start = i * FEC_SHARD_SIZE;
                let end = (start + FEC_SHARD_SIZE).min(body.len());
                let mut shard = vec![0u8; FEC_SHARD_SIZE];
                shard[..end - start].copy_from_slice(&body[start..end]);
                shard
            } else {
                vec![0u8; FEC_SHARD_SIZE]
            }
        })
        .collect();

    rs.encode(&mut shards).ok()?;

    let parity_shards_u16 = u16::try_from(parity).ok()?;
    let data_shards_u16 = u16::try_from(primary).ok()?;
    Some(
        shards
            .into_iter()
            .enumerate()
            .skip(primary)
            .enumerate()
            .map(|(shard_idx, (_, payload))| VideoPacket::Parity {
                display,
                stream_epoch,
                frame_seq,
                data_shards: data_shards_u16,
                parity_shards: parity_shards_u16,
                shard_index: u16::try_from(shard_idx).expect("parity count fits in u16"),
                total_body_len,
                meta: meta.clone(),
                payload: Bytes::from(payload),
            })
            .collect(),
    )
}

/// Reassembled frame produced by [`FrameReassembler::handle`].
///
/// `body` is `Bytes` rather than `Vec<u8>` so the decoder side can
/// slice / clone it without copying when forwarding to a worker
/// thread. The reassembler still performs one final
/// `BytesMut::with_capacity(total_len)` + per-fragment
/// `extend_from_slice` to land the body contiguously — most decoder
/// APIs (libavcodec's `avcodec_send_packet`) require a single `&[u8]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReassembledFrame {
    pub display: u8,
    pub stream_epoch: u32,
    pub frame_seq: u32,
    pub meta: VideoFrameMeta,
    pub body: Bytes,
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
    fragments: Vec<Option<Bytes>>,
    meta: Option<VideoFrameMeta>,
    /// When the first fragment for this frame arrived. Used by the
    /// wall-clock timeout in `prune_old`.
    first_seen: Instant,
    /// Parity shards received so far. `None` slots are missing
    /// parity packets. Sized lazily on the first Parity packet for
    /// this frame; left empty when no Parity arrived.
    parity_shards: Vec<Option<Bytes>>,
    /// Set on the first Parity packet for the frame. `0` (= "no
    /// parity"), set non-zero once any Parity arrives. The
    /// reassembler will only attempt RS recovery when this is
    /// non-zero AND `received_primaries + received_parity >=
    /// fragment_count`.
    parity_count: u16,
    /// Set on the first Parity packet. The reassembler needs this
    /// to trim zero-padding from the reconstructed last shard.
    total_body_len: Option<u32>,
    /// Latches `true` the first time an RS reconstruct attempt
    /// runs for this frame. If reconstruct failed (impossible for
    /// well-formed shards over QUIC, but the RS crate returns
    /// Result), don't retry on every subsequent packet — that
    /// would be quadratic work for no benefit.
    recovery_attempted: bool,
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
        // Wire-level sizing validation. Reject before any allocation —
        // a forged packet with oversized fragment_count, parity_shards,
        // or total_body_len would otherwise trigger a multi-MB
        // `resize_with` / `BytesMut::with_capacity` per malicious
        // packet, and the pending HashMap stacks them. The legitimate
        // sender's caps are documented at MAX_FRAGMENTS_PER_FRAME /
        // MAX_FRAME_BODY_BYTES; anything above that is malformed or
        // hostile. Bump `fragments_lost` (the existing receive-side
        // drop counter) so the rejection is observable in metrics.
        if let Some(reason) = validate_packet_sizing(&packet) {
            tracing::warn!(reason, "dropping malformed VideoPacket (wire-validation)");
            self.fragments_lost = self.fragments_lost.saturating_add(1);
            return None;
        }

        let (display, stream_epoch, frame_seq) = packet.route_key();

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
            parity_shards: Vec::new(),
            parity_count: 0,
            total_body_len: None,
            recovery_attempted: false,
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
            VideoPacket::Parity {
                data_shards,
                parity_shards,
                shard_index,
                total_body_len,
                meta,
                payload,
                ..
            } => {
                // Treat data_shards as authoritative for sizing the
                // primaries vector — by the time a Parity packet
                // arrives, the receiver may not yet have seen the
                // First (out-of-order datagrams).
                ensure_capacity(&mut entry.fragments, data_shards as usize);
                if entry.fragment_count == 0 {
                    entry.fragment_count = data_shards;
                }
                if entry.parity_count == 0 {
                    entry.parity_count = parity_shards;
                    entry.total_body_len = Some(total_body_len);
                }
                // Parity replicates meta so a lost First can still
                // be recovered. First overwrites this with its own
                // meta if both arrive — same envelope contents
                // either way.
                if entry.meta.is_none() {
                    entry.meta = Some(meta.into_meta());
                }
                let idx = shard_index as usize;
                ensure_capacity(&mut entry.parity_shards, parity_shards as usize);
                if entry.parity_shards[idx].is_none() {
                    entry.parity_shards[idx] = Some(payload);
                }
            }
        }

        // Fast path: every primary arrived. Skip RS decode and run
        // the existing concat path, byte-identical to the pre-FEC
        // behavior.
        if entry.fragment_count > 0
            && entry.received_count == entry.fragment_count
            && entry.meta.is_some()
        {
            return self.finalize_primary(key, display, stream_epoch, frame_seq);
        }

        // Recovery path: enough total shards have arrived to run RS
        // decode and reconstruct the missing primaries.
        if entry.fragment_count > 0
            && entry.parity_count > 0
            && entry.meta.is_some()
            && !entry.recovery_attempted
            && enough_for_rs_decode(entry)
        {
            return self.finalize_with_recovery(key, display, stream_epoch, frame_seq);
        }

        None
    }

    fn finalize_primary(
        &mut self,
        key: FrameKey,
        display: u8,
        stream_epoch: u32,
        frame_seq: u32,
    ) -> Option<ReassembledFrame> {
        let pending = self
            .pending
            .remove(&key)
            .expect("entry exists, we just inserted into it");
        let meta = pending.meta.expect("checked Some above");
        let total_len: usize = pending
            .fragments
            .iter()
            .filter_map(|f| f.as_ref().map(bytes::Bytes::len))
            .sum();
        let mut buf = BytesMut::with_capacity(total_len);
        for fragment in pending.fragments.into_iter().flatten() {
            buf.extend_from_slice(&fragment);
        }
        let body = buf.freeze();
        Some(ReassembledFrame {
            display,
            stream_epoch,
            frame_seq,
            meta,
            body,
        })
    }

    fn finalize_with_recovery(
        &mut self,
        key: FrameKey,
        display: u8,
        stream_epoch: u32,
        frame_seq: u32,
    ) -> Option<ReassembledFrame> {
        use reed_solomon_erasure::galois_8::ReedSolomon;

        let pending = self.pending.remove(&key)?;
        let meta = pending.meta.clone()?;
        let primary = pending.fragment_count as usize;
        let parity = pending.parity_count as usize;
        let total_body_len = pending.total_body_len? as usize;

        let rs = ReedSolomon::new(primary, parity).ok()?;

        // Build a Vec<Option<Vec<u8>>> of length primary + parity.
        // Each Some shard is zero-padded to FEC_SHARD_SIZE; missing
        // shards are None. RS decode fills in the missing primaries
        // (we don't care about reconstructing missing parities).
        let mut shards: Vec<Option<Vec<u8>>> = Vec::with_capacity(primary + parity);
        for i in 0..primary {
            shards.push(pending.fragments.get(i).and_then(|opt| {
                opt.as_ref().map(|b| {
                    let mut shard = vec![0u8; FEC_SHARD_SIZE];
                    shard[..b.len()].copy_from_slice(b);
                    shard
                })
            }));
        }
        for i in 0..parity {
            shards.push(
                pending
                    .parity_shards
                    .get(i)
                    .and_then(|opt| opt.as_ref().map(|b| b.to_vec())),
            );
        }

        if rs.reconstruct(&mut shards).is_err() {
            // Reconstruction failure should be unreachable over
            // QUIC (authenticated datagrams + valid N/K). Re-insert
            // with `recovery_attempted = true` so future packets
            // don't re-trigger the same wasted RS attempt — the
            // frame will instead time out via `prune_old`.
            tracing::trace!(
                primary,
                parity,
                "RS reconstruct failed; will not retry this frame"
            );
            let mut pending = pending;
            pending.recovery_attempted = true;
            self.pending.insert(key, pending);
            return None;
        }

        // Concatenate the recovered primaries, trimming the last
        // shard down to its actual length so we don't ship the
        // zero padding as part of the body.
        let mut buf = BytesMut::with_capacity(total_body_len);
        for (i, shard) in shards.into_iter().take(primary).enumerate() {
            let Some(data) = shard else {
                // Should be unreachable after a successful
                // reconstruct, but guard so we never publish a
                // half-reconstructed body.
                tracing::warn!(i, "RS reconstruct returned None primary shard");
                return None;
            };
            let start = i * FEC_SHARD_SIZE;
            let end = (start + FEC_SHARD_SIZE).min(total_body_len);
            buf.extend_from_slice(&data[..end - start]);
        }

        Some(ReassembledFrame {
            display,
            stream_epoch,
            frame_seq,
            meta,
            body: buf.freeze(),
        })
    }

    fn prune_old(&mut self) {
        let max_age = self.max_age;
        let max_pending_age = self.max_pending_age;
        let now = Instant::now();
        // Disjoint borrow: `pending` (mut) and `latest_seq` (shared)
        // are distinct fields; bind locally to make this explicit so
        // we don't have to clone the map on every fragment.
        let latest = &self.latest_seq;
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

/// Decide whether `pending` has enough shards to run RS decode.
/// Requires (primary slots filled OR parity slots filled) ≥ data_shards
/// AND parity availability > 0.
fn enough_for_rs_decode(pending: &Pending) -> bool {
    let parity_received = pending
        .parity_shards
        .iter()
        .filter(|s| s.is_some())
        .count() as u16;
    pending.received_count + parity_received >= pending.fragment_count
}

fn ensure_capacity(v: &mut Vec<Option<Bytes>>, len: usize) {
    if v.len() < len {
        v.resize_with(len, || None);
    }
}

/// Wire-validation guard for [`FrameReassembler::handle`]. Returns
/// `None` for legitimate packets and `Some(reason)` for any packet
/// whose declared sizing exceeds the receive-side caps. Called before
/// any `pending` entry insert or `ensure_capacity` call, so a
/// rejected packet costs only the deserialisation work — no
/// `resize_with` or `BytesMut::with_capacity` runs on its claimed
/// dimensions.
fn validate_packet_sizing(packet: &VideoPacket) -> Option<&'static str> {
    match packet {
        VideoPacket::First { fragment_count, .. } => {
            if *fragment_count as usize > MAX_FRAGMENTS_PER_FRAME {
                return Some("First.fragment_count exceeds MAX_FRAGMENTS_PER_FRAME");
            }
        }
        VideoPacket::Continuation {
            fragment_index, ..
        } => {
            if *fragment_index as usize >= MAX_FRAGMENTS_PER_FRAME {
                return Some("Continuation.fragment_index exceeds MAX_FRAGMENTS_PER_FRAME");
            }
        }
        VideoPacket::Parity {
            data_shards,
            parity_shards,
            shard_index,
            total_body_len,
            ..
        } => {
            if *data_shards as usize > MAX_FRAGMENTS_PER_FRAME {
                return Some("Parity.data_shards exceeds MAX_FRAGMENTS_PER_FRAME");
            }
            if *parity_shards as usize > MAX_FRAGMENTS_PER_FRAME {
                return Some("Parity.parity_shards exceeds MAX_FRAGMENTS_PER_FRAME");
            }
            if *shard_index >= *parity_shards {
                return Some("Parity.shard_index out of range");
            }
            if *total_body_len as usize > MAX_FRAME_BODY_BYTES {
                return Some("Parity.total_body_len exceeds MAX_FRAME_BODY_BYTES");
            }
        }
    }
    None
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    fn dummy_envelope() -> VideoFrameMetaEnvelope {
        VideoFrameMetaEnvelope::V1(VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (128, 128),
        })
    }

    #[test]
    fn validate_rejects_oversized_first_fragment_count() {
        let packet = VideoPacket::First {
            display: 0,
            stream_epoch: 0,
            frame_seq: 0,
            fragment_count: u16::MAX,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&packet).is_some());
    }

    #[test]
    fn validate_rejects_oversized_continuation_index() {
        let packet = VideoPacket::Continuation {
            display: 0,
            stream_epoch: 0,
            frame_seq: 0,
            fragment_index: u16::MAX,
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&packet).is_some());
    }

    #[test]
    fn validate_rejects_oversized_parity_shards() {
        let packet = VideoPacket::Parity {
            display: 0,
            stream_epoch: 0,
            frame_seq: 0,
            data_shards: 1,
            parity_shards: u16::MAX,
            shard_index: 0,
            total_body_len: 1000,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&packet).is_some());
    }

    #[test]
    fn validate_rejects_oversized_total_body_len() {
        let packet = VideoPacket::Parity {
            display: 0,
            stream_epoch: 0,
            frame_seq: 0,
            data_shards: 1,
            parity_shards: 1,
            shard_index: 0,
            total_body_len: u32::MAX,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&packet).is_some());
    }

    #[test]
    fn validate_rejects_parity_shard_index_out_of_range() {
        let packet = VideoPacket::Parity {
            display: 0,
            stream_epoch: 0,
            frame_seq: 0,
            data_shards: 1,
            parity_shards: 2,
            shard_index: 2,
            total_body_len: 1000,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&packet).is_some());
    }

    #[test]
    fn validate_accepts_legitimate_packets() {
        let first = VideoPacket::First {
            display: 0,
            stream_epoch: 0,
            frame_seq: 0,
            fragment_count: 10,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&first).is_none());
        let parity = VideoPacket::Parity {
            display: 0,
            stream_epoch: 0,
            frame_seq: 0,
            data_shards: 10,
            parity_shards: 2,
            shard_index: 1,
            total_body_len: 11000,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&parity).is_none());
    }

    #[test]
    fn handle_rejected_packet_bumps_fragments_lost() {
        let mut reassembler = FrameReassembler::new();
        let (_, before) = reassembler.loss_counters();
        let crafted = VideoPacket::Parity {
            display: 0,
            stream_epoch: 0,
            frame_seq: 0,
            data_shards: u16::MAX,
            parity_shards: u16::MAX,
            shard_index: 0,
            total_body_len: u32::MAX,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(reassembler.handle(crafted).is_none());
        let (_, after) = reassembler.loss_counters();
        assert_eq!(after, before + 1);
    }
}
