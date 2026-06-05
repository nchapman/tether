//! Video datagram format + fragmentation / reassembly helpers.
//!
//! Every video frame — IDR keyframes and P-frames alike — rides the single
//! unreliable datagram channel, sliced into [`VideoPacket`]s and protected by
//! Reed-Solomon FEC. There is no separate reliable keyframe path: a split
//! transport gives no ordering guarantee between channels, so an IDR could be
//! overtaken by the P-frames that depend on it (the green-screen-on-connect
//! class of bug). One channel keeps an IDR and its dependents inherently
//! ordered; the client gates its decoder on the first IDR and re-requests it
//! while waiting.
//!
//! ## FEC layout
//!
//! A frame body is split into `K` uniform **primary shards** of `shard_size`
//! bytes (the last shard is short on the wire, zero-padded for the RS math).
//! The primaries are partitioned into one or more contiguous **FEC blocks**,
//! each independently Reed-Solomon coded so a large IDR stays loss-protected
//! without exceeding RS's 255-shards-per-block GF(2⁸) ceiling. A single block
//! (`K <= per-block ceiling`) is the common case; multi-block only kicks in for
//! large frames. The receiver derives the identical block layout from `K` +
//! `fec_pct`, so only those two values travel on the wire — see [`fec_layout`].
//!
//! ## Datagram sizing
//!
//! `shard_size` is chosen per frame from the connection's real datagram budget
//! minus the encoded packet header (including the variable meta envelope), so
//! every emitted datagram fits the path MTU — see
//! [`FrameFragmenter::fragment`]. The uniform-shard requirement means the
//! `First`/`Parity` packets (which carry the meta envelope) set the size and
//! the leaner `Continuation` packets leave a little headroom unused.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::control::VideoStreamId;
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
/// stream_id info, ROI hints, encoder QP feedback, etc.) must land as a new
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
    /// Convenience for receivers: collapse the envelope into the
    /// legacy `VideoFrameMeta` shape. Future variants update this
    /// method to project their richer payload back onto the
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
/// `(stream_id, stream_epoch, frame_seq)`.
///
/// `stream_id` identifies which host stream_id the frame came from. The
/// current host always uses `stream_id = 0` (single-monitor capture), but
/// the field is present so multi-monitor support can land later as a
/// pure additive change.
/// Each stream_id gets its own encoder thread and therefore its own
/// `stream_epoch` + `frame_seq` counters (cribbed from RustDesk's
/// `video_threads: HashMap<usize, _>` pattern).
///
/// `stream_epoch` is `u32` (varint-encoded as 1 byte for typical values) so
/// a long-lived session that restarts the encoder cannot wrap and reuse a
/// prior epoch (which would let the client misattribute fragments at the
/// wrong resolution / codec / hw context). The host bumps `stream_epoch`
/// whenever the encoder is restarted (resize, codec switch, HW context
/// loss). Clients drop all packets from prior epochs.
///
/// The frame descriptor (`fragment_count`, `fec_pct`, `shard_size`,
/// `total_body_len`) rides on `First` and on every `Parity` packet so a frame
/// whose `First` is lost can still be reconstructed from parity. `Continuation`
/// packets stay lean — the descriptor is recoverable from any other shard of
/// the frame.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoPacket {
    First {
        stream_id: VideoStreamId,
        stream_epoch: u32,
        frame_seq: u32,
        /// Total primary (data) shards across all FEC blocks for this frame.
        fragment_count: u16,
        /// Parity ratio the sender used, as a percentage of primaries per
        /// block (`0` = no FEC). The receiver derives the identical block
        /// layout + per-block parity count from `fragment_count` + this —
        /// see [`fec_layout`].
        fec_pct: u8,
        /// Uniform shard size for this frame, in bytes. Every shard
        /// (primary + parity) is this size for the RS math; the last
        /// primary is shorter on the wire and zero-padded during recovery.
        shard_size: u32,
        /// Total frame body length, to size the reassembly buffer and trim
        /// zero-padding off a reconstructed last shard.
        total_body_len: u32,
        meta: VideoFrameMetaEnvelope,
        /// Encoded payload slice (shard 0). `Bytes` (refcounted) rather than
        /// `Vec<u8>` so the fragmenter can produce per-fragment payloads via
        /// `Bytes::slice` (refcount bump, no copy) and the host can pass the
        /// encoder's output straight through to the QUIC `send_datagram`
        /// path without a per-fragment `to_vec()`.
        payload: Bytes,
    },
    Continuation {
        stream_id: VideoStreamId,
        stream_epoch: u32,
        frame_seq: u32,
        /// Global primary shard index, `1..fragment_count`.
        fragment_index: u16,
        payload: Bytes,
    },
    /// Reed-Solomon parity shard for one FEC block of a frame. Only emitted
    /// when the fragmenter is constructed with `fec_percentage > 0`.
    ///
    /// `block_index` selects the FEC block (derived layout); `parity_index`
    /// is `0..m` within that block's `m` parity shards. The frame descriptor
    /// is replicated here so a lost `First` doesn't make the frame
    /// unrecoverable. Each parity payload is exactly `shard_size` bytes.
    Parity {
        stream_id: VideoStreamId,
        stream_epoch: u32,
        frame_seq: u32,
        fragment_count: u16,
        fec_pct: u8,
        shard_size: u32,
        total_body_len: u32,
        /// Which FEC block this parity shard belongs to.
        block_index: u16,
        /// Parity shard index within the block, `0..m`.
        parity_index: u16,
        meta: VideoFrameMetaEnvelope,
        payload: Bytes,
    },
}

impl VideoPacket {
    /// Exact serialized wire size, in bytes. Used by the host's
    /// packet pacer to spread datagrams across the frame interval and
    /// by the fragmenter to size shards against the datagram budget.
    /// Uses `bincode::serialized_size` so the answer tracks the
    /// real wire shape automatically as the protocol evolves
    /// (envelope variants, new input-echo fields, etc.) — no
    /// manual header-constant maintenance.
    ///
    /// Returns `0` if serialization fails (only possible with a
    /// programmer error like a non-finite f32, which our wire
    /// schema doesn't expose).
    ///
    /// Note: on the wire each packet is wrapped in
    /// `crate::Datagram::Video(packet)`, which adds one byte for
    /// the outer enum discriminant ([`DATAGRAM_WRAPPER_BYTES`]).
    #[must_use]
    pub fn wire_size(&self) -> usize {
        crate::encode(self).map(|b| b.len()).unwrap_or(0)
    }

    /// `(stream_id, stream_epoch, frame_seq)` accessor common to all
    /// variants. Used by the reassembler to route packets without
    /// repeating the match in five places.
    #[must_use]
    pub fn route_key(&self) -> (VideoStreamId, u32, u32) {
        match self {
            Self::First {
                stream_id,
                stream_epoch,
                frame_seq,
                ..
            }
            | Self::Continuation {
                stream_id,
                stream_epoch,
                frame_seq,
                ..
            }
            | Self::Parity {
                stream_id,
                stream_epoch,
                frame_seq,
                ..
            } => (*stream_id, *stream_epoch, *frame_seq),
        }
    }
}

/// One byte for the outer `crate::Datagram::Video(_)` enum discriminant the
/// packet is wrapped in before it hits the wire. Accounted for when the
/// fragmenter sizes shards against the datagram budget.
pub const DATAGRAM_WRAPPER_BYTES: usize = 1;

/// Slack added on top of the measured empty-payload header when computing the
/// shard size, covering the payload length-prefix growing under bincode's
/// varint encoding: a length < 251 takes 1 byte (the empty-payload sample
/// used to measure the header), while a real shard ≥ 251 bytes takes 3 (a
/// marker byte + u16) — a 2-byte growth, plus 2 bytes of margin.
const SHARD_HEADER_SAFETY: usize = 4;

/// Floor on the per-frame shard size. Reached only if the meta envelope is so
/// large it leaves less than this under the datagram budget (a pathological
/// input-echo batch on a tiny-MTU path) — in which case fitting the meta in
/// one datagram is physically impossible and the frame is shipped at the floor
/// regardless.
pub const MIN_SHARD_SIZE: usize = 256;

/// Hard ceiling on primary shards per FEC block, regardless of parity
/// ratio. Caps at the GF(2⁸) Reed-Solomon ceiling of 255 total shards
/// (primary + parity); `(255 * 100) / (100 + fec_pct)` floors as `fec_pct`
/// rises — see [`max_primary_shards_for_pct`].
///
/// 212 is the value at the default 20% parity. Frames whose primary count
/// exceeds this are split across multiple FEC blocks (each `<=` this), so a
/// large IDR stays loss-protected rather than falling back to no-FEC.
pub const FEC_MAX_PRIMARY_SHARDS: usize = 212;

/// Upper bound on the parity ratio a legitimate sender advertises. Bounds the
/// per-block parity count the receiver derives (and therefore allocates) from
/// `fec_pct`. 100% (one parity shard per primary) is already far past any
/// useful operating point; the production default is 20%.
pub const MAX_FEC_PCT: u8 = 100;

/// Per-`fec_percentage` ceiling on primary shards per FEC block. Beyond this,
/// Reed-Solomon's GF(2⁸) limit of 255 total shards (primary + parity) is
/// exceeded and `ReedSolomon::new` rejects the block.
///
/// Capped at [`FEC_MAX_PRIMARY_SHARDS`] so the default 20% case keeps the
/// historical 212-shard block size.
#[must_use]
pub fn max_primary_shards_for_pct(fec_percentage: u8) -> usize {
    if fec_percentage == 0 {
        return FEC_MAX_PRIMARY_SHARDS;
    }
    let dynamic = (255usize * 100) / (100 + fec_percentage as usize);
    dynamic.min(FEC_MAX_PRIMARY_SHARDS)
}

/// Hard ceiling on the per-frame primary shard count enforced by
/// [`FrameReassembler`]. A legitimate sender produces
/// `ceil(body_len / shard_size)` primaries; at the project's bitrate budget
/// even a large 4K IDR sits well under this. 4096 covers any realistic frame
/// (≈4.5 MB body at a ~1100-byte shard) while bounding the receive-side
/// allocation a forged `fragment_count` could request. Above this ceiling the
/// receiver drops the packet rather than allocating the requested space.
pub const MAX_FRAGMENTS_PER_FRAME: usize = 4096;

/// Hard ceiling on `total_body_len`, in bytes — the largest body any
/// legitimate sender could fragment under [`MAX_FRAGMENTS_PER_FRAME`] at the
/// soft datagram payload budget.
pub const MAX_FRAME_BODY_BYTES: usize = MAX_FRAGMENTS_PER_FRAME * crate::MAX_DATAGRAM_PAYLOAD;

/// Hard cap on the number of simultaneously-pending (incomplete) frames the
/// [`FrameReassembler`] buffers. The per-stream `max_age` window and the
/// `max_pending_age` wall-clock timeout bound steady-state memory but not the
/// *instantaneous* entry count: a peer that sends one fragment each for a flood
/// of distinct `(stream_id, stream_epoch, frame_seq)` keys — never completing any
/// — accumulates entries faster than the prune can evict them, and each new
/// entry's descriptor can pre-allocate up to [`MAX_FRAGMENTS_PER_FRAME`] shard
/// slots. This cap bounds that worst case: once reached, fragments for *new*
/// keys are dropped (frames already pending still complete). The legitimate
/// working set is a few frames per active stream_id (the `max_age` window), so
/// 256 is generous headroom.
pub const MAX_PENDING_FRAMES: usize = 256;

/// One FEC block's position in the frame: `(primary_start, primary_count,
/// parity_count)`. The primaries are the global shards
/// `primary_start..primary_start + primary_count`.
type BlockSpec = (usize, usize, usize);

/// Deterministic FEC block layout shared by sender and receiver. Given the
/// total primary count `K` and the parity ratio, partitions the primaries into
/// contiguous blocks (the first `K % block_count` blocks get one extra shard)
/// and computes each block's parity count. Returns one [`BlockSpec`] per block.
///
/// `block_count = ceil(K / per-block ceiling)` keeps every block within RS's
/// 255-shard GF(2⁸) limit at the configured parity ratio. With `fec_pct == 0`
/// there is a single block and zero parity.
#[must_use]
pub fn fec_layout(shard_count: usize, fec_pct: u8) -> Vec<BlockSpec> {
    if shard_count == 0 {
        return Vec::new();
    }
    let block_count = if fec_pct == 0 {
        1
    } else {
        let ceiling = max_primary_shards_for_pct(fec_pct);
        shard_count.div_ceil(ceiling).max(1)
    }
    .min(shard_count);

    let base = shard_count / block_count;
    let extra = shard_count % block_count;
    let mut out = Vec::with_capacity(block_count);
    let mut start = 0;
    for b in 0..block_count {
        let k_b = base + usize::from(b < extra);
        let m_b = compute_parity_count(k_b, fec_pct);
        out.push((start, k_b, m_b));
        start += k_b;
    }
    out
}

/// `ceil(primary * pct / 100)`, but never zero when `pct > 0`. A
/// "fec_percentage = 1" + primary = 50 frame would otherwise produce
/// 0 parity shards (round-down), defeating the configuration.
fn compute_parity_count(primary: usize, fec_percentage: u8) -> usize {
    if fec_percentage == 0 {
        return 0;
    }
    let raw = primary
        .saturating_mul(fec_percentage as usize)
        .div_ceil(100);
    raw.max(1)
}

/// Splits a video frame body into a sequence of `VideoPacket`s sized to
/// fit inside the QUIC datagram budget. Owns the per-stream `frame_seq`
/// counter; bump `stream_epoch` via [`Self::bump_epoch`] whenever the
/// underlying encoder is restarted.
pub struct FrameFragmenter {
    stream_id: VideoStreamId,
    stream_epoch: u32,
    next_frame_seq: u32,
    /// Parity ratio as a percentage of primary shards per FEC block. `0`
    /// disables FEC entirely (no `Parity` packets emitted).
    fec_percentage: u8,
}

impl FrameFragmenter {
    pub fn new(stream_id: impl Into<VideoStreamId>) -> Self {
        Self::new_with_fec(stream_id, 0)
    }

    /// Construct a fragmenter with the given parity ratio. `0` disables FEC;
    /// positive values emit additional `VideoPacket::Parity` packets after the
    /// primaries of each `fragment()` call.
    pub fn new_with_fec(stream_id: impl Into<VideoStreamId>, fec_percentage: u8) -> Self {
        Self {
            stream_id: stream_id.into(),
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

    pub fn stream_id(&self) -> VideoStreamId {
        self.stream_id
    }

    pub fn stream_epoch(&self) -> u32 {
        self.stream_epoch
    }

    pub fn bump_epoch(&mut self) {
        self.stream_epoch = self.stream_epoch.wrapping_add(1);
        self.next_frame_seq = 0;
    }

    /// Fragment a frame body into one or more packets sized to fit
    /// `datagram_budget` bytes on the wire (the connection's real
    /// `max_datagram_size`, clamped to [`crate::MAX_DATAGRAM_PAYLOAD`]). The
    /// shard size is derived from the budget minus the encoded packet header —
    /// including this frame's meta envelope — so every emitted datagram,
    /// `First`/`Parity` included, encodes to at most `datagram_budget` bytes.
    ///
    /// `meta` rides in fragment 0 (and is replicated on every parity packet).
    /// An empty body still produces a single [`VideoPacket::First`] with
    /// `fragment_count = 1`. Per-fragment payloads are `Bytes::slice`s — a
    /// refcount bump on the underlying buffer, not a copy.
    pub fn fragment(
        &mut self,
        meta: VideoFrameMeta,
        body: Bytes,
        datagram_budget: usize,
    ) -> Vec<VideoPacket> {
        let frame_seq = self.next_frame_seq;
        self.next_frame_seq = self.next_frame_seq.wrapping_add(1);

        let mut meta = meta;
        // The meta envelope rides on First/Parity; an `input_echo` so large
        // that the header alone wouldn't leave MIN_SHARD_SIZE of payload makes
        // the budget physically unsatisfiable. Drop the excess echo IDs (they
        // are latency telemetry, not stream data) so every datagram fits.
        cap_input_echo(&mut meta, datagram_budget);
        let envelope = VideoFrameMetaEnvelope::V1(meta);
        let shard_size = shard_size_for_budget(datagram_budget, &envelope);
        let body_len = body.len();
        let primary_shards = if body_len == 0 {
            1
        } else {
            body_len.div_ceil(shard_size)
        };

        if primary_shards > MAX_FRAGMENTS_PER_FRAME {
            // Larger than the receiver will reassemble. Unreachable at the
            // project's bitrate budget (a ~4.5 MB frame at a ~1100-byte
            // shard); log rather than silently ship an unrecoverable frame.
            tracing::warn!(
                primary_shards,
                shard_size,
                body_len,
                "frame exceeds MAX_FRAGMENTS_PER_FRAME; receiver will reject it"
            );
        }

        let fec_pct = self.fec_percentage;
        let fragment_count = u16::try_from(primary_shards).unwrap_or(u16::MAX);
        let shard_size_u32 = u32::try_from(shard_size).unwrap_or(u32::MAX);
        let total_body_len = u32::try_from(body_len).unwrap_or(u32::MAX);

        let mut packets = Vec::new();

        // Primaries: First (shard 0, carries meta) + Continuations.
        let first_end = shard_size.min(body_len);
        packets.push(VideoPacket::First {
            stream_id: self.stream_id,
            stream_epoch: self.stream_epoch,
            frame_seq,
            fragment_count,
            fec_pct,
            shard_size: shard_size_u32,
            total_body_len,
            meta: envelope.clone(),
            payload: body.slice(..first_end),
        });
        let mut offset = first_end;
        let mut idx: u16 = 1;
        while offset < body_len {
            let end = (offset + shard_size).min(body_len);
            packets.push(VideoPacket::Continuation {
                stream_id: self.stream_id,
                stream_epoch: self.stream_epoch,
                frame_seq,
                fragment_index: idx,
                payload: body.slice(offset..end),
            });
            offset = end;
            idx += 1;
        }

        // Parity: one independent RS block per `fec_layout` entry.
        if fec_pct > 0 {
            for (block_index, &(start, k_b, m_b)) in
                fec_layout(primary_shards, fec_pct).iter().enumerate()
            {
                if m_b == 0 {
                    continue;
                }
                let Some(parity_payloads) = encode_block_parity(&body, start, k_b, m_b, shard_size)
                else {
                    // Unreachable: `fec_layout` keeps `k_b + m_b <= 255`, the
                    // only thing `ReedSolomon::new` rejects. Ship the block's
                    // primaries without parity rather than panic.
                    tracing::warn!(
                        block_index,
                        k_b,
                        m_b,
                        "reed-solomon construction failed; block shipped without parity"
                    );
                    continue;
                };
                for (parity_index, payload) in parity_payloads.into_iter().enumerate() {
                    packets.push(VideoPacket::Parity {
                        stream_id: self.stream_id,
                        stream_epoch: self.stream_epoch,
                        frame_seq,
                        fragment_count,
                        fec_pct,
                        shard_size: shard_size_u32,
                        total_body_len,
                        block_index: u16::try_from(block_index).unwrap_or(u16::MAX),
                        parity_index: u16::try_from(parity_index).unwrap_or(u16::MAX),
                        meta: envelope.clone(),
                        payload,
                    });
                }
            }
        }

        packets
    }
}

/// Upper bound on the per-datagram overhead (header + meta envelope + the
/// `Datagram::Video` wrapper + safety slack) for the largest packet variant.
/// Measured by encoding empty-payload sample `First`/`Parity` packets with
/// max-magnitude field values, so it bounds the real header regardless of the
/// frame's eventual shard/block counts.
#[must_use]
fn header_overhead(envelope: &VideoFrameMetaEnvelope) -> usize {
    let first = VideoPacket::First {
        stream_id: VideoStreamId(u32::MAX),
        stream_epoch: u32::MAX,
        frame_seq: u32::MAX,
        fragment_count: u16::MAX,
        fec_pct: u8::MAX,
        shard_size: u32::MAX,
        total_body_len: u32::MAX,
        meta: envelope.clone(),
        payload: Bytes::new(),
    };
    let parity = VideoPacket::Parity {
        stream_id: VideoStreamId(u32::MAX),
        stream_epoch: u32::MAX,
        frame_seq: u32::MAX,
        fragment_count: u16::MAX,
        fec_pct: u8::MAX,
        shard_size: u32::MAX,
        total_body_len: u32::MAX,
        block_index: u16::MAX,
        parity_index: u16::MAX,
        meta: envelope.clone(),
        payload: Bytes::new(),
    };
    first.wire_size().max(parity.wire_size()) + DATAGRAM_WRAPPER_BYTES + SHARD_HEADER_SAFETY
}

/// Per-frame shard size: the datagram budget minus the largest packet header,
/// so even a `First`/`Parity` datagram encodes to at most `datagram_budget`
/// bytes once wrapped in `Datagram::Video`. Floored at [`MIN_SHARD_SIZE`].
#[must_use]
fn shard_size_for_budget(datagram_budget: usize, envelope: &VideoFrameMetaEnvelope) -> usize {
    datagram_budget
        .saturating_sub(header_overhead(envelope))
        .max(MIN_SHARD_SIZE)
}

/// Drop trailing `input_echo` IDs until the meta-bearing header leaves at least
/// [`MIN_SHARD_SIZE`] of payload within `datagram_budget`. Keeps the
/// every-datagram-fits guarantee under a pathological input burst (the echo is
/// latency telemetry, so losing the tail degrades a metric, not the stream).
fn cap_input_echo(meta: &mut VideoFrameMeta, datagram_budget: usize) {
    if meta.input_echo.event_ids.is_empty() {
        return;
    }
    // Cost of the header with no echo IDs at all — the floor a First/Parity
    // pays before any echo.
    let mut probe = meta.clone();
    probe.input_echo.event_ids.clear();
    let base = header_overhead(&VideoFrameMetaEnvelope::V1(probe));

    // Bytes left for echo IDs while still fitting MIN_SHARD_SIZE of payload.
    // Each u64 ID costs at most 9 bytes under bincode's varint encoding; the
    // slack covers the Vec length prefix, which grows from 1 byte to at most 3
    // (marker + u16) as the count crosses 251 — 2 bytes, rounded up to 8 for
    // headroom. The `every_fragment_fits...input_echo` test would catch a
    // too-tight value.
    const ECHO_ID_MAX_BYTES: usize = 9;
    const ECHO_PREFIX_SLACK: usize = 8;
    let avail = datagram_budget.saturating_sub(base + MIN_SHARD_SIZE + ECHO_PREFIX_SLACK);
    let max_ids = avail / ECHO_ID_MAX_BYTES;

    if meta.input_echo.event_ids.len() > max_ids {
        let dropped = meta.input_echo.event_ids.len() - max_ids;
        meta.input_echo.event_ids.truncate(max_ids);
        tracing::debug!(
            dropped,
            max_ids,
            datagram_budget,
            "input echo truncated to fit datagram budget"
        );
    }
}

/// Build the `m` parity shards for one FEC block covering global primary
/// shards `start..start + k`. Returns `None` only if `ReedSolomon::new`
/// rejects `(k, m)` (impossible for the counts [`fec_layout`] permits).
fn encode_block_parity(
    body: &Bytes,
    start: usize,
    k: usize,
    m: usize,
    shard_size: usize,
) -> Option<Vec<Bytes>> {
    use reed_solomon_erasure::galois_8::ReedSolomon;

    let rs = ReedSolomon::new(k, m).ok()?;
    // (k + m) uniform shards: primaries copied from the body region (last
    // zero-padded), parity left zero, then encoded in place.
    let mut shards: Vec<Vec<u8>> = (0..(k + m))
        .map(|i| {
            let mut shard = vec![0u8; shard_size];
            if i < k {
                let s = (start + i) * shard_size;
                if s < body.len() {
                    let e = (s + shard_size).min(body.len());
                    shard[..e - s].copy_from_slice(&body[s..e]);
                }
            }
            shard
        })
        .collect();

    rs.encode(&mut shards).ok()?;
    Some(shards.into_iter().skip(k).map(Bytes::from).collect())
}

/// Reassembled frame produced by [`FrameReassembler::handle`].
///
/// `body` is `Bytes` rather than `Vec<u8>` so the decoder side can
/// slice / clone it without copying when forwarding to a worker
/// thread.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReassembledFrame {
    pub stream_id: VideoStreamId,
    pub stream_epoch: u32,
    pub frame_seq: u32,
    pub meta: VideoFrameMeta,
    pub body: Bytes,
}

/// Buffers in-flight fragments by `(stream_id, stream_epoch, frame_seq)`
/// and emits a [`ReassembledFrame`] when all primaries for a key have
/// arrived (directly or via per-block RS recovery). Drops fragments
/// belonging to frames more than `max_age` frames behind the latest seen on
/// that stream, so a permanent loss can't leak memory.
pub struct FrameReassembler {
    pending: HashMap<FrameKey, Pending>,
    latest_seq: HashMap<StreamKey, u32>,
    /// Highest `frame_seq` we've successfully returned a
    /// `ReassembledFrame` for, per stream. Used to drop late packets
    /// (typically `VideoPacket::Parity` arriving after all primaries
    /// already finalized the frame) without re-creating a ghost
    /// `pending` entry that would otherwise time out via
    /// `max_pending_age` and falsely inflate `frames_dropped`.
    finalized_seq: HashMap<StreamKey, u32>,
    max_age: u32,
    /// Wall-clock cap on how long a pending (incomplete) frame stays
    /// in the buffer before being evicted. Belt-and-braces alongside
    /// `max_age`: a near-quiet stream that suddenly stops can leave a
    /// half-reassembled frame sitting around because no newer
    /// fragments arrive to advance `latest_seq` past the eviction
    /// threshold.
    max_pending_age: Duration,
    /// Cumulative count of frames the reassembler started but pruned
    /// (timed out past `max_age`) before completing.
    frames_dropped: u64,
    /// Cumulative count of fragments rejected as stale (older than
    /// `max_age` behind the latest seq seen on their stream) or
    /// malformed (wire-validation rejection).
    fragments_lost: u64,
}

type FrameKey = (VideoStreamId, u32, u32);
type StreamKey = (VideoStreamId, u32);

struct Pending {
    /// Total primary shards (`K`). `0` until a `First`/`Parity` sets the
    /// descriptor — continuations alone can't determine it.
    fragment_count: u16,
    fec_pct: u8,
    shard_size: u32,
    total_body_len: Option<u32>,
    /// Primaries received so far (directly or reconstructed).
    received_count: u16,
    fragments: Vec<Option<Bytes>>,
    meta: Option<VideoFrameMeta>,
    /// When the first fragment for this frame arrived — for the wall-clock
    /// timeout in `prune_old`.
    first_seen: Instant,
    /// Set once a `First`/`Parity` provides `fragment_count` + `fec_pct`,
    /// fixing the FEC block layout and sizing `parity` / the recovery state.
    descriptor_known: bool,
    /// Parity shards per FEC block: `parity[block][parity_index]`. Sized when
    /// the descriptor is set; empty for `fec_pct == 0`.
    parity: Vec<Vec<Option<Bytes>>>,
    /// Latches per block once RS recovery has been attempted, so a failed
    /// reconstruct (unreachable over QUIC) isn't retried on every packet.
    block_recovery_attempted: Vec<bool>,
}

impl Pending {
    fn new() -> Self {
        Self {
            fragment_count: 0,
            fec_pct: 0,
            shard_size: 0,
            total_body_len: None,
            received_count: 0,
            fragments: Vec::new(),
            meta: None,
            first_seen: Instant::now(),
            descriptor_known: false,
            parity: Vec::new(),
            block_recovery_attempted: Vec::new(),
        }
    }

    /// Fix the FEC layout from the first `First`/`Parity` to arrive. Sizes the
    /// primaries vector to `K`, the per-block parity vectors, and the
    /// per-block recovery latch. Idempotent — later descriptor-bearing packets
    /// (which carry identical values) are no-ops.
    fn ensure_descriptor(&mut self, k: u16, fec_pct: u8, shard_size: u32, total_body_len: u32) {
        if self.descriptor_known {
            return;
        }
        self.descriptor_known = true;
        self.fragment_count = k;
        self.fec_pct = fec_pct;
        self.shard_size = shard_size;
        self.total_body_len = Some(total_body_len);
        ensure_capacity(&mut self.fragments, k as usize);
        // A Continuation received before this descriptor may have allocated and
        // counted a slot at an index >= k. That's only reachable from a
        // malformed/hostile peer — a legit sender never emits a fragment_index
        // >= fragment_count — but if left in place the out-of-range slot keeps
        // `received_count` above `fragment_count` forever, so the
        // `received_count == fragment_count` completion check is never true and
        // the frame never finalizes even once every real primary arrives. Drop
        // the out-of-range slots and recompute the count from the in-range ones.
        self.fragments.truncate(k as usize);
        self.received_count = u16::try_from(self.fragments.iter().filter(|s| s.is_some()).count())
            .unwrap_or(u16::MAX);
        let layout = fec_layout(k as usize, fec_pct);
        self.parity = layout.iter().map(|&(_, _, m)| vec![None; m]).collect();
        self.block_recovery_attempted = vec![false; layout.len()];
    }
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
            finalized_seq: HashMap::new(),
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
    /// Default is 500 ms.
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
        // Wire-level sizing validation. Reject before any allocation — a
        // forged packet with oversized counts / body length would otherwise
        // trigger a multi-MB allocation per packet, and the pending HashMap
        // stacks them. Bump `fragments_lost` so the rejection is observable.
        if let Some(reason) = validate_packet_sizing(&packet) {
            tracing::warn!(reason, "dropping malformed VideoPacket (wire-validation)");
            self.fragments_lost = self.fragments_lost.saturating_add(1);
            return None;
        }

        let (stream_id, stream_epoch, frame_seq) = packet.route_key();
        let stream_key = (stream_id, stream_epoch);
        let latest = *self
            .latest_seq
            .entry(stream_key)
            .and_modify(|s| *s = (*s).max(frame_seq))
            .or_insert(frame_seq);

        if latest.saturating_sub(frame_seq) > self.max_age {
            tracing::trace!(
                "dropping stale fragment: stream_id={} epoch={} seq={} latest={}",
                stream_id,
                stream_epoch,
                frame_seq,
                latest
            );
            self.fragments_lost = self.fragments_lost.saturating_add(1);
            return None;
        }

        self.prune_old();

        let key = (stream_id, stream_epoch, frame_seq);
        // Drop packets for a frame that's already finalized AND no longer has a
        // pending entry — the FEC late-parity case. Without this gate, the
        // trailing parity packets resurrect a ghost entry that prunes-as-
        // dropped and storms the client's recovery loop.
        if !self.pending.contains_key(&key) {
            if let Some(&final_seq) = self.finalized_seq.get(&stream_key) {
                if frame_seq <= final_seq {
                    tracing::trace!(
                        stream_id = %stream_id,
                        stream_epoch,
                        frame_seq,
                        final_seq,
                        "dropping late packet for already-finalized frame"
                    );
                    return None;
                }
            }
        }

        // Hard cap on concurrent pending frames (see [`MAX_PENDING_FRAMES`]). A
        // fragment for a *new* key that would exceed the cap is dropped rather
        // than allocating another descriptor, bounding the memory a peer can
        // pin by flooding distinct keys (e.g. unique `stream_epoch`s) without
        // ever completing a frame. Frames already pending are unaffected and
        // still complete. Counted as a fragment loss (overload, not a genuine
        // frame drop), which deliberately does not drive recovery.
        if !self.pending.contains_key(&key) && self.pending.len() >= MAX_PENDING_FRAMES {
            self.fragments_lost = self.fragments_lost.saturating_add(1);
            return None;
        }

        let entry = self.pending.entry(key).or_insert_with(Pending::new);

        match packet {
            VideoPacket::First {
                fragment_count,
                fec_pct,
                shard_size,
                total_body_len,
                meta,
                payload,
                ..
            } => {
                entry.ensure_descriptor(fragment_count, fec_pct, shard_size, total_body_len);
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
                // Once the descriptor is known, reject an out-of-range index
                // outright (a legit sender never emits idx >= fragment_count);
                // storing it would inflate `received_count` past
                // `fragment_count` and wedge finalization. Before the
                // descriptor arrives we can't know `k`, so store optimistically
                // and let `ensure_descriptor` reconcile.
                if entry.descriptor_known && idx >= entry.fragment_count as usize {
                    self.fragments_lost = self.fragments_lost.saturating_add(1);
                    return None;
                }
                ensure_capacity(&mut entry.fragments, idx + 1);
                if entry.fragments[idx].is_none() {
                    entry.fragments[idx] = Some(payload);
                    entry.received_count += 1;
                }
            }
            VideoPacket::Parity {
                fragment_count,
                fec_pct,
                shard_size,
                total_body_len,
                block_index,
                parity_index,
                meta,
                payload,
                ..
            } => {
                entry.ensure_descriptor(fragment_count, fec_pct, shard_size, total_body_len);
                // Parity replicates meta so a lost First can still be
                // recovered. First overwrites with its own meta if both arrive
                // — same envelope contents either way.
                if entry.meta.is_none() {
                    entry.meta = Some(meta.into_meta());
                }
                let b = block_index as usize;
                let p = parity_index as usize;
                if let Some(block) = entry.parity.get_mut(b) {
                    if let Some(slot) = block.get_mut(p) {
                        if slot.is_none() {
                            *slot = Some(payload);
                        }
                    }
                }
            }
        }

        // Fast path: every primary arrived. Byte-identical to the pre-FEC
        // concat behavior.
        if entry.descriptor_known
            && entry.received_count == entry.fragment_count
            && entry.meta.is_some()
        {
            return self.finalize(key, stream_id, stream_epoch, frame_seq);
        }

        // Recovery path: try to rebuild missing primaries from parity, per
        // block. Finalizes if recovery completes the frame.
        if entry.descriptor_known
            && entry.fec_pct > 0
            && entry.meta.is_some()
            && self.try_recover(key)
        {
            return self.finalize(key, stream_id, stream_epoch, frame_seq);
        }

        None
    }

    /// Attempt per-block RS recovery for the frame at `key`. Fills any
    /// reconstructable missing primaries into `fragments` and returns whether
    /// every primary is now present.
    fn try_recover(&mut self, key: FrameKey) -> bool {
        let Some(entry) = self.pending.get_mut(&key) else {
            return false;
        };
        let k = entry.fragment_count as usize;
        let shard_size = entry.shard_size as usize;
        let layout = fec_layout(k, entry.fec_pct);

        for (b, &(start, k_b, m_b)) in layout.iter().enumerate() {
            if entry
                .block_recovery_attempted
                .get(b)
                .copied()
                .unwrap_or(true)
            {
                continue;
            }
            let present_primary = (start..start + k_b)
                .filter(|&i| entry.fragments.get(i).is_some_and(Option::is_some))
                .count();
            if present_primary == k_b {
                // Block already complete — latch so later packets don't
                // re-scan its primaries on every parity arrival.
                entry.block_recovery_attempted[b] = true;
                continue;
            }
            let present_parity = entry
                .parity
                .get(b)
                .map(|block| block.iter().filter(|s| s.is_some()).count())
                .unwrap_or(0);
            if present_primary + present_parity < k_b {
                continue; // not enough shards to reconstruct this block yet
            }

            entry.block_recovery_attempted[b] = true;
            if let Some(recovered) = reconstruct_block(
                &entry.fragments,
                &entry.parity[b],
                start,
                k_b,
                m_b,
                shard_size,
            ) {
                for (i, shard) in recovered.into_iter().enumerate() {
                    let g = start + i;
                    if entry.fragments[g].is_none() {
                        entry.fragments[g] = Some(shard);
                        entry.received_count += 1;
                    }
                }
            }
        }

        entry.descriptor_known && entry.received_count == entry.fragment_count
    }

    fn finalize(
        &mut self,
        key: FrameKey,
        stream_id: VideoStreamId,
        stream_epoch: u32,
        frame_seq: u32,
    ) -> Option<ReassembledFrame> {
        let pending = self.pending.remove(&key)?;
        let meta = pending.meta?;
        let k = pending.fragment_count as usize;
        let shard_size = pending.shard_size as usize;
        let total = pending.total_body_len? as usize;

        let mut buf = BytesMut::with_capacity(total);
        for i in 0..k {
            let Some(shard) = pending.fragments.get(i).and_then(Option::as_ref) else {
                // Unreachable on the finalize paths (all primaries present),
                // but guard so we never publish a half-assembled body.
                tracing::warn!(i, "finalize found a missing primary shard; dropping frame");
                return None;
            };
            // Trim each shard to its expected on-wire length: full `shard_size`
            // for all but the last, the remainder for the last. A no-op for
            // directly-received shards; for an RS-reconstructed last shard it
            // strips the zero padding.
            let start = i.saturating_mul(shard_size);
            let expected = shard_size.min(total.saturating_sub(start));
            let take = expected.min(shard.len());
            buf.extend_from_slice(&shard[..take]);
        }

        self.mark_finalized(stream_id, stream_epoch, frame_seq);
        Some(ReassembledFrame {
            stream_id,
            stream_epoch,
            frame_seq,
            meta,
            body: buf.freeze(),
        })
    }

    /// Record `frame_seq` as the latest finalized frame on this
    /// `(stream_id, stream_epoch)` stream. Idempotent + monotonic.
    fn mark_finalized(&mut self, stream_id: VideoStreamId, stream_epoch: u32, frame_seq: u32) {
        self.finalized_seq
            .entry((stream_id, stream_epoch))
            .and_modify(|s| *s = (*s).max(frame_seq))
            .or_insert(frame_seq);
    }

    fn prune_old(&mut self) {
        let max_age = self.max_age;
        let max_pending_age = self.max_pending_age;
        let now = Instant::now();
        let latest = &self.latest_seq;
        let before = self.pending.len();
        self.pending.retain(|(d, e, seq), pending| {
            if now.duration_since(pending.first_seen) > max_pending_age {
                return false;
            }
            latest
                .get(&(*d, *e))
                .is_none_or(|l| l.saturating_sub(*seq) <= max_age)
        });
        let pruned = before.saturating_sub(self.pending.len());
        self.frames_dropped = self.frames_dropped.saturating_add(pruned as u64);

        // Bound the per-stream watermark maps. Both would otherwise grow one
        // entry per (stream_id, stream_epoch) ever seen — a peer spamming an
        // incrementing stream_epoch could leak memory without bound. Keep only
        // the newest epoch per stream_id (the live stream for a legitimately
        // monotonic sender) plus any stream with a frame still pending;
        // everything else is a dead epoch whose late packets we no longer need
        // to recognize. Bounds both maps to O(pending), itself capped by the
        // age + wall-clock eviction above.
        if self.latest_seq.len() > 1 || self.finalized_seq.len() > 1 {
            let pending_streams: HashSet<StreamKey> =
                self.pending.keys().map(|&(d, e, _)| (d, e)).collect();
            retain_live_streams(&mut self.latest_seq, &pending_streams);
            retain_live_streams(&mut self.finalized_seq, &pending_streams);
        }
    }
}

/// Retain only the newest `stream_epoch` per stream_id plus any stream still
/// referenced by a pending frame; drop superseded (dead-epoch) entries.
fn retain_live_streams(map: &mut HashMap<StreamKey, u32>, pending: &HashSet<StreamKey>) {
    if map.len() <= 1 {
        return;
    }
    let mut newest: HashMap<VideoStreamId, u32> = HashMap::new();
    for &(d, e) in map.keys() {
        newest
            .entry(d)
            .and_modify(|m| *m = (*m).max(e))
            .or_insert(e);
    }
    map.retain(|k, _| {
        let (d, e) = *k;
        newest.get(&d) == Some(&e) || pending.contains(k)
    });
}

/// Reconstruct one FEC block's `k` primaries from whatever primary +
/// parity shards are present. Returns the `k` primary shards (each padded to
/// `shard_size`; the last is trimmed by the caller) or `None` if RS decode
/// fails (unreachable over QUIC with valid `(k, m)`).
fn reconstruct_block(
    fragments: &[Option<Bytes>],
    parity: &[Option<Bytes>],
    start: usize,
    k: usize,
    m: usize,
    shard_size: usize,
) -> Option<Vec<Bytes>> {
    use reed_solomon_erasure::galois_8::ReedSolomon;

    let rs = ReedSolomon::new(k, m).ok()?;
    let mut shards: Vec<Option<Vec<u8>>> = Vec::with_capacity(k + m);
    for i in 0..k {
        shards.push(
            fragments
                .get(start + i)
                .and_then(Option::as_ref)
                .map(|b| pad_to(b, shard_size)),
        );
    }
    for slot in parity.iter().take(m) {
        shards.push(slot.as_ref().map(|b| pad_to(b, shard_size)));
    }
    // `parity` shorter than `m` would leave the vector under-length; pad with
    // None so RS sees the full (k + m) shape.
    while shards.len() < k + m {
        shards.push(None);
    }

    rs.reconstruct(&mut shards).ok()?;

    let mut out = Vec::with_capacity(k);
    for shard in shards.into_iter().take(k) {
        out.push(Bytes::from(shard?));
    }
    Some(out)
}

/// Copy `b` into a fresh `shard_size`-byte zero-padded buffer (RS requires
/// uniform-length shards). `b` is never longer than `shard_size`.
fn pad_to(b: &Bytes, shard_size: usize) -> Vec<u8> {
    let mut shard = vec![0u8; shard_size];
    let n = b.len().min(shard_size);
    shard[..n].copy_from_slice(&b[..n]);
    shard
}

fn ensure_capacity(v: &mut Vec<Option<Bytes>>, len: usize) {
    if v.len() < len {
        v.resize_with(len, || None);
    }
}

/// Wire-validation guard for [`FrameReassembler::handle`]. Returns `None` for
/// legitimate packets and `Some(reason)` for any packet whose declared sizing
/// exceeds the receive-side caps. Called before any `pending` insert or
/// `ensure_capacity`, so a rejected packet costs only deserialization.
fn validate_packet_sizing(packet: &VideoPacket) -> Option<&'static str> {
    /// Shared descriptor checks for `First`/`Parity`.
    fn check_descriptor(
        fragment_count: u16,
        fec_pct: u8,
        shard_size: u32,
        total_body_len: u32,
    ) -> Option<&'static str> {
        if fragment_count == 0 {
            return Some("descriptor fragment_count is zero");
        }
        if fragment_count as usize > MAX_FRAGMENTS_PER_FRAME {
            return Some("descriptor fragment_count exceeds MAX_FRAGMENTS_PER_FRAME");
        }
        if fec_pct > MAX_FEC_PCT {
            return Some("descriptor fec_pct exceeds MAX_FEC_PCT");
        }
        if shard_size == 0 || shard_size as usize > crate::MAX_DATAGRAM_PAYLOAD {
            return Some("descriptor shard_size out of range");
        }
        if total_body_len as usize > MAX_FRAME_BODY_BYTES {
            return Some("descriptor total_body_len exceeds MAX_FRAME_BODY_BYTES");
        }
        // `K` shards of `shard_size` must be able to hold `total_body_len` —
        // guards an inconsistent (K, shard_size, total) triple from mis-sizing
        // the recovery buffer. We deliberately don't also enforce the lower
        // bound (that `K-1` shards would be too few): an over-claimed
        // `fragment_count` merely fails to finalize, which the pending-frame
        // age/wall-clock/count caps already bound.
        let capacity = (fragment_count as usize).saturating_mul(shard_size as usize);
        if (total_body_len as usize) > capacity {
            return Some("descriptor total_body_len exceeds fragment_count * shard_size");
        }
        None
    }

    match packet {
        VideoPacket::First {
            fragment_count,
            fec_pct,
            shard_size,
            total_body_len,
            ..
        } => check_descriptor(*fragment_count, *fec_pct, *shard_size, *total_body_len),
        VideoPacket::Continuation { fragment_index, .. } => {
            if *fragment_index == 0 {
                return Some("Continuation.fragment_index is zero (First's slot)");
            }
            if *fragment_index as usize >= MAX_FRAGMENTS_PER_FRAME {
                return Some("Continuation.fragment_index exceeds MAX_FRAGMENTS_PER_FRAME");
            }
            None
        }
        VideoPacket::Parity {
            fragment_count,
            fec_pct,
            shard_size,
            total_body_len,
            block_index,
            parity_index,
            ..
        } => {
            if let Some(reason) =
                check_descriptor(*fragment_count, *fec_pct, *shard_size, *total_body_len)
            {
                return Some(reason);
            }
            if *fec_pct == 0 {
                return Some("Parity packet with fec_pct = 0");
            }
            // Absolute bounds; the precise block_index < block_count and
            // parity_index < m checks happen at store time against the derived
            // layout (out-of-range shards are dropped silently there).
            if *block_index as usize >= MAX_FRAGMENTS_PER_FRAME {
                return Some("Parity.block_index exceeds MAX_FRAGMENTS_PER_FRAME");
            }
            if *parity_index as usize >= MAX_FRAGMENTS_PER_FRAME {
                return Some("Parity.parity_index exceeds MAX_FRAGMENTS_PER_FRAME");
            }
            None
        }
    }
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
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 0,
            fragment_count: u16::MAX,
            fec_pct: 20,
            shard_size: 1100,
            total_body_len: 1000,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&packet).is_some());
    }

    #[test]
    fn validate_rejects_zero_fragment_count() {
        let packet = VideoPacket::First {
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 0,
            fragment_count: 0,
            fec_pct: 0,
            shard_size: 1100,
            total_body_len: 0,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&packet).is_some());
    }

    #[test]
    fn validate_rejects_oversized_continuation_index() {
        let packet = VideoPacket::Continuation {
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 0,
            fragment_index: u16::MAX,
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&packet).is_some());
    }

    #[test]
    fn validate_rejects_continuation_index_zero() {
        // Shard 0 is always the `First`; a Continuation claiming index 0 is
        // malformed and must be rejected before it reaches the reassembler.
        let packet = VideoPacket::Continuation {
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 0,
            fragment_index: 0,
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&packet).is_some());
    }

    #[test]
    fn validate_rejects_oversized_shard_size() {
        let packet = VideoPacket::First {
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 0,
            fragment_count: 1,
            fec_pct: 0,
            shard_size: u32::MAX,
            total_body_len: 1000,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&packet).is_some());
    }

    #[test]
    fn validate_rejects_oversized_total_body_len() {
        let packet = VideoPacket::Parity {
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 0,
            fragment_count: 1,
            fec_pct: 20,
            shard_size: 1100,
            total_body_len: u32::MAX,
            block_index: 0,
            parity_index: 0,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&packet).is_some());
    }

    #[test]
    fn validate_rejects_total_body_len_above_shard_capacity() {
        // 2 shards × 1100 = 2200 capacity; 3000 can't fit.
        let packet = VideoPacket::First {
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 0,
            fragment_count: 2,
            fec_pct: 0,
            shard_size: 1100,
            total_body_len: 3000,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&packet).is_some());
    }

    #[test]
    fn validate_accepts_legitimate_packets() {
        let first = VideoPacket::First {
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 0,
            fragment_count: 10,
            fec_pct: 20,
            shard_size: 1100,
            total_body_len: 10_000,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&first).is_none());
        let parity = VideoPacket::Parity {
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 0,
            fragment_count: 10,
            fec_pct: 20,
            shard_size: 1100,
            total_body_len: 10_000,
            block_index: 0,
            parity_index: 1,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(validate_packet_sizing(&parity).is_none());
    }

    #[test]
    fn handle_rejected_packet_bumps_fragments_lost() {
        let mut reassembler = FrameReassembler::new();
        let (_, before) = reassembler.loss_counters();
        let crafted = VideoPacket::First {
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 0,
            fragment_count: u16::MAX,
            fec_pct: 20,
            shard_size: 1100,
            total_body_len: u32::MAX,
            meta: dummy_envelope(),
            payload: Bytes::new(),
        };
        assert!(reassembler.handle(crafted).is_none());
        let (_, after) = reassembler.loss_counters();
        assert_eq!(after, before + 1);
    }

    /// A peer spamming an ever-incrementing `stream_epoch` must not grow the
    /// reassembler's per-stream watermark maps without bound. Feeds 1000
    /// distinct single-shard epochs and asserts both maps stay tiny — only the
    /// newest live epoch (plus any pending) is retained.
    #[test]
    fn incrementing_stream_epoch_does_not_leak_watermark_maps() {
        let mut reassembler = FrameReassembler::new();
        for epoch in 0..1000u32 {
            let packet = VideoPacket::First {
                stream_id: VideoStreamId(0),
                stream_epoch: epoch,
                frame_seq: 0,
                fragment_count: 1,
                fec_pct: 0,
                shard_size: 16,
                total_body_len: 4,
                meta: dummy_envelope(),
                payload: Bytes::from_static(&[0u8; 4]),
            };
            // Each single-shard frame finalizes immediately.
            assert!(reassembler.handle(packet).is_some());
        }
        assert!(
            reassembler.latest_seq.len() <= 4,
            "latest_seq leaked: {} entries",
            reassembler.latest_seq.len()
        );
        assert!(
            reassembler.finalized_seq.len() <= 4,
            "finalized_seq leaked: {} entries",
            reassembler.finalized_seq.len()
        );
    }

    /// A peer flooding distinct `stream_epoch`s with *incomplete* frames (a
    /// lone `First` of a multi-shard frame that never completes) must not grow
    /// the `pending` buffer without bound. Unlike the watermark-map test above
    /// these frames never finalize, so they exercise the `MAX_PENDING_FRAMES`
    /// hard cap rather than the wall-clock/age prune. Each `First` declares the
    /// maximum `fragment_count`, the worst case for per-entry allocation.
    #[test]
    fn flood_of_incomplete_frames_is_capped_at_max_pending_frames() {
        let mut reassembler = FrameReassembler::new();
        let frag_count = u16::try_from(MAX_FRAGMENTS_PER_FRAME).unwrap();
        let flood = u32::try_from(MAX_PENDING_FRAMES * 4).unwrap();
        for epoch in 0..flood {
            let packet = VideoPacket::First {
                stream_id: VideoStreamId(0),
                stream_epoch: epoch,
                frame_seq: 0,
                // Multi-shard so the lone First never completes the frame.
                fragment_count: frag_count,
                fec_pct: 20,
                shard_size: 1100,
                total_body_len: 1100 * u32::from(frag_count),
                meta: dummy_envelope(),
                payload: Bytes::from_static(&[0u8; 16]),
            };
            // Never finalizes (only 1 of fragment_count primaries present).
            assert!(reassembler.handle(packet).is_none());
        }
        assert!(
            reassembler.pending.len() <= MAX_PENDING_FRAMES,
            "pending buffer exceeded the cap: {} entries (cap {})",
            reassembler.pending.len(),
            MAX_PENDING_FRAMES
        );
        // The over-cap fragments are counted as losses, not silently ignored.
        assert!(
            reassembler.loss_counters().1 > 0,
            "dropped-over-cap fragments must register as fragment losses"
        );
    }

    /// A malformed/hostile peer can send a `Continuation` with a
    /// `fragment_index` >= the frame's eventual `fragment_count`. If that
    /// out-of-range fragment (arriving before the descriptor) were left to
    /// inflate `received_count`, the `received_count == fragment_count`
    /// completion check would never be true and the frame would never finalize
    /// even once every real primary arrived. The reassembler must reconcile the
    /// count when the descriptor arrives so a legitimate frame still completes.
    #[test]
    fn out_of_range_continuation_does_not_wedge_finalization() {
        let mut fragmenter = FrameFragmenter::new_with_fec(0u8, 0); // primaries only
        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (128, 128),
        };
        let body = Bytes::from(vec![0x7Eu8; 3 * 1000]);
        let packets = fragmenter.fragment(meta, body.clone(), crate::MAX_DATAGRAM_PAYLOAD);
        assert!(
            packets.len() >= 3,
            "3 KB at fec_pct=0 should be ≥3 primary packets"
        );

        let mut reassembler = FrameReassembler::new();
        // Inject a bogus out-of-range continuation for the same key BEFORE the
        // descriptor (First) arrives — stored optimistically at a high index.
        let bogus = VideoPacket::Continuation {
            stream_id: VideoStreamId(0),
            stream_epoch: 0,
            frame_seq: 0,
            fragment_index: 99,
            payload: Bytes::from_static(&[0xFFu8; 16]),
        };
        assert!(reassembler.handle(bogus).is_none());

        // Deliver the real packets; the frame must still finalize byte-equal.
        let mut finalized = None;
        for pkt in packets {
            if let Some(frame) = reassembler.handle(pkt) {
                finalized = Some(frame);
            }
        }
        let frame =
            finalized.expect("frame must finalize despite the earlier out-of-range continuation");
        assert_eq!(frame.body, body, "reassembled body must be byte-equal");
    }

    /// Steady-state FEC: all primaries arrive before any parity, frame
    /// finalises cleanly, then the late parity packets must NOT resurrect a
    /// ghost pending entry (which would prune-as-dropped and storm the
    /// client's recovery loop).
    #[test]
    fn late_parity_after_finalize_does_not_create_ghost_entry() {
        let mut fragmenter = FrameFragmenter::new_with_fec(0u8, 20);
        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (128, 128),
        };
        // Three-shard frame so we get 3 primary + ≥1 parity.
        let body = Bytes::from(vec![0xAAu8; 3 * 1000]);
        let packets = fragmenter.fragment(meta, body, crate::MAX_DATAGRAM_PAYLOAD);
        let parity_count = packets
            .iter()
            .filter(|p| matches!(p, VideoPacket::Parity { .. }))
            .count();
        assert!(parity_count >= 1, "expected ≥1 parity packet");

        let mut reassembler = FrameReassembler::new();
        let (drops_before, lost_before) = reassembler.loss_counters();

        let (primaries, parities): (Vec<_>, Vec<_>) = packets.into_iter().partition(|p| {
            matches!(
                p,
                VideoPacket::First { .. } | VideoPacket::Continuation { .. }
            )
        });
        let mut finalised = None;
        for packet in primaries {
            if let Some(frame) = reassembler.handle(packet) {
                finalised = Some(frame);
            }
        }
        assert!(
            finalised.is_some(),
            "frame must finalise from primaries alone"
        );

        for packet in parities {
            assert!(
                reassembler.handle(packet).is_none(),
                "late parity must not produce a second frame for the same seq"
            );
        }

        let (drops_after, lost_after) = reassembler.loss_counters();
        assert_eq!(drops_before, drops_after, "no frames should drop");
        assert_eq!(lost_before, lost_after, "no fragments should count as lost");
    }

    /// The positive half of the recovery-trigger contract the client relies on
    /// (`recovery_warranted`): a frame that starts but never completes bumps
    /// `frames_dropped` (not `fragments_lost`) when `prune_old` evicts it for
    /// falling more than `max_age` behind the latest seq. The straggler/
    /// malformed half is covered by `handle_rejected_packet_bumps_fragments_lost`.
    #[test]
    fn incomplete_frame_pruned_past_max_age_bumps_frames_dropped() {
        let mut fragmenter = FrameFragmenter::new_with_fec(0u8, 20);
        let meta = VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (128, 128),
        };

        let mut reassembler = FrameReassembler::new();
        let (drops_before, lost_before) = reassembler.loss_counters();

        // Frame 0: multi-shard, but feed only its `First` so it stays pending.
        let incomplete = fragmenter.fragment(
            meta.clone(),
            Bytes::from(vec![0x5Au8; 3 * 1000]),
            crate::MAX_DATAGRAM_PAYLOAD,
        );
        let first = incomplete
            .into_iter()
            .find(|p| matches!(p, VideoPacket::First { .. }))
            .expect("multi-shard frame has a First");
        assert!(
            reassembler.handle(first).is_none(),
            "a lone First must not finalize a multi-shard frame"
        );

        // Frames 1..=5: single-shard, fed complete so each finalizes and
        // advances latest_seq. With max_age = 4, frame 0 falls out of the
        // window once latest reaches 5 and prune_old evicts it.
        for _ in 0..5 {
            let pkts = fragmenter.fragment(
                meta.clone(),
                Bytes::from(vec![0u8; 16]),
                crate::MAX_DATAGRAM_PAYLOAD,
            );
            let first = pkts
                .into_iter()
                .find(|p| matches!(p, VideoPacket::First { .. }))
                .expect("single-shard frame has a First");
            reassembler.handle(first);
        }

        let (drops_after, lost_after) = reassembler.loss_counters();
        assert!(
            drops_after > drops_before,
            "the never-completed frame 0 must be counted as a dropped frame \
             (drops {drops_before} -> {drops_after})"
        );
        assert_eq!(
            lost_after, lost_before,
            "a pruned incomplete frame is a frame drop, not a fragment loss"
        );
    }
}
