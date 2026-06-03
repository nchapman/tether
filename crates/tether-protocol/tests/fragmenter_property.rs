//! Property test for `FrameFragmenter` ↔ `FrameReassembler` under
//! random loss patterns spanning multiple frames.
//!
//! Hand-written cases for the happy-path round-trip (no loss) live in
//! `src/lib.rs#mod tests`. This file targets the multi-frame loss
//! accounting that hand-written cases can't enumerate efficiently —
//! specifically the `latest_seq`-driven stale-fragment classification
//! that bumps `fragments_lost` — plus the FEC block-layout invariants
//! (`fec_layout` partitions primaries within the RS 255-shard ceiling, and the
//! receiver re-derives the sender's exact layout from the wire descriptor)
//! that the transport review flagged as untested.

// Test-only sequence numbers derived from small loop indices; casts are in range.
#![allow(clippy::cast_possible_truncation)]

use bytes::Bytes;
use proptest::prelude::*;
use tether_protocol::video::{
    FrameFragmenter, FrameReassembler, HostFrameTiming, VideoFrameMeta, VideoPacket,
};

const MAX_AGE: u32 = 4;
const NUM_FRAMES: usize = 12;
/// Datagram budget the fragmenter sizes shards against in these tests.
const BUDGET: usize = tether_protocol::MAX_DATAGRAM_PAYLOAD;

fn meta() -> VideoFrameMeta {
    VideoFrameMeta {
        dimensions: (16, 16),
        keyframe: false,
        timing: HostFrameTiming::default(),
        input_echo: Default::default(),
    }
}

/// The uniform shard size the fragmenter derives for [`meta`] at [`BUDGET`].
/// Probed at runtime (it depends on the encoded meta + header) by reading the
/// `First` payload length of a large multi-shard frame, so the two-packet body
/// strategy can target it without hard-coding the budget arithmetic.
fn probe_shard_size() -> usize {
    let mut frag = FrameFragmenter::new(0);
    let pkts = frag.fragment(meta(), Bytes::from(vec![0u8; 50_000]), BUDGET);
    match &pkts[0] {
        VideoPacket::First { payload, .. } => payload.len(),
        other => panic!("expected First, got {other:?}"),
    }
}

/// Per-frame keep-mask of exactly 2 booleans paired with an `extra` body
/// length in `1..=900`. The test turns `extra` into a body of
/// `shard_size + extra` bytes, which fragments to exactly two packets (the
/// probed `shard_size` is comfortably above 900). The LAST frame's first
/// fragment is forced kept in-test so phase 1 advances `latest_seq` to
/// NUM_FRAMES-1, making every dropped fragment more than MAX_AGE behind it
/// classifiably stale in phase 2.
fn trace_strategy() -> impl Strategy<Value = Vec<(usize, [bool; 2])>> {
    proptest::collection::vec(
        (1usize..=900, [any::<bool>(), any::<bool>()]),
        NUM_FRAMES..=NUM_FRAMES,
    )
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// Simulate two-phase delivery: phase 1 sends every kept fragment
    /// in chronological frame order; phase 2 sends every dropped
    /// fragment in chronological frame order *after* phase 1 has
    /// established a high `latest_seq` via the forcibly-kept last
    /// frame's first fragment.
    ///
    /// Invariant under test: `fragments_lost` increments by exactly
    /// 1 for each dropped fragment whose `frame_seq` is more than
    /// `MAX_AGE` behind `latest_seq` *at the moment that fragment
    /// arrives in phase 2*. Since phase 1 has already pushed
    /// `latest_seq` to NUM_FRAMES-1 (via the forced last-frame
    /// fragment), phase-2 fragments from frame `i` are stale iff
    /// `(NUM_FRAMES-1) - i > MAX_AGE` ⇔ `i < NUM_FRAMES-1-MAX_AGE`.
    ///
    /// This catches the mutation that removes
    /// `self.fragments_lost = ...saturating_add(1)` in the
    /// stale-fragment branch.
    #[test]
    fn fragments_lost_equals_late_arrivals(trace in trace_strategy()) {
        // Force last frame's first fragment kept to guarantee
        // latest_seq reaches NUM_FRAMES-1 by end of phase 1.
        let mut trace = trace;
        trace[NUM_FRAMES - 1].1[0] = true;

        let shard_size = probe_shard_size();
        let mut fragmenter = FrameFragmenter::new(0);
        let frame_packets: Vec<([bool; 2], Vec<VideoPacket>)> = trace
            .iter()
            .map(|(extra, mask)| {
                let body = Bytes::from(vec![0u8; shard_size + extra]);
                let pkts = fragmenter.fragment(meta(), body, BUDGET);
                // shard_size < body <= 2*shard_size ⇒ exactly 2 packets;
                // any deviation would invalidate the mask shape.
                assert_eq!(pkts.len(), 2, "body should fragment to 2 packets");
                (*mask, pkts)
            })
            .collect();

        let mut reassembler = FrameReassembler::new();

        // Phase 1: deliver kept fragments in chronological order.
        for (mask, pkts) in &frame_packets {
            for (i, pkt) in pkts.iter().enumerate() {
                if mask[i] {
                    let _ = reassembler.handle(pkt.clone());
                }
            }
        }
        let baseline_lost = reassembler.loss_counters().1;

        // Phase 2: deliver dropped fragments in chronological order.
        // By construction (last-frame's first kept), latest_seq is at
        // NUM_FRAMES-1 entering phase 2.
        let last_seq = (NUM_FRAMES as u32) - 1;
        let mut expected_lost = 0u64;
        for (frame_idx, (mask, pkts)) in frame_packets.iter().enumerate() {
            let frame_seq = frame_idx as u32;
            let is_stale = last_seq.saturating_sub(frame_seq) > MAX_AGE;
            for (i, pkt) in pkts.iter().enumerate() {
                if !mask[i] {
                    if is_stale {
                        expected_lost += 1;
                    }
                    let _ = reassembler.handle(pkt.clone());
                }
            }
        }
        let final_lost = reassembler.loss_counters().1;
        prop_assert_eq!(
            final_lost - baseline_lost,
            expected_lost,
            "fragments_lost must bump once per stale-fragment arrival"
        );
    }

    /// Across any random interleaving of `fragment` + `bump_epoch`
    /// calls, emitted packets satisfy:
    /// - `(stream_epoch, frame_seq)` pairs are non-decreasing within
    ///   each `stream_epoch` value.
    /// - `frame_seq` resets to 0 on each `bump_epoch`.
    #[test]
    fn epoch_and_seq_monotonic_across_random_interleaving(
        ops in proptest::collection::vec(any::<bool>(), 0..200usize)
    ) {
        let mut fragmenter = FrameFragmenter::new(0);
        let mut last_seq_per_epoch: std::collections::HashMap<u32, Option<u32>> =
            std::collections::HashMap::new();

        for op in ops {
            if op {
                let prev_epoch = fragmenter.stream_epoch();
                fragmenter.bump_epoch();
                let new_epoch = fragmenter.stream_epoch();
                prop_assert_eq!(new_epoch, prev_epoch.wrapping_add(1));
                prop_assert!(
                    last_seq_per_epoch.get(&new_epoch).copied().flatten().is_none(),
                    "new epoch starts with no observed seq"
                );
            } else {
                let pkts = fragmenter.fragment(meta(), Bytes::from(vec![0u8; 100]), BUDGET);
                for pkt in pkts {
                    let (_, epoch, seq) = pkt.route_key();
                    let entry = last_seq_per_epoch.entry(epoch).or_insert(None);
                    if let Some(last) = *entry {
                        prop_assert!(
                            seq >= last,
                            "seq within epoch {} went backwards: {} -> {}",
                            epoch, last, seq
                        );
                    }
                    *entry = Some(seq);
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 512,
        ..ProptestConfig::default()
    })]

    /// The FEC block layout is the load-bearing invariant of the whole scheme:
    /// sender and receiver each call `fec_layout(K, fec_pct)` to derive an
    /// *identical* block structure from the wire descriptor alone. This sweeps
    /// the full `(K, fec_pct)` space the wire bounds allow and asserts the
    /// layout is always reconstructible without panic or corruption:
    /// - blocks are contiguous and cover exactly K primaries (no gap/overlap),
    /// - every block stays under RS's 255-shard GF(2⁸) ceiling (a violation
    ///   would panic `ReedSolomon::new` at runtime),
    /// - parity presence matches the ratio (zero iff `fec_pct == 0`).
    ///
    /// Pure math, no I/O — the catch-net the QUIC/transport review asked for so
    /// an edit to the block-splitting arithmetic can't silently produce an
    /// un-encodable or un-recoverable layout.
    #[test]
    fn fec_layout_partitions_primaries_and_respects_rs_ceiling(
        k in 1usize..=tether_protocol::video::MAX_FRAGMENTS_PER_FRAME,
        fec_pct in 0u8..=tether_protocol::video::MAX_FEC_PCT,
    ) {
        let layout = tether_protocol::video::fec_layout(k, fec_pct);
        prop_assert!(!layout.is_empty(), "non-zero K must yield at least one block");

        let mut next_start = 0usize;
        let mut primary_total = 0usize;
        for (start, k_b, m_b) in &layout {
            prop_assert_eq!(*start, next_start, "blocks must be contiguous (no gap/overlap)");
            prop_assert!(*k_b >= 1, "every block carries at least one primary");
            if fec_pct == 0 {
                // No RS encode happens, so the 255 ceiling doesn't apply; there
                // is a single zero-parity block spanning all primaries.
                prop_assert_eq!(*m_b, 0, "fec_pct == 0 ⇒ no parity");
            } else {
                prop_assert!(*m_b >= 1, "fec_pct > 0 ⇒ ≥1 parity shard per block");
                prop_assert!(
                    k_b + m_b <= 255,
                    "block {}+{} exceeds RS 255-shard ceiling (K={}, fec_pct={})",
                    *k_b, *m_b, k, fec_pct
                );
            }
            next_start += *k_b;
            primary_total += *k_b;
        }
        prop_assert_eq!(primary_total, k, "blocks must cover exactly K primaries");
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 192,
        ..ProptestConfig::default()
    })]

    /// End-to-end agreement: the parity the fragmenter actually emits must
    /// equal what a receiver, reading only the wire descriptor
    /// `(fragment_count, fec_pct)` off the `First`, would re-derive via
    /// `fec_layout`. Spans single- and multi-block frames. Catches the
    /// sender/receiver layout-drift bug class where the receiver can't
    /// reconstruct the blocks the sender encoded (e.g. an edit to one side's
    /// parity math, or a descriptor that no longer carries enough to re-derive
    /// the layout).
    #[test]
    fn receiver_rederives_emitted_fec_layout_from_descriptor(
        body_len in 1usize..=400_000,
        fec_pct in 1u8..=50,
    ) {
        let mut frag = FrameFragmenter::new_with_fec(0, fec_pct);
        let pkts = frag.fragment(meta(), Bytes::from(vec![0x33u8; body_len]), BUDGET);

        // The descriptor the receiver reads off the wire.
        let (fragment_count, wire_fec_pct) = pkts
            .iter()
            .find_map(|p| match p {
                VideoPacket::First { fragment_count, fec_pct, .. } => {
                    Some((*fragment_count as usize, *fec_pct))
                }
                _ => None,
            })
            .expect("a fragmented frame always has a First");

        let parity_emitted = pkts
            .iter()
            .filter(|p| matches!(p, VideoPacket::Parity { .. }))
            .count();
        let primaries_emitted = pkts.len() - parity_emitted;

        prop_assert_eq!(
            primaries_emitted, fragment_count,
            "emitted primary count must equal the descriptor's fragment_count"
        );

        let layout = tether_protocol::video::fec_layout(fragment_count, wire_fec_pct);
        let parity_expected: usize = layout.iter().map(|(_, _, m_b)| *m_b).sum();
        prop_assert_eq!(
            parity_emitted, parity_expected,
            "emitted parity must equal the layout the receiver re-derives \
             (K={}, fec_pct={})",
            fragment_count, wire_fec_pct
        );
    }
}
