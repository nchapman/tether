//! Property test for `FrameFragmenter` ↔ `FrameReassembler` under
//! random loss patterns spanning multiple frames.
//!
//! Hand-written cases for the happy-path round-trip (no loss) live in
//! `src/lib.rs#mod tests`. This file targets the multi-frame loss
//! accounting that hand-written cases can't enumerate efficiently —
//! specifically the `latest_seq`-driven stale-fragment classification
//! that bumps `fragments_lost`.

// Test-only sequence numbers derived from small loop indices; casts are in range.
#![allow(clippy::cast_possible_truncation)]

use bytes::Bytes;
use proptest::prelude::*;
use tether_protocol::video::{
    FrameFragmenter, FrameReassembler, HostFrameTiming, VideoFrameMeta, VideoPacket,
};

const MAX_AGE: u32 = 4;
const NUM_FRAMES: usize = 12;

fn meta() -> VideoFrameMeta {
    VideoFrameMeta {
        dimensions: (16, 16),
        keyframe: false,
        timing: HostFrameTiming::default(),
        input_echo: Default::default(),
    }
}

/// Body that fragments into exactly 2 packets (FIRST_PAYLOAD_BUDGET =
/// 1100, CONTINUATION_PAYLOAD_BUDGET = 1180 in `video.rs`).
fn two_packet_body_strategy() -> impl Strategy<Value = Bytes> {
    // [1101, 2280]: > FIRST_PAYLOAD_BUDGET (1100) so a continuation
    // exists, ≤ FIRST + CONTINUATION (1100 + 1180) so exactly one
    // continuation is emitted.
    (1101usize..=2280).prop_map(|n| Bytes::from(vec![0u8; n]))
}

/// Per-frame keep-mask of exactly 2 booleans. The proptest forces the
/// LAST frame's first fragment to be kept (asserted in-test) so
/// phase-1 reliably advances `latest_seq` to NUM_FRAMES-1 — making
/// every dropped fragment from frames more than MAX_AGE behind that
/// classifiably stale in phase 2.
fn trace_strategy() -> impl Strategy<Value = Vec<(Bytes, [bool; 2])>> {
    proptest::collection::vec(
        (two_packet_body_strategy(), [any::<bool>(), any::<bool>()]),
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

        let mut fragmenter = FrameFragmenter::new(0);
        let frame_packets: Vec<([bool; 2], Vec<VideoPacket>)> = trace
            .iter()
            .map(|(body, mask)| {
                let pkts = fragmenter.fragment(meta(), body.clone());
                // Bodies are chosen to fragment to exactly 2 packets;
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
                let pkts = fragmenter.fragment(meta(), Bytes::from(vec![0u8; 100]));
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
