//! Loss recovery on the client decode side: turn each incoming audio datagram
//! into the PCM frames to play, healing gaps from the RED tail before falling
//! back to PLC.
//!
//! This owns the [`OpusDecoder`] and the per-stream sequence state so the
//! stateful orchestration — recover-from-RED, conceal, resync — lives next to
//! the codec and is unit-testable with real Opus frames, rather than being
//! trapped in the client binary's playback thread. The classification half
//! ([`classify_gap`]) is pure; [`LossRecovery::accept`] drives the decoder.

use bytes::Bytes;

use crate::{AudioFrame, OpusDecoder};

/// Max frames concealed/recovered for one sequence gap. Caps work per packet so
/// a crafted or huge jump can't insert a long silence or spin. 8 frames is
/// ~40 ms at the default 5 ms frames — plenty, since PLC quality degrades past
/// a few frames anyway, after which it's better to declare a dropout and let
/// the buffer resync.
pub const MAX_CONCEAL: u32 = 8;

/// What an arriving `frame_seq` means relative to the last one played. Pure
/// classification, separated from the decode/IO it drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeqGap {
    /// Gap of 1 — in order, nothing missing.
    InOrder,
    /// `missing` frames lost but within the conceal window → recover from the
    /// RED tail where present, else PLC, then play the arriving frame.
    Conceal { missing: u32 },
    /// A forward gap past the conceal window: a burst too long to hide, so it's
    /// an audible dropout. We don't synthesise a long silence; play the arriving
    /// frame and let the jitter buffer resync. `missing` is for accounting.
    Dropout { missing: u32 },
    /// Duplicate (gap 0) or a backward/reordered late packet (gap wraps past the
    /// halfway mark): not loss. Dropped so it can't double-play or yank the
    /// sequence backward.
    Stale,
}

/// Classify `seq` against the last delivered `prev` using forward wrapping
/// distance. A gap of `2..=max_conceal+1` is concealable loss; a larger forward
/// gap (up to the wrap midpoint) is an un-concealable burst dropout; gap 0 or a
/// backward gap is a duplicate/reorder. Splitting forward-loss from backward-
/// reorder keeps the loss counters honest on a reordering Wi-Fi path.
fn classify_gap(prev: u32, seq: u32, max_conceal: u32) -> SeqGap {
    let gap = seq.wrapping_sub(prev);
    if gap == 0 {
        SeqGap::Stale
    } else if gap == 1 {
        SeqGap::InOrder
    } else if gap <= max_conceal + 1 {
        SeqGap::Conceal { missing: gap - 1 }
    } else if gap <= u32::MAX / 2 {
        SeqGap::Dropout { missing: gap - 1 }
    } else {
        SeqGap::Stale
    }
}

/// Cumulative loss/recovery counters, for the client's periodic stats log. The
/// playback buffer's own health counters (underruns, drops) can't tell network
/// loss from clock drift or scheduling; these split it out.
#[derive(Debug, Default, Clone, Copy)]
pub struct RecoveryStats {
    /// Lost frames healed from the RED tail — real decoded audio, no click.
    pub recovered_frames: u64,
    /// Lost frames the tail couldn't cover, filled by PLC (soft).
    pub concealed_frames: u64,
    /// Frames lost in bursts past the conceal window (audible gaps).
    pub dropout_frames: u64,
    /// Number of such burst-dropout events.
    pub dropouts: u64,
    /// Duplicate / reordered-late packets dropped.
    pub stale_packets: u64,
    /// Decode failures (corrupt payload) on a primary or recovered frame.
    pub decode_errors: u64,
}

/// Owns the Opus decoder and per-stream sequence state; turns each incoming
/// audio datagram into zero or more PCM frames to play.
pub struct LossRecovery {
    decoder: OpusDecoder,
    last_seq: Option<u32>,
    stats: RecoveryStats,
}

impl LossRecovery {
    /// Wrap a decoder. The first accepted packet establishes the sequence base.
    #[must_use]
    pub fn new(decoder: OpusDecoder) -> Self {
        Self {
            decoder,
            last_seq: None,
            stats: RecoveryStats::default(),
        }
    }

    /// Cumulative counters snapshot.
    #[must_use]
    pub fn stats(&self) -> RecoveryStats {
        self.stats
    }

    /// Process one audio datagram: classify the gap from the previous frame,
    /// recover or conceal any missing frames, then decode the primary payload.
    /// `submit` is called for each PCM frame to play, **in order**. `redundant`
    /// is the RED tail, newest-first (`redundant[k]` is `seq - (k + 1)`).
    pub fn accept(
        &mut self,
        seq: u32,
        payload: &[u8],
        redundant: &[Bytes],
        mut submit: impl FnMut(&AudioFrame),
    ) {
        let mut last_played_seq = self.last_seq;
        if let Some(prev) = self.last_seq {
            match classify_gap(prev, seq, MAX_CONCEAL) {
                SeqGap::InOrder => {}
                SeqGap::Conceal { missing } => {
                    // Fill missing frames oldest-first. Frame `seq - j` lives at
                    // `redundant[j - 1]`, so iterate j high→low. Recover from the
                    // tail when present, else PLC. PLC and recovered decodes both
                    // advance decoder state, so the later frames stay coherent.
                    for j in (1..=missing).rev() {
                        let missing_seq = seq.wrapping_sub(j);
                        let mut played = false;
                        if let Some(red) = redundant.get((j - 1) as usize) {
                            match self.decoder.decode(red) {
                                Ok(pcm) => {
                                    self.stats.recovered_frames += 1;
                                    submit(&pcm);
                                    played = true;
                                }
                                Err(e) => {
                                    self.stats.decode_errors += 1;
                                    tracing::warn!(error = %e, "RED recovery decode failed; concealing");
                                }
                            }
                        }
                        if !played {
                            self.stats.concealed_frames += 1;
                            let pcm = self.decoder.conceal();
                            submit(&pcm);
                        }
                        last_played_seq = Some(missing_seq);
                    }
                }
                SeqGap::Dropout { missing } => {
                    self.stats.dropout_frames += u64::from(missing);
                    self.stats.dropouts += 1;
                    // The gap is too long to fill, but advance the decoder once
                    // with a null-packet PLC so it resyncs its predictor state
                    // before the next real frame. Output discarded — we're not
                    // filling the gap, just keeping decoder state coherent.
                    let _ = self.decoder.conceal();
                }
                SeqGap::Stale => {
                    self.stats.stale_packets += 1;
                    return;
                }
            }
        }
        match self.decoder.decode(payload) {
            Ok(pcm) => {
                submit(&pcm);
                self.last_seq = Some(seq);
            }
            Err(e) => {
                self.stats.decode_errors += 1;
                tracing::warn!(error = %e, "opus decode failed; dropping audio frame");
                // `last_seq` tracks frames actually emitted to playback. A
                // corrupt primary did not advance the playback clock, so leave
                // the gap visible to the next packet; its RED tail or PLC can
                // then repair the missing slot instead of silently shortening
                // the stream.
                self.last_seq = last_played_seq;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_pattern::sine_frame;
    use crate::{OpusConfig, OpusEncoder};
    use bytes::Bytes;

    // --- pure classification ---

    #[test]
    fn in_order_and_duplicate() {
        assert_eq!(classify_gap(10, 11, MAX_CONCEAL), SeqGap::InOrder);
        assert_eq!(classify_gap(10, 10, MAX_CONCEAL), SeqGap::Stale);
    }

    #[test]
    fn small_forward_gap_is_concealable_loss() {
        assert_eq!(
            classify_gap(10, 12, MAX_CONCEAL),
            SeqGap::Conceal { missing: 1 }
        );
        assert_eq!(
            classify_gap(10, 10 + MAX_CONCEAL + 1, MAX_CONCEAL),
            SeqGap::Conceal {
                missing: MAX_CONCEAL
            }
        );
    }

    #[test]
    fn forward_gap_past_window_is_an_audible_dropout() {
        assert_eq!(
            classify_gap(10, 10 + MAX_CONCEAL + 2, MAX_CONCEAL),
            SeqGap::Dropout {
                missing: MAX_CONCEAL + 1
            }
        );
    }

    #[test]
    fn backward_gap_is_a_reorder_not_loss() {
        assert_eq!(classify_gap(100, 95, MAX_CONCEAL), SeqGap::Stale);
        assert_eq!(classify_gap(1, 0, MAX_CONCEAL), SeqGap::Stale);
    }

    #[test]
    fn forward_gap_is_classified_correctly_across_the_seq_wrap() {
        assert_eq!(classify_gap(u32::MAX, 0, MAX_CONCEAL), SeqGap::InOrder);
        assert_eq!(
            classify_gap(u32::MAX - 1, 0, MAX_CONCEAL),
            SeqGap::Conceal { missing: 1 }
        );
    }

    // --- recovery orchestration through real Opus frames ---

    /// `(frame_seq, primary payload, RED tail newest-first)` — the wire shape of
    /// one audio datagram, as the test feeds it to `LossRecovery::accept`.
    type RedPacket = (u32, Bytes, Vec<Bytes>);

    /// Encode `n` frames of 440 Hz sine into Opus payloads, with a depth-1 RED
    /// tail (previous payload) per frame — the production wire shape.
    fn encode_with_red(n: usize) -> (OpusConfig, Vec<RedPacket>) {
        let cfg = OpusConfig::default();
        let mut enc = OpusEncoder::new(cfg).unwrap();
        let fs = enc.frame_size();
        let mut out = Vec::new();
        let mut seq = 0u32;
        let mut prev: Option<Bytes> = None;
        for i in 0..n {
            let frame = sine_frame(cfg.sample_rate, cfg.channels, 440.0, fs, i * fs);
            for payload in enc.encode(&frame.samples).unwrap() {
                let red = prev.iter().cloned().collect();
                out.push((seq, payload.clone(), red));
                prev = Some(payload);
                seq += 1;
            }
        }
        (cfg, out)
    }

    /// A clean stream (no loss) decodes every frame in order, with no recovery,
    /// concealment, or dropouts counted.
    #[test]
    fn clean_stream_decodes_every_frame() {
        let (cfg, packets) = encode_with_red(20);
        let mut rec = LossRecovery::new(OpusDecoder::new(cfg).unwrap());
        let mut played = 0usize;
        for (seq, payload, red) in &packets {
            rec.accept(*seq, payload, red, |pcm| {
                assert_eq!(pcm.frames(), cfg.frame_size());
                played += 1;
            });
        }
        assert_eq!(played, packets.len());
        let s = rec.stats();
        assert_eq!(s.recovered_frames, 0);
        assert_eq!(s.concealed_frames, 0);
        assert_eq!(s.dropout_frames, 0);
        assert_eq!(s.stale_packets, 0);
    }

    /// An isolated lost frame is recovered from the next packet's RED tail —
    /// real decoded audio (carries energy), not concealment.
    #[test]
    fn isolated_loss_is_recovered_from_red_not_concealed() {
        let (cfg, packets) = encode_with_red(20);
        let mut rec = LossRecovery::new(OpusDecoder::new(cfg).unwrap());
        let mut played = Vec::new();
        for (seq, payload, red) in &packets {
            // Drop frame 10 (interior); frame 11 carries it in redundant[0].
            if *seq == 10 {
                continue;
            }
            rec.accept(*seq, payload, red, |pcm| played.push(pcm.clone()));
        }
        let s = rec.stats();
        assert_eq!(s.recovered_frames, 1, "the dropped frame is RED-recovered");
        assert_eq!(
            s.concealed_frames, 0,
            "no PLC needed for a depth-1 covered loss"
        );
        assert_eq!(s.dropouts, 0);
        // The recovered frame carries real signal energy.
        let recovered = &played[10];
        let peak = recovered
            .samples
            .iter()
            .fold(0.0f32, |a, &s| a.max(s.abs()));
        assert!(
            peak > 0.01,
            "recovered frame must carry energy, peak={peak}"
        );
        assert!(!recovered.is_silent());
    }

    /// A corrupt primary payload should not advance `last_seq`: no PCM reached
    /// playback for that sequence number. The next in-order datagram must still
    /// see the one-frame gap and recover it from RED instead of silently
    /// shortening the audio clock by one frame.
    #[test]
    fn corrupt_primary_keeps_gap_recoverable_from_next_red_tail() {
        let (cfg, packets) = encode_with_red(12);
        let mut rec = LossRecovery::new(OpusDecoder::new(cfg).unwrap());
        let mut played = 0usize;

        for (seq, payload, red) in &packets[..5] {
            rec.accept(*seq, payload, red, |_| played += 1);
        }
        assert_eq!(played, 5);

        let (bad_seq, _bad_payload, bad_red) = &packets[5];
        let corrupt = Bytes::from(vec![0u8; 4096]);
        rec.accept(*bad_seq, &corrupt, bad_red, |_| played += 1);
        assert_eq!(played, 5, "corrupt primary emitted no PCM");
        assert_eq!(rec.stats().decode_errors, 1);

        let (seq, payload, red) = &packets[6];
        rec.accept(*seq, payload, red, |_| played += 1);
        assert_eq!(played, 7, "RED-recovered frame 5 plus primary frame 6");
        let s = rec.stats();
        assert_eq!(s.recovered_frames, 1);
        assert_eq!(s.concealed_frames, 0);
        assert_eq!(s.dropouts, 0);
    }

    /// Two adjacent losses with depth-1 RED: the newer is recovered from the
    /// surviving packet's tail, the older falls back to PLC (the tail can't
    /// reach it). Order is preserved and the decoder stays coherent.
    #[test]
    fn burst_beyond_depth_recovers_what_it_can_and_conceals_the_rest() {
        let (cfg, packets) = encode_with_red(20);
        let mut rec = LossRecovery::new(OpusDecoder::new(cfg).unwrap());
        let mut played = 0usize;
        for (seq, payload, red) in &packets {
            // Drop frames 9 and 10. Frame 11's depth-1 tail covers only 10.
            if *seq == 9 || *seq == 10 {
                continue;
            }
            rec.accept(*seq, payload, red, |_| played += 1);
        }
        let s = rec.stats();
        assert_eq!(s.recovered_frames, 1, "frame 10 recovered from RED");
        assert_eq!(s.concealed_frames, 1, "frame 9 beyond the tail → PLC");
        assert_eq!(s.dropouts, 0, "a 2-gap stays within the conceal window");
        // Every gap slot is filled (recovered or concealed), so the play count
        // matches the original frame count.
        assert_eq!(played, packets.len());
    }

    /// A duplicate/reordered-late packet is dropped: it neither double-plays nor
    /// pulls the sequence backward (which would fake a dropout on the next real
    /// packet).
    #[test]
    fn stale_packet_is_dropped() {
        let (cfg, packets) = encode_with_red(10);
        let mut rec = LossRecovery::new(OpusDecoder::new(cfg).unwrap());
        let mut played = 0usize;
        for (seq, payload, red) in &packets[..5] {
            rec.accept(*seq, payload, red, |_| played += 1);
        }
        // Re-deliver frame 2 (already played) as a late duplicate.
        let (seq, payload, red) = &packets[2];
        rec.accept(*seq, payload, red, |_| played += 1);
        assert_eq!(rec.stats().stale_packets, 1);
        assert_eq!(played, 5, "the duplicate did not play again");
        // A subsequent in-order frame (5) still plays — the sequence base wasn't
        // yanked backward by the stale packet.
        let (seq, payload, red) = &packets[5];
        rec.accept(*seq, payload, red, |_| played += 1);
        assert_eq!(played, 6);
        assert_eq!(
            rec.stats().dropouts,
            0,
            "no phantom dropout after the stale"
        );
    }

    /// A burst past the conceal window counts as a dropout (one resync PLC, no
    /// gap fill) and the post-gap frame still decodes.
    #[test]
    fn long_gap_is_a_dropout_and_resyncs() {
        let (cfg, packets) = encode_with_red(40);
        let mut rec = LossRecovery::new(OpusDecoder::new(cfg).unwrap());
        let mut played = 0usize;
        // Play 0..5, then jump to 30 (gap of 25 ≫ MAX_CONCEAL).
        for (seq, payload, red) in &packets[..5] {
            rec.accept(*seq, payload, red, |_| played += 1);
        }
        let (seq, payload, red) = &packets[30];
        rec.accept(*seq, payload, red, |pcm| {
            assert_eq!(pcm.frames(), cfg.frame_size());
            played += 1;
        });
        let s = rec.stats();
        assert_eq!(s.dropouts, 1);
        assert!(s.dropout_frames > u64::from(MAX_CONCEAL));
        assert_eq!(s.recovered_frames, 0, "RED can't span a 25-frame gap");
        assert_eq!(played, 6, "5 clean + the post-gap frame; gap not filled");
    }
}
