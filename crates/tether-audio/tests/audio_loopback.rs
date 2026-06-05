//! Audio loopback: the real Opus codec driven through the production
//! `Datagram::Audio` wire channel (`DuplexVideoChannel`). Three cases: lossless
//! round-trip; 1% random loss exercising pure PLC concealment (RED off); and
//! isolated loss healed end-to-end through the production `LossRecovery` seam
//! with depth-1 RED. Together they prove the stream survives loss (the issue's
//! "survives 1% packet loss" bar) both by concealment and by RED recovery.

use std::sync::Arc;

use tether_audio::test_pattern::sine_frame;
use tether_audio::{LossRecovery, OpusConfig, OpusDecoder, OpusEncoder, RedundancyBuffer};
use tether_protocol::audio::AudioPacket;
use tether_protocol::MonoNanos;
use tether_transport::test_support::{
    video_duplex_pair, video_duplex_pair_with_depth, LossyChannel, LossyConfig,
};
use tether_transport::{Datagram, VideoChannel};

/// Encode `n_frames` of 440 Hz sine into one `AudioPacket` per Opus frame, with
/// a monotonic `frame_seq` the receiver uses for gap detection and a
/// `redundancy_depth`-deep RED tail (0 disables it) — the production host's send
/// shape. `payload`/`redundant` are `Bytes`, so no copy is needed here either.
fn encode_packets(
    enc: &mut OpusEncoder,
    cfg: OpusConfig,
    n_frames: usize,
    redundancy_depth: usize,
) -> Vec<AudioPacket> {
    let fs = enc.frame_size();
    let mut packets = Vec::new();
    let mut seq = 0u32;
    let mut red = RedundancyBuffer::new(redundancy_depth);
    for i in 0..n_frames {
        let frame = sine_frame(cfg.sample_rate, cfg.channels, 440.0, fs, i * fs);
        for payload in enc.encode(&frame.samples).unwrap() {
            let redundant = red.next_tail(&payload);
            packets.push(AudioPacket::Opus {
                stream_epoch: 0,
                frame_seq: seq,
                t_capture: MonoNanos(u64::from(seq) * fs as u64),
                payload,
                redundant,
            });
            seq += 1;
        }
    }
    packets
}

#[tokio::test]
async fn audio_round_trips_through_datagram_channel() {
    let cfg = OpusConfig::default();
    let (host, client) = video_duplex_pair();
    let mut enc = OpusEncoder::new(cfg).unwrap();
    let mut dec = OpusDecoder::new(cfg).unwrap();

    let packets = encode_packets(&mut enc, cfg, 50, 0);
    let sent = packets.len();
    assert!(sent >= 48, "expected ~50 frames, got {sent}");
    for p in &packets {
        host.send_datagram(&Datagram::Audio(p.clone())).unwrap();
    }
    assert_eq!(
        host.datagrams_dropped(),
        0,
        "lossless channel drops nothing"
    );

    let mut decoded = 0usize;
    let mut peak = 0.0f32;
    for _ in 0..sent {
        let Datagram::Audio(AudioPacket::Opus { payload, .. }) =
            client.recv_datagram().await.unwrap()
        else {
            panic!("expected an audio datagram");
        };
        let pcm = dec.decode(&payload).unwrap();
        assert_eq!(pcm.frames(), enc.frame_size());
        decoded += 1;
        peak = peak.max(pcm.samples.iter().fold(0.0, |a, &s| a.max(s.abs())));
    }
    assert_eq!(decoded, sent, "every sent frame decoded");
    assert!(
        peak > 0.1,
        "decoded audio carries signal energy, peak={peak}"
    );
}

#[tokio::test]
async fn audio_survives_one_percent_loss_with_concealment() {
    let cfg = OpusConfig::default();
    // RED depth 0 deliberately: this test exercises the *pure PLC concealment*
    // path in isolation (no redundancy), so the concealment-vs-drops accounting
    // below is unambiguous. RED-driven recovery is covered separately by
    // `audio_recovers_isolated_loss_through_loss_recovery`.
    // Channel depth comfortably above the packet count: the only drops should be
    // the LossyChannel's, not silent channel-overflow drops (which `drop_log`
    // wouldn't count, leaving the receiver waiting on packets that never come).
    let (host_raw, client) = video_duplex_pair_with_depth(4096);
    let host = Arc::new(LossyChannel::new(
        host_raw.clone(),
        LossyConfig {
            drop_probability: 0.01,
            reorder_window: 0,
            seed: 0xA0D10,
        },
    ));
    let mut enc = OpusEncoder::new(cfg).unwrap();
    let mut dec = OpusDecoder::new(cfg).unwrap();

    let packets = encode_packets(&mut enc, cfg, 400, 0);
    let sent = packets.len() as u64;
    for p in &packets {
        host.send_datagram(&Datagram::Audio(p.clone())).unwrap();
    }
    let dropped = host.drop_log().len() as u64;
    let delivered = sent - dropped;
    assert!(
        dropped > 0,
        "1% loss over {sent} frames should drop at least one"
    );
    assert!(
        delivered as f64 >= sent as f64 * 0.95,
        "delivered {delivered}/{sent} — far more than 1% lost"
    );

    // Drain delivered packets in order; conceal interior gaps the way the
    // client jitter buffer does, and confirm no delivered packet wedges the
    // decoder.
    let mut last_seq: Option<u32> = None;
    let mut decoded = 0u64;
    let mut concealed = 0u64;
    for _ in 0..delivered {
        let Datagram::Audio(AudioPacket::Opus {
            frame_seq, payload, ..
        }) = client.recv_datagram().await.unwrap()
        else {
            panic!("expected an audio datagram");
        };
        // Conceal interior gaps the way the production client does: only a
        // small forward gap counts as loss, capped so a reorder or crafted
        // sequence can't spin or insert a long silence.
        const MAX_CONCEAL: u32 = 8;
        if let Some(prev) = last_seq {
            let gap = frame_seq.wrapping_sub(prev);
            if (2..=MAX_CONCEAL + 1).contains(&gap) {
                for _ in 0..gap - 1 {
                    let c = dec.conceal();
                    assert_eq!(c.frames(), cfg.frame_size());
                    concealed += 1;
                }
            }
        }
        let pcm = dec.decode(&payload).unwrap();
        assert_eq!(pcm.frames(), enc.frame_size());
        decoded += 1;
        last_seq = Some(frame_seq);
    }

    assert_eq!(
        decoded, delivered,
        "every delivered frame decoded without error"
    );
    // Interior gaps are concealed once each; trailing/leading drops can't be
    // detected from sequence numbers, so concealment never exceeds drops.
    assert!(
        concealed > 0,
        "at least one interior gap should be concealed"
    );
    assert!(
        concealed <= dropped,
        "concealed {concealed} > dropped {dropped}"
    );
}

/// End-to-end through the production `LossRecovery` seam: depth-1 RED recovers
/// every *isolated* lost frame from the next datagram's tail, decoding real
/// audio with zero concealment — the win RED buys over PLC. Drives the actual
/// library recovery code (not an inline reimplementation) over real transport
/// loss, with drops engineered to be interior and never adjacent so depth-1
/// always covers them.
#[tokio::test]
async fn audio_recovers_isolated_loss_through_loss_recovery() {
    let cfg = OpusConfig::default();
    let (host, client) = video_duplex_pair_with_depth(4096);
    let mut enc = OpusEncoder::new(cfg).unwrap();
    let mut recovery = LossRecovery::new(OpusDecoder::new(cfg).unwrap());

    // Depth-1 RED: each datagram carries the previous frame's payload.
    let packets = encode_packets(&mut enc, cfg, 200, 1);
    let total = packets.len();

    // Drop isolated interior frames (every 7th, so never two adjacent, and
    // never the first or last). The datagram *after* each drop carries the
    // lost frame in `redundant[0]`, so a depth-1 tail covers all of them.
    let mut dropped = 0u64;
    for (i, p) in packets.iter().enumerate() {
        if i % 7 == 3 && i != 0 && i != total - 1 {
            dropped += 1;
            continue;
        }
        host.send_datagram(&Datagram::Audio(p.clone())).unwrap();
    }
    let delivered = total as u64 - dropped;
    assert!(
        dropped > 0,
        "the drop pattern must remove at least one frame"
    );

    // Feed each delivered datagram through the real `LossRecovery`, counting the
    // PCM frames it emits (recovered + concealed + primary all flow to `submit`).
    let mut played = 0u64;
    for _ in 0..delivered {
        let Datagram::Audio(AudioPacket::Opus {
            frame_seq,
            payload,
            redundant,
            ..
        }) = client.recv_datagram().await.unwrap()
        else {
            panic!("expected an audio datagram");
        };
        recovery.accept(frame_seq, &payload, &redundant, |pcm| {
            assert_eq!(pcm.frames(), cfg.frame_size());
            played += 1;
        });
    }

    let s = recovery.stats();
    assert_eq!(
        s.recovered_frames, dropped,
        "every isolated drop recovered from RED exactly once"
    );
    assert_eq!(
        s.concealed_frames, 0,
        "depth-1 RED covers every isolated loss — no PLC needed"
    );
    assert_eq!(s.dropouts, 0, "no gap exceeded the conceal window");
    // Every delivered primary plus every recovered gap frame reached playback,
    // so the playback clock saw the full original frame count.
    assert_eq!(
        played, total as u64,
        "recovered + delivered = original frames"
    );
}
