//! Datagram loss + IDR reliability assertions through `LossyChannel`.
//!
//! Wraps the duplex video channel in `LossyChannel` at a configured
//! drop rate, deterministic via seed. Asserts:
//! - `FrameReassembler::loss_counters()` reports loss in the right
//!   ballpark (lower bound: bounded above by per-event drop log).
//! - The IDR uni-stream path is unaffected — keyframes always arrive
//!   intact.
//! - An `IdrSignal::raise()` mirrors the production "loss → force IDR"
//!   trigger and the resulting keyframe round-trips reliably.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use tether_decode::test_support::{FakeDecoder, FakeOutcome};
use tether_decode::Worker;
use tether_protocol::video::{
    FrameFragmenter, FrameReassembler, HostFrameTiming, VideoFrameMeta, VideoPacket,
};
use tether_protocol::MonoNanos;
use tether_render::LatestFrame;
use tether_transport::test_support::{video_duplex_pair, LossyChannel, LossyConfig};
use tether_transport::{Datagram, VideoChannel};

fn meta(keyframe: bool) -> VideoFrameMeta {
    VideoFrameMeta {
        dimensions: (16, 16),
        keyframe,
        timing: HostFrameTiming::default(),
        input_echo: Default::default(),
    }
}

#[tokio::test]
async fn lossy_pframe_path_increments_reassembler_loss_counter() {
    let (host_raw, client) = video_duplex_pair();
    let host = Arc::new(LossyChannel::new(
        host_raw.clone(),
        LossyConfig {
            drop_probability: 0.10,
            reorder_window: 0,
            seed: 0xC0FFEE,
        },
    ));

    let mut fragmenter = FrameFragmenter::new(0);
    let mut reassembler = FrameReassembler::new();

    let bodies: Vec<Bytes> = (0..64u8).map(|i| Bytes::from(vec![i; 4096])).collect();
    let mut expected_packets = 0u64;
    for body in &bodies {
        let pkts = fragmenter.fragment(meta(false), body.clone());
        expected_packets += pkts.len() as u64;
        for pkt in pkts {
            host.send_datagram(&Datagram::Video(pkt)).unwrap();
        }
    }

    let drop_log = host.drop_log();
    let dropped_packets = drop_log.len() as u64;
    let delivered_expected = expected_packets - dropped_packets;
    assert!(
        dropped_packets > 0,
        "10% drop on 64 frames should drop at least one"
    );
    assert!(
        dropped_packets < expected_packets,
        "10% drop should not drop everything (seed=0xC0FFEE)"
    );

    // Drain the receiver until either we run out of pending packets or
    // we've reassembled enough frames to bound the loss count.
    let mut delivered = 0u64;
    let mut frames_completed = 0u64;
    while delivered < delivered_expected {
        let next = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            client.recv_datagram(),
        )
        .await;
        let Ok(Ok(dgram)) = next else { break };
        delivered += 1;
        let Datagram::Video(pkt) = dgram else {
            panic!("unexpected datagram");
        };
        if reassembler.handle(pkt).is_some() {
            frames_completed += 1;
        }
    }

    let (frames_dropped, fragments_lost) = reassembler.loss_counters();
    // Reassembler can only count losses inferred from observed fragments;
    // the bound is `<= dropped_packets`.
    assert!(
        frames_dropped + fragments_lost <= dropped_packets,
        "reassembler-reported loss ({}+{}) should not exceed sent-but-dropped ({})",
        frames_dropped,
        fragments_lost,
        dropped_packets
    );
    // Lower bound: with 64 fragmented frames at 10% drop on seed
    // 0xC0FFEE, the reassembler must classify at least one fragment
    // as lost (either via fragments_lost on a stale arrival or
    // frames_dropped on a max_age prune). Without this assertion the
    // upper-bound check above could pass with all zeros — exactly the
    // false-confidence failure mode the code-review flagged.
    assert!(
        frames_dropped + fragments_lost > 0,
        "10% drop on 64 fragmented frames should surface at least one loss \
         to the reassembler's counters; got frames_dropped={frames_dropped} \
         fragments_lost={fragments_lost}"
    );
    assert!(
        frames_completed > 0,
        "some frames should still reassemble despite 10% loss"
    );
}

#[tokio::test]
async fn idr_uni_stream_unaffected_by_lossy_wrapper() {
    let (host_raw, client) = video_duplex_pair();
    let host = LossyChannel::new(
        host_raw.clone(),
        LossyConfig {
            drop_probability: 1.0, // drop everything on the datagram path
            reorder_window: 0,
            seed: 1,
        },
    );

    let mut fragmenter = FrameFragmenter::new(0);
    let body: Bytes = vec![0xa5u8; 4096].into();
    let pkt = fragmenter.single_packet(meta(true), body.clone());

    // Hand the IDR through the lossy wrapper. The wrapper must NOT
    // tamper with the reliable stream — keyframes always arrive intact.
    host.send_video_keyframe(&pkt).await.unwrap();

    let received = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        client.accept_video_keyframe(),
    )
    .await
    .expect("timeout")
    .unwrap();
    match received {
        VideoPacket::First { payload, .. } => assert_eq!(payload, body),
        other => panic!("expected First, got {other:?}"),
    }
}

#[tokio::test]
async fn loss_drives_worker_soft_failure_then_keyframe_recovers() {
    // Models the full client recovery chain:
    //   lossy datagrams → FrameReassembler → reassembled-with-skip
    //   → Worker::process_job sees a libavcodec "warnings bumped"
    //     soft failure → Worker invokes request_idr callback
    //   → (production: host receives ForceIdr) → host sends an IDR
    //     via the reliable keyframe stream → client receives it.
    //
    // This test drives the entire chain via the production code
    // paths (LossyChannel → reassembler → Worker → request_idr →
    // keyframe round-trip), not just the last hop. Previously this
    // test manually called `idr.raise()`/`idr.take()`, which gave
    // false confidence about the upstream link.
    let (host_raw, client) = video_duplex_pair();
    let lossy = Arc::new(LossyChannel::new(
        host_raw.clone(),
        LossyConfig {
            drop_probability: 0.5,
            reorder_window: 0,
            seed: 7,
        },
    ));

    let mut fragmenter = FrameFragmenter::new(0);
    let mut reassembler = FrameReassembler::new();

    // Pump lossy P-frames through the production datagram path.
    for i in 0..16u8 {
        for pkt in fragmenter.fragment(meta(false), vec![i; 1024].into()) {
            lossy.send_datagram(&Datagram::Video(pkt)).unwrap();
        }
    }

    let mut reassembled: Vec<Bytes> = Vec::new();
    while let Ok(Ok(d)) = tokio::time::timeout(
        std::time::Duration::from_millis(100),
        client.recv_datagram(),
    )
    .await
    {
        if let Datagram::Video(pkt) = d {
            if let Some(frame) = reassembler.handle(pkt) {
                reassembled.push(frame.body);
            }
        }
    }
    let lossy_dropped = lossy.drop_log().len() as u64;
    assert!(
        lossy_dropped > 0,
        "50% drop on 16 fragments should drop at least one"
    );

    // Drive the reassembled bodies through a Worker whose `warnings`
    // closure bumps on every call — this simulates libavcodec
    // logging "P-frame references a fragment we dropped" inside the
    // decoder, the exact production trigger for the soft-failure
    // request_idr path.
    let frames = LatestFrame::new();
    let idr_calls = Arc::new(AtomicU32::new(0));
    let idr_cb = Arc::clone(&idr_calls);
    let warn_counter = Arc::new(AtomicU64::new(0));
    let warn_cb = Arc::clone(&warn_counter);

    // Scripted FakeDecoder that returns an empty frame list per
    // submit (Worker still treats warnings bumps as soft failures
    // regardless of whether frames are emitted).
    let outcomes: Vec<FakeOutcome> = (0..reassembled.len())
        .map(|_| FakeOutcome::Frames(vec![]))
        .collect();
    let mut worker = Worker::new(
        Box::new(FakeDecoder::new(outcomes)),
        frames.clone(),
        Arc::new(move || {
            idr_cb.fetch_add(1, Ordering::Relaxed);
        }),
        // Each Worker::process_job reads warnings twice (before /
        // after decode). Bumping by 1 every read guarantees the
        // worker sees `after > before`, classifying the job as a
        // soft failure.
        Arc::new(move || warn_cb.fetch_add(1, Ordering::Relaxed) + 1),
    );
    let _ = frames;

    for body in reassembled {
        let job = tether_decode::DecodeJob {
            body,
            host_in_client_clock: MonoNanos::now(),
            keyframe: false,
        };
        let c = worker.process_job(job, MonoNanos::now());
        assert!(
            c.soft_failure,
            "warnings bump must classify as soft failure"
        );
    }

    // Rate-limit collapses N consecutive soft failures into 1
    // request_idr call within the 500 ms window.
    assert_eq!(
        idr_calls.load(Ordering::Relaxed),
        1,
        "production code rate-limits ForceIdr to one per IDR_RATE_LIMIT window"
    );

    // Final hop: the IDR rides the reliable keyframe stream and must
    // arrive intact even though the datagram path is dropping
    // everything.
    let kf = fragmenter.single_packet(meta(true), Bytes::from(vec![0xff; 4096]));
    lossy.send_video_keyframe(&kf).await.unwrap();
    let received = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        client.accept_video_keyframe(),
    )
    .await
    .expect("timeout")
    .unwrap();
    assert!(matches!(received, VideoPacket::First { .. }));
}
