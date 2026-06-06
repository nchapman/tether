//! End-to-end loopback: fragmenter → DuplexVideoChannel →
//! reassembler → `tether_decode::Worker` (with FakeDecoder) →
//! LatestFrame.
//!
//! Two flavors:
//! 1. Steady-state: N P-frames in, N frames eventually in
//!    `LatestFrame` (the last one stays after the consumer drains).
//! 2. Backpressure: producer faster than consumer; `Worker::process_job`
//!    reports a non-zero `render_drops` count matching the
//!    `produced - consumed` arithmetic.

// Drop counts are small test values; the u64 -> u32 casts are in range.
#![allow(clippy::cast_possible_truncation)]

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use tether_codec::Frame as CodecFrame;
use tether_decode::test_support::{FakeDecoder, FakeOutcome};
use tether_decode::Worker;
use tether_protocol::video::{FrameFragmenter, FrameReassembler, HostFrameTiming, VideoFrameMeta};
use tether_protocol::MonoNanos;
use tether_render::{Frame as RenderFrame, LatestFrame};
use tether_transport::test_support::video_duplex_pair;
use tether_transport::{Datagram, VideoChannel};

fn meta() -> VideoFrameMeta {
    VideoFrameMeta {
        dimensions: (16, 16),
        keyframe: false,
        timing: HostFrameTiming::default(),
        input_echo: Default::default(),
    }
}

fn solid_frame(width: u32, height: u32, luma: u8) -> CodecFrame {
    use tether_codec::DecodedFrame;
    let y_len = (width as usize) * (height as usize);
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    let uv_len = cw * ch * 2;
    CodecFrame::Cpu(DecodedFrame {
        width,
        height,
        pts: None,
        y: vec![luma; y_len],
        uv: vec![0x80; uv_len],
    })
}

fn make_worker_with(outcomes: Vec<FakeOutcome>) -> (Worker, LatestFrame, Arc<AtomicU32>) {
    let frames = LatestFrame::new();
    let idr_calls = Arc::new(AtomicU32::new(0));
    let idr_cb = Arc::clone(&idr_calls);
    let worker = Worker::new(
        Box::new(FakeDecoder::new(outcomes)),
        frames.clone(),
        Arc::new(move || {
            idr_cb.fetch_add(1, Ordering::Relaxed);
        }),
        Arc::new(|| 0u64),
    );
    (worker, frames, idr_calls)
}

#[tokio::test]
async fn pframes_flow_through_full_chain_to_latest_frame() {
    let (host, client) = video_duplex_pair();
    let mut fragmenter = FrameFragmenter::new(0u8);
    let mut reassembler = FrameReassembler::new();

    let bodies: Vec<Bytes> = (0..4u8).map(|i| Bytes::from(vec![i; 1024])).collect();
    for body in &bodies {
        for pkt in fragmenter.fragment(meta(), body.clone(), tether_protocol::MAX_DATAGRAM_PAYLOAD)
        {
            host.send_datagram(&Datagram::Video(pkt)).unwrap();
        }
    }

    // Receive + reassemble.
    let mut bodies_in = Vec::new();
    while bodies_in.len() < bodies.len() {
        let dgram = tokio::time::timeout(std::time::Duration::from_secs(2), client.recv_datagram())
            .await
            .unwrap()
            .unwrap();
        let Datagram::Video(pkt) = dgram else {
            panic!()
        };
        if let Some(f) = reassembler.handle(pkt) {
            bodies_in.push(f.body);
        }
    }
    assert_eq!(bodies_in, bodies);

    // Now run those reassembled bodies through the decoder worker:
    // each `process_job` decodes one frame and pushes it into
    // LatestFrame. The four `Frames` outcomes are scripted to emit a
    // single solid frame per submit.
    let outcomes: Vec<FakeOutcome> = (0..4u8)
        .map(|i| FakeOutcome::Frames(vec![solid_frame(16, 16, i)]))
        .collect();
    let (mut worker, frames, idr_calls) = make_worker_with(outcomes);

    let mut render_drops_total = 0u32;
    for (i, body) in bodies_in.into_iter().enumerate() {
        let job = tether_decode::DecodeJob {
            body,
            host_in_client_clock: MonoNanos::now(),
            keyframe: i == 0,
        };
        let c = worker.process_job(job, MonoNanos::now());
        assert!(!c.decode_err);
        assert!(!c.soft_failure);
        render_drops_total += c.render_drops;
    }

    // No IDR requests on the happy path.
    assert_eq!(idr_calls.load(Ordering::Relaxed), 0);
    // The first 3 frames each displace the previous; the last one
    // stays in the slot.
    assert_eq!(render_drops_total, 3, "3 displacements over 4 frames");
    let last = frames.take().expect("frame in slot");
    match last {
        RenderFrame::Cpu(c) => assert_eq!(c.y[0], 3, "last frame's luma wins"),
        _ => panic!("expected Cpu"),
    }
}

#[tokio::test]
async fn render_drops_accounting_is_exact() {
    // Push 10 frames through the worker, draining LatestFrame after
    // every 3rd. The render_drop count from process_job is "would
    // have been a drop if no one was watching"; summed across all
    // jobs it must equal `produced - consumed - 1` (the -1 because
    // the very last frame stays in the slot).
    //
    // This is a strict accounting check on the LatestFrame
    // displacement counter, not a concurrency test — the
    // `concurrent_producer_consumer_*` test below covers the racy
    // production scenario.
    let n = 10u32;
    let drain_every = 3u32;

    let outcomes: Vec<FakeOutcome> = (0..n)
        .map(|i| FakeOutcome::Frames(vec![solid_frame(8, 8, (i & 0xff) as u8)]))
        .collect();
    let (mut worker, frames, _idr_calls) = make_worker_with(outcomes);

    let observed_drops = Arc::new(AtomicU64::new(0));
    let mut produced = 0u32;
    let mut consumed = 0u32;
    for i in 0..n {
        let job = tether_decode::DecodeJob {
            body: Bytes::from(vec![0u8; 32]),
            host_in_client_clock: MonoNanos::now(),
            keyframe: i == 0,
        };
        let c = worker.process_job(job, MonoNanos::now());
        produced += 1;
        observed_drops.fetch_add(u64::from(c.render_drops), Ordering::Relaxed);
        if (i + 1) % drain_every == 0 {
            // Consumer pulls the latest frame, leaving the slot empty.
            if frames.take().is_some() {
                consumed += 1;
            }
        }
    }
    // Drain the final frame so the slot is empty at end-of-test.
    let final_in_slot = if frames.take().is_some() { 1 } else { 0 };

    // Invariant: produced = consumed + final_in_slot + drops.
    let drops = observed_drops.load(Ordering::Relaxed) as u32;
    assert_eq!(
        produced,
        consumed + final_in_slot + drops,
        "produced ({produced}) = consumed ({consumed}) + slot ({final_in_slot}) + drops ({drops})"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_producer_consumer_under_load_preserves_invariant() {
    // Race a real producer thread against a real consumer task on the
    // same LatestFrame. Asserts the accounting invariant
    // `produced = consumed + final_slot + drops` holds under
    // concurrent set/take on the slot, not just under the synchronous
    // arithmetic the previous test covers.
    let frames = LatestFrame::new();
    let produced = 200u32;
    let drops_seen = Arc::new(AtomicU64::new(0));
    let consumed = Arc::new(AtomicU32::new(0));

    // Producer: pushes 200 frames as fast as it can.
    let producer_frames = frames.clone();
    let drops_seen_p = Arc::clone(&drops_seen);
    let producer = tokio::task::spawn_blocking(move || {
        for i in 0..produced {
            let displaced =
                producer_frames.set(tether_render::Frame::Cpu(tether_render::CpuFrame {
                    width: 4,
                    height: 4,
                    y: vec![(i & 0xff) as u8; 16],
                    uv: vec![0x80; 8],
                    t_capture_client_clock: None,
                }));
            if displaced.is_some() {
                drops_seen_p.fetch_add(1, Ordering::Relaxed);
            }
        }
    });

    // Consumer: drains opportunistically until the producer is done
    // and the slot is empty.
    let consumer_frames = frames.clone();
    let consumed_c = Arc::clone(&consumed);
    let consumer = tokio::task::spawn_blocking(move || {
        let mut idle_polls = 0u32;
        // Give the producer a head start by polling until idle
        // settles to "many empty reads in a row" — guarantees the
        // producer has finished pushing.
        while idle_polls < 100 {
            if consumer_frames.take().is_some() {
                consumed_c.fetch_add(1, Ordering::Relaxed);
                idle_polls = 0;
            } else {
                idle_polls += 1;
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        }
    });

    producer.await.unwrap();
    consumer.await.unwrap();

    let drops = drops_seen.load(Ordering::Relaxed) as u32;
    let consumed_n = consumed.load(Ordering::Relaxed);
    // After consumer drained to idle, slot must be empty.
    assert!(
        frames.take().is_none(),
        "consumer should have drained the slot"
    );
    assert_eq!(
        produced,
        consumed_n + drops,
        "produced ({produced}) = consumed ({consumed_n}) + drops ({drops}); \
         no frame should be silently lost between producer and consumer"
    );
}
