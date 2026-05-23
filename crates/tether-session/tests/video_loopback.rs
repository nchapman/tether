//! Loopback for the video datagram + IDR uni-stream paths.
//!
//! Drives the production `FrameFragmenter` → `DuplexVideoChannel` →
//! `FrameReassembler` chain in the lossless case and asserts every
//! body byte arrives intact.

use bytes::Bytes;
use tether_protocol::video::{
    FrameFragmenter, FrameReassembler, HostFrameTiming, VideoFrameMeta, VideoPacket,
};
use tether_transport::test_support::video_duplex_pair;
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
async fn n_pframes_round_trip_through_duplex_video_channel() {
    let (host, client) = video_duplex_pair();
    let mut fragmenter = FrameFragmenter::new(0);
    let mut reassembler = FrameReassembler::new();

    let bodies: Vec<Bytes> = (0..16u8)
        .map(|i| {
            let mut v = vec![0u8; 4096];
            v.fill(i);
            Bytes::from(v)
        })
        .collect();

    for body in &bodies {
        for pkt in fragmenter.fragment(meta(false), body.clone()) {
            host.send_datagram(&Datagram::Video(pkt)).unwrap();
        }
    }

    let mut received = Vec::new();
    while received.len() < bodies.len() {
        let dgram = tokio::time::timeout(std::time::Duration::from_secs(2), client.recv_datagram())
            .await
            .expect("timeout")
            .unwrap();
        let Datagram::Video(pkt) = dgram else {
            panic!("expected Video datagram, got {dgram:?}");
        };
        if let Some(frame) = reassembler.handle(pkt) {
            received.push(frame.body);
        }
    }

    assert_eq!(received, bodies, "all bodies should round-trip byte-equal");
    assert_eq!(
        host.datagrams_dropped(),
        0,
        "no drops on the lossless channel"
    );
    assert_eq!(
        reassembler.loss_counters(),
        (0, 0),
        "no loss observed on the lossless channel"
    );
}

#[tokio::test]
async fn idr_uni_stream_delivers_keyframe_intact() {
    let (host, client) = video_duplex_pair();
    let mut fragmenter = FrameFragmenter::new(0);

    let body: Bytes = vec![0xa5u8; 16 * 1024].into();
    let pkt = fragmenter.single_packet(meta(true), body.clone());
    host.send_video_keyframe(&pkt).await.unwrap();

    let received = tokio::time::timeout(std::time::Duration::from_secs(2), client.accept_video_keyframe())
        .await
        .expect("timeout")
        .unwrap();
    match received {
        VideoPacket::First {
            payload,
            fragment_count,
            ..
        } => {
            assert_eq!(fragment_count, 1, "IDRs ride one packet");
            assert_eq!(payload, body, "IDR body should round-trip byte-equal");
        }
        other => panic!("expected First, got {other:?}"),
    }
}
