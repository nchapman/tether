//! Loopback for the unified video datagram path (IDR keyframes + P-frames).
//!
//! Drives the production `FrameFragmenter` → `DuplexVideoChannel` →
//! `FrameReassembler` chain in the lossless case and asserts every
//! body byte arrives intact.

use bytes::Bytes;
use tether_protocol::video::{FrameFragmenter, FrameReassembler, HostFrameTiming, VideoFrameMeta};
use tether_protocol::MAX_DATAGRAM_PAYLOAD;
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
        for pkt in fragmenter.fragment(meta(false), body.clone(), MAX_DATAGRAM_PAYLOAD) {
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
async fn idr_keyframe_round_trips_through_datagram_channel() {
    // IDRs ride the same FEC'd datagram channel as P-frames — fragmented,
    // sent as datagrams, reassembled. Asserts a multi-shard keyframe body
    // arrives intact over the lossless channel.
    let (host, client) = video_duplex_pair();
    let mut fragmenter = FrameFragmenter::new_with_fec(0, 20);
    let mut reassembler = FrameReassembler::new();

    let body: Bytes = vec![0xa5u8; 16 * 1024].into();
    let packets = fragmenter.fragment(meta(true), body.clone(), MAX_DATAGRAM_PAYLOAD);
    assert!(packets.len() > 1, "a 16 KB IDR must span multiple shards");
    for pkt in packets {
        host.send_datagram(&Datagram::Video(pkt)).unwrap();
    }

    let mut got = None;
    while got.is_none() {
        let dgram = tokio::time::timeout(std::time::Duration::from_secs(2), client.recv_datagram())
            .await
            .expect("timeout")
            .unwrap();
        let Datagram::Video(pkt) = dgram else {
            panic!("expected Video datagram, got {dgram:?}");
        };
        got = reassembler.handle(pkt);
    }
    let frame = got.unwrap();
    assert!(
        frame.meta.keyframe,
        "reassembled frame must be the keyframe"
    );
    assert_eq!(frame.body, body, "IDR body should round-trip byte-equal");
}
