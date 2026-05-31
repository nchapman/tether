//! Smoke test for cross-epoch fragment independence at the
//! integration level. Unit coverage for the same invariant lives
//! alongside `FrameFragmenter`/`FrameReassembler` in
//! `tether-protocol/src/lib.rs`; this test exercises the same
//! behaviour through the production duplex video channel so the
//! trait + serializer + decode chain is also covered.
//!
//! What we assert: after `bump_epoch`, fragments under the new epoch
//! reassemble independently of any in-flight fragments under the old
//! epoch — bodies never fuse across epochs even when `frame_seq`
//! resets to 0 in the new epoch.

use bytes::Bytes;
use tether_protocol::video::{FrameFragmenter, FrameReassembler, HostFrameTiming, VideoFrameMeta};
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

#[tokio::test]
async fn frames_across_epoch_bump_reassemble_independently() {
    let (host, client) = video_duplex_pair();
    let mut fragmenter = FrameFragmenter::new(0);
    let mut reassembler = FrameReassembler::new();

    // Send a complete frame in epoch 0 with a distinctive body.
    let epoch0_body = Bytes::from(vec![0xa5u8; 4096]);
    for pkt in fragmenter.fragment(meta(), epoch0_body.clone()) {
        host.send_datagram(&Datagram::Video(pkt)).unwrap();
    }

    // Bump epoch and send another complete frame with a different body.
    // frame_seq resets to 0 on the new epoch; the two frames share
    // (display=0, frame_seq=0) and would fuse if the reassembler keyed
    // only on those.
    fragmenter.bump_epoch();
    assert_eq!(fragmenter.stream_epoch(), 1);
    let epoch1_body = Bytes::from(vec![0x5au8; 4096]);
    for pkt in fragmenter.fragment(meta(), epoch1_body.clone()) {
        host.send_datagram(&Datagram::Video(pkt)).unwrap();
    }

    let mut completed = Vec::new();
    while completed.len() < 2 {
        let dgram = tokio::time::timeout(std::time::Duration::from_secs(1), client.recv_datagram())
            .await
            .expect("timeout")
            .unwrap();
        let Datagram::Video(pkt) = dgram else {
            panic!()
        };
        if let Some(f) = reassembler.handle(pkt) {
            completed.push((f.stream_epoch, f.body));
        }
    }

    // Both frames reassemble; bodies stay distinct.
    let epochs: Vec<u32> = completed.iter().map(|(e, _)| *e).collect();
    assert!(epochs.contains(&0));
    assert!(epochs.contains(&1));
    let body_e0 = &completed.iter().find(|(e, _)| *e == 0).unwrap().1;
    let body_e1 = &completed.iter().find(|(e, _)| *e == 1).unwrap().1;
    assert_eq!(body_e0, &epoch0_body, "epoch 0 body intact");
    assert_eq!(body_e1, &epoch1_body, "epoch 1 body intact");
    assert_ne!(body_e0[0], body_e1[0], "bodies are distinct (no fusion)");
}
