//! Loopback test for `InputChannel`: 100 events sent through the
//! in-memory duplex pair must arrive in order on the other side.

// Test event ids derived from small loop indices; cast is in range.
#![allow(clippy::cast_possible_truncation)]

use tether_protocol::input::{HidUsage, InputEvent, InputEventKind, Modifiers};
use tether_protocol::MonoNanos;
use tether_transport::test_support::input_duplex_pair;
use tether_transport::InputChannel;

fn make_event(event_id: u64) -> InputEvent {
    InputEvent {
        event_id,
        t_client: MonoNanos::ZERO,
        device_id: 0,
        kind: InputEventKind::KeyDown {
            key: HidUsage(0x07_0004 + (event_id as u32)),
            modifiers: Modifiers::default(),
        },
    }
}

#[tokio::test]
async fn one_hundred_events_round_trip_in_order() {
    let (host, client) = input_duplex_pair();

    let sender = tokio::spawn(async move {
        for i in 0..100u64 {
            client.send_input(&make_event(i)).await.unwrap();
        }
    });

    for expected in 0..100u64 {
        let evt = tokio::time::timeout(std::time::Duration::from_secs(2), host.recv_input())
            .await
            .expect("timeout waiting for event")
            .expect("recv_input failed");
        assert_eq!(evt.event_id, expected, "events arrive in send order");
    }

    sender.await.unwrap();
}
