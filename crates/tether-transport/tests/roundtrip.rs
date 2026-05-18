use std::net::Ipv4Addr;

use tether_protocol::{
    control::ControlMessage,
    cursor::CursorPacket,
    input::{InputEvent, InputEventKind, MouseButton},
    video::{HostFrameTiming, InputEchoBatch, VideoFrameMeta, VideoPacket},
    MonoNanos,
};
use tether_transport::{Client, Datagram, Server};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn roundtrip_datagrams_control_input() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_env_filter("tether_transport=trace,quinn=warn")
        .try_init();

    let server = Server::bind((Ipv4Addr::LOCALHOST, 0).into()).await?;
    let server_addr = server.local_addr()?;
    let fingerprint = server.fingerprint();

    let server_task = tokio::spawn(async move {
        let conn = server.accept().await.expect("server closed")?;

        // Receive a video datagram
        match conn.recv_datagram().await? {
            Datagram::Video(VideoPacket::First { frame_seq, .. }) => {
                assert_eq!(frame_seq, 7);
            }
            other => panic!("expected video First, got {other:?}"),
        }

        // Receive a cursor datagram
        match conn.recv_datagram().await? {
            Datagram::Cursor(CursorPacket::Position { x, y, .. }) => {
                assert_eq!(x, 100);
                assert_eq!(y, 200);
            }
            other => panic!("expected cursor Position, got {other:?}"),
        }

        // Send a ForceIdr back on the control stream
        conn.send_control(&ControlMessage::ForceIdr).await?;

        // Receive an input event
        let evt = conn.recv_input().await?;
        assert_eq!(evt.event_id, 42);

        anyhow::Ok(())
    });

    let client = Client::new()?;
    let conn = client
        .connect(server_addr, "tether-host", fingerprint)
        .await?;

    let video = VideoPacket::First {
        stream_epoch: 0,
        frame_seq: 7,
        fragment_count: 1,
        meta: VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: true,
            input_echo: InputEchoBatch::default(),
        },
        payload: vec![0u8; 100],
    };
    conn.send_datagram(&Datagram::Video(video))?;

    let cursor = CursorPacket::Position {
        t_capture: MonoNanos::now(),
        x: 100,
        y: 200,
        visible: true,
    };
    conn.send_datagram(&Datagram::Cursor(cursor))?;

    let ctl = conn.recv_control().await?;
    assert!(
        matches!(ctl, ControlMessage::ForceIdr),
        "expected ForceIdr, got {ctl:?}",
    );

    let evt = InputEvent {
        event_id: 42,
        t_client: MonoNanos::now(),
        kind: InputEventKind::MouseButton {
            button: MouseButton::Left,
            pressed: true,
        },
    };
    conn.send_input(&evt).await?;

    server_task.await??;

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_fingerprint_rejects_wrong_cert() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    let server = Server::bind((Ipv4Addr::LOCALHOST, 0).into()).await?;
    let server_addr = server.local_addr()?;

    // Drive the server's accept loop in the background so the handshake
    // actually progresses to the point of failing.
    tokio::spawn(async move {
        let _ = server.accept().await;
    });

    let client = Client::new()?;
    let wrong = [0u8; 32];
    let result = client.connect(server_addr, "tether-host", wrong).await;
    assert!(
        result.is_err(),
        "connection should have failed but got {result:?}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversize_datagram_is_rejected_locally() -> anyhow::Result<()> {
    let server = Server::bind((Ipv4Addr::LOCALHOST, 0).into()).await?;
    let server_addr = server.local_addr()?;
    let fingerprint = server.fingerprint();

    tokio::spawn(async move {
        let _ = server.accept().await;
    });

    let client = Client::new()?;
    let conn = client
        .connect(server_addr, "tether-host", fingerprint)
        .await?;

    // Construct a video packet whose encoded form exceeds MAX_DATAGRAM_PAYLOAD.
    let oversized = VideoPacket::First {
        stream_epoch: 0,
        frame_seq: 0,
        fragment_count: 1,
        meta: VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
        },
        payload: vec![0u8; 4096],
    };
    let err = conn.send_datagram(&Datagram::Video(oversized));
    assert!(
        matches!(err, Err(tether_transport::TransportError::FrameTooLarge { .. })),
        "expected FrameTooLarge, got {err:?}"
    );
    Ok(())
}
