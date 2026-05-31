use std::net::Ipv4Addr;

use tether_protocol::{
    control::ControlMessage,
    cursor::{ClientCursorPacket, HostCursorPacket},
    input::{InputEvent, InputEventKind, Modifiers, MouseButton},
    video::{HostFrameTiming, InputEchoBatch, VideoFrameMeta, VideoFrameMetaEnvelope, VideoPacket},
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

        // Receive a host-cursor datagram (host → client direction).
        match conn.recv_datagram().await? {
            Datagram::HostCursor(HostCursorPacket::Position { x, y, .. }) => {
                assert_eq!(x, 100);
                assert_eq!(y, 200);
            }
            other => panic!("expected host cursor Position, got {other:?}"),
        }

        // Receive a client-cursor datagram (client → host direction).
        match conn.recv_datagram().await? {
            Datagram::ClientCursor(c) => {
                assert_eq!(c.seq, 1);
                assert!((c.x - 0.5).abs() < 1e-6);
                assert!(c.visible);
            }
            other => panic!("expected client cursor, got {other:?}"),
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
        display: 0,
        stream_epoch: 0,
        frame_seq: 7,
        fragment_count: 1,
        meta: VideoFrameMetaEnvelope::V1(VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: true,
            input_echo: InputEchoBatch::default(),
            dimensions: (320, 240),
        }),
        payload: bytes::Bytes::from(vec![0u8; 100]),
    };
    conn.send_datagram(&Datagram::Video(video))?;

    let host_cursor = HostCursorPacket::Position {
        t_capture: MonoNanos::now(),
        x: 100,
        y: 200,
        visible: true,
    };
    conn.send_datagram(&Datagram::HostCursor(host_cursor))?;

    let client_cursor = ClientCursorPacket {
        seq: 1,
        display_idx: 0,
        x: 0.5,
        y: 0.5,
        visible: true,
        t_client: MonoNanos::now(),
    };
    conn.send_datagram(&Datagram::ClientCursor(client_cursor))?;

    let ctl = conn.recv_control().await?;
    assert!(
        matches!(ctl, ControlMessage::ForceIdr),
        "expected ForceIdr, got {ctl:?}",
    );

    let evt = InputEvent {
        event_id: 42,
        t_client: MonoNanos::now(),
        device_id: 0,
        kind: InputEventKind::MouseButton {
            button: MouseButton::Left,
            pressed: true,
            modifiers: Modifiers::default(),
        },
    };
    conn.send_input(&evt).await?;

    server_task.await??;

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mutual_tls_exposes_peer_fingerprint_and_matching_exporter() -> anyhow::Result<()> {
    // With mutual TLS the host must see the client's cert fingerprint (the
    // identity it will check against the paired-clients allowlist), and both
    // ends must derive the *same* TLS exporter for the same label/context —
    // the channel binding the pairing key-confirmation rides on. A mismatch
    // here would silently break MITM resistance, so pin both.
    let server = Server::bind((Ipv4Addr::LOCALHOST, 0).into()).await?;
    let server_addr = server.local_addr()?;
    let host_fingerprint = server.fingerprint();

    let client = Client::new()?;
    let client_fingerprint = client.fingerprint();

    let server_task = tokio::spawn(async move {
        let conn = server.accept().await.expect("server closed")?;
        let exporter = conn.export_keying_material(b"tether test label", b"ctx", 32)?;
        let peer_fp = conn.peer_cert_fingerprint();
        // Keep the connection alive until the client has read its own
        // exporter, then hand both observations back.
        conn.send_control(&ControlMessage::ForceIdr).await?;
        let _ = conn.recv_control().await?;
        anyhow::Ok((exporter, peer_fp))
    });

    let conn = client
        .connect(server_addr, "tether-host", host_fingerprint)
        .await?;
    let client_exporter = conn.export_keying_material(b"tether test label", b"ctx", 32)?;
    // The host's cert fingerprint, observed from the client side.
    let observed_host_fp = conn.peer_cert_fingerprint();
    let _ = conn.recv_control().await?;
    conn.send_control(&ControlMessage::ForceIdr).await?;

    let (host_exporter, observed_client_fp) = server_task.await??;

    assert_eq!(
        host_exporter, client_exporter,
        "TLS exporter must agree on both ends of the same connection"
    );
    assert_eq!(
        observed_client_fp,
        Some(client_fingerprint),
        "host must observe the client's cert fingerprint"
    );
    assert_eq!(
        observed_host_fp,
        Some(host_fingerprint),
        "client must observe the host's cert fingerprint"
    );
    // A different label must yield a different exporter (binding is real).
    let other = conn.export_keying_material(b"different label", b"ctx", 32)?;
    assert_ne!(other, client_exporter);
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
async fn input_stream_wrong_role_errors() -> anyhow::Result<()> {
    let server = Server::bind((Ipv4Addr::LOCALHOST, 0).into()).await?;
    let server_addr = server.local_addr()?;
    let fingerprint = server.fingerprint();

    let server_task = tokio::spawn(async move {
        let conn = server.accept().await.expect("server closed")?;
        // Host opens the input stream as Recv. Calling send_input on it
        // must surface the role mismatch as a typed error rather than
        // silently misbehaving.
        let evt = InputEvent {
            event_id: 1,
            t_client: MonoNanos::now(),
            device_id: 0,
            kind: InputEventKind::MouseButton {
                button: MouseButton::Left,
                pressed: true,
                modifiers: Modifiers::default(),
            },
        };
        let result = conn.send_input(&evt).await;
        assert!(
            matches!(
                result,
                Err(tether_transport::TransportError::InputStreamWrongRole)
            ),
            "expected InputStreamWrongRole, got {result:?}"
        );
        anyhow::Ok(())
    });

    let client = Client::new()?;
    let conn = client
        .connect(server_addr, "tether-host", fingerprint)
        .await?;

    // Client opens the input stream as Send. recv_input must error
    // immediately, not block waiting for data that can never arrive.
    let result = conn.recv_input().await;
    assert!(
        matches!(
            result,
            Err(tether_transport::TransportError::InputStreamWrongRole)
        ),
        "expected InputStreamWrongRole, got {result:?}"
    );

    server_task.await??;
    Ok(())
}

// Note: receive-side `MAX_FRAMED_MESSAGE` cap is exercised at the
// codec layer via `tether_protocol::decode::<T>` — bincode rejects a
// length-prefix mismatch when the body doesn't match the schema.
// Direct test of the length-prefix check inside `read_framed` would
// require either a test-only raw write hook on Connection or a unit
// test inside connection.rs against a quinn SendStream pair. The
// public send_video_keyframe path below exercises the same code
// inside read_framed_with_max via MAX_VIDEO_STREAM_MESSAGE.

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
        display: 0,
        stream_epoch: 0,
        frame_seq: 0,
        fragment_count: 1,
        meta: VideoFrameMetaEnvelope::V1(VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: false,
            input_echo: InputEchoBatch::default(),
            dimensions: (320, 240),
        }),
        payload: bytes::Bytes::from(vec![0u8; 4096]),
    };
    let err = conn.send_datagram(&Datagram::Video(oversized));
    assert!(
        matches!(
            err,
            Err(tether_transport::TransportError::FrameTooLarge { .. })
        ),
        "expected FrameTooLarge, got {err:?}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn video_keyframe_stream_roundtrips() -> anyhow::Result<()> {
    // Reliable-IDR path: host opens a uni stream, writes one
    // length-framed VideoPacket::First with fragment_count=1, and
    // finishes. Client accepts the stream and reads the same packet
    // back. Exercises send_video_keyframe + accept_video_keyframe and
    // the underlying write_framed_with_max / read_framed_with_max at
    // the larger video-stream cap.
    let server = Server::bind((Ipv4Addr::LOCALHOST, 0).into()).await?;
    let server_addr = server.local_addr()?;
    let fingerprint = server.fingerprint();

    // 200 KiB body — comfortably inside MAX_VIDEO_STREAM_MESSAGE (2 MiB)
    // and large enough that the underlying QUIC stream actually has to
    // segment, exercising the partial-read path inside read_exact.
    let body: Vec<u8> = (0..200_000).map(|i| (i & 0xff) as u8).collect();
    let body_for_server = body.clone();

    let server_task = tokio::spawn(async move {
        let conn = server.accept().await.expect("server closed")?;
        let packet = VideoPacket::First {
            display: 0,
            stream_epoch: 7,
            frame_seq: 42,
            fragment_count: 1,
            meta: VideoFrameMetaEnvelope::V1(VideoFrameMeta {
                timing: HostFrameTiming::default(),
                keyframe: true,
                input_echo: InputEchoBatch::default(),
                dimensions: (1920, 1080),
            }),
            payload: bytes::Bytes::from(body_for_server),
        };
        conn.send_video_keyframe(&packet).await?;
        // Wait for the client to confirm receipt by sending a
        // control message. Otherwise the conn drops here and quinn
        // closes the connection before the keyframe stream's
        // fin/data has been flushed to the wire.
        let _ = conn.recv_control().await?;
        anyhow::Ok(())
    });

    let client = Client::new()?;
    let conn = client
        .connect(server_addr, "tether-host", fingerprint)
        .await?;
    let received = conn.accept_video_keyframe().await?;
    match received {
        VideoPacket::First {
            stream_epoch,
            frame_seq,
            fragment_count,
            payload,
            ..
        } => {
            assert_eq!(stream_epoch, 7);
            assert_eq!(frame_seq, 42);
            assert_eq!(fragment_count, 1);
            assert_eq!(payload.as_ref(), body.as_slice());
        }
        other => panic!("expected First, got {other:?}"),
    }
    // Tell the server we got it so it can exit cleanly.
    conn.send_control(&ControlMessage::ForceIdr).await?;
    server_task.await??;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversize_video_keyframe_is_rejected_on_send() -> anyhow::Result<()> {
    // The send-side check inside write_framed_with_max guards against
    // accidentally building a keyframe larger than the receive-side
    // cap. Test it locally — exceeds MAX_VIDEO_STREAM_MESSAGE (2 MiB).
    let server = Server::bind((Ipv4Addr::LOCALHOST, 0).into()).await?;
    let server_addr = server.local_addr()?;
    let fingerprint = server.fingerprint();

    // Hold the server-side connection alive while the client builds
    // and submits its (rejected) packet. The reject happens locally
    // inside write_framed_with_max — the connection itself stays
    // healthy — but quinn still needs the server's conn alive for
    // open_uni to succeed in the first place.
    let server_task = tokio::spawn(async move {
        let conn = server.accept().await.expect("server closed")?;
        // Hold the connection until the client tells us it's done.
        let _ = conn.recv_control().await?;
        anyhow::Ok(())
    });

    let client = Client::new()?;
    let conn = client
        .connect(server_addr, "tether-host", fingerprint)
        .await?;

    let oversized = VideoPacket::First {
        display: 0,
        stream_epoch: 0,
        frame_seq: 0,
        fragment_count: 1,
        meta: VideoFrameMetaEnvelope::V1(VideoFrameMeta {
            timing: HostFrameTiming::default(),
            keyframe: true,
            input_echo: InputEchoBatch::default(),
            dimensions: (3840, 2160),
        }),
        payload: bytes::Bytes::from(vec![0u8; tether_transport::MAX_VIDEO_STREAM_MESSAGE + 1]),
    };
    let err = conn.send_video_keyframe(&oversized).await;
    assert!(
        matches!(
            err,
            Err(tether_transport::TransportError::FrameTooLarge { .. })
        ),
        "expected FrameTooLarge, got {err:?}"
    );
    conn.send_control(&ControlMessage::ForceIdr).await?;
    server_task.await??;
    Ok(())
}
