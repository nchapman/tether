//! End-to-end session loopback. Wires `HostSession::accept` and
//! `ClientSession::connect` together through a `DuplexControlChannel`
//! pair so the handshake's clock-sync probe, profile negotiation,
//! typed rejection flow, and bit-depth validation get exercised
//! in CI without a real QUIC pair (or any UDP socket at all).
//!
//! These tests close the "no end-to-end session loopback" gap that
//! used to mean session-level bugs only surfaced in live sessions.
//! The phantom-latency bug (slow build closure biasing the
//! `ClockSync` offset by ~50ms) lived entirely in this gap before
//! the trait abstraction made loopback testable.

use std::sync::Arc;

use tether_protocol::control::{
    ChromaSubsampling, ClientDisplayMetrics, ClientHello, CodecKind, ControlMessage,
    DisplayDescriptor, DisplayId, DisplayMode, DisplayModeStatus, NegotiatedVideo, PixelFormat,
    RequestId, ServerHello, VideoColorSpec, VideoProfile, VideoStreamId, Viewport,
    CLOCK_SYNC_PROBE_SAMPLES,
};
use tether_protocol::MonoNanos;
use tether_session::{
    AcceptError, ClientSession, ClientSessionConfig, ConnectError, HostSession, HostSessionConfig,
};
use tether_transport::test_support::duplex_pair;
use tether_transport::ControlChannel;

/// Construct a paired host + client session config used by most
/// happy-path tests.
fn cfgs() -> (HostSessionConfig, ClientSessionConfig) {
    let host = HostSessionConfig {
        server_name: "test-host".to_string(),
        audio_config: None,
        displays: vec![test_display()],
    };
    let client = ClientSessionConfig {
        client_name: "test-client".to_string(),
        client_decode_profiles: vec![VideoProfile::HEVC_8BIT_420, VideoProfile::H264_8BIT_420],
        viewport: None,
    };
    (host, client)
}

fn test_display() -> DisplayDescriptor {
    let mode = DisplayMode::new(1280, 720, 60_000);
    DisplayDescriptor {
        id: DisplayId(0),
        name: "test".into(),
        scale_num: 1,
        scale_den: 1,
        primary: true,
        position: (0, 0),
        current_mode: mode,
        available_modes: vec![mode],
        can_set_mode: false,
    }
}

fn test_server_hello(profile: VideoProfile) -> ServerHello {
    ServerHello {
        server_name: "test-host".to_string(),
        video: NegotiatedVideo {
            stream_id: VideoStreamId(0),
            display_id: DisplayId(0),
            profile,
            pixel_format: PixelFormat::Nv12,
            color_space: VideoColorSpec::sdr_desktop(),
        },
        audio: None,
        displays: vec![test_display()],
        accepted_features: vec![],
    }
}

async fn answer_clock_probe(channel: &dyn ControlChannel) {
    for _ in 0..CLOCK_SYNC_PROBE_SAMPLES {
        match channel.recv_control().await.unwrap() {
            ControlMessage::ClockProbeRequest { t0_sender } => {
                channel
                    .send_control(&ControlMessage::ClockProbeResponse(
                        tether_protocol::control::ClockProbe {
                            t0_sender,
                            t1_receiver_recv: MonoNanos::now(),
                            t2_receiver_send: MonoNanos::now(),
                        },
                    ))
                    .await
                    .unwrap();
            }
            other => panic!("expected ClockProbeRequest, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn happy_path_handshake_completes_with_negotiated_profile() {
    let (host_chan, client_chan) = duplex_pair();
    let (host_cfg, client_cfg) = cfgs();

    let host_chan: Arc<dyn ControlChannel> = host_chan;
    let client_chan: Arc<dyn ControlChannel> = client_chan;
    let host_chan_for_probe = host_chan.clone();

    let host_task = tokio::spawn(async move {
        let session = HostSession::accept(host_chan, host_cfg, |client_caps| {
            // Trivial selector: prefer HEVC if mutually supported,
            // else H.264 — mirrors the production `pick_supported_profile`
            // preference order minus the tether-probe dep.
            [VideoProfile::HEVC_8BIT_420, VideoProfile::H264_8BIT_420]
                .into_iter()
                .find(|p| client_caps.contains(p))
        })
        .await?;
        answer_clock_probe(host_chan_for_probe.as_ref()).await;
        Ok::<_, AcceptError>(session)
    });
    let client_task =
        tokio::spawn(async move { ClientSession::connect(client_chan, client_cfg).await });

    let host = host_task.await.unwrap().unwrap();
    let client = client_task.await.unwrap().unwrap();

    assert_eq!(host.negotiated, VideoProfile::HEVC_8BIT_420);
    assert_eq!(client.negotiated, VideoProfile::HEVC_8BIT_420);

    // Loopback RTT is in-memory — should be tiny. Use a generous
    // bound so a slow CI runner doesn't flake.
    assert!(
        client.clock_sync.rtt_nanos < 200_000_000,
        "expected sub-200ms RTT on loopback, got {} ns",
        client.clock_sync.rtt_nanos
    );
    // Offset on loopback should be near zero — both stamps come from
    // the same monotonic clock. Anything more than ~10ms means the
    // handshake closure is doing too much work.
    assert!(
        client.clock_sync.offset_nanos.unsigned_abs() < 10_000_000,
        "expected sub-10ms clock offset on loopback, got {} ns",
        client.clock_sync.offset_nanos
    );
}

#[tokio::test]
async fn clock_probe_burst_rejects_noisy_high_rtt_sample() {
    let (host_chan, client_chan) = duplex_pair();
    let client_cfg = ClientSessionConfig {
        client_name: "test-client".to_string(),
        client_decode_profiles: vec![VideoProfile::H264_8BIT_420],
        viewport: None,
    };

    let host_chan_dyn: Arc<dyn ControlChannel> = host_chan;
    let client_chan_dyn: Arc<dyn ControlChannel> = client_chan;

    let host_task = tokio::spawn(async move {
        let _hello = host_chan_dyn.recv_client_hello().await.unwrap();
        host_chan_dyn
            .send_server_hello(test_server_hello(VideoProfile::H264_8BIT_420))
            .await
            .unwrap();

        for sample_idx in 0..CLOCK_SYNC_PROBE_SAMPLES {
            let t0_sender = match host_chan_dyn.recv_control().await.unwrap() {
                ControlMessage::ClockProbeRequest { t0_sender } => t0_sender,
                other => panic!("expected ClockProbeRequest, got {other:?}"),
            };
            let probe = if sample_idx == 0 {
                // Simulate the exact failure shape from issue #58: one queued
                // startup sample with an RTT-sized offset bias. The reducer
                // should reject it once cleaner samples arrive.
                tether_protocol::control::ClockProbe {
                    t0_sender,
                    t1_receiver_recv: MonoNanos(t0_sender.0.saturating_add(300_000_000)),
                    t2_receiver_send: t0_sender,
                }
            } else {
                // A near-zero-queue sample. In loopback, `t2` is effectively
                // the client's `t3`, so the offset and RTT should both be tiny.
                tether_protocol::control::ClockProbe {
                    t0_sender,
                    t1_receiver_recv: t0_sender,
                    t2_receiver_send: MonoNanos::now(),
                }
            };
            host_chan_dyn
                .send_control(&ControlMessage::ClockProbeResponse(probe))
                .await
                .unwrap();
        }
    });

    let client = ClientSession::connect(client_chan_dyn, client_cfg)
        .await
        .unwrap();
    host_task.await.unwrap();

    assert!(
        client.clock_sync.offset_nanos.unsigned_abs() < 5_000_000,
        "expected clean min-RTT sample, got offset={}ns rtt={}ns",
        client.clock_sync.offset_nanos,
        client.clock_sync.rtt_nanos
    );
    assert!(
        client.clock_sync.rtt_nanos < 5_000_000,
        "expected low-RTT sample, got offset={}ns rtt={}ns",
        client.clock_sync.offset_nanos,
        client.clock_sync.rtt_nanos
    );
}

#[tokio::test]
async fn no_mutual_profile_sends_typed_rejection_then_errors() {
    let (host_chan, client_chan) = duplex_pair();
    let (host_cfg, mut client_cfg) = cfgs();
    // Client advertises only Av1 (which the test selector ignores) —
    // forces no-match.
    client_cfg.client_decode_profiles = vec![VideoProfile {
        codec: CodecKind::Av1,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 8,
    }];

    let host_chan_dyn: Arc<dyn ControlChannel> = host_chan;
    let client_chan_dyn: Arc<dyn ControlChannel> = client_chan;

    let host_task = tokio::spawn(async move {
        HostSession::accept(host_chan_dyn, host_cfg, |client_caps| {
            // Selector only ever picks HEVC/H264 — the client's Av1
            // never matches.
            [VideoProfile::HEVC_8BIT_420, VideoProfile::H264_8BIT_420]
                .into_iter()
                .find(|p| client_caps.contains(p))
        })
        .await
    });

    let client_task = tokio::spawn(async move {
        let connect_err = ClientSession::connect(client_chan_dyn, client_cfg)
            .await
            .map(|_| ())
            .expect_err("expected HandshakeRejected");
        assert!(
            matches!(connect_err, ConnectError::HandshakeRejected { .. }),
            "expected HandshakeRejected, got: {connect_err:?}"
        );
    });

    let host_err = host_task
        .await
        .unwrap()
        .map(|_| ())
        .expect_err("expected no-match");
    assert!(
        matches!(host_err, AcceptError::NoProfileIntersection { .. }),
        "expected NoProfileIntersection, got: {host_err:?}"
    );
    client_task.await.unwrap();
}

#[tokio::test]
async fn double_send_server_hello_corrupts_the_stream() {
    // Pin the ordering invariant the `ControlChannel` trait documents
    // but doesn't enforce: `send_server_hello` is valid exactly once
    // per session, after `recv_client_hello`. Calling it twice puts a
    // second `ServerHello` frame on the wire after the client has
    // moved on to reading `ControlMessage`s — prost decode/conversion fails.
    //
    // Production orchestration goes through the `HostHandshake` typestate
    // wrapper, which makes this double-call uncompilable. This regression test
    // pins the lower-level `ControlChannel` escape hatch that tests and custom
    // harnesses can still call directly.
    let (host_chan, client_chan) = duplex_pair();
    let host_chan_dyn: Arc<dyn ControlChannel> = host_chan;

    // Client side: complete the handshake normally, then `recv_control`
    // — that read should see the (corrupt) second ServerHello and fail
    // protobuf/control conversion.
    let client_task = tokio::spawn(async move {
        let _ = client_chan
            .client_handshake(ClientHello {
                client_name: "test".into(),
                decode_profiles: vec![VideoProfile::H264_8BIT_420],
                initial_viewport: None,
                input_capabilities: tether_protocol::control::InputCapabilities::default(),
                requested_features: vec![],
            })
            .await
            .unwrap();
        // The second ServerHello sits on the stream; reading it as a
        // ControlMessage fails prost conversion.
        let second = client_chan.recv_control().await;
        assert!(
            second.is_err(),
            "a second send_server_hello on the same channel must not \
             decode as a ControlMessage; got: {second:?}"
        );
    });

    let _hello = host_chan_dyn.recv_client_hello().await.unwrap();
    let placeholder = test_server_hello(VideoProfile::H264_8BIT_420);
    host_chan_dyn
        .send_server_hello(placeholder.clone())
        .await
        .unwrap();
    // Misuse: call it again. The client task asserts the corruption
    // becomes visible on the next read.
    host_chan_dyn.send_server_hello(placeholder).await.unwrap();

    client_task.await.unwrap();
}

#[tokio::test]
async fn host_picks_unadvertised_profile_client_refuses() {
    let (host_chan, client_chan) = duplex_pair();
    let (host_cfg, mut client_cfg) = cfgs();
    // Client advertises only H264. The selector (closure below) is
    // adversarial: it ignores the client's list and picks HEVC anyway.
    client_cfg.client_decode_profiles = vec![VideoProfile::H264_8BIT_420];

    let host_chan_dyn: Arc<dyn ControlChannel> = host_chan;
    let client_chan_dyn: Arc<dyn ControlChannel> = client_chan;

    let host_task = tokio::spawn(async move {
        HostSession::accept(host_chan_dyn, host_cfg, |_client_caps| {
            // Buggy host: picks something the client didn't advertise.
            Some(VideoProfile::HEVC_8BIT_420)
        })
        .await
    });
    let client_task = tokio::spawn(async move {
        let err = ClientSession::connect(client_chan_dyn, client_cfg)
            .await
            .map(|_| ())
            .expect_err("client should refuse unadvertised profile");
        assert!(
            matches!(err, ConnectError::ProfileNotAdvertised { chosen, .. }
                if chosen == VideoProfile::HEVC_8BIT_420),
            "expected ProfileNotAdvertised(HEVC), got: {err:?}"
        );
    });

    // Host side reports success — it has no way to know the client
    // refused. Any later shutdown flows from the *client* back to the host
    // via the regular session-end path, not via HostSession::accept.
    let host_session = host_task.await.unwrap().unwrap();
    assert_eq!(host_session.negotiated, VideoProfile::HEVC_8BIT_420);
    client_task.await.unwrap();
    match host_session.channel.recv_control().await.unwrap() {
        ControlMessage::Goodbye { reason, code, .. } => {
            assert_eq!(code, tether_protocol::control::GoodbyeCode::ProtocolError);
            assert!(
                reason.contains("unadvertised video profile"),
                "unexpected goodbye reason: {reason}"
            );
        }
        other => panic!("expected client Goodbye after invalid profile, got {other:?}"),
    }
}

#[tokio::test]
async fn client_filters_unknown_bit_depth_from_host_advert() {
    let (host_chan, client_chan) = duplex_pair();
    let host_cfg = HostSessionConfig {
        server_name: "buggy-host".to_string(),
        audio_config: None,
        displays: vec![test_display()],
    };
    let client_cfg = ClientSessionConfig {
        client_name: "test-client".to_string(),
        client_decode_profiles: vec![VideoProfile::HEVC_8BIT_420],
        viewport: None,
    };

    let host_chan_dyn: Arc<dyn ControlChannel> = host_chan;
    let client_chan_dyn: Arc<dyn ControlChannel> = client_chan;

    let host_task = tokio::spawn(async move {
        HostSession::accept(host_chan_dyn, host_cfg, |_client_caps| {
            // Adversarial host picks a profile with a bit_depth this
            // build doesn't know — a hypothetical 12-bit future profile
            // or a malformed peer.
            Some(VideoProfile {
                codec: CodecKind::Hevc,
                chroma: ChromaSubsampling::Yuv420,
                bit_depth: 12,
            })
        })
        .await
    });
    let client_task = tokio::spawn(async move {
        let err = ClientSession::connect(client_chan_dyn, client_cfg)
            .await
            .map(|_| ())
            .expect_err("client should refuse unknown bit_depth");
        assert!(
            matches!(err, ConnectError::UnknownBitDepth(12, _)),
            "expected UnknownBitDepth(12, _), got: {err:?}"
        );
    });

    // HostSession itself succeeds because it sent a syntactically valid
    // ServerHello, but the client reports the protocol error on the control
    // stream before returning its local validation failure.
    let host = host_task.await.unwrap().unwrap();
    client_task.await.unwrap();
    match host.channel.recv_control().await.unwrap() {
        ControlMessage::Goodbye { reason, code, .. } => {
            assert_eq!(code, tether_protocol::control::GoodbyeCode::ProtocolError);
            assert!(
                reason.contains("unknown bit_depth 12"),
                "unexpected goodbye reason: {reason}"
            );
        }
        other => panic!("expected client Goodbye after unknown bit_depth, got {other:?}"),
    }
}

#[tokio::test]
async fn host_filters_unknown_bit_depths_keeps_known_ones() {
    let (host_chan, client_chan) = duplex_pair();
    let host_cfg = HostSessionConfig {
        server_name: "test-host".to_string(),
        audio_config: None,
        displays: vec![test_display()],
    };
    // Client advertises a 12-bit profile (unknown) and a real
    // 8-bit HEVC profile. The host should filter the unknown depth
    // out and still find a mutual 8-bit match.
    let client_cfg = ClientSessionConfig {
        client_name: "future-client".to_string(),
        client_decode_profiles: vec![
            VideoProfile {
                codec: CodecKind::Hevc,
                chroma: ChromaSubsampling::Yuv444,
                bit_depth: 12,
            },
            VideoProfile::HEVC_8BIT_420,
        ],
        viewport: None,
    };

    let host_chan_dyn: Arc<dyn ControlChannel> = host_chan;
    let client_chan_dyn: Arc<dyn ControlChannel> = client_chan;
    let host_chan_for_probe = host_chan_dyn.clone();

    let host_task = tokio::spawn(async move {
        let session = HostSession::accept(host_chan_dyn, host_cfg, |client_caps| {
            // Selector receives the filtered list — assert it does
            // NOT contain the 12-bit profile.
            assert!(
                !client_caps
                    .iter()
                    .any(|p| p.bit_depth == 12),
                "selector should not see 12-bit profile after host-side filter; saw: {client_caps:?}"
            );
            client_caps
                .iter()
                .find(|p| **p == VideoProfile::HEVC_8BIT_420)
                .copied()
        })
        .await?;
        answer_clock_probe(host_chan_for_probe.as_ref()).await;
        Ok::<_, AcceptError>(session)
    });
    let client_task =
        tokio::spawn(async move { ClientSession::connect(client_chan_dyn, client_cfg).await });

    let host = host_task.await.unwrap().unwrap();
    let client = client_task.await.unwrap().unwrap();
    assert_eq!(host.negotiated, VideoProfile::HEVC_8BIT_420);
    assert_eq!(client.negotiated, VideoProfile::HEVC_8BIT_420);
}

#[tokio::test]
async fn client_handshake_uses_typed_negotiated_video() {
    let (host_chan, client_chan) = duplex_pair();
    let client_cfg = ClientSessionConfig {
        client_name: "test-client".to_string(),
        client_decode_profiles: vec![VideoProfile::HEVC_8BIT_420],
        viewport: None,
    };

    let host_chan_dyn: Arc<dyn ControlChannel> = host_chan;
    let client_chan_dyn: Arc<dyn ControlChannel> = client_chan;

    let host_task = tokio::spawn(async move {
        let _hello = host_chan_dyn.recv_client_hello().await.unwrap();
        let server = test_server_hello(VideoProfile::HEVC_8BIT_420);
        host_chan_dyn.send_server_hello(server).await.unwrap();
        answer_clock_probe(host_chan_dyn.as_ref()).await;
        // Swallow the ForceIdr the client sends right after the
        // handshake completes.
        let _ = host_chan_dyn.recv_control().await.unwrap();
    });
    let client = ClientSession::connect(client_chan_dyn, client_cfg)
        .await
        .unwrap();
    host_task.await.unwrap();

    assert_eq!(client.negotiated, VideoProfile::HEVC_8BIT_420);
    assert_eq!(client.negotiated_video.profile, VideoProfile::HEVC_8BIT_420);
}

#[tokio::test]
async fn goodbye_during_clock_probe_aborts_connect() {
    let (host_chan, client_chan) = duplex_pair();
    let client_cfg = ClientSessionConfig {
        client_name: "test-client".to_string(),
        client_decode_profiles: vec![VideoProfile::H264_8BIT_420],
        viewport: None,
    };

    let host_chan_dyn: Arc<dyn ControlChannel> = host_chan;
    let client_chan_dyn: Arc<dyn ControlChannel> = client_chan;

    let host_task = tokio::spawn(async move {
        let _hello = host_chan_dyn.recv_client_hello().await.unwrap();
        host_chan_dyn
            .send_server_hello(test_server_hello(VideoProfile::H264_8BIT_420))
            .await
            .unwrap();
        match host_chan_dyn.recv_control().await.unwrap() {
            ControlMessage::ClockProbeRequest { .. } => {
                host_chan_dyn
                    .send_control(&ControlMessage::Goodbye {
                        reason: "probe denied".into(),
                        code: tether_protocol::control::GoodbyeCode::ProtocolError,
                        final_stats: None,
                    })
                    .await
                    .unwrap();
            }
            other => panic!("expected ClockProbeRequest, got {other:?}"),
        }
    });

    let err = ClientSession::connect(client_chan_dyn, client_cfg)
        .await
        .map(|_| ())
        .expect_err("Goodbye during clock probe should abort connect");
    host_task.await.unwrap();
    assert!(
        matches!(err, ConnectError::PeerGoodbyeDuringClockProbe { ref reason, .. } if reason == "probe denied"),
        "expected PeerGoodbyeDuringClockProbe, got {err:?}"
    );
}

#[tokio::test]
async fn stale_clock_probe_responses_abort_connect_after_budget() {
    let (host_chan, client_chan) = duplex_pair();
    let client_cfg = ClientSessionConfig {
        client_name: "test-client".to_string(),
        client_decode_profiles: vec![VideoProfile::H264_8BIT_420],
        viewport: None,
    };

    let host_chan_dyn: Arc<dyn ControlChannel> = host_chan;
    let client_chan_dyn: Arc<dyn ControlChannel> = client_chan;

    let host_task = tokio::spawn(async move {
        let _hello = host_chan_dyn.recv_client_hello().await.unwrap();
        host_chan_dyn
            .send_server_hello(test_server_hello(VideoProfile::H264_8BIT_420))
            .await
            .unwrap();
        for _ in 0..64 {
            if host_chan_dyn
                .send_control(&ControlMessage::ClockProbeResponse(
                    tether_protocol::control::ClockProbe {
                        t0_sender: MonoNanos::ZERO,
                        t1_receiver_recv: MonoNanos::ZERO,
                        t2_receiver_send: MonoNanos::ZERO,
                    },
                ))
                .await
                .is_err()
            {
                break;
            }
        }
    });

    let err = ClientSession::connect(client_chan_dyn, client_cfg)
        .await
        .map(|_| ())
        .expect_err("stale probe responses should not keep connect pending forever");
    host_task.await.unwrap();
    assert!(
        matches!(err, ConnectError::ClockProbeIgnoredMessageLimit { .. }),
        "expected ClockProbeIgnoredMessageLimit, got {err:?}"
    );
}

// Convenience: a malformed peer (truncated or garbage on the wire)
// shouldn't deadlock the test — if it ever did, this test would time
// out and surface that.
#[tokio::test]
async fn dropped_client_during_handshake_surfaces_as_transport_error() {
    let (host_chan, client_chan) = duplex_pair();
    let host_chan_dyn: Arc<dyn ControlChannel> = host_chan;
    drop(client_chan); // client dies before sending ClientHello

    let res = HostSession::accept(
        host_chan_dyn,
        HostSessionConfig {
            server_name: "test".into(),
            audio_config: None,
            displays: vec![test_display()],
        },
        |_| Some(VideoProfile::H264_8BIT_420),
    )
    .await;
    let err = res
        .map(|_| ())
        .expect_err("dropped client should produce transport error");
    assert!(
        matches!(err, AcceptError::Transport(_)),
        "expected Transport(_), got: {err:?}"
    );
}

#[tokio::test]
async fn initial_viewport_round_trips_via_client_hello() {
    // The client config's `viewport` populates `ClientHello::initial_viewport`,
    // and the host receives it on its session struct. Loopback covers
    // both sides — guards against a future refactor dropping the
    // field in either direction.
    let (host_chan, client_chan) = duplex_pair();
    let (host_cfg, mut client_cfg) = cfgs();
    client_cfg.viewport = Some(Viewport::new(1280, 720));

    let host_chan: Arc<dyn ControlChannel> = host_chan;
    let client_chan: Arc<dyn ControlChannel> = client_chan;
    let host_chan_for_probe = host_chan.clone();

    let host_task = tokio::spawn(async move {
        let session = HostSession::accept(host_chan, host_cfg, |client_caps| {
            client_caps.iter().copied().next()
        })
        .await
        .unwrap();
        answer_clock_probe(host_chan_for_probe.as_ref()).await;
        session
    });
    let client_task = tokio::spawn(async move {
        ClientSession::connect(client_chan, client_cfg)
            .await
            .unwrap()
    });

    let host = host_task.await.unwrap();
    let _client = client_task.await.unwrap();
    assert_eq!(
        host.client_hello.initial_viewport,
        Some(Viewport::new(1280, 720))
    );
}

#[tokio::test]
async fn mid_session_set_viewport_round_trips_on_control_stream() {
    // After the handshake, the client sending SetViewportHint
    // arrives on the host's recv_control. This is the wire half of
    // the production flow; the host binary's send-thread side
    // (encoder rebuild, force_idr raise) is covered by host-side
    // unit tests on the dim helper.
    let (host_chan, client_chan) = duplex_pair();
    let (host_cfg, client_cfg) = cfgs();

    let host_chan_for_session: Arc<dyn ControlChannel> = host_chan.clone();
    let client_chan_for_session: Arc<dyn ControlChannel> = client_chan.clone();
    let host_chan_for_probe = host_chan_for_session.clone();

    let host_task = tokio::spawn(async move {
        let session = HostSession::accept(host_chan_for_session, host_cfg, |client_caps| {
            client_caps.iter().copied().next()
        })
        .await
        .unwrap();
        answer_clock_probe(host_chan_for_probe.as_ref()).await;
        session
    });
    let client_task = tokio::spawn(async move {
        ClientSession::connect(client_chan_for_session, client_cfg)
            .await
            .unwrap()
    });
    let _host = host_task.await.unwrap();
    let _client = client_task.await.unwrap();

    // Now the post-handshake message.
    let host_chan_recv: Arc<dyn ControlChannel> = host_chan;
    let client_chan_send: Arc<dyn ControlChannel> = client_chan;
    let send_task = tokio::spawn(async move {
        client_chan_send
            .send_control(&ControlMessage::SetViewportHint {
                stream_id: VideoStreamId(0),
                viewport: Viewport::new(640, 480),
            })
            .await
            .unwrap();
    });
    let recv_task = tokio::spawn(async move {
        // The client implementation sends a ForceIdr immediately after
        // a successful handshake (so the host emits a fresh keyframe
        // for the new decoder); swallow it so the SetViewportHint
        // is the message we assert on.
        loop {
            match host_chan_recv.recv_control().await.unwrap() {
                ControlMessage::ForceIdr => continue,
                other => return other,
            }
        }
    });
    send_task.await.unwrap();
    let msg = recv_task.await.unwrap();
    match msg {
        ControlMessage::SetViewportHint {
            stream_id,
            viewport: v,
        } => {
            assert_eq!(stream_id, VideoStreamId(0));
            assert_eq!(v, Viewport::new(640, 480));
        }
        other => panic!("expected SetViewportHint, got {other:?}"),
    }
}

#[tokio::test]
async fn mid_session_client_display_metrics_round_trips_on_control_stream() {
    // ClientDisplayMetrics is display-mode-matching input, not a mode-change
    // request. Loopback pins that it rides the same post-handshake control
    // stream as SetViewportHint without being dropped by session plumbing.
    let (host_chan, client_chan) = duplex_pair();
    let (host_cfg, client_cfg) = cfgs();

    let host_chan_for_session: Arc<dyn ControlChannel> = host_chan.clone();
    let client_chan_for_session: Arc<dyn ControlChannel> = client_chan.clone();
    let host_chan_for_probe = host_chan_for_session.clone();

    let host_task = tokio::spawn(async move {
        let session = HostSession::accept(host_chan_for_session, host_cfg, |client_caps| {
            client_caps.iter().copied().next()
        })
        .await
        .unwrap();
        answer_clock_probe(host_chan_for_probe.as_ref()).await;
        session
    });
    let client_task = tokio::spawn(async move {
        ClientSession::connect(client_chan_for_session, client_cfg)
            .await
            .unwrap()
    });
    let _host = host_task.await.unwrap();
    let _client = client_task.await.unwrap();

    let metrics = ClientDisplayMetrics {
        display_id: 3,
        mode: DisplayMode::new(2560, 1440, 144_000),
        scale_num: 3,
        scale_den: 2,
        safe_area: None,
    };

    let host_chan_recv: Arc<dyn ControlChannel> = host_chan;
    let client_chan_send: Arc<dyn ControlChannel> = client_chan;
    let expected = metrics.clone();
    let send_task = tokio::spawn(async move {
        client_chan_send
            .send_control(&ControlMessage::ClientDisplayMetrics(metrics))
            .await
            .unwrap();
    });
    let recv_task = tokio::spawn(async move {
        loop {
            match host_chan_recv.recv_control().await.unwrap() {
                ControlMessage::ForceIdr => continue,
                other => return other,
            }
        }
    });
    send_task.await.unwrap();
    let msg = recv_task.await.unwrap();
    match msg {
        ControlMessage::ClientDisplayMetrics(metrics) => assert_eq!(metrics, expected),
        other => panic!("expected ClientDisplayMetrics, got {other:?}"),
    }
}

#[tokio::test]
async fn mid_session_client_stats_round_trips_on_control_stream() {
    let (host_chan, client_chan) = duplex_pair();
    let (host_cfg, client_cfg) = cfgs();

    let host_chan_for_session: Arc<dyn ControlChannel> = host_chan.clone();
    let client_chan_for_session: Arc<dyn ControlChannel> = client_chan.clone();
    let host_chan_for_probe = host_chan_for_session.clone();

    let host_task = tokio::spawn(async move {
        let session = HostSession::accept(host_chan_for_session, host_cfg, |client_caps| {
            client_caps.iter().copied().next()
        })
        .await
        .unwrap();
        answer_clock_probe(host_chan_for_probe.as_ref()).await;
        session
    });
    let client_task = tokio::spawn(async move {
        ClientSession::connect(client_chan_for_session, client_cfg)
            .await
            .unwrap()
    });
    let _host = host_task.await.unwrap();
    let _client = client_task.await.unwrap();

    client_chan
        .send_control(&ControlMessage::ClientStats {
            window_ms: 1001,
            frames_received: 62,
            incomplete_frames: 3,
            fragment_loss_events: 5,
            rtt_us: 7000,
            fec_recovered_frames: 7,
            fec_recovered_fragments: 11,
        })
        .await
        .unwrap();

    loop {
        match host_chan.recv_control().await.unwrap() {
            ControlMessage::ForceIdr => continue,
            ControlMessage::ClientStats {
                window_ms,
                frames_received,
                incomplete_frames,
                fragment_loss_events,
                rtt_us,
                fec_recovered_frames,
                fec_recovered_fragments,
            } => {
                assert_eq!(window_ms, 1001);
                assert_eq!(frames_received, 62);
                assert_eq!(incomplete_frames, 3);
                assert_eq!(fragment_loss_events, 5);
                assert_eq!(rtt_us, 7000);
                assert_eq!(fec_recovered_frames, 7);
                assert_eq!(fec_recovered_fragments, 11);
                break;
            }
            other => panic!("expected ClientStats, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn display_mode_request_returns_unsupported_until_platform_wired() {
    let (host_chan, client_chan) = duplex_pair();
    let (host_cfg, client_cfg) = cfgs();

    let host_chan_for_session: Arc<dyn ControlChannel> = host_chan.clone();
    let client_chan_for_session: Arc<dyn ControlChannel> = client_chan.clone();
    let host_chan_for_probe = host_chan_for_session.clone();

    let host_task = tokio::spawn(async move {
        let session = HostSession::accept(host_chan_for_session, host_cfg, |client_caps| {
            client_caps.iter().copied().next()
        })
        .await
        .unwrap();
        answer_clock_probe(host_chan_for_probe.as_ref()).await;
        session
    });
    let client_task = tokio::spawn(async move {
        ClientSession::connect(client_chan_for_session, client_cfg)
            .await
            .unwrap()
    });
    let _host = host_task.await.unwrap();
    let _client = client_task.await.unwrap();

    let host_chan_recv: Arc<dyn ControlChannel> = host_chan;
    let client_chan_send: Arc<dyn ControlChannel> = client_chan.clone();
    let client_chan_recv: Arc<dyn ControlChannel> = client_chan;
    let request_id = RequestId(99);
    let display_id = DisplayId(0);
    let mode = DisplayMode::new(1920, 1080, 60_000);

    let host_reply_task = tokio::spawn(async move {
        loop {
            match host_chan_recv.recv_control().await.unwrap() {
                ControlMessage::ForceIdr => continue,
                ControlMessage::SetDisplayMode {
                    request_id,
                    display_id,
                    ..
                } => {
                    host_chan_recv
                        .send_control(&ControlMessage::DisplayModeResult {
                            request_id,
                            display_id,
                            status: DisplayModeStatus::Unsupported,
                            actual_mode: None,
                        })
                        .await
                        .unwrap();
                    return;
                }
                other => panic!("expected SetDisplayMode, got {other:?}"),
            }
        }
    });

    client_chan_send
        .send_control(&ControlMessage::SetDisplayMode {
            request_id,
            display_id,
            mode,
            restore_on_disconnect: true,
        })
        .await
        .unwrap();
    let msg = client_chan_recv.recv_control().await.unwrap();
    host_reply_task.await.unwrap();
    match msg {
        ControlMessage::DisplayModeResult {
            request_id: got_request,
            display_id: got_display,
            status,
            actual_mode,
        } => {
            assert_eq!(got_request, request_id);
            assert_eq!(got_display, display_id);
            assert_eq!(status, DisplayModeStatus::Unsupported);
            assert_eq!(actual_mode, None);
        }
        other => panic!("expected DisplayModeResult, got {other:?}"),
    }
}
