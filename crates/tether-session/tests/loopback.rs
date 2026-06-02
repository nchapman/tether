//! End-to-end session loopback. Wires `HostSession::accept` and
//! `ClientSession::connect` together through a `DuplexControlChannel`
//! pair so the handshake's clock-sync math, profile negotiation,
//! Goodbye-on-no-match flow, and bit-depth validation get exercised
//! in CI without a real QUIC pair (or any UDP socket at all).
//!
//! These tests close the "no end-to-end session loopback" gap that
//! used to mean session-level bugs only surfaced in live sessions.
//! The phantom-latency bug (slow build closure biasing the
//! `ClockSync` offset by ~50ms) lived entirely in this gap before
//! the trait abstraction made loopback testable.

use std::sync::Arc;

use tether_protocol::control::{
    ChromaSubsampling, ClientHello, CodecKind, ControlMessage, ServerHello, VideoProfile, Viewport,
};
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
    };
    let client = ClientSessionConfig {
        client_name: "test-client".to_string(),
        client_decode_profiles: vec![VideoProfile::HEVC_8BIT_420, VideoProfile::H264_8BIT_420],
        viewport: None,
    };
    (host, client)
}

#[tokio::test]
async fn happy_path_handshake_completes_with_negotiated_profile() {
    let (host_chan, client_chan) = duplex_pair();
    let (host_cfg, client_cfg) = cfgs();

    let host_chan: Arc<dyn ControlChannel> = host_chan;
    let client_chan: Arc<dyn ControlChannel> = client_chan;

    let host_task = tokio::spawn(async move {
        HostSession::accept(host_chan, host_cfg, |client_caps| {
            // Trivial selector: prefer HEVC if mutually supported,
            // else H.264 — mirrors the production `pick_supported_profile`
            // preference order minus the tether-probe dep.
            [VideoProfile::HEVC_8BIT_420, VideoProfile::H264_8BIT_420]
                .into_iter()
                .find(|p| client_caps.contains(p))
        })
        .await
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
async fn no_mutual_profile_sends_goodbye_then_errors() {
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

    // From the client side, the placeholder ServerHello (carrying H.264
    // as a stand-in) surfaces as ProfileNotAdvertised because H.264 is
    // not in our advert. We deliberately don't try to drain the
    // following Goodbye through the raw channel: that would peek
    // under the `ClientSession` abstraction, and if `connect` is ever
    // changed to drain the Goodbye itself the test would hang. The
    // host-side `NoProfileIntersection` assertion below is what pins
    // the "Goodbye was sent" half — `HostSession::accept`'s code path
    // is the only way that error is returned, and `send_control` of
    // the Goodbye is unconditional on that path.
    let client_task = tokio::spawn(async move {
        let connect_err = ClientSession::connect(client_chan_dyn, client_cfg)
            .await
            .map(|_| ())
            .expect_err("expected ProfileNotAdvertised");
        assert!(
            matches!(connect_err, ConnectError::ProfileNotAdvertised { .. }),
            "expected ProfileNotAdvertised, got: {connect_err:?}"
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
    // moved on to reading `ControlMessage`s — bincode decode fails.
    //
    // This regression test exists to keep future refactors of
    // `HostSession::accept` honest. The proper fix is a typestate
    // wrapper that makes the double-call uncompilable; until that
    // lands, this catches the misuse at runtime.
    use tether_protocol::control::{ClientHelloV1, ServerHelloV1, VideoColorSpec};
    use tether_protocol::MonoNanos;

    let (host_chan, client_chan) = duplex_pair();
    let host_chan_dyn: Arc<dyn ControlChannel> = host_chan;

    // Client side: complete the handshake normally, then `recv_control`
    // — that read should see the (corrupt) second ServerHello and
    // bincode-fail.
    let client_task = tokio::spawn(async move {
        let _ = client_chan
            .client_handshake(ClientHello::V1(ClientHelloV1 {
                client_name: "test".into(),
                preferred_codecs: vec![CodecKind::H264],
                max_resolution: None,
                viewport: None,
                clock_probe_t0: MonoNanos::ZERO,
                extensions: Default::default(),
                resume_token: None,
            }))
            .await
            .unwrap();
        // The second ServerHello sits on the stream; reading it as a
        // ControlMessage fails the bincode decode.
        let second = client_chan.recv_control().await;
        assert!(
            second.is_err(),
            "a second send_server_hello on the same channel must not \
             decode as a ControlMessage; got: {second:?}"
        );
    });

    let (_, t1) = host_chan_dyn.recv_client_hello().await.unwrap();
    let placeholder = ServerHello::V1(ServerHelloV1 {
        server_name: "test".into(),
        chosen_codec: CodecKind::H264,
        chosen_chroma: ChromaSubsampling::Yuv420,
        color_space: VideoColorSpec::sdr_desktop(),
        resolution: (0, 0),
        clock_probe_t0_echo: MonoNanos::ZERO,
        t1_server_recv: MonoNanos::ZERO,
        t2_server_send: MonoNanos::ZERO,
        extensions: Default::default(),
        resume_token: None,
    });
    host_chan_dyn
        .send_server_hello(placeholder.clone(), MonoNanos::ZERO, t1)
        .await
        .unwrap();
    // Misuse: call it again. The client task asserts the corruption
    // becomes visible on the next read.
    host_chan_dyn
        .send_server_hello(placeholder, MonoNanos::ZERO, t1)
        .await
        .unwrap();

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
    // refused. The Goodbye(InternalError) flows from the *client*
    // back to the host via the regular session-end path, not via
    // HostSession::accept.
    let host_session = host_task.await.unwrap().unwrap();
    assert_eq!(host_session.negotiated, VideoProfile::HEVC_8BIT_420);
    client_task.await.unwrap();
}

#[tokio::test]
async fn client_filters_unknown_bit_depth_from_host_advert() {
    let (host_chan, client_chan) = duplex_pair();
    let host_cfg = HostSessionConfig {
        server_name: "buggy-host".to_string(),
        audio_config: None,
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

    // Host doesn't see the refusal (it succeeds on its side); the
    // client is the gate for unknown depths.
    let _host = host_task.await.unwrap().unwrap();
    client_task.await.unwrap();
}

#[tokio::test]
async fn host_filters_unknown_bit_depths_keeps_known_ones() {
    let (host_chan, client_chan) = duplex_pair();
    let host_cfg = HostSessionConfig {
        server_name: "test-host".to_string(),
        audio_config: None,
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

    let host_task = tokio::spawn(async move {
        HostSession::accept(host_chan_dyn, host_cfg, |client_caps| {
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
        .await
    });
    let client_task =
        tokio::spawn(async move { ClientSession::connect(client_chan_dyn, client_cfg).await });

    let host = host_task.await.unwrap().unwrap();
    let client = client_task.await.unwrap().unwrap();
    assert_eq!(host.negotiated, VideoProfile::HEVC_8BIT_420);
    assert_eq!(client.negotiated, VideoProfile::HEVC_8BIT_420);
}

#[tokio::test]
async fn client_handshake_decodes_legacy_inline_fields_when_extension_absent() {
    // A "legacy host" that doesn't emit the structured encode-profile
    // extension. Hand-build the ServerHello so it has only the inline
    // chosen_codec / chosen_chroma fields populated; the client must
    // synthesize an 8-bit VideoProfile.
    let (host_chan, client_chan) = duplex_pair();
    let client_cfg = ClientSessionConfig {
        client_name: "test-client".to_string(),
        // 8-bit HEVC must be advertised so the post-resolve
        // ProfileNotAdvertised check passes.
        client_decode_profiles: vec![VideoProfile::HEVC_8BIT_420],
        viewport: None,
    };

    let host_chan_dyn: Arc<dyn ControlChannel> = host_chan;
    let client_chan_dyn: Arc<dyn ControlChannel> = client_chan;

    // Drive the host side manually so we can build a "no-extension"
    // ServerHello — `HostSession::accept` always inserts the
    // pixel-format and encode-profile extensions, so we don't go
    // through it here.
    let host_task = tokio::spawn(async move {
        let (_hello, t1) = host_chan_dyn.recv_client_hello().await.unwrap();
        let server = ServerHello::V1(tether_protocol::control::ServerHelloV1 {
            server_name: "legacy".into(),
            chosen_codec: CodecKind::Hevc,
            chosen_chroma: ChromaSubsampling::Yuv420,
            color_space: tether_protocol::control::VideoColorSpec::sdr_desktop(),
            resolution: (1920, 1080),
            clock_probe_t0_echo: tether_protocol::MonoNanos::ZERO,
            t1_server_recv: tether_protocol::MonoNanos::ZERO,
            t2_server_send: tether_protocol::MonoNanos::ZERO,
            // No extension map — emulates a host predating the
            // structured advert.
            extensions: Default::default(),
            resume_token: None,
        });
        host_chan_dyn
            .send_server_hello(server, tether_protocol::MonoNanos::ZERO, t1)
            .await
            .unwrap();
        // Swallow the ForceIdr the client sends right after the
        // handshake completes.
        let _ = host_chan_dyn.recv_control().await.unwrap();
    });
    let client = ClientSession::connect(client_chan_dyn, client_cfg)
        .await
        .unwrap();
    host_task.await.unwrap();

    // Synthesized from inline fields with bit_depth = 8.
    assert_eq!(client.negotiated, VideoProfile::HEVC_8BIT_420);
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
    // The client config's `viewport` populates `ClientHelloV1::viewport`,
    // and the host receives it on its session struct. Loopback covers
    // both sides — guards against a future refactor dropping the
    // field in either direction.
    let (host_chan, client_chan) = duplex_pair();
    let (host_cfg, mut client_cfg) = cfgs();
    client_cfg.viewport = Some(Viewport::new(1280, 720));

    let host_chan: Arc<dyn ControlChannel> = host_chan;
    let client_chan: Arc<dyn ControlChannel> = client_chan;

    let host_task = tokio::spawn(async move {
        HostSession::accept(host_chan, host_cfg, |client_caps| {
            client_caps.iter().copied().next()
        })
        .await
        .unwrap()
    });
    let client_task = tokio::spawn(async move {
        ClientSession::connect(client_chan, client_cfg)
            .await
            .unwrap()
    });

    let host = host_task.await.unwrap();
    let _client = client_task.await.unwrap();
    assert_eq!(host.client_hello.viewport, Some(Viewport::new(1280, 720)));
}

#[tokio::test]
async fn mid_session_set_viewport_round_trips_on_control_stream() {
    // After the handshake, the client sending SetClientViewport
    // arrives on the host's recv_control. This is the wire half of
    // the production flow; the host binary's send-thread side
    // (encoder rebuild, force_idr raise) is covered by host-side
    // unit tests on the dim helper.
    let (host_chan, client_chan) = duplex_pair();
    let (host_cfg, client_cfg) = cfgs();

    let host_chan_for_session: Arc<dyn ControlChannel> = host_chan.clone();
    let client_chan_for_session: Arc<dyn ControlChannel> = client_chan.clone();

    let host_task = tokio::spawn(async move {
        HostSession::accept(host_chan_for_session, host_cfg, |client_caps| {
            client_caps.iter().copied().next()
        })
        .await
        .unwrap()
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
            .send_control(&ControlMessage::SetClientViewport(Viewport::new(640, 480)))
            .await
            .unwrap();
    });
    let recv_task = tokio::spawn(async move {
        // The client implementation sends a ForceIdr immediately after
        // a successful handshake (so the host emits a fresh keyframe
        // for the new decoder); swallow it so the SetClientViewport
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
        ControlMessage::SetClientViewport(v) => {
            assert_eq!(v, Viewport::new(640, 480));
        }
        other => panic!("expected SetClientViewport, got {other:?}"),
    }
}
