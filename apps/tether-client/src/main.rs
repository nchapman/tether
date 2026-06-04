//! Tether client — connects to a host, reassembles incoming video
//! frames, decodes them (HEVC/H.264 via VAAPI when available), and
//! presents them in a wgpu window.
//!
//! Recv loop runs on a tokio task draining `recv_datagram`: all video — IDR
//! keyframes and P-frames alike — arrives as FEC'd datagrams and feeds one
//! reassembler (cursor + audio datagrams share the channel). Decode runs on a
//! dedicated `std::thread` (`tether-decode`) so a GPU-driver stall in
//! libavcodec → libva can't starve the QUIC recv loop. Render is one-deep so a
//! slow renderer drops frames rather than back-pressuring upstream.
//!
//! Usage: `tether-client <host_addr> <cert_fingerprint_hex>`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::bounded;
use tether_decode::{DecodeCompletion, DecodeJob};
use tether_input::{WinitTranslator, WireEvent};
use tether_ipc::{EngineEvent, Reporter};
use tether_protocol::audio::AudioPacket;
use tether_protocol::control::{ControlMessage, GoodbyeCode, Viewport};
use tether_protocol::video::{FrameReassembler, VideoPacket};
use tether_protocol::MonoNanos;
use tether_render::LatestFrame;
use tether_render::RenderEvent;
use tether_session::{ClientSession, ClientSessionConfig, ConnectError};
use tether_transport::{Client, Datagram, ServerAuth};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

mod client_pairing;
use client_pairing::HostAuth;

// Initial window size — the actual frame dimensions come from
// `VideoFrameMeta::dimensions` once frames start arriving and the window
// will pick them up automatically because tether-render reallocates its
// texture on dimension change.
const INITIAL_WIDTH: u32 = 1280;
const INITIAL_HEIGHT: u32 = 720;

/// Whether a `FrameReassembler::handle()` loss-counter delta warrants a
/// `RequestRecovery`. `before`/`after` are `(frames_dropped, fragments_lost)`
/// snapshots taken around the `handle()` call.
///
/// Fires only on a `frames_dropped` increase — a frame started but pruned
/// before completing, the genuine "this frame will never arrive" signal.
/// Never fires on `fragments_lost` alone: that counts stale stragglers (a late
/// fragment for an already-finalized or already-pruned frame) and malformed
/// packets, neither of which is independently actionable. Triggering on them
/// would emit spurious recovery IDRs that add bandwidth and worsen congestion
/// on an already-lossy path; a frame that truly won't complete still bumps
/// `frames_dropped` when it's pruned, so the real signal survives.
fn recovery_warranted(before: (u64, u64), after: (u64, u64)) -> bool {
    after.0 > before.0
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> anyhow::Result<()> {
    // Parse args first so `--ipc` routes tracing off stdout (reserved for
    // the JSON-lines protocol) before the subscriber is installed.
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let ipc = raw_args.iter().any(|a| a == "--ipc");
    let reporter = Reporter::from_ipc_flag(ipc);

    // `_tracing_guard` keeps the non-blocking writer's worker thread
    // alive. Dropping it flushes pending log lines and shuts the
    // worker down; binding it for the duration of `main` ensures
    // logs aren't truncated at process exit.
    let _tracing_guard = init_tracing(reporter.is_json());

    // Parse args: positional host addr (and optional explicit fingerprint),
    // plus `--pin <PIN>` (first-contact pairing) and `--label <name>` (display
    // name to record for the host). `--ipc` was already consumed above.
    let CliArgs {
        addr,
        fingerprint_hex,
        pin,
        label,
        audio: audio_enabled,
    } = parse_cli_args(&raw_args)?;

    // Decide how to authenticate the host. Precedence: an explicit `--pin`
    // means first-contact pairing; otherwise an explicit fingerprint or a
    // known-hosts entry for this address means a pinned reconnect.
    let config_dir = client_config_dir()?;
    let known_hosts_path = config_dir.join("known_hosts.json");
    let known_hosts = tether_pairing::KnownHosts::load(&known_hosts_path).map_err(|e| {
        // Fail closed: a corrupt known-hosts file could otherwise drop pinning.
        let msg = format!("failed to load {}: {e}", known_hosts_path.display());
        reporter.emit(&EngineEvent::Error {
            message: msg.clone(),
        });
        anyhow::anyhow!(msg)
    })?;

    let (server_auth, mode) = if let Some(pin) = pin {
        (ServerAuth::TrustOnFirstPair, HostAuth::FirstContact { pin })
    } else if let Some(fp_hex) = &fingerprint_hex {
        // Explicit fingerprint reconnect. If this address is already pinned,
        // the supplied value must match — otherwise this would be a silent
        // trust downgrade (re-pinning a known host to an attacker-supplied
        // fingerprint without a PIN). Re-pairing requires --pin.
        let supplied = hex_decode(fp_hex)?;
        if let Some(known) = known_hosts.fingerprint(&addr.to_string()) {
            if known != supplied {
                let msg = format!(
                    "supplied fingerprint for {addr} does not match the pinned one; \
                     use --pin to re-pair after verifying the host"
                );
                reporter.emit(&EngineEvent::Error {
                    message: msg.clone(),
                });
                anyhow::bail!(msg);
            }
        }
        (ServerAuth::Pinned(supplied), HostAuth::Resume)
    } else if let Some(fp) = known_hosts.fingerprint(&addr.to_string()) {
        (ServerAuth::Pinned(fp), HostAuth::Resume)
    } else {
        let msg = format!(
            "unknown host {addr}: pass --pin <PIN> to pair (the host shows a PIN \
             under \"Add a device\"), or pass the host fingerprint"
        );
        reporter.emit(&EngineEvent::Error {
            message: msg.clone(),
        });
        anyhow::bail!(msg);
    };

    reporter.emit(&EngineEvent::Connecting {
        host: addr.to_string(),
    });

    let client = Client::with_identity(&config_dir)?;
    let client_fp = client.fingerprint();
    let pending = match client
        .connect_pending(addr, "tether-host", server_auth)
        .await
    {
        Ok(p) => p,
        Err(e) => {
            reporter.emit(&EngineEvent::Error {
                message: format!("connect failed: {e}"),
            });
            return Err(e.into());
        }
    };
    let is_first_contact = matches!(mode, HostAuth::FirstContact { .. });
    let (conn, host_fp) = match client_pairing::establish(pending, &mode, client_fp).await {
        Ok(pair) => pair,
        Err(e) => {
            reporter.emit(&EngineEvent::Error {
                message: format!("pairing failed: {e}"),
            });
            return Err(e);
        }
    };

    // Persist known-hosts: on first contact, pin the host so the next connect
    // is one-click; on every successful connect (first-contact or resume),
    // stamp the time so the shell's address book can show recency.
    {
        let mut known_hosts = known_hosts;
        let addr_key = addr.to_string();
        let now = unix_now();
        if is_first_contact {
            let label = label.unwrap_or_else(|| addr_key.clone());
            known_hosts.insert(addr_key.clone(), &host_fp, label, now);
        }
        known_hosts.set_last_connected(&addr_key, now);
        if let Err(e) = known_hosts.save(&known_hosts_path) {
            warn!(error = %e, "connected but failed to persist known-hosts; first-contact reconnect may need --pin again");
        }
    }

    let conn = Arc::new(conn);
    info!(remote = %conn.remote_address(), "connected to host");

    // Client video decode capabilities. The probe in tether-probe does
    // a real encode + decode round trip per profile against the live
    // driver — see crates/tether-codec/src/profile_probe.rs for why a
    // construction-only probe wasn't enough. macOS clients on M-series
    // silicon advertise HEVC 4:4:4 here (VT decodes Main444 to a
    // `'444v'` IOSurface and the renderer's biplanar path handles it).
    // Linux clients keep 4:4:4 if the VAAPI driver supports it. The
    // function returns profiles in PROFILE_PREFERENCE order so logs
    // look natural.
    let mut client_decode_profiles = tether_probe::client_decode_profiles();
    // Renderer capability gate for 10-bit. The codec decode probe can't
    // see whether the *renderer* can present 10-bit, so each backend's
    // `supports_10bit_render` answers that (the platform-specific reason
    // lives in its doc comment): on Linux/macOS it's the wgpu adapter's
    // `TEXTURE_FORMAT_16BIT_NORM`; on Windows it's D3D11 P010 texture
    // support. Filter 10-bit profiles out of our advert when the renderer
    // can't present them so the host's negotiator never picks one we
    // can't actually render.
    if !tether_render::supports_10bit_render().await {
        let before = client_decode_profiles.len();
        client_decode_profiles.retain(|p| p.bit_depth == 8);
        let dropped = before - client_decode_profiles.len();
        if dropped > 0 {
            info!(
                dropped_profiles = dropped,
                "renderer cannot present 10-bit; dropping 10-bit profiles \
                 from the decode-capability advert"
            );
        }
    }
    if client_decode_profiles.is_empty() {
        let message = "no hardware video decoder is available on this client \
             (no codec in PROFILE_PREFERENCE constructed). Tether requires \
             GPU decode; there is no software fallback."
            .to_string();
        reporter.emit(&EngineEvent::Error {
            message: message.clone(),
        });
        anyhow::bail!(message);
    }

    // Application-layer handshake: identify ourselves, advertise our
    // decode profiles, resolve + validate the host's pick, and prime
    // the host with a `ForceIdr` so the next frame is a keyframe.
    // The clock-sync probe round-trip happens inside `client_handshake`
    // so latency logs are wall-clock-accurate from the first frame.
    // ClientSession takes the channel through the `ControlChannel`
    // trait object so it's mockable in tests. The original
    // `Arc<Connection>` stays in `conn` for the rest of `main` — the
    // recv tasks below use concrete-`Connection` methods (datagram,
    // keyframe-stream accept, input send) that aren't on the trait.
    let session = ClientSession::connect(
        conn.clone() as Arc<dyn tether_transport::ControlChannel>,
        ClientSessionConfig {
            client_name: "tether-client".to_string(),
            client_decode_profiles: client_decode_profiles.clone(),
            // Initial viewport = the window's logical size at connect
            // time. The renderer's first WindowEvent::Resized will fire
            // shortly after the window is created (the WM sizes it to
            // the actual physical pixels for the display's scale
            // factor) and the viewport debouncer task below will follow
            // up with a SetClientViewport reflecting the physical dims.
            // Sending an initial guess here means the first encoded
            // frame is already at roughly-the-right size instead of
            // wasting one frame at native capture dims.
            viewport: Some(Viewport::new(INITIAL_WIDTH, INITIAL_HEIGHT)),
        },
    )
    .await;
    let session = match session {
        Ok(s) => s,
        Err(e) => {
            let err = match e {
                ConnectError::ProfileNotAdvertised { .. }
                | ConnectError::InvalidEncodeProfile { .. }
                | ConnectError::UnknownBitDepth(_, _) => anyhow::anyhow!("{e}"),
                ConnectError::Transport(t) => anyhow::Error::from(t),
            };
            reporter.emit(&EngineEvent::Error {
                message: format!("handshake failed: {err}"),
            });
            return Err(err);
        }
    };
    let ClientSession {
        channel: _,
        negotiated: negotiated_profile,
        server_hello,
        clock_sync,
        client_decode_profiles: _,
    } = session;

    reporter.emit(&EngineEvent::Connected {
        host: addr.to_string(),
        profile: format!(
            "{:?} {:?} {}-bit",
            negotiated_profile.codec, negotiated_profile.chroma, negotiated_profile.bit_depth
        ),
    });

    // Single-slot drop-oldest channel: the renderer always wants the
    // freshest decoded frame, not a queued backlog. Cheap clone (Arc
    // inside); the decoder thread takes one, the renderer takes
    // another.
    let frames = LatestFrame::new();

    // Shared cursor state. The control-stream task uploads sprites
    // and activates the current shape; the datagram-recv task pushes
    // position updates; the renderer's overlay pass reads each
    // frame. Cloned through to all three consumers (cheap `Arc`).
    let cursor_channel = tether_render::CursorChannel::new();

    // Receive-side control loop. Today the host doesn't initiate any
    // typed messages we need to act on, but the Extension escape and
    // future variants (CursorShape, DisplayList, StreamPause/Resume)
    // arrive here, so the loop exists from V1 onward.
    {
        let conn = conn.clone();
        let cursor_channel_ctrl = cursor_channel.clone();
        tokio::spawn(async move {
            loop {
                match conn.recv_control().await {
                    Ok(ControlMessage::ForceIdr) => {
                        tracing::trace!("host sent ForceIdr (no-op on client)");
                    }
                    Ok(ControlMessage::RequestRecovery { .. }) => {
                        // Client never receives RequestRecovery — it's
                        // a client→host signal. A host that ever sends
                        // it is misbehaving; log and ignore.
                        tracing::warn!("host sent RequestRecovery (wrong direction); ignoring");
                    }
                    Ok(ControlMessage::SetCursorMode { mode }) => {
                        // The host echoes our SetCursorMode on every
                        // toggle. No client-side action needed beyond
                        // a trace log — the client already changed
                        // its cursor-grab state to drive the send.
                        tracing::debug!(?mode, "host echoed cursor mode");
                    }
                    Ok(ControlMessage::ClockProbeRequest { t0_sender }) => {
                        let t1 = MonoNanos::now();
                        let response = ControlMessage::ClockProbeResponse(
                            tether_protocol::control::ClockProbe {
                                t0_sender,
                                t1_receiver_recv: t1,
                                t2_receiver_send: MonoNanos::now(),
                            },
                        );
                        if let Err(e) = conn.send_control(&response).await {
                            warn!(error = ?e, "clock probe response failed; ending control loop");
                            return;
                        }
                    }
                    Ok(ControlMessage::ClockProbeResponse(_)) => {
                        tracing::trace!("unsolicited clock probe response; ignoring");
                    }
                    Ok(ControlMessage::Goodbye { reason, code }) => {
                        info!(%reason, ?code, "host said goodbye");
                        return;
                    }
                    Ok(ControlMessage::Extension { key, payload }) => {
                        tracing::debug!(
                            key = %key,
                            payload_len = payload.len(),
                            "unknown control extension; ignoring"
                        );
                    }
                    Ok(ControlMessage::CursorShape {
                        id,
                        hotspot,
                        width,
                        height,
                        format,
                        pixels,
                    }) => {
                        // The wire pixel format is always Rgba8 today
                        // (`CursorPixelFormat::Rgba8`). New variants
                        // would land alongside renderer-side
                        // conversion; until then we drop unknown
                        // formats rather than rendering garbage.
                        use tether_protocol::cursor::CursorPixelFormat;
                        if !matches!(format, CursorPixelFormat::Rgba8) {
                            tracing::warn!(
                                id,
                                ?format,
                                "unsupported cursor pixel format; dropping shape"
                            );
                            continue;
                        }
                        info!(
                            id,
                            ?hotspot,
                            width,
                            height,
                            pixel_bytes = pixels.len(),
                            "received cursor shape; enqueuing for renderer upload",
                        );
                        cursor_channel_ctrl.with(|state| {
                            state.enqueue_shape(
                                id,
                                u32::from(width),
                                u32::from(height),
                                u32::from(hotspot.0),
                                u32::from(hotspot.1),
                                pixels,
                            );
                            // The first shape arrives before any
                            // `CursorUseShape` and before any position
                            // update; activate it eagerly so the
                            // overlay starts drawing as soon as the
                            // first position datagram arrives.
                            state.activate(id);
                        });
                    }
                    Ok(ControlMessage::CursorUseShape { id }) => {
                        tracing::debug!(id, "host activated cursor shape");
                        cursor_channel_ctrl.with(|state| state.activate(id));
                    }
                    Ok(ControlMessage::DisplayList { displays }) => {
                        info!(count = displays.len(), "host display topology");
                        for d in &displays {
                            info!(
                                id = d.id,
                                name = %d.name,
                                width = d.width,
                                height = d.height,
                                refresh_mhz = d.refresh_mhz,
                                scale = format!("{}/{}", d.scale_num, d.scale_den),
                                primary = d.primary,
                                position = ?d.position,
                                "  display"
                            );
                        }
                    }
                    Ok(ControlMessage::SetActiveDisplays { .. }) => {
                        // Client-originated; misrouted if seen on the client side.
                        tracing::debug!(
                            "unexpected client→host SetActiveDisplays arrived on client; ignoring"
                        );
                    }
                    Ok(
                        ControlMessage::StreamReady { .. }
                        | ControlMessage::StreamPause { .. }
                        | ControlMessage::StreamResume { .. }
                        | ControlMessage::ClientStats { .. }
                        | ControlMessage::SetClientViewport(_),
                    ) => {
                        // Client-originated; misrouted if seen on the client side.
                        tracing::debug!(
                            "unexpected client→host control message arrived on client; ignoring"
                        );
                    }
                    Err(e) => {
                        warn!(error = ?e, "control recv failed; ending control loop");
                        return;
                    }
                }
            }
        });
    }

    let conn_recv = conn.clone();
    let recv_clock_sync = clock_sync;
    let decode_profile = negotiated_profile;
    let conn_ready = conn.clone();
    let cursor_channel_datagram = cursor_channel.clone();

    // Decode runs on a dedicated std::thread so a GPU-driver stall
    // (vaSyncSurface, vaExportSurfaceHandle, etc.) inside the
    // libavcodec → libva path can't starve the QUIC recv loop's
    // tokio task. The recv loop hands ReassembledFrames over a
    // bounded crossbeam channel; if the decoder falls behind, the
    // recv loop drops frames at the channel rather than blocking on
    // the await.
    let (decode_job_tx, decode_job_rx) = bounded::<DecodeJob>(8);
    let (decode_completion_tx, decode_completion_rx) =
        crossbeam_channel::unbounded::<DecodeCompletion>();
    // Oneshot from the decode thread back to the recv task: lets the
    // recv loop wait for decoder construction before announcing
    // StreamReady to the host. If the decoder fails to build, the
    // sender drops and `await` returns Err.
    let (decoder_ready_tx, decoder_ready_rx) = tokio::sync::oneshot::channel::<()>();
    let runtime_handle = tokio::runtime::Handle::current();
    let conn_for_decode = conn.clone();
    let frames_for_decode = frames.clone();
    // ForceIdr send is fire-and-forget — spawned onto the tokio
    // runtime because the decoder thread is a `std::thread`. A failed
    // send is logged but not retried; the next decode error after
    // `IDR_RATE_LIMIT` re-triggers.
    let request_idr: Arc<dyn Fn() + Send + Sync + 'static> = Arc::new(move || {
        let conn = conn_for_decode.clone();
        runtime_handle.spawn(async move {
            if let Err(e) = conn.send_control(&ControlMessage::ForceIdr).await {
                warn!(error = ?e, "ForceIdr send failed");
            }
        });
    });
    let warnings: Arc<dyn Fn() -> u64 + Send + Sync + 'static> =
        Arc::new(tether_codec::av_log::warning_or_above_count);
    // Windows decode is GPU-resident: the native D3D11 renderer opens the
    // decoder's NT shared handles directly (no cross-API import), so the
    // decoder always exports `Frame::Gpu` there. Other platforms export
    // GPU-resident frames through their own backends and ignore this flag.
    let gpu_export = cfg!(target_os = "windows");
    info!(gpu_export, "decode export mode resolved");
    tether_decode::run_thread(
        decode_profile,
        decode_job_rx,
        decode_completion_tx,
        frames_for_decode,
        request_idr,
        warnings,
        decoder_ready_tx,
        gpu_export,
    );

    // Audio: if enabled and the host advertised an `AudioConfig`, stand up the
    // decode → jitter buffer → cpal playback path on its own thread.
    // `audio_tx` feeds it from the datagram recv loop below; `audio_active`
    // gates `StreamReady.audio`. Both are moved into the recv task.
    let (audio_tx, audio_active) = setup_audio_playback(audio_enabled, &server_hello);

    let conn_for_recovery_send = conn.clone();
    tokio::spawn(async move {
        let mut reassembler = FrameReassembler::new();
        if decoder_ready_rx.await.is_err() {
            // Decoder construction failed and the thread dropped the
            // sender without signalling ready. The host has no other
            // way to learn we won't ever render its frames, so send a
            // Goodbye(InternalError) before exiting — otherwise the
            // host keeps encoding into a black hole until idle
            // timeout, and the user sees a frozen window with no
            // explanation. `say_goodbye_with_code` closes the
            // connection as part of its shutdown.
            error!(
                "decode thread failed to initialise; sending Goodbye(InternalError) and exiting"
            );
            say_goodbye_with_code(
                &conn_ready,
                "client decoder failed to initialise",
                GoodbyeCode::InternalError,
            )
            .await;
            return;
        }
        // Decoder is up; tell the host to start streaming. `audio` is true only
        // when we built a playback path for the host's advertised AudioConfig —
        // the host's audio gate keys off this.
        if let Err(e) = conn_ready
            .send_control(&ControlMessage::StreamReady {
                video: true,
                audio: audio_active,
            })
            .await
        {
            warn!(error = ?e, "StreamReady send failed; host will not emit video");
        }
        let mut frame_count: u64 = 0;
        // Reassembler cumulative counters at the start of the current
        // stats window. Diff against the live counters to compute the
        // per-interval drop and fragment-loss rates for ClientStats.
        let mut last_frames_dropped: u64 = 0;
        let mut last_fragments_lost: u64 = 0;
        // Most-recent frame_seq the reassembler *completed*. Quoted in
        // RequestRecovery when the reassembler observes a stale drop, and
        // doubles as the "have we received any frame yet" sentinel that gates
        // the recovery request. NOTE: this is "reassembled", NOT "decoded
        // cleanly" — a frame can reassemble (FEC rebuilt every shard) yet be a
        // P-frame referencing an earlier frame the decoder concealed. The host
        // only logs the value and always responds with a full IDR, so the
        // distinction is harmless today; it must not be used to select an
        // encoder reference without a decode-success signal we don't have. See
        // the `RequestRecovery` doc in tether-protocol.
        let mut last_reassembled_frame_seq: Option<u32> = None;
        // Last time we emitted a RequestRecovery. Rate-limits the
        // signal to one every IDR_RATE_LIMIT (500 ms) so a burst
        // of drops collapses into a single recovery action — same
        // cadence the decoder thread's auto-IDR uses.
        let mut last_request_recovery_at: Option<MonoNanos> = None;
        // Sum of decode call wall-clocks across the frames in the
        // current log window, surfaced as avg_decode_ms on the
        // frame-stats line. Same shape as the host's avg_encode_ms
        // metric so the two together show where pipeline time is
        // going.
        let mut decode_latency_sum_ns: u64 = 0;
        // Sum of capture-to-recv ages over the window. The previous
        // implementation logged the last frame's age which is
        // misleading when the metric is supposed to summarise a
        // second of behaviour; averaging across frames gives an
        // actually-meaningful number.
        let mut latency_sum_ns: u64 = 0;
        // Sum of t_send (host clock, translated via clock_sync) to
        // local recv times. Isolates the network leg from compute
        // — pair with avg_encode_ms / avg_decode_ms to attribute
        // any latency budget movement to the right component.
        let mut network_latency_sum_ns: u64 = 0;
        // Bytes off the wire (encoded H.264 payloads after
        // reassembly). With matching host kbps_out, a divergence
        // means packets are being dropped between host and client.
        let mut bytes_received: u64 = 0;
        // Decode failures in the window. Steady non-zero values
        // mean we're losing IDR/SPS/PPS frames or the bitstream is
        // getting corrupted between encode and decode.
        let mut decode_errors: u32 = 0;
        // Frames the decoder produced but displaced before the
        // renderer could pick them up (LatestFrame is single-slot
        // drop-oldest). Non-zero means the render thread isn't
        // keeping up with arrival rate — typically a wgpu/present
        // pacing issue, not a codec issue.
        let mut render_drops: u32 = 0;
        // ForceIdr control messages we sent in the window. Pairs
        // with the host's kf_per_s — if these match, the storm of
        // keyframes on the host is *our* fault, not the encoder
        // misbehaving.
        let mut idr_requests: u32 = 0;
        // Frames the recv loop reassembled but couldn't enqueue for
        // decode because the bounded channel was full. Non-zero means
        // the decoder is falling behind the network — a strong signal
        // to drop quality or alert the user.
        let mut decode_queue_drops: u32 = 0;
        // Decode completions folded into the current stats window.
        // Used as the divisor for avg_decode_ms so the metric stays
        // honest when frame_count (recv-side) and completions drift
        // under backpressure.
        let mut decode_completion_count: u64 = 0;
        let mut last_log = Instant::now();
        // Cursor datagram observability — separate cadence so a
        // chatty cursor channel doesn't bury the video stats line.
        let mut cursor_pos_packets: u64 = 0;
        let mut last_cursor_log = std::time::Instant::now();

        loop {
            // All video — IDR keyframes and P-frames alike — arrives on the
            // unreliable datagram channel and feeds the same reassembler; there
            // is no separate reliable keyframe stream to race. Cursor and audio
            // datagrams share the channel and are dispatched out here.
            let packet: VideoPacket = match conn_recv.recv_datagram().await {
                Ok(Datagram::Video(p)) => p,
                Ok(Datagram::HostCursor(hc)) => {
                    // Position datagrams ride latest-wins; the overlay's render
                    // pass reads the most recent value each frame.
                    use tether_protocol::cursor::HostCursorPacket;
                    match hc {
                        HostCursorPacket::Position { x, y, visible, .. } => {
                            // The host position is ignored for rendering —
                            // the overlay is anchored to the local pointer
                            // (zero round-trip lag). Only the host's
                            // visibility intent is consumed here.
                            cursor_channel_datagram.with(|state| {
                                state.set_host_visible(visible);
                            });
                            cursor_pos_packets += 1;
                            if last_cursor_log.elapsed() >= std::time::Duration::from_secs(2) {
                                info!(
                                    cursor_pos_packets,
                                    last_x = x,
                                    last_y = y,
                                    visible,
                                    "cursor position datagrams"
                                );
                                last_cursor_log = std::time::Instant::now();
                            }
                        }
                    }
                    continue;
                }
                Ok(Datagram::ClientCursor(_)) => {
                    // Client-originated cursor packets should never come back to
                    // the client; ignore defensively.
                    continue;
                }
                Ok(Datagram::Audio(AudioPacket::Opus {
                    frame_seq, payload, ..
                })) => {
                    // Forward to the audio decode thread; drop on a full channel
                    // (decoder behind) — audio is loss-tolerant.
                    if let Some(tx) = &audio_tx {
                        let _ = tx.try_send((frame_seq, payload));
                    }
                    continue;
                }
                Err(e) => {
                    // Promoted from warn → error: this is terminal for the video
                    // stream and the user otherwise sees a frozen last-frame with
                    // no indication anything broke. Also close the connection
                    // explicitly so the host learns about it instead of waiting
                    // for the idle timeout.
                    error!(error = ?e, "datagram recv failed; closing connection and ending recv loop");
                    conn_recv.close(1, b"recv failed");
                    break;
                }
            };

            // Snapshot loss counters around the handle() so we can see if
            // this packet's processing caused the reassembler to give up on
            // a frame. We gate recovery on `frames_dropped` only — the count
            // of frames started but pruned incomplete. That is the genuine
            // "a frame will never complete" signal worth a recovery IDR.
            //
            // We deliberately do NOT trigger on `fragments_lost`: that counter
            // bumps when a *straggler* fragment arrives for a frame that was
            // already finalized or pruned, or when a malformed packet is
            // rejected. Neither is independently actionable — by the time a
            // late fragment shows up the frame is already gone (recovered or
            // abandoned), so firing on it just emits spurious recovery IDRs
            // that add bandwidth and worsen congestion exactly when the path
            // is already lossy. A frame that truly won't complete still bumps
            // `frames_dropped` when `prune_old` evicts it, so the real signal
            // is not lost — only the noise.
            let pre_loss = reassembler.loss_counters();
            let result = reassembler.handle(packet);
            let post_loss = reassembler.loss_counters();
            if recovery_warranted(pre_loss, post_loss) {
                if let Some(last_reassembled) = last_reassembled_frame_seq {
                    let now = MonoNanos::now();
                    let rate_limit_ns = 500_000_000u64;
                    let fire = last_request_recovery_at
                        .is_none_or(|t| now.saturating_sub(t) > rate_limit_ns);
                    if fire {
                        last_request_recovery_at = Some(now);
                        let send_conn = conn_for_recovery_send.clone();
                        tokio::spawn(async move {
                            if let Err(e) = send_conn
                                .send_control(&ControlMessage::RequestRecovery {
                                    last_reassembled_frame_id: last_reassembled,
                                })
                                .await
                            {
                                tracing::debug!(
                                    error = ?e,
                                    "RequestRecovery send failed; host will fall back to its own auto-IDR"
                                );
                            }
                        });
                    }
                }
            }
            let Some(frame) = result else { continue };
            // Reassembled, not yet decoded — see the declaration comment.
            last_reassembled_frame_seq = Some(frame.frame_seq);
            let now = MonoNanos::now();
            // Host timestamps -> client clock via the handshake
            // offset. host_in_client_clock is the moment the
            // host captured the frame; send_in_client_clock is
            // the moment the host handed it to QUIC. The
            // difference between them and `now` decomposes
            // total latency into capture-to-send (host
            // pipeline) and send-to-recv (network + reassembly).
            let host_in_client_clock =
                recv_clock_sync.remote_to_local(frame.meta.timing.t_capture_userspace);
            let send_in_client_clock = recv_clock_sync.remote_to_local(frame.meta.timing.t_send);
            let age_ns = now.saturating_sub(host_in_client_clock);
            let network_ns = now.saturating_sub(send_in_client_clock);
            frame_count += 1;
            latency_sum_ns = latency_sum_ns.saturating_add(age_ns);
            network_latency_sum_ns = network_latency_sum_ns.saturating_add(network_ns);
            bytes_received = bytes_received.saturating_add(frame.body.len() as u64);

            // Hand the reassembled frame to the decode thread.
            // Bounded channel + try_send means a stalled decoder
            // doesn't block the recv loop — we drop the frame and
            // count the loss so the stats line surfaces it.
            let job = DecodeJob {
                body: frame.body,
                host_in_client_clock,
                keyframe: frame.meta.keyframe,
            };
            if decode_job_tx.try_send(job).is_err() {
                decode_queue_drops = decode_queue_drops.saturating_add(1);
            }

            // Drain decode completions that arrived since the last
            // iteration. Non-blocking — if the decoder hasn't
            // produced anything yet, we'll pick it up next time
            // round. Folds per-frame metrics into the stats window
            // the recv loop owns.
            while let Ok(c) = decode_completion_rx.try_recv() {
                decode_completion_count = decode_completion_count.saturating_add(1);
                decode_latency_sum_ns = decode_latency_sum_ns.saturating_add(c.decode_duration_ns);
                if c.decode_err || c.soft_failure {
                    decode_errors = decode_errors.saturating_add(1);
                }
                render_drops = render_drops.saturating_add(c.render_drops);
                if c.idr_request_fired {
                    idr_requests = idr_requests.saturating_add(1);
                }
            }

            if last_log.elapsed() >= std::time::Duration::from_secs(1) {
                let window_secs = last_log.elapsed().as_secs_f64();
                // ClientStats — host uses this to drive future
                // adaptive bitrate / FEC strength / codec
                // downshift decisions. Counters are diffed
                // against last window so the wire field is a
                // per-interval rate; rtt_ewma_us is whole-
                // session EWMA on the QUIC RTT.
                let (frames_dropped_now, fragments_lost_now) = reassembler.loss_counters();
                let frames_dropped_delta =
                    u32::try_from(frames_dropped_now.saturating_sub(last_frames_dropped))
                        .unwrap_or(u32::MAX);
                let fragments_lost_delta =
                    u32::try_from(fragments_lost_now.saturating_sub(last_fragments_lost))
                        .unwrap_or(u32::MAX);
                last_frames_dropped = frames_dropped_now;
                last_fragments_lost = fragments_lost_now;
                // window_secs is a small positive duration; round-to-i64 is intentional.
                #[allow(clippy::cast_possible_truncation)]
                let interval_ms =
                    u32::try_from((window_secs * 1000.0).round() as i64).unwrap_or(u32::MAX);
                let rtt_ewma_us =
                    u32::try_from(conn_recv.rtt().as_micros().min(u128::from(u32::MAX)))
                        .unwrap_or(u32::MAX);
                let stats = ControlMessage::ClientStats {
                    interval_ms,
                    frames_received: u32::try_from(frame_count).unwrap_or(u32::MAX),
                    frames_dropped: frames_dropped_delta,
                    fragments_lost: fragments_lost_delta,
                    rtt_ewma_us,
                };
                let conn_stats = conn_recv.clone();
                tokio::spawn(async move {
                    if let Err(e) = conn_stats.send_control(&stats).await {
                        tracing::trace!(error = ?e, "ClientStats send failed");
                    }
                });

                // Divide by completions, not recv-side frame_count:
                // under backpressure they diverge (frame_count grows,
                // completions lag). Dividing by frame_count would
                // understate decode time exactly when it matters.
                #[allow(clippy::cast_precision_loss)] // u64 well under 2^53
                let avg_decode_ms = if decode_completion_count > 0 {
                    (decode_latency_sum_ns as f64 / decode_completion_count as f64) / 1_000_000.0
                } else {
                    0.0
                };
                #[allow(clippy::cast_precision_loss)]
                let avg_latency_ms = if frame_count > 0 {
                    (latency_sum_ns as f64 / frame_count as f64) / 1_000_000.0
                } else {
                    0.0
                };
                #[allow(clippy::cast_precision_loss)]
                let avg_network_ms = if frame_count > 0 {
                    (network_latency_sum_ns as f64 / frame_count as f64) / 1_000_000.0
                } else {
                    0.0
                };
                #[allow(clippy::cast_precision_loss)]
                let kbps_in = if window_secs > 0.0 {
                    (bytes_received as f64 * 8.0 / 1000.0) / window_secs
                } else {
                    0.0
                };
                info!(
                    frames_per_s = frame_count,
                    latency_ms = format!("{avg_latency_ms:.2}"),
                    network_ms = format!("{avg_network_ms:.2}"),
                    avg_decode_ms = format!("{avg_decode_ms:.2}"),
                    kbps_in = format!("{kbps_in:.0}"),
                    decode_errs = decode_errors,
                    render_drops = render_drops,
                    idr_reqs = idr_requests,
                    decode_queue_drops,
                    "frame stats"
                );
                frame_count = 0;
                decode_latency_sum_ns = 0;
                latency_sum_ns = 0;
                network_latency_sum_ns = 0;
                bytes_received = 0;
                decode_errors = 0;
                render_drops = 0;
                idr_requests = 0;
                decode_queue_drops = 0;
                decode_completion_count = 0;
                last_log = Instant::now();
            }
        }
    });

    // Bridge winit window events from the render thread into a tokio
    // task that owns the wire. UnboundedSender is safe to call from
    // the render thread (sync) and the receiver runs inside the tokio
    // runtime where the send paths are async. Cursor goes on the
    // unreliable datagram channel; everything else on the reliable
    // input stream.
    let (events_tx, mut events_rx) = mpsc::unbounded_channel::<RenderEvent>();
    let (viewport_tx, mut viewport_rx) = mpsc::unbounded_channel::<(u32, u32)>();
    let conn_input = conn.clone();
    tokio::spawn(async move {
        let mut translator = WinitTranslator::new();
        while let Some(render_event) = events_rx.recv().await {
            // Forward CursorModeChanged separately on the control
            // stream so the host learns of the toggle. The input
            // translator skips these events.
            if let RenderEvent::CursorModeChanged(mode) = render_event {
                let send_conn = conn_input.clone();
                tokio::spawn(async move {
                    let msg = ControlMessage::SetCursorMode { mode };
                    if let Err(e) = send_conn.send_control(&msg).await {
                        tracing::debug!(error = ?e, "SetCursorMode send failed");
                    }
                });
                continue;
            }
            for wire in translator.translate(render_event) {
                match wire {
                    WireEvent::Input(evt) => {
                        if let Err(e) = conn_input.send_input(&evt).await {
                            error!(error = ?e, "send_input failed; ending input loop");
                            return;
                        }
                    }
                    WireEvent::Cursor(pkt) => {
                        if let Err(e) = conn_input.send_datagram(&Datagram::ClientCursor(pkt)) {
                            // Cursor packets are best-effort by design
                            // — log at debug and keep going. A burst
                            // of failures means quinn's send queue is
                            // saturated, which a moving cursor will
                            // self-recover from.
                            tracing::debug!(error = ?e, "cursor datagram drop");
                        }
                    }
                }
            }
        }
    });

    // Viewport debouncer task. Drag-resizing fires
    // `WindowEvent::Resized` continuously (often >100 events/second);
    // sending a `SetClientViewport` per event would have the host
    // rebuild its encoder on every pixel of drag — expensive on
    // Metal/DX12 where pipeline compile is hundreds of ms, and produces
    // a stream of one-frame DimMismatch drops in the scaler. Coalesce
    // by waiting for ~150 ms of quiescence before forwarding the most
    // recent size. 150 ms is the standard UI-debounce floor — fast
    // enough to feel live, slow enough to filter drag noise. Zero-dim
    // sizes (minimised window) are dropped — encoding at 0×0 would
    // panic, and the host's current dims are still appropriate for
    // when the window comes back.
    let conn_viewport = conn.clone();
    tokio::spawn(async move {
        use std::time::Duration;
        let debounce = Duration::from_millis(150);
        let mut pending: Option<(u32, u32)> = None;
        loop {
            // Either receive a new size, or fire the pending one after
            // the debounce window elapses with no new event.
            let next = match pending {
                Some(_) => tokio::time::timeout(debounce, viewport_rx.recv()).await,
                None => Ok(viewport_rx.recv().await),
            };
            match next {
                Ok(Some(size)) => {
                    pending = Some(size);
                }
                Ok(None) => {
                    // Sender dropped. Fire any pending before exiting
                    // so the last resize event still makes it.
                    if let Some((w, h)) = pending {
                        let viewport = Viewport::new(w, h);
                        if viewport.is_valid() {
                            let _ = conn_viewport
                                .send_control(&ControlMessage::SetClientViewport(viewport))
                                .await;
                        }
                    }
                    return;
                }
                Err(_) => {
                    if let Some((w, h)) = pending.take() {
                        let viewport = Viewport::new(w, h);
                        if !viewport.is_valid() {
                            continue;
                        }
                        if let Err(e) = conn_viewport
                            .send_control(&ControlMessage::SetClientViewport(viewport))
                            .await
                        {
                            warn!(error = ?e, "SetClientViewport send failed; viewport task exiting");
                            return;
                        }
                        info!(width = w, height = h, "sent SetClientViewport to host");
                    }
                }
            }
        }
    });

    let on_event: tether_render::EventSink = Box::new(move |evt| {
        // Resize events fork off to the viewport debouncer; everything
        // else (input, cursor, focus) goes to the existing input task.
        // Render must not block on a slow consumer — UnboundedSender
        // drops on send-after-close, which is what we want when either
        // consumer task has exited.
        if let RenderEvent::Resized { width, height } = evt {
            let _ = viewport_tx.send((width, height));
            return;
        }
        let _ = events_tx.send(evt);
    });

    // Ctrl-C handler: winit's event loop owns the main thread once
    // `tether_render::run` starts, so an interrupt can't naturally fall
    // through to the `say_goodbye` block below. Catch it here, send
    // Goodbye on the host's behalf, then `std::process::exit`. The exit
    // skips destructors (wgpu device drop, winit window cleanup, tokio
    // runtime shutdown). That's safe today because this process owns
    // no state that needs cleanup before exit: no on-disk caches to
    // flush, no in-flight telemetry to drain, no other peers depending
    // on a graceful close beyond the Goodbye we already sent. Revisit
    // before adding anything in those categories.
    {
        let conn = conn.clone();
        tokio::spawn(async move {
            if let Err(e) = tokio::signal::ctrl_c().await {
                warn!(error = %e, "ctrl-c handler failed; exiting anyway");
                std::process::exit(1);
            }
            info!("ctrl-c received, sending Goodbye and exiting");
            reporter.emit(&EngineEvent::Disconnected {
                reason: "interrupted".to_string(),
            });
            say_goodbye(&conn, "client interrupted").await;
            std::process::exit(0);
        });
    }

    // When shell-driven, a `Stop` on stdin (or stdin EOF — the shell
    // died) closes the session the same way Ctrl-C does: the render loop
    // owns the main thread, so like the Ctrl-C handler this task sends
    // Goodbye and exits the process directly rather than trying to unwind
    // through winit.
    if reporter.is_json() {
        let conn = conn.clone();
        tokio::spawn(async move {
            wait_for_stdin_stop().await;
            info!("shell stop received; sending Goodbye and exiting");
            reporter.emit(&EngineEvent::Disconnected {
                reason: "stopped by shell".to_string(),
            });
            say_goodbye(&conn, "client stopped").await;
            std::process::exit(0);
        });
    }

    // Render loop blocks until the user closes the window. The
    // host's advertised color spec drives the renderer's EOTF
    // dispatch — for desktop captures (`sdr_desktop`) this is the
    // sRGB path, eliminating the BT.709-vs-sRGB transfer-curve
    // mismatch the spec-blind chain previously had to absorb.
    let render_result = tether_render::run(
        "tether-client",
        (INITIAL_WIDTH, INITIAL_HEIGHT),
        server_hello.color_space,
        negotiated_profile.chroma,
        negotiated_profile.bit_depth,
        frames,
        cursor_channel,
        Some(on_event),
    );

    // Normal window-close path. Notify the host so it can tear down its
    // capture, encoder, and libei session immediately instead of waiting
    // for QUIC's idle timeout.
    let reason = match &render_result {
        Ok(()) => "window closed".to_string(),
        Err(e) => format!("render error: {e}"),
    };
    reporter.emit(&EngineEvent::Disconnected { reason });
    say_goodbye(&conn, "client closing").await;

    // In `--ipc` mode the stdin stop-watcher parks a `tokio::io::stdin()`
    // blocking read that can't be cancelled, so letting `main` return would
    // hang the runtime's drop on that stuck thread — exit the process directly
    // instead (same rationale as the Ctrl-C handler above). In plain CLI mode
    // there's no such watcher, so return normally and let destructors run —
    // notably `_tracing_guard`, whose drop flushes buffered log lines.
    if ipc {
        std::process::exit(if render_result.is_ok() { 0 } else { 1 });
    }
    match render_result {
        Ok(()) => Ok(()),
        Err(e) => Err(anyhow::anyhow!("render error: {e}")),
    }
}

/// Block until the shell sends a `Stop` command, closes our stdin (the
/// shell process died), or stdin errors — all three mean "shut down."
/// Mirrors the host's `spawn_stdin_stop_watcher`, but the client races it
/// against the render loop via a direct process exit rather than a
/// shutdown notify.
async fn wait_for_stdin_stop() {
    use tokio::io::{AsyncBufReadExt, BufReader};
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match tether_ipc::ShellCommand::from_line(line) {
                    Ok(tether_ipc::ShellCommand::Stop) => return,
                    // Host-only pairing commands: a well-formed command that
                    // doesn't apply to the client. Ignore rather than error.
                    Ok(tether_ipc::ShellCommand::StartPairing { .. })
                    | Ok(tether_ipc::ShellCommand::RevokePeer { .. })
                    | Ok(tether_ipc::ShellCommand::ListPeers) => {
                        warn!(?line, "ignoring host-only command on client stdin");
                    }
                    Err(e) => {
                        warn!(error = %e, line, "ignoring unrecognized stdin command");
                    }
                }
            }
            // EOF or read error: the shell is gone — treat as stop.
            Ok(None) | Err(_) => return,
        }
    }
}

/// Send a `ControlMessage::Goodbye` and close the connection. The
/// sleep between send and close is a known-imperfect interim: the
/// correct primitive is "wait for the peer to ack the Goodbye bytes,"
/// which Quinn exposes only via `SendStream::finish` + `stopped()` on
/// a stream we're willing to terminate. Our control stream lives for
/// the whole connection, so a clean implementation needs a dedicated
/// uni stream for shutdown — tracked as a separate task and required
/// before real-network testing.
///
/// Until then: wait `2 * rtt`, floored at 20 ms and capped at 200 ms.
/// `send_control` returns when bytes are in Quinn's send buffer (not
/// when ack'd); two round trips covers send + propagation + ack on
/// any sane link. The floor handles loopback (RTT in the microseconds)
/// and the cap bounds shutdown latency on a high-RTT link where the
/// peer may already be unreachable anyway.
async fn say_goodbye(conn: &tether_transport::Connection, reason: &str) {
    say_goodbye_with_code(conn, reason, GoodbyeCode::Clean).await;
}

/// Variant that lets the caller signal *why* the session ended.
/// Use [`GoodbyeCode::InternalError`] for fatal local failures (decoder
/// init failed, render thread died) so the host's session-end log
/// distinguishes a genuine crash from a user closing the window.
async fn say_goodbye_with_code(
    conn: &tether_transport::Connection,
    reason: &str,
    code: GoodbyeCode,
) {
    use std::time::Duration;
    let msg = ControlMessage::Goodbye {
        reason: reason.to_string(),
        code,
    };
    if let Err(e) = conn.send_control(&msg).await {
        warn!(error = ?e, "send Goodbye failed; host will fall back to timeout");
    } else {
        let wait = (2 * conn.rtt()).clamp(Duration::from_millis(20), Duration::from_millis(200));
        tokio::time::sleep(wait).await;
    }
    conn.close(0, reason.as_bytes());
}

fn init_tracing(ipc: bool) -> tracing_appender::non_blocking::WorkerGuard {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    // Non-blocking writer offloads formatting and the write syscall
    // to a dedicated worker thread. Critical for the client because
    // the FFmpeg log callback bridges into tracing from the decoder
    // thread (which is also the QUIC datagram recv loop's thread);
    // a synchronous writer there would stall the recv loop and the
    // input-send task during decode-error storms.
    //
    // In IPC mode stdout is reserved for the JSON-lines protocol the
    // shell parses, so logs go to stderr instead. Both branches yield
    // the same `(NonBlocking, WorkerGuard)` type — `non_blocking` erases
    // the inner writer behind its channel.
    let (writer, guard) = if ipc {
        tracing_appender::non_blocking(std::io::stderr())
    } else {
        tracing_appender::non_blocking(std::io::stdout())
    };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(writer)
        .init();
    guard
}

/// Parsed command-line arguments for the client.
#[derive(Debug)]
struct CliArgs {
    addr: SocketAddr,
    /// Explicit host fingerprint (optional): pins the host cert for a reconnect
    /// without consulting known-hosts. Mutually informative with `--pin`, which
    /// takes precedence.
    fingerprint_hex: Option<String>,
    /// `--pin <PIN>`: present ⇒ first-contact pairing mode.
    pin: Option<String>,
    /// `--label <name>`: display name to record for the host on first pair.
    label: Option<String>,
    /// Play host audio when the host advertises it. On by default; `--no-audio`
    /// disables the client's playback path entirely.
    audio: bool,
}

/// Parse the client CLI. Positional[0] is the host address (required);
/// positional[1] is an optional explicit fingerprint. `--pin`/`--label` consume
/// the following token as their value; `--ipc` is handled earlier and ignored
/// here; other `--flags` are ignored with a warning.
fn parse_cli_args(raw_args: &[String]) -> anyhow::Result<CliArgs> {
    let mut positional: Vec<&str> = Vec::new();
    let mut pin = None;
    let mut label = None;
    let mut audio = true;
    let mut it = raw_args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--ipc" => {}
            "--no-audio" => audio = false,
            "--pin" => pin = Some(take_flag_value(&mut it, "--pin")?),
            "--label" => label = Some(take_flag_value(&mut it, "--label")?),
            other if other.starts_with("--") => {
                warn!(flag = other, "ignoring unknown flag");
            }
            other => positional.push(other),
        }
    }
    if matches!(pin.as_deref(), Some("")) {
        anyhow::bail!("--pin value must not be empty");
    }
    let addr: SocketAddr = positional
        .first()
        .ok_or_else(|| anyhow::anyhow!("missing host address argument"))?
        .parse()?;
    let fingerprint_hex = positional.get(1).map(|s| s.to_string());
    Ok(CliArgs {
        addr,
        fingerprint_hex,
        pin,
        label,
        audio,
    })
}

/// Consume the next token as a flag's value. Rejects a missing value and a
/// value that looks like another flag (e.g. `--pin --label x`), which would
/// otherwise silently swallow the next flag as the value.
fn take_flag_value<'a>(
    it: &mut impl Iterator<Item = &'a String>,
    flag: &str,
) -> anyhow::Result<String> {
    match it.next() {
        Some(v) if v.starts_with("--") => {
            anyhow::bail!("{flag} requires a value, but got the flag '{v}'")
        }
        Some(v) => Ok(v.clone()),
        None => anyhow::bail!("{flag} requires a value"),
    }
}

/// `(frame_seq, opus_payload)` handed from the datagram recv loop to the audio
/// playback thread; the sequence number drives gap-detected concealment.
type AudioFrameMsg = (u32, Vec<u8>);

/// If audio is enabled and the host advertised an `AudioConfig`, spawn the
/// playback thread and return a sender the recv loop feeds Opus frames to,
/// plus `true` (audio active). Otherwise return `(None, false)` and run
/// video-only. Never fatal — a missing output device just means no sound.
fn setup_audio_playback(
    enabled: bool,
    server_hello: &tether_protocol::control::ServerHelloV1,
) -> (Option<crossbeam_channel::Sender<AudioFrameMsg>>, bool) {
    if !enabled {
        info!("audio disabled via --no-audio");
        return (None, false);
    }
    let Some(audio_cfg) = server_hello
        .extensions
        .get(tether_protocol::audio::AUDIO_CONFIG_EXTENSION_KEY)
        .and_then(|b| tether_protocol::decode::<tether_protocol::audio::AudioConfig>(b).ok())
    else {
        info!("host advertised no audio; running video-only");
        return (None, false);
    };

    // Validate the host-advertised config before it sizes any buffer. The
    // `AudioConfig` is attacker-controllable; an out-of-range `sample_rate_hz`
    // would size the playback ring's allocation into gigabytes (OOM), and an
    // out-of-range `channels` would drive decoder plane indexing.
    // v1 ships mono/stereo at an Opus-native rate; reject anything else.
    // This is the accept-list the client honours from the host. 8 kHz and
    // 16 kHz are valid Opus rates but are excluded as a policy choice — no
    // current capture backend produces them, so narrowing the negotiation
    // window keeps the surface small. (`OpusDecoder::new` independently rejects
    // any (rate, frame-duration) pair whose frame size libopus would refuse, so
    // a future config bug fails loudly rather than dropping every packet.)
    const OPUS_RATES: [u32; 3] = [12_000, 24_000, 48_000];
    if !OPUS_RATES.contains(&audio_cfg.sample_rate_hz) || !(1..=2).contains(&audio_cfg.channels) {
        warn!(
            sample_rate = audio_cfg.sample_rate_hz,
            channels = audio_cfg.channels,
            "host advertised an unsupported audio config; running video-only"
        );
        return (None, false);
    }

    let opus_cfg = tether_audio::OpusConfig {
        sample_rate: audio_cfg.sample_rate_hz,
        channels: audio_cfg.channels,
        ..tether_audio::OpusConfig::default()
    };
    // ~640 ms of 10 ms frames — generous headroom so the recv loop never
    // blocks; the jitter buffer downstream bounds actual playback latency.
    let (tx, rx) = crossbeam_channel::bounded::<AudioFrameMsg>(64);
    match std::thread::Builder::new()
        .name("tether-client-audio".into())
        .spawn(move || run_audio_playback(opus_cfg, rx))
    {
        Ok(_) => {
            info!(
                sample_rate = audio_cfg.sample_rate_hz,
                channels = audio_cfg.channels,
                "audio playback enabled"
            );
            (Some(tx), true)
        }
        Err(e) => {
            warn!(error = %e, "failed to spawn audio thread; running video-only");
            (None, false)
        }
    }
}

/// Audio playback thread: decode incoming Opus frames, conceal sequence gaps,
/// and push PCM into the playback ring feeding the cpal output stream. Owns the
/// `AudioPlayer` (and thus the cpal stream) for its lifetime; returns when the
/// channel closes (session ending), dropping the player to stop playback.
fn run_audio_playback(
    cfg: tether_audio::OpusConfig,
    rx: crossbeam_channel::Receiver<AudioFrameMsg>,
) {
    let (player, sink) = match tether_audio::AudioPlayer::with_defaults(cfg) {
        Ok(pair) => pair,
        Err(e) => {
            warn!(error = %e, "audio output device unavailable; no playback");
            return;
        }
    };
    let mut decoder = match tether_audio::OpusDecoder::new(cfg) {
        Ok(d) => d,
        Err(e) => {
            warn!(error = %e, "opus decoder init failed; no playback");
            return;
        }
    };

    // Cap concealment per packet so a crafted/huge sequence jump can't insert a
    // long silence or spin. ~80 ms at 10 ms frames.
    const MAX_CONCEAL: u32 = 8;
    let mut last_seq: Option<u32> = None;

    // Periodic playback-health snapshot. The ring's drift/underrun behaviour is
    // otherwise invisible: cap-and-drop silently absorbs overflow and silence
    // fills underruns, so without this a multi-minute session shows nothing.
    // Frame-driven (audio arrives ~every 10 ms), like the host's send-stats —
    // when audio stops, so does the log. 2 s matches the video stats cadence.
    const STATS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);
    let mut last_stats_log = std::time::Instant::now();
    // Peak |sample| of decoded audio since the last log. ~0 means the frames
    // arriving are silent (the silence is upstream — capture/encode), which
    // distinguishes that from a local output-routing problem where non-zero
    // audio is played but not heard.
    let mut peak: f32 = 0.0;

    while let Ok((seq, payload)) = rx.recv() {
        if let Some(prev) = last_seq {
            // Forward distance from the last delivered frame. A small positive
            // gap is real loss → conceal the missing frames. A gap of 1 is the
            // in-order case (nothing missing); 0 (duplicate) or a large value
            // (reordered late packet, or a future epoch reset) is *not* treated
            // as loss, so a backward/late packet can't trigger spurious silence.
            let gap = seq.wrapping_sub(prev);
            if (2..=MAX_CONCEAL + 1).contains(&gap) {
                for _ in 0..gap - 1 {
                    sink.submit(&decoder.conceal());
                }
            }
        }
        match decoder.decode(&payload) {
            Ok(pcm) => {
                for &s in &pcm.samples {
                    peak = peak.max(s.abs());
                }
                sink.submit(&pcm);
            }
            Err(e) => warn!(error = %e, "opus decode failed; dropping audio frame"),
        }
        last_seq = Some(seq);

        if last_stats_log.elapsed() >= STATS_INTERVAL {
            let (underruns, dropped_samples, buffered) = sink.stats();
            // `buffered` is interleaved samples; convert to wall-clock so drift
            // toward an underrun (→ 0 ms) or the latency cap is legible at a
            // glance. Cumulative counters climbing across snapshots flags a
            // clock-drift problem the cap-and-drop policy is papering over.
            let frames_buffered = buffered / usize::from(cfg.channels).max(1);
            let buffered_ms = frames_buffered * 1000 / (cfg.sample_rate as usize).max(1);
            info!(
                underruns,
                dropped_samples,
                buffered_ms,
                peak = format!("{peak:.4}"),
                "audio playback stats"
            );
            last_stats_log = std::time::Instant::now();
            peak = 0.0;
        }
    }

    // Channel closed: drop the player to stop the output stream.
    drop(player);
}

/// Directory the client caches its identity (`client_cert.der`/`client_key.der`)
/// and `known_hosts.json` in. Delegates to the shared
/// [`tether_pairing::config_dir`] so the host, the client, and the Tauri shell
/// all resolve the same location.
fn client_config_dir() -> anyhow::Result<PathBuf> {
    Ok(tether_pairing::config_dir()?)
}

/// Seconds since the Unix epoch, for stamping when a host was paired. A
/// pre-epoch clock clamps to 0 rather than failing the connect.
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn hex_decode(s: &str) -> anyhow::Result<[u8; 32]> {
    if s.len() != 64 {
        anyhow::bail!("fingerprint must be 64 hex chars, got {}", s.len());
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        let hi = u8::from_str_radix(&s[i * 2..=i * 2], 16)
            .map_err(|_| anyhow::anyhow!("bad hex at byte {i}"))?;
        let lo = u8::from_str_radix(&s[i * 2 + 1..=i * 2 + 1], 16)
            .map_err(|_| anyhow::anyhow!("bad hex at byte {i}"))?;
        *byte = (hi << 4) | lo;
    }
    Ok(out)
}

/// Cross-crate decode→render format-agreement check for the Windows
/// native D3D11 client. Two crates each carry a `profile → DXGI_FORMAT`
/// table for the GPU-resident decode path:
///
///   * `tether_codec::d3d11::expected_decode_dxgi_format` — the
///     `DXGI_FORMAT` the D3D11VA decoder emits as `D3D11DecodedTexture`
///     for a negotiated profile (NV12 for 8-bit, P010 for 10-bit; `None`
///     for the 4:2:0-only platform's unsupported chromas).
///   * `tether_render::decode_plane_srv_formats` — the plane-SRV formats
///     the native D3D11 renderer binds to sample that texture, or `None`
///     if it rejects the format.
///
/// The chain is `D3D11VA decode → shared handle → renderer import`. The
/// invariant: **every format the decoder can emit for a negotiable
/// profile is one the renderer accepts.** A miss silently drops (or
/// errors out) every frame after decode — the exact shape of macOS bug
/// `621badc`, where the renderer rejected the decoder's `'x420'` for the
/// first live Main10 session. The two tables sit on parallel tracks
/// (different crates, only joined at runtime), so the `#[ignore]`d
/// hardware roundtrip can't catch their drift in default CI — this
/// no-GPU unit test does, communicating via the neutral `DXGI_FORMAT`
/// `u32` so neither test crate needs the `windows` dependency.
#[cfg(all(test, target_os = "windows"))]
mod windows_format_tables {
    use tether_codec::d3d11::expected_decode_dxgi_format;
    use tether_probe::PROFILE_PREFERENCE;
    use tether_render::decode_plane_srv_formats;

    #[test]
    fn decoder_output_is_subset_of_renderer_accept() {
        let mut covered = 0;
        for profile in PROFILE_PREFERENCE {
            let Some(fmt) = expected_decode_dxgi_format(*profile) else {
                // 4:4:4 / unsupported: no Windows decode path, nothing to
                // hand the renderer. Verified in tether-codec's co-located
                // `expected_decode_format_rejects_unmodeled_profiles`.
                continue;
            };
            assert!(
                decode_plane_srv_formats(fmt).is_some(),
                "D3D11VA decoder emits DXGI format {fmt:#x} for profile {profile:?}, but the \
                 native D3D11 renderer's import path rejects it — every frame of a negotiated \
                 session would be dropped after decode (bug 621badc shape)."
            );
            covered += 1;
        }
        // Guard against a vacuous pass: the negotiator's preference list
        // must contain at least the 8-bit and 10-bit 4:2:0 profiles the
        // Windows client actually decodes, or the loop above asserts
        // nothing.
        assert!(
            covered >= 2,
            "expected ≥2 Windows-decodable profiles (NV12 8-bit + P010 10-bit) in \
             PROFILE_PREFERENCE, only {covered} produced a decode format"
        );
    }
}

#[cfg(test)]
mod arg_tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn positional_addr_and_optional_fingerprint() {
        let parsed = parse_cli_args(&args(&["127.0.0.1:7654"])).expect("addr only");
        assert_eq!(parsed.addr.to_string(), "127.0.0.1:7654");
        assert!(parsed.fingerprint_hex.is_none());
        assert!(parsed.pin.is_none());

        let parsed = parse_cli_args(&args(&["127.0.0.1:7654", "deadbeef"])).expect("addr + fp");
        assert_eq!(parsed.fingerprint_hex.as_deref(), Some("deadbeef"));
    }

    #[test]
    fn pin_and_label_consume_their_values() {
        let parsed = parse_cli_args(&args(&[
            "--pin",
            "12345678",
            "127.0.0.1:7654",
            "--label",
            "my laptop",
        ]))
        .expect("flags + positional");
        assert_eq!(parsed.pin.as_deref(), Some("12345678"));
        assert_eq!(parsed.label.as_deref(), Some("my laptop"));
        assert_eq!(parsed.addr.to_string(), "127.0.0.1:7654");
    }

    #[test]
    fn flag_as_pin_value_is_rejected() {
        // `--pin --label x` must not swallow `--label` as the PIN.
        let err = parse_cli_args(&args(&["--pin", "--label", "x", "127.0.0.1:7654"]))
            .expect_err("flag-as-value must error");
        assert!(err.to_string().contains("--pin"));
    }

    #[test]
    fn empty_pin_is_rejected() {
        let err =
            parse_cli_args(&args(&["--pin", "", "127.0.0.1:7654"])).expect_err("empty pin errors");
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn missing_addr_is_rejected() {
        assert!(parse_cli_args(&args(&["--pin", "12345678"])).is_err());
    }

    #[test]
    fn missing_flag_value_is_rejected() {
        assert!(parse_cli_args(&args(&["127.0.0.1:7654", "--pin"])).is_err());
    }

    #[test]
    fn recovery_fires_only_on_a_pruned_frame_not_a_straggler() {
        // (frames_dropped, fragments_lost)
        // A frame pruned incomplete (frames_dropped++) → recover.
        assert!(recovery_warranted((0, 0), (1, 0)));
        // A stale straggler / malformed packet (fragments_lost++ only) →
        // do NOT recover; it's not independently actionable.
        assert!(!recovery_warranted((0, 0), (0, 1)));
        // Both moving still recovers — the dropped frame is the reason.
        assert!(recovery_warranted((0, 0), (1, 1)));
        // Nothing moved → nothing to recover.
        assert!(!recovery_warranted((3, 7), (3, 7)));
    }
}
