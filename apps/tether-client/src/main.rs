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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Instant;

use bytes::Bytes;
use crossbeam_channel::bounded;
use tether_decode::{DecodeEvent, DecodeJob, EpochDropReason};
use tether_input::{WinitTranslator, WireEvent};
use tether_ipc::{EngineEvent, Reporter};
use tether_protocol::audio::AudioPacket;
use tether_protocol::control::{
    ClientDisplayMetrics, ClockSync, ControlMessage, DisplayDescriptor, GoodbyeCode, ServerHello,
    VideoStreamId, Viewport, CLOCK_SYNC_PROBE_SAMPLES,
};
use tether_protocol::video::{FrameReassembler, VideoPacket};
use tether_protocol::MonoNanos;
use tether_render::LatestFrame;
use tether_render::RenderEvent;
use tether_session::{
    log_peer_session_summary, ClientSession, ClientSessionConfig, ConnectError, SessionSummaryState,
};
use tether_transport::{Client, Datagram, ServerAuth};
use tokio::sync::{mpsc, watch};
use tracing::{debug, error, info, warn};

mod client_pairing;
use client_pairing::HostAuth;

// Fallback startup window size when the host cannot advertise a valid display.
// Normal sessions use the host primary display's physical mode and let
// tether-render cap that to the client monitor for FitNoUpscale startup.
const FALLBACK_INITIAL_WIDTH: u32 = 1280;
const FALLBACK_INITIAL_HEIGHT: u32 = 720;
const CLOCK_RESYNC_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

#[derive(Default)]
struct ClockResyncState {
    pending: Vec<MonoNanos>,
    samples: Vec<ClockSync>,
}

fn lock_clock_resync(state: &Mutex<ClockResyncState>) -> MutexGuard<'_, ClockResyncState> {
    state.lock().unwrap_or_else(|poisoned| {
        warn!("clock resync lock poisoned; recovering pending probe state");
        poisoned.into_inner()
    })
}

fn read_clock_sync(state: &RwLock<ClockSync>) -> RwLockReadGuard<'_, ClockSync> {
    state.read().unwrap_or_else(|poisoned| {
        warn!("clock sync read lock poisoned; recovering last sample");
        poisoned.into_inner()
    })
}

fn write_clock_sync(state: &RwLock<ClockSync>) -> RwLockWriteGuard<'_, ClockSync> {
    state.write().unwrap_or_else(|poisoned| {
        warn!("clock sync write lock poisoned; recovering last sample");
        poisoned.into_inner()
    })
}

/// Whether a `FrameReassembler::handle()` loss-counter delta warrants a
/// `RequestRecovery`. `before`/`after` are reassembler
/// `(incomplete_frames, fragment_loss_events)` snapshots taken around the
/// `handle()` call.
///
/// Fires only on an incomplete-frame increase — a frame started but pruned
/// before completing, the genuine "this frame will never arrive" signal.
/// Never fires on fragment-loss events alone: that counts stale stragglers
/// (a late fragment for an already-finalized or already-pruned frame) and
/// malformed packets, neither of which is independently actionable. Triggering
/// on them would emit spurious recovery IDRs that add bandwidth and worsen
/// congestion on an already-lossy path; a frame that truly won't complete
/// still bumps the incomplete-frame counter when it's pruned, so the real
/// signal survives.
fn recovery_warranted(before: (u64, u64), after: (u64, u64)) -> bool {
    after.0 > before.0
}

fn drain_latest_valid_viewport(
    mut latest: Viewport,
    rx: &mut mpsc::UnboundedReceiver<(u32, u32)>,
) -> Viewport {
    while let Ok((width, height)) = rx.try_recv() {
        let candidate = Viewport::new(width, height);
        if candidate.is_valid() {
            latest = candidate;
        }
    }
    latest
}

fn initial_video_size_px_from_displays(displays: &[DisplayDescriptor]) -> (u32, u32) {
    displays
        .iter()
        .find(|display| {
            display.primary && display.current_mode.width > 0 && display.current_mode.height > 0
        })
        .or_else(|| {
            displays
                .iter()
                .find(|display| display.current_mode.width > 0 && display.current_mode.height > 0)
        })
        .map(|display| (display.current_mode.width, display.current_mode.height))
        .unwrap_or((FALLBACK_INITIAL_WIDTH, FALLBACK_INITIAL_HEIGHT))
}

fn should_send_viewport(last_sent: Option<Viewport>, next: Viewport) -> bool {
    next.is_valid() && last_sent != Some(next)
}

fn should_send_client_display_metrics(
    last_sent: Option<&ClientDisplayMetrics>,
    next: &ClientDisplayMetrics,
) -> bool {
    last_sent != Some(next)
}

#[allow(clippy::cast_precision_loss)]
fn ns_to_ms(ns: u64, samples: u64) -> f64 {
    if samples == 0 {
        0.0
    } else {
        ns as f64 / 1_000_000.0
    }
}

#[derive(Default)]
struct DecodeEventWindow {
    latency_sum_ns: u64,
    latency_min_ns: u64,
    latency_max_ns: u64,
    decode_errors: u32,
    render_drops: u32,
    idr_requests: u32,
    queue_drops: u32,
    stale_epoch_drops: u32,
    epoch_throttle_drops: u32,
    completion_count: u64,
}

impl DecodeEventWindow {
    fn record_completion(&mut self, c: tether_decode::DecodeCompletion) {
        self.completion_count = self.completion_count.saturating_add(1);
        self.latency_sum_ns = self.latency_sum_ns.saturating_add(c.decode_duration_ns);
        if self.completion_count == 1 {
            self.latency_min_ns = c.decode_duration_ns;
        } else {
            self.latency_min_ns = self.latency_min_ns.min(c.decode_duration_ns);
        }
        self.latency_max_ns = self.latency_max_ns.max(c.decode_duration_ns);
        if c.decode_err || c.soft_failure {
            self.decode_errors = self.decode_errors.saturating_add(1);
        }
        self.render_drops = self.render_drops.saturating_add(c.render_drops);
        if c.idr_request_fired {
            self.idr_requests = self.idr_requests.saturating_add(1);
        }
    }

    fn record_epoch_drop(&mut self, reason: EpochDropReason) {
        match reason {
            EpochDropReason::Stale => {
                self.stale_epoch_drops = self.stale_epoch_drops.saturating_add(1);
            }
            EpochDropReason::RebuildRateLimited => {
                self.epoch_throttle_drops = self.epoch_throttle_drops.saturating_add(1);
            }
        }
    }
}

fn drain_decode_events(
    rx: &crossbeam_channel::Receiver<DecodeEvent>,
    summary: &SessionSummaryState,
    mut window: Option<&mut DecodeEventWindow>,
) {
    while let Ok(event) = rx.try_recv() {
        match event {
            DecodeEvent::Completion(c) => {
                if c.decode_err || c.soft_failure {
                    summary.video.decode_errors.fetch_add(1, Ordering::Relaxed);
                }
                summary
                    .video
                    .render_drop_frames
                    .fetch_add(u64::from(c.render_drops), Ordering::Relaxed);
                if c.idr_request_fired {
                    summary.video.idr_requests.fetch_add(1, Ordering::Relaxed);
                }
                if let Some(w) = window.as_deref_mut() {
                    w.record_completion(c);
                }
            }
            DecodeEvent::EpochDrop { reason, .. } => {
                match reason {
                    EpochDropReason::Stale => {
                        summary
                            .video
                            .decode_stale_epoch_drop_frames
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    EpochDropReason::RebuildRateLimited => {
                        summary
                            .video
                            .decode_epoch_throttle_drop_frames
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                if let Some(w) = window.as_deref_mut() {
                    w.record_epoch_drop(reason);
                }
            }
        }
    }
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
    let stdin_stop = spawn_stdin_stop_signal(reporter.is_json());

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
    let connect_result = if let Some(rx) = stdin_stop.clone() {
        tokio::select! {
            () = wait_for_stdin_stop_signal(rx) => {
                reporter.emit(&EngineEvent::Disconnected {
                    reason: "stopped by shell".to_string(),
                });
                return Ok(());
            }
            result = client.connect_pending(addr, "tether-host", server_auth) => result,
        }
    } else {
        client
            .connect_pending(addr, "tether-host", server_auth)
            .await
    };
    let pending = match connect_result {
        Ok(p) => p,
        Err(e) => {
            reporter.emit(&EngineEvent::Error {
                message: format!("connect failed: {e}"),
            });
            return Err(e.into());
        }
    };
    let is_first_contact = matches!(mode, HostAuth::FirstContact { .. });
    let pairing_result = if let Some(rx) = stdin_stop.clone() {
        tokio::select! {
            () = wait_for_stdin_stop_signal(rx) => {
                reporter.emit(&EngineEvent::Disconnected {
                    reason: "stopped by shell".to_string(),
                });
                return Ok(());
            }
            result = client_pairing::establish(pending, &mode, client_fp) => result,
        }
    } else {
        client_pairing::establish(pending, &mode, client_fp).await
    };
    let (conn, host_fp) = match pairing_result {
        Ok(pair) => pair,
        Err(e) => {
            reporter.emit(&EngineEvent::Error {
                message: format!("pairing failed: {e}"),
            });
            return Err(e);
        }
    };

    // Persist known-hosts: on first contact, pin the host so the next connect
    // is one-click; on every successful connect to an entry that still exists,
    // stamp the time so the shell's address book can show recency. Reload right
    // before saving so a shell-side Forget during connect does not get
    // resurrected by the startup snapshot.
    {
        let addr_key = addr.to_string();
        let now = unix_now();
        let save_result = (|| -> std::io::Result<()> {
            let mut known_hosts = tether_pairing::KnownHosts::load(&known_hosts_path)?;
            if is_first_contact {
                let label = label.unwrap_or_else(|| addr_key.clone());
                known_hosts.insert(addr_key.clone(), &host_fp, label, now);
                known_hosts.set_last_connected(&addr_key, now);
                known_hosts.save(&known_hosts_path)?;
            } else if known_hosts.fingerprint(&addr_key) == Some(host_fp) {
                known_hosts.set_last_connected(&addr_key, now);
                known_hosts.save(&known_hosts_path)?;
            } else {
                debug!(
                    host = %addr_key,
                    "connected host is no longer in known-hosts; not rewriting forgotten entry"
                );
            }
            Ok(())
        })();
        if let Err(e) = save_result {
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
    // Renderer capability gate. The codec decode probe can't see whether the
    // renderer can present a profile, so each backend answers per profile. That
    // matters for Linux 4:4:4 10-bit: it uses a packed 10:10:10:2 texture path,
    // not the 16-bit biplanar feature gate used by P010/P410.
    let before_render_gate = client_decode_profiles.len();
    let mut renderable_profiles = Vec::with_capacity(client_decode_profiles.len());
    for profile in client_decode_profiles {
        if tether_render::supports_video_profile_render(profile).await {
            renderable_profiles.push(profile);
        }
    }
    let dropped = before_render_gate - renderable_profiles.len();
    client_decode_profiles = renderable_profiles;
    if dropped > 0 {
        info!(
            dropped_profiles = dropped,
            "renderer cannot present some decoded profiles; dropping them \
             from the decode-capability advert"
        );
    }
    let forced_video_profile = match tether_probe::forced_video_profile_from_env() {
        Ok(profile) => profile,
        Err(e) => {
            reporter.emit(&EngineEvent::Error { message: e.clone() });
            anyhow::bail!(e);
        }
    };
    if let Some(profile) = forced_video_profile {
        let before = client_decode_profiles.len();
        client_decode_profiles.retain(|p| *p == profile);
        info!(
            forced_profile = ?profile,
            profiles_before_force = before,
            profiles_after_force = client_decode_profiles.len(),
            env = tether_probe::FORCE_VIDEO_PROFILE_ENV,
            "forced video profile applied to client decode-capability advert"
        );
    }
    if client_decode_profiles.is_empty() {
        let message = if let Some(profile) = forced_video_profile {
            format!(
                "{}={profile:?} was requested, but this client cannot decode/render it",
                tether_probe::FORCE_VIDEO_PROFILE_ENV
            )
        } else {
            "no hardware video decoder is available on this client \
                 (no codec in PROFILE_PREFERENCE constructed). Tether requires \
                 GPU decode; there is no software fallback."
                .to_string()
        };
        reporter.emit(&EngineEvent::Error {
            message: message.clone(),
        });
        anyhow::bail!(message);
    }

    // Application-layer handshake: identify ourselves, advertise our
    // decode profiles, resolve + validate the host's pick, and prime
    // the host with a `ForceIdr` so the next frame is a keyframe.
    // The clock-sync probe round-trip happens immediately after the
    // typed handshake, so latency logs are wall-clock-accurate from
    // the first frame.
    // ClientSession takes the channel through the `ControlChannel`
    // trait object so it's mockable in tests. The original
    // `Arc<Connection>` stays in `conn` for the rest of `main` — the
    // recv tasks below use concrete-`Connection` methods (datagram,
    // input send, connection stats) that aren't on the trait.
    let session = ClientSession::connect(
        conn.clone() as Arc<dyn tether_transport::ControlChannel>,
        ClientSessionConfig {
            client_name: "tether-client".to_string(),
            client_decode_profiles: client_decode_profiles.clone(),
            // Startup video is gated on the first real renderer viewport.
            // A guessed logical size here causes the host to build a doomed
            // first encoder epoch on HiDPI clients, then immediately rebuild
            // when the physical viewport arrives.
            viewport: None,
        },
    )
    .await;
    let session = match session {
        Ok(s) => s,
        Err(e) => {
            let err = match e {
                ConnectError::ProfileNotAdvertised { .. }
                | ConnectError::UnknownBitDepth(_, _)
                | ConnectError::HandshakeRejected { .. }
                | ConnectError::PeerGoodbyeDuringClockProbe { .. }
                | ConnectError::ClockProbeIgnoredMessageLimit { .. } => anyhow::anyhow!("{e}"),
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
        negotiated_video: _negotiated_video,
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
    let session_summary = Arc::new(SessionSummaryState::new(
        "client",
        negotiated_profile,
        false,
    ));
    let shutdown_notice_sent = Arc::new(AtomicBool::new(false));
    let clock_sync_state = Arc::new(RwLock::new(clock_sync));
    let clock_resync_state = Arc::new(Mutex::new(ClockResyncState::default()));

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

    // Renderer resize events drive startup video readiness. The host must not
    // begin streaming until it has the client's real physical viewport, so the
    // viewport task sends SetViewportHint before StreamReady.
    let (viewport_tx, mut viewport_rx) = mpsc::unbounded_channel::<(u32, u32)>();
    let (display_metrics_tx, mut display_metrics_rx) =
        mpsc::unbounded_channel::<tether_protocol::control::ClientDisplayMetrics>();
    let (decoder_ready_for_startup_tx, decoder_ready_for_startup_rx) = watch::channel(false);
    let (decode_event_tx, decode_event_rx) = crossbeam_channel::unbounded::<DecodeEvent>();
    let decode_event_rx = Arc::new(decode_event_rx);

    // Receive-side control loop. Today the host doesn't initiate any
    // typed messages we need to act on, but the Extension escape and
    // future variants (CursorShape, DisplayList, StreamPause/Resume)
    // arrive here, so the loop exists from V1 onward.
    {
        let conn = conn.clone();
        let clock_resync_state = clock_resync_state.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(CLOCK_RESYNC_INTERVAL);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            ticker.tick().await; // Skip the immediate tick; handshake just sampled.
            loop {
                ticker.tick().await;
                {
                    let mut state = lock_clock_resync(&clock_resync_state);
                    if !state.pending.is_empty() {
                        warn!(
                            pending = state.pending.len(),
                            samples = state.samples.len(),
                            "clock resync probe burst did not complete before next interval; restarting"
                        );
                        state.pending.clear();
                        state.samples.clear();
                    }
                    state.pending.reserve(CLOCK_SYNC_PROBE_SAMPLES);
                    state.samples.reserve(CLOCK_SYNC_PROBE_SAMPLES);
                }

                for _ in 0..CLOCK_SYNC_PROBE_SAMPLES {
                    let t0 = MonoNanos::now();
                    {
                        let mut state = lock_clock_resync(&clock_resync_state);
                        state.pending.push(t0);
                    }
                    if let Err(e) = conn
                        .send_control(&ControlMessage::ClockProbeRequest { t0_sender: t0 })
                        .await
                    {
                        let mut state = lock_clock_resync(&clock_resync_state);
                        state.pending.clear();
                        state.samples.clear();
                        warn!(error = ?e, "clock resync probe send failed; ending resync task");
                        return;
                    }
                }
            }
        });
    }
    {
        let conn = conn.clone();
        let cursor_channel_ctrl = cursor_channel.clone();
        let session_summary = session_summary.clone();
        let shutdown_notice_sent = shutdown_notice_sent.clone();
        let decode_event_rx = decode_event_rx.clone();
        let clock_sync_state = clock_sync_state.clone();
        let clock_resync_state = clock_resync_state.clone();
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
                    Ok(ControlMessage::ClockProbeResponse(probe)) => {
                        let selected = {
                            let mut state = lock_clock_resync(&clock_resync_state);
                            let Some(pos) =
                                state.pending.iter().position(|t0| *t0 == probe.t0_sender)
                            else {
                                tracing::trace!("unsolicited clock probe response; ignoring");
                                continue;
                            };
                            let t0 = state.pending.swap_remove(pos);
                            let t3 = MonoNanos::now();
                            state.samples.push(ClockSync::from_probe(
                                t0,
                                probe.t1_receiver_recv,
                                probe.t2_receiver_send,
                                t3,
                            ));
                            if state.samples.len() == CLOCK_SYNC_PROBE_SAMPLES {
                                let min_rtt =
                                    state.samples.iter().map(|s| s.rtt_nanos).min().unwrap_or(0);
                                let max_rtt =
                                    state.samples.iter().map(|s| s.rtt_nanos).max().unwrap_or(0);
                                let samples = std::mem::take(&mut state.samples);
                                state.pending.clear();
                                ClockSync::best_sample(samples).map(|sync| (sync, min_rtt, max_rtt))
                            } else {
                                None
                            }
                        };
                        if let Some((selected, min_rtt, max_rtt)) = selected {
                            let previous = *read_clock_sync(&clock_sync_state);
                            *write_clock_sync(&clock_sync_state) = selected;
                            let offset_delta_us = ((i128::from(selected.offset_nanos)
                                - i128::from(previous.offset_nanos))
                                / 1_000)
                                .clamp(i128::from(i64::MIN), i128::from(i64::MAX));
                            let offset_delta_us =
                                i64::try_from(offset_delta_us).expect("clamped to i64 range");
                            info!(
                                event = "clock_sync",
                                phase = "resync",
                                samples = CLOCK_SYNC_PROBE_SAMPLES,
                                selected_rtt_us = selected.rtt_nanos / 1_000,
                                min_rtt_us = min_rtt / 1_000,
                                max_rtt_us = max_rtt / 1_000,
                                clock_offset_us = selected.offset_nanos / 1_000,
                                offset_delta_us,
                                "refreshed clock-sync sample"
                            );
                        }
                    }
                    Ok(ControlMessage::Goodbye {
                        reason,
                        code,
                        final_stats,
                    }) => {
                        info!(event = "peer_goodbye", %reason, ?code, "host said goodbye");
                        log_peer_session_summary("host", final_stats.as_deref());
                        say_goodbye_once(
                            &conn,
                            "client acknowledged host goodbye",
                            GoodbyeCode::Clean,
                            &session_summary,
                            &shutdown_notice_sent,
                            Some(&decode_event_rx),
                        )
                        .await;
                        return;
                    }
                    Ok(ControlMessage::Extension(msg)) => {
                        warn!(
                            key = %msg.key,
                            version = msg.version,
                            request_id = %msg.request_id,
                            payload_len = msg.payload.len(),
                            "unnegotiated control extension; closing session"
                        );
                        let reason = format!("unnegotiated extension {}", msg.key);
                        say_goodbye_once(
                            &conn,
                            reason.as_str(),
                            GoodbyeCode::ProtocolError,
                            &session_summary,
                            &shutdown_notice_sent,
                            Some(&decode_event_rx),
                        )
                        .await;
                        return;
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
                                id = %d.id,
                                name = %d.name,
                                width = d.current_mode.width,
                                height = d.current_mode.height,
                                refresh_millihz = d.current_mode.refresh_millihz,
                                scale = format!("{}/{}", d.scale_num, d.scale_den),
                                primary = d.primary,
                                position = ?d.position,
                                can_set_mode = d.can_set_mode,
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
                        | ControlMessage::SetViewportHint { .. }
                        | ControlMessage::SetDisplayMode { .. }
                        | ControlMessage::ClientDisplayMetrics { .. },
                    ) => {
                        // Client-originated; misrouted if seen on the client side.
                        tracing::debug!(
                            "unexpected client→host control message arrived on client; ignoring"
                        );
                    }
                    Ok(ControlMessage::DisplayModeResult {
                        request_id,
                        display_id,
                        status,
                        actual_mode,
                    }) => {
                        tracing::debug!(
                            %request_id,
                            %display_id,
                            ?status,
                            ?actual_mode,
                            "display mode result"
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
    let recv_clock_sync = clock_sync_state.clone();
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
        decode_event_tx,
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
    let (audio_tx, audio_active) =
        setup_audio_playback(audio_enabled, &server_hello, session_summary.clone());
    session_summary.set_audio_active(audio_active);

    let conn_for_recovery_send = conn.clone();
    let session_summary_for_recv = session_summary.clone();
    let shutdown_notice_for_recv = shutdown_notice_sent.clone();
    let decode_event_rx_for_recv = decode_event_rx.clone();
    tokio::spawn(async move {
        let mut reassembler = FrameReassembler::new();
        if decoder_ready_rx.await.is_err() {
            // Decoder construction failed and the thread dropped the
            // sender without signalling ready. The host has no other
            // way to learn we won't ever render its frames, so send a
            // Goodbye(InternalError) before exiting — otherwise the
            // host keeps encoding into a black hole until idle
            // timeout, and the user sees a frozen window with no
            // explanation. `say_goodbye_once` closes the
            // connection as part of its shutdown.
            error!(
                "decode thread failed to initialise; sending Goodbye(InternalError) and exiting"
            );
            say_goodbye_once(
                &conn_ready,
                "client decoder failed to initialise",
                GoodbyeCode::InternalError,
                &session_summary_for_recv,
                &shutdown_notice_for_recv,
                Some(&decode_event_rx_for_recv),
            )
            .await;
            return;
        }
        // Decoder is up, but video startup still waits for the renderer's real
        // physical viewport. The viewport task sends SetViewportHint and then
        // StreamReady on the same control stream so the host opens the gate only
        // after it has the dimensions it will encode for.
        let _ = decoder_ready_for_startup_tx.send(true);
        info!("client decoder ready; waiting for first viewport before StreamReady");
        let mut frame_count: u64 = 0;
        // Reassembler cumulative counters at the start of the current stats
        // window. Diff against the live counters to compute per-window
        // incomplete-frame and fragment-loss-event counts for ClientStats.
        let mut last_incomplete_frames: u64 = 0;
        let mut last_fragment_loss_events: u64 = 0;
        let mut last_fec_recovered_frames: u64 = 0;
        let mut last_fec_recovered_fragments: u64 = 0;
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
        // Sum of capture-to-recv ages over the window. The previous
        // implementation logged the last frame's age which is
        // misleading when the metric is supposed to summarise a
        // second of behaviour; averaging across frames gives an
        // actually-meaningful number.
        let mut latency_sum_ns: u64 = 0;
        let mut latency_min_ns: u64 = 0;
        let mut latency_max_ns: u64 = 0;
        // Sum of t_send (host clock, translated via clock_sync) to
        // local recv times. Isolates the network leg from compute
        // — pair with avg_encode_ms / avg_decode_ms to attribute
        // any latency budget movement to the right component.
        let mut network_latency_sum_ns: u64 = 0;
        let mut network_latency_min_ns: u64 = 0;
        let mut network_latency_max_ns: u64 = 0;
        // Bytes off the wire (encoded H.264 payloads after
        // reassembly). With matching host kbps_out, a divergence
        // means packets are being dropped between host and client.
        let mut bytes_received: u64 = 0;
        // Decode-side events folded into the current stats window. Decode
        // completions drive avg/min/max decode latency; epoch drops are tracked
        // separately so they explain missing completions without skewing timing.
        let mut decode_window = DecodeEventWindow::default();
        let mut last_log = Instant::now();
        let mut decode_event_tick = tokio::time::interval(std::time::Duration::from_millis(100));
        decode_event_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Cursor datagram observability — separate cadence so a
        // chatty cursor channel doesn't bury the video stats line.
        let mut cursor_pos_packets: u64 = 0;
        let mut last_cursor_log = std::time::Instant::now();

        loop {
            // All video — IDR keyframes and P-frames alike — arrives on the
            // unreliable datagram channel and feeds the same reassembler; there
            // is no separate reliable keyframe stream to race. Cursor and audio
            // datagrams share the channel and are dispatched out here.
            let datagram = tokio::select! {
                result = conn_recv.recv_datagram() => result,
                _ = decode_event_tick.tick() => {
                    drain_decode_events(
                        &decode_event_rx_for_recv,
                        &session_summary_for_recv,
                        Some(&mut decode_window),
                    );
                    continue;
                }
            };
            let packet: VideoPacket = match datagram {
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
                    stream_epoch,
                    frame_seq,
                    payload,
                    redundant,
                    ..
                })) => {
                    // Forward to the audio decode thread; drop on a full channel
                    // (decoder behind) — audio is loss-tolerant, but the drop
                    // is surfaced separately from packets_received.
                    if let Some(tx) = &audio_tx {
                        if tx
                            .try_send(AudioFrameMsg {
                                stream_epoch,
                                seq: frame_seq,
                                payload,
                                redundant,
                            })
                            .is_err()
                        {
                            session_summary_for_recv
                                .audio
                                .decode_queue_drop_packets
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    session_summary_for_recv
                        .audio
                        .packets_received
                        .fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                Err(e) => {
                    drain_decode_events(
                        &decode_event_rx_for_recv,
                        &session_summary_for_recv,
                        Some(&mut decode_window),
                    );
                    if shutdown_notice_for_recv.load(Ordering::Acquire)
                        || e.is_clean_shutdown_recv()
                    {
                        info!(error = ?e, "datagram recv ended during clean shutdown");
                    } else {
                        // This is terminal for the video stream and the user
                        // otherwise sees a frozen last-frame with no indication
                        // anything broke. Also close explicitly so the host
                        // learns about it instead of waiting for idle timeout.
                        error!(error = ?e, "datagram recv failed; closing connection and ending recv loop");
                        conn_recv.close(1, b"recv failed");
                    }
                    break;
                }
            };

            // Snapshot loss counters around the handle() so we can see if
            // this packet's processing caused the reassembler to give up on
            // a frame. We gate recovery on incomplete frames only — the count
            // of frames started but pruned incomplete. That is the genuine "a
            // frame will never complete" signal worth a recovery IDR.
            //
            // We deliberately do NOT trigger on fragment-loss events: that
            // counter bumps when a *straggler* fragment arrives for a frame
            // that was already finalized or pruned, or when a malformed packet
            // is rejected. Neither is independently actionable — by the time a
            // late fragment shows up the frame is already gone (recovered or
            // abandoned), so firing on it just emits spurious recovery IDRs
            // that add bandwidth and worsen congestion exactly when the path
            // is already lossy. A frame that truly won't complete still bumps
            // incomplete frames when `prune_old` evicts it, so the real signal
            // is not lost — only the noise.
            let pre_loss = reassembler.loss_counters();
            let pre_recovery = reassembler.recovery_counters();
            let result = reassembler.handle(packet);
            let post_loss = reassembler.loss_counters();
            let post_recovery = reassembler.recovery_counters();
            session_summary_for_recv
                .video
                .incomplete_frames
                .fetch_add(post_loss.0.saturating_sub(pre_loss.0), Ordering::Relaxed);
            session_summary_for_recv
                .video
                .fragment_loss_events
                .fetch_add(post_loss.1.saturating_sub(pre_loss.1), Ordering::Relaxed);
            session_summary_for_recv
                .video
                .fec_recovered_frames
                .fetch_add(
                    post_recovery.0.saturating_sub(pre_recovery.0),
                    Ordering::Relaxed,
                );
            session_summary_for_recv
                .video
                .fec_recovered_fragments
                .fetch_add(
                    post_recovery.1.saturating_sub(pre_recovery.1),
                    Ordering::Relaxed,
                );
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
            let current_clock_sync = *read_clock_sync(&recv_clock_sync);
            let host_in_client_clock =
                current_clock_sync.remote_to_local(frame.meta.timing.t_capture_userspace);
            let send_in_client_clock = current_clock_sync.remote_to_local(frame.meta.timing.t_send);
            let age_ns = now.saturating_sub(host_in_client_clock);
            let network_ns = now.saturating_sub(send_in_client_clock);
            frame_count += 1;
            latency_sum_ns = latency_sum_ns.saturating_add(age_ns);
            network_latency_sum_ns = network_latency_sum_ns.saturating_add(network_ns);
            if frame_count == 1 {
                latency_min_ns = age_ns;
                network_latency_min_ns = network_ns;
            } else {
                latency_min_ns = latency_min_ns.min(age_ns);
                network_latency_min_ns = network_latency_min_ns.min(network_ns);
            }
            latency_max_ns = latency_max_ns.max(age_ns);
            network_latency_max_ns = network_latency_max_ns.max(network_ns);
            let frame_body_len = frame.body.len() as u64;
            bytes_received = bytes_received.saturating_add(frame_body_len);
            session_summary_for_recv
                .video
                .frames_received
                .fetch_add(1, Ordering::Relaxed);
            session_summary_for_recv
                .video
                .bytes_received
                .fetch_add(frame_body_len, Ordering::Relaxed);
            if frame.meta.keyframe {
                session_summary_for_recv
                    .video
                    .keyframes
                    .fetch_add(1, Ordering::Relaxed);
            }

            // Hand the reassembled frame to the decode thread.
            // Bounded channel + try_send means a stalled decoder
            // doesn't block the recv loop — we drop the frame and
            // count the loss so the stats line surfaces it.
            // `frame.stream_epoch` is the reassembler's authority on which
            // epoch this frame belongs to (it keys fragments by epoch). The
            // decode worker rebuilds the decoder when the epoch advances —
            // the host bumps it on a resolution/codec change, and an in-place
            // reconfigure corrupts the AV1 decoder on AMD.
            let job = DecodeJob {
                body: frame.body,
                host_in_client_clock,
                keyframe: frame.meta.keyframe,
                stream_epoch: frame.stream_epoch,
            };
            if decode_job_tx.try_send(job).is_err() {
                decode_window.queue_drops = decode_window.queue_drops.saturating_add(1);
                session_summary_for_recv
                    .video
                    .decode_queue_drop_frames
                    .fetch_add(1, Ordering::Relaxed);
            }

            // Drain decode completions that arrived since the last
            // iteration. Non-blocking — if the decoder hasn't
            // produced anything yet, we'll pick it up next time
            // round. Folds per-frame metrics into the stats window
            // the recv loop owns.
            drain_decode_events(
                &decode_event_rx_for_recv,
                &session_summary_for_recv,
                Some(&mut decode_window),
            );

            if last_log.elapsed() >= std::time::Duration::from_secs(1) {
                let window_secs = last_log.elapsed().as_secs_f64();
                // ClientStats — host uses this to drive future
                // adaptive bitrate / FEC strength / codec
                // downshift decisions. Counters are per-window; rtt_us is the
                // current QUIC RTT estimate.
                let (incomplete_frames_now, fragment_loss_events_now) = reassembler.loss_counters();
                let incomplete_frames =
                    u32::try_from(incomplete_frames_now.saturating_sub(last_incomplete_frames))
                        .unwrap_or(u32::MAX);
                let fragment_loss_events = u32::try_from(
                    fragment_loss_events_now.saturating_sub(last_fragment_loss_events),
                )
                .unwrap_or(u32::MAX);
                last_incomplete_frames = incomplete_frames_now;
                last_fragment_loss_events = fragment_loss_events_now;
                let (fec_recovered_frames_now, fec_recovered_fragments_now) =
                    reassembler.recovery_counters();
                let fec_recovered_frames = u32::try_from(
                    fec_recovered_frames_now.saturating_sub(last_fec_recovered_frames),
                )
                .unwrap_or(u32::MAX);
                let fec_recovered_fragments = u32::try_from(
                    fec_recovered_fragments_now.saturating_sub(last_fec_recovered_fragments),
                )
                .unwrap_or(u32::MAX);
                last_fec_recovered_frames = fec_recovered_frames_now;
                last_fec_recovered_fragments = fec_recovered_fragments_now;
                // window_secs is a small positive duration; round-to-i64 is intentional.
                #[allow(clippy::cast_possible_truncation)]
                let window_ms =
                    u32::try_from((window_secs * 1000.0).round() as i64).unwrap_or(u32::MAX);
                let rtt_us = u32::try_from(conn_recv.rtt().as_micros().min(u128::from(u32::MAX)))
                    .unwrap_or(u32::MAX);
                let stats = ControlMessage::ClientStats {
                    window_ms,
                    frames_received: u32::try_from(frame_count).unwrap_or(u32::MAX),
                    incomplete_frames,
                    fragment_loss_events,
                    rtt_us,
                    fec_recovered_frames,
                    fec_recovered_fragments,
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
                let avg_decode_ms = if decode_window.completion_count > 0 {
                    (decode_window.latency_sum_ns as f64 / decode_window.completion_count as f64)
                        / 1_000_000.0
                } else {
                    0.0
                };
                let min_decode_ms =
                    ns_to_ms(decode_window.latency_min_ns, decode_window.completion_count);
                let max_decode_ms =
                    ns_to_ms(decode_window.latency_max_ns, decode_window.completion_count);
                #[allow(clippy::cast_precision_loss)]
                let avg_latency_ms = if frame_count > 0 {
                    (latency_sum_ns as f64 / frame_count as f64) / 1_000_000.0
                } else {
                    0.0
                };
                let min_latency_ms = ns_to_ms(latency_min_ns, frame_count);
                let max_latency_ms = ns_to_ms(latency_max_ns, frame_count);
                #[allow(clippy::cast_precision_loss)]
                let avg_network_ms = if frame_count > 0 {
                    (network_latency_sum_ns as f64 / frame_count as f64) / 1_000_000.0
                } else {
                    0.0
                };
                let min_network_ms = ns_to_ms(network_latency_min_ns, frame_count);
                let max_network_ms = ns_to_ms(network_latency_max_ns, frame_count);
                #[allow(clippy::cast_precision_loss)]
                let kbps_in = if window_secs > 0.0 {
                    (bytes_received as f64 * 8.0 / 1000.0) / window_secs
                } else {
                    0.0
                };
                #[allow(clippy::cast_precision_loss)]
                let fps = if window_secs > 0.0 {
                    frame_count as f64 / window_secs
                } else {
                    0.0
                };
                info!(
                    frames = frame_count,
                    fps,
                    avg_latency_ms,
                    min_latency_ms,
                    max_latency_ms,
                    avg_network_ms,
                    min_network_ms,
                    max_network_ms,
                    avg_decode_ms,
                    min_decode_ms,
                    max_decode_ms,
                    kbps_in,
                    decode_errors = decode_window.decode_errors,
                    render_drop_frames = decode_window.render_drops,
                    idr_requests = decode_window.idr_requests,
                    decode_queue_drop_frames = decode_window.queue_drops,
                    decode_stale_epoch_drop_frames = decode_window.stale_epoch_drops,
                    decode_epoch_throttle_drop_frames = decode_window.epoch_throttle_drops,
                    fec_recovered_frames,
                    fec_recovered_fragments,
                    "frame stats"
                );
                frame_count = 0;
                latency_sum_ns = 0;
                latency_min_ns = 0;
                latency_max_ns = 0;
                network_latency_sum_ns = 0;
                network_latency_min_ns = 0;
                network_latency_max_ns = 0;
                bytes_received = 0;
                decode_window = DecodeEventWindow::default();
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

    {
        let conn = conn.clone();
        tokio::spawn(async move {
            let mut last_sent = None;
            while let Some(metrics) = display_metrics_rx.recv().await {
                if !should_send_client_display_metrics(last_sent.as_ref(), &metrics) {
                    continue;
                }
                if let Err(e) = conn
                    .send_control(&ControlMessage::ClientDisplayMetrics(metrics.clone()))
                    .await
                {
                    warn!(
                        error = ?e,
                        "ClientDisplayMetrics send failed; display metrics task exiting"
                    );
                    return;
                }
                info!(
                    client_display_id = metrics.display_id,
                    width = metrics.mode.width,
                    height = metrics.mode.height,
                    refresh_millihz = metrics.mode.refresh_millihz,
                    scale = format!("{}/{}", metrics.scale_num, metrics.scale_den),
                    safe_area = ?metrics.safe_area,
                    "sent client display metrics to host"
                );
                last_sent = Some(metrics);
            }
        });
    }

    // Viewport debouncer task. Drag-resizing fires
    // `WindowEvent::Resized` continuously (often >100 events/second);
    // sending a `SetViewportHint` per event would have the host
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
        let mut startup_sent = false;
        let mut last_sent_viewport: Option<Viewport> = None;
        let mut decoder_ready_for_startup_rx = decoder_ready_for_startup_rx;
        loop {
            // Either receive a new size, or fire the pending one after
            // the debounce window elapses with no new event.
            let next = match pending {
                Some(_) => tokio::time::timeout(debounce, viewport_rx.recv()).await,
                None => Ok(viewport_rx.recv().await),
            };
            match next {
                Ok(Some(size)) => {
                    let viewport = Viewport::new(size.0, size.1);
                    if !viewport.is_valid() {
                        continue;
                    }
                    if !startup_sent {
                        while !*decoder_ready_for_startup_rx.borrow_and_update() {
                            if decoder_ready_for_startup_rx.changed().await.is_err() {
                                warn!(
                                    "decoder readiness channel closed before initial viewport; \
                                     viewport task exiting"
                                );
                                return;
                            }
                        }
                        let viewport = drain_latest_valid_viewport(viewport, &mut viewport_rx);
                        if should_send_viewport(last_sent_viewport, viewport) {
                            if let Err(e) = conn_viewport
                                .send_control(&ControlMessage::SetViewportHint {
                                    stream_id: VideoStreamId(0),
                                    viewport,
                                })
                                .await
                            {
                                warn!(
                                    error = ?e,
                                    "initial SetViewportHint send failed; viewport task exiting"
                                );
                                return;
                            }
                            last_sent_viewport = Some(viewport);
                            info!(
                                width = viewport.width,
                                height = viewport.height,
                                "sent initial SetViewportHint to host"
                            );
                        }
                        if let Err(e) = conn_viewport
                            .send_control(&ControlMessage::StreamReady {
                                video: true,
                                audio: audio_active,
                            })
                            .await
                        {
                            warn!(error = ?e, "StreamReady send failed; host will not emit video");
                            return;
                        }
                        info!(
                            event = "stream_ready",
                            video = true,
                            audio = audio_active,
                            "client signalled StreamReady after initial viewport"
                        );
                        startup_sent = true;
                        pending = None;
                        continue;
                    }
                    pending = Some(size);
                }
                Ok(None) => {
                    // Sender dropped. Fire any pending before exiting
                    // so the last resize event still makes it.
                    if startup_sent {
                        if let Some((w, h)) = pending {
                            let viewport = Viewport::new(w, h);
                            if should_send_viewport(last_sent_viewport, viewport) {
                                let _ = conn_viewport
                                    .send_control(&ControlMessage::SetViewportHint {
                                        stream_id: VideoStreamId(0),
                                        viewport,
                                    })
                                    .await;
                            }
                        }
                    }
                    return;
                }
                Err(_) => {
                    if let Some((w, h)) = pending.take() {
                        let viewport = Viewport::new(w, h);
                        if !should_send_viewport(last_sent_viewport, viewport) {
                            continue;
                        }
                        if let Err(e) = conn_viewport
                            .send_control(&ControlMessage::SetViewportHint {
                                stream_id: VideoStreamId(0),
                                viewport,
                            })
                            .await
                        {
                            warn!(error = ?e, "SetViewportHint send failed; viewport task exiting");
                            return;
                        }
                        last_sent_viewport = Some(viewport);
                        info!(width = w, height = h, "sent SetViewportHint to host");
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
        if let RenderEvent::ClientDisplayMetrics(metrics) = evt {
            let _ = display_metrics_tx.send(metrics);
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
        let session_summary = session_summary.clone();
        let shutdown_notice_sent = shutdown_notice_sent.clone();
        let decode_event_rx = decode_event_rx.clone();
        tokio::spawn(async move {
            if let Err(e) = tokio::signal::ctrl_c().await {
                warn!(error = %e, "ctrl-c handler failed; exiting anyway");
                std::process::exit(1);
            }
            info!("ctrl-c received, sending Goodbye and exiting");
            reporter.emit(&EngineEvent::Disconnected {
                reason: "interrupted".to_string(),
            });
            say_goodbye(
                &conn,
                "client interrupted",
                &session_summary,
                &shutdown_notice_sent,
                Some(&decode_event_rx),
            )
            .await;
            std::process::exit(0);
        });
    }

    // When shell-driven, a `Stop` on stdin (or stdin EOF — the shell
    // died) closes the session the same way Ctrl-C does: the render loop
    // owns the main thread, so like the Ctrl-C handler this task sends
    // Goodbye and exits the process directly rather than trying to unwind
    // through winit.
    if let Some(rx) = stdin_stop {
        let conn = conn.clone();
        let session_summary = session_summary.clone();
        let shutdown_notice_sent = shutdown_notice_sent.clone();
        let decode_event_rx = decode_event_rx.clone();
        tokio::spawn(async move {
            wait_for_stdin_stop_signal(rx).await;
            info!("shell stop received; sending Goodbye and exiting");
            reporter.emit(&EngineEvent::Disconnected {
                reason: "stopped by shell".to_string(),
            });
            say_goodbye(
                &conn,
                "client stopped",
                &session_summary,
                &shutdown_notice_sent,
                Some(&decode_event_rx),
            )
            .await;
            std::process::exit(0);
        });
    }

    let initial_video_size_px = initial_video_size_px_from_displays(&server_hello.displays);
    info!(
        host_mode_px_width = initial_video_size_px.0,
        host_mode_px_height = initial_video_size_px.1,
        "selected host display mode as initial render target"
    );

    // Render loop blocks until the user closes the window. The
    // host's advertised color spec drives the renderer's EOTF
    // dispatch — for desktop captures (`sdr_desktop`) this is the
    // sRGB path, eliminating the BT.709-vs-sRGB transfer-curve
    // mismatch the spec-blind chain previously had to absorb.
    let render_result = tether_render::run(
        "tether-client",
        initial_video_size_px,
        server_hello.video.color_space,
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
    say_goodbye(
        &conn,
        "client closing",
        &session_summary,
        &shutdown_notice_sent,
        Some(&decode_event_rx),
    )
    .await;

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

fn spawn_stdin_stop_signal(enabled: bool) -> Option<tokio::sync::watch::Receiver<bool>> {
    if !enabled {
        return None;
    }
    let (tx, rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        wait_for_stdin_stop().await;
        let _ = tx.send(true);
    });
    Some(rx)
}

async fn wait_for_stdin_stop_signal(mut rx: tokio::sync::watch::Receiver<bool>) {
    if *rx.borrow() {
        return;
    }
    while rx.changed().await.is_ok() {
        if *rx.borrow() {
            return;
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
async fn say_goodbye(
    conn: &tether_transport::Connection,
    reason: &str,
    summary: &SessionSummaryState,
    sent: &AtomicBool,
    decode_events: Option<&crossbeam_channel::Receiver<DecodeEvent>>,
) {
    say_goodbye_once(
        conn,
        reason,
        GoodbyeCode::Clean,
        summary,
        sent,
        decode_events,
    )
    .await;
}

/// Send local final stats once and close the connection. The `sent` guard
/// prevents a reciprocal Goodbye from echoing forever when both peers exchange
/// shutdown summaries.
async fn say_goodbye_once(
    conn: &tether_transport::Connection,
    reason: &str,
    code: GoodbyeCode,
    summary: &SessionSummaryState,
    sent: &AtomicBool,
    decode_events: Option<&crossbeam_channel::Receiver<DecodeEvent>>,
) {
    if sent.swap(true, Ordering::AcqRel) {
        return;
    }
    say_goodbye_with_code(conn, reason, code, summary, decode_events).await;
}

/// Variant that lets the caller signal *why* the session ended.
/// Use [`GoodbyeCode::InternalError`] for fatal local failures (decoder
/// init failed, render thread died) so the host's session-end log
/// distinguishes a genuine crash from a user closing the window.
async fn say_goodbye_with_code(
    conn: &tether_transport::Connection,
    reason: &str,
    code: GoodbyeCode,
    summary: &SessionSummaryState,
    decode_events: Option<&crossbeam_channel::Receiver<DecodeEvent>>,
) {
    use std::time::Duration;
    if let Some(rx) = decode_events {
        drain_decode_events(rx, summary, None);
    }
    let msg = ControlMessage::Goodbye {
        reason: reason.to_string(),
        code,
        final_stats: Some(Box::new(summary.snapshot())),
    };
    if let Err(e) = conn.send_control(&msg).await {
        warn!(error = ?e, "send Goodbye failed; host will fall back to timeout");
    } else {
        info!(
            event = "session_teardown",
            reason,
            ?code,
            "client sent Goodbye"
        );
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

/// One audio datagram handed from the recv loop to the playback thread. The
/// sequence number drives gap-detected concealment; `redundant` is the RED tail
/// (previous payloads, newest-first) used to recover a lost frame without a
/// concealment click.
struct AudioFrameMsg {
    stream_epoch: u32,
    seq: u32,
    payload: Bytes,
    redundant: Vec<Bytes>,
}

/// If audio is enabled and the host advertised an `AudioConfig`, spawn the
/// playback thread and return a sender the recv loop feeds Opus frames to,
/// plus `true` (audio active). Otherwise return `(None, false)` and run
/// video-only. Never fatal — a missing output device just means no sound.
fn setup_audio_playback(
    enabled: bool,
    server_hello: &ServerHello,
    summary: Arc<SessionSummaryState>,
) -> (Option<crossbeam_channel::Sender<AudioFrameMsg>>, bool) {
    const AUDIO_STARTUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

    if !enabled {
        info!("audio disabled via --no-audio");
        return (None, false);
    }
    let Some(audio_cfg) = server_hello.audio.clone() else {
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
    let (startup_tx, startup_rx) = std::sync::mpsc::sync_channel::<Result<(), String>>(1);
    match std::thread::Builder::new()
        .name("tether-client-audio".into())
        .spawn(move || run_audio_playback(opus_cfg, rx, summary, startup_tx))
    {
        Ok(_) => {
            match startup_rx.recv_timeout(AUDIO_STARTUP_TIMEOUT) {
                Ok(Ok(())) => {
                    info!(
                        sample_rate = audio_cfg.sample_rate_hz,
                        channels = audio_cfg.channels,
                        "audio playback enabled"
                    );
                    (Some(tx), true)
                }
                Ok(Err(reason)) => {
                    warn!(%reason, "audio playback unavailable; running video-only");
                    (None, false)
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    warn!(
                        timeout_ms = AUDIO_STARTUP_TIMEOUT.as_millis(),
                        "audio playback startup timed out; running video-only"
                    );
                    (None, false)
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    warn!("audio playback thread exited before reporting readiness; running video-only");
                    (None, false)
                }
            }
        }
        Err(e) => {
            warn!(error = %e, "failed to spawn audio thread; running video-only");
            (None, false)
        }
    }
}

/// Audio playback thread: decode incoming Opus frames, recover/conceal sequence
/// gaps (via [`tether_audio::LossRecovery`]), and push PCM into the playback
/// ring feeding the cpal output stream. Owns the `AudioPlayer` (and thus the
/// cpal stream) for its lifetime; returns when the channel closes (session
/// ending), dropping the player to stop playback.
fn run_audio_playback(
    cfg: tether_audio::OpusConfig,
    rx: crossbeam_channel::Receiver<AudioFrameMsg>,
    summary: Arc<SessionSummaryState>,
    startup_tx: std::sync::mpsc::SyncSender<Result<(), String>>,
) {
    let (player, sink) = match tether_audio::AudioPlayer::with_defaults(cfg) {
        Ok(pair) => pair,
        Err(e) => {
            let _ = startup_tx.send(Err(format!("audio output device unavailable: {e}")));
            return;
        }
    };
    let decoder = match tether_audio::OpusDecoder::new(cfg) {
        Ok(d) => d,
        Err(e) => {
            let _ = startup_tx.send(Err(format!("opus decoder init failed: {e}")));
            return;
        }
    };
    if startup_tx.send(Ok(())).is_err() {
        return;
    }
    // Owns the decoder + sequence state; turns each datagram into PCM frames,
    // healing losses from the RED tail before falling back to PLC. The loss
    // counters it accumulates (recovered/concealed/dropout/stale) are surfaced
    // in the stats log below — on a clean LAN all stay ~0; recovered_frames
    // climbing while concealed/dropout stay low means RED is doing its job.
    let mut recovery = tether_audio::LossRecovery::new(decoder);

    // The host runs one encoder for the whole session, so stream_epoch is a
    // constant 0. We don't handle an audio encoder restart (decoder reset +
    // sequence rebase) yet; guard against silently decoding cross-epoch state
    // with stale decoder/RED context if that path is ever half-wired on the
    // host — fail loud (warn once) and drop the foreign-epoch packets.
    let mut audio_epoch: Option<u32> = None;
    let mut epoch_mismatch_logged = false;

    // Periodic playback-health snapshot. The ring's drift/underrun behaviour is
    // otherwise invisible: cap-and-drop silently absorbs overflow and silence
    // fills underruns, so without this a multi-minute session shows nothing.
    // Frame-driven (audio arrives ~every 5 ms), like the host's send-stats —
    // when audio stops, so does the log. 2 s matches the video stats cadence.
    // Counters are logged as per-interval deltas (matching the video stats
    // line), so a transient loss spike stands out instead of being buried in a
    // climbing session total — hence the `prev_*` snapshots below.
    const STATS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);
    let mut last_stats_log = std::time::Instant::now();
    let mut prev_underruns: u64 = 0;
    let mut prev_dropped_samples: u64 = 0;
    let mut prev_recovery = tether_audio::RecoveryStats::default();
    let mut summary_prev_underruns: u64 = 0;
    let mut summary_prev_dropped_samples: u64 = 0;
    let mut summary_prev_recovery = tether_audio::RecoveryStats::default();
    // Peak |sample| of decoded audio since the last log. ~0 means the frames
    // arriving are silent (the silence is upstream — capture/encode), which
    // distinguishes that from a local output-routing problem where non-zero
    // audio is played but not heard.
    let mut peak: f32 = 0.0;

    while let Ok(AudioFrameMsg {
        stream_epoch,
        seq,
        payload,
        redundant,
    }) = rx.recv()
    {
        match audio_epoch {
            None => audio_epoch = Some(stream_epoch),
            Some(expected) if stream_epoch != expected => {
                if !epoch_mismatch_logged {
                    epoch_mismatch_logged = true;
                    warn!(
                        expected,
                        got = stream_epoch,
                        "audio stream_epoch changed mid-session; decoder/RED reset is \
                         not wired — dropping foreign-epoch packets"
                    );
                }
                continue;
            }
            Some(_) => {}
        }

        recovery.accept(seq, &payload, &redundant, |pcm| {
            for &s in &pcm.samples {
                peak = peak.max(s.abs());
            }
            sink.submit(pcm);
        });
        let (summary_underruns, summary_dropped_samples, _) = sink.stats();
        let summary_recovery = recovery.stats();
        record_audio_playback_summary_delta(
            &summary,
            summary_underruns,
            summary_prev_underruns,
            summary_dropped_samples,
            summary_prev_dropped_samples,
            summary_recovery,
            summary_prev_recovery,
        );
        summary_prev_underruns = summary_underruns;
        summary_prev_dropped_samples = summary_dropped_samples;
        summary_prev_recovery = summary_recovery;

        if last_stats_log.elapsed() >= STATS_INTERVAL {
            let (underruns, dropped_samples, buffered) = sink.stats();
            let s = recovery.stats();
            // `buffered` is interleaved samples → wall-clock so drift toward an
            // underrun (→ 0 ms) or the latency cap is legible at a glance. The
            // counters are deltas over this interval (totals climb forever and
            // bury a transient spike); buffered_ms / peak are instantaneous /
            // per-interval already.
            let frames_buffered = buffered / usize::from(cfg.channels).max(1);
            let buffered_ms = frames_buffered * 1000 / (cfg.sample_rate as usize).max(1);
            info!(
                underruns = underruns - prev_underruns,
                dropped_samples = dropped_samples - prev_dropped_samples,
                buffered_ms,
                recovered_frames = s.recovered_frames - prev_recovery.recovered_frames,
                concealed_frames = s.concealed_frames - prev_recovery.concealed_frames,
                dropout_frames = s.dropout_frames - prev_recovery.dropout_frames,
                dropouts = s.dropouts - prev_recovery.dropouts,
                stale_packets = s.stale_packets - prev_recovery.stale_packets,
                decode_errors = s.decode_errors - prev_recovery.decode_errors,
                peak,
                "audio playback stats"
            );
            prev_underruns = underruns;
            prev_dropped_samples = dropped_samples;
            prev_recovery = s;
            last_stats_log = std::time::Instant::now();
            peak = 0.0;
        }
    }

    let (underruns, dropped_samples, _) = sink.stats();
    let s = recovery.stats();
    record_audio_playback_summary_delta(
        &summary,
        underruns,
        summary_prev_underruns,
        dropped_samples,
        summary_prev_dropped_samples,
        s,
        summary_prev_recovery,
    );

    // Channel closed: drop the player to stop the output stream.
    drop(player);
}

fn record_audio_playback_summary_delta(
    summary: &SessionSummaryState,
    underruns: u64,
    prev_underruns: u64,
    dropped_samples: u64,
    prev_dropped_samples: u64,
    recovery: tether_audio::RecoveryStats,
    prev_recovery: tether_audio::RecoveryStats,
) {
    summary
        .audio
        .underruns
        .fetch_add(underruns.saturating_sub(prev_underruns), Ordering::Relaxed);
    summary.audio.dropped_samples.fetch_add(
        dropped_samples.saturating_sub(prev_dropped_samples),
        Ordering::Relaxed,
    );
    summary.audio.recovered_frames.fetch_add(
        recovery
            .recovered_frames
            .saturating_sub(prev_recovery.recovered_frames),
        Ordering::Relaxed,
    );
    summary.audio.concealed_frames.fetch_add(
        recovery
            .concealed_frames
            .saturating_sub(prev_recovery.concealed_frames),
        Ordering::Relaxed,
    );
    summary.audio.dropout_frames.fetch_add(
        recovery
            .dropout_frames
            .saturating_sub(prev_recovery.dropout_frames),
        Ordering::Relaxed,
    );
    summary.audio.dropouts.fetch_add(
        recovery.dropouts.saturating_sub(prev_recovery.dropouts),
        Ordering::Relaxed,
    );
    summary.audio.stale_packets.fetch_add(
        recovery
            .stale_packets
            .saturating_sub(prev_recovery.stale_packets),
        Ordering::Relaxed,
    );
    summary.audio.decode_errors.fetch_add(
        recovery
            .decode_errors
            .saturating_sub(prev_recovery.decode_errors),
        Ordering::Relaxed,
    );
}

/// Directory the client caches its identity (`client_cert.der`/`client_key.der`)
/// and `known_hosts.json` in. Delegates to the shared
/// [`tether_pairing::config_dir`] so the host, the client, and the Tauri shell
/// all resolve the same release/dev channel location.
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
    use tether_protocol::control::{DisplayId, DisplayMode};

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    fn display(id: u32, width: u32, height: u32, primary: bool) -> DisplayDescriptor {
        let mode = DisplayMode::new(width, height, 60_000);
        DisplayDescriptor {
            id: DisplayId(id),
            name: format!("display-{id}"),
            scale_num: 1,
            scale_den: 1,
            primary,
            position: (0, 0),
            current_mode: mode,
            available_modes: vec![mode],
            can_set_mode: false,
        }
    }

    #[test]
    fn initial_video_size_prefers_valid_primary_display() {
        let displays = vec![display(0, 1920, 1080, false), display(1, 3024, 1952, true)];
        assert_eq!(initial_video_size_px_from_displays(&displays), (3024, 1952));
    }

    #[test]
    fn initial_video_size_falls_back_to_first_valid_display() {
        let displays = vec![display(0, 0, 1080, true), display(1, 1920, 1200, false)];
        assert_eq!(initial_video_size_px_from_displays(&displays), (1920, 1200));
    }

    #[test]
    fn initial_video_size_uses_fallback_when_displays_are_invalid() {
        assert_eq!(initial_video_size_px_from_displays(&[]), (1280, 720));
        assert_eq!(
            initial_video_size_px_from_displays(&[display(0, 0, 1080, true)]),
            (1280, 720)
        );
    }

    #[test]
    fn should_send_viewport_rejects_invalid_and_duplicate_sizes() {
        let viewport = Viewport::new(1920, 1080);

        assert!(should_send_viewport(None, viewport));
        assert!(!should_send_viewport(Some(viewport), viewport));
        assert!(should_send_viewport(
            Some(viewport),
            Viewport::new(1280, 720)
        ));
        assert!(!should_send_viewport(None, Viewport::new(0, 720)));
    }

    #[test]
    fn should_send_client_display_metrics_rejects_duplicates() {
        let metrics = ClientDisplayMetrics {
            display_id: 0,
            mode: DisplayMode::new(2560, 1440, 60_000),
            scale_num: 2,
            scale_den: 1,
            safe_area: None,
        };

        assert!(should_send_client_display_metrics(None, &metrics));
        assert!(!should_send_client_display_metrics(
            Some(&metrics),
            &metrics
        ));

        let mut moved = metrics.clone();
        moved.display_id = 1;
        assert!(should_send_client_display_metrics(Some(&metrics), &moved));

        let mut scaled = metrics.clone();
        scaled.scale_num = 3;
        scaled.scale_den = 2;
        assert!(should_send_client_display_metrics(Some(&metrics), &scaled));
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
    fn no_audio_flag_disables_playback_request() {
        let parsed = parse_cli_args(&args(&["--no-audio", "127.0.0.1:7654"])).expect("valid args");
        assert!(!parsed.audio);

        let parsed = parse_cli_args(&args(&["127.0.0.1:7654"])).expect("valid args");
        assert!(parsed.audio);
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
        // (incomplete_frames, fragment_loss_events)
        // A frame pruned incomplete (incomplete_frames++) → recover.
        assert!(recovery_warranted((0, 0), (1, 0)));
        // A stale straggler / malformed packet (fragment_loss_events++ only)
        // → do NOT recover; it's not independently actionable.
        assert!(!recovery_warranted((0, 0), (0, 1)));
        // Both moving still recovers — the dropped frame is the reason.
        assert!(recovery_warranted((0, 0), (1, 1)));
        // Nothing moved → nothing to recover.
        assert!(!recovery_warranted((3, 7), (3, 7)));
    }

    #[test]
    fn startup_viewport_drain_uses_newest_queued_valid_size() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        tx.send((1600, 900)).unwrap();
        tx.send((0, 0)).unwrap();
        tx.send((1920, 1080)).unwrap();

        let selected = drain_latest_valid_viewport(Viewport::new(1280, 720), &mut rx);

        assert_eq!(selected, Viewport::new(1920, 1080));
        assert!(
            rx.try_recv().is_err(),
            "queued resize events should be drained"
        );
    }

    #[test]
    fn hex_decode_accepts_exact_32_byte_lowercase_fingerprint() {
        let decoded =
            hex_decode("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
                .expect("valid fingerprint");
        assert_eq!(
            decoded,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31,
            ]
        );
    }

    #[test]
    fn hex_decode_rejects_bad_length_and_bad_digits() {
        let short = hex_decode("00").expect_err("short fingerprint must fail");
        assert!(short.to_string().contains("64 hex chars"));

        let bad_digit =
            hex_decode("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1g")
                .expect_err("non-hex digit must fail");
        assert!(bad_digit.to_string().contains("bad hex at byte 31"));
    }

    #[test]
    fn clock_resync_lock_recovers_from_poisoned_state() {
        let state = Arc::new(Mutex::new(ClockResyncState::default()));
        let poisoned = state.clone();
        let _ = std::thread::spawn(move || {
            let mut state = poisoned.lock().unwrap();
            state.pending.push(MonoNanos(7));
            panic!("poison clock resync state");
        })
        .join();

        let mut recovered = lock_clock_resync(&state);
        assert_eq!(recovered.pending, vec![MonoNanos(7)]);
        recovered.samples.push(ClockSync {
            offset_nanos: 1,
            rtt_nanos: 2,
            sampled_at_local: MonoNanos(3),
        });
        assert_eq!(recovered.samples.len(), 1);
    }

    #[test]
    fn clock_sync_lock_recovers_from_poisoned_state() {
        let initial = ClockSync {
            offset_nanos: 0,
            rtt_nanos: 1,
            sampled_at_local: MonoNanos(2),
        };
        let state = Arc::new(RwLock::new(initial));
        let poisoned = state.clone();
        let _ = std::thread::spawn(move || {
            let mut state = poisoned.write().unwrap();
            state.offset_nanos = 42;
            panic!("poison clock sync state");
        })
        .join();

        assert_eq!(read_clock_sync(&state).offset_nanos, 42);
        write_clock_sync(&state).rtt_nanos = 99;
        assert_eq!(read_clock_sync(&state).rtt_nanos, 99);
    }
}
