//! Tether host — captures the local display and streams it to a client.
//!
//! v0: raw BGRA frames fragmented over QUIC datagrams, single client per
//! host, no codec. Real capture (PipeWire/portal on Linux, ScreenCaptureKit
//! on macOS) is the default; pass `--test-pattern` to fall back to the
//! synthetic gradient generator (useful for headless dev or as a fallback
//! when the portal isn't available).
//!
//! Usage: `tether-host [--test-pattern] [bind_addr]`
//! (`bind_addr` defaults to `127.0.0.1:7654`).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use crossbeam_channel::Receiver;
#[cfg(target_os = "linux")]
use tether_capture::GpuCapturedSource;
use tether_capture::{
    CapturedFrame, CursorEvent, CursorSource, DamageHint, DamageSignal, HashDamage,
    PixelFormat, PlaceholderCursorSource,
};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use tether_codec::GpuEncoderFrame;
use tether_codec::{build_encoder, Encoder};
#[cfg(target_os = "linux")]
use tether_codec::{DmaBufFrame, DmaBufLayer, DmaBufObject};
#[cfg(target_os = "linux")]
use tether_gpuconvert::{
    Bgra2P010DmaBuf, Nv12DmaBuf, Nv12DmaBufFrame, P010DmaBufFrame, Yuv444DmaBuf, Yuv444DmaBufFrame,
};
#[cfg(target_os = "linux")]
use tether_scaler::{Pipelines as ScalerPipelines, Scaler, ScalerError};
use tether_protocol::control::{
    ChromaSubsampling, CodecKind, ControlMessage, VideoProfile, Viewport,
};
use tether_protocol::video::{
    FrameFragmenter, HostFrameTimingBuilder, InputEchoBatch, VideoFrameMeta,
};
use tether_protocol::MonoNanos;
use tether_session::{
    AbrConfig, AbrController, AbrSample, AcceptError, HostSession, HostSessionConfig,
};
use tether_transport::{AbrSnapshot, Connection, Datagram, Server};
use tokio::sync::Mutex as TokioMutex;
use tokio::task::JoinSet;
use tracing::{info, warn};

/// Default target frame rate. Sunshine and Apollo run desktop / game
/// streaming at 60 fps by default; tether matches. The host's encoder
/// time_base, the test-pattern source, and (in the future) the
/// PipeWire format negotiation all use this. Per-frame budget at 60 fps
/// is 16.6 ms; current Intel iGPU encode times sit around 7–8 ms, so
/// there's headroom.
const ENCODER_FPS: u32 = 60;

/// VBR target bitrate. The encoder is allowed to overshoot for
/// motion-heavy frames and undershoot on static content. Calibrated
/// for 1080p60 H.264 ≈ 8 Mbps and roughly scales linearly with
/// resolution × fps. HEVC sessions get a 0.7× multiplier inside the
/// derivation step (~30% more efficient at the same visual quality;
/// conservative estimate, refined when we benchmark). For now the
/// constant is the H.264 1080p60 floor; multi-resolution scaling is W8.
const ENCODER_BITRATE_KBPS: u32 = 8_000;

const TEST_PATTERN_WIDTH: u32 = 320;
const TEST_PATTERN_HEIGHT: u32 = 240;
const TEST_PATTERN_FPS: u32 = 60;

/// Latest `ClientStats` observation, drained by the encode-and-send
/// thread on each loop iteration. The control recv task writes here
/// every time `ControlMessage::ClientStats` arrives (~1 Hz). A
/// `std::sync::Mutex` is fine because the lock is held for a single
/// option-swap; the send thread isn't a tokio worker so we don't need
/// `tokio::sync::Mutex`.
type LatestClientStats = Arc<StdMutex<Option<ClientStatsObservation>>>;

/// Latest client viewport, with sequence counter so the send thread
/// can tell whether anything changed without diffing the whole value.
/// Set once from `ClientHelloV1::viewport` at session start, then
/// overwritten by `ControlMessage::SetClientViewport` on each resize.
type LatestViewport = Arc<StdMutex<ViewportState>>;

#[derive(Debug, Clone, Copy, Default)]
struct ViewportState {
    /// `None` means "let the host pick native." A valid viewport
    /// (both dims > 0) is what the encoder targets.
    viewport: Option<Viewport>,
    /// Bumped on every write. The send thread reads its last-seen
    /// sequence and only acts when this advances — saves the slot
    /// comparison work when nothing changed.
    seq: u64,
}

/// One window of client-side telemetry. Mirrors the fields in
/// [`ControlMessage::ClientStats`] that the ABR controller actually
/// consumes; `interval_ms` and `frames_received` are dropped at the
/// boundary because the controller takes wall-clock dt itself and
/// doesn't need the success-count.
#[derive(Debug, Clone, Copy)]
struct ClientStatsObservation {
    frames_dropped: u32,
    fragments_lost: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Both host (encoder) and client (decoder) call av_log::install(),
    // so FFmpeg messages can land on either side's hot thread. The
    // encoder is quieter in steady state but the same non-blocking
    // rationale applies — a synchronous stdout writer would stall
    // whichever thread libavcodec calls into.
    let _tracing_guard = init_tracing();

    let (bind, use_test_pattern) = parse_args()?;

    let cert_dir = persistent_cert_dir()?;
    let server = Server::bind_persistent(bind, &cert_dir).await?;
    let local = server.local_addr()?;
    let fingerprint = server.fingerprint();
    let fp_hex = hex_encode(&fingerprint);

    println!("tether-host listening on {local}");
    println!("cert fingerprint: {fp_hex}");
    println!("cert dir:        {} (rm to rotate)", cert_dir.display());
    println!("client cmd:      tether-client {local} {fp_hex}");

    // Warm the SCK capture-capability cache *before* the first client
    // connects. The probe runs eight short `SCStream::start_capture`
    // cycles (~300–900 ms total wall time) and triggers the macOS
    // ScreenRecording TCC prompt on first run. Running it lazily on
    // first-connection inside `handle_client` straddles the
    // application-layer clock-probe RTT measurement, biasing the
    // negotiated `clock_offset` by ~half the probe duration and
    // producing phantom hundred-millisecond `latency_ms` values for
    // every frame stat that session. Running it here moves the
    // probe out of the handshake's critical path entirely; the TCC
    // prompt also fires on host launch (better UX than on first
    // remote connect).
    //
    // Warm the unified capability probe. `tether_probe::host_supported_profiles`
    // round-trips every `PROFILE_PREFERENCE` entry through the full
    // production chain (capture → bridge → encoder → decoder) and
    // OnceLock-caches the result for process lifetime. This is one
    // call that replaces what used to be three caches:
    //   - tether-codec's supported_profiles
    //   - tether-host's SCK_CAPS_CACHE (macOS)
    //   - tether-host's LINUX_P010_DELIVERABLE_CACHE
    // Running it here keeps the probe off the handshake's critical
    // path — see the historical comment about clock-sync offset bias.
    //
    // Test-pattern mode skips the probe entirely: no real capture
    // backend runs (no PipeWire portal, no SCK), and on macOS
    // probing SCK would trigger an unnecessary screen-recording
    // permission prompt. In test-pattern mode `handle_client` uses
    // a hardcoded conservative profile set (the H.264 4:2:0 8-bit
    // floor) — the encoder still constructs lazily for that single
    // profile, so a broken-encoder host still fails loud.
    if !use_test_pattern {
        tokio::task::spawn_blocking(|| {
            let _ = tether_probe::host_supported_profiles();
        })
        .await
        .ok();
    }

    // Reconnect loop: one host process serves a stream of clients,
    // each in its own `handle_client` lifecycle. Per-session state —
    // encoder, libei injector, recv tasks — is fully owned by
    // `handle_client` and dropped when it returns; nothing leaks
    // between sessions. Errors from a single session are logged and
    // the loop continues, so a malformed client or a transient
    // decoder failure can't take the host process down.
    //
    // Ctrl-C is raced at two levels: `handle_client`'s inner select
    // tears the current session down cleanly (sends Goodbye, closes
    // the QUIC connection, joins the send thread). The outer races
    // below catch the same signal so the reconnect loop also exits —
    // without them, the inner handler suppresses the kernel's default
    // SIGINT-kills-process behavior (once tokio's signal handler is
    // installed it stays installed) and subsequent Ctrl-Cs are
    // silently queued.
    loop {
        let conn = tokio::select! {
            biased;
            ctrl_c = tokio::signal::ctrl_c() => {
                if let Err(e) = ctrl_c {
                    warn!(error = %e, "ctrl-c handler failed; shutting down anyway");
                } else {
                    info!("ctrl-c received at main loop; shutting down");
                }
                break;
            }
            accept_res = server.accept() => match accept_res {
                Some(Ok(c)) => Arc::new(c),
                Some(Err(e)) => {
                    warn!(error = ?e, "server.accept failed; continuing");
                    continue;
                }
                None => {
                    warn!("server closed; ending main loop");
                    break;
                }
            },
        };
        info!(remote = %conn.remote_address(), "client connected");

        let host_encode_profiles: Vec<VideoProfile> = if use_test_pattern {
            vec![VideoProfile::H264_8BIT_420]
        } else {
            tether_probe::host_encode_profiles()
        };
        tracing::debug!(
            host_encode_profiles = ?host_encode_profiles,
            "host video encode capabilities (capture-bridge filtered)"
        );

        let cfg = HostSessionConfig {
            server_name: "tether-host".to_string(),
        };
        // `HostSession::accept` takes the channel through the
        // `ControlChannel` trait object so it's mockable in tests.
        // We keep the original `Arc<Connection>` in scope for the rest
        // of `handle_client`, which uses concrete-type methods
        // (video keyframe streams, datagram send, input recv) that
        // are outside the `ControlChannel` surface.
        let session = match HostSession::accept(
            conn.clone() as Arc<dyn tether_transport::ControlChannel>,
            cfg,
            |client_caps| tether_probe::pick_supported_profile(&host_encode_profiles, client_caps),
        )
        .await
        {
            Ok(s) => s,
            Err(AcceptError::NoProfileIntersection { client }) => {
                // HostSession has already sent Goodbye(InternalError).
                // We log the host list ourselves — the error doesn't
                // carry it because the selector (which closes over it)
                // is the only thing that saw both sides.
                warn!(
                    host_encode_profiles = ?host_encode_profiles,
                    client_decode_profiles = ?client,
                    "no mutual video profile; session ended"
                );
                continue;
            }
            Err(AcceptError::Transport(e)) => {
                warn!(error = ?e, "handshake transport error; session ended");
                continue;
            }
        };

        tokio::select! {
            biased;
            ctrl_c = tokio::signal::ctrl_c() => {
                if let Err(e) = ctrl_c {
                    warn!(error = %e, "ctrl-c handler failed; shutting down anyway");
                } else {
                    info!("ctrl-c received during session; shutting down");
                }
                // `handle_client`'s future drops here, which drops the
                // per-session graph (encoder thread, recv tasks, libei
                // injector). The connection close on the dropped
                // `Arc<Connection>` notifies the client.
                break;
            }
            res = handle_client(session, conn, use_test_pattern) => {
                if let Err(e) = res {
                    warn!(error = ?e, "session ended with error; accepting next client");
                }
            }
        }
    }

    server.close_and_wait(0, b"host shutdown").await;
    Ok(())
}

/// Owns every piece of per-connection state — encoder thread, injector,
/// recv tasks. When this function returns the whole graph drops together:
/// the libei session releases, the QUIC connection closes, the encoder
/// frees its hardware context. Anything that needs to survive across
/// reconnects must live in `main`, not here.
async fn handle_client(
    session: HostSession,
    conn: Arc<Connection>,
    use_test_pattern: bool,
) -> anyhow::Result<()> {
    // The handshake (decode-profile negotiation, pixel-format advert,
    // Goodbye-on-no-match) ran in `HostSession::accept`. Unpack the
    // per-connection state it produced.
    //
    // `session.channel` is dropped here: it's an `Arc<dyn ControlChannel>`
    // pointing to the same `Connection` that `conn` holds, just
    // type-erased for the session-level abstraction. The recv tasks
    // and send thread below need concrete-`Connection` methods
    // (video keyframe streams, datagram send, input recv), so we use
    // `conn` directly.
    //
    // Bind `_client_decode_profiles` and `_client_hello` because the
    // immediate orchestration below doesn't read them — they're kept on
    // the session for callers (logs, diagnostics, future adaptive
    // policy) and accessible via the field names if needed.
    let HostSession {
        channel: _,
        negotiated: chosen_profile,
        client_hello,
        client_decode_profiles: _client_decode_profiles,
        idr_signal: force_idr,
        stream_ready,
    } = session;
    let initial_viewport = client_hello
        .viewport
        .filter(|v| v.is_valid());

    // `use_test_pattern` from here on only switches the capture source;
    // the handshake-time profile floor it implied is already baked into
    // the negotiated profile.

    // Cursor source. Behind the [`CursorSource`] trait so a future
    // per-platform impl (Wayland SPA_META_Cursor parser, macOS NSCursor
    // poller) can drop in without touching this block. The placeholder
    // emits one 16×16 checkerboard at startup so the wire is
    // exercised — that's the same shape the previous inline code sent.
    drain_initial_cursor(&conn, &mut PlaceholderCursorSource::new()).await;

    // DisplayList: one entry, single-monitor placeholder. Real values
    // (refresh rate from the PipeWire stream, scale + position from
    // the compositor) get filled in when the capture backend grows a
    // display-enumeration API. The send is here so the client gets
    // the topology *before* any video arrives.
    {
        let display = tether_protocol::control::DisplayDescriptor {
            id: 0,
            name: String::new(),
            // Resolution is not known until the first frame arrives;
            // use 0 as "to be replaced." Future: defer DisplayList
            // until the capture backend has reported its real dims.
            width: 0,
            height: 0,
            refresh_mhz: 60_000,
            scale_num: 1,
            scale_den: 1,
            primary: true,
            position: (0, 0),
        };
        let msg = ControlMessage::DisplayList {
            displays: vec![display],
        };
        if let Err(e) = conn.send_control(&msg).await {
            warn!(error = ?e, "initial DisplayList send failed; continuing anyway");
        }
    }

    // Acquire a capture stream — either real platform capture or the
    // synthetic test pattern fallback. Real Linux capture is async (the
    // portal handshake awaits a user permission dialog); test pattern is
    // sync. Both end up as a `Receiver<CapturedFrame>` so the send loop
    // is identical.
    let frames = pick_capture_source(use_test_pattern, chosen_profile).await?;

    // Force-IDR signal + stream-readiness gate were created by
    // `HostSession::accept` and destructured at the top of this fn.
    // `force_idr`: control-stream recv task `raise`s it on
    // `ControlMessage::ForceIdr`; the capture/encode thread `take`s it
    // each frame. Coalescing comes for free — N raises between two
    // takes produce one keyframe. See `tether_session::IdrSignal`.

    // Shared display-dimensions channel: the capture thread learns the
    // real host display size on the first frame and posts (w, h) here;
    // the dims-follower task reads it and feeds the injector via
    // set_display_size. We use a single-slot watch so the injector
    // always reads the latest known dims even if it polls late.
    let (display_dims_tx, display_dims_rx) =
        tokio::sync::watch::channel::<Option<(u32, u32)>>(None);

    // Capture + send runs on a dedicated OS thread per the expert review:
    // the hot path doesn't share the tokio runtime with anything else.
    // We keep the JoinHandle so the disconnect path can wait for the
    // thread to actually exit before we return — otherwise the encoder
    // and capture receiver would still be live in the background while
    // a follow-on session tried to grab the same resources. The shutdown
    // flag breaks the send loop out of its idle wait on a quiet desktop,
    // where `frames.recv` would otherwise block past disconnect detection.
    let send_shutdown = Arc::new(AtomicBool::new(false));
    // Drop-oldest single-slot mailbox for the most recent `ClientStats`
    // window. Written by the control recv task, drained by the
    // encode-and-send thread on each loop iteration.
    let latest_client_stats: LatestClientStats = Arc::new(StdMutex::new(None));
    // Latest viewport-target the client has communicated. Seeded
    // from ClientHello; mutated by SetClientViewport.
    let latest_viewport: LatestViewport = Arc::new(StdMutex::new(ViewportState {
        viewport: initial_viewport,
        seq: if initial_viewport.is_some() { 1 } else { 0 },
    }));
    // `stream_ready`: gate at the head of the send loop. The client
    // signals it has built its decoders by sending
    // `ControlMessage::StreamReady`; until then we drop captured frames
    // so the first ~100-500 ms don't race the client's decoder
    // construction and render as garbage. Bound to the session in
    // `HostSession::accept`.
    let conn_send = conn.clone();
    let force_idr_for_send = force_idr.clone();
    let send_shutdown_for_thread = send_shutdown.clone();
    let stream_ready_for_thread = stream_ready.clone();
    let latest_client_stats_for_send = latest_client_stats.clone();
    let latest_viewport_for_send = latest_viewport.clone();
    // Keyframes ride a reliable per-IDR QUIC unidirectional stream
    // rather than the unreliable datagram path used for P-frames.
    // The sync send thread is not a tokio worker, so it calls into
    // the async send via `Handle::block_on`. We send keyframes
    // synchronously (block the send thread until the IDR is queued
    // into quinn) instead of routing them through an mpsc to a
    // separate task, because:
    //   1. Ordering: a P-frame's stream_epoch must agree with the
    //      IDR's. The earlier mpsc design opened a window where
    //      `bump_epoch()` could fire between IDR enqueue and IDR
    //      wire-write, mis-attributing the IDR to the old epoch.
    //   2. Cost: keyframes are request-driven and infrequent. The
    //      blocking write is ~1 ms on LAN (open_uni + write_all +
    //      finish into quinn's send buffer). That frame's latency
    //      budget already absorbs an extra round trip for
    //      reliability.
    let runtime_handle_for_send = tokio::runtime::Handle::current();
    let conn_keyframe = conn.clone();
    let send_handle = std::thread::Builder::new()
        .name("tether-host-send".into())
        .spawn(move || {
            run_capture_and_send(
                conn_send,
                frames,
                force_idr_for_send,
                display_dims_tx,
                send_shutdown_for_thread,
                chosen_profile,
                stream_ready_for_thread,
                runtime_handle_for_send,
                conn_keyframe,
                latest_client_stats_for_send,
                latest_viewport_for_send,
            )
        })?;

    // Per-connection injector. The libei session lives inside the
    // injector; when the last Arc drops, libei releases. We deliberately
    // hand the three clones to the three recv tasks and don't keep a
    // fourth reference in this scope — otherwise the original outlives
    // the tasks, refcount never hits zero, and the host's mouse stays
    // grabbed until the process exits (which is the bug that prompted
    // this whole rewrite). The recv tasks themselves are owned by the
    // JoinSet below, so tasks.shutdown() is what triggers the final drop.
    //
    // tokio::sync::Mutex is the right primitive: the lock is held only
    // for the duration of one enigo call (microseconds), but both
    // holders are async, and a std::sync::Mutex held across an await
    // would risk blocking a tokio worker.
    let injector = Arc::new(TokioMutex::new(
        tether_input::inject::default_injector().await,
    ));

    let mut tasks: JoinSet<()> = JoinSet::new();

    // Control recv: react to ForceIdr and clock-probe requests on the
    // reliable control stream. Goodbye returns immediately so the
    // disconnect path runs as soon as the client signals; unknown
    // messages are logged at trace and the loop continues. We never
    // crash on a control packet.
    {
        let conn = conn.clone();
        let force_idr = force_idr.clone();
        let stream_ready_ctl = stream_ready.clone();
        let latest_client_stats_for_ctl = latest_client_stats.clone();
        let latest_viewport_for_ctl = latest_viewport.clone();
        let force_idr_for_viewport = force_idr.clone();
        tasks.spawn(async move {
            loop {
                match conn.recv_control().await {
                    Ok(ControlMessage::ForceIdr) => {
                        tracing::debug!("client requested IDR");
                        force_idr.raise();
                    }
                    Ok(ControlMessage::StreamReady { video, audio }) => {
                        info!(
                            video,
                            audio, "client signalled StreamReady; opening the gate"
                        );
                        stream_ready_ctl.store(true, Ordering::Release);
                    }
                    Ok(ControlMessage::StreamPause { display }) => {
                        let display_id = display;
                        info!(display_id, "client paused stream (no-op today)");
                    }
                    Ok(ControlMessage::StreamResume { display }) => {
                        let display_id = display;
                        info!(display_id, "client resumed stream (no-op today)");
                        // Force a fresh IDR so the client can latch
                        // onto the resumed stream without a partial GOP.
                        force_idr.raise();
                    }
                    Ok(ControlMessage::ClientStats {
                        interval_ms,
                        frames_received,
                        frames_dropped,
                        fragments_lost,
                        rtt_ewma_us,
                    }) => {
                        info!(
                            interval_ms,
                            frames_received,
                            frames_dropped,
                            fragments_lost,
                            rtt_ewma_us,
                            "client stats"
                        );
                        // Drop-oldest: the ABR controller only needs
                        // the most recent window. If the send thread
                        // hasn't drained yet (slow encoder loop), the
                        // older sample is discarded — we'd rather act
                        // on fresh telemetry than queue stale data.
                        *latest_client_stats_for_ctl.lock().unwrap() =
                            Some(ClientStatsObservation {
                                frames_dropped,
                                fragments_lost,
                            });
                    }
                    Ok(ControlMessage::SetClientViewport(v)) => {
                        info!(width = v.width, height = v.height, "client viewport changed");
                        // Latch the new viewport. The send thread
                        // notices the seq bump on its next iteration
                        // and rebuilds the encoder ONLY if encode
                        // dims actually change — on a GPU session the
                        // dims stay at capture and the rebuild is
                        // skipped (no GPU scaler yet). We force an
                        // IDR regardless so the client sees a clean
                        // cut on whatever the new dimensions turn
                        // out to be. On a GPU session that means the
                        // IDR fires mid-GOP with no epoch bump —
                        // harmless to the client (a normal IDR), but
                        // worth a debug breadcrumb in traces.
                        let next = if v.is_valid() { Some(v) } else { None };
                        let mut guard = latest_viewport_for_ctl.lock().unwrap();
                        guard.viewport = next;
                        guard.seq = guard.seq.wrapping_add(1);
                        drop(guard);
                        tracing::debug!(
                            width = v.width,
                            height = v.height,
                            "SetClientViewport: forcing IDR; encoder rebuild fires only if \
                             encode dims change (GPU sessions stay at capture dims until the \
                             wgpu scaler lands)"
                        );
                        force_idr_for_viewport.raise();
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
                        // Host doesn't currently initiate re-probes, but if
                        // we ever do, the matching response handler goes here.
                        tracing::trace!("unsolicited clock probe response; ignoring");
                    }
                    Ok(ControlMessage::Goodbye { reason, code }) => {
                        info!(%reason, ?code, "client said goodbye");
                        return;
                    }
                    Ok(ControlMessage::Extension { key, payload }) => {
                        tracing::debug!(
                            key = %key,
                            payload_len = payload.len(),
                            "unknown control extension; ignoring"
                        );
                    }
                    Ok(
                        ControlMessage::CursorShape { .. } | ControlMessage::CursorUseShape { .. },
                    ) => {
                        // Host-originated; receiving one here means the
                        // client misrouted. Log and drop.
                        tracing::debug!(
                            "unexpected host→client cursor message arrived on host; ignoring"
                        );
                    }
                    Ok(ControlMessage::DisplayList { .. }) => {
                        // Host-originated; misrouted if seen here.
                        tracing::debug!(
                            "unexpected host→client DisplayList arrived on host; ignoring"
                        );
                    }
                    Ok(ControlMessage::SetActiveDisplays { displays }) => {
                        // Single-display host today — log the request
                        // and ignore. The selection mechanic plugs in
                        // when multi-display capture lands.
                        info!(
                            ?displays,
                            "client requested display subset; ignoring (single-display host)"
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

    // Display-dimensions follower: any change to the capture's
    // negotiated resolution pushes new pixel dims into the injector.
    // Exits naturally when the send thread drops its display_dims_tx,
    // or via tasks.shutdown() during disconnect.
    {
        let injector = injector.clone();
        let mut rx = display_dims_rx;
        tasks.spawn(async move {
            while rx.changed().await.is_ok() {
                // Copy the value out and drop the borrow guard before
                // awaiting on the injector lock — `watch::Ref` is not Send,
                // so holding it across an .await fails the Send bound.
                let dims = *rx.borrow();
                if let Some((w, h)) = dims {
                    injector.lock().await.set_display_size(w, h);
                }
            }
        });
    }

    // Input recv: drain the client's input stream and feed each event
    // into the host's injection backend.
    {
        let conn = conn.clone();
        let injector = injector.clone();
        tasks.spawn(async move {
            loop {
                match conn.recv_input().await {
                    Ok(evt) => {
                        tracing::trace!(
                            event_id = evt.event_id,
                            t_client_ns = evt.t_client.0,
                            kind = ?evt.kind,
                            "input event"
                        );
                        let mut inj = injector.lock().await;
                        if let Err(e) = inj.inject(&evt) {
                            warn!(error = %e, "injector rejected event; dropping");
                        }
                    }
                    Err(e) => {
                        warn!(error = ?e, "input recv failed; ending input task");
                        return;
                    }
                }
            }
        });
    }

    // Datagram recv: cursor packets ride the unreliable channel for
    // latency. Video and host-cursor datagrams flow the other
    // direction; they should never arrive here, but we match
    // defensively so a misbehaving client can't crash the host.
    //
    // This consumes the third (and final) injector clone in this
    // scope; after the spawn, the only references are in the three
    // task closures, and `injector` is no longer in scope.
    {
        let conn = conn.clone();
        let injector = injector;
        tasks.spawn(async move {
            loop {
                match conn.recv_datagram().await {
                    Ok(Datagram::ClientCursor(c)) => {
                        let mut inj = injector.lock().await;
                        if let Err(e) = inj.inject_cursor(&c) {
                            warn!(error = %e, "cursor inject failed; dropping");
                        }
                    }
                    Ok(Datagram::Video(_)) | Ok(Datagram::HostCursor(_)) => {
                        tracing::trace!("unexpected host-direction datagram on host; ignoring");
                    }
                    Err(e) => {
                        warn!(error = ?e, "datagram recv failed; ending datagram task");
                        return;
                    }
                }
            }
        });
    }

    // Wait for any signal that the session is over:
    //   - Ctrl-C: user wants out
    //   - any per-connection task exited: disconnect, Goodbye, or recv error
    // Whichever fires first, the cleanup path below runs.
    tokio::select! {
        ctrl_c = tokio::signal::ctrl_c() => {
            if let Err(e) = ctrl_c {
                warn!(error = %e, "ctrl-c handler failed; tearing down anyway");
            } else {
                info!("ctrl-c received, ending session");
            }
        }
        res = tasks.join_next() => {
            match res {
                Some(Ok(())) => info!("per-connection task ended; tearing down"),
                Some(Err(e)) => warn!(error = ?e, "per-connection task failed; tearing down"),
                None => warn!("joined empty task set; tearing down"),
            }
        }
    }

    // Close the QUIC connection. This makes send_datagram error in the
    // send thread, breaking it out of the encode loop, and tells any
    // still-alive recv tasks that the peer is gone. Cheap and idempotent.
    conn.close(0, b"session ended");

    // Tell the send thread to stop polling the capture channel. Paired
    // with the recv_timeout inside the loop so a static desktop (no new
    // frames) can't keep us blocked in `frames.recv` past disconnect.
    send_shutdown.store(true, Ordering::Relaxed);

    // Abort and await the remaining recv tasks. Each one holds an Arc
    // clone of the injector; once shutdown() returns they're all dropped,
    // leaving the injector at refcount 0 (we deliberately didn't keep
    // a clone in this scope), so the injector drops here and libei
    // releases the host's mouse and keyboard.
    tasks.shutdown().await;

    // Wait for the send thread to actually exit. spawn_blocking lets us
    // await the std::thread::join without parking a tokio worker. We
    // swallow the result — a panicked send thread is logged inside the
    // join, and we're tearing down anyway.
    let _ = tokio::task::spawn_blocking(move || send_handle.join()).await;

    Ok(())
}

/// Encoder paired with the input dimensions it was configured for, so we
/// can detect a resolution change in the capture stream and recreate the
/// encoder (plus bump the wire-side stream epoch) before the next frame.
/// `Box<dyn Encoder>` lets the probe swap hardware backends in without
/// the encode loop knowing which one it got.
struct EncoderSlot {
    encoder: Box<dyn Encoder>,
    /// Capture-side dimensions. Bumping these means PipeWire / SCK
    /// re-negotiated the source — same rebuild path as a viewport
    /// change.
    capture_width: u32,
    capture_height: u32,
    /// Encoder output dimensions, i.e. what gets sent over the wire.
    /// Equals capture dims when no viewport is in play; otherwise
    /// equals the letterboxed-and-aligned viewport.
    width: u32,
    height: u32,
    /// `(Viewport, seq)` of the viewport directive that produced
    /// `width`/`height`. `seq == 0` means "no viewport in effect."
    /// The send loop compares `seq` against the shared mailbox to
    /// detect mid-session resize requests without diffing the
    /// Viewport itself.
    viewport_seq: u64,
    /// Adaptive bitrate state. `None` if the encoder reports
    /// `supports_changing_bitrate() == false` — we still send video,
    /// we just don't try to retune. A resolution change destroys the
    /// whole slot (and the controller along with it), which is the
    /// right reset point: the cumulative quinn counters keep ticking
    /// but a freshly-built encoder starts at its own baseline.
    abr: Option<AbrState>,
    /// Lazily-built BGRA→NV12 + DMA-BUF bridge for the zero-copy
    /// capture→encode path. `NotYetBuilt` while the stream is SHM-only
    /// or before any Gpu frame has been seen; `Ready` after the first
    /// successful build for this resolution. No `Failed` state — the
    /// startup probe (`importable_dmabuf_modifiers`) gates whether the
    /// compositor ever offers DMA-BUF in the first place, so failure at
    /// this layer would be an anomalous device-loss / OOM and is
    /// surfaced as a fatal error that exits the send loop rather than a
    /// silent per-frame drop.
    #[cfg(target_os = "linux")]
    bridge: BridgeState,
    /// Mitchell-Netravali downscaler inserted between PipeWire's BGRA
    /// dma-buf import and the gpuconvert bridge when the client's
    /// viewport calls for a smaller encode than the capture size.
    /// `None` when `capture_dims == encode_dims` (1:1 pass-through;
    /// the bridge consumes the imported texture directly). Lazily
    /// built alongside the bridge so it can reuse the bridge's wgpu
    /// device — see `build_scaler_for_slot`.
    ///
    /// macOS host has no equivalent here: SCK already does the
    /// scaling at capture time via `SCStreamConfiguration.width/
    /// height`. Unifying the Mac host path on this shader is the
    /// phase-2 follow-up tracked in ARCHITECTURE.md.
    #[cfg(target_os = "linux")]
    scaler: Option<Scaler>,
}

/// Per-encoder adaptive bitrate state.
struct AbrState {
    controller: AbrController,
    /// Snapshot of quinn's cumulative path counters at the previous
    /// `observe` call. The controller takes deltas itself.
    last_quinn: AbrSnapshot,
    /// Wall-clock instant of the previous `observe` call.
    last_observed_at: Instant,
    /// Bitrate currently programmed into the encoder. Lets us skip
    /// the syscall when the controller hasn't moved.
    last_applied_kbps: u32,
}

#[cfg(target_os = "linux")]
enum BridgeState {
    NotYetBuilt,
    Ready(GpuConvertBridge),
}

/// Negotiated-chroma-specific gpuconvert bridge. The encoder picks
/// which one to build during lazy init; from then on the variant is
/// fixed for the encoder's lifetime (a chroma switch needs a full
/// encoder rebuild, same as a resolution change).
#[cfg(target_os = "linux")]
enum GpuConvertBridge {
    Nv12(Nv12DmaBuf),
    Yuv444(Yuv444DmaBuf),
    P010(Bgra2P010DmaBuf),
}

/// Outcome of one Gpu-frame encode attempt. Distinguishes per-frame
/// failures (drop this frame, keep going) from bridge construction
/// failure (no recovery; exit the send loop). Per-frame errors are
/// usually transient (a single bad PipeWire buffer); bridge-init failure
/// after the startup probe succeeded indicates device loss or OOM and
/// the client would just freeze if we silently dropped every subsequent
/// frame.
#[cfg(target_os = "linux")]
enum GpuEncodeOutcome {
    Packets(Vec<tether_codec::EncodedPacket>),
    DropFrame(anyhow::Error),
    Fatal(anyhow::Error),
}

/// Borrow the bridge's wgpu device + queue for sharing with the
/// scaler. Both are cheap-to-clone handle types (Arc-backed
/// internally) so cloning here costs an atomic increment, not a real
/// resource alloc.
#[cfg(target_os = "linux")]
fn bridge_device_queue(b: &GpuConvertBridge) -> (wgpu::Device, wgpu::Queue) {
    match b {
        GpuConvertBridge::Nv12(b) => (b.device().clone(), b.queue().clone()),
        GpuConvertBridge::Yuv444(b) => (b.device().clone(), b.queue().clone()),
        GpuConvertBridge::P010(b) => (b.device().clone(), b.queue().clone()),
    }
}

/// Compile scaler pipelines + construct a Scaler for the given (src,
/// dst). The Scaler holds the pipelines internally via Arc, so we
/// don't need to retain a separate handle in the slot — when the
/// slot is destroyed (on resolution change) the pipelines drop with
/// it. Future "rebuild scaler with new dims but keep the pipelines"
/// fast-path would expose `pipelines.clone()` here for the caller to
/// stash; today there's no such caller.
#[cfg(target_os = "linux")]
fn build_scaler_for_slot(
    device: wgpu::Device,
    queue: wgpu::Queue,
    src_dims: (u32, u32),
    dst_dims: (u32, u32),
) -> Result<Scaler, ScalerError> {
    let pipelines = Arc::new(ScalerPipelines::build(&device));
    Scaler::new(pipelines, device, queue, src_dims, dst_dims)
}

/// If `scaler` is `Some`, run it on `imported` and return a reference
/// to the scaler's output texture. Otherwise return `imported`. The
/// lifetime ties to whichever is borrowed.
#[cfg(target_os = "linux")]
fn scale_if_needed<'a>(
    scaler: Option<&'a Scaler>,
    imported: &'a wgpu::Texture,
) -> Result<&'a wgpu::Texture, anyhow::Error> {
    match scaler {
        Some(s) => s.scale(imported).map_err(|e| {
            // DimMismatch is the resize-race signal — capture
            // changed dims before the slot rebuilt. One frame drop
            // is acceptable; a persistent stream of these points at
            // a bookkeeping bug (slot.capture_width drifting off
            // live capture dims) that the next slot rebuild won't
            // fix. Log every occurrence so the pattern is visible.
            if matches!(e, ScalerError::DimMismatch { .. }) {
                warn!(error = %e, "scaler dim mismatch on encode; dropping frame");
            }
            anyhow::anyhow!("Scaler::scale: {e}")
        }),
        None => Ok(imported),
    }
}

/// Encode one PipeWire-supplied DMA-BUF frame through the zero-copy
/// pipeline: import BGRA into wgpu, compute BGRA→(NV12|YUV444) onto
/// exported DMA-BUF planes, hand them to the encoder's `encode_gpu`.
///
/// The bridge variant matches the negotiated chroma — NV12 for 4:2:0,
/// YUV444 for HEVC Main444. Chosen on lazy init and fixed for the
/// encoder's lifetime (chroma switch needs a full encoder rebuild,
/// same as resolution change).
#[cfg(target_os = "linux")]
fn encode_gpu_frame(
    slot: &mut EncoderSlot,
    chroma: tether_protocol::control::ChromaSubsampling,
    bit_depth: u8,
    gpu: tether_capture::GpuCapturedFrame,
    pts: i64,
    force_keyframe: bool,
) -> GpuEncodeOutcome {
    use tether_protocol::control::ChromaSubsampling;

    let bridge = match &mut slot.bridge {
        BridgeState::Ready(b) => b,
        BridgeState::NotYetBuilt => {
            let built = match (chroma, bit_depth) {
                (ChromaSubsampling::Yuv420, 8) => {
                    match pollster::block_on(Nv12DmaBuf::new(slot.width, slot.height)) {
                        Ok(b) => GpuConvertBridge::Nv12(b),
                        Err(e) => {
                            return GpuEncodeOutcome::Fatal(anyhow::anyhow!(
                                "Nv12 gpuconvert bridge init failed for {}x{} after \
                                 startup probe succeeded — device loss or OOM: {e}",
                                slot.width,
                                slot.height,
                            ));
                        }
                    }
                }
                (ChromaSubsampling::Yuv420, 10) => {
                    match pollster::block_on(Bgra2P010DmaBuf::new(slot.width, slot.height)) {
                        Ok(b) => GpuConvertBridge::P010(b),
                        Err(e) => {
                            return GpuEncodeOutcome::Fatal(anyhow::anyhow!(
                                "P010 gpuconvert bridge init failed for {}x{} after \
                                 startup probe succeeded — device loss or OOM: {e}",
                                slot.width,
                                slot.height,
                            ));
                        }
                    }
                }
                (ChromaSubsampling::Yuv444, 8) => {
                    match pollster::block_on(Yuv444DmaBuf::new(slot.width, slot.height)) {
                        Ok(b) => GpuConvertBridge::Yuv444(b),
                        Err(e) => {
                            return GpuEncodeOutcome::Fatal(anyhow::anyhow!(
                                "Yuv444 gpuconvert bridge init failed for {}x{} after \
                                 startup probe succeeded — device loss or OOM: {e}",
                                slot.width,
                                slot.height,
                            ));
                        }
                    }
                }
                // Yuv444 10-bit (XV30) has no bridge; tether-probe's
                // capture-stage check is supposed to keep it out of the
                // negotiated set. Reaching this arm is a contract
                // violation — fail loud rather than silently dropping
                // every frame.
                (chroma, bit_depth) => {
                    return GpuEncodeOutcome::Fatal(anyhow::anyhow!(
                        "no gpuconvert bridge for negotiated profile chroma={:?} \
                         bit_depth={} — tether_probe::host_supported_profiles \
                         should have filtered this profile out before negotiation",
                        chroma,
                        bit_depth,
                    ));
                }
            };
            info!(
                width = slot.width,
                height = slot.height,
                chroma = ?chroma,
                "gpuconvert bridge initialised for zero-copy DMA-BUF encode"
            );
            slot.bridge = BridgeState::Ready(built);

            // Build the Mitchell downscaler if the client viewport
            // means we're encoding smaller than the capture. Sharing
            // the bridge's wgpu device keeps the imported BGRA
            // texture, the scaler scratch, and the chroma output all
            // on the same GPU context — no cross-device copies.
            if slot.capture_width != slot.width || slot.capture_height != slot.height {
                let BridgeState::Ready(b) = &slot.bridge else {
                    unreachable!()
                };
                let (device, queue) = bridge_device_queue(b);
                match build_scaler_for_slot(
                    device,
                    queue,
                    (slot.capture_width, slot.capture_height),
                    (slot.width, slot.height),
                ) {
                    Ok(scaler) => {
                        info!(
                            capture_w = slot.capture_width,
                            capture_h = slot.capture_height,
                            encode_w = slot.width,
                            encode_h = slot.height,
                            mip_levels = scaler.mip_levels(),
                            "Mitchell scaler built for viewport downscale"
                        );
                        slot.scaler = Some(scaler);
                    }
                    Err(e) => {
                        return GpuEncodeOutcome::Fatal(anyhow::anyhow!(
                            "scaler construction failed for {}x{} -> {}x{}: {e}",
                            slot.capture_width,
                            slot.capture_height,
                            slot.width,
                            slot.height,
                        ));
                    }
                }
            }

            let BridgeState::Ready(b) = &mut slot.bridge else {
                unreachable!()
            };
            b
        }
    };

    let GpuCapturedSource::DmaBuf(dmabuf) = gpu.source;
    let tether_capture::CapturedDmaBuf {
        fourcc,
        fd,
        stride,
        offset,
        modifier,
    } = dmabuf;
    let _ = fourcc; // PipeWire-side fourcc is informational; the
                    // shader treats input as BGRA regardless.

    // Helper: if the slot has a scaler, run it on the imported BGRA
    // and return the scaled texture; otherwise return the imported
    // texture unchanged. The returned reference borrows either from
    // `imported` (no-scale) or from `scaler.output()` (scaler held by
    // the slot, lifetime-bound to slot).
    //
    // Coded as a `match` not a helper because the borrow checker
    // doesn't love mixing borrows of `imported` (local) and
    // `slot.scaler` (caller-owned) through a function boundary.
    let codec_frame = match bridge {
        GpuConvertBridge::Nv12(b) => {
            let imported = match b.import_bgra_dmabuf(
                fd,
                modifier,
                stride,
                offset,
                slot.capture_width,
                slot.capture_height,
            ) {
                Ok(t) => t,
                Err(e) => {
                    return GpuEncodeOutcome::DropFrame(anyhow::anyhow!(
                        "import_bgra_dmabuf (nv12 bridge): {e}"
                    ));
                }
            };
            let bridge_input = match scale_if_needed(slot.scaler.as_ref(), &imported) {
                Ok(t) => t,
                Err(e) => return GpuEncodeOutcome::DropFrame(e),
            };
            let nv12 = match b.convert(bridge_input) {
                Ok(f) => f,
                Err(e) => {
                    return GpuEncodeOutcome::DropFrame(anyhow::anyhow!(
                        "Nv12DmaBuf::convert: {e}"
                    ));
                }
            };
            drop(imported);
            nv12_dmabuf_to_codec_frame(nv12)
        }
        GpuConvertBridge::Yuv444(b) => {
            let imported = match b.import_bgra_dmabuf(
                fd,
                modifier,
                stride,
                offset,
                slot.capture_width,
                slot.capture_height,
            ) {
                Ok(t) => t,
                Err(e) => {
                    return GpuEncodeOutcome::DropFrame(anyhow::anyhow!(
                        "import_bgra_dmabuf (yuv444 bridge): {e}"
                    ));
                }
            };
            let bridge_input = match scale_if_needed(slot.scaler.as_ref(), &imported) {
                Ok(t) => t,
                Err(e) => return GpuEncodeOutcome::DropFrame(e),
            };
            let yuv = match b.convert(bridge_input) {
                Ok(f) => f,
                Err(e) => {
                    return GpuEncodeOutcome::DropFrame(anyhow::anyhow!(
                        "Yuv444DmaBuf::convert: {e}"
                    ));
                }
            };
            drop(imported);
            yuv444_dmabuf_to_codec_frame(yuv)
        }
        GpuConvertBridge::P010(b) => {
            let imported = match b.import_bgra_dmabuf(
                fd,
                modifier,
                stride,
                offset,
                slot.capture_width,
                slot.capture_height,
            ) {
                Ok(t) => t,
                Err(e) => {
                    return GpuEncodeOutcome::DropFrame(anyhow::anyhow!(
                        "import_bgra_dmabuf (p010 bridge): {e}"
                    ));
                }
            };
            let bridge_input = match scale_if_needed(slot.scaler.as_ref(), &imported) {
                Ok(t) => t,
                Err(e) => return GpuEncodeOutcome::DropFrame(e),
            };
            let p010 = match b.convert(bridge_input) {
                Ok(f) => f,
                Err(e) => {
                    return GpuEncodeOutcome::DropFrame(anyhow::anyhow!(
                        "Bgra2P010DmaBuf::convert: {e}"
                    ));
                }
            };
            drop(imported);
            p010_dmabuf_to_codec_frame(p010)
        }
    };

    match slot
        .encoder
        .encode_gpu(GpuEncoderFrame::DmaBuf(&codec_frame), pts, force_keyframe)
    {
        Ok(packets) => GpuEncodeOutcome::Packets(packets),
        Err(e) => GpuEncodeOutcome::DropFrame(anyhow::anyhow!("encode_gpu: {e}")),
    }
}

/// Encode one ScreenCaptureKit-supplied IOSurface through the
/// VideoToolbox zero-copy path. Simpler than the Linux equivalent —
/// no gpuconvert bridge, no NV12 conversion, no `BridgeState`: SCK
/// hands us NV12 IOSurfaces directly and the encoder consumes them
/// as-is via `CVPixelBufferCreateWithIOSurface`.
///
/// The capture-side `release_guard` keeps the underlying IOSurface
/// alive for the duration of this call; the encoder's
/// `submit_iosurface` performs a fresh CFRetain on the wrapping
/// CVPixelBuffer so the surface stays valid for the encoder's
/// async work after we return.
#[cfg(target_os = "macos")]
fn encode_iosurface_frame(
    slot: &mut EncoderSlot,
    gpu: tether_capture::GpuCapturedFrame,
    pts: i64,
    force_keyframe: bool,
) -> anyhow::Result<Vec<tether_codec::EncodedPacket>> {
    let tether_capture::GpuCapturedSource::IOSurface(iosurface) = gpu.source;
    let codec_frame = tether_codec::IOSurfaceFrame {
        surface: iosurface.surface,
        pixel_format: iosurface.pixel_format,
        width: iosurface.width,
        height: iosurface.height,
    };
    let packets = slot.encoder.encode_gpu(
        GpuEncoderFrame::IOSurface(&codec_frame),
        pts,
        force_keyframe,
    )?;
    // `gpu` (and its `release_guard`) falls out of scope at function
    // end, releasing the capture-side CMSampleBuffer + IOSurface
    // retains. By that point `submit_iosurface` has already taken its
    // own CFRetain on the wrapping CVPixelBuffer, so the surface
    // outlives this scope as needed by the encoder. Ordering here is
    // not load-bearing — no explicit `drop` needed.
    Ok(packets)
}

/// Build a `DmaBufFrame` (NV12, one object, two layers — Y as R8 and
/// UV as GR88, both pointing at `object_index=0` with their offsets
/// within the shared allocation) from the bridge's per-call descriptor.
/// FFmpeg's `av_hwframe_map(DRM_PRIME → VAAPI)` only accepts NV12 as a
/// single DRM object with planes at distinct offsets — separate-fd
/// NV12 fails with `"VAAPI can only map frames made from a single DRM
/// object"`. The bridge allocates one shared `VkDeviceMemory` for both
/// planes precisely to produce this shape.
#[cfg(target_os = "linux")]
fn nv12_dmabuf_to_codec_frame(out: Nv12DmaBufFrame) -> DmaBufFrame {
    DmaBufFrame {
        fourcc: u32::from_le_bytes(*b"NV12"),
        objects: vec![DmaBufObject {
            fd: out.fd,
            size: out.size,
            drm_format_modifier: out.modifier,
        }],
        layers: vec![
            DmaBufLayer {
                drm_format: u32::from_le_bytes(*b"R8  "),
                num_planes: 1,
                object_index: [0, 0, 0, 0],
                offset: [
                    u32::try_from(out.y_offset).expect("Y plane offset fits in u32"),
                    0,
                    0,
                    0,
                ],
                pitch: [
                    u32::try_from(out.y_stride).expect("Y plane stride fits in u32"),
                    0,
                    0,
                    0,
                ],
            },
            DmaBufLayer {
                drm_format: u32::from_le_bytes(*b"GR88"),
                num_planes: 1,
                object_index: [0, 0, 0, 0],
                offset: [
                    u32::try_from(out.uv_offset).expect("UV plane offset fits in u32"),
                    0,
                    0,
                    0,
                ],
                pitch: [
                    u32::try_from(out.uv_stride).expect("UV plane stride fits in u32"),
                    0,
                    0,
                    0,
                ],
            },
        ],
    }
}

/// Build a `DmaBufFrame` for the P010 (HEVC Main10 / 4:2:0 10-bit) path.
/// Thin adapter over `tether_codec::build_p010_dmabuf_frame` — the
/// shared helper takes raw fields so tether-codec doesn't need to
/// depend on tether-gpuconvert just to know about `P010DmaBufFrame`.
#[cfg(target_os = "linux")]
fn p010_dmabuf_to_codec_frame(out: P010DmaBufFrame) -> DmaBufFrame {
    tether_codec::build_p010_dmabuf_frame(
        out.fd,
        out.size,
        out.modifier,
        out.y_offset,
        out.y_stride,
        out.uv_offset,
        out.uv_stride,
    )
}

/// Build a `DmaBufFrame` for the YUV 4:4:4 path: one DRM object, one
/// `XYUV` (DRM_FORMAT_XYUV8888) layer, one packed plane (32 bpp).
///
/// Why packed (not planar): ffmpeg 8.x's `vaapi_drm_format_map` has
/// no entry for DRM_FORMAT_YUV444 / planar YUV444P / three-R8-layer
/// shapes — `av_hwframe_map(DRM_PRIME → VAAPI)` rejects them all
/// with "DRM format not supported by VAAPI". DRM_FORMAT_XYUV8888 IS
/// in the table (maps to VA_FOURCC_XYUV / AV_PIX_FMT_VUYX) and is
/// accepted as input to HEVC Main 4:4:4 encode. See
/// `crates/tether-gpuconvert/src/dmabuf_export/shared_yuv444.rs`.
#[cfg(target_os = "linux")]
fn yuv444_dmabuf_to_codec_frame(out: Yuv444DmaBufFrame) -> DmaBufFrame {
    DmaBufFrame {
        fourcc: u32::from_le_bytes(*b"XYUV"),
        objects: vec![DmaBufObject {
            fd: out.fd,
            size: out.size,
            drm_format_modifier: out.modifier,
        }],
        layers: vec![DmaBufLayer {
            drm_format: u32::from_le_bytes(*b"XYUV"),
            num_planes: 1,
            object_index: [0, 0, 0, 0],
            offset: [
                u32::try_from(out.offset).expect("plane offset fits in u32"),
                0,
                0,
                0,
            ],
            pitch: [
                u32::try_from(out.stride).expect("plane stride fits in u32"),
                0,
                0,
                0,
            ],
        }],
    }
}

/// Drain any pending events from the cursor source and forward them
/// to the client on the reliable control stream. Called at session
/// start so the placeholder's one-shot shape lands before the client
/// gets its first video frame; when a real cursor source replaces
/// the placeholder this same call covers initial-state delivery.
///
/// Per-session sprite cache keeps `CursorShape` (pixels) from re-
/// sending when the platform reports the same id twice in a row —
/// the second hit becomes `CursorUseShape { id }`.
async fn drain_initial_cursor<S: CursorSource>(conn: &Connection, source: &mut S) {
    // Forward-looking cache: a future real cursor source emits the
    // same arrow id every time the pointer enters the desktop, so a
    // `Shape` -> `UseShape` dedup pays off. The placeholder emits a
    // single unique id so this cache is empty-allocated overhead
    // today — kept because the real impl will need it.
    let mut seen_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
    loop {
        match source.next_event() {
            CursorEvent::Idle => break,
            CursorEvent::Shape(shape) => {
                let id = shape.id;
                let already_sent = !seen_ids.insert(id);
                if !already_sent {
                    // Deposit the pixels into the client's cache.
                    // `CursorShape` defines but does NOT activate; the
                    // protocol pairs it with `CursorUseShape` so the
                    // host can swap between cached cursors without
                    // resending pixels.
                    let msg = ControlMessage::CursorShape {
                        id,
                        hotspot: shape.hotspot,
                        width: shape.width,
                        height: shape.height,
                        format: shape.format,
                        pixels: shape.pixels,
                    };
                    if let Err(e) = conn.send_control(&msg).await {
                        warn!(error = ?e, id, "CursorShape send failed; continuing anyway");
                    }
                }
                // Always activate — applies to both first-send and
                // cached-switch cases. Without this, defining a shape
                // never becomes the visible cursor on the client.
                if let Err(e) = conn
                    .send_control(&ControlMessage::CursorUseShape { id })
                    .await
                {
                    warn!(error = ?e, id, "CursorUseShape send failed; continuing anyway");
                }
            }
        }
    }
}

/// Encoder alignment in pixels. H.264 / HEVC both want even widths
/// (chroma sampling); HEVC Main444 hardware encoders empirically
/// reject widths that aren't a multiple of 16. Round to 16 to cover
/// all cases — the worst-case waste is 15 pixels per side, which is
/// invisible at any reasonable client window size.
const ENCODER_ALIGN: u32 = 16;

/// Compute the encoder-output dimensions for a given capture size
/// and client viewport. The viewport bounds the longest edge; we
/// preserve aspect ratio (letterbox at the client; never stretch)
/// and clamp to [`ENCODER_ALIGN`]. `None` returns the capture dims
/// (with the same alignment guarantee).
fn encode_dims_for_viewport(
    capture_w: u32,
    capture_h: u32,
    viewport: Option<Viewport>,
) -> (u32, u32) {
    // No active viewport → pass capture dims through unchanged. H.264
    // and HEVC carry crop offsets in SPS for non-16-aligned heights
    // (e.g. the very common 1920x1080), so the encoder handles it.
    // Flooring to 16 here would silently drop 8 rows on a 1080p host
    // for no reason.
    let Some(v) = viewport.filter(|v| v.is_valid()) else {
        return (capture_w, capture_h);
    };
    // Fit capture inside viewport at fixed aspect ratio: scale by the
    // smaller of the two ratios. Never upscale — the client renderer
    // can scale up for cheap; we're not paying the encoder cost to do
    // it host-side.
    let scale_w = f64::from(v.width) / f64::from(capture_w);
    let scale_h = f64::from(v.height) / f64::from(capture_h);
    let scale = scale_w.min(scale_h).min(1.0);
    let target_w = (f64::from(capture_w) * scale).round() as u32;
    let target_h = (f64::from(capture_h) * scale).round() as u32;
    // When viewport IS in play we floor to 16: the encoder rejects
    // misaligned dims on HEVC Main444 hardware, and floor (not
    // nearest) is what keeps us inside the viewport budget.
    let aligned_w = (target_w / ENCODER_ALIGN) * ENCODER_ALIGN;
    let aligned_h = (target_h / ENCODER_ALIGN) * ENCODER_ALIGN;
    (aligned_w.max(ENCODER_ALIGN), aligned_h.max(ENCODER_ALIGN))
}

/// Pure CPU bilinear BGRA resize. Used on the [`CapturedFrame::Cpu`]
/// path when a viewport contracts the output below capture dims.
/// Allocates a fresh `Vec<u8>` per call — the alternative (caller-
/// owned scratch) would require lifetime gymnastics for a per-frame
/// cost that's already dominated by encoder time.
///
/// Tradeoff: GPU paths (DMA-BUF on Linux, IOSurface on macOS) don't
/// reach this function. A real GPU compute scaler is the planned
/// follow-up; for now those paths encode at capture dims regardless
/// of viewport.
fn resize_bgra_bilinear(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    debug_assert_eq!((src_w * src_h * 4) as usize, src.len());
    let mut out = vec![0u8; (dst_w * dst_h * 4) as usize];
    if dst_w == 0 || dst_h == 0 {
        return out;
    }
    // Standard inverse-mapping bilinear. Computes a single
    // floating-point sample per channel; on a 1920x1200 → 1280x800
    // resize this lands at ~5 ms on a recent x86 core. Adequate for
    // the test-pattern / SHM-fallback path. If the GPU scaler ever
    // lands ahead of schedule, this becomes dead code.
    let sx_step = f64::from(src_w) / f64::from(dst_w);
    let sy_step = f64::from(src_h) / f64::from(dst_h);
    for y in 0..dst_h {
        let sy = (f64::from(y) + 0.5) * sy_step - 0.5;
        let y0 = sy.floor().max(0.0) as u32;
        let y1 = (y0 + 1).min(src_h - 1);
        let dy = (sy - f64::from(y0)).clamp(0.0, 1.0);
        for x in 0..dst_w {
            let sx = (f64::from(x) + 0.5) * sx_step - 0.5;
            let x0 = sx.floor().max(0.0) as u32;
            let x1 = (x0 + 1).min(src_w - 1);
            let dx = (sx - f64::from(x0)).clamp(0.0, 1.0);
            let i00 = ((y0 * src_w + x0) * 4) as usize;
            let i01 = ((y0 * src_w + x1) * 4) as usize;
            let i10 = ((y1 * src_w + x0) * 4) as usize;
            let i11 = ((y1 * src_w + x1) * 4) as usize;
            let out_idx = ((y * dst_w + x) * 4) as usize;
            for c in 0..4 {
                let p00 = f64::from(src[i00 + c]);
                let p01 = f64::from(src[i01 + c]);
                let p10 = f64::from(src[i10 + c]);
                let p11 = f64::from(src[i11 + c]);
                let top = p00 * (1.0 - dx) + p01 * dx;
                let bot = p10 * (1.0 - dx) + p11 * dx;
                let v = top * (1.0 - dy) + bot * dy;
                out[out_idx + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Fold one observation into the encoder's ABR controller and apply
/// the resulting bitrate target if it changed. No-op when the encoder
/// reports `supports_changing_bitrate() == false` (slot.abr is None).
fn tick_abr(slot: &mut EncoderSlot, conn: &Connection, stats: ClientStatsObservation) {
    let Some(abr) = slot.abr.as_mut() else {
        return;
    };
    let now = Instant::now();
    let dt = now.saturating_duration_since(abr.last_observed_at);
    let quinn = conn.quinn_stats();
    let sample = AbrSample {
        rtt: quinn.rtt,
        congestion_events_delta: quinn
            .congestion_events
            .saturating_sub(abr.last_quinn.congestion_events),
        lost_packets_delta: quinn
            .lost_packets
            .saturating_sub(abr.last_quinn.lost_packets),
        client_fragments_lost: stats.fragments_lost,
        client_frames_dropped: stats.frames_dropped,
    };
    abr.last_quinn = quinn;
    abr.last_observed_at = now;
    let decision = abr.controller.observe(dt, sample);
    // The controller also publishes `target_fps`, but no capture
    // backend takes an FPS hint at runtime yet — capture rates are
    // fixed at startup. The FPS gear runs anyway because client
    // `frames_dropped` is the signal that distinguishes
    // network-side stress from decoder-side stress (the former
    // should drop bitrate, the latter should drop FPS). When a
    // capture backend grows a `set_target_fps`, this is where it
    // gets wired.
    let _ = decision.target_fps;
    if decision.target_kbps != abr.last_applied_kbps {
        match slot.encoder.set_bitrate_kbps(decision.target_kbps) {
            Ok(()) => {
                info!(
                    from_kbps = abr.last_applied_kbps,
                    to_kbps = decision.target_kbps,
                    rtt_ms = sample.rtt.as_millis() as u64,
                    congestion_events_delta = sample.congestion_events_delta,
                    lost_packets_delta = sample.lost_packets_delta,
                    client_fragments_lost = sample.client_fragments_lost,
                    client_frames_dropped = sample.client_frames_dropped,
                    "abr: bitrate retuned"
                );
                abr.last_applied_kbps = decision.target_kbps;
            }
            Err(e) => {
                // The controller already advanced its internal gear
                // (cooldown reset, current = target). Sync our
                // tracking variable to that so the controller and
                // the integrator agree on "current" — otherwise the
                // next loop tries the *same* failing target again
                // each iteration while the controller has already
                // moved on. The encoder stays at whatever bitrate it
                // last accepted, which is the safest fallback.
                warn!(error = %e, target_kbps = decision.target_kbps, "abr: set_bitrate_kbps failed; reconciling tracking with controller state");
                abr.last_applied_kbps = decision.target_kbps;
            }
        }
    }
}

fn run_capture_and_send(
    conn: Arc<Connection>,
    frames: Receiver<CapturedFrame>,
    force_idr: tether_session::IdrSignal,
    display_dims_tx: tokio::sync::watch::Sender<Option<(u32, u32)>>,
    shutdown: Arc<AtomicBool>,
    chosen_profile: VideoProfile,
    stream_ready: Arc<AtomicBool>,
    runtime: tokio::runtime::Handle,
    keyframe_conn: Arc<Connection>,
    latest_client_stats: LatestClientStats,
    latest_viewport: LatestViewport,
) {
    let mut fragmenter = FrameFragmenter::new(0);
    let mut stats = tether_session::EncodeStatsWindow::new(std::time::Duration::from_secs(2));
    let mut slot: Option<EncoderSlot> = None;
    let mut pts: i64 = 0;
    // Frame-change classifier. CPU frames get a strided hash; GPU
    // frames bypass it (zero-copy path mustn't read back). Resolution
    // changes are caught by the fingerprint's (w, h, format) tuple,
    // so we don't need to reset this alongside the slot rebuild.
    let mut damage: Box<dyn DamageSignal> = Box::new(HashDamage::new());

    // Poll the capture channel on a short tick so a quiet desktop
    // (PipeWire stops delivering frames when nothing changes on screen)
    // doesn't trap us in a blocking recv past the point where the
    // disconnect path wants us to exit. The 100 ms wake-up adds zero
    // latency to actual frame delivery — `recv_timeout` returns as
    // soon as a frame lands — and ~10 wake-ups/sec of idle CPU is
    // negligible next to the encoder's own load.
    loop {
        if shutdown.load(Ordering::Relaxed) {
            info!("send-thread shutdown signalled; exiting");
            break;
        }
        let frame = match frames.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(f) => f,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };
        // Drop frames captured before the client said it was ready to
        // decode. This is the StreamReady gate — without it, frames
        // from the first ~100-500 ms race the client's decoder
        // construction. We drop rather than buffer because (a)
        // captured frames go stale fast, and (b) the encoder isn't
        // built yet on this thread either, so there's nothing to feed.
        if !stream_ready.load(Ordering::Acquire) {
            continue;
        }
        let frame_width = frame.width();
        let frame_height = frame.height();
        let mut timing = {
            let (k, u) = frame.timestamps();
            HostFrameTimingBuilder::captured(k, u)
        };

        // Reject CPU frames in non-BGRA formats up front (no encoder
        // path consumes them today). GPU frames already passed format
        // gating at PipeWire negotiation time — see
        // spa_format_to_drm_fourcc in tether-capture.
        if let CapturedFrame::Cpu(ref cpu) = frame {
            if cpu.format != PixelFormat::Bgra8 {
                warn!(?cpu.format, "h264 encoder only accepts BGRA; skipping frame");
                continue;
            }
        }

        // Damage-skip gate. If the classifier says the frame is
        // bit-identical to the previous one AND nobody is waiting on
        // a forced IDR, drop it on the floor — the client's
        // renderer is already showing this image. The IdrSignal peek
        // here is non-consuming; the take() further down still owns
        // the clear. Forced IDRs win over the damage skip
        // unconditionally: a client mid-reconnect requests an IDR to
        // bootstrap, and swallowing it would deadlock the session.
        if matches!(damage.classify(&frame), DamageHint::Unchanged) && !force_idr.peek() {
            continue;
        }

        // Encoder is lazily created on the first frame (we don't know
        // capture resolution at startup) and recreated whenever the
        // capture source changes resolution mid-stream (Linux portal
        // output switch, future multi-monitor handoff) OR the client
        // requests a viewport-driven resize. Bumping the fragmenter
        // epoch makes the receiver discard any pre-resize fragments
        // still in flight instead of fusing them with the first
        // post-resize keyframe — that's exactly what
        // `VideoPacket::stream_epoch` exists for.
        let (current_viewport, current_viewport_seq) = {
            let g = latest_viewport.lock().unwrap();
            (g.viewport, g.seq)
        };
        // Viewport-driven downscale applies per frame source:
        //   - CPU frames: `resize_bgra_bilinear` does the work below.
        //   - Linux GPU dma-buf: `tether-scaler` (Mitchell-Netravali
        //     in linear-light) runs inside `encode_gpu_frame` between
        //     PipeWire's BGRA import and the chroma bridge.
        //   - macOS GPU IOSurface: until SCK's `SCStreamConfiguration.
        //     width/height` is plumbed (phase-2), the IOSurface
        //     arrives at capture dims; the encoder + VideoToolbox
        //     session expect frames at their configured dims, so we
        //     hold `encode_dims = capture_dims` here. Removing this
        //     branch without the SCK plumbing would feed mis-sized
        //     pixel buffers to VT and produce encoder rejection or
        //     silently wrong output.
        let viewport_applies = !matches!(frame, CapturedFrame::Gpu(_)) || cfg!(target_os = "linux");
        let (encode_width, encode_height) = if viewport_applies {
            encode_dims_for_viewport(frame_width, frame_height, current_viewport)
        } else {
            (frame_width, frame_height)
        };
        // viewport_seq isn't part of the rebuild check: only an actual
        // change in encoder dimensions warrants tearing down the
        // encoder + bumping stream_epoch. A viewport change that
        // leaves dims unchanged (GPU path while no scaler exists; an
        // upscale-rejected viewport on any path) shouldn't churn the
        // session. The seq is tracked for diagnostic logging only.
        let needs_recreate = slot.as_ref().is_none_or(|s| {
            s.capture_width != frame_width
                || s.capture_height != frame_height
                || s.width != encode_width
                || s.height != encode_height
        });
        if needs_recreate {
            if let Some(old) = slot.as_ref() {
                info!(
                    old_capture = old.capture_width,
                    new_capture = frame_width,
                    old_encode_w = old.width,
                    new_encode_w = encode_width,
                    old_encode_h = old.height,
                    new_encode_h = encode_height,
                    viewport_seq = current_viewport_seq,
                    "dimensions changed; recreating encoder, bumping stream epoch"
                );
                fragmenter.bump_epoch();
            }
            let _ = display_dims_tx.send(Some((frame_width, frame_height)));
            // Single-element preference list: the handshake already
            // picked one codec, and a mid-session codec switch would
            // require coordinated client decoder rebuild. We pass the
            // list-form for API symmetry with the initial handshake
            // probe; per-resize cost is one construction attempt.
            let baseline_kbps =
                derive_bitrate_kbps(chosen_profile, encode_width, encode_height, ENCODER_FPS);
            slot = match build_encoder(
                chosen_profile,
                encode_width,
                encode_height,
                ENCODER_FPS,
                baseline_kbps,
            ) {
                Ok((_profile, e)) => {
                    info!(
                        backend = e.name(),
                        hardware = e.is_hardware(),
                        codec = ?chosen_profile.codec,
                        chroma = ?chosen_profile.chroma,
                        bit_depth = chosen_profile.bit_depth,
                        capture_width = frame_width,
                        capture_height = frame_height,
                        encode_width,
                        encode_height,
                        fps = ENCODER_FPS,
                        baseline_kbps,
                        abr = e.supports_changing_bitrate(),
                        "encoder initialised"
                    );
                    let abr = e.supports_changing_bitrate().then(|| AbrState {
                        controller: AbrController::new(AbrConfig::new(
                            baseline_kbps,
                            ENCODER_FPS,
                        )),
                        last_quinn: conn.quinn_stats(),
                        last_observed_at: Instant::now(),
                        last_applied_kbps: baseline_kbps,
                    });
                    // Diagnostic: if the controller's floor equals the
                    // baseline (true for very small encode sizes —
                    // 320x240 test pattern, or an aggressive
                    // viewport) the bitrate gear has zero working
                    // range. Surface it once at encoder init so a
                    // future "ABR isn't doing anything" investigation
                    // finds the cause in the logs.
                    if abr.is_some()
                        && AbrConfig::new(baseline_kbps, ENCODER_FPS).floor_kbps == baseline_kbps
                    {
                        info!(
                            baseline_kbps,
                            "ABR enabled but floor == baseline; bitrate control range is zero \
                             (encode dims too small for the configured floor)"
                        );
                    }
                    Some(EncoderSlot {
                        encoder: e,
                        capture_width: frame_width,
                        capture_height: frame_height,
                        width: encode_width,
                        height: encode_height,
                        viewport_seq: current_viewport_seq,
                        abr,
                        #[cfg(target_os = "linux")]
                        bridge: BridgeState::NotYetBuilt,
                        #[cfg(target_os = "linux")]
                        scaler: None,
                    })
                }
                Err(e) => {
                    // The probe approved this profile at handshake on a
                    // 128×128 scratch surface, but real dims rejected.
                    // Common Main444 trip-wire: hardware accepts the
                    // probe size but rejects 16-pixel-misaligned real
                    // dimensions. Surface this to the client as a clean
                    // Goodbye(InternalError) rather than a silent black
                    // window — the client otherwise sits forever
                    // waiting for video that won't arrive.
                    warn!(
                        error = %e,
                        codec = ?chosen_profile.codec,
                        chroma = ?chosen_profile.chroma,
                        encode_width,
                        encode_height,
                        "encoder init failed; sending Goodbye(InternalError) and exiting send loop"
                    );
                    let goodbye_conn = conn.clone();
                    let reason = format!(
                        "host could not construct {:?} {:?} encoder for {}x{}: {}",
                        chosen_profile.codec, chosen_profile.chroma, encode_width, encode_height, e
                    );
                    let _ = runtime.block_on(goodbye_conn.send_control(&ControlMessage::Goodbye {
                        reason,
                        code: tether_protocol::control::GoodbyeCode::InternalError,
                    }));
                    return;
                }
            };
        }
        let slot_mut = slot.as_mut().expect("slot populated above");

        // ABR tick. Only fires when the client has reported a fresh
        // window; ClientStats arrives at ~1 Hz, the encode loop runs at
        // ~60 Hz, so most iterations bypass this entirely. Quinn's
        // cumulative counters are read here so the controller sees a
        // snapshot that's coherent with the client's window.
        if let Some(stats) = latest_client_stats.lock().unwrap().take() {
            tick_abr(slot_mut, &conn, stats);
        }

        // Swap-and-zero: at most one forced keyframe per request, even
        // if multiple ForceIdr messages arrive between encode calls.
        let force_kf = force_idr.take();
        timing.encode_submit();
        let encoded = match frame {
            CapturedFrame::Cpu(ref cpu) => {
                // If the encoder is sized below the captured frame
                // (viewport-driven downscale), bilinear-resize the
                // BGRA into the encoder's expected dimensions before
                // handing it off. Capture-size == encode-size avoids
                // the allocation + copy entirely.
                let result = if slot_mut.width == cpu.width && slot_mut.height == cpu.height {
                    slot_mut.encoder.encode_bgra(&cpu.data, pts, force_kf)
                } else {
                    let scaled = resize_bgra_bilinear(
                        &cpu.data,
                        cpu.width,
                        cpu.height,
                        slot_mut.width,
                        slot_mut.height,
                    );
                    slot_mut.encoder.encode_bgra(&scaled, pts, force_kf)
                };
                match result {
                    Ok(e) => e,
                    Err(e) => {
                        warn!(error = %e, "encode failed; dropping frame");
                        continue;
                    }
                }
            }
            #[cfg(target_os = "linux")]
            CapturedFrame::Gpu(gpu) => match encode_gpu_frame(
                slot_mut,
                chosen_profile.chroma,
                chosen_profile.bit_depth,
                gpu,
                pts,
                force_kf,
            ) {
                GpuEncodeOutcome::Packets(p) => p,
                GpuEncodeOutcome::DropFrame(e) => {
                    warn!(error = %e, "GPU encode failed; dropping frame");
                    continue;
                }
                GpuEncodeOutcome::Fatal(e) => {
                    tracing::error!(
                        error = %e,
                        "GPU encode bridge collapsed; sending Goodbye(InternalError) and exiting send loop"
                    );
                    let goodbye_conn = conn.clone();
                    let reason = format!("host GPU encode bridge collapsed: {e}");
                    let _ = runtime.block_on(goodbye_conn.send_control(&ControlMessage::Goodbye {
                        reason,
                        code: tether_protocol::control::GoodbyeCode::InternalError,
                    }));
                    return;
                }
            },
            #[cfg(target_os = "macos")]
            CapturedFrame::Gpu(gpu) => match encode_iosurface_frame(slot_mut, gpu, pts, force_kf) {
                Ok(p) => p,
                Err(e) => {
                    warn!(error = %e, "IOSurface encode failed; dropping frame");
                    continue;
                }
            },
            #[cfg(not(any(target_os = "linux", target_os = "macos")))]
            CapturedFrame::Gpu(_) => {
                warn!("Gpu CapturedFrame on an unsupported build; dropping");
                continue;
            }
        };
        timing.encode_done();
        let encode_delta_ns = timing.encode_delta_ns();
        pts += 1;

        // Concatenate all packets the encoder spat out for this input
        // frame into one wire payload. With tune=zerolatency this is
        // almost always exactly one packet, so the common case moves
        // the encoder's `Bytes` straight through — no copy. Multi-
        // packet frames (rare; happens when the encoder emits SPS/PPS
        // + IDR as separate AVPackets) coalesce into a single
        // pre-sized `BytesMut`.
        let mut keyframe = false;
        let body: tether_codec::bytes::Bytes = match encoded.len() {
            0 => continue,
            1 => {
                let pkt = encoded.into_iter().next().expect("len == 1");
                keyframe = pkt.keyframe;
                pkt.data
            }
            _ => {
                let total_len: usize = encoded.iter().map(|p| p.data.len()).sum();
                let mut buf = tether_codec::bytes::BytesMut::with_capacity(total_len);
                for pkt in encoded {
                    if pkt.keyframe {
                        keyframe = true;
                    }
                    buf.extend_from_slice(&pkt.data);
                }
                buf.freeze()
            }
        };
        if body.is_empty() {
            continue;
        }
        stats.record_frame(encode_delta_ns, body.len() as u64, keyframe);

        let meta = VideoFrameMeta {
            timing: timing.finish(),
            keyframe,
            input_echo: InputEchoBatch::default(),
            dimensions: (frame_width, frame_height),
        };

        if keyframe {
            // Keyframes ride a reliable per-IDR QUIC uni stream. The
            // single_packet path doesn't chunk into datagram-sized
            // pieces because the stream layer handles segmentation;
            // the receiver's reassembler sees a fragment_count=1
            // packet that completes immediately. Frame_seq still
            // advances so the next P-frame fragments slot in
            // sequentially.
            //
            // Synchronous block_on (rather than an mpsc to a separate
            // task) keeps strict ordering between the IDR and the
            // P-frames that follow — see the comment on the spawn
            // site for the epoch-race rationale.
            let packet = fragmenter.single_packet(meta, body);
            if let Err(e) = runtime.block_on(keyframe_conn.send_video_keyframe(&packet)) {
                warn!(error = ?e, "send_video_keyframe failed, terminating send loop");
                return;
            }
        } else {
            let packets = fragmenter.fragment(meta, body);
            for packet in packets {
                if let Err(e) = conn.send_datagram(&Datagram::Video(packet)) {
                    warn!(error = ?e, "send_datagram failed, terminating send loop");
                    return;
                }
            }
        }

        if stats.should_emit() {
            if let Some(snap) = stats.snapshot_and_reset() {
                let kf_per_s = if snap.window_secs > 0.0 {
                    f64::from(snap.keyframe_count) / snap.window_secs
                } else {
                    0.0
                };
                info!(
                    frames = snap.frame_count,
                    avg_encode_ms = format!("{:.2}", snap.avg_encode_ms),
                    kbps_out = format!("{:.0}", snap.kbps_out),
                    kf_per_s = format!("{kf_per_s:.2}"),
                    "send stats"
                );
            }
        }
    }
    info!("send loop exiting");
}

/// Directory the host caches its self-signed cert + key in. Default
/// is `$HOME/.tether/`; override with `$TETHER_CERT_DIR` for testing
/// or sharing between host instances. We deliberately don't follow
/// XDG paths — the file pair is small and operationally important,
/// and a single well-known location ("look under ~/.tether") is
/// easier to talk about in docs than "wherever XDG_DATA_HOME points".
fn persistent_cert_dir() -> anyhow::Result<PathBuf> {
    if let Some(dir) = std::env::var_os("TETHER_CERT_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let home = std::env::var_os("HOME").ok_or_else(|| {
        anyhow::anyhow!("neither $TETHER_CERT_DIR nor $HOME is set; can't choose a cert directory")
    })?;
    Ok(PathBuf::from(home).join(".tether"))
}

fn parse_args() -> anyhow::Result<(SocketAddr, bool)> {
    let mut bind: SocketAddr = "127.0.0.1:7654".parse().expect("static literal");
    let mut use_test_pattern = false;
    for arg in std::env::args().skip(1) {
        if arg == "--test-pattern" {
            use_test_pattern = true;
        } else if arg == "--help" || arg == "-h" {
            eprintln!("usage: tether-host [--test-pattern] [bind_addr]");
            std::process::exit(0);
        } else {
            bind = arg.parse()?;
        }
    }
    Ok((bind, use_test_pattern))
}

async fn pick_capture_source(
    force_test_pattern: bool,
    chosen_profile: VideoProfile,
) -> anyhow::Result<Receiver<CapturedFrame>> {
    if force_test_pattern {
        info!(
            width = TEST_PATTERN_WIDTH,
            height = TEST_PATTERN_HEIGHT,
            fps = TEST_PATTERN_FPS,
            "capture source: test-pattern (forced)"
        );
        return Ok(tether_capture::test_pattern::start(
            TEST_PATTERN_WIDTH,
            TEST_PATTERN_HEIGHT,
            TEST_PATTERN_FPS,
        ));
    }
    real_capture(chosen_profile).await
}

#[cfg(target_os = "linux")]
async fn real_capture(_chosen_profile: VideoProfile) -> anyhow::Result<Receiver<CapturedFrame>> {
    info!("capture source: linux (PipeWire + xdg-desktop-portal)");
    // Query which DRM modifiers our wgpu/Vulkan importer can consume for
    // the BGRA-family capture formats PipeWire emits. PipeWire then
    // negotiates a DMA-BUF format with the compositor restricted to the
    // intersection of (compositor-supported, this list). Empty result —
    // or any query failure — drops us to SHM-only, which is the
    // intended fallback path; the goal is "never advertise a modifier we
    // can't actually import."
    //
    // AR24 (DRM_FORMAT_ARGB8888) and XR24 (DRM_FORMAT_XRGB8888) both map
    // to vk::Format::B8G8R8A8_UNORM on the importer side, so the
    // modifier sets are identical; querying once suffices.
    let modifiers =
        match tether_gpuconvert::importable_dmabuf_modifiers(u32::from_le_bytes(*b"AR24")).await {
            Ok(m) if !m.is_empty() => {
                info!(
                    count = m.len(),
                    "advertised DMA-BUF modifiers to compositor"
                );
                m
            }
            Ok(_) => {
                warn!("GPU importer reports zero DRM modifiers; DMA-BUF disabled, SHM only");
                Vec::new()
            }
            Err(e) => {
                warn!(error = %e, "modifier query failed; DMA-BUF disabled, SHM only");
                Vec::new()
            }
        };
    tether_capture::linux::start(modifiers)
        .await
        .map_err(anyhow::Error::from)
}

#[cfg(target_os = "macos")]
async fn real_capture(chosen_profile: VideoProfile) -> anyhow::Result<Receiver<CapturedFrame>> {
    // Pick the SCK pixel format that matches the negotiated encoder
    // profile. The encoder's `submit_iosurface` cross-checks the
    // delivered IOSurface fourcc against this; a mismatch would refuse
    // the zero-copy fast path. SCK's `start_capture` triggers the
    // macOS ScreenRecording TCC prompt on first run; subsequent runs
    // reuse the granted permission.
    let pixel_format = match tether_capture::macos::sck_pixel_format_for_profile(chosen_profile) {
        tether_capture::macos::SckCapabilityCheck::Supported(p) => p,
        tether_capture::macos::SckCapabilityCheck::Unsupported => {
            anyhow::bail!(
                "no SCK pixel format models the negotiated profile {:?} — the \
                 capture-bridge filter should have prevented this profile from \
                 reaching negotiation",
                chosen_profile
            );
        }
    };
    tether_capture::macos::start(pixel_format)
        .await
        .map_err(anyhow::Error::from)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
async fn real_capture(_chosen_profile: VideoProfile) -> anyhow::Result<Receiver<CapturedFrame>> {
    warn!("no real capture backend on this platform yet; falling back to test-pattern");
    Ok(tether_capture::test_pattern::start(
        TEST_PATTERN_WIDTH,
        TEST_PATTERN_HEIGHT,
        TEST_PATTERN_FPS,
    ))
}

fn init_tracing() -> tracing_appender::non_blocking::WorkerGuard {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let (writer, guard) = tracing_appender::non_blocking(std::io::stdout());
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(writer)
        .init();
    guard
}

/// Default VBR target bitrate as a function of resolution, fps,
/// codec, and chroma. Anchored at 1080p60 H.264 4:2:0 = 8 Mbps (the
/// [`ENCODER_BITRATE_KBPS`] floor); scales linearly with `pixels × fps`;
/// HEVC gets a 0.7× codec multiplier (conservative ~30% efficiency
/// gain over H.264 at the same visual quality); 4:4:4 gets a 1.4×
/// chroma multiplier on top because the encoder is now carrying 3×
/// the chroma samples (vs. 4:2:0's 1×) and rate-control only absorbs
/// some of the cost — without the bump, 4:4:4 sessions ship blocky
/// chroma in the same budget that was sized for subsampled video.
/// Clamped to a sane band so a tiny test pattern doesn't get a
/// starvation-tier bitrate and a huge display doesn't blow the LAN.

fn derive_bitrate_kbps(profile: VideoProfile, width: u32, height: u32, fps: u32) -> u32 {
    const REFERENCE_PIXELS: u64 = 1920 * 1080;
    const REFERENCE_FPS: u64 = 60;
    const REFERENCE_KBPS_H264: u64 = ENCODER_BITRATE_KBPS as u64;

    let pixels = u64::from(width).saturating_mul(u64::from(height));
    let scaled = REFERENCE_KBPS_H264
        .saturating_mul(pixels)
        .saturating_mul(u64::from(fps))
        / (REFERENCE_PIXELS * REFERENCE_FPS).max(1);

    let codec_scaled = match profile.codec {
        CodecKind::H264 => scaled,
        // HEVC: 70% of H.264 for similar visual quality.
        CodecKind::Hevc => scaled * 7 / 10,
        // AV1: not yet supported; if it gets here the encoder build
        // will fail anyway. 60% is the standard reference number.
        CodecKind::Av1 => scaled * 6 / 10,
    };

    // 4:4:4 has 3× the chroma samples of 4:2:0 (full-res U + V vs.
    // half-res interleaved UV). Empirically HEVC Main444 needs
    // ~1.3–1.5× the 4:2:0 bitrate to maintain quality parity on
    // mixed content; less penalty (closer to 1.1×) on pure UI where
    // chroma is mostly piecewise-constant. 1.4× is the conservative
    // middle of that band.
    let chroma_scaled = match profile.chroma {
        ChromaSubsampling::Yuv420 => codec_scaled,
        ChromaSubsampling::Yuv444 => codec_scaled * 14 / 10,
    };

    // 10-bit precision: extra 2 bits per sample means ~25% more entropy
    // per pixel before motion-compensated prediction and entropy coding
    // recover most of it. ~1.2× is the standard reference multiplier
    // for 10-bit at the same perceptual quality as 8-bit at the same
    // chroma. Without this term, 10-bit sessions ship at the 8-bit
    // budget and visibly lose precision on gradients — defeating the
    // reason a host advertised the 10-bit profile in the first place.
    let depth_scaled = match profile.bit_depth {
        10 => chroma_scaled * 12 / 10,
        // 8 and any future depth fall back to no multiplier; the probe
        // layer is the authority on which depths actually ship.
        _ => chroma_scaled,
    };

    depth_scaled.clamp(500, 30_000) as u32
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    //! Regression tests for `handle_client`'s lifecycle invariants. These
    //! aren't integration tests against the real Connection/libei stack —
    //! they enforce the *pattern* the production code uses, so a future
    //! refactor that breaks the pattern (and re-introduces the bug it
    //! exists to prevent) fails CI instead of failing on a user's desk.
    //!
    //! The full subprocess-level integration test is tracked separately;
    //! see the project task list.
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::task::JoinSet;

    #[test]
    fn encode_dims_no_viewport_passes_capture_dims_through() {
        // No viewport → capture dims pass through unmodified. H.264 /
        // HEVC SPS carries crop signalling for non-16-aligned heights
        // like 1080; flooring would silently lose 8 rows. The
        // alignment floor only kicks in when a viewport budget
        // requires it.
        assert_eq!(encode_dims_for_viewport(1920, 1080, None), (1920, 1080));
        assert_eq!(encode_dims_for_viewport(1366, 768, None), (1366, 768));
    }

    #[test]
    fn encode_dims_viewport_smaller_letterboxes() {
        // 3840x2160 captured, client viewport 1280x720. Aspect
        // matches (16:9), so we expect 1280x720 exactly (both
        // 16-aligned).
        assert_eq!(
            encode_dims_for_viewport(3840, 2160, Some(Viewport::new(1280, 720))),
            (1280, 720)
        );
    }

    #[test]
    fn encode_dims_viewport_aspect_mismatch_letterboxes_at_smaller_axis() {
        // 1920x1080 (16:9) captured, client window 1280x1024 (5:4).
        // We must fit inside 1280x1024 without stretching: the height
        // ratio (1024/1080 = 0.948) is smaller than the width ratio
        // (1280/1920 = 0.667), so width is the binding edge: 1920 *
        // 0.667 = 1280, height = 1080 * 0.667 = 720. Client letterboxes
        // the 1280x720 inside its 1280x1024 window.
        assert_eq!(
            encode_dims_for_viewport(1920, 1080, Some(Viewport::new(1280, 1024))),
            (1280, 720)
        );
    }

    #[test]
    fn encode_dims_viewport_larger_does_not_upscale() {
        // Client window is bigger than the capture. We don't pay
        // for upscaling — encoder stays at capture dims, client
        // upscales locally with its renderer.
        assert_eq!(
            encode_dims_for_viewport(1280, 720, Some(Viewport::new(3840, 2160))),
            (1280, 720)
        );
    }

    #[test]
    fn encode_dims_invalid_viewport_falls_back() {
        // A peer that sends (0, 0) is explicitly opting out. Same
        // behaviour as `None`: capture dims passed through, no
        // alignment floor.
        assert_eq!(
            encode_dims_for_viewport(1920, 1080, Some(Viewport::new(0, 720))),
            (1920, 1080)
        );
    }

    #[test]
    fn encode_dims_rounds_down_to_alignment() {
        // Off-aspect viewport that lands the output on non-16-aligned
        // dims. We must round DOWN — the encoder rejects unaligned
        // input on hardware Main444 backends, and overshooting would
        // mean we ship more pixels than the client asked for.
        let (w, h) = encode_dims_for_viewport(1920, 1080, Some(Viewport::new(1000, 600)));
        assert_eq!(w % 16, 0);
        assert_eq!(h % 16, 0);
        assert!(w <= 1000 && h <= 600);
    }

    #[test]
    fn resize_bgra_identity_for_same_dims() {
        // Same in/out dims should be a faithful bitwise copy. We
        // route around this case in the encode branch to avoid the
        // allocation, but the function itself must still be correct
        // when called.
        let src = vec![0x11u8, 0x22, 0x33, 0xFF, 0x44, 0x55, 0x66, 0xFF];
        let out = resize_bgra_bilinear(&src, 2, 1, 2, 1);
        assert_eq!(out, src);
    }

    #[test]
    fn resize_bgra_downscale_preserves_solid_color() {
        // A solid BGRA fill must come back as the same color at
        // every output pixel — bilinear must not introduce drift on
        // a constant image.
        let src = vec![0x80u8; 64 * 64 * 4];
        let out = resize_bgra_bilinear(&src, 64, 64, 32, 32);
        assert!(out.iter().all(|&v| v == 0x80));
    }

    /// Increments a shared counter once, when its sole owner drops it.
    /// Stand-in for the `LibeiInjector` (which we can't construct in a
    /// unit test — it needs a real portal session) while exercising the
    /// same Arc-refcount lifecycle the production injector relies on.
    struct DropCounter(Arc<AtomicUsize>);
    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// `handle_client` deliberately hands the *last* injector Arc to
    /// the dgram task by move rather than clone. The reason is the bug
    /// this test exists to prevent: if a fourth Arc lives in the
    /// `handle_client` stack frame, every spawned task can exit and the
    /// injector's refcount still won't hit zero — the libei session
    /// stays grabbed, and the host's mouse and keyboard stay frozen
    /// until the process itself exits. This was a real shipped bug
    /// (mouse lockup after client disconnect, May 2026).
    ///
    /// The assertion: after the analogous three-spawn pattern and a
    /// `JoinSet::shutdown`, the `DropCounter` standing in for the
    /// injector must have dropped exactly once. If a future edit
    /// regresses the pattern (e.g. changes the third spawn from
    /// `let inj = injector;` to `let inj = injector.clone();`), this
    /// test fails with `drops == 0`.
    #[tokio::test]
    async fn injector_drops_after_recv_tasks_shut_down() {
        let drops = Arc::new(AtomicUsize::new(0));

        let injector = Arc::new(DropCounter(drops.clone()));
        let mut tasks: JoinSet<()> = JoinSet::new();

        // Mirror the production spawn pattern: two clones, then a
        // move for the last one so no Arc survives in this scope.
        {
            let inj = injector.clone();
            tasks.spawn(async move {
                let _hold = inj;
                std::future::pending::<()>().await
            });
        }
        {
            let inj = injector.clone();
            tasks.spawn(async move {
                let _hold = inj;
                std::future::pending::<()>().await
            });
        }
        {
            // The load-bearing move. Replace with `.clone()` to
            // reproduce the bug this test guards against — the
            // assertion below will fail with `drops == 0` because
            // the outer `injector` will still hold the last ref
            // when we check.
            let inj = injector;
            tasks.spawn(async move {
                let _hold = inj;
                std::future::pending::<()>().await
            });
        }

        tasks.shutdown().await;

        // The check has to happen *before* this function returns —
        // otherwise even buggy code (extra outer Arc) would pass,
        // because the stray Arc would drop at function exit and the
        // counter would still read 1 at that point. We need to catch
        // "still alive after shutdown" while we're still inside the
        // scope where the stray ref would have lived.
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "expected injector to drop after tasks.shutdown(); drops={} means a stray Arc clone survived past the spawned tasks (libei lockup bug)",
            drops.load(Ordering::SeqCst)
        );
    }

    /// **Cross-table fourcc consistency** for the macOS IOSurface
    /// path. Four crates each carry a `(chroma, bit_depth) → fourcc`
    /// table:
    ///
    ///   * `tether_capture::macos::sck_pixel_format_for_profile` — the
    ///     fourcc SCK will deliver for a chosen `VideoProfile`.
    ///   * `tether_codec::videotoolbox::encoder::iosurface_fourcc_matches`
    ///     — the fourccs the VT encoder accepts as zero-copy input.
    ///   * `tether_codec::videotoolbox::probe::expected_iosurface_fourccs`
    ///     — the fourccs the VT decoder is allowed to emit for a
    ///     given profile (= the probe's accept set on the decode
    ///     side of the round-trip).
    ///   * `tether_render::accepts_iosurface_fourcc` — the fourccs the
    ///     renderer's IOSurface import path accepts.
    ///
    /// The full pipeline is `SCK → encoder → decoder → renderer`. For
    /// each profile we negotiate, the per-link invariants are:
    ///
    ///   * **SCK output ⊆ encoder accept** — the encoder must accept
    ///     whatever the capture layer delivers. A miss here crashes
    ///     the first frame after handshake.
    ///   * **Decoder output (probe expected) ⊆ renderer accept** —
    ///     the renderer must accept anything the VT decoder might
    ///     emit. A miss here silently drops every frame after
    ///     decode, exactly the shape of bug `621badc` (renderer
    ///     rejected `'x420'` for HEVC Main10).
    ///
    /// This is a unit test on purpose. The bug shipped to a real
    /// session because no probe or unit test compared the four
    /// tables, and the existing `videotoolbox_round_trip_chroma_matrix`
    /// stops at the decoder's IOSurface — the renderer is on a
    /// parallel track. Catching drift in default CI is cheaper than
    /// catching it on a user's desk.
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_iosurface_fourcc_tables_agree_across_crates() {
        use tether_capture::macos::sck_pixel_format_for_profile;
        use tether_codec::videotoolbox::encoder::iosurface_fourcc_matches;
        use tether_codec::videotoolbox::expected_iosurface_fourccs;
        use tether_probe::PROFILE_PREFERENCE;
        use tether_render::accepts_iosurface_fourcc;

        // Walk every profile the negotiator could pick.
        for profile in PROFILE_PREFERENCE {
            let chroma = profile.chroma;
            let bd = profile.bit_depth;

            // (1) SCK output ⊆ encoder accept.
            // If SCK has a mapping for this profile, whatever it
            // would deliver must be in the encoder's accept set.
            if let Some(fourcc) = sck_pixel_format_for_profile(*profile).fourcc() {
                assert!(
                    iosurface_fourcc_matches(chroma, bd, fourcc),
                    "SCK delivers 0x{fourcc:08x} for profile {profile:?} but the VT encoder \
                     does not accept it via submit_iosurface. This would crash at first frame."
                );
            }

            // (2) Probe expected ⊆ renderer accept.
            // Every fourcc the VT decoder might emit for a confirmed
            // round-trip of this profile must be in the renderer's
            // accept set. Bug 621badc was this invariant broken for
            // (Yuv420, 10) — probe expected `'x420'` but renderer
            // only accepted `'P010'`, so the renderer rejected every
            // frame of the first live Main10 session.
            for &fourcc in expected_iosurface_fourccs(*profile) {
                assert!(
                    accepts_iosurface_fourcc(chroma, bd, fourcc),
                    "VT probe expects the decoder may emit 0x{fourcc:08x} for profile \
                     {profile:?} but the renderer's IOSurface import rejects it. \
                     Renderer would drop every frame of a negotiated session."
                );
            }
        }
    }

    /// Companion to `macos_iosurface_fourcc_tables_agree_across_crates`:
    /// for profiles that are *not* in PROFILE_PREFERENCE today (e.g.
    /// 4:2:2 — checked-in fixtures exist but no wire support yet),
    /// nothing should accept anything. Guards against a half-wired
    /// future profile that's accepted by one table but unhandled by
    /// the others.
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_iosurface_fourcc_tables_reject_unmodeled_profiles() {
        use tether_codec::videotoolbox::encoder::iosurface_fourcc_matches;
        use tether_codec::videotoolbox::probe::expected_iosurface_fourccs;
        use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};
        use tether_render::accepts_iosurface_fourcc;

        // 12-bit isn't in the model; nor is 4:2:2.
        let bogus = VideoProfile {
            codec: CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 12,
        };
        assert!(
            expected_iosurface_fourccs(bogus).is_empty(),
            "probe should not expect any fourcc for a 12-bit profile"
        );
        // Every plausible fourcc must be rejected by encoder + renderer.
        for label in ["420v", "x420", "xf20", "P010", "444v", "xf44", "P410"] {
            let bytes: [u8; 4] = label.as_bytes().try_into().unwrap();
            let fourcc = u32::from_be_bytes(bytes);
            assert!(
                !iosurface_fourcc_matches(bogus.chroma, bogus.bit_depth, fourcc),
                "encoder accepted fourcc {label} for unmodeled 12-bit profile"
            );
            assert!(
                !accepts_iosurface_fourcc(bogus.chroma, bogus.bit_depth, fourcc),
                "renderer accepted fourcc {label} for unmodeled 12-bit profile"
            );
        }
    }
}
