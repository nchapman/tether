//! Tether host — captures the local display and streams it to a client.
//!
//! v0: raw BGRA frames fragmented over QUIC datagrams, single client per
//! host, no codec. Real capture (PipeWire/portal on Linux, ScreenCaptureKit
//! on macOS) is the default; pass `--test-pattern` to fall back to the
//! synthetic gradient generator (useful for headless dev or as a fallback
//! when the portal isn't available).
//!
//! Usage: `tether-host [--test-pattern] [bind_addr]`
//! (`bind_addr` defaults to `127.0.0.1:7374`).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex, MutexGuard};
use std::time::Instant;

#[cfg(target_os = "linux")]
use tether_capture::GpuCapturedSource;
use tether_capture::{
    CapturedFrame, CursorEvent, CursorSource, DamageHint, DamageSignal, FrameReceiver, HashDamage,
    PixelFormat, PlaceholderCursorSource,
};
#[cfg(not(target_os = "windows"))]
use tether_codec::build_encoder;
#[cfg(target_os = "windows")]
use tether_codec::build_encoder_d3d11;
use tether_codec::Encoder;
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
use tether_codec::GpuEncoderFrame;
#[cfg(target_os = "linux")]
use tether_codec::{DmaBufFrame, DmaBufLayer, DmaBufObject};
#[cfg(target_os = "macos")]
use tether_gpuconvert::nv12_iosurface::{
    BgraIOSurfaceBridge, BridgeError as IOSurfaceBridgeError, PooledIOSurface,
};
#[cfg(target_os = "linux")]
use tether_gpuconvert::{
    Bgra2P010DmaBuf, Bgra2Xv30DmaBuf, Nv12DmaBuf, Nv12DmaBufFrame, P010DmaBufFrame,
    Xv30DmaBufFrame, Yuv444DmaBuf, Yuv444DmaBufFrame, Yuv444pDmaBuf, Yuv444pDmaBufFrame,
};
use tether_ipc::{EngineEvent, Reporter};
use tether_protocol::audio::AudioPacket;
use tether_protocol::control::{
    ChromaSubsampling, CodecKind, ControlMessage, DisplayDescriptor, DisplayModeStatus,
    GoodbyeCode, VideoProfile, VideoStreamId, Viewport,
};
use tether_protocol::video::{
    FrameFragmenter, HostFrameTimingBuilder, InputEchoBatch, VideoFrameMeta, VideoPacket,
};
use tether_protocol::MonoNanos;
#[cfg(target_os = "linux")]
use tether_scaler::{Pipelines as ScalerPipelines, Scaler, ScalerError};
use tether_session::{
    log_peer_session_summary, AbrConfig, AbrController, AbrSample, AcceptError, HostSession,
    HostSessionConfig, SessionSummaryState,
};
use tether_transport::{AbrSnapshot, Connection, Datagram, Server};
use tokio::sync::Mutex as TokioMutex;
use tokio::task::JoinSet;
use tracing::{debug, info, warn};

mod pairing;
use pairing::{ActiveSession, Authorized, PairingState, RefusedReason};

#[cfg(target_os = "linux")]
mod setup_input;

/// Registers the live session in the shared `active` slot on creation and
/// clears it on drop. Using a guard keeps the invariant "the slot is `Some`
/// iff a session is live" on *every* exit path — clean end, revocation, a
/// handshake-failure `continue`, or a ctrl-c / shutdown `break` — without a
/// manual clear on each.
struct ActiveSessionGuard {
    slot: Arc<StdMutex<Option<ActiveSession>>>,
}

impl ActiveSessionGuard {
    fn register(slot: Arc<StdMutex<Option<ActiveSession>>>, session: ActiveSession) -> Self {
        *lock_host_state(&slot, "active session") = Some(session);
        Self { slot }
    }
}

impl Drop for ActiveSessionGuard {
    fn drop(&mut self) {
        // Tolerate a poisoned lock rather than double-panicking on the way out.
        *lock_host_state(&self.slot, "active session") = None;
    }
}

fn lock_host_state<'a, T>(lock: &'a StdMutex<T>, name: &str) -> MutexGuard<'a, T> {
    lock.lock().unwrap_or_else(|poisoned| {
        warn!(
            state = name,
            "host shared-state mutex poisoned; recovering inner state"
        );
        poisoned.into_inner()
    })
}

/// Default target frame rate. Sunshine and Apollo run desktop / game
/// streaming at 60 fps by default; tether matches. The host's encoder
/// time_base, the test-pattern source, and (in the future) the
/// PipeWire format negotiation all use this. Per-frame budget at 60 fps
/// is 16.6 ms; current Intel iGPU encode times sit around 7–8 ms, so
/// there's headroom.
const ENCODER_FPS: u32 = 60;

/// Server-side defense throttle on viewport-driven encoder rebuilds. The
/// client already debounces resize events (~150 ms), so a well-behaved client
/// drives at most one rebuild per settled size. This throttle bounds a client
/// that *doesn't* debounce (buggy or hostile) from forcing a full
/// encoder-teardown + `stream_epoch` bump every frame — which would churn the
/// hardware encode session (fragile on some drivers) and storm the client's
/// decoder rebuilds. Set above the client debounce so a single legitimate
/// resize is never delayed; only sub-throttle bursts coalesce to the latest
/// viewport. Capture-source resolution changes and the first build are NOT
/// throttled — they aren't client-controlled.
const VIEWPORT_REBUILD_THROTTLE: std::time::Duration = std::time::Duration::from_millis(250);

/// VBR target bitrate. The encoder is allowed to overshoot for
/// motion-heavy frames and undershoot on static content. Calibrated
/// for 1080p60 H.264 ≈ 10 Mbps and roughly scales linearly with
/// resolution × fps. HEVC sessions get a 0.7× multiplier inside the
/// derivation step (~30% more efficient at the same visual quality;
/// conservative estimate, refined when we benchmark). For now the
/// constant is the H.264 1080p60 floor; multi-resolution scaling is W8.
const ENCODER_BITRATE_KBPS: u32 = 10_000;

const TEST_PATTERN_WIDTH: u32 = 320;
const TEST_PATTERN_HEIGHT: u32 = 240;
const TEST_PATTERN_FPS: u32 = 60;

fn initial_display_descriptors(use_test_pattern: bool) -> Vec<DisplayDescriptor> {
    if use_test_pattern {
        return vec![tether_capture::test_pattern_display(
            TEST_PATTERN_WIDTH,
            TEST_PATTERN_HEIGHT,
            TEST_PATTERN_FPS.saturating_mul(1000),
        )];
    }

    match tether_capture::display_list() {
        Ok(displays) => displays,
        Err(e) => {
            warn!(
                error = %e,
                "display topology enumeration failed; advertising synthetic primary display"
            );
            vec![tether_capture::test_pattern_display(1280, 720, 60_000)]
        }
    }
}

/// Upper bound on the per-connection authorization exchange (allowlist resume
/// or first-contact pairing). A peer that stalls mid-exchange is dropped so it
/// can't wedge the accept loop indefinitely. The client connects only *after*
/// the user has entered the PIN (the PIN is a launch argument, not typed mid-
/// connection), so the exchange itself runs at machine speed — 30 s is ample
/// headroom for a slow link without inviting a long stall.
///
/// The accept loop authorizes one peer at a time, so a hostile peer that opens
/// a connection and stalls holds *this* slot for the full timeout. If it stalls
/// a first-contact `Pair` after the window was taken (burn-on-attempt), the
/// window is also spent for the duration, so a legitimate device can't pair
/// until the stall clears and the operator re-opens the window. Acceptable for
/// the single-client alpha; concurrent authorization is the planned fix.
const AUTHORIZE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Latest `ClientStats` observation, drained by the encode-and-send
/// thread on each loop iteration. The control recv task writes here
/// every time `ControlMessage::ClientStats` arrives (~1 Hz). A
/// `std::sync::Mutex` is fine because the lock is held for a single
/// option-swap; the send thread isn't a tokio worker so we don't need
/// `tokio::sync::Mutex`.
type LatestClientStats = Arc<StdMutex<Option<ClientStatsObservation>>>;

/// Latest client viewport, with sequence counter so the send thread
/// can tell whether anything changed without diffing the whole value.
/// Set once from `ClientHello::initial_viewport` at session start, then
/// overwritten by `ControlMessage::SetViewportHint` on each resize.
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

impl ViewportState {
    fn update_if_changed(&mut self, next: Option<Viewport>) -> bool {
        if self.viewport == next {
            return false;
        }
        self.viewport = next;
        self.seq = self.seq.wrapping_add(1);
        true
    }
}

/// One window of client-side telemetry. Mirrors the fields in
/// [`ControlMessage::ClientStats`] that the ABR controller actually
/// consumes; `window_ms` and `frames_received` are dropped at the
/// boundary because the controller takes wall-clock dt itself and
/// doesn't need the success-count.
#[derive(Debug, Clone, Copy)]
struct ClientStatsObservation {
    incomplete_frames: u32,
    fragment_loss_events: u32,
}

fn host_summary_audio_active(host_audio_available: bool, client_audio_ready: bool) -> bool {
    host_audio_available && client_audio_ready
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse args first so `--ipc` can route tracing off stdout *before*
    // the subscriber is installed: in IPC mode stdout is reserved for the
    // JSON-lines protocol the shell parses, so logs go to stderr instead.
    let args = parse_args()?;

    // `--setup-input` is a standalone maintenance action: install the
    // udev rule (via pkexec) and exit before any server setup. Linux-only;
    // elsewhere it's a no-op with a note.
    if args.setup_input {
        #[cfg(target_os = "linux")]
        {
            std::process::exit(setup_input::run());
        }
        #[cfg(not(target_os = "linux"))]
        {
            eprintln!("--setup-input is only needed on Linux (uinput); nothing to do.");
            std::process::exit(0);
        }
    }

    let Args {
        bind,
        use_test_pattern,
        ipc,
        audio: audio_enabled,
        ..
    } = args;
    let reporter = Reporter::from_ipc_flag(ipc);
    let forced_video_profile = match tether_probe::forced_video_profile_from_env() {
        Ok(profile) => profile,
        Err(e) => {
            reporter.emit(&EngineEvent::Error { message: e.clone() });
            if reporter.is_json() {
                std::process::exit(1);
            }
            anyhow::bail!(e);
        }
    };

    // Both host (encoder) and client (decoder) call av_log::install(),
    // so FFmpeg messages can land on either side's hot thread. The
    // encoder is quieter in steady state but the same non-blocking
    // rationale applies — a synchronous writer would stall whichever
    // thread libavcodec calls into.
    let _tracing_guard = init_tracing(reporter.is_json());

    // When the shell drives us, watch stdin for commands (or EOF, i.e. the
    // shell died, which trips a shutdown notify that both the accept loop and
    // the in-session select race below). The watcher is spawned once the
    // pairing state exists — it also handles `StartPairing` / `RevokePeer`.
    let shutdown = Arc::new(tokio::sync::Notify::new());

    let cert_dir = persistent_cert_dir()?;
    let server = match Server::bind_persistent(bind, &cert_dir).await {
        Ok(s) => s,
        Err(e) => {
            // Surface bind failures (e.g. the port already in use) to the
            // shell instead of dying silently.
            reporter.emit(&EngineEvent::Error {
                message: format!("failed to bind {bind}: {e}"),
            });
            // In IPC mode the stdin stop-watcher's blocking read would hang
            // the runtime drop, so exit the process directly; the
            // standalone CLI has no such thread and returns for a clean
            // anyhow-formatted error.
            if reporter.is_json() {
                std::process::exit(1);
            }
            return Err(e.into());
        }
    };
    let local = server.local_addr()?;
    let fingerprint = server.fingerprint();
    let fp_hex = hex_encode(&fingerprint);

    // Load the paired-clients allowlist. A corrupt file is fatal (fail closed):
    // silently treating a damaged allowlist as empty would force re-pairing and
    // mask tampering. A missing file is first-run (empty store).
    let paired_path = cert_dir.join("paired_clients.json");
    let paired_store = match tether_pairing::PairedStore::load(&paired_path) {
        Ok(s) => s,
        Err(e) => {
            reporter.emit(&EngineEvent::Error {
                message: format!(
                    "failed to load paired-clients allowlist {}: {e}",
                    paired_path.display()
                ),
            });
            if reporter.is_json() {
                std::process::exit(1);
            }
            return Err(e.into());
        }
    };
    let pairing_state = PairingState {
        paired: Arc::new(StdMutex::new(paired_store)),
        paired_path: Arc::new(paired_path),
        window: Arc::new(StdMutex::new(None)),
        active: Arc::new(StdMutex::new(None)),
        host_fp: fingerprint,
    };

    // Now that the pairing state exists, start the stdin command watcher in
    // IPC mode (Stop / StartPairing / RevokePeer).
    if reporter.is_json() {
        spawn_stdin_command_watcher(shutdown.clone(), pairing_state.clone(), reporter);
    }

    reporter.emit(&EngineEvent::Listening {
        addr: local.to_string(),
        fingerprint: fp_hex.clone(),
    });
    if !reporter.is_json() {
        // Human-only flavor with no IPC-event equivalent.
        println!("cert dir:        {} (rm to rotate)", cert_dir.display());
    }

    // Warm the unified capability probe. `tether_probe::host_supported_profiles`
    // round-trips every `PROFILE_PREFERENCE` entry through the full
    // production chain (capture → bridge → encoder → decoder) and
    // OnceLock-caches the result for process lifetime. This replaces
    // the older per-backend cache warmers with one host-authoritative
    // answer.
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
    // On Windows, pre-create the DXGI capture device BEFORE the codec
    // probe. AMF's probe activity corrupts in-process DXGI output
    // enumeration (EnumOutputs returns 0 outputs after AMF runs), so the
    // capture device must be enumerated first. That same device is later
    // stored in `SHARED_D3D11_DEVICE` (see `real_capture`) and reused as
    // the encoder's D3D11 device, so capture and encode share one device
    // and the VP blit needs no cross-device texture copy.
    #[cfg(target_os = "windows")]
    if !use_test_pattern {
        match tether_capture::windows::pre_create() {
            Ok(pre) => {
                *lock_host_state(&PRECREATED_CAPTURE, "precreated capture") = Some(pre);
            }
            Err(e) => {
                tracing::warn!(error = %e, "DXGI pre-create failed; capture will retry later");
            }
        }
    }

    // On a Linux NVIDIA host, pin every NVIDIA codec subsystem (NVENC/NVDEC
    // CUDA context, the EGL→CUDA importer, the NVDEC surface pool's Vulkan
    // device) to the physical GPU the dma-buf producer will use. The producer
    // leads: read the GPU its HighPerformance wgpu adapter picks and pin to it
    // BEFORE the capability probe and any encoder/decoder construct, so the
    // whole zero-copy path lands on one GPU on multi-GPU hosts. Unpinned
    // (non-NVIDIA, or no Vulkan adapter) keeps FFmpeg's default device, i.e.
    // the original single-GPU behavior.
    #[cfg(target_os = "linux")]
    if !use_test_pattern && tether_codec::nvenc::nvidia_gpu_present() {
        match tether_gpuconvert::gpu_select::preferred_device_uuid().await {
            Some(uuid) => tether_codec::nvenc::pin_gpu_uuid(uuid),
            // NVIDIA present (sysfs) but no Vulkan adapter to read a UUID from —
            // the producer and the CUDA/EGL side can't be aligned. On a
            // single-GPU host the default device still works; on a multi-GPU
            // host this is a latent producer/encoder GPU mismatch that faults on
            // the first real encode, so log loudly rather than as a soft warning.
            None => tracing::error!(
                "NVIDIA host but no Vulkan adapter found to pin the GPU; the codec \
                 path will use FFmpeg's default CUDA device — on a multi-GPU host \
                 the dma-buf producer and encoder may land on different GPUs and \
                 fault. Check the Vulkan driver/ICD installation."
            ),
        }
    }

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
        let pending = tokio::select! {
            biased;
            ctrl_c = tokio::signal::ctrl_c() => {
                if let Err(e) = ctrl_c {
                    warn!(error = %e, "ctrl-c handler failed; shutting down anyway");
                } else {
                    info!("ctrl-c received at main loop; shutting down");
                }
                break;
            }
            _ = shutdown.notified() => {
                info!("shell stop received at main loop; shutting down");
                break;
            }
            accept_res = server.accept_pending() => match accept_res {
                Some(Ok(p)) => p,
                Some(Err(e)) => {
                    warn!(error = ?e, "server.accept_pending failed; continuing");
                    continue;
                }
                None => {
                    warn!("server closed; ending main loop");
                    break;
                }
            },
        };
        let peer = pending.remote_address();
        info!(remote = %peer, "incoming connection; authorizing");

        // Authorize before any session exists: allowlist hit (Resume) or a
        // windowed first-contact pairing. An unpaired client with no open
        // window is refused here, so the input-injection path is structurally
        // unreachable for it. The timeout stops a stalled or hostile peer from
        // wedging the accept loop — dropping the future closes its connection.
        let (conn, fp) = match tokio::time::timeout(
            AUTHORIZE_TIMEOUT,
            pairing::authorize(pending, &pairing_state, Instant::now()),
        )
        .await
        {
            Ok(Authorized::Session {
                pending,
                fp,
                newly_paired,
            }) => {
                // Authorized: promote to a full session (wire control + input).
                let conn = match pending.into_connection().await {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(remote = %peer, error = ?e, "promoting authorized peer failed");
                        continue;
                    }
                };
                if let Some(label) = newly_paired {
                    reporter.emit(&EngineEvent::Paired {
                        peer: peer.to_string(),
                        label,
                    });
                    // Refresh the shell's device list with the new entry.
                    reporter.emit(&EngineEvent::PeerList {
                        peers: pairing_state.peer_list(),
                    });
                }
                (Arc::new(conn), fp)
            }
            Ok(Authorized::Refused(RefusedReason::PairingRequired)) => {
                info!(remote = %peer, "unpaired client refused (no pairing window)");
                reporter.emit(&EngineEvent::PairingRequired {
                    peer: peer.to_string(),
                });
                continue;
            }
            Ok(Authorized::Refused(RefusedReason::Protocol(msg))) => {
                warn!(remote = %peer, reason = %msg, "pending connection refused");
                continue;
            }
            Err(_elapsed) => {
                warn!(remote = %peer, "authorization timed out; dropping connection");
                continue;
            }
        };

        info!(remote = %peer, "client authorized");
        reporter.emit(&EngineEvent::PeerConnected {
            peer: peer.to_string(),
        });

        // Register the live session so `RevokePeer` can tear it down (by closing
        // its connection). The guard clears the slot when this loop iteration
        // ends, however it ends. `revoked` lets the session-end path below
        // attribute the disconnect to a revocation rather than a clean exit.
        let revoked = Arc::new(AtomicBool::new(false));
        let _active_guard = ActiveSessionGuard::register(
            pairing_state.active.clone(),
            ActiveSession {
                fp,
                conn: conn.clone(),
                revoked: revoked.clone(),
            },
        );

        let host_encode_profiles: Vec<VideoProfile> = if use_test_pattern {
            vec![VideoProfile::H264_8BIT_420]
        } else {
            tether_probe::host_encode_profiles()
        };
        tracing::debug!(
            host_encode_profiles = ?host_encode_profiles,
            forced_video_profile = ?forced_video_profile,
            "host video encode capabilities (capture-bridge filtered)"
        );

        // Advertise audio only when enabled *and* this platform has a capture
        // backend, so a client never opts into audio we can't deliver. Both
        // ends use the fixed default Opus config (48 kHz stereo).
        let audio_config = (audio_enabled && tether_audio::capture::is_supported())
            .then(|| tether_audio::OpusConfig::default().wire_config());
        let display_descriptors =
            Arc::new(StdMutex::new(initial_display_descriptors(use_test_pattern)));
        let cfg = HostSessionConfig {
            server_name: "tether-host".to_string(),
            audio_config,
            displays: lock_host_state(&display_descriptors, "display descriptors").clone(),
        };
        // `HostSession::accept` takes the channel through the
        // `ControlChannel` trait object so it's mockable in tests.
        // We keep the original `Arc<Connection>` in scope for the rest
        // of `handle_client`, which uses concrete-type methods
        // (datagram send/recv, input recv, connection stats) that
        // are outside the `ControlChannel` surface.
        let session = match HostSession::accept(
            conn.clone() as Arc<dyn tether_transport::ControlChannel>,
            cfg,
            |client_caps| {
                tether_probe::pick_supported_profile_with_force(
                    &host_encode_profiles,
                    client_caps,
                    forced_video_profile,
                )
            },
        )
        .await
        {
            Ok(s) => s,
            Err(AcceptError::NoProfileIntersection { client }) => {
                // HostSession has already sent a typed handshake rejection.
                // We log the host list ourselves — the error doesn't
                // carry it because the selector (which closes over it)
                // is the only thing that saw both sides.
                warn!(
                    host_encode_profiles = ?host_encode_profiles,
                    client_decode_profiles = ?client,
                    "no mutual video profile; session ended"
                );
                reporter.emit(&EngineEvent::PeerDisconnected {
                    reason: "no mutual video profile".to_string(),
                });
                continue;
            }
            Err(AcceptError::Transport(e)) => {
                warn!(error = ?e, "handshake transport error; session ended");
                reporter.emit(&EngineEvent::PeerDisconnected {
                    reason: format!("handshake error: {e}"),
                });
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
            _ = shutdown.notified() => {
                info!("shell stop received during session; shutting down");
                break;
            }
            res = handle_client(
                session,
                conn,
                use_test_pattern,
                audio_enabled,
                display_descriptors,
            ) => {
                let reason = if revoked.load(Ordering::Relaxed) {
                    // The session ended because the operator revoked this peer
                    // (its connection was closed out from under handle_client).
                    "revoked".to_string()
                } else {
                    match res {
                        Ok(()) => "clean".to_string(),
                        Err(e) => {
                            warn!(error = ?e, "session ended with error; accepting next client");
                            e.to_string()
                        }
                    }
                };
                reporter.emit(&EngineEvent::PeerDisconnected { reason });
            }
        }

        // `_active_guard` drops here (or on any `continue`/`break` above),
        // clearing the active-session slot.
    }

    server.close_and_wait(0, b"host shutdown").await;
    Ok(())
}

/// Build the per-connection input injector for the current platform. On
/// Linux this lazily requests the `/dev/uinput` permission via a GUI
/// PolicyKit prompt on first connection (mirroring the screen-capture
/// portal) and retries before falling back to no-op; other platforms use
/// the library's plain connect-or-noop default.
async fn build_injector() -> Box<dyn tether_input::inject::Injector> {
    #[cfg(target_os = "linux")]
    {
        setup_input::linux_injector().await
    }
    #[cfg(not(target_os = "linux"))]
    {
        tether_input::inject::default_injector().await
    }
}

/// Owns every piece of per-connection state — encoder thread, injector,
/// recv tasks. When this function returns the whole graph drops together:
/// the injector releases its virtual input devices, the QUIC connection
/// closes, the encoder frees its hardware context. Anything that needs to
/// survive across reconnects must live in `main`, not here.
async fn handle_client(
    session: HostSession,
    conn: Arc<Connection>,
    use_test_pattern: bool,
    audio_enabled: bool,
    display_descriptors: Arc<StdMutex<Vec<DisplayDescriptor>>>,
) -> anyhow::Result<()> {
    // The handshake (decode-profile negotiation, display topology advert,
    // typed rejection on no-match) ran in `HostSession::accept`. Unpack the
    // per-connection state it produced.
    //
    // `session.channel` is dropped here: it's an `Arc<dyn ControlChannel>`
    // pointing to the same `Connection` that `conn` holds, just
    // type-erased for the session-level abstraction. The recv tasks
    // and send thread below need concrete-`Connection` methods
    // (datagram send/recv, input recv, connection stats), so we use
    // `conn` directly.
    //
    // Bind `_client_decode_profiles` and `_client_hello` because the
    // immediate orchestration below doesn't read them — they're kept on
    // the session for callers (logs, diagnostics, future adaptive
    // policy) and accessible via the field names if needed.
    let HostSession {
        channel: _,
        negotiated: chosen_profile,
        negotiated_video: _negotiated_video,
        client_hello,
        client_decode_profiles: _client_decode_profiles,
        idr_signal: force_idr,
        stream_ready,
    } = session;
    let initial_viewport = client_hello.initial_viewport.filter(|v| v.is_valid());
    if let Some(v) = initial_viewport {
        info!(
            width = v.width,
            height = v.height,
            "client hello included an initial viewport; waiting for live viewport hint before streaming"
        );
    }
    let audio_active = audio_enabled && tether_audio::capture::is_supported();
    let session_summary = Arc::new(SessionSummaryState::new(
        "host",
        chosen_profile,
        audio_active,
    ));

    // `use_test_pattern` from here on only switches the capture source;
    // the handshake-time profile floor it implied is already baked into
    // the negotiated profile.

    // Cursor pump runs as its own task below (after the JoinSet is
    // built) so it owns the cursor source for the lifetime of the
    // session — startup shape delivery, ongoing position datagrams,
    // and per-sprite-change shape forwarding are one loop.

    // Acquire a capture stream — either real platform capture or the
    // synthetic test pattern fallback. Real Linux capture is async (the
    // portal handshake awaits a user permission dialog); test pattern is
    // sync. Both end up as a `Receiver<CapturedFrame>` so the send loop
    // is identical.
    // Drop the CaptureHandle's FPS-control side: no production
    // backend honours runtime FPS retunes today (PipeWire format
    // renegotiation and SCK `minimumFrameInterval` updates are
    // per-backend follow-up work). Re-grab the Arc clone via
    // `capture_handle.fps_handle()` when the first backend wires
    // through. See `tether_capture::CaptureHandle` docs.
    let mut capture_handle = pick_capture_source(use_test_pattern, chosen_profile, None).await?;
    // Take the per-backend cursor source out before we drop the
    // handle. Wayland/PipeWire fills this with a `SPA_META_Cursor`
    // parser; the test-pattern and macOS-stub backends leave it
    // `None`, in which case we fall back to the placeholder so the
    // wire-level cursor protocol stays exercised.
    let cursor_source: Box<dyn CursorSource> = capture_handle
        .take_cursor_source()
        .unwrap_or_else(|| Box::new(PlaceholderCursorSource::new()));
    let frames = capture_handle.into_rx();

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

    // Cursor coordinate-frame watch: the encode loop posts
    // `(capture_w, capture_h, encode_w, encode_h)` on every encoder
    // slot rebuild so the cursor pump can rescale positions + sprite
    // dims into the encode-pixel space the client actually renders
    // against. Without this, a Retina macOS host (capture 3024×1952)
    // negotiated down to a 1104×720 encode would send a sprite at
    // 2.74× the correct visual size and a position 2.74× off.
    let (cursor_frame_tx, cursor_frame_rx) =
        tokio::sync::watch::channel::<Option<CursorFrameDims>>(None);

    // Capture + send runs on a dedicated OS thread per the expert review:
    // the hot path doesn't share the tokio runtime with anything else.
    // We keep the JoinHandle so the disconnect path can wait for the
    // thread to actually exit before we return — otherwise the encoder
    // and capture receiver would still be live in the background while
    // a follow-on session tried to grab the same resources. The shutdown
    // flag breaks the send loop out of its idle wait on a quiet desktop,
    // where `frames.recv` would otherwise block past disconnect detection.
    let send_shutdown = Arc::new(AtomicBool::new(false));
    // Notified when the send thread exits for any reason — fatal encoder
    // init failure, GPU bridge collapse, capture-channel disconnect, or
    // a clean shutdown response. Lets `handle_client`'s `select!`
    // unblock immediately on a fatal send-side exit instead of waiting
    // for the client to process Goodbye and drop the QUIC connection.
    // Notify is idempotent — a benign notify during clean shutdown is a
    // no-op.
    let send_exited = Arc::new(tokio::sync::Notify::new());
    let send_exited_for_thread = send_exited.clone();
    // Set once this side has sent a typed shutdown reason on the control
    // stream. Prevents the final teardown from overwriting a specific protocol
    // error / fatal send-loop Goodbye with a generic clean "session ended"
    // notice, while still allowing a reciprocal stats-bearing Goodbye when the
    // peer initiates shutdown.
    let shutdown_notice_sent = Arc::new(AtomicBool::new(false));
    // Drop-oldest single-slot mailbox for the most recent `ClientStats`
    // window. Written by the control recv task, drained by the
    // encode-and-send thread on each loop iteration.
    let latest_client_stats: LatestClientStats = Arc::new(StdMutex::new(None));
    // Latest viewport-target the client has communicated. Startup requires a
    // live SetViewportHint from the renderer before the video gate opens; the
    // older ClientHello hint is ignored because the app can only know the real
    // physical viewport after the window exists.
    let latest_viewport: LatestViewport = Arc::new(StdMutex::new(ViewportState {
        viewport: None,
        seq: 0,
    }));
    // `stream_ready`: gate at the head of the send loop. The client
    // signals it has built its decoders by sending
    // `ControlMessage::StreamReady`; until then we drop captured frames
    // so the first ~100-500 ms don't race the client's decoder
    // construction and render as garbage. Bound to the session in
    // `HostSession::accept`.
    // Loss-recovery signal: client's RequestRecovery raises this with
    let conn_send = conn.clone();
    let force_idr_for_send = force_idr.clone();
    let send_shutdown_for_thread = send_shutdown.clone();
    let stream_ready_for_thread = stream_ready.clone();
    let video_ready_requested = Arc::new(AtomicBool::new(false));
    let latest_client_stats_for_send = latest_client_stats.clone();
    let latest_viewport_for_send = latest_viewport.clone();
    let shutdown_notice_for_send = shutdown_notice_sent.clone();
    let session_summary_for_send = session_summary.clone();
    // The sync send thread is not a tokio worker, so it uses the
    // runtime handle only for the few async control sends needed on
    // fatal exits. Regular video output is synchronous datagram send:
    // every frame, IDR and P alike, goes through FrameFragmenter and
    // quinn's non-blocking datagram queue.
    let runtime_handle_for_send = tokio::runtime::Handle::current();
    let send_handle = std::thread::Builder::new()
        .name("tether-host-send".into())
        .spawn(move || {
            run_capture_and_send(
                conn_send,
                frames,
                force_idr_for_send,
                display_dims_tx,
                cursor_frame_tx,
                send_shutdown_for_thread,
                chosen_profile,
                stream_ready_for_thread,
                runtime_handle_for_send,
                latest_client_stats_for_send,
                latest_viewport_for_send,
                send_exited_for_thread,
                shutdown_notice_for_send,
                session_summary_for_send,
            )
        })?;

    // Audio sender: capture system audio → Opus → unreliable `Datagram::Audio`,
    // on its own thread parallel to the video send thread. `audio_ready` is the
    // client's `StreamReady.audio`; until it's set we drop captured audio so we
    // don't stream into a client with no playback. Reuses `send_shutdown` to
    // tear down with the session. Skipped (and left silent) when audio is off
    // or the platform has no capture backend.
    let audio_ready = Arc::new(AtomicBool::new(false));
    let audio_handle = if audio_active {
        let conn_audio = conn.clone();
        let audio_shutdown = send_shutdown.clone();
        let audio_ready_thread = audio_ready.clone();
        let session_summary_for_audio = session_summary.clone();
        Some(
            std::thread::Builder::new()
                .name("tether-host-audio".into())
                .spawn(move || {
                    run_audio_capture_and_send(
                        conn_audio,
                        audio_shutdown,
                        audio_ready_thread,
                        session_summary_for_audio,
                    );
                })?,
        )
    } else {
        None
    };

    // Per-connection injector. The backend's virtual devices live inside
    // the injector; when the last Arc drops, they're released. We
    // deliberately hand the three clones to the three recv tasks and don't
    // keep a fourth reference in this scope — otherwise the original
    // outlives the tasks, refcount never hits zero, and the host's input
    // devices stay open until the process exits (which is the bug that
    // prompted this whole rewrite). The recv tasks themselves are owned by
    // the JoinSet below, so tasks.shutdown() is what triggers the final
    // drop.
    //
    // tokio::sync::Mutex is the right primitive: the lock is held only
    // for the duration of one inject call (microseconds), but both
    // holders are async, and a std::sync::Mutex held across an await
    // would risk blocking a tokio worker.
    let injector = Arc::new(TokioMutex::new(build_injector().await));

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
        let video_ready_requested_ctl = video_ready_requested.clone();
        let audio_ready_ctl = audio_ready.clone();
        let host_audio_available_ctl = audio_active;
        let latest_client_stats_for_ctl = latest_client_stats.clone();
        let latest_viewport_for_ctl = latest_viewport.clone();
        let force_idr_for_viewport = force_idr.clone();
        let shutdown_notice_for_ctl = shutdown_notice_sent.clone();
        let session_summary_for_ctl = session_summary.clone();
        tasks.spawn(async move {
            // Per-message rate limit for IDR-triggering control messages.
            // A client flooding ForceIdr / RequestRecovery on the
            // reliable control stream could otherwise pin the encoder
            // into perpetual-keyframe mode (bitrate explodes,
            // session degrades to a slideshow). 250 ms floor matches
            // the minimum useful keyframe cadence — faster bursts
            // are coalesced through IdrSignal anyway, so the cap
            // costs nothing in normal operation.
            let mut last_idr_request: Option<std::time::Instant> = None;
            const IDR_REQUEST_MIN_INTERVAL: std::time::Duration =
                std::time::Duration::from_millis(250);
            loop {
                match conn.recv_control().await {
                    Ok(ControlMessage::ForceIdr) => {
                        let now = std::time::Instant::now();
                        if last_idr_request
                            .is_some_and(|t| now.duration_since(t) < IDR_REQUEST_MIN_INTERVAL)
                        {
                            tracing::trace!("ForceIdr rate-limited");
                            continue;
                        }
                        last_idr_request = Some(now);
                        tracing::debug!("client requested IDR");
                        force_idr.raise();
                    }
                    Ok(ControlMessage::SetCursorMode { mode }) => {
                        // Host-side state is advisory: dispatch on
                        // whichever input variant arrives. Log the
                        // transition + echo the ack so the client
                        // knows the host saw it.
                        tracing::info!(?mode, "client switched cursor mode");
                        // Ack inline rather than tokio::spawn per
                        // message: a client that flaps cursor mode
                        // would otherwise grow the task queue
                        // unboundedly. The control recv loop is
                        // single-threaded; brief send blocking on
                        // backpressure is fine and self-limiting.
                        let ack = ControlMessage::SetCursorMode { mode };
                        if let Err(e) = conn.send_control(&ack).await {
                            tracing::debug!(error = ?e, "SetCursorMode ack send failed");
                        }
                    }
                    Ok(ControlMessage::RequestRecovery {
                        last_reassembled_frame_id,
                    }) => {
                        let now = std::time::Instant::now();
                        if last_idr_request
                            .is_some_and(|t| now.duration_since(t) < IDR_REQUEST_MIN_INTERVAL)
                        {
                            tracing::trace!(
                                last_reassembled_frame_id,
                                "RequestRecovery rate-limited"
                            );
                            continue;
                        }
                        last_idr_request = Some(now);
                        // Recovery means a full IDR. `last_reassembled_frame_id`
                        // is logged for diagnostics only — it is the client's
                        // reassembler high-water mark, not a decode-verified
                        // reference, so it can't drive LTR re-prediction even if
                        // an encoder supported it. RFI is intentionally not
                        // implemented (see the `RequestRecovery` doc); FEC + the
                        // bounded GOP cover its failure mode.
                        tracing::info!(
                            last_reassembled_frame_id,
                            "client requested recovery; emitting forced IDR"
                        );
                        force_idr.raise();
                    }
                    Ok(ControlMessage::StreamReady { video, audio }) => {
                        info!(
                            event = "stream_ready",
                            video, audio, "client signalled StreamReady"
                        );
                        if video {
                            video_ready_requested_ctl.store(true, Ordering::Release);
                            let has_viewport =
                                lock_host_state(&latest_viewport_for_ctl, "latest viewport")
                                    .viewport
                                    .is_some();
                            if has_viewport {
                                stream_ready_ctl.store(true, Ordering::Release);
                                // Defensive IDR at gate-open. A fresh encoder's
                                // first frame is already an IDR, so this is usually
                                // redundant — but if StreamReady lands before the
                                // encoder lazy-inits, the pending flag guarantees
                                // the first encoded frame is still forced to a
                                // keyframe. Coalesces harmlessly via IdrSignal.
                                // Matches Moonlight/Sunshine forcing an IDR at
                                // stream start.
                                force_idr.raise();
                            } else {
                                info!(
                                    "video StreamReady received before viewport; waiting for \
                                     SetViewportHint before opening video gate"
                                );
                            }
                        }
                        // Open the audio gate too; the host audio thread drops
                        // captured frames until the client says it can play them.
                        audio_ready_ctl.store(audio, Ordering::Release);
                        session_summary_for_ctl.set_audio_active(host_summary_audio_active(
                            host_audio_available_ctl,
                            audio,
                        ));
                    }
                    Ok(ControlMessage::StreamPause { stream_id }) => {
                        info!(%stream_id, "client paused stream (no-op today)");
                    }
                    Ok(ControlMessage::StreamResume { stream_id }) => {
                        info!(%stream_id, "client resumed stream (no-op today)");
                        // Force a fresh IDR so the client can latch
                        // onto the resumed stream without a partial GOP.
                        force_idr.raise();
                    }
                    Ok(ControlMessage::ClientStats {
                        window_ms,
                        frames_received,
                        incomplete_frames,
                        fragment_loss_events,
                        rtt_us,
                        fec_recovered_frames,
                        fec_recovered_fragments,
                    }) => {
                        info!(
                            window_ms,
                            frames_received,
                            incomplete_frames,
                            fragment_loss_events,
                            rtt_us,
                            fec_recovered_frames,
                            fec_recovered_fragments,
                            "client stats"
                        );
                        // Drop-oldest: the ABR controller only needs
                        // the most recent window. If the send thread
                        // hasn't drained yet (slow encoder loop), the
                        // older sample is discarded — we'd rather act
                        // on fresh telemetry than queue stale data.
                        *lock_host_state(&latest_client_stats_for_ctl, "latest client stats") =
                            Some(ClientStatsObservation {
                                incomplete_frames,
                                fragment_loss_events,
                            });
                    }
                    Ok(ControlMessage::SetViewportHint {
                        stream_id,
                        viewport: v,
                    }) => {
                        info!(
                            %stream_id,
                            width = v.width,
                            height = v.height,
                            "client viewport changed"
                        );
                        // Latch the new viewport. The send thread
                        // notices the seq bump on its next iteration
                        // and rebuilds the encoder only if encode dims
                        // actually change. We force an IDR regardless
                        // so the client sees a clean cut on whatever
                        // dimensions the backend chooses.
                        let next = if v.is_valid() {
                            Some(v)
                        } else {
                            warn!(
                                width = v.width,
                                height = v.height,
                                "invalid viewport hint; video gate remains closed until a valid viewport arrives"
                            );
                            None
                        };
                        let mut guard =
                            lock_host_state(&latest_viewport_for_ctl, "latest viewport");
                        let changed = guard.update_if_changed(next);
                        drop(guard);
                        if !changed {
                            tracing::trace!(
                                width = v.width,
                                height = v.height,
                                "duplicate viewport hint; skipping seq bump and IDR"
                            );
                            continue;
                        }
                        if next.is_some()
                            && video_ready_requested_ctl.load(Ordering::Acquire)
                            && !stream_ready_ctl.load(Ordering::Acquire)
                        {
                            info!(
                                width = v.width,
                                height = v.height,
                                "initial viewport received; opening video gate"
                            );
                            stream_ready_ctl.store(true, Ordering::Release);
                        }
                        tracing::debug!(
                            width = v.width,
                            height = v.height,
                            "SetViewportHint: forcing IDR; encoder rebuild fires only if \
                             encode dims change"
                        );
                        force_idr_for_viewport.raise();
                    }
                    Ok(ControlMessage::ClientDisplayMetrics(metrics)) => {
                        info!(
                            client_display_id = metrics.display_id,
                            width = metrics.mode.width,
                            height = metrics.mode.height,
                            refresh_millihz = metrics.mode.refresh_millihz,
                            scale = format!("{}/{}", metrics.scale_num, metrics.scale_den),
                            safe_area = ?metrics.safe_area,
                            "client display metrics"
                        );
                    }
                    Ok(ControlMessage::SetDisplayMode {
                        request_id,
                        display_id,
                        ..
                    }) => {
                        let response = ControlMessage::DisplayModeResult {
                            request_id,
                            display_id,
                            status: DisplayModeStatus::Unsupported,
                            actual_mode: None,
                        };
                        if let Err(e) = conn.send_control(&response).await {
                            warn!(
                                error = ?e,
                                %request_id,
                                %display_id,
                                "display mode result send failed; ending control loop"
                            );
                            return;
                        }
                    }
                    Ok(ControlMessage::DisplayModeResult { .. }) => {
                        tracing::trace!("unsolicited display mode result; ignoring");
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
                    Ok(ControlMessage::Goodbye {
                        reason,
                        code,
                        final_stats,
                    }) => {
                        info!(event = "peer_goodbye", %reason, ?code, "client said goodbye");
                        log_peer_session_summary("client", final_stats.as_deref());
                        if !shutdown_notice_for_ctl.swap(true, Ordering::AcqRel) {
                            let ack_reason = "host acknowledged client goodbye";
                            let sent = send_goodbye_notice(
                                &conn,
                                ack_reason,
                                GoodbyeCode::Clean,
                                &session_summary_for_ctl,
                            )
                            .await;
                            info!(
                                event = "session_teardown",
                                reason = ack_reason,
                                code = ?GoodbyeCode::Clean,
                                sent,
                                "host session shutdown notice"
                            );
                        }
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
                        if !shutdown_notice_for_ctl.swap(true, Ordering::AcqRel) {
                            let sent = send_goodbye_notice(
                                &conn,
                                reason.as_str(),
                                GoodbyeCode::ProtocolError,
                                &session_summary_for_ctl,
                            )
                            .await;
                            info!(
                                event = "session_teardown",
                                reason = reason.as_str(),
                                code = ?GoodbyeCode::ProtocolError,
                                sent,
                                "host session shutdown notice"
                            );
                        }
                        return;
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
        let conn = conn.clone();
        let display_descriptors = display_descriptors.clone();
        let mut rx = display_dims_rx;
        tasks.spawn(async move {
            while rx.changed().await.is_ok() {
                // Copy the value out and drop the borrow guard before
                // awaiting on the injector lock — `watch::Ref` is not Send,
                // so holding it across an .await fails the Send bound.
                let dims = *rx.borrow();
                if let Some((w, h)) = dims {
                    injector.lock().await.set_display_size(w, h);
                    let next = {
                        let mut guard =
                            lock_host_state(&display_descriptors, "display descriptors");
                        let next = tether_capture::display_list_with_primary_mode(
                            guard.clone(),
                            w,
                            h,
                            ENCODER_FPS.saturating_mul(1000),
                        );
                        if *guard == next {
                            None
                        } else {
                            *guard = next.clone();
                            Some(next)
                        }
                    };
                    if let Some(displays) = next {
                        if let Err(e) = conn
                            .send_control(&ControlMessage::DisplayList { displays })
                            .await
                        {
                            warn!(
                                error = ?e,
                                "display list update send failed; ending display follower"
                            );
                            return;
                        }
                    }
                }
            }
        });
    }

    // Input recv: drain the client's input stream and feed each event
    // into the host's injection backend.
    {
        let conn = conn.clone();
        let injector = injector.clone();
        let shutdown_notice_for_input = shutdown_notice_sent.clone();
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
                        if shutdown_notice_for_input.load(Ordering::Acquire)
                            || e.is_clean_shutdown_recv()
                        {
                            info!(error = ?e, "input recv stopped during session shutdown");
                        } else {
                            warn!(error = ?e, "input recv failed; ending input task");
                        }
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
        let shutdown_notice_for_datagram = shutdown_notice_sent.clone();
        tasks.spawn(async move {
            loop {
                match conn.recv_datagram().await {
                    Ok(Datagram::ClientCursor(c)) => {
                        let mut inj = injector.lock().await;
                        if let Err(e) = inj.inject_cursor(&c) {
                            warn!(error = %e, "cursor inject failed; dropping");
                        }
                    }
                    Ok(Datagram::Video(_))
                    | Ok(Datagram::HostCursor(_))
                    | Ok(Datagram::Audio(_)) => {
                        tracing::trace!("unexpected host-direction datagram on host; ignoring");
                    }
                    Err(e) => {
                        if shutdown_notice_for_datagram.load(Ordering::Acquire)
                            || e.is_clean_shutdown_recv()
                        {
                            info!(error = ?e, "datagram recv stopped during session shutdown");
                        } else {
                            warn!(error = ?e, "datagram recv failed; ending datagram task");
                        }
                        return;
                    }
                }
            }
        });
    }

    // Cursor pump: forwards shape changes on the reliable control
    // stream, position updates on the unreliable cursor datagram
    // channel. One task drives both so id-dedup state and latest-
    // position state share a single owner. Ending this task on a
    // send error tears the session down through the JoinSet just
    // like the other recv tasks.
    {
        let conn = conn.clone();
        let cursor_frame_rx = cursor_frame_rx.clone();
        tasks.spawn(async move {
            pump_cursor(conn, cursor_source, cursor_frame_rx).await;
        });
    }

    // Wait for any signal that the session is over:
    //   - Ctrl-C: user wants out
    //   - any per-connection task exited: disconnect, Goodbye, or recv error
    // Whichever fires first, the cleanup path below runs.
    let (teardown_reason, teardown_code) = tokio::select! {
        ctrl_c = tokio::signal::ctrl_c() => {
            if let Err(e) = ctrl_c {
                warn!(error = %e, "ctrl-c handler failed; tearing down anyway");
            } else {
                info!("ctrl-c received, ending session");
            }
            ("host interrupted", GoodbyeCode::Clean)
        }
        res = tasks.join_next() => {
            match res {
                Some(Ok(())) => info!("per-connection task ended; tearing down"),
                Some(Err(e)) => warn!(error = ?e, "per-connection task failed; tearing down"),
                None => warn!("joined empty task set; tearing down"),
            }
            ("host session task ended", GoodbyeCode::Clean)
        }
        () = send_exited.notified() => {
            // Send thread exited (fatal encoder init failure, GPU
            // bridge collapse, capture-channel disconnect, etc.).
            // On fatal exits the Goodbye is already in flight; on a
            // clean capture-channel disconnect the connection close
            // below serves as the implicit goodbye. Either way, tear
            // down immediately rather than waiting for the client to
            // ack-by-disconnect.
            info!("send thread exited; tearing down");
            ("host send loop ended", GoodbyeCode::Clean)
        }
    };

    if !shutdown_notice_sent.swap(true, Ordering::AcqRel) {
        let sent =
            send_goodbye_notice(&conn, teardown_reason, teardown_code, &session_summary).await;
        info!(
            event = "session_teardown",
            reason = teardown_reason,
            code = ?teardown_code,
            sent,
            "host session shutdown notice"
        );
    }

    // Tell the send threads to stop before closing QUIC. Paired with the
    // recv_timeout inside the loops so a static desktop / idle audio source
    // can't keep us blocked in `recv` past disconnect. Setting this first also
    // lets in-flight datagram sends classify ConnectionLost as clean shutdown
    // noise instead of a fatal transport failure.
    send_shutdown.store(true, Ordering::Relaxed);

    // Close the QUIC connection. This makes any blocked send/recv path error
    // out and tells still-alive tasks that the peer is gone. Cheap and
    // idempotent.
    conn.close(0, b"session ended");

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

    // Same for the audio thread, if we started one. `send_shutdown` (set above)
    // is its stop signal too.
    if let Some(audio_handle) = audio_handle {
        let _ = tokio::task::spawn_blocking(move || audio_handle.join()).await;
    }

    Ok(())
}

async fn send_goodbye_notice(
    conn: &Connection,
    reason: &str,
    code: GoodbyeCode,
    summary: &SessionSummaryState,
) -> bool {
    let msg = ControlMessage::Goodbye {
        reason: reason.to_string(),
        code,
        final_stats: Some(Box::new(summary.snapshot())),
    };
    match conn.send_control(&msg).await {
        Ok(()) => {
            let wait = (2 * conn.rtt()).clamp(
                std::time::Duration::from_millis(20),
                std::time::Duration::from_millis(200),
            );
            tokio::time::sleep(wait).await;
            true
        }
        Err(e) => {
            warn!(error = ?e, reason, ?code, "send Goodbye failed");
            false
        }
    }
}

fn send_goodbye_notice_blocking(
    runtime: &tokio::runtime::Handle,
    conn: &Connection,
    sent: &AtomicBool,
    reason: &str,
    code: GoodbyeCode,
    summary: &SessionSummaryState,
) {
    if sent.swap(true, Ordering::AcqRel) {
        return;
    }
    let sent = runtime.block_on(send_goodbye_notice(conn, reason, code, summary));
    info!(
        event = "session_teardown",
        reason,
        ?code,
        sent,
        "host session shutdown notice"
    );
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
    #[cfg(target_os = "linux")]
    scaler: Option<Scaler>,
    /// Compiled Mitchell shader pipelines, cached across scaler
    /// rebuilds. Without this, every viewport change recompiles
    /// five WGSL compute pipelines from scratch — on Vulkan that's
    /// a few hundred microseconds, on Metal / DX12 it's 50–200 ms
    /// per pipeline. The `Scaler` itself holds a clone of this
    /// `Arc` (so the actual `Pipelines` only drop when both
    /// references are gone), but storing it on the slot lets the
    /// next scaler reuse the compiled pipelines without going
    /// through `ScalerPipelines::build` again.
    #[cfg(target_os = "linux")]
    scaler_pipelines: Option<Arc<ScalerPipelines>>,
    /// macOS BGRA IOSurface -> optional Mitchell scale -> NV12/P010
    /// IOSurface bridge. Unlike the older YUV-plane bridge, this keeps
    /// SCK capture in BGRA until the final VideoToolbox input
    /// conversion, matching the Linux/Windows visual-quality shape.
    #[cfg(target_os = "macos")]
    bgra_iosurface_bridge: Option<BgraIOSurfaceBridge>,
    /// VideoToolbox-compatible destination IOSurface fourcc for the
    /// negotiated 4:2:0 profile (`420v` or `x420`).
    #[cfg(target_os = "macos")]
    iosurface_encode_fourcc: u32,
    /// The PooledIOSurface used by the previous macOS frame. Held
    /// across one frame so VideoToolbox's async encode can drain the
    /// CVPixelBuffer that wraps the IOSurface before the bridge recycles
    /// the slot.
    #[cfg(target_os = "macos")]
    prev_pooled: Option<PooledIOSurface>,
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

// `Ready` is the steady-state variant and lives for the encoder's whole
// lifetime; boxing it would only add an indirection on the hot path without
// shrinking any frequently-moved value, so the size skew is accepted.
#[cfg(target_os = "linux")]
#[allow(clippy::large_enum_variant)]
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
    Yuv444p(Yuv444pDmaBuf),
    P010(Bgra2P010DmaBuf),
    Xv30(Bgra2Xv30DmaBuf),
}

/// Outcome of one Gpu-frame encode attempt. Distinguishes per-frame
/// failures (drop this frame, keep going) from bridge construction
/// failure (no recovery; exit the send loop). Per-frame errors are
/// usually transient (a single bad PipeWire buffer); bridge-init failure
/// after the startup probe succeeded indicates device loss or OOM and
/// the client would just freeze if we silently dropped every subsequent
/// frame.
#[cfg(any(target_os = "linux", target_os = "macos"))]
enum GpuEncodeOutcome {
    Packets(Vec<tether_codec::EncodedPacket>),
    DropFrame(anyhow::Error),
    #[cfg(any(target_os = "linux", target_os = "macos"))]
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
        GpuConvertBridge::Yuv444p(b) => (b.device().clone(), b.queue().clone()),
        GpuConvertBridge::P010(b) => (b.device().clone(), b.queue().clone()),
        GpuConvertBridge::Xv30(b) => (b.device().clone(), b.queue().clone()),
    }
}

/// Build a Mitchell scaler for the given (src, dst), reusing the
/// pre-compiled pipelines from `cached_pipelines` if present.
/// Lazily compiles on first call per slot; subsequent calls (e.g.
/// a viewport change) clone the cached Arc instead of re-running
/// the shader compilation.
#[cfg(target_os = "linux")]
fn build_scaler_for_slot(
    device: wgpu::Device,
    queue: wgpu::Queue,
    cached_pipelines: &mut Option<Arc<ScalerPipelines>>,
    src_dims: (u32, u32),
    dst_dims: (u32, u32),
) -> Result<Scaler, ScalerError> {
    let pipelines = cached_pipelines
        .get_or_insert_with(|| Arc::new(ScalerPipelines::build(&device)))
        .clone();
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
/// pipeline: import BGRA into wgpu, compute BGRA→NV12/P010/YUV444 onto
/// exported DMA-BUF planes, hand them to the encoder's `encode_gpu`.
///
/// The bridge variant matches the negotiated chroma/depth. NVIDIA's planar
/// YUV444P branch is currently only reached if the live probe proves the
/// driver accepts `YU24` dma-buf import; tested NVIDIA EGL stacks reject it,
/// so NVIDIA Linux does not advertise 4:4:4 today. Chosen on lazy init and
/// fixed for the encoder's lifetime (chroma switch needs a full encoder
/// rebuild, same as resolution change).
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
                    if tether_codec::nvenc::nvidia_gpu_present() {
                        match pollster::block_on(Yuv444pDmaBuf::new(slot.width, slot.height)) {
                            Ok(b) => GpuConvertBridge::Yuv444p(b),
                            Err(e) => {
                                return GpuEncodeOutcome::Fatal(anyhow::anyhow!(
                                    "Yuv444p gpuconvert bridge init failed for {}x{} after \
                                     startup probe succeeded — device loss or OOM: {e}",
                                    slot.width,
                                    slot.height,
                                ));
                            }
                        }
                    } else {
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
                }
                (ChromaSubsampling::Yuv444, 10) => {
                    match pollster::block_on(Bgra2Xv30DmaBuf::new(slot.width, slot.height)) {
                        Ok(b) => GpuConvertBridge::Xv30(b),
                        Err(e) => {
                            return GpuEncodeOutcome::Fatal(anyhow::anyhow!(
                                "XV30 gpuconvert bridge init failed for {}x{} after \
                                 startup probe succeeded — device loss or OOM: {e}",
                                slot.width,
                                slot.height,
                            ));
                        }
                    }
                }
                // Any future profile combination not enumerated above.
                // tether_probe::host_supported_profiles must filter
                // unsupported (chroma, bit_depth) tuples out of the
                // negotiated set; reaching this arm is a contract
                // violation, not a transient failure.
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
                    &mut slot.scaler_pipelines,
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
        GpuConvertBridge::Yuv444p(b) => {
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
                        "import_bgra_dmabuf (yuv444p bridge): {e}"
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
                        "Yuv444pDmaBuf::convert: {e}"
                    ));
                }
            };
            drop(imported);
            yuv444p_dmabuf_to_codec_frame(yuv)
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
        GpuConvertBridge::Xv30(b) => {
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
                        "import_bgra_dmabuf (xv30 bridge): {e}"
                    ));
                }
            };
            let bridge_input = match scale_if_needed(slot.scaler.as_ref(), &imported) {
                Ok(t) => t,
                Err(e) => return GpuEncodeOutcome::DropFrame(e),
            };
            let xv30 = match b.convert(bridge_input) {
                Ok(f) => f,
                Err(e) => {
                    return GpuEncodeOutcome::DropFrame(anyhow::anyhow!(
                        "Bgra2Xv30DmaBuf::convert: {e}"
                    ));
                }
            };
            drop(imported);
            xv30_dmabuf_to_codec_frame(xv30)
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

/// Encode one ScreenCaptureKit-supplied BGRA IOSurface through the
/// VideoToolbox zero-copy path. The host bridge converts BGRA into the
/// negotiated VideoToolbox input family (NV12/P010/NV24/P410-style IOSurface)
/// before the encoder consumes it via `CVPixelBufferCreateWithIOSurface`.
///
/// The capture-side `release_guard` keeps the underlying IOSurface
/// alive for the duration of this call; the encoder's
/// `submit_iosurface` performs a fresh CFRetain on the wrapping
/// CVPixelBuffer so the surface stays valid for the encoder's
/// async work after we return.
#[cfg(target_os = "macos")]
fn encode_iosurface_frame(
    slot: &mut EncoderSlot,
    macos_gpu: &mut Option<MacosGpuState>,
    gpu: tether_capture::GpuCapturedFrame,
    pts: i64,
    force_keyframe: bool,
) -> GpuEncodeOutcome {
    let tether_capture::GpuCapturedSource::IOSurface(iosurface) = gpu.source;
    let src_frame = tether_codec::IOSurfaceFrame {
        surface: iosurface.surface,
        pixel_format: iosurface.pixel_format,
        width: iosurface.width,
        height: iosurface.height,
    };
    if is_macos_bgra_fourcc(src_frame.pixel_format) {
        if slot.iosurface_encode_fourcc == 0 {
            return GpuEncodeOutcome::Fatal(anyhow::anyhow!(
                "BGRA IOSurface capture requires a 4:2:0 VideoToolbox destination fourcc"
            ));
        }
        if macos_gpu.is_none() {
            match MacosGpuState::new() {
                Ok(state) => *macos_gpu = Some(state),
                Err(e) => {
                    return GpuEncodeOutcome::Fatal(anyhow::anyhow!(
                        "macOS BGRA IOSurface bridge device init failed: {e}"
                    ));
                }
            }
        }
        let macos_gpu = macos_gpu.as_ref().expect("just initialised");
        let rebuild_bridge = slot.bgra_iosurface_bridge.as_ref().is_none_or(|bridge| {
            bridge.src_dims() != (slot.capture_width, slot.capture_height)
                || bridge.dst_dims() != (slot.width, slot.height)
        });
        if rebuild_bridge {
            let bridge = match macos_gpu.build_bgra_bridge(
                (slot.capture_width, slot.capture_height),
                (slot.width, slot.height),
                slot.iosurface_encode_fourcc,
            ) {
                Ok(bridge) => bridge,
                Err(e) => {
                    return GpuEncodeOutcome::Fatal(anyhow::anyhow!(
                        "BGRA IOSurface bridge construction failed for {}x{} -> {}x{}: {e}",
                        slot.capture_width,
                        slot.capture_height,
                        slot.width,
                        slot.height,
                    ));
                }
            };
            info!(
                capture_w = slot.capture_width,
                capture_h = slot.capture_height,
                encode_w = slot.width,
                encode_h = slot.height,
                dst_fourcc = format_args!("0x{:08x}", slot.iosurface_encode_fourcc),
                "BGRA IOSurface bridge built for macOS host encode"
            );
            slot.bgra_iosurface_bridge = Some(bridge);
        }

        let bridge = slot.bgra_iosurface_bridge.as_ref().expect("just built");
        let pooled = match bridge.convert_to_iosurface(&src_frame) {
            Ok(pooled) => pooled,
            Err(IOSurfaceBridgeError::PoolExhausted { depth }) => {
                return GpuEncodeOutcome::DropFrame(anyhow::anyhow!(
                    "BGRA IOSurface pool exhausted (depth {depth})"
                ));
            }
            Err(e) => {
                return GpuEncodeOutcome::Fatal(anyhow::anyhow!(
                    "BGRA IOSurface convert failed structurally: {e}"
                ));
            }
        };
        let packets = match slot.encoder.encode_gpu(
            GpuEncoderFrame::IOSurface(&pooled.frame),
            pts,
            force_keyframe,
        ) {
            Ok(p) => p,
            Err(e) => return GpuEncodeOutcome::DropFrame(anyhow::anyhow!("encode_gpu: {e}")),
        };
        slot.prev_pooled = Some(pooled);
        return GpuEncodeOutcome::Packets(packets);
    }

    slot.prev_pooled = None;
    match slot
        .encoder
        .encode_gpu(GpuEncoderFrame::IOSurface(&src_frame), pts, force_keyframe)
    {
        Ok(p) => GpuEncodeOutcome::Packets(p),
        Err(e) => GpuEncodeOutcome::DropFrame(anyhow::anyhow!("encode_gpu: {e}")),
    }
}

#[cfg(target_os = "macos")]
fn is_macos_bgra_fourcc(fourcc: u32) -> bool {
    fourcc == u32::from_be_bytes(*b"BGRA")
}

#[cfg(target_os = "macos")]
fn macos_iosurface_encode_fourcc(profile: VideoProfile) -> Option<u32> {
    use tether_codec::macos_interop::{
        NV12_VIDEO_RANGE_FOURCC, NV24_VIDEO_RANGE_FOURCC, X420_FOURCC, X444_FOURCC,
    };
    match (profile.chroma, profile.bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => Some(NV12_VIDEO_RANGE_FOURCC),
        (ChromaSubsampling::Yuv420, 10) => Some(X420_FOURCC),
        (ChromaSubsampling::Yuv444, 8) => Some(NV24_VIDEO_RANGE_FOURCC),
        (ChromaSubsampling::Yuv444, 10) => Some(X444_FOURCC),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
struct MacosGpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[cfg(target_os = "macos")]
impl MacosGpuState {
    fn new() -> anyhow::Result<Self> {
        let (device, queue, caps) =
            pollster::block_on(tether_gpuconvert::nv12_iosurface::build_bridge_device())?;
        tracing::info!(
            supports_10bit = caps.supports_10bit,
            "macOS BGRA IOSurface bridge wgpu device built"
        );
        Ok(Self { device, queue })
    }

    fn build_bgra_bridge(
        &self,
        src_dims: (u32, u32),
        dst_dims: (u32, u32),
        dst_fourcc: u32,
    ) -> anyhow::Result<BgraIOSurfaceBridge> {
        BgraIOSurfaceBridge::new(
            self.device.clone(),
            self.queue.clone(),
            src_dims,
            dst_dims,
            dst_fourcc,
        )
        .map_err(|e| anyhow::anyhow!("BgraIOSurfaceBridge::new: {e}"))
    }
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

/// Build a `DmaBufFrame` for NVIDIA's diagnostic planar YUV444P path: one
/// DRM object, one `YU24` layer with three full-resolution R8 planes. Thin
/// adapter over `tether_codec::build_yuv444p_dmabuf_frame` so production and
/// probe descriptors cannot drift if a driver starts accepting this import.
#[cfg(target_os = "linux")]
fn yuv444p_dmabuf_to_codec_frame(out: Yuv444pDmaBufFrame) -> DmaBufFrame {
    tether_codec::build_yuv444p_dmabuf_frame(
        out.fd,
        out.size,
        out.modifier,
        out.y_offset,
        out.y_stride,
        out.u_offset,
        out.u_stride,
        out.v_offset,
        out.v_stride,
    )
}

/// Build a `DmaBufFrame` for the XV30 (HEVC Main 4:4:4 10-bit) path:
/// one DRM object, one `XV30` layer, one packed plane (32 bpp, layout
/// `[31:30]X | [29:20]V | [19:10]U | [9:0]Y`). FFmpeg consumes this as
/// `AV_PIX_FMT_XV30LE` via `vaapi_drm_format_map`. The construction
/// itself is delegated to `tether_codec::build_xv30_dmabuf_frame` so
/// the production send loop and the startup probe share one source of
/// truth — cross-table consistency tests in `tether-render::dmabuf_test`
/// pin the fourcc family.
#[cfg(target_os = "linux")]
fn xv30_dmabuf_to_codec_frame(out: Xv30DmaBufFrame) -> DmaBufFrame {
    tether_codec::build_xv30_dmabuf_frame(out.fd, out.size, out.modifier, out.offset, out.stride)
}

/// Long-running cursor pump. Owns the per-backend
/// [`CursorSource`] for the lifetime of the session and drives two
/// wire paths off it:
///
/// - **Shape changes** (reliable control stream): each unique sprite
///   id is sent as a `CursorShape` with full pixel bytes the first
///   time we see it, then activated with `CursorUseShape`. Subsequent
///   activations of the same id skip the pixel re-send so a
///   compositor that recycles arrows / text-beams / hand cursors
///   pays for them once.
/// - **Position** (unreliable `Datagram::HostCursor`): polled at
///   ~120 Hz, debounced — we only emit when the snapshot actually
///   changed, so an idle desktop keeps the channel quiet enough for
///   the host-side native-damage path to flag the frame as unchanged
///   and skip the encoder entirely.
///
/// Returning from this function ends the JoinSet task and tears down
/// the session. The match arms below treat a send error as fatal
/// because there's no useful retry — the connection is already gone.
/// Dims the cursor pump needs to translate from the cursor source's
/// native coordinate frame (capture pixels) into the encode-pixel
/// frame the client renders against. Updated by the encode loop on
/// each encoder slot rebuild.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CursorFrameDims {
    capture_w: u32,
    capture_h: u32,
    encode_w: u32,
    encode_h: u32,
}

/// Dedup + debounce state carried across cursor-pump ticks. Pulled
/// out so the decision logic in [`cursor_tick`] is a pure function
/// of `(state, source, dims) -> effects` and can be unit-tested
/// without the async runtime + Connection wired in.
#[derive(Default)]
struct CursorPumpState {
    seen_ids: std::collections::HashSet<u64>,
    last_pos: Option<tether_capture::CursorPosition>,
    positions_sent: u64,
    shapes_sent: u64,
}

/// What one cursor-pump tick decided to send. Pure data; the async
/// loop turns each variant into a wire call.
#[derive(Debug, PartialEq)]
enum CursorEffect {
    /// New sprite the client hasn't cached — deposit pixels.
    Shape(ControlMessage),
    /// Activate an already-cached id (or the just-deposited one).
    UseShape(u64),
    /// Latest-wins position datagram.
    Position(tether_protocol::cursor::HostCursorPacket),
}

/// Step the cursor-pump state once. Drains every buffered shape
/// event from `source`, emits a deduped `Shape`/`UseShape` pair per
/// new id (just `UseShape` for repeats), then emits a single
/// `Position` if the latest snapshot differs from what was sent
/// last tick.
///
/// When `dims` is `Some`, both the shape and the position are
/// rescaled from capture-pixel space into encode-pixel space (Linux
/// has capture==encode so callers there pass `None`; macOS Retina
/// can run capture/encode ~2.7×). Re-hashing inside
/// `rescale_shape_to_frame` keeps the wire id tied to the rescaled
/// pixels so a viewport change can't surface a stale cached bitmap.
///
/// Pure function — no I/O, no clock side effects beyond `t_capture`
/// on the position packet. The `now` parameter is injected so tests
/// can pin it.
// Cursor positions are rescaled into encode-pixel space; the round-then-cast
// to i32 is intentional and bounded by realistic screen dimensions.
#[allow(clippy::cast_possible_truncation)]
fn cursor_tick(
    state: &mut CursorPumpState,
    source: &mut dyn CursorSource,
    dims: Option<CursorFrameDims>,
    now: MonoNanos,
) -> Vec<CursorEffect> {
    let mut effects = Vec::new();
    loop {
        match source.next_event() {
            CursorEvent::Idle => break,
            CursorEvent::Shape(mut shape) => {
                if let Some(frame) = dims {
                    shape = tether_capture::rescale_shape_to_frame(
                        shape,
                        (frame.capture_w, frame.capture_h),
                        (frame.encode_w, frame.encode_h),
                    );
                }
                let id = shape.id;
                if state.seen_ids.insert(id) {
                    effects.push(CursorEffect::Shape(ControlMessage::CursorShape {
                        id,
                        hotspot: shape.hotspot,
                        width: shape.width,
                        height: shape.height,
                        format: shape.format,
                        pixels: shape.pixels,
                    }));
                    // Counts *distinct sprites deposited* on the wire,
                    // not Shape events processed. Repeat-id touches
                    // emit only `UseShape` and don't bump this.
                    state.shapes_sent += 1;
                }
                effects.push(CursorEffect::UseShape(id));
            }
        }
    }
    if let Some(pos) = source.poll_position() {
        if state.last_pos != Some(pos) {
            state.last_pos = Some(pos);
            // Rescale position into encode-pixel space (same reason
            // as the sprite). Per-axis ratios because
            // `encode_dims_for_viewport` happens to preserve aspect
            // today (rx ≈ ry) but that's not a wire contract the
            // client enforces — scaling per-axis stays correct if a
            // future path drops uniform-scale.
            let (px, py) = match dims {
                Some(frame) if frame.capture_w > 0 && frame.capture_h > 0 => {
                    let rx = f64::from(frame.encode_w) / f64::from(frame.capture_w);
                    let ry = f64::from(frame.encode_h) / f64::from(frame.capture_h);
                    (
                        (f64::from(pos.x) * rx).round() as i32,
                        (f64::from(pos.y) * ry).round() as i32,
                    )
                }
                _ => (pos.x, pos.y),
            };
            effects.push(CursorEffect::Position(
                tether_protocol::cursor::HostCursorPacket::Position {
                    t_capture: now,
                    x: px,
                    y: py,
                    visible: pos.visible,
                },
            ));
            state.positions_sent += 1;
        }
    }
    effects
}

/// Classify a control-send error so the cursor pump can distinguish
/// "connection is gone, give up" from "this one message couldn't be
/// serialized/sent, skip it." Treating every error as fatal would
/// collapse the entire session over a single oversize cursor sprite
/// (see `MAX_FRAMED_MESSAGE` — macOS can return cursors larger than
/// the control-stream cap when the user has the accessibility cursor
/// size enlarged).
fn is_fatal_send_error(err: &tether_transport::TransportError) -> bool {
    use tether_transport::TransportError as E;
    match err {
        // Local errors — connection is fine, just this message failed.
        E::FrameTooLarge { .. } | E::Codec(_) => false,
        // Anything that touches the QUIC connection state is fatal —
        // no point trying further sends.
        _ => true,
    }
}

async fn pump_cursor(
    conn: Arc<Connection>,
    mut source: Box<dyn CursorSource>,
    mut cursor_frame_rx: tokio::sync::watch::Receiver<Option<CursorFrameDims>>,
) {
    info!(
        "cursor pump started; waiting for first encoder slot rebuild to learn capture/encode dims"
    );
    // Block until the encode loop publishes its first dims. Sending
    // anything before that races against the encoder setup: the
    // cursor source produces in capture-pixel space, the client
    // renders against decoded-video pixel space, and without the
    // dims we'd emit at the wrong scale. The very first sprite would
    // then poison the client's shape cache (see id-derived-from-
    // pixels in `cursor::rescale_shape_to_frame`) — `CursorUseShape`
    // dedup would keep referring to the un-rescaled bitmap forever.
    if cursor_frame_rx.borrow().is_none() {
        if let Err(e) = cursor_frame_rx.changed().await {
            warn!(error = ?e, "cursor frame-dims watch closed before first publish; ending cursor pump");
            return;
        }
    }
    info!(dims = ?*cursor_frame_rx.borrow(), "cursor pump unblocked with encoder dims");
    let mut state = CursorPumpState::default();
    let mut last_log = std::time::Instant::now();
    // 120 Hz is the upper bound a typical desktop generates pointer
    // motion at; sleeping for one tick collapses any sub-tick
    // updates into a single latest-wins datagram, which is exactly
    // what the unreliable channel wants.
    let mut tick = tokio::time::interval(std::time::Duration::from_millis(8));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tick.tick().await;
        let dims = *cursor_frame_rx.borrow();
        let effects = cursor_tick(&mut state, source.as_mut(), dims, MonoNanos::now());
        for effect in effects {
            match effect {
                CursorEffect::Shape(msg) => {
                    let (id, w, h, n) = match &msg {
                        ControlMessage::CursorShape {
                            id,
                            width,
                            height,
                            pixels,
                            ..
                        } => (*id, *width, *height, pixels.len()),
                        _ => unreachable!("CursorEffect::Shape always carries CursorShape"),
                    };
                    match conn.send_control(&msg).await {
                        Ok(()) => debug!(id, w, h, bytes = n, "sent CursorShape"),
                        Err(e) if is_fatal_send_error(&e) => {
                            warn!(error = ?e, id, "CursorShape send failed on a fatal connection error; ending cursor pump");
                            return;
                        }
                        Err(e) => {
                            // Local serialization error (e.g. sprite
                            // larger than the control-stream cap).
                            // Drop the shape from `seen_ids` so a
                            // future re-fetch can try again, but
                            // keep the pump alive — collapsing the
                            // session over one weird cursor would
                            // be way worse than missing it.
                            warn!(error = ?e, id, w, h, bytes = n, "CursorShape send failed (non-fatal); skipping shape and continuing");
                            state.seen_ids.remove(&id);
                            // `continue` skips this tick's paired
                            // `CursorEffect::UseShape(id)` (cursor_tick
                            // always pushes them together) — sending it
                            // before the client has the bitmap would
                            // reference an empty cache slot. Next tick
                            // re-emits both because `seen_ids` no longer
                            // contains the id.
                            continue;
                        }
                    }
                }
                CursorEffect::UseShape(id) => {
                    match conn
                        .send_control(&ControlMessage::CursorUseShape { id })
                        .await
                    {
                        Ok(()) => {}
                        Err(e) if is_fatal_send_error(&e) => {
                            warn!(error = ?e, id, "CursorUseShape send failed on a fatal connection error; ending cursor pump");
                            return;
                        }
                        Err(e) => {
                            warn!(error = ?e, id, "CursorUseShape send failed (non-fatal); continuing");
                        }
                    }
                }
                CursorEffect::Position(pkt) => {
                    if let Err(e) = conn.send_datagram(&Datagram::HostCursor(pkt)) {
                        warn!(error = ?e, "HostCursor datagram send failed; ending cursor pump");
                        return;
                    }
                }
            }
        }
        if last_log.elapsed() >= std::time::Duration::from_secs(2) {
            info!(
                positions_sent = state.positions_sent,
                shapes_sent = state.shapes_sent,
                seen_shape_ids = state.seen_ids.len(),
                "cursor pump stats"
            );
            last_log = std::time::Instant::now();
        }
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod cursor_pump_tests {
    use super::*;
    use tether_capture::{CursorPosition, CursorShapeEvent};
    use tether_protocol::cursor::{CursorPixelFormat, HostCursorPacket};

    /// Scripted source: pops `events` FIFO on each `next_event` call,
    /// returns `position` on every `poll_position`. Lets us drive
    /// `cursor_tick` without a real PipeWire stream.
    struct ScriptedSource {
        events: std::collections::VecDeque<CursorEvent>,
        position: Option<CursorPosition>,
    }
    impl CursorSource for ScriptedSource {
        fn next_event(&mut self) -> CursorEvent {
            self.events.pop_front().unwrap_or(CursorEvent::Idle)
        }
        fn poll_position(&mut self) -> Option<CursorPosition> {
            self.position
        }
    }

    fn shape(id: u64) -> CursorShapeEvent {
        CursorShapeEvent {
            id,
            width: 16,
            height: 16,
            hotspot: (1, 2),
            format: CursorPixelFormat::Rgba8,
            pixels: vec![0xAB; 16 * 16 * 4],
        }
    }

    fn now() -> MonoNanos {
        MonoNanos(1)
    }

    #[test]
    fn idle_tick_emits_nothing() {
        let mut state = CursorPumpState::default();
        let mut src = ScriptedSource {
            events: Default::default(),
            position: None,
        };
        assert!(cursor_tick(&mut state, &mut src, None, now()).is_empty());
    }

    #[test]
    fn new_shape_emits_shape_then_use_shape() {
        let mut state = CursorPumpState::default();
        let mut src = ScriptedSource {
            events: vec![CursorEvent::Shape(shape(42))].into(),
            position: None,
        };
        let effects = cursor_tick(&mut state, &mut src, None, now());
        assert_eq!(effects.len(), 2);
        assert!(matches!(
            effects[0],
            CursorEffect::Shape(ControlMessage::CursorShape { id: 42, .. })
        ));
        assert_eq!(effects[1], CursorEffect::UseShape(42));
        assert!(state.seen_ids.contains(&42));
        assert_eq!(state.shapes_sent, 1);
    }

    #[test]
    fn repeat_shape_emits_use_shape_only() {
        let mut state = CursorPumpState::default();
        state.seen_ids.insert(42);
        let mut src = ScriptedSource {
            events: vec![CursorEvent::Shape(shape(42))].into(),
            position: None,
        };
        let effects = cursor_tick(&mut state, &mut src, None, now());
        assert_eq!(effects, vec![CursorEffect::UseShape(42)]);
    }

    #[test]
    fn moved_position_emits_datagram() {
        let mut state = CursorPumpState::default();
        let mut src = ScriptedSource {
            events: Default::default(),
            position: Some(CursorPosition {
                x: 100,
                y: 200,
                visible: true,
            }),
        };
        let effects = cursor_tick(&mut state, &mut src, None, now());
        assert_eq!(effects.len(), 1);
        match &effects[0] {
            CursorEffect::Position(HostCursorPacket::Position { x, y, visible, .. }) => {
                assert_eq!(*x, 100);
                assert_eq!(*y, 200);
                assert!(*visible);
            }
            other => panic!("expected Position, got {other:?}"),
        }
        assert_eq!(state.positions_sent, 1);
    }

    #[test]
    fn unchanged_position_emits_nothing_second_tick() {
        let mut state = CursorPumpState::default();
        let pos = CursorPosition {
            x: 50,
            y: 60,
            visible: true,
        };
        let mut src = ScriptedSource {
            events: Default::default(),
            position: Some(pos),
        };
        // First tick: position changed (from None to Some) → emit.
        let first = cursor_tick(&mut state, &mut src, None, now());
        assert_eq!(first.len(), 1);
        // Second tick with the same snapshot: debounce → no emit.
        let second = cursor_tick(&mut state, &mut src, None, now());
        assert!(
            second.is_empty(),
            "identical position must not produce a second datagram (got {second:?})"
        );
        assert_eq!(state.positions_sent, 1);
    }

    #[test]
    fn visibility_flip_emits_new_datagram_even_at_same_xy() {
        let mut state = CursorPumpState::default();
        state.last_pos = Some(CursorPosition {
            x: 50,
            y: 60,
            visible: true,
        });
        state.positions_sent = 1;
        let mut src = ScriptedSource {
            events: Default::default(),
            position: Some(CursorPosition {
                x: 50,
                y: 60,
                visible: false,
            }),
        };
        let effects = cursor_tick(&mut state, &mut src, None, now());
        assert_eq!(effects.len(), 1, "visibility flip must emit a datagram");
        let CursorEffect::Position(HostCursorPacket::Position { visible, .. }) = effects[0] else {
            panic!("expected Position effect");
        };
        assert!(!visible);
    }

    #[test]
    fn multiple_distinct_shapes_in_one_tick_each_get_shape_pair() {
        let mut state = CursorPumpState::default();
        let mut src = ScriptedSource {
            events: vec![
                CursorEvent::Shape(shape(1)),
                CursorEvent::Shape(shape(2)),
                CursorEvent::Shape(shape(1)), // repeat: just UseShape
            ]
            .into(),
            position: None,
        };
        let effects = cursor_tick(&mut state, &mut src, None, now());
        // 1 → Shape + UseShape; 2 → Shape + UseShape; 1 again → UseShape only
        assert_eq!(effects.len(), 5);
        // shapes_sent counts distinct sprites deposited, not events
        // processed — the repeat of id 1 emits UseShape only and does
        // not bump the counter.
        assert_eq!(state.shapes_sent, 2);
        assert_eq!(state.seen_ids.len(), 2);
    }
}

/// Visible stream dimensions are kept even for 4:2:0 luma/chroma
/// geometry. Hardware paths that need coarser coded or pitch alignment
/// should pad/crop internally instead of bending the displayed aspect
/// ratio.
const VISIBLE_DIM_ALIGN: u32 = 2;

/// Decide whether to (re)build the encoder this iteration. The first build and
/// a capture-source resolution change always rebuild (not client-controlled). A
/// viewport-driven encode-dim change is *throttled*: deferred when it lands
/// within [`VIEWPORT_REBUILD_THROTTLE`] of the previous rebuild. This is the
/// server-side defense against a client that doesn't debounce resizes — it
/// bounds encoder-teardown + `stream_epoch` churn without delaying a single
/// legitimate resize (the prior rebuild is seeded far in the past).
fn should_recreate_encoder(
    first_build: bool,
    capture_changed: bool,
    encode_changed: bool,
    elapsed_since_rebuild: std::time::Duration,
) -> bool {
    if first_build || capture_changed {
        return true;
    }
    // Only a viewport-driven encode-dim change remains; throttle it.
    encode_changed && elapsed_since_rebuild >= VIEWPORT_REBUILD_THROTTLE
}

/// Compute the encoder-output dimensions for a given capture size
/// and client viewport. The viewport bounds the longest edge; we
/// preserve aspect ratio (letterbox at the client; never stretch)
/// and clamp viewport-driven outputs to [`VISIBLE_DIM_ALIGN`]. `None`
/// returns the capture dims unchanged.
// Scaled dimensions are non-negative (scale clamped to <= 1.0, dims positive)
// and far below u32::MAX; the round-then-cast is intentional.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
    // Fit capture inside viewport at fixed aspect ratio. Never upscale
    // — the client renderer can scale up for cheap; we're not paying
    // the encoder cost to do it host-side.
    let scale_w = f64::from(v.width) / f64::from(capture_w);
    let scale_h = f64::from(v.height) / f64::from(capture_h);
    if scale_w >= 1.0 && scale_h >= 1.0 {
        return (capture_w, capture_h);
    }

    if scale_w <= scale_h {
        let aligned_w = align_floor(v.width.min(capture_w)).max(VISIBLE_DIM_ALIGN);
        let exact_h = f64::from(aligned_w) * f64::from(capture_h) / f64::from(capture_w);
        let aligned_h = align_nearest_within(exact_h, v.height.min(capture_h));
        (aligned_w, aligned_h)
    } else {
        let aligned_h = align_floor(v.height.min(capture_h)).max(VISIBLE_DIM_ALIGN);
        let exact_w = f64::from(aligned_h) * f64::from(capture_w) / f64::from(capture_h);
        let aligned_w = align_nearest_within(exact_w, v.width.min(capture_w));
        (aligned_w, aligned_h)
    }
}

fn align_floor(value: u32) -> u32 {
    (value / VISIBLE_DIM_ALIGN) * VISIBLE_DIM_ALIGN
}

// Alignment can make the exact aspect-preserving dimension impossible.
// Pick the nearest aligned dimension that still fits the viewport/source cap.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn align_nearest_within(exact: f64, limit: u32) -> u32 {
    let floor = align_floor(exact.floor() as u32);
    let ceil = floor.saturating_add(VISIBLE_DIM_ALIGN);
    if ceil <= limit && (f64::from(ceil) - exact).abs() < (exact - f64::from(floor)).abs() {
        ceil.max(VISIBLE_DIM_ALIGN)
    } else {
        floor.max(VISIBLE_DIM_ALIGN)
    }
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
// Sample coordinates are floored/clamped to >= 0 and bounded by source dims;
// output samples are clamped to [0, 255] before the u8 cast. All casts intentional.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn resize_bgra_bilinear(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
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
// rtt_ms is a logging-only value; RTT in milliseconds never approaches u64::MAX.
#[allow(clippy::cast_possible_truncation)]
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
        client_fragment_loss_events: stats.fragment_loss_events,
        client_incomplete_frames: stats.incomplete_frames,
    };
    abr.last_quinn = quinn;
    abr.last_observed_at = now;
    let decision = abr.controller.observe(dt, sample);
    if decision.target_kbps != abr.last_applied_kbps {
        match slot.encoder.set_bitrate_kbps(decision.target_kbps) {
            Ok(()) => {
                info!(
                    from_kbps = abr.last_applied_kbps,
                    to_kbps = decision.target_kbps,
                    rtt_ms = sample.rtt.as_millis() as u64,
                    congestion_events_delta = sample.congestion_events_delta,
                    lost_packets_delta = sample.lost_packets_delta,
                    client_fragment_loss_events = sample.client_fragment_loss_events,
                    client_incomplete_frames = sample.client_incomplete_frames,
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

/// Host-local per-stage latency accumulator. Complements
/// `EncodeStatsWindow::avg_encode_ms` (the encode stage) with the two
/// stages it doesn't see: the capture→dequeue handoff age (time a frame
/// waited between capture and the encode loop picking it up) and the
/// fragment+send duration. Averaged over the stats window and logged
/// beside `avg_encode_ms`, this shows which stage owns end-to-end host
/// latency — the breakdown the native-encode pass needs to know whether
/// the FFmpeg async floor or something else dominates.
#[derive(Default)]
struct StageLatency {
    frames: u64,
    capture_age_ns: u64,
    send_ns: u64,
    min_capture_age_ns: u64,
    max_capture_age_ns: u64,
    min_send_ns: u64,
    max_send_ns: u64,
}

impl StageLatency {
    fn record(&mut self, capture_age_ns: u64, send_ns: u64) {
        if self.frames == 0 {
            self.min_capture_age_ns = capture_age_ns;
            self.min_send_ns = send_ns;
        } else {
            self.min_capture_age_ns = self.min_capture_age_ns.min(capture_age_ns);
            self.min_send_ns = self.min_send_ns.min(send_ns);
        }
        self.frames += 1;
        self.capture_age_ns += capture_age_ns;
        self.send_ns += send_ns;
        self.max_capture_age_ns = self.max_capture_age_ns.max(capture_age_ns);
        self.max_send_ns = self.max_send_ns.max(send_ns);
    }

    /// Mean of `sum_ns` over recorded frames, in milliseconds.
    fn avg_ms(sum_ns: u64, frames: u64) -> f64 {
        if frames == 0 {
            0.0
        } else {
            sum_ns as f64 / frames as f64 / 1_000_000.0
        }
    }

    fn ns_to_ms(ns: u64, frames: u64) -> f64 {
        if frames == 0 {
            0.0
        } else {
            ns as f64 / 1_000_000.0
        }
    }
}

/// Capture system audio, Opus-encode it, and send each frame as an unreliable
/// `Datagram::Audio`. Runs on its own thread; returns when `shutdown` is set,
/// the capture backend disconnects, or the connection dies. Audio is dropped
/// (not buffered) until `audio_ready` — the client's `StreamReady.audio` — is
/// set, so we never stream into a client without playback.
fn run_audio_capture_and_send(
    conn: Arc<Connection>,
    shutdown: Arc<AtomicBool>,
    audio_ready: Arc<AtomicBool>,
    summary: Arc<SessionSummaryState>,
) {
    let opus_cfg = tether_audio::OpusConfig::default();
    let capture = match tether_audio::capture::start(opus_cfg) {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "audio capture unavailable; session will be silent");
            return;
        }
    };
    let mut encoder = match tether_audio::OpusEncoder::new(opus_cfg) {
        Ok(e) => e,
        Err(e) => {
            warn!(error = %e, "opus encoder init failed; session will be silent");
            return;
        }
    };

    // RFC-2198-style RED: attach the previous frames' Opus payloads to each
    // datagram so the client can recover lost packets without a concealment
    // click. Audio has no transport FEC, and Opus in-band FEC doesn't fit (LBRR
    // is SILK-only; our CELT-only config emits none), so this is the loss lever
    // that fits. Depth 2 covers a 2-frame burst — the common short Wi-Fi
    // dropout — at zero added latency (RED recovers retroactively) for ~3× the
    // tiny audio bitrate. (Comparable to Moonlight's 4+2 audio FEC.) Its history
    // carries old payloads, so it must be reset alongside the encoder if a
    // stream_epoch bump is ever wired (else the new epoch's first datagrams ship
    // cross-epoch copies).
    const AUDIO_REDUNDANCY_DEPTH: usize = 2;
    let mut redundancy = tether_audio::RedundancyBuffer::new(AUDIO_REDUNDANCY_DEPTH);

    let mut frame_seq: u32 = 0;
    // Capture-side health, logged every interval: how many frames the backend
    // delivered and the peak |sample| among them. A peak of ~0 means the
    // backend is handing us silence (e.g. a PipeWire sink monitor attached to
    // the wrong/idle sink) even though buffers are flowing — the decisive
    // signal for "host plays sound but the client hears nothing".
    const AUDIO_STATS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);
    let mut last_stats_log = std::time::Instant::now();
    let mut peak: f32 = 0.0;
    let mut frames_captured: u64 = 0;
    loop {
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        let frame = match capture
            .rx
            .recv_timeout(std::time::Duration::from_millis(100))
        {
            Ok(f) => f,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                warn!("audio capture channel disconnected; ending audio sender");
                break;
            }
        };
        frames_captured += 1;
        summary.audio.capture_frames.fetch_add(1, Ordering::Relaxed);
        for &s in &frame.samples {
            peak = peak.max(s.abs());
        }
        if last_stats_log.elapsed() >= AUDIO_STATS_INTERVAL {
            info!(
                frames_captured,
                peak,
                gated = !audio_ready.load(Ordering::Acquire),
                "audio capture stats"
            );
            last_stats_log = std::time::Instant::now();
            peak = 0.0;
            frames_captured = 0;
        }
        // Drop until the client is ready to play; capturing meanwhile keeps the
        // pipeline warm so audio starts promptly once the gate opens.
        if !audio_ready.load(Ordering::Acquire) {
            continue;
        }
        let packets = match encoder.encode(&frame.samples) {
            Ok(p) => p,
            Err(e) => {
                warn!(error = %e, "opus encode failed; dropping audio frame");
                continue;
            }
        };
        for payload in packets {
            // The RED tail must be built in send order (it records each payload
            // for the following frames), so compute it before the move below.
            // Both `payload` and the tail are `Bytes`, so they ride to the wire
            // struct with no copy.
            let redundant = redundancy.next_tail(&payload);
            let packet = AudioPacket::Opus {
                // Audio runs a single encoder for the whole session (no
                // mid-stream sample-rate/device switch today), so the epoch is
                // constant. If a device/rate switch is ever added, bump this on
                // the restart AND reset the client decoder for the new epoch in
                // the same change — don't pre-wire one half.
                stream_epoch: 0,
                frame_seq,
                t_capture: MonoNanos::now(),
                payload,
                redundant,
            };
            frame_seq = frame_seq.wrapping_add(1);
            if let Err(e) = conn.send_datagram(&Datagram::Audio(packet)) {
                if e.is_transient_send() {
                    // MTU shrank mid-stream (PLPMTUD / path change): drop this
                    // audio packet and keep the stream alive, same as the video
                    // send loop. Audio is unreliable anyway; a single dropped
                    // packet is concealed by the client's PLC.
                    tracing::debug!(error = ?e, "dropping audio packet on transient send error");
                    continue;
                }
                if shutdown.load(Ordering::Acquire) {
                    info!(error = ?e, "audio sender stopped during session shutdown");
                    return;
                }
                warn!(error = ?e, "audio datagram send failed; ending audio sender");
                return;
            }
            summary.audio.packets_sent.fetch_add(1, Ordering::Relaxed);
        }
    }
    capture.stop();
}

// The capture/send loop wires together independently-owned channels, signals,
// and shared flags; bundling them into a struct would only obscure the seam.
// The as_nanos()->u64 latency cast is logging-only and never overflows.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn run_capture_and_send(
    conn: Arc<Connection>,
    frames: FrameReceiver,
    force_idr: tether_session::IdrSignal,
    display_dims_tx: tokio::sync::watch::Sender<Option<(u32, u32)>>,
    cursor_frame_tx: tokio::sync::watch::Sender<Option<CursorFrameDims>>,
    shutdown: Arc<AtomicBool>,
    chosen_profile: VideoProfile,
    stream_ready: Arc<AtomicBool>,
    runtime: tokio::runtime::Handle,
    latest_client_stats: LatestClientStats,
    latest_viewport: LatestViewport,
    send_exited: Arc<tokio::sync::Notify>,
    shutdown_notice_sent: Arc<AtomicBool>,
    summary: Arc<SessionSummaryState>,
) {
    // RAII: notify handle_client on *any* exit from this function —
    // including the six fatal-return sites below, a panic that's
    // caught by the JoinHandle, or the clean loop-end on
    // capture-channel disconnect. Putting this on a guard means a
    // future fatal-return site doesn't have to remember to call
    // notify itself.
    struct ExitNotifier(Arc<tokio::sync::Notify>);
    impl Drop for ExitNotifier {
        fn drop(&mut self) {
            self.0.notify_one();
        }
    }
    let _exit_notifier = ExitNotifier(send_exited);
    // FEC parity ratio applied to P-frame datagrams. 20% is the
    // default — enough to absorb single-digit packet loss without
    // a bandwidth penalty most LANs will notice. Disable by
    // setting to 0 (a future `tether.cap.fec` negotiation will let
    // the client opt-out).
    //
    // Wire-level packet pacing was tried (commit bc57420) and
    // removed: `wait_for_slot` slept on the encode/send thread, so
    // any pacing tail directly gated the next frame's capture. On
    // a 10 Mbps baseline that turned 60 fps loopback into ~38 fps
    // and inflated end-to-end latency by 20-50 ms. RustDesk bursts
    // freely and relies on encoder bitrate as the only throttle;
    // Apollo paces but on a decoupled `videoBroadcastThread` with
    // a queue between encoder and sender. Re-adding wire pacing
    // would need that decoupling, not the in-line sleep we had.
    const FEC_PERCENTAGE: u8 = 20;
    let mut fragmenter = FrameFragmenter::new_with_fec(VideoStreamId(0), FEC_PERCENTAGE);
    let mut stats = tether_session::EncodeStatsWindow::new(std::time::Duration::from_secs(2));
    // Frames sacrificed because a datagram send failed transiently (the path
    // MTU shrank mid-frame). Reset each stats window and logged alongside the
    // per-window rates, so a flapping path shows up as recent drops rather than
    // a silent monotonic total.
    let mut transient_send_drops: u64 = 0;
    let mut datagrams_sent: u64 = 0;
    let mut parity_datagrams_sent: u64 = 0;
    let mut max_datagrams_per_frame: u64 = 0;
    let mut max_frame_bytes: u64 = 0;
    let mut max_keyframe_bytes: u64 = 0;
    let mut forced_idr_misses: u64 = 0;
    // Per-stage latency (handoff + send), averaged over the same window
    // as `stats` and logged alongside it. See [`StageLatency`].
    let mut stage_latency = StageLatency::default();
    let mut slot: Option<EncoderSlot> = None;
    // When the encoder was last rebuilt, for the viewport-rebuild throttle
    // (`VIEWPORT_REBUILD_THROTTLE`). Seeded in the past so the first
    // viewport-driven rebuild after startup is never throttled.
    let mut last_encoder_rebuild = Instant::now() - VIEWPORT_REBUILD_THROTTLE;
    #[cfg(target_os = "macos")]
    let mut macos_gpu_state: Option<MacosGpuState> = None;
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
        let (k, u) = frame.timestamps();
        // Handoff age: how long this frame waited between capture and the
        // encode loop dequeuing it. With the single-slot drop-oldest
        // mailbox this should stay near zero; a rising value means the
        // encoder can't keep up and frames are queuing (or being evicted).
        let capture_age_ns = MonoNanos::now().0.saturating_sub(u.0);
        let mut timing = HostFrameTimingBuilder::captured(k, u);

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
            let g = lock_host_state(&latest_viewport, "latest viewport");
            (g.viewport, g.seq)
        };
        // Viewport-driven downscale applies per frame source:
        //   - CPU frames: `resize_bgra_bilinear` does the work below.
        //   - Linux GPU dma-buf: `tether-scaler` (Mitchell-Netravali
        //     in linear-light) runs inside `encode_gpu_frame` between
        //     PipeWire's BGRA import and the chroma bridge.
        //   - macOS GPU IOSurface: capture full-res BGRA from SCK,
        //     then `BgraIOSurfaceBridge` runs MPS Lanczos before
        //     converting into VT-ready NV12/P010/NV24/P410-family
        //     IOSurfaces for the negotiated profile.
        let (encode_width, encode_height) =
            encode_dims_for_viewport(frame_width, frame_height, current_viewport);
        // viewport_seq isn't part of the rebuild check: only an actual
        // change in encoder dimensions warrants tearing down the
        // encoder + bumping stream_epoch. A viewport change that
        // leaves dims unchanged (GPU path while no scaler exists; an
        // upscale-rejected viewport on any path) shouldn't churn the
        // session. The seq is tracked for diagnostic logging only.
        let first_build = slot.is_none();
        let capture_changed = slot
            .as_ref()
            .is_some_and(|s| s.capture_width != frame_width || s.capture_height != frame_height);
        let encode_changed = slot
            .as_ref()
            .is_some_and(|s| s.width != encode_width || s.height != encode_height);
        // Throttle viewport-driven rebuilds (server-side defense); while
        // deferred, encoding continues at the current dims (the client
        // letterboxes) and a later iteration rebuilds to the newest viewport.
        let needs_recreate = should_recreate_encoder(
            first_build,
            capture_changed,
            encode_changed,
            Instant::now().duration_since(last_encoder_rebuild),
        );

        if needs_recreate {
            last_encoder_rebuild = Instant::now();
            // Drop the previous encoder BEFORE constructing its
            // replacement. `slot.take()` moves the old `EncoderSlot` out
            // (slot = None) and it drops at the end of this block —
            // releasing its hardware session and D3D11 frame pool before
            // the new encoder allocates its own. Required for QSV: two
            // live QSV sessions on one D3D11 device fail child-texture
            // creation with DXGI_ERROR_INVALID_CALL.
            if let Some(old) = slot.take() {
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
            let _ = cursor_frame_tx.send(Some(CursorFrameDims {
                capture_w: frame_width,
                capture_h: frame_height,
                encode_w: encode_width,
                encode_h: encode_height,
            }));
            // Single-element preference list: the handshake already
            // picked one codec, and a mid-session codec switch would
            // require coordinated client decoder rebuild. We pass the
            // list-form for API symmetry with the initial handshake
            // probe; per-resize cost is one construction attempt.
            let baseline_kbps =
                derive_bitrate_kbps(chosen_profile, encode_width, encode_height, ENCODER_FPS);
            #[cfg(target_os = "windows")]
            let encoder_result = {
                // Real capture stores the shared DXGI device in
                // SHARED_D3D11_DEVICE (`real_capture`), and the encoder
                // reuses it for a zero-copy VP blit. Test-pattern mode
                // never runs `real_capture`, so no shared device exists —
                // fall back to null device pointers (FFmpeg self-creates a
                // d3d11va device, as on the probe path) and vendor 0,
                // which selects the vendor-agnostic Media Foundation
                // encoder. The CPU test-pattern frames take the
                // `encode_bgra` path either way, so this stays correct.
                let (dev_ptr, ctx_ptr, vendor_id) = match SHARED_D3D11_DEVICE.get() {
                    Some(dev) => {
                        let (dev_ptr, ctx_ptr) = dev.device_ptrs();
                        (dev_ptr, ctx_ptr, dev.vendor_id)
                    }
                    None => (std::ptr::null_mut(), std::ptr::null_mut(), 0),
                };
                build_encoder_d3d11(
                    chosen_profile,
                    encode_width,
                    encode_height,
                    ENCODER_FPS,
                    baseline_kbps,
                    dev_ptr,
                    ctx_ptr,
                    vendor_id,
                )
            };
            #[cfg(not(target_os = "windows"))]
            let encoder_result = build_encoder(
                chosen_profile,
                encode_width,
                encode_height,
                ENCODER_FPS,
                baseline_kbps,
            );
            slot = match encoder_result {
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
                        controller: AbrController::new(AbrConfig::new(baseline_kbps)),
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
                    if abr.is_some() && AbrConfig::new(baseline_kbps).floor_kbps == baseline_kbps {
                        info!(
                            baseline_kbps,
                            "ABR enabled but floor == baseline; bitrate control range is zero \
                             (encode dims too small for the configured floor)"
                        );
                    }
                    #[cfg(target_os = "macos")]
                    let iosurface_encode_fourcc =
                        macos_iosurface_encode_fourcc(chosen_profile).unwrap_or(0);
                    Some(EncoderSlot {
                        encoder: e,
                        capture_width: frame_width,
                        capture_height: frame_height,
                        width: encode_width,
                        height: encode_height,
                        abr,
                        #[cfg(target_os = "linux")]
                        bridge: BridgeState::NotYetBuilt,
                        #[cfg(target_os = "linux")]
                        scaler: None,
                        #[cfg(target_os = "linux")]
                        scaler_pipelines: None,
                        #[cfg(target_os = "macos")]
                        bgra_iosurface_bridge: None,
                        #[cfg(target_os = "macos")]
                        iosurface_encode_fourcc,
                        #[cfg(target_os = "macos")]
                        prev_pooled: None,
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
                    let reason = format!(
                        "host could not construct {:?} {:?} encoder for {}x{}: {}",
                        chosen_profile.codec, chosen_profile.chroma, encode_width, encode_height, e
                    );
                    send_goodbye_notice_blocking(
                        &runtime,
                        &conn,
                        &shutdown_notice_sent,
                        reason.as_str(),
                        GoodbyeCode::InternalError,
                        &summary,
                    );
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
        if let Some(stats) = lock_host_state(&latest_client_stats, "latest client stats").take() {
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
                    let reason = format!("host GPU encode bridge collapsed: {e}");
                    send_goodbye_notice_blocking(
                        &runtime,
                        &conn,
                        &shutdown_notice_sent,
                        reason.as_str(),
                        GoodbyeCode::InternalError,
                        &summary,
                    );
                    return;
                }
            },
            #[cfg(target_os = "macos")]
            CapturedFrame::Gpu(gpu) => {
                match encode_iosurface_frame(slot_mut, &mut macos_gpu_state, gpu, pts, force_kf) {
                    GpuEncodeOutcome::Packets(p) => p,
                    GpuEncodeOutcome::DropFrame(e) => {
                        warn!(error = %e, "IOSurface encode failed; dropping frame");
                        continue;
                    }
                    GpuEncodeOutcome::Fatal(e) => {
                        tracing::error!(
                            error = %e,
                            "IOSurface encode bridge collapsed; sending Goodbye(InternalError) and exiting send loop"
                        );
                        let reason = format!("host IOSurface encode bridge collapsed: {e}");
                        send_goodbye_notice_blocking(
                            &runtime,
                            &conn,
                            &shutdown_notice_sent,
                            reason.as_str(),
                            GoodbyeCode::InternalError,
                            &summary,
                        );
                        return;
                    }
                }
            }
            #[cfg(target_os = "windows")]
            CapturedFrame::Gpu(gpu) => {
                use windows::core::Interface;
                let tether_capture::GpuCapturedSource::D3D11Texture(ref tex) = gpu.source;
                // DXGI_FORMAT is a non-negative enum.
                #[allow(clippy::cast_sign_loss)]
                let format = tex.format.0 as u32;
                let frame = tether_codec::D3D11TextureFrame {
                    texture: tex.texture.as_raw(),
                    device: tex.device.device.as_raw(),
                    device_context: tex.device.context.as_raw(),
                    width: tex.width,
                    height: tex.height,
                    format,
                };
                match slot_mut.encoder.encode_gpu(
                    GpuEncoderFrame::D3D11Texture(&frame),
                    pts,
                    force_kf,
                ) {
                    Ok(p) => p,
                    Err(tether_codec::CodecError::UnsupportedInputFormat) => {
                        tracing::error!(
                            "D3D11 GPU encode input path unavailable; sending Goodbye(InternalError) and exiting send loop"
                        );
                        let reason = "host D3D11 GPU encode input path unavailable";
                        send_goodbye_notice_blocking(
                            &runtime,
                            &conn,
                            &shutdown_notice_sent,
                            reason,
                            GoodbyeCode::InternalError,
                            &summary,
                        );
                        return;
                    }
                    Err(e) => {
                        warn!(error = %e, "D3D11 GPU encode failed; dropping frame");
                        continue;
                    }
                }
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
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
            0 => {
                if force_kf {
                    forced_idr_misses = forced_idr_misses.saturating_add(1);
                    summary
                        .video
                        .forced_idr_misses
                        .fetch_add(1, Ordering::Relaxed);
                    warn!(pts, "encoder produced no packet for a forced-IDR request");
                }
                continue;
            }
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
            if force_kf {
                forced_idr_misses = forced_idr_misses.saturating_add(1);
                summary
                    .video
                    .forced_idr_misses
                    .fetch_add(1, Ordering::Relaxed);
                warn!(
                    pts,
                    "encoder produced an empty body for a forced-IDR request"
                );
            }
            continue;
        }
        let body_len = body.len() as u64;
        if force_kf && !keyframe {
            forced_idr_misses = forced_idr_misses.saturating_add(1);
            summary
                .video
                .forced_idr_misses
                .fetch_add(1, Ordering::Relaxed);
            warn!(
                pts,
                body_len, "encoder did not emit a keyframe for a forced-IDR request"
            );
        }
        let meta = VideoFrameMeta {
            timing: timing.finish(),
            keyframe,
            input_echo: InputEchoBatch::default(),
            dimensions: (frame_width, frame_height),
        };

        let send_t0 = std::time::Instant::now();
        // All frames — IDR keyframes and P-frames alike — ride the FEC'd
        // datagram channel. A single channel keeps an IDR inherently ordered
        // ahead of the P-frames that depend on it (no cross-channel overtaking)
        // and large IDRs get multi-block FEC. Shards are sized to the
        // connection's real datagram MTU (clamped to the soft payload target)
        // minus the encoded header, so no datagram exceeds the path MTU even
        // under an input-echo burst. Burst freely — see the comment at the top
        // of `run_capture_and_send` for why wire-level pacing was removed.
        let budget = conn
            .max_datagram_size()
            .map_or(tether_protocol::MAX_DATAGRAM_PAYLOAD, |m| {
                m.min(tether_protocol::MAX_DATAGRAM_PAYLOAD)
            });
        let mut sent_frame = true;
        let packets = fragmenter.fragment(meta, body, budget);
        let planned_frame_datagrams = u64::try_from(packets.len()).unwrap_or(u64::MAX);
        for packet in packets {
            let is_parity = matches!(&packet, VideoPacket::Parity { .. });
            if let Err(e) = conn.send_datagram(&Datagram::Video(packet)) {
                if e.is_transient_send() {
                    // MTU shrank mid-frame (PLPMTUD black-hole / path change),
                    // so a shard fragmented against the old budget no longer
                    // fits. Every remaining shard of this frame is the same
                    // size, so drop the rest of the frame and keep the session
                    // alive — the next frame re-fragments against the new MTU.
                    // FEC can't rebuild a lost contiguous tail, so this frame is
                    // sacrificed, but a single dropped frame is recoverable and
                    // a torn-down session is not.
                    transient_send_drops += 1;
                    summary
                        .video
                        .transient_send_drop_frames
                        .fetch_add(1, Ordering::Relaxed);
                    tracing::debug!(error = ?e, "dropping frame on transient datagram send error");
                    sent_frame = false;
                    break;
                }
                if shutdown.load(Ordering::Acquire) || shutdown_notice_sent.load(Ordering::Acquire)
                {
                    info!(error = ?e, "video send loop stopped during session shutdown");
                    return;
                }
                let reason = format!("host video datagram send failed: {e:?}");
                warn!(error = ?e, "send_datagram failed (fatal), sending Goodbye(InternalError)");
                send_goodbye_notice_blocking(
                    &runtime,
                    &conn,
                    &shutdown_notice_sent,
                    reason.as_str(),
                    GoodbyeCode::InternalError,
                    &summary,
                );
                return;
            }
            datagrams_sent = datagrams_sent.saturating_add(1);
            summary.video.datagrams_sent.fetch_add(1, Ordering::Relaxed);
            if is_parity {
                parity_datagrams_sent = parity_datagrams_sent.saturating_add(1);
                summary
                    .video
                    .parity_datagrams_sent
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        if sent_frame {
            max_datagrams_per_frame = max_datagrams_per_frame.max(planned_frame_datagrams);
            max_frame_bytes = max_frame_bytes.max(body_len);
            summary.video.frames_sent.fetch_add(1, Ordering::Relaxed);
            summary
                .video
                .bytes_sent
                .fetch_add(body_len, Ordering::Relaxed);
            summary
                .video
                .max_datagrams_per_frame
                .fetch_max(planned_frame_datagrams, Ordering::Relaxed);
            summary
                .video
                .max_frame_bytes
                .fetch_max(body_len, Ordering::Relaxed);
            if keyframe {
                summary.video.keyframes.fetch_add(1, Ordering::Relaxed);
                max_keyframe_bytes = max_keyframe_bytes.max(body_len);
                summary
                    .video
                    .max_keyframe_bytes
                    .fetch_max(body_len, Ordering::Relaxed);
            }
            stats.record_frame(encode_delta_ns, body_len, keyframe);
            // Accumulated only for frames that complete encode + send;
            // damage-skipped, rebuild-failed, and transient-send-dropped frames
            // are excluded, so this is the handoff age of frames that actually
            // reach the wire.
            stage_latency.record(capture_age_ns, send_t0.elapsed().as_nanos() as u64);
        }

        if stats.should_emit() {
            if let Some(snap) = stats.snapshot_and_reset() {
                let kf_per_s = if snap.window_secs > 0.0 {
                    f64::from(snap.keyframe_count) / snap.window_secs
                } else {
                    0.0
                };
                // Per-stage host latency breakdown: capture_age_ms is the
                // handoff (capture→dequeue), avg_encode_ms is the encoder
                // (the FFmpeg send_frame→packet floor), send_ms is
                // fragment+wire. Their sum ≈ host-side end-to-end latency.
                info!(
                    frames = snap.frame_count,
                    avg_capture_age_ms =
                        StageLatency::avg_ms(stage_latency.capture_age_ns, stage_latency.frames),
                    min_capture_age_ms = StageLatency::ns_to_ms(
                        stage_latency.min_capture_age_ns,
                        stage_latency.frames
                    ),
                    max_capture_age_ms = StageLatency::ns_to_ms(
                        stage_latency.max_capture_age_ns,
                        stage_latency.frames
                    ),
                    avg_encode_ms = snap.avg_encode_ms,
                    min_encode_ms = snap.min_encode_ms,
                    max_encode_ms = snap.max_encode_ms,
                    avg_send_ms = StageLatency::avg_ms(stage_latency.send_ns, stage_latency.frames),
                    min_send_ms =
                        StageLatency::ns_to_ms(stage_latency.min_send_ns, stage_latency.frames),
                    max_send_ms =
                        StageLatency::ns_to_ms(stage_latency.max_send_ns, stage_latency.frames),
                    kbps_out = snap.kbps_out,
                    keyframes_per_s = kf_per_s,
                    transient_send_drop_frames = transient_send_drops,
                    datagrams_sent,
                    parity_datagrams_sent,
                    max_datagrams_per_frame,
                    max_frame_bytes = max_frame_bytes.max(snap.max_frame_bytes),
                    max_keyframe_bytes,
                    forced_idr_misses,
                    "send stats"
                );
                stage_latency = StageLatency::default();
                // Per-window like the rates above: zero so the next log shows
                // drops in that window, not a monotonic lifetime total.
                transient_send_drops = 0;
                datagrams_sent = 0;
                parity_datagrams_sent = 0;
                max_datagrams_per_frame = 0;
                max_frame_bytes = 0;
                max_keyframe_bytes = 0;
                forced_idr_misses = 0;
            }
        }
    }
    info!("send loop exiting");
}

/// Directory the host caches its self-signed cert + key in. Default is
/// `$HOME/.tether/` for release and `$HOME/.tether-dev/` for the dev channel;
/// override with `$TETHER_CERT_DIR` for testing or sharing between host
/// instances. We deliberately don't follow XDG paths — the file pair is small
/// and operationally important, and a single well-known per-channel location
/// is easier to talk about in docs than "wherever XDG_DATA_HOME points".
/// Delegates to the shared [`tether_pairing::config_dir`] so the host, the
/// client, and the Tauri shell all resolve the same location.
fn persistent_cert_dir() -> anyhow::Result<PathBuf> {
    Ok(tether_pairing::config_dir()?)
}

struct Args {
    bind: SocketAddr,
    use_test_pattern: bool,
    ipc: bool,
    /// Linux only: install the `/dev/uinput` udev rule and exit. Parsed
    /// on every platform so the flag gives a clear message rather than
    /// being mistaken for a bind address elsewhere.
    setup_input: bool,
    /// Stream system audio (Opus) alongside video. On by default; the host
    /// degrades to a silent session if no capture backend is available.
    audio: bool,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut bind: SocketAddr = "127.0.0.1:7374".parse().expect("static literal");
    let mut use_test_pattern = false;
    let mut ipc = false;
    let mut setup_input = false;
    let mut audio = true;
    for arg in std::env::args().skip(1) {
        if arg == "--test-pattern" {
            use_test_pattern = true;
        } else if arg == "--ipc" {
            ipc = true;
        } else if arg == "--setup-input" {
            setup_input = true;
        } else if arg == "--no-audio" {
            audio = false;
        } else if arg == "--help" || arg == "-h" {
            eprintln!(
                "usage: tether-host [--test-pattern] [--ipc] [--setup-input] [--no-audio] [bind_addr]\n\
                 \n\
                 --setup-input  (Linux) grant /dev/uinput access for input injection, then exit\n\
                 --no-audio     disable system-audio (Opus) streaming"
            );
            std::process::exit(0);
        } else {
            bind = arg.parse()?;
        }
    }
    Ok(Args {
        bind,
        use_test_pattern,
        ipc,
        setup_input,
        audio,
    })
}

/// In `--ipc` mode, watch stdin for shell commands. `Stop` (or stdin EOF / a
/// read error — the shell process died) trips `shutdown` so no engine is ever
/// orphaned by a crashed shell. `StartPairing` opens a pairing window and emits
/// the PIN; `RevokePeer` removes a paired client and drops any live session
/// from it. Unrecognized lines are logged and skipped.
fn spawn_stdin_command_watcher(
    shutdown: Arc<tokio::sync::Notify>,
    pairing_state: PairingState,
    reporter: Reporter,
) {
    use tokio::io::{AsyncBufReadExt, BufReader};
    tokio::spawn(async move {
        let mut lines = BufReader::new(tokio::io::stdin()).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    match tether_ipc::ShellCommand::from_line(line) {
                        Ok(tether_ipc::ShellCommand::Stop) => {
                            info!("shell sent stop");
                            shutdown.notify_one();
                            return;
                        }
                        Ok(tether_ipc::ShellCommand::StartPairing { label }) => {
                            let pin = pairing_state.open_window(label);
                            info!("pairing window opened");
                            reporter.emit(&EngineEvent::PairingPin {
                                pin,
                                expires_in_secs: pairing::PAIRING_WINDOW.as_secs(),
                            });
                        }
                        Ok(tether_ipc::ShellCommand::RevokePeer { fingerprint }) => {
                            match pairing_state.revoke(&fingerprint) {
                                Ok(removed) => {
                                    info!(removed, "revoke peer requested");
                                    // Push the updated list so the UI reflects the removal.
                                    reporter.emit(&EngineEvent::PeerList {
                                        peers: pairing_state.peer_list(),
                                    });
                                }
                                Err(e) => {
                                    warn!(error = %e, "revoke peer failed");
                                    reporter.emit(&EngineEvent::Error {
                                        message: format!("failed to revoke peer: {e}"),
                                    });
                                }
                            }
                        }
                        Ok(tether_ipc::ShellCommand::ListPeers) => {
                            reporter.emit(&EngineEvent::PeerList {
                                peers: pairing_state.peer_list(),
                            });
                        }
                        Err(e) => {
                            warn!(error = %e, line, "ignoring unrecognized stdin command");
                        }
                    }
                }
                Ok(None) => {
                    info!("stdin closed; treating as stop");
                    shutdown.notify_one();
                    return;
                }
                Err(e) => {
                    warn!(error = %e, "stdin read error; treating as stop");
                    shutdown.notify_one();
                    return;
                }
            }
        }
    });
}

async fn pick_capture_source(
    force_test_pattern: bool,
    chosen_profile: VideoProfile,
    initial_viewport: Option<Viewport>,
) -> anyhow::Result<tether_capture::CaptureHandle> {
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
    real_capture(chosen_profile, initial_viewport).await
}

#[cfg(target_os = "linux")]
async fn real_capture(
    _chosen_profile: VideoProfile,
    _initial_viewport: Option<Viewport>,
) -> anyhow::Result<tether_capture::CaptureHandle> {
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
async fn real_capture(
    _chosen_profile: VideoProfile,
    _initial_viewport: Option<Viewport>,
) -> anyhow::Result<tether_capture::CaptureHandle> {
    // Capture BGRA for every macOS host profile. The Metal bridge is
    // the single live conversion path: MPS Lanczos handles viewport
    // downscale, then the compute kernel writes the VideoToolbox input
    // fourcc for the negotiated chroma/bit-depth.
    tether_capture::macos::start(tether_capture::macos::sck_bgra_pixel_format())
        .await
        .map_err(anyhow::Error::from)
}

#[cfg(target_os = "windows")]
static SHARED_D3D11_DEVICE: std::sync::OnceLock<tether_capture::windows::D3D11Device> =
    std::sync::OnceLock::new();

#[cfg(target_os = "windows")]
static PRECREATED_CAPTURE: std::sync::Mutex<Option<tether_capture::windows::PreCreatedCapture>> =
    std::sync::Mutex::new(None);

#[cfg(target_os = "windows")]
async fn real_capture(
    _chosen_profile: VideoProfile,
    _initial_viewport: Option<Viewport>,
) -> anyhow::Result<tether_capture::CaptureHandle> {
    info!("capture source: windows (DXGI Desktop Duplication)");
    let pre = lock_host_state(&PRECREATED_CAPTURE, "precreated capture").take();
    let (handle, d3d11_device) = match pre {
        Some(p) => tether_capture::windows::start_with(p).map_err(anyhow::Error::from)?,
        None => tether_capture::windows::start_with(
            tether_capture::windows::pre_create().map_err(anyhow::Error::from)?,
        )
        .map_err(anyhow::Error::from)?,
    };
    let _ = SHARED_D3D11_DEVICE.set(d3d11_device);
    Ok(handle)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
async fn real_capture(
    _chosen_profile: VideoProfile,
    _initial_viewport: Option<Viewport>,
) -> anyhow::Result<tether_capture::CaptureHandle> {
    warn!("no real capture backend on this platform yet; falling back to test-pattern");
    Ok(tether_capture::test_pattern::start(
        TEST_PATTERN_WIDTH,
        TEST_PATTERN_HEIGHT,
        TEST_PATTERN_FPS,
    ))
}

fn init_tracing(ipc: bool) -> tracing_appender::non_blocking::WorkerGuard {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    // In IPC mode stdout is reserved for the JSON-lines protocol, so logs
    // go to stderr; the standalone CLI keeps logs on stdout as before.
    // Both branches yield the same `(NonBlocking, WorkerGuard)` type — the
    // inner writer is erased behind `non_blocking`'s channel.
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

/// Default VBR target bitrate as a function of resolution, fps,
/// codec, and chroma. Anchored at 1080p60 H.264 4:2:0 = 10 Mbps (the
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
        // AV1: ~60% of H.264 for similar visual quality (standard
        // reference number). Wired on VAAPI (Linux) and D3D11 (Windows).
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

    #[test]
    fn host_audio_summary_requires_host_and_client_readiness() {
        assert!(host_summary_audio_active(true, true));
        assert!(!host_summary_audio_active(true, false));
        assert!(!host_summary_audio_active(false, true));
        assert!(!host_summary_audio_active(false, false));
    }

    fn profile(codec: CodecKind, chroma: ChromaSubsampling, bit_depth: u8) -> VideoProfile {
        VideoProfile {
            codec,
            chroma,
            bit_depth,
        }
    }

    #[test]
    fn derive_bitrate_anchors_h264_1080p60_floor() {
        let p = profile(CodecKind::H264, ChromaSubsampling::Yuv420, 8);
        assert_eq!(derive_bitrate_kbps(p, 1920, 1080, 60), ENCODER_BITRATE_KBPS);
    }

    #[test]
    fn derive_bitrate_applies_codec_chroma_and_depth_multipliers() {
        let hevc_main = profile(CodecKind::Hevc, ChromaSubsampling::Yuv420, 8);
        let hevc_444 = profile(CodecKind::Hevc, ChromaSubsampling::Yuv444, 8);
        let hevc_444_10 = profile(CodecKind::Hevc, ChromaSubsampling::Yuv444, 10);
        let av1_10 = profile(CodecKind::Av1, ChromaSubsampling::Yuv420, 10);

        assert_eq!(derive_bitrate_kbps(hevc_main, 1920, 1080, 60), 7_000);
        assert_eq!(derive_bitrate_kbps(hevc_444, 1920, 1080, 60), 9_800);
        assert_eq!(derive_bitrate_kbps(hevc_444_10, 1920, 1080, 60), 11_760);
        assert_eq!(derive_bitrate_kbps(av1_10, 1920, 1080, 60), 7_200);
    }

    #[test]
    fn derive_bitrate_clamps_extreme_resolutions() {
        let p = profile(CodecKind::H264, ChromaSubsampling::Yuv420, 8);
        assert_eq!(derive_bitrate_kbps(p, 64, 64, 30), 500);
        assert_eq!(derive_bitrate_kbps(p, 7680, 4320, 120), 30_000);
    }

    #[test]
    fn lock_host_state_recovers_poisoned_mutex() {
        let lock = Arc::new(StdMutex::new(7u32));
        let poisoned = lock.clone();
        let result = std::panic::catch_unwind(move || {
            let mut guard = poisoned.lock().unwrap();
            *guard = 11;
            panic!("poison test lock");
        });
        assert!(result.is_err());

        let mut guard = lock_host_state(&lock, "test");
        assert_eq!(*guard, 11);
        *guard = 13;
        drop(guard);

        assert_eq!(*lock_host_state(&lock, "test"), 13);
    }

    #[test]
    fn viewport_state_skips_duplicate_updates() {
        let mut state = ViewportState::default();
        let viewport = Some(Viewport::new(1920, 1080));

        assert!(state.update_if_changed(viewport));
        assert_eq!(state.seq, 1);
        assert_eq!(state.viewport, viewport);

        assert!(!state.update_if_changed(viewport));
        assert_eq!(state.seq, 1);

        assert!(state.update_if_changed(Some(Viewport::new(1280, 720))));
        assert_eq!(state.seq, 2);

        assert!(state.update_if_changed(None));
        assert_eq!(state.seq, 3);
        assert!(!state.update_if_changed(None));
        assert_eq!(state.seq, 3);
    }

    #[test]
    fn viewport_rebuild_throttle_defers_rapid_changes_only() {
        use std::time::Duration;
        let under = VIEWPORT_REBUILD_THROTTLE / 2;
        let over = VIEWPORT_REBUILD_THROTTLE + Duration::from_millis(1);

        // First build and capture-source changes always rebuild, regardless of
        // how recent the last rebuild was — they aren't client-controlled.
        assert!(should_recreate_encoder(true, false, false, Duration::ZERO));
        assert!(should_recreate_encoder(false, true, false, under));

        // A viewport-driven encode change is throttled within the window...
        assert!(
            !should_recreate_encoder(false, false, true, under),
            "rapid viewport change within the throttle window must be deferred"
        );
        // ...and allowed once the window has passed.
        assert!(
            should_recreate_encoder(false, false, true, over),
            "viewport change past the throttle window must rebuild"
        );

        // No change at all → never rebuild.
        assert!(!should_recreate_encoder(false, false, false, over));

        // A capture change is never throttled even if an encode change rode
        // along inside the window (real source change wins).
        assert!(should_recreate_encoder(false, true, true, under));
    }

    #[test]
    fn stage_latency_averages_over_recorded_frames() {
        let mut s = StageLatency::default();
        // No frames recorded → zero, not a divide-by-zero.
        assert_eq!(StageLatency::avg_ms(s.capture_age_ns, s.frames), 0.0);

        // 1 ms + 3 ms handoff, 2 ms + 4 ms send over two frames.
        s.record(1_000_000, 2_000_000);
        s.record(3_000_000, 4_000_000);
        assert_eq!(s.frames, 2);
        assert!((StageLatency::avg_ms(s.capture_age_ns, s.frames) - 2.0).abs() < 1e-9);
        assert!((StageLatency::avg_ms(s.send_ns, s.frames) - 3.0).abs() < 1e-9);
        assert!((StageLatency::ns_to_ms(s.min_capture_age_ns, s.frames) - 1.0).abs() < 1e-9);
        assert!((StageLatency::ns_to_ms(s.max_capture_age_ns, s.frames) - 3.0).abs() < 1e-9);
        assert!((StageLatency::ns_to_ms(s.min_send_ns, s.frames) - 2.0).abs() < 1e-9);
        assert!((StageLatency::ns_to_ms(s.max_send_ns, s.frames) - 4.0).abs() < 1e-9);
    }
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
        // matches (16:9), so we expect 1280x720 exactly.
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
    fn encode_dims_viewport_scales_macos_hidpi_without_aspect_bend() {
        // Regression for the macOS host blur investigation: the old
        // 16x16 clamp turned this into 1728x1104, visibly changing
        // the source aspect ratio.
        assert_eq!(
            encode_dims_for_viewport(3024, 1952, Some(Viewport::new(1920, 1116))),
            (1728, 1116)
        );
    }

    #[test]
    fn encode_dims_rounds_to_even_chroma_safe_visible_grid() {
        // Off-aspect viewport that lands the output on odd dimensions.
        // The visible stream snaps only to a small chroma-safe grid;
        // any coarser coded alignment belongs in backend padding/crop,
        // not in the displayed aspect ratio.
        let (w, h) = encode_dims_for_viewport(1920, 1080, Some(Viewport::new(1000, 600)));
        assert_eq!(w % 2, 0);
        assert_eq!(h % 2, 0);
        assert!(w <= 1000 && h <= 600);
    }

    #[test]
    fn encode_dims_five_k_two_k_viewport_uses_nearest_even_height() {
        // 7680x3232 on a 1600-wide viewport lands at 673.33 px high.
        // Even-only snapping chooses 674, avoiding the larger aspect
        // bend introduced by a 4-pixel grid.
        assert_eq!(
            encode_dims_for_viewport(7680, 3232, Some(Viewport::new(1600, 900))),
            (1600, 674)
        );
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
    ///   * `macos_iosurface_encode_fourcc` — the fourcc the macOS
    ///     Metal bridge will deliver for a chosen `VideoProfile`.
    ///   * `tether_codec::videotoolbox::encoder::iosurface_fourcc_matches`
    ///     — the fourccs the VT encoder accepts as zero-copy input.
    ///   * `tether_codec::videotoolbox::probe::expected_iosurface_fourccs`
    ///     — the fourccs the VT decoder is allowed to emit for a
    ///     given profile (= the probe's accept set on the decode
    ///     side of the round-trip).
    ///   * `tether_render::accepts_iosurface_fourcc` — the fourccs the
    ///     renderer's IOSurface import path accepts.
    ///
    /// The full pipeline is `SCK BGRA → Metal bridge → encoder → decoder → renderer`. For
    /// each profile we negotiate, the per-link invariants are:
    ///
    ///   * **Bridge output ⊆ encoder accept** — the encoder must accept
    ///     whatever the Metal bridge produces. A miss here crashes
    ///     the first bridged frame after handshake.
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
        use tether_codec::videotoolbox::encoder::iosurface_fourcc_matches;
        use tether_codec::videotoolbox::expected_iosurface_fourccs;
        use tether_probe::PROFILE_PREFERENCE;
        use tether_render::accepts_iosurface_fourcc;

        // Walk every profile the negotiator could pick.
        for profile in PROFILE_PREFERENCE {
            let chroma = profile.chroma;
            let bd = profile.bit_depth;

            // (1) Metal bridge output ⊆ encoder + renderer accept.
            let Some(fourcc) = macos_iosurface_encode_fourcc(*profile) else {
                continue;
            };
            assert!(
                iosurface_fourcc_matches(chroma, bd, fourcc),
                "Metal bridge produces 0x{fourcc:08x} for profile {profile:?} but the VT encoder \
                 does not accept it via submit_iosurface. This would crash at first bridged frame."
            );
            assert!(
                accepts_iosurface_fourcc(chroma, bd, fourcc),
                "Metal bridge produces 0x{fourcc:08x} for profile {profile:?} but the renderer \
                 rejects that IOSurface family. The encode/decode loopback would not be displayable."
            );

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
        use tether_codec::videotoolbox::expected_iosurface_fourccs;
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
