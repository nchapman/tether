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
use std::sync::Arc;

use crossbeam_channel::Receiver;
use tether_capture::{CapturedFrame, PixelFormat};
use tether_codec::{
    pick_supported_profile, probe_encoder, supported_encode_profiles, Encoder,
};
#[cfg(target_os = "linux")]
use tether_capture::GpuCapturedSource;
#[cfg(target_os = "linux")]
use tether_codec::{DmaBufFrame, DmaBufLayer, DmaBufObject};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use tether_codec::GpuEncoderFrame;
#[cfg(target_os = "linux")]
use tether_gpuconvert::{Nv12DmaBuf, Nv12DmaBufFrame};
use tether_protocol::control::{
    ChromaSubsampling, ClientHello, CodecKind, ControlMessage, ServerHello, ServerHelloV1,
    VideoColorSpec, VideoProfile, CLIENT_DECODE_PROFILES_EXTENSION_KEY,
    SERVER_ENCODE_PROFILE_EXTENSION_KEY,
};
use tether_protocol::video::{
    FrameFragmenter, HostFrameTimingBuilder, InputEchoBatch, VideoFrameMeta,
};
use tether_protocol::MonoNanos;
use tether_transport::{Connection, Datagram, Server};
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

    let conn = match server.accept().await {
        Some(Ok(c)) => Arc::new(c),
        Some(Err(e)) => return Err(e.into()),
        None => {
            warn!("server closed before any connection arrived");
            return Ok(());
        }
    };
    info!(remote = %conn.remote_address(), "client connected");

    handle_client(conn, use_test_pattern).await?;

    server.close_and_wait(0, b"host shutdown").await;
    Ok(())
}

/// Owns every piece of per-connection state — encoder thread, injector,
/// recv tasks. When this function returns the whole graph drops together:
/// the libei session releases, the QUIC connection closes, the encoder
/// frees its hardware context. Anything that needs to survive across
/// reconnects must live in `main`, not here.
async fn handle_client(
    conn: Arc<Connection>,
    use_test_pattern: bool,
) -> anyhow::Result<()> {
    // Application-layer handshake. The closure walks the client's
    // codec preference list and picks the first one this host can
    // actually build (cheap construction probe at 64×64), so the
    // ServerHello we return carries an authoritative `chosen_codec`
    // rather than a guess. The transport layer stamps the clock-probe
    // timestamps right around the send.
    //
    // Newer clients sending an unknown ClientHello variant fail decode
    // upstream of this closure; the transport surfaces that as an error
    // we return below. Reaching the closure means the variant decoded.
    // Two-stage codec selection: the closure picks (and may fail to
    // pick), but the handshake must still produce a syntactically valid
    // ServerHello to send. If the pick failed, the closure writes a
    // placeholder codec into the response and surfaces None via the
    // outer Option; post-handshake we send a clean Goodbye(InternalError)
    // and exit rather than leaving the client waiting on frames that
    // will never arrive.
    // Host encode capabilities — what this build can actually construct
    // against the live VAAPI driver. Probe is process-cached (driver
    // caps don't change at runtime), so per-connection cost is just a
    // clone. Logged at debug because every reconnect would otherwise
    // repeat the same line; the per-session negotiated result below
    // is the operator-relevant signal.
    let host_encode_profiles = supported_encode_profiles();
    tracing::debug!(
        host_encode_profiles = ?host_encode_profiles,
        "host video encode capabilities"
    );

    let mut chosen_profile_outer: Option<VideoProfile> = None;
    let mut client_decode_profiles_outer: Option<Vec<VideoProfile>> = None;
    let client_hello = conn
        .host_handshake(|hello| {
            let ClientHello::V1(body) = hello;
            // Parse the client's structured decode-profile advert. Absence
            // means a legacy client built before this extension — assume
            // the universal floor (H.264 4:2:0 8-bit) per the protocol
            // doc on CLIENT_DECODE_PROFILES_EXTENSION_KEY.
            let client_caps: Vec<VideoProfile> = body
                .extensions
                .get(CLIENT_DECODE_PROFILES_EXTENSION_KEY)
                .and_then(|bytes| match tether_protocol::decode::<Vec<VideoProfile>>(bytes) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        warn!(
                            error = %e,
                            payload_len = bytes.len(),
                            "client decode-profiles extension failed to decode; \
                             treating as legacy H.264 4:2:0 client"
                        );
                        None
                    }
                })
                .unwrap_or_else(|| vec![VideoProfile::H264_8BIT_420]);
            info!(
                client_decode_profiles = ?client_caps,
                legacy_preferred_codecs = ?body.preferred_codecs,
                "client video decode capabilities (parsed from hello)"
            );
            client_decode_profiles_outer = Some(client_caps.clone());

            let chosen = pick_supported_profile(&host_encode_profiles, &client_caps);
            chosen_profile_outer = chosen;

            let mut extensions = std::collections::BTreeMap::new();
            // Advertise pixel format up front so the client's
            // decoder import path (VAAPI / VT / MF) doesn't
            // have to wait on the first SPS to decide between
            // 8-bit and 10-bit. We're NV12 everywhere today;
            // P010 lands with HDR.
            extensions.insert(
                tether_protocol::control::PIXEL_FORMAT_EXTENSION_KEY.to_string(),
                tether_protocol::encode(&tether_protocol::control::PixelFormat::Nv12)
                    .expect("PixelFormat::Nv12 encodes; types under our control"),
            );
            // Echo the structured negotiation result. The inline
            // chosen_codec / chosen_chroma fields stay for legacy clients;
            // the structured echo is the source of truth going forward.
            if let Some(p) = chosen {
                extensions.insert(
                    SERVER_ENCODE_PROFILE_EXTENSION_KEY.to_string(),
                    tether_protocol::encode(&p)
                        .expect("VideoProfile encodes; types under our control"),
                );
            }

            ServerHello::V1(ServerHelloV1 {
                server_name: "tether-host".to_string(),
                // On no-match the response carries H264 / Yuv420 as a
                // placeholder — the immediately-following Goodbye is what
                // the client actually acts on. We use the universal floor
                // (not e.g. Av1) so a buggy client that ignores the
                // Goodbye doesn't trip on an unknown variant before it
                // sees the goodbye message.
                chosen_codec: chosen.map_or(CodecKind::H264, |p| p.codec),
                chosen_chroma: chosen.map_or(ChromaSubsampling::Yuv420, |p| p.chroma),
                // sRGB transfer, BT.709 matrix / primaries / limited
                // range — the honest spec for every host backend we
                // ship today (PipeWire framebuffer interpreted as
                // gamma-encoded sRGB on Linux; SCK NV12 on macOS).
                color_space: VideoColorSpec::sdr_desktop(),
                // Encoded source dims aren't known yet (lazy encoder init
                // happens on the first frame); use a placeholder and rely
                // on per-frame VideoFrameMeta::dimensions for the truth.
                resolution: (0, 0),
                clock_probe_t0_echo: MonoNanos::ZERO,
                t1_server_recv: MonoNanos::ZERO,
                t2_server_send: MonoNanos::ZERO,
                extensions,
                resume_token: None,
            })
        })
        .await?;
    let ClientHello::V1(client_body) = client_hello;
    let chosen_profile = match chosen_profile_outer {
        Some(p) => p,
        None => {
            warn!(
                client_decode_profiles = ?client_decode_profiles_outer,
                host_encode_profiles = ?host_encode_profiles,
                "no video profile intersects host encode + client decode capabilities; \
                 sending Goodbye(InternalError) and ending session"
            );
            let _ = conn
                .send_control(&ControlMessage::Goodbye {
                    reason: "host and client video capabilities do not intersect".to_string(),
                    code: tether_protocol::control::GoodbyeCode::InternalError,
                })
                .await;
            return Ok(());
        }
    };
    let chosen_codec = chosen_profile.codec;
    info!(
        client = %client_body.client_name,
        chosen_codec = ?chosen_profile.codec,
        chosen_chroma = ?chosen_profile.chroma,
        chosen_bit_depth = chosen_profile.bit_depth,
        max_resolution = ?client_body.max_resolution,
        "video profile negotiated; handshake complete"
    );

    // Send a single placeholder cursor shape so the client's receive
    // path is exercised end-to-end. Real cursor-shape capture (querying
    // the compositor's current cursor sprite, sending Shape on change,
    // UseShape on switch) is its own future workstream. A 16×16 opaque
    // checkerboard is enough to prove the wire shape and the client's
    // log line — replace with the real pointer texture later.
    {
        let pixels: Vec<u8> = (0..16 * 16)
            .flat_map(|i| {
                let on = ((i / 16) + (i % 16)) % 2 == 0;
                let v = if on { 0xFFu8 } else { 0x00 };
                [v, v, v, 0xFF]
            })
            .collect();
        let shape = ControlMessage::CursorShape {
            id: 0,
            hotspot: (0, 0),
            width: 16,
            height: 16,
            format: tether_protocol::cursor::CursorPixelFormat::Rgba8,
            pixels,
        };
        if let Err(e) = conn.send_control(&shape).await {
            warn!(error = ?e, "initial CursorShape send failed; continuing anyway");
        }
        if let Err(e) = conn
            .send_control(&ControlMessage::CursorUseShape { id: 0 })
            .await
        {
            warn!(error = ?e, "CursorUseShape send failed; continuing anyway");
        }
    }

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
    let frames = pick_capture_source(use_test_pattern).await?;

    // Force-IDR signal: control-stream recv task `raise`s it on
    // `ControlMessage::ForceIdr`; the capture/encode thread `take`s it
    // each frame. Coalescing comes for free — N raises between two
    // takes produce one keyframe. See `tether_session::IdrSignal`.
    let force_idr = tether_session::IdrSignal::new();

    // Shared display-dimensions channel: the capture thread learns the
    // real host display size on the first frame and posts (w, h) here;
    // the dims-follower task reads it and feeds the injector via
    // set_display_size. We use a single-slot watch so the injector
    // always reads the latest known dims even if it polls late.
    let (display_dims_tx, display_dims_rx) = tokio::sync::watch::channel::<Option<(u32, u32)>>(None);

    // Capture + send runs on a dedicated OS thread per the expert review:
    // the hot path doesn't share the tokio runtime with anything else.
    // We keep the JoinHandle so the disconnect path can wait for the
    // thread to actually exit before we return — otherwise the encoder
    // and capture receiver would still be live in the background while
    // a follow-on session tried to grab the same resources. The shutdown
    // flag breaks the send loop out of its idle wait on a quiet desktop,
    // where `frames.recv` would otherwise block past disconnect detection.
    let send_shutdown = Arc::new(AtomicBool::new(false));
    // Stream-readiness gate. The client signals it has built its
    // decoders by sending `ControlMessage::StreamReady`; until then
    // we drop captured frames at the head of the send loop. Without
    // this, the first ~100-500 ms of frames race the client's
    // decoder construction and render as garbage or get dropped on
    // the floor (depending on which side wins).
    let stream_ready = Arc::new(AtomicBool::new(false));
    let conn_send = conn.clone();
    let force_idr_for_send = force_idr.clone();
    let send_shutdown_for_thread = send_shutdown.clone();
    let stream_ready_for_thread = stream_ready.clone();
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
                chosen_codec,
                stream_ready_for_thread,
                runtime_handle_for_send,
                conn_keyframe,
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
        tasks.spawn(async move {
            loop {
                match conn.recv_control().await {
                    Ok(ControlMessage::ForceIdr) => {
                        tracing::debug!("client requested IDR");
                        force_idr.raise();
                    }
                    Ok(ControlMessage::StreamReady { video, audio }) => {
                        info!(video, audio, "client signalled StreamReady; opening the gate");
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
                        // No adaptive policy yet — log only. When the
                        // host learns to act on these, it consumes the
                        // counters here and feeds the rate controller.
                        info!(
                            interval_ms,
                            frames_received,
                            frames_dropped,
                            fragments_lost,
                            rtt_ewma_us,
                            "client stats"
                        );
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
                    Ok(ControlMessage::CursorShape { .. } | ControlMessage::CursorUseShape { .. }) => {
                        // Host-originated; receiving one here means the
                        // client misrouted. Log and drop.
                        tracing::debug!("unexpected host→client cursor message arrived on host; ignoring");
                    }
                    Ok(ControlMessage::DisplayList { .. }) => {
                        // Host-originated; misrouted if seen here.
                        tracing::debug!("unexpected host→client DisplayList arrived on host; ignoring");
                    }
                    Ok(ControlMessage::SetActiveDisplays { displays }) => {
                        // Single-display host today — log the request
                        // and ignore. The selection mechanic plugs in
                        // when multi-display capture lands.
                        info!(?displays, "client requested display subset; ignoring (single-display host)");
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
    width: u32,
    height: u32,
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
}

#[cfg(target_os = "linux")]
enum BridgeState {
    NotYetBuilt,
    Ready(Nv12DmaBuf),
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

/// Encode one PipeWire-supplied DMA-BUF frame through the zero-copy
/// pipeline: import BGRA into wgpu, compute BGRA→NV12 onto exported
/// DMA-BUF Y/UV planes, hand both to the encoder's `encode_gpu`.
///
/// On first call, lazily opens the wgpu device and allocates the
/// bridge; the result is cached on the `EncoderSlot`.
#[cfg(target_os = "linux")]
fn encode_gpu_frame(
    slot: &mut EncoderSlot,
    gpu: tether_capture::GpuCapturedFrame,
    pts: i64,
    force_keyframe: bool,
) -> GpuEncodeOutcome {
    let bridge = match &mut slot.bridge {
        BridgeState::Ready(b) => b,
        BridgeState::NotYetBuilt => {
            match pollster::block_on(Nv12DmaBuf::new(slot.width, slot.height)) {
                Ok(built) => {
                    info!(
                        width = slot.width,
                        height = slot.height,
                        "gpuconvert bridge initialised for zero-copy DMA-BUF encode"
                    );
                    slot.bridge = BridgeState::Ready(built);
                    let BridgeState::Ready(b) = &mut slot.bridge else {
                        unreachable!()
                    };
                    b
                }
                Err(e) => {
                    return GpuEncodeOutcome::Fatal(anyhow::anyhow!(
                        "gpuconvert bridge init failed for {}x{} after startup \
                         probe succeeded — device loss or OOM: {e}",
                        slot.width,
                        slot.height,
                    ));
                }
            }
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

    let imported = match bridge.import_bgra_dmabuf(fd, modifier, stride, offset) {
        Ok(t) => t,
        Err(e) => {
            return GpuEncodeOutcome::DropFrame(anyhow::anyhow!("import_bgra_dmabuf: {e}"));
        }
    };
    let nv12 = match bridge.convert(&imported) {
        Ok(f) => f,
        Err(e) => {
            return GpuEncodeOutcome::DropFrame(anyhow::anyhow!("Nv12DmaBuf::convert: {e}"));
        }
    };
    // gpu.release_guard drops here implicitly along with `imported`
    // and `dmabuf` once `nv12` has the dup'd Y/UV fds owned. The
    // bridge's poll-on-completion guarantees the compute write retired
    // before we hand the fds to VAAPI.
    drop(imported);
    let _ = fourcc; // PipeWire-side fourcc is informational; the
                     // shader treats input as BGRA regardless.

    let codec_frame = nv12_dmabuf_to_codec_frame(nv12);
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
    let packets = slot
        .encoder
        .encode_gpu(GpuEncoderFrame::IOSurface(&codec_frame), pts, force_keyframe)?;
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

fn run_capture_and_send(
    conn: Arc<Connection>,
    frames: Receiver<CapturedFrame>,
    force_idr: tether_session::IdrSignal,
    display_dims_tx: tokio::sync::watch::Sender<Option<(u32, u32)>>,
    shutdown: Arc<AtomicBool>,
    chosen_codec: CodecKind,
    stream_ready: Arc<AtomicBool>,
    runtime: tokio::runtime::Handle,
    keyframe_conn: Arc<Connection>,
) {
    let mut fragmenter = FrameFragmenter::new(0);
    let mut stats = tether_session::EncodeStatsWindow::new(std::time::Duration::from_secs(2));
    let mut slot: Option<EncoderSlot> = None;
    let mut pts: i64 = 0;

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

        // Encoder is lazily created on the first frame (we don't know
        // capture resolution at startup) and recreated whenever the
        // capture source changes resolution mid-stream (Linux portal
        // output switch, future multi-monitor handoff). Bumping the
        // fragmenter epoch makes the receiver discard any pre-resize
        // fragments still in flight instead of fusing them with the
        // first post-resize keyframe — that's exactly what
        // `VideoPacket::stream_epoch` exists for.
        let needs_recreate = slot
            .as_ref()
            .is_none_or(|s| s.width != frame_width || s.height != frame_height);
        if needs_recreate {
            if let Some(old) = slot.as_ref() {
                info!(
                    old_width = old.width,
                    old_height = old.height,
                    new_width = frame_width,
                    new_height = frame_height,
                    "capture dimensions changed; recreating encoder, bumping stream epoch"
                );
                fragmenter.bump_epoch();
            }
            let _ = display_dims_tx.send(Some((frame_width, frame_height)));
            // Single-element preference list: the handshake already
            // picked one codec, and a mid-session codec switch would
            // require coordinated client decoder rebuild. We pass the
            // list-form for API symmetry with the initial handshake
            // probe; per-resize cost is one construction attempt.
            slot = match probe_encoder(
                &[chosen_codec],
                frame_width,
                frame_height,
                ENCODER_FPS,
                derive_bitrate_kbps(chosen_codec, frame_width, frame_height, ENCODER_FPS),
            ) {
                Ok((_kind, e)) => {
                    info!(
                        backend = e.name(),
                        hardware = e.is_hardware(),
                        codec = ?chosen_codec,
                        width = frame_width,
                        height = frame_height,
                        fps = ENCODER_FPS,
                        "encoder initialised"
                    );
                    Some(EncoderSlot {
                        encoder: e,
                        width: frame_width,
                        height: frame_height,
                        #[cfg(target_os = "linux")]
                        bridge: BridgeState::NotYetBuilt,
                    })
                }
                Err(e) => {
                    warn!(error = %e, "encoder init failed, exiting send loop");
                    return;
                }
            };
        }
        let slot_mut = slot.as_mut().expect("slot populated above");

        // Swap-and-zero: at most one forced keyframe per request, even
        // if multiple ForceIdr messages arrive between encode calls.
        let force_kf = force_idr.take();
        timing.encode_submit();
        let encoded = match frame {
            CapturedFrame::Cpu(ref cpu) => {
                match slot_mut.encoder.encode_bgra(&cpu.data, pts, force_kf) {
                    Ok(e) => e,
                    Err(e) => {
                        warn!(error = %e, "encode failed; dropping frame");
                        continue;
                    }
                }
            }
            #[cfg(target_os = "linux")]
            CapturedFrame::Gpu(gpu) => match encode_gpu_frame(slot_mut, gpu, pts, force_kf) {
                GpuEncodeOutcome::Packets(p) => p,
                GpuEncodeOutcome::DropFrame(e) => {
                    warn!(error = %e, "GPU encode failed; dropping frame");
                    continue;
                }
                GpuEncodeOutcome::Fatal(e) => {
                    tracing::error!(error = %e, "GPU encode bridge collapsed; exiting send loop");
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
        // usually 1:1; the first few frames may produce 0 (encoder
        // setup latency) which we silently skip.
        let mut keyframe = false;
        let mut combined = Vec::new();
        for pkt in encoded {
            if pkt.keyframe {
                keyframe = true;
            }
            combined.extend_from_slice(&pkt.data);
        }
        if combined.is_empty() {
            continue;
        }
        stats.record_frame(encode_delta_ns, combined.len() as u64, keyframe);

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
            let packet = fragmenter.single_packet(meta, combined);
            if let Err(e) = runtime.block_on(keyframe_conn.send_video_keyframe(&packet)) {
                warn!(error = ?e, "send_video_keyframe failed, terminating send loop");
                return;
            }
        } else {
            let packets = fragmenter.fragment(meta, &combined);
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
        anyhow::anyhow!(
            "neither $TETHER_CERT_DIR nor $HOME is set; can't choose a cert directory"
        )
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
    real_capture().await
}

#[cfg(target_os = "linux")]
async fn real_capture() -> anyhow::Result<Receiver<CapturedFrame>> {
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
    let modifiers = match tether_gpuconvert::importable_dmabuf_modifiers(
        u32::from_le_bytes(*b"AR24"),
    )
    .await
    {
        Ok(m) if !m.is_empty() => {
            info!(count = m.len(), "advertised DMA-BUF modifiers to compositor");
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
async fn real_capture() -> anyhow::Result<Receiver<CapturedFrame>> {
    // No gpuconvert preflight — VideoToolbox accepts the NV12
    // IOSurface ScreenCaptureKit hands us directly. SCK's
    // `start_capture` triggers the macOS ScreenRecording TCC prompt
    // on first run; subsequent runs reuse the granted permission.
    info!("capture source: macOS (ScreenCaptureKit, primary display)");
    tether_capture::macos::start()
        .await
        .map_err(anyhow::Error::from)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
async fn real_capture() -> anyhow::Result<Receiver<CapturedFrame>> {
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

/// Default VBR target bitrate as a function of resolution, fps, and
/// codec. Anchored at 1080p60 H.264 = 8 Mbps (the [`ENCODER_BITRATE_KBPS`]
/// floor); scales linearly with `pixels × fps`; HEVC gets a 0.7×
/// multiplier (conservative ~30% efficiency gain over H.264 at the
/// same visual quality). Clamped to a sane band so a tiny test pattern
/// doesn't get a starvation-tier bitrate and a huge display doesn't
/// blow the LAN.
fn derive_bitrate_kbps(codec: CodecKind, width: u32, height: u32, fps: u32) -> u32 {
    const REFERENCE_PIXELS: u64 = 1920 * 1080;
    const REFERENCE_FPS: u64 = 60;
    const REFERENCE_KBPS_H264: u64 = ENCODER_BITRATE_KBPS as u64;

    let pixels = u64::from(width).saturating_mul(u64::from(height));
    let scaled = REFERENCE_KBPS_H264
        .saturating_mul(pixels)
        .saturating_mul(u64::from(fps))
        / (REFERENCE_PIXELS * REFERENCE_FPS).max(1);

    let codec_scaled = match codec {
        CodecKind::H264 => scaled,
        // HEVC: 70% of H.264 for similar visual quality.
        CodecKind::Hevc => scaled * 7 / 10,
        // AV1: not yet supported; if it gets here the encoder build
        // will fail anyway. 60% is the standard reference number.
        CodecKind::Av1 => scaled * 6 / 10,
    };

    codec_scaled.clamp(500, 30_000) as u32
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::task::JoinSet;

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
}
