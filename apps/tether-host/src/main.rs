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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam_channel::Receiver;
use tether_capture::{CapturedFrame, PixelFormat};
use tether_codec::{Encoder, H264Encoder};
use tether_protocol::control::{
    ChromaSubsampling, CodecKind, ColorSpace, ControlMessage, ServerHello,
};
use tether_protocol::video::{FrameFragmenter, HostFrameTiming, InputEchoBatch, VideoFrameMeta};
use tether_protocol::{MonoNanos, PROTOCOL_VERSION};
use tether_transport::{Connection, Datagram, Server};
use tracing::{info, warn};

const ENCODER_BITRATE_KBPS: u32 = 4_000;
const ENCODER_FPS: u32 = 30;

const TEST_PATTERN_WIDTH: u32 = 320;
const TEST_PATTERN_HEIGHT: u32 = 240;
const TEST_PATTERN_FPS: u32 = 30;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let (bind, use_test_pattern) = parse_args()?;

    let server = Server::bind(bind).await?;
    let local = server.local_addr()?;
    let fingerprint = server.fingerprint();
    let fp_hex = hex_encode(&fingerprint);

    println!("tether-host listening on {local}");
    println!("cert fingerprint: {fp_hex}");
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

    // Application-layer handshake. The closure picks the codec from the
    // client's preference list (H264-only for now), and the transport
    // layer stamps the clock-probe timestamps right around the send to
    // keep the offset measurement tight. Hello protocol-version
    // mismatches are fatal per the v0 policy in control.rs.
    let client_hello = conn
        .host_handshake(|hello| ServerHello {
            protocol_version: PROTOCOL_VERSION,
            server_name: "tether-host".to_string(),
            chosen_codec: pick_codec(&hello.preferred_codecs),
            chosen_chroma: ChromaSubsampling::Yuv420,
            color_space: ColorSpace::Bt709Limited,
            // Encoded source dims aren't known yet (lazy encoder init
            // happens on the first frame); use a placeholder and rely
            // on per-frame VideoFrameMeta::dimensions for the truth.
            resolution: (0, 0),
            clock_probe_t0_echo: MonoNanos::ZERO,
            t1_server_recv: MonoNanos::ZERO,
            t2_server_send: MonoNanos::ZERO,
        })
        .await?;
    if client_hello.protocol_version != PROTOCOL_VERSION {
        warn!(
            client_version = client_hello.protocol_version,
            host_version = PROTOCOL_VERSION,
            "protocol version mismatch; closing"
        );
        conn.close(0, b"protocol version mismatch");
        return Ok(());
    }
    info!(
        client = %client_hello.client_name,
        codecs = ?client_hello.preferred_codecs,
        max_resolution = ?client_hello.max_resolution,
        "handshake complete"
    );

    // Acquire a capture stream — either real platform capture or the
    // synthetic test pattern fallback. Real Linux capture is async (the
    // portal handshake awaits a user permission dialog); test pattern is
    // sync. Both end up as a `Receiver<CapturedFrame>` so the send loop
    // is identical.
    let frames = pick_capture_source(use_test_pattern).await?;

    // Force-IDR signal: control-stream recv task sets it on
    // `ControlMessage::ForceIdr`; the capture/encode thread checks it
    // each frame via swap. AtomicBool is the cheapest cross-thread
    // primitive that fits the "one-shot until next frame" semantic.
    let force_idr = Arc::new(AtomicBool::new(false));

    // Force-IDR signal already created above. Shared display-dimensions
    // channel: the capture thread learns the real host display size on
    // the first frame and posts (w, h) here; the input recv loop reads
    // it and feeds the injector via set_display_size. We use a single-
    // slot watch so the injector always reads the latest known dims
    // even if it polls late.
    let (display_dims_tx, display_dims_rx) = tokio::sync::watch::channel::<Option<(u32, u32)>>(None);

    // Capture + send runs on a dedicated OS thread per the expert review:
    // the hot path doesn't share the tokio runtime with anything else.
    // Naming the JoinHandle is informational — dropping it doesn't kill
    // the thread in Rust, and v0 has no clean way to signal the PipeWire
    // main loop to stop, so we rely on process exit to tear capture down.
    let conn_send = conn.clone();
    let force_idr_for_send = force_idr.clone();
    let _send_handle = std::thread::Builder::new()
        .name("tether-host-send".into())
        .spawn(move || run_capture_and_send(conn_send, frames, force_idr_for_send, display_dims_tx))?;

    // Control recv: react to ForceIdr and clock-probe requests on the
    // reliable control stream. Goodbye triggers a clean shutdown by
    // closing the connection — the recv loops everywhere else then
    // return Err and exit. Unknown messages are logged at trace; we
    // never crash on a control packet.
    let conn_control = conn.clone();
    let force_idr_for_ctrl = force_idr.clone();
    tokio::spawn(async move {
        loop {
            match conn_control.recv_control().await {
                Ok(ControlMessage::ForceIdr) => {
                    tracing::debug!("client requested IDR");
                    force_idr_for_ctrl.store(true, Ordering::Relaxed);
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
                    if let Err(e) = conn_control.send_control(&response).await {
                        warn!(error = ?e, "clock probe response failed; ending control loop");
                        return;
                    }
                }
                Ok(ControlMessage::ClockProbeResponse(_)) => {
                    // Host doesn't currently initiate re-probes, but if
                    // we ever do, the matching response handler goes here.
                    tracing::trace!("unsolicited clock probe response; ignoring");
                }
                Ok(ControlMessage::Goodbye { reason }) => {
                    info!(%reason, "client said goodbye");
                    return;
                }
                Err(e) => {
                    warn!(error = ?e, "control recv failed; ending control loop");
                    return;
                }
            }
        }
    });

    // Input recv: drain the client's input stream and feed each event
    // into the host's injection backend. Backend selection happens
    // before the recv loops start so the user sees any portal prompt
    // up front; a backend init failure is non-fatal — we fall back to
    // a noop injector that just logs.
    //
    // The injector is shared between the reliable input-stream task
    // and the unreliable cursor-datagram task; tokio's Mutex is the
    // right primitive because both call sites are async and the
    // critical section (a single enigo call) is short. A std Mutex
    // would risk blocking a tokio worker if libei ever takes longer
    // than a few microseconds.
    use std::sync::Arc as StdArc;
    use tokio::sync::Mutex as TokioMutex;
    let injector = StdArc::new(TokioMutex::new(
        tether_input::inject::default_injector().await,
    ));

    // Display-dimensions follower: any change to the capture's
    // negotiated resolution pushes new pixel dims into the injector.
    // Lives in its own task so the recv loops below stay focused.
    let injector_for_dims = injector.clone();
    let mut display_dims_watch = display_dims_rx.clone();
    tokio::spawn(async move {
        while display_dims_watch.changed().await.is_ok() {
            // Copy the value out and drop the borrow guard before
            // awaiting on the injector lock — `watch::Ref` is not Send,
            // so holding it across an .await fails the Send bound.
            let dims = *display_dims_watch.borrow();
            if let Some((w, h)) = dims {
                injector_for_dims.lock().await.set_display_size(w, h);
            }
        }
    });

    let conn_input = conn.clone();
    let injector_for_input = injector.clone();
    tokio::spawn(async move {
        loop {
            match conn_input.recv_input().await {
                Ok(evt) => {
                    tracing::trace!(
                        event_id = evt.event_id,
                        t_client_ns = evt.t_client.0,
                        kind = ?evt.kind,
                        "input event"
                    );
                    let mut inj = injector_for_input.lock().await;
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

    // Datagram recv: cursor packets ride the unreliable channel for
    // latency. Video and host-cursor datagrams flow the other
    // direction; they should never arrive here, but we match
    // defensively so a misbehaving client can't crash the host.
    let conn_dgram = conn.clone();
    let injector_for_dgram = injector.clone();
    tokio::spawn(async move {
        loop {
            match conn_dgram.recv_datagram().await {
                Ok(Datagram::ClientCursor(c)) => {
                    let mut inj = injector_for_dgram.lock().await;
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

    // Block until Ctrl-C, then shut down gracefully. A ctrl-c registration
    // error is itself a reason to shut down, not to bail out and skip the
    // close path — log and proceed either way.
    if let Err(e) = tokio::signal::ctrl_c().await {
        warn!(error = %e, "ctrl-c handler failed; shutting down anyway");
    } else {
        info!("ctrl-c received, shutting down");
    }
    conn.close(0, b"host shutdown");
    server.close_and_wait(0, b"host shutdown").await;
    Ok(())
}

/// Encoder paired with the input dimensions it was configured for, so we
/// can detect a resolution change in the capture stream and recreate the
/// encoder (plus bump the wire-side stream epoch) before the next frame.
struct EncoderSlot {
    encoder: H264Encoder,
    width: u32,
    height: u32,
}

fn run_capture_and_send(
    conn: Arc<Connection>,
    frames: Receiver<CapturedFrame>,
    force_idr: Arc<AtomicBool>,
    display_dims_tx: tokio::sync::watch::Sender<Option<(u32, u32)>>,
) {
    let mut fragmenter = FrameFragmenter::new(0);
    let mut frame_count: u64 = 0;
    let mut last_log = std::time::Instant::now();
    let mut slot: Option<EncoderSlot> = None;
    let mut pts: i64 = 0;

    while let Ok(frame) = frames.recv() {
        if frame.format != PixelFormat::Bgra8 {
            warn!(
                ?frame.format,
                "h264 encoder only accepts BGRA in v0; skipping frame"
            );
            continue;
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
            .is_none_or(|s| s.width != frame.width || s.height != frame.height);
        if needs_recreate {
            if let Some(old) = slot.as_ref() {
                info!(
                    old_width = old.width,
                    old_height = old.height,
                    new_width = frame.width,
                    new_height = frame.height,
                    "capture dimensions changed; recreating encoder, bumping stream epoch"
                );
                fragmenter.bump_epoch();
            }
            // Push the new display dims to anything that cares (the
            // input injector uses these to scale normalised cursor
            // coords into pixels). send() only fails if every receiver
            // is gone, which means the host is shutting down anyway.
            let _ = display_dims_tx.send(Some((frame.width, frame.height)));
            slot = match H264Encoder::new_bgra(
                frame.width,
                frame.height,
                ENCODER_FPS,
                ENCODER_BITRATE_KBPS,
            ) {
                Ok(e) => {
                    info!(
                        width = frame.width,
                        height = frame.height,
                        fps = ENCODER_FPS,
                        kbps = ENCODER_BITRATE_KBPS,
                        "h264 encoder initialised"
                    );
                    Some(EncoderSlot {
                        encoder: e,
                        width: frame.width,
                        height: frame.height,
                    })
                }
                Err(e) => {
                    warn!(error = %e, "encoder init failed, exiting send loop");
                    return;
                }
            };
        }
        let enc = &mut slot.as_mut().expect("slot populated above").encoder;

        // Swap-and-zero: at most one forced keyframe per request, even
        // if multiple ForceIdr messages arrive between encode calls.
        let force_kf = force_idr.swap(false, Ordering::Relaxed);
        let t_encode_submit = MonoNanos::now();
        let encoded = match enc.encode_bgra(&frame.data, pts, force_kf) {
            Ok(e) => e,
            Err(e) => {
                warn!(error = %e, "encode failed; dropping frame");
                continue;
            }
        };
        let t_encode_done = MonoNanos::now();
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

        let t_send = MonoNanos::now();
        let meta = VideoFrameMeta {
            timing: HostFrameTiming {
                t_capture_kernel: frame.t_capture_kernel,
                t_capture_userspace: frame.t_capture_userspace,
                t_encode_submit,
                t_encode_done,
                t_send,
            },
            keyframe,
            input_echo: InputEchoBatch::default(),
            dimensions: (frame.width, frame.height),
        };

        let packets = fragmenter.fragment(meta, &combined);
        for packet in packets {
            if let Err(e) = conn.send_datagram(&Datagram::Video(packet)) {
                warn!(error = ?e, "send_datagram failed, terminating send loop");
                return;
            }
        }

        frame_count += 1;
        if last_log.elapsed() >= std::time::Duration::from_secs(2) {
            info!(
                frames = frame_count,
                "sent {} encoded frames in last 2s",
                frame_count
            );
            frame_count = 0;
            last_log = std::time::Instant::now();
        }
    }
    info!("capture channel closed, send loop exiting");
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
    tether_capture::linux::start()
        .await
        .map_err(anyhow::Error::from)
}

#[cfg(not(target_os = "linux"))]
async fn real_capture() -> anyhow::Result<Receiver<CapturedFrame>> {
    warn!("no real capture backend on this platform yet; falling back to test-pattern");
    Ok(tether_capture::test_pattern::start(
        TEST_PATTERN_WIDTH,
        TEST_PATTERN_HEIGHT,
        TEST_PATTERN_FPS,
    ))
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

/// Pick the first codec from the client's preference list that we can
/// actually encode. H264 is the only one we ship today; HEVC and AV1
/// will land later.
fn pick_codec(preferred: &[CodecKind]) -> CodecKind {
    for k in preferred {
        if matches!(k, CodecKind::H264) {
            return *k;
        }
    }
    CodecKind::H264
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}
