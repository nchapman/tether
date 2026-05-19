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
use std::sync::Arc;

use crossbeam_channel::Receiver;
use tether_capture::CapturedFrame;
use tether_protocol::video::{FrameFragmenter, HostFrameTiming, InputEchoBatch, VideoFrameMeta};
use tether_protocol::MonoNanos;
use tether_transport::{Connection, Datagram, Server};
use tracing::{info, warn};

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

    // Acquire a capture stream — either real platform capture or the
    // synthetic test pattern fallback. Real Linux capture is async (the
    // portal handshake awaits a user permission dialog); test pattern is
    // sync. Both end up as a `Receiver<CapturedFrame>` so the send loop
    // is identical.
    let frames = pick_capture_source(use_test_pattern).await?;

    // Capture + send runs on a dedicated OS thread per the expert review:
    // the hot path doesn't share the tokio runtime with anything else.
    let conn_send = conn.clone();
    std::thread::Builder::new()
        .name("tether-host-send".into())
        .spawn(move || run_capture_and_send(conn_send, frames))?;

    // Block until Ctrl-C, then shut down gracefully.
    tokio::signal::ctrl_c().await?;
    info!("ctrl-c received, shutting down");
    conn.close(0, b"host shutdown");
    server.close_and_wait(0, b"host shutdown").await;
    Ok(())
}

fn run_capture_and_send(conn: Arc<Connection>, frames: Receiver<CapturedFrame>) {
    let mut fragmenter = FrameFragmenter::new(0);
    let mut frame_count: u64 = 0;
    let mut last_log = std::time::Instant::now();

    while let Ok(frame) = frames.recv() {
        let t_send = MonoNanos::now();
        let meta = VideoFrameMeta {
            timing: HostFrameTiming {
                t_capture_kernel: frame.t_capture_kernel,
                t_capture_userspace: frame.t_capture_userspace,
                // No real encoder yet — submit and done both land at "send".
                t_encode_submit: t_send,
                t_encode_done: t_send,
                t_send,
            },
            // Without a codec every frame is independent; mark all as
            // keyframes so the receiver can decode (i.e. trivially copy)
            // any single frame.
            keyframe: true,
            input_echo: InputEchoBatch::default(),
            dimensions: (frame.width, frame.height),
        };

        let packets = fragmenter.fragment(meta, &frame.data);
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
                "sent {} frames in last 2s",
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

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}
