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
use tether_capture::{CapturedFrame, PixelFormat};
use tether_codec::{Encoder, H264Encoder};
use tether_protocol::video::{FrameFragmenter, HostFrameTiming, InputEchoBatch, VideoFrameMeta};
use tether_protocol::MonoNanos;
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
    let mut encoder: Option<H264Encoder> = None;
    let mut pts: i64 = 0;

    while let Ok(frame) = frames.recv() {
        if frame.format != PixelFormat::Bgra8 {
            warn!(
                ?frame.format,
                "h264 encoder only accepts BGRA in v0; skipping frame"
            );
            continue;
        }

        // Lazy-init the encoder on the first frame: we need its
        // dimensions, and capture might be a resolution we didn't know
        // at startup. A subsequent resolution change would require
        // recreating the encoder + bumping the stream epoch — deferred
        // to a later task.
        let enc = match encoder.as_mut() {
            Some(e) => e,
            None => {
                match H264Encoder::new_bgra(
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
                        encoder.insert(e)
                    }
                    Err(e) => {
                        warn!(error = %e, "encoder init failed, exiting send loop");
                        return;
                    }
                }
            }
        };

        let t_encode_submit = MonoNanos::now();
        let encoded = match enc.encode_bgra(&frame.data, pts, false) {
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

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}
