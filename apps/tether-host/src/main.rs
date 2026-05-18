//! Tether host — captures the local display and streams it to a client.
//!
//! v0 walking skeleton: synthetic test-pattern capture, no codec, raw
//! BGRA frames fragmented over QUIC datagrams. Single client per host.
//! Real capture (ScreenCaptureKit / PipeWire) and a codec come later.
//!
//! Usage: `tether-host [bind_addr]` (defaults to `127.0.0.1:7654`).

use std::net::SocketAddr;
use std::sync::Arc;

use tether_capture::test_pattern;
use tether_protocol::video::{FrameFragmenter, HostFrameTiming, InputEchoBatch, VideoFrameMeta};
use tether_protocol::MonoNanos;
use tether_transport::{Connection, Datagram, Server};
use tracing::{info, warn};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const FPS: u32 = 30;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let bind: SocketAddr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:7654".into())
        .parse()?;

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

    // Capture + send runs on a dedicated OS thread per the expert review:
    // the hot path doesn't share the tokio runtime with anything else.
    let conn_send = conn.clone();
    std::thread::Builder::new()
        .name("tether-host-send".into())
        .spawn(move || run_capture_and_send(conn_send))?;

    // Block until Ctrl-C, then shut down gracefully.
    tokio::signal::ctrl_c().await?;
    info!("ctrl-c received, shutting down");
    conn.close(0, b"host shutdown");
    server.close_and_wait(0, b"host shutdown").await;
    Ok(())
}

fn run_capture_and_send(conn: Arc<Connection>) {
    let frames = test_pattern::start(WIDTH, HEIGHT, FPS);
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
