//! Tether client — connects to a host, reassembles incoming video frames,
//! and presents them in a wgpu window.
//!
//! v0 walking skeleton: raw BGRA frames over QUIC datagrams, no codec,
//! no jitter buffer beyond the single-frame-deep render channel.
//!
//! Usage: `tether-client <host_addr> <cert_fingerprint_hex>`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::bounded;
use tether_codec::{Decoder, H264Decoder};
use tether_protocol::video::FrameReassembler;
use tether_protocol::MonoNanos;
use tether_render::RawFrame;
use tether_transport::{Client, Datagram};
use tracing::{error, info, warn};

// Initial window size — the actual frame dimensions come from
// `VideoFrameMeta::dimensions` once frames start arriving and the window
// will pick them up automatically because tether-render reallocates its
// texture on dimension change.
const INITIAL_WIDTH: u32 = 1280;
const INITIAL_HEIGHT: u32 = 720;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let mut args = std::env::args().skip(1);
    let addr: SocketAddr = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing host address argument"))?
        .parse()?;
    let fingerprint_hex = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing fingerprint argument"))?;
    let fingerprint = hex_decode(&fingerprint_hex)?;

    let client = Client::new()?;
    let conn = client.connect(addr, "tether-host", fingerprint).await?;
    let conn = Arc::new(conn);
    info!(remote = %conn.remote_address(), "connected to host");

    // Render channel: producer is the recv loop, consumer is the wgpu
    // window. Bounded(2) with drop-newest semantics matches the rest of
    // the project.
    let (frame_tx, frame_rx) = bounded::<RawFrame>(2);

    let conn_recv = conn.clone();
    tokio::spawn(async move {
        let mut reassembler = FrameReassembler::new();
        let mut decoder = match H264Decoder::new() {
            Ok(d) => d,
            Err(e) => {
                warn!(error = %e, "h264 decoder init failed; aborting recv loop");
                return;
            }
        };
        let mut frame_count: u64 = 0;
        let mut last_log = Instant::now();

        loop {
            match conn_recv.recv_datagram().await {
                Ok(Datagram::Video(packet)) => {
                    let Some(frame) = reassembler.handle(packet) else { continue };
                    let now = MonoNanos::now();
                    let age_ns = now.saturating_sub(frame.meta.timing.t_capture_userspace);
                    frame_count += 1;
                    if last_log.elapsed() >= std::time::Duration::from_secs(1) {
                        info!(
                            frames_per_s = frame_count,
                            latency_ms = age_ns as f64 / 1_000_000.0,
                            "frame stats"
                        );
                        frame_count = 0;
                        last_log = Instant::now();
                    }

                    let decoded = match decoder.decode(&frame.body) {
                        Ok(d) => d,
                        Err(e) => {
                            warn!(error = %e, "h264 decode failed; dropping packet");
                            continue;
                        }
                    };
                    for dec in decoded {
                        let rgba = bgra_to_rgba(&dec.data);
                        let raw = RawFrame {
                            width: dec.width,
                            height: dec.height,
                            data: rgba,
                        };
                        // Drop on full — render is intentionally one-deep.
                        let _ = frame_tx.try_send(raw);
                    }
                }
                Ok(Datagram::Cursor(_)) => {}
                Err(e) => {
                    // Promoted from warn → error: this is terminal for the
                    // video stream and the user otherwise sees a frozen
                    // last-frame with no indication anything broke. Also
                    // close the connection explicitly so the host learns
                    // about it instead of waiting for the idle timeout.
                    error!(error = ?e, "datagram recv failed; closing connection and ending recv loop");
                    conn_recv.close(1, b"recv failed");
                    break;
                }
            }
        }
    });

    // Render loop blocks until the user closes the window.
    tether_render::run(
        "tether-client",
        (INITIAL_WIDTH, INITIAL_HEIGHT),
        frame_rx,
    )?;
    Ok(())
}

fn bgra_to_rgba(bgra: &[u8]) -> Vec<u8> {
    debug_assert_eq!(bgra.len() % 4, 0, "BGRA buffer must be a multiple of 4 bytes");
    let mut out = Vec::with_capacity(bgra.len());
    for px in bgra.chunks_exact(4) {
        out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
    }
    out
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
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
