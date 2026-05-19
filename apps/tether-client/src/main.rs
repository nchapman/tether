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
use tether_input::WinitTranslator;
use tether_protocol::control::{ClientHello, CodecKind, ControlMessage};
use tether_protocol::video::FrameReassembler;
use tether_protocol::{MonoNanos, PROTOCOL_VERSION};
use tether_render::{RawFrame, RenderEvent};
use tether_transport::{Client, Datagram};
use tokio::sync::mpsc;
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

    // Application-layer handshake: identify ourselves, request a codec,
    // and use the embedded probe to compute a host↔client clock offset
    // so latency logs are wall-clock-accurate from the first frame.
    let hello = ClientHello {
        protocol_version: PROTOCOL_VERSION,
        client_name: "tether-client".to_string(),
        preferred_codecs: vec![CodecKind::H264],
        max_resolution: None,
        clock_probe_t0: MonoNanos::ZERO,
    };
    let (server_hello, clock_sync) = conn.client_handshake(hello).await?;
    if server_hello.protocol_version != PROTOCOL_VERSION {
        anyhow::bail!(
            "protocol version mismatch: client={PROTOCOL_VERSION}, server={}",
            server_hello.protocol_version
        );
    }
    info!(
        server = %server_hello.server_name,
        codec = ?server_hello.chosen_codec,
        resolution = ?server_hello.resolution,
        rtt_us = clock_sync.rtt_nanos / 1_000,
        clock_offset_us = clock_sync.offset_nanos / 1_000,
        "handshake complete"
    );

    // Render channel: producer is the recv loop, consumer is the wgpu
    // window. Bounded(2) with drop-newest semantics matches the rest of
    // the project.
    let (frame_tx, frame_rx) = bounded::<RawFrame>(2);

    // First IDR request goes out immediately after the handshake: the
    // host's encoder always emits IDR on its very first frame, but if
    // capture hasn't started yet (portal prompt still up, etc.) we
    // need to make sure the *next* frame we see is a keyframe instead
    // of joining the host's P-frame chain mid-GOP.
    if let Err(e) = conn.send_control(&ControlMessage::ForceIdr).await {
        warn!(error = ?e, "initial ForceIdr send failed; continuing anyway");
    }

    let conn_recv = conn.clone();
    let recv_clock_sync = clock_sync;
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
        // Rate-limit ForceIdr requests so a corrupt stream doesn't
        // turn into a keyframe storm. 500ms matches the human "is this
        // still broken?" cadence; anything tighter just wastes
        // bitrate on duplicate IDRs that haven't even been encoded yet.
        let mut last_idr_request: Option<Instant> = None;
        const IDR_RATE_LIMIT: std::time::Duration = std::time::Duration::from_millis(500);

        loop {
            match conn_recv.recv_datagram().await {
                Ok(Datagram::Video(packet)) => {
                    let Some(frame) = reassembler.handle(packet) else { continue };
                    let now = MonoNanos::now();
                    // Host timestamp -> client clock via the handshake
                    // offset, so this is true glass-to-glass-ish (we
                    // still don't sample present-time on the GPU). On
                    // first frame this can saturate-to-zero if the
                    // offset hasn't actually advanced past zero yet —
                    // log will read 0 ms, which is fine for a single
                    // frame.
                    let host_in_client_clock =
                        recv_clock_sync.remote_to_local(frame.meta.timing.t_capture_userspace);
                    let age_ns = now.saturating_sub(host_in_client_clock);
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
                            // A decode error usually means a P-frame
                            // arrived without its IDR (or with a dropped
                            // fragment that corrupted the slice). Asking
                            // the host for a fresh IDR is the only way
                            // out — without it we stay garbled until
                            // the next periodic IDR (up to one GOP).
                            let now = Instant::now();
                            if last_idr_request.is_none_or(|t| now.duration_since(t) > IDR_RATE_LIMIT) {
                                let conn = conn_recv.clone();
                                tokio::spawn(async move {
                                    if let Err(e) = conn.send_control(&ControlMessage::ForceIdr).await {
                                        warn!(error = ?e, "ForceIdr send failed");
                                    }
                                });
                                last_idr_request = Some(now);
                            }
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

    // Bridge winit window events from the render thread into a tokio
    // task that owns the connection's input stream. UnboundedSender is
    // safe to call from the render thread (sync) and the receiver runs
    // inside the tokio runtime where send_input is async.
    let (events_tx, mut events_rx) = mpsc::unbounded_channel::<RenderEvent>();
    let conn_input = conn.clone();
    tokio::spawn(async move {
        let mut translator = WinitTranslator::new();
        while let Some(render_event) = events_rx.recv().await {
            for input_event in translator.translate(render_event) {
                if let Err(e) = conn_input.send_input(&input_event).await {
                    error!(error = ?e, "send_input failed; ending input loop");
                    return;
                }
            }
        }
    });

    let on_event: tether_render::EventSink = Box::new(move |evt| {
        // Render must not block on a slow consumer. UnboundedSender drops
        // its message on send-after-close, which is exactly what we want
        // when the input task has exited.
        let _ = events_tx.send(evt);
    });

    // Render loop blocks until the user closes the window.
    tether_render::run(
        "tether-client",
        (INITIAL_WIDTH, INITIAL_HEIGHT),
        frame_rx,
        Some(on_event),
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
