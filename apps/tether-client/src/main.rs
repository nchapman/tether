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
use tether_codec::{probe_decoder, Decoder};
use tether_input::{WinitTranslator, WireEvent};
use tether_protocol::control::{ClientHello, CodecKind, ControlMessage};
use tether_protocol::video::FrameReassembler;
use tether_protocol::{MonoNanos, PROTOCOL_VERSION};
use tether_render::{Frame, RenderEvent};
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
    let (frame_tx, frame_rx) = bounded::<Frame>(2);

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
        let mut decoder: Box<dyn Decoder> = match probe_decoder() {
            Ok(d) => {
                info!(
                    backend = d.name(),
                    hardware = d.is_hardware(),
                    "decoder initialised"
                );
                d
            }
            Err(e) => {
                warn!(error = %e, "decoder init failed; aborting recv loop");
                return;
            }
        };
        let mut frame_count: u64 = 0;
        // Sum of decode call wall-clocks across the frames in the
        // current log window, surfaced as avg_decode_ms on the
        // frame-stats line. Same shape as the host's avg_encode_ms
        // metric so the two together show where pipeline time is
        // going.
        let mut decode_latency_sum_ns: u64 = 0;
        // Sum of capture-to-recv ages over the window. The previous
        // implementation logged the last frame's age which is
        // misleading when the metric is supposed to summarise a
        // second of behaviour; averaging across frames gives an
        // actually-meaningful number.
        let mut latency_sum_ns: u64 = 0;
        // Sum of t_send (host clock, translated via clock_sync) to
        // local recv times. Isolates the network leg from compute
        // — pair with avg_encode_ms / avg_decode_ms to attribute
        // any latency budget movement to the right component.
        let mut network_latency_sum_ns: u64 = 0;
        // Bytes off the wire (encoded H.264 payloads after
        // reassembly). With matching host kbps_out, a divergence
        // means packets are being dropped between host and client.
        let mut bytes_received: u64 = 0;
        // Decode failures in the window. Steady non-zero values
        // mean we're losing IDR/SPS/PPS frames or the bitstream is
        // getting corrupted between encode and decode.
        let mut decode_errors: u32 = 0;
        // Frames the recv loop produced but the render channel
        // couldn't accept (bounded(2), drop-newest). Non-zero
        // means the render thread isn't keeping up with arrival
        // rate — typically a wgpu/present pacing issue, not a
        // codec issue.
        let mut render_drops: u32 = 0;
        // ForceIdr control messages we sent in the window. Pairs
        // with the host's kf_per_s — if these match, the storm of
        // keyframes on the host is *our* fault, not the encoder
        // misbehaving.
        let mut idr_requests: u32 = 0;
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
                    // Host timestamps -> client clock via the handshake
                    // offset. host_in_client_clock is the moment the
                    // host captured the frame; send_in_client_clock is
                    // the moment the host handed it to QUIC. The
                    // difference between them and `now` decomposes
                    // total latency into capture-to-send (host
                    // pipeline) and send-to-recv (network + reassembly).
                    let host_in_client_clock =
                        recv_clock_sync.remote_to_local(frame.meta.timing.t_capture_userspace);
                    let send_in_client_clock =
                        recv_clock_sync.remote_to_local(frame.meta.timing.t_send);
                    let age_ns = now.saturating_sub(host_in_client_clock);
                    let network_ns = now.saturating_sub(send_in_client_clock);
                    frame_count += 1;
                    latency_sum_ns = latency_sum_ns.saturating_add(age_ns);
                    network_latency_sum_ns = network_latency_sum_ns.saturating_add(network_ns);
                    bytes_received = bytes_received.saturating_add(frame.body.len() as u64);

                    let t_decode_start = MonoNanos::now();
                    let decoded = match decoder.decode(&frame.body) {
                        Ok(d) => d,
                        Err(e) => {
                            warn!(error = %e, "h264 decode failed; dropping packet");
                            decode_errors = decode_errors.saturating_add(1);
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
                                idr_requests = idr_requests.saturating_add(1);
                            }
                            continue;
                        }
                    };
                    let t_decode_done = MonoNanos::now();
                    decode_latency_sum_ns = decode_latency_sum_ns
                        .saturating_add(t_decode_done.saturating_sub(t_decode_start));

                    for dec in decoded {
                        // YUV planes go straight to the render texture
                        // upload — no per-frame CPU pixel format
                        // conversion in our code anymore. The
                        // BGRA↔RGBA bounce that used to live here
                        // turned into a free GPU sample in the YUV
                        // fragment shader.
                        let raw = Frame {
                            width: dec.width,
                            height: dec.height,
                            y: dec.y,
                            uv: dec.uv,
                            t_capture_client_clock: Some(host_in_client_clock),
                        };
                        // Drop on full — render is intentionally one-deep.
                        if frame_tx.try_send(raw).is_err() {
                            render_drops = render_drops.saturating_add(1);
                        }
                    }

                    if last_log.elapsed() >= std::time::Duration::from_secs(1) {
                        let window_secs = last_log.elapsed().as_secs_f64();
                        #[allow(clippy::cast_precision_loss)] // u64 frame_count well under 2^53
                        let avg_decode_ms = if frame_count > 0 {
                            (decode_latency_sum_ns as f64 / frame_count as f64) / 1_000_000.0
                        } else {
                            0.0
                        };
                        #[allow(clippy::cast_precision_loss)]
                        let avg_latency_ms = if frame_count > 0 {
                            (latency_sum_ns as f64 / frame_count as f64) / 1_000_000.0
                        } else {
                            0.0
                        };
                        #[allow(clippy::cast_precision_loss)]
                        let avg_network_ms = if frame_count > 0 {
                            (network_latency_sum_ns as f64 / frame_count as f64) / 1_000_000.0
                        } else {
                            0.0
                        };
                        #[allow(clippy::cast_precision_loss)]
                        let kbps_in = if window_secs > 0.0 {
                            (bytes_received as f64 * 8.0 / 1000.0) / window_secs
                        } else {
                            0.0
                        };
                        info!(
                            frames_per_s = frame_count,
                            latency_ms = format!("{avg_latency_ms:.2}"),
                            network_ms = format!("{avg_network_ms:.2}"),
                            avg_decode_ms = format!("{avg_decode_ms:.2}"),
                            kbps_in = format!("{kbps_in:.0}"),
                            decode_errs = decode_errors,
                            render_drops = render_drops,
                            idr_reqs = idr_requests,
                            "frame stats"
                        );
                        frame_count = 0;
                        decode_latency_sum_ns = 0;
                        latency_sum_ns = 0;
                        network_latency_sum_ns = 0;
                        bytes_received = 0;
                        decode_errors = 0;
                        render_drops = 0;
                        idr_requests = 0;
                        last_log = Instant::now();
                    }
                }
                Ok(Datagram::HostCursor(_)) => {
                    // Host cursor sprite/position rendering isn't wired
                    // on this side yet; the wire slot is reserved.
                }
                Ok(Datagram::ClientCursor(_)) => {
                    // Client-originated cursor packets should never
                    // come back to the client; ignore defensively.
                }
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
            for wire in translator.translate(render_event) {
                match wire {
                    WireEvent::Input(evt) => {
                        if let Err(e) = conn_input.send_input(&evt).await {
                            error!(error = ?e, "send_input failed; ending input loop");
                            return;
                        }
                    }
                    WireEvent::Cursor(pkt) => {
                        if let Err(e) =
                            conn_input.send_datagram(&Datagram::ClientCursor(pkt))
                        {
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

    let on_event: tether_render::EventSink = Box::new(move |evt| {
        // Render must not block on a slow consumer. UnboundedSender drops
        // its message on send-after-close, which is exactly what we want
        // when the input task has exited.
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
        tokio::spawn(async move {
            if let Err(e) = tokio::signal::ctrl_c().await {
                warn!(error = %e, "ctrl-c handler failed; exiting anyway");
                std::process::exit(1);
            }
            info!("ctrl-c received, sending Goodbye and exiting");
            say_goodbye(&conn, "client interrupted").await;
            std::process::exit(0);
        });
    }

    // Render loop blocks until the user closes the window.
    tether_render::run(
        "tether-client",
        (INITIAL_WIDTH, INITIAL_HEIGHT),
        frame_rx,
        Some(on_event),
    )?;

    // Normal window-close path. Notify the host so it can tear down its
    // capture, encoder, and libei session immediately instead of waiting
    // for QUIC's idle timeout.
    say_goodbye(&conn, "client closing").await;
    Ok(())
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
async fn say_goodbye(conn: &tether_transport::Connection, reason: &str) {
    use std::time::Duration;
    let msg = ControlMessage::Goodbye {
        reason: reason.to_string(),
    };
    if let Err(e) = conn.send_control(&msg).await {
        warn!(error = ?e, "send Goodbye failed; host will fall back to timeout");
    } else {
        let wait = (2 * conn.rtt()).clamp(Duration::from_millis(20), Duration::from_millis(200));
        tokio::time::sleep(wait).await;
    }
    conn.close(0, reason.as_bytes());
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
