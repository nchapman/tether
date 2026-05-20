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
use tether_codec::{probe_encoder_bgra, Encoder};
#[cfg(target_os = "linux")]
use tether_capture::GpuCapturedSource;
#[cfg(target_os = "linux")]
use tether_codec::{DmaBufFrame, DmaBufLayer, DmaBufObject, GpuEncoderFrame};
#[cfg(target_os = "linux")]
use tether_gpuconvert::{Nv12DmaBuf, Nv12DmaBufFrame};
use tether_protocol::control::{
    ChromaSubsampling, CodecKind, ColorSpace, ControlMessage, ServerHello,
};
use tether_protocol::video::{
    FrameFragmenter, HostFrameTimingBuilder, InputEchoBatch, VideoFrameMeta,
};
use tether_protocol::{MonoNanos, PROTOCOL_VERSION};
use tether_transport::{Connection, Datagram, Server};
use tokio::sync::Mutex as TokioMutex;
use tokio::task::JoinSet;
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
    let conn_send = conn.clone();
    let force_idr_for_send = force_idr.clone();
    let send_shutdown_for_thread = send_shutdown.clone();
    let send_handle = std::thread::Builder::new()
        .name("tether-host-send".into())
        .spawn(move || {
            run_capture_and_send(
                conn_send,
                frames,
                force_idr_for_send,
                display_dims_tx,
                send_shutdown_for_thread,
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
        tasks.spawn(async move {
            loop {
                match conn.recv_control().await {
                    Ok(ControlMessage::ForceIdr) => {
                        tracing::debug!("client requested IDR");
                        force_idr.store(true, Ordering::Relaxed);
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
    force_idr: Arc<AtomicBool>,
    display_dims_tx: tokio::sync::watch::Sender<Option<(u32, u32)>>,
    shutdown: Arc<AtomicBool>,
) {
    let mut fragmenter = FrameFragmenter::new(0);
    let mut frame_count: u64 = 0;
    // Sum of (t_encode_done - t_encode_submit) across frames in the
    // current log window. Pairs with frame_count to produce an average
    // encode latency per stats log — the headline number for
    // confirming a HW encoder swap actually moved the needle without
    // having to instrument every frame.
    let mut encode_latency_sum_ns: u64 = 0;
    // Bytes of encoded H.264 produced this window (sum of EncodedPacket
    // payloads). Drives kbps_out so the user can see whether the
    // encoder is actually hitting the configured bitrate target.
    let mut encoded_bytes_sum: u64 = 0;
    // Keyframes emitted this window. Expected value with GOP=fps is
    // ~1/s; spikes above that mean the client is hammering ForceIdr
    // (network loss or decoder errors), which is a useful signal
    // independent of why it's happening.
    let mut keyframe_count: u32 = 0;
    let mut last_log = std::time::Instant::now();
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
            slot = match probe_encoder_bgra(
                frame_width,
                frame_height,
                ENCODER_FPS,
                ENCODER_BITRATE_KBPS,
            ) {
                Ok(e) => {
                    info!(
                        backend = e.name(),
                        hardware = e.is_hardware(),
                        width = frame_width,
                        height = frame_height,
                        fps = ENCODER_FPS,
                        kbps = ENCODER_BITRATE_KBPS,
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
        let force_kf = force_idr.swap(false, Ordering::Relaxed);
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
            #[cfg(not(target_os = "linux"))]
            CapturedFrame::Gpu(_) => {
                warn!("Gpu CapturedFrame on a non-Linux build; dropping");
                continue;
            }
        };
        timing.encode_done();
        encode_latency_sum_ns = encode_latency_sum_ns.saturating_add(timing.encode_delta_ns());
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
        encoded_bytes_sum = encoded_bytes_sum.saturating_add(combined.len() as u64);
        if keyframe {
            keyframe_count = keyframe_count.saturating_add(1);
        }

        let meta = VideoFrameMeta {
            timing: timing.finish(),
            keyframe,
            input_echo: InputEchoBatch::default(),
            dimensions: (frame_width, frame_height),
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
            let window_secs = last_log.elapsed().as_secs_f64();
            #[allow(clippy::cast_precision_loss)] // u64 frame_count well under 2^53
            let avg_encode_ms = if frame_count > 0 {
                (encode_latency_sum_ns as f64 / frame_count as f64) / 1_000_000.0
            } else {
                0.0
            };
            #[allow(clippy::cast_precision_loss)]
            let kbps_out = if window_secs > 0.0 {
                (encoded_bytes_sum as f64 * 8.0 / 1000.0) / window_secs
            } else {
                0.0
            };
            let kf_per_s = if window_secs > 0.0 {
                f64::from(keyframe_count) / window_secs
            } else {
                0.0
            };
            info!(
                frames = frame_count,
                avg_encode_ms = format!("{avg_encode_ms:.2}"),
                kbps_out = format!("{kbps_out:.0}"),
                kf_per_s = format!("{kf_per_s:.2}"),
                "send stats"
            );
            frame_count = 0;
            encode_latency_sum_ns = 0;
            encoded_bytes_sum = 0;
            keyframe_count = 0;
            last_log = std::time::Instant::now();
        }
    }
    info!("send loop exiting");
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
