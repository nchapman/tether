//! Decoder worker thread and its core per-job logic.
//!
//! `run_thread` spawns the dedicated `std::thread`. `Worker::process_job`
//! is the pure per-job state machine — it takes an explicit `now:
//! MonoNanos` so tests can drive the IDR-rate-limit window deterministically
//! without spinning up a thread, channels, or paused tokio time.

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{Receiver as XbReceiver, Sender as XbSender};
use tether_codec::{build_decoder, bytes::Bytes, CodecError, Frame as CodecFrame};
use tether_protocol::control::VideoProfile;
use tether_protocol::MonoNanos;
use tether_render::{CpuFrame, Frame, GpuFrame as RenderGpuFrame, LatestFrame};
use tracing::{error, info, warn};

/// Minimum spacing between recovery-IDR requests. A corrupt stream
/// shouldn't turn into a keyframe storm; 500 ms matches the human
/// "is this still broken?" cadence and bounds wasted bitrate when
/// repeated decode failures arrive faster than the host could
/// possibly respond to.
///
/// Public so tests can reference the exact constant rather than
/// hard-coding 500.
pub const IDR_RATE_LIMIT: Duration = Duration::from_millis(500);

/// One frame's worth of work handed from the recv (tokio) task to the
/// decode (std::thread) worker. Keeping `host_in_client_clock` here
/// (already translated through clock_sync on the recv side) lets the
/// decode thread stamp each rendered frame without having to know
/// anything about the clock-sync handshake.
#[derive(Debug)]
pub struct DecodeJob {
    pub body: Bytes,
    pub host_in_client_clock: MonoNanos,
}

/// Per-frame metrics shipped back from the decode thread to the recv
/// loop, which owns the per-second stats log. `idr_request_fired`
/// reflects whether the worker *actually* invoked `request_idr`
/// (post rate-limit), not just whether it wanted to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeCompletion {
    pub decode_duration_ns: u64,
    pub decode_err: bool,
    pub soft_failure: bool,
    pub render_drops: u32,
    pub idr_request_fired: bool,
}

/// Per-thread mutable state plus its injected dependencies.
///
/// `request_idr` is fire-and-forget: errors from the underlying
/// `ControlMessage::ForceIdr` send are not surfaced. This matches the
/// production behavior at the original call site — a failure here is
/// logged inside the callback and the next decode error after the
/// rate-limit window re-triggers.
///
/// `warnings` returns the running count of libavcodec messages at
/// warning level or above. Production passes a closure wrapping
/// `tether_codec::av_log::warning_or_above_count`; tests pass a closure
/// over an `Arc<AtomicU64>` so they can simulate soft failures without
/// touching any global state and without breaking `cargo test`
/// parallelism.
pub struct Worker {
    decoder: Box<dyn tether_codec::Decoder>,
    frames: LatestFrame,
    request_idr: Arc<dyn Fn() + Send + Sync + 'static>,
    warnings: Arc<dyn Fn() -> u64 + Send + Sync + 'static>,
    last_idr_request: Option<MonoNanos>,
}

impl Worker {
    /// Construct a worker around an existing decoder. Used by tests
    /// that swap in a `FakeDecoder`. Production code goes through
    /// [`run_thread`], which builds the real decoder from a
    /// `VideoProfile`.
    pub fn new(
        decoder: Box<dyn tether_codec::Decoder>,
        frames: LatestFrame,
        request_idr: Arc<dyn Fn() + Send + Sync + 'static>,
        warnings: Arc<dyn Fn() -> u64 + Send + Sync + 'static>,
    ) -> Self {
        Self {
            decoder,
            frames,
            request_idr,
            warnings,
            last_idr_request: None,
        }
    }

    /// Decode one job and update the rendering / IDR state.
    ///
    /// `now` is the worker's view of wall-clock at the start of this
    /// job; production threads call `MonoNanos::now()` once per loop
    /// iteration. Tests supply a synthetic value to drive the rate
    /// limiter deterministically.
    pub fn process_job(&mut self, job: DecodeJob, now: MonoNanos) -> DecodeCompletion {
        // submit + drain. The trait swallows ffmpeg's drain/flushed
        // sentinels and returns them as `Ok(None)`, so any `Err` we
        // see here is a real decode failure — never a benign "need
        // more input".
        //
        // Some failure modes (HEVC RPS reconstruction — "Could not
        // find ref with POC N") don't surface through the API at all:
        // libavcodec internally logs the error and skips the
        // undecodable NALU, returning Ok. We bracket the call with a
        // snapshot of the warning-or-above counter so we can treat
        // "libavcodec was unhappy during this packet" as a soft
        // decode failure worth a ForceIdr, even when `decoder.submit`
        // / `next_frame` reported Ok.
        let warnings_before = (self.warnings)();
        let mut decoded: Vec<CodecFrame> = Vec::new();
        let mut decode_err: Option<tether_codec::CodecError> =
            self.decoder.submit(job.body.as_ref()).err();
        if decode_err.is_none() {
            loop {
                match self.decoder.next_frame() {
                    Ok(Some(f)) => decoded.push(f),
                    Ok(None) => break,
                    Err(e) => {
                        decode_err = Some(e);
                        break;
                    }
                }
            }
        }
        let warnings_after = (self.warnings)();
        let avlog_warnings = warnings_after.saturating_sub(warnings_before);
        let decode_duration_ns = MonoNanos::now().saturating_sub(now);

        // Render any frames we *did* successfully decode before
        // reporting the error. With async_depth=0 and no B-frames
        // this is almost always 0 or 1 frames; the loop is here so a
        // mid-drain failure doesn't silently throw away good output.
        let mut render_drops: u32 = 0;
        for dec in decoded {
            let raw = match dec {
                CodecFrame::Cpu(c) => Frame::Cpu(CpuFrame {
                    width: c.width,
                    height: c.height,
                    y: c.y,
                    uv: c.uv,
                    t_capture_client_clock: Some(job.host_in_client_clock),
                }),
                CodecFrame::Gpu(g) => {
                    let (w, h, _pts, source, guard) = g.into_parts();
                    Frame::Gpu(RenderGpuFrame {
                        width: w,
                        height: h,
                        t_capture_client_clock: Some(job.host_in_client_clock),
                        source,
                        guard,
                    })
                }
            };
            // LatestFrame.set displaces the previous frame; count each
            // displacement as a render drop (the renderer never saw
            // the displaced frame). On steady state with a
            // keeping-up renderer, `set` mostly returns None.
            if self.frames.set(raw).is_some() {
                render_drops = render_drops.saturating_add(1);
            }
        }

        // Two distinct failure shapes share the same recovery path.
        // (1) Hard failure: `decoder.submit` or `next_frame` returned
        //     an Err (rare; almost always a truly corrupt slice).
        // (2) Soft failure: libavcodec internally logged at warning
        //     or above during this packet — the common case for
        //     "P-frame references a fragment we dropped on the wire,"
        //     which the HEVC decoder skips silently until the next
        //     IDR. Either way, the only recovery is a fresh IDR.
        let soft_failure = decode_err.is_none() && avlog_warnings > 0;
        if let Some(e) = decode_err.as_ref() {
            warn!(error = %e, "decode failed; dropping packet");
        } else if soft_failure {
            // Don't emit per-packet — the av_log bridge already routed
            // the underlying libavcodec message into tracing. A trace
            // here keeps the metric attributable without doubling up.
            tracing::trace!(
                avlog_warnings,
                "libavcodec warned during decode; treating as soft failure"
            );
        }
        let mut idr_request_fired = false;
        if decode_err.is_some() || soft_failure {
            let rate_limit_ns = IDR_RATE_LIMIT.as_nanos() as u64;
            let fire = self
                .last_idr_request
                .is_none_or(|t| now.saturating_sub(t) > rate_limit_ns);
            if fire {
                (self.request_idr)();
                self.last_idr_request = Some(now);
                idr_request_fired = true;
            }
        }

        DecodeCompletion {
            decode_duration_ns,
            decode_err: decode_err.is_some(),
            soft_failure,
            render_drops,
            idr_request_fired,
        }
    }
}

/// Spawn the decoder worker thread.
///
/// Shutdown contract: the recv task drops its `XbSender<DecodeJob>` when
/// its spawn future returns; that causes `job_rx.recv()` to return
/// `Err` and the loop exits. The ctrl-c path uses `std::process::exit`
/// and bypasses this entirely; that's fine because the worker owns no
/// state needing flush.
///
/// `ready_tx` is sent on after the decoder is constructed; the recv
/// task awaits it before forwarding fragments. If the decoder build
/// fails, `ready_tx` is dropped without a send, which the recv task
/// treats as a fatal session error.
#[allow(clippy::too_many_arguments)]
pub fn run_thread(
    profile: VideoProfile,
    job_rx: XbReceiver<DecodeJob>,
    completion_tx: XbSender<DecodeCompletion>,
    frames: LatestFrame,
    request_idr: Arc<dyn Fn() + Send + Sync + 'static>,
    warnings: Arc<dyn Fn() -> u64 + Send + Sync + 'static>,
    ready_tx: tokio::sync::oneshot::Sender<()>,
) -> JoinHandle<()> {
    let decoder_init = build_decoder(profile);
    run_thread_with_init(
        profile,
        decoder_init,
        job_rx,
        completion_tx,
        frames,
        request_idr,
        warnings,
        ready_tx,
    )
}

/// Lower-level entry point that takes a pre-built decoder (or the
/// error from trying). Production calls [`run_thread`], which wraps
/// `build_decoder`. Tests inject an `Err(...)` here to exercise the
/// init-failure path without needing real hardware, and inject
/// `Ok(Box<FakeDecoder>)` to exercise the full thread lifecycle
/// against scripted decode outcomes.
#[allow(clippy::too_many_arguments)]
pub fn run_thread_with_init(
    profile: VideoProfile,
    decoder_init: Result<Box<dyn tether_codec::Decoder>, CodecError>,
    job_rx: XbReceiver<DecodeJob>,
    completion_tx: XbSender<DecodeCompletion>,
    frames: LatestFrame,
    request_idr: Arc<dyn Fn() + Send + Sync + 'static>,
    warnings: Arc<dyn Fn() -> u64 + Send + Sync + 'static>,
    ready_tx: tokio::sync::oneshot::Sender<()>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("tether-decode".into())
        .spawn(move || {
            let decoder: Box<dyn tether_codec::Decoder> = match decoder_init {
                Ok(d) => {
                    info!(
                        backend = d.name(),
                        hardware = d.is_hardware(),
                        codec = ?profile.codec,
                        chroma = ?profile.chroma,
                        bit_depth = profile.bit_depth,
                        "decoder initialised"
                    );
                    d
                }
                Err(e) => {
                    error!(error = %e, codec = ?profile.codec, "decoder init failed; aborting decode thread");
                    // Dropping ready_tx without sending signals the
                    // recv task that decoder construction failed; it
                    // tears the session down rather than streaming
                    // into a sink-hole.
                    drop(ready_tx);
                    return;
                }
            };
            let _ = ready_tx.send(());

            let mut worker = Worker::new(decoder, frames, request_idr, warnings);
            while let Ok(job) = job_rx.recv() {
                let now = MonoNanos::now();
                let completion = worker.process_job(job, now);
                // Send completion. If the recv loop has exited, the
                // receiver is dropped — that's expected at session
                // end; ignore.
                let _ = completion_tx.send(completion);
            }
            info!("decode thread exiting");
        })
        .expect("spawn tether-decode thread")
}
