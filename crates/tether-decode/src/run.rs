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

/// Consecutive transient decode failures the worker tolerates with
/// `decoder.flush()` between attempts before escalating to a full
/// rebuild. Three matches Moonlight's classification: a single
/// hiccup, a quick second, and one last benefit-of-the-doubt before
/// declaring the codec context wedged. The loop is responsible for
/// honouring `RecoveryAction::Rebuild` (calling `build_decoder` +
/// `replace_decoder`) and for tracking the per-session rebuild
/// budget.
pub const TRANSIENT_FAILURE_THRESHOLD: u32 = 3;

/// Hard cap on full decoder rebuilds in one session. Beyond this we
/// classify the failure as fatal — driver crash or persistent
/// bitstream corruption that no amount of restart can paper over —
/// and the thread exits cleanly via the Goodbye path rather than
/// looping forever.
pub const REBUILD_BUDGET: u32 = 10;

/// No-output watchdog window. If submits keep arriving but no
/// frame has been produced for this long, the decoder is wedged
/// — typical cause is HEVC RPS reconstruction silently skipping
/// every NALU because a key reference was lost on the wire. The
/// worker requests a flush + IDR on the first window expiry and
/// escalates to a rebuild on the second.
pub const NO_OUTPUT_WATCHDOG: Duration = Duration::from_millis(1500);

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
    /// Recovery escalation the run-thread should take after this job.
    /// `None` means "carry on"; `Some(Rebuild)` instructs the loop to
    /// tear the decoder down and call `build_decoder` again (subject
    /// to the rebuild budget).
    pub recovery: Option<RecoveryAction>,
}

/// What the worker is asking the run-thread to do beyond the
/// per-job rate-limited `request_idr` callback. Carried back inside
/// [`DecodeCompletion::recovery`] so the loop can branch on it
/// without inspecting the worker's private counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Consecutive transient failures crossed [`TRANSIENT_FAILURE_THRESHOLD`].
    /// `Worker::flush()` did not unwedge the codec context; the loop
    /// should call `build_decoder(profile)` and hand the new
    /// instance back via [`Worker::replace_decoder`]. Subject to
    /// [`REBUILD_BUDGET`].
    Rebuild,
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
    /// Consecutive failures (hard `decode_err` OR soft warning bump)
    /// since the last successfully-rendered frame. Resets to 0 on
    /// any job that produces ≥1 frame and triggers no warning bump.
    consecutive_failures: u32,
    /// Wall-clock instant of the most recent successful render-side
    /// frame. Drives the no-output watchdog; `None` until the very
    /// first frame lands.
    last_successful_decode: Option<MonoNanos>,
    /// Has the watchdog already fired in the current "no output"
    /// window? Prevents a stuck decoder from triggering a flush +
    /// IDR on every job — one shot per window, then escalate.
    watchdog_armed: bool,
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
            consecutive_failures: 0,
            last_successful_decode: None,
            watchdog_armed: false,
        }
    }

    /// Swap in a freshly-built decoder after a recovery rebuild.
    /// Resets failure counters so the new instance starts with a
    /// clean slate.
    pub fn replace_decoder(&mut self, new_decoder: Box<dyn tether_codec::Decoder>) {
        self.decoder = new_decoder;
        self.consecutive_failures = 0;
        self.watchdog_armed = false;
        // `last_successful_decode` stays as-is: the watchdog clock
        // shouldn't reset just because we rebuilt; if the rebuild
        // doesn't produce a frame either, we want the *original*
        // wedge window to escalate to fatal promptly.
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
        let produced_frame = !decoded.is_empty();
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

        // Update failure / watchdog tracking. If the decoder emitted
        // at least one frame this packet, count it as success even
        // when libavcodec also logged a warning during the decode
        // (RPS reconstruction skipped one NALU but the rest of the
        // GOP still came through). Flushing on a partially-good
        // packet would discard the decoder's reference-frame pool
        // and turn one imperfect frame into a guaranteed cascade
        // of failures. The IDR request is already queued above —
        // the next IDR will rebuild the reference set cleanly.
        if produced_frame {
            self.consecutive_failures = 0;
            self.last_successful_decode = Some(now);
            self.watchdog_armed = false;
        } else if decode_err.is_some() || soft_failure {
            self.consecutive_failures = self.consecutive_failures.saturating_add(1);
            // Transient recovery: flush the codec context between
            // failures so a wedged state (lost SPS reference, stale
            // RPS) doesn't carry across packets. flush() is cheap
            // and idempotent; ignore its error.
            if let Err(e) = self.decoder.flush() {
                tracing::warn!(error = %e, "decoder flush failed during transient recovery");
            }
        }

        // No-output watchdog. If the most recent successful frame
        // is older than the watchdog window AND submits keep
        // arriving (we're in process_job, so yes), fire an extra
        // request_idr + flush — one-shot per window — and escalate
        // to a rebuild if the wedge persists past two windows.
        let recovery = self.evaluate_recovery(now);

        DecodeCompletion {
            decode_duration_ns,
            decode_err: decode_err.is_some(),
            soft_failure,
            render_drops,
            idr_request_fired,
            recovery,
        }
    }

    fn evaluate_recovery(&mut self, now: MonoNanos) -> Option<RecoveryAction> {
        // Transient threshold crossed → ask the loop to rebuild.
        if self.consecutive_failures >= TRANSIENT_FAILURE_THRESHOLD {
            return Some(RecoveryAction::Rebuild);
        }

        // Watchdog: only meaningful once we've decoded at least one
        // frame and have a baseline. The first IDR after connect
        // hasn't landed yet otherwise; no-output is expected.
        let Some(last) = self.last_successful_decode else {
            return None;
        };
        let elapsed_ns = now.saturating_sub(last);
        let window_ns = NO_OUTPUT_WATCHDOG.as_nanos() as u64;
        if elapsed_ns < window_ns {
            return None;
        }

        if !self.watchdog_armed {
            // First window expiry: flush + IDR, arm the watchdog
            // so we don't re-fire on the very next packet. The IDR
            // rate-limit still applies; if it just fired we skip
            // here too.
            self.watchdog_armed = true;
            if let Err(e) = self.decoder.flush() {
                tracing::warn!(error = %e, "decoder flush failed during watchdog recovery");
            }
            let rate_limit_ns = IDR_RATE_LIMIT.as_nanos() as u64;
            let fire = self
                .last_idr_request
                .is_none_or(|t| now.saturating_sub(t) > rate_limit_ns);
            if fire {
                (self.request_idr)();
                self.last_idr_request = Some(now);
            }
            tracing::warn!(
                elapsed_ms = elapsed_ns / 1_000_000,
                "decoder produced no output for {NO_OUTPUT_WATCHDOG:?}; flushed and requested IDR"
            );
            return None;
        }

        // Watchdog already armed, still no output one window
        // later → escalate.
        if elapsed_ns >= window_ns.saturating_mul(2) {
            tracing::warn!(
                elapsed_ms = elapsed_ns / 1_000_000,
                "decoder still produced no output after watchdog flush; requesting rebuild"
            );
            return Some(RecoveryAction::Rebuild);
        }
        None
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
            let mut rebuilds_used: u32 = 0;
            while let Ok(job) = job_rx.recv() {
                let now = MonoNanos::now();
                let completion = worker.process_job(job, now);
                let recovery = completion.recovery;
                // Send completion. If the recv loop has exited, the
                // receiver is dropped — that's expected at session
                // end; ignore.
                let _ = completion_tx.send(completion);

                if matches!(recovery, Some(RecoveryAction::Rebuild)) {
                    if rebuilds_used >= REBUILD_BUDGET {
                        error!(
                            rebuilds_used,
                            budget = REBUILD_BUDGET,
                            "decoder rebuild budget exhausted; exiting decode thread"
                        );
                        break;
                    }
                    rebuilds_used = rebuilds_used.saturating_add(1);
                    // `profile` is fixed for the session lifetime
                    // today; any future per-session renegotiation
                    // requires a full session restart (the
                    // surrounding QUIC session is torn down when
                    // the decode thread exits).
                    match build_decoder(profile) {
                        Ok(new) => {
                            info!(
                                rebuilds_used,
                                backend = new.name(),
                                "decoder rebuilt after persistent failure"
                            );
                            worker.replace_decoder(new);
                        }
                        Err(e) => {
                            error!(
                                error = %e,
                                rebuilds_used,
                                "decoder rebuild failed; exiting decode thread"
                            );
                            break;
                        }
                    }
                }
            }
            info!("decode thread exiting");
        })
        .expect("spawn tether-decode thread")
}
