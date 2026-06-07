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

/// Minimum spacing between epoch-driven decoder rebuilds. An epoch bump means
/// the host changed resolution / codec / hw-context; each rebuild tears down
/// and recreates the hardware decode session (D3D11VA on Windows, VAAPI on
/// Linux, VideoToolbox on macOS), which is fragile under rapid churn on some
/// drivers (e.g. AMD RDNA 4). Legitimate resizes
/// are already debounced ~150 ms client-side, so this only bites a host that
/// bumps the epoch every frame (drag-resize edge or a misbehaving host): such
/// new-epoch frames are dropped until the window passes, then we rebuild to the
/// newest epoch. Below the 150 ms viewport debounce so normal resizes never
/// drop.
pub const EPOCH_REBUILD_MIN_INTERVAL: Duration = Duration::from_millis(100);

/// One frame's worth of work handed from the recv (tokio) task to the
/// decode (std::thread) worker. Keeping `host_in_client_clock` here
/// (already translated through clock_sync on the recv side) lets the
/// decode thread stamp each rendered frame without having to know
/// anything about the clock-sync handshake.
#[derive(Debug)]
pub struct DecodeJob {
    pub body: Bytes,
    pub host_in_client_clock: MonoNanos,
    /// Whether this frame is a keyframe (IDR). Used by the worker to
    /// skip non-IDR frames after a decoder rebuild — a fresh decoder
    /// has no parameter sets and will fail on P-frames.
    pub keyframe: bool,
    /// Wire `stream_epoch` this frame belongs to. The host bumps it on a
    /// resolution / codec / hw-context change (see `VideoPacket`); when it
    /// changes, the decode worker rebuilds the decoder so the new-epoch IDR
    /// lands in a fresh context. Feeding a mid-stream resolution change into
    /// a running D3D11VA AV1 decoder corrupts/crashes it on some GPUs
    /// (observed: AMD RDNA 4), so a clean rebuild is required, not an
    /// in-place reconfigure.
    pub stream_epoch: u32,
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

/// Message emitted by the decode thread to the recv loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeEvent {
    /// A job reached [`Worker::process_job`]; fold into decode latency and
    /// recovery stats.
    Completion(DecodeCompletion),
    /// A reassembled frame was intentionally discarded before decode because
    /// its stream epoch could not be submitted to the current decoder.
    EpochDrop {
        reason: EpochDropReason,
        stream_epoch: u32,
    },
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

/// What the run-loop should do with a job given its wire `stream_epoch`,
/// returned by [`Worker::classify_epoch`]. The epoch is host-monotonic, so it
/// is treated as a high-water mark: only a forward advance rebuilds; an older
/// epoch is a late straggler and must never reach the (new-resolution) decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpochAction {
    /// Same epoch as the current decoder — decode normally.
    Decode,
    /// Epoch advanced — rebuild the decoder, then decode this (new-epoch) frame.
    Rebuild,
    /// Drop without decoding. Carries the reason so stats can distinguish a
    /// harmless late straggler from resize/reconfigure churn.
    Drop(EpochDropReason),
}

/// Why an epoch-classified job was dropped before decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpochDropReason {
    /// Older than the decoder epoch high-water mark; typically a late FEC
    /// completion from before a resize/reconfigure.
    Stale,
    /// Newer than the decoder epoch, but arrived inside
    /// [`EPOCH_REBUILD_MIN_INTERVAL`]; dropped to bound hardware-context churn.
    RebuildRateLimited,
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
    /// Set after a decoder rebuild. While true, non-IDR frames are
    /// discarded without submission — a freshly-built decoder has no
    /// parameter sets and will fail on P-frames, burning through the
    /// rebuild budget in a death spiral. Cleared on the first IDR.
    awaiting_idr: bool,
    /// P-frames discarded since entering the current `awaiting_idr`
    /// window. Drives a re-request of the keyframe so a lost or
    /// un-honoured initial IDR can't strand the stream indefinitely;
    /// reset to 0 when an IDR finally arrives.
    awaiting_idr_discards: u64,
    /// Highest wire `stream_epoch` seen (host-monotonic high-water mark). The
    /// host bumps it on a resolution / codec / hw-context change; an advance
    /// means the decoder must be rebuilt before the new-epoch IDR is fed in
    /// (see [`Self::classify_epoch`]). `None` until the first job.
    current_epoch: Option<u32>,
    /// When the last epoch-driven rebuild happened, to rate-limit rebuilds
    /// (see [`EPOCH_REBUILD_MIN_INTERVAL`]). `None` until the first job.
    last_epoch_rebuild: Option<MonoNanos>,
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
            awaiting_idr: false,
            awaiting_idr_discards: 0,
            current_epoch: None,
            last_epoch_rebuild: None,
        }
    }

    /// Classify a job by its wire `stream_epoch` against the current decoder's
    /// epoch, deciding whether to decode, rebuild-then-decode, or drop.
    ///
    /// The epoch is host-monotonic (bumped on a resolution / codec / hw-context
    /// change), so it is a high-water mark:
    ///   - **Same** epoch → [`EpochAction::Decode`].
    ///   - **Advance** → [`EpochAction::Rebuild`]: rebuild the decoder BEFORE
    ///     submitting, so the new-resolution IDR initialises a fresh decoder
    ///     rather than forcing an in-place reconfigure (which corrupts/crashes
    ///     the D3D11VA AV1 decoder on AMD RDNA 4). Rate-limited by
    ///     [`EPOCH_REBUILD_MIN_INTERVAL`]: an advance arriving too soon after
    ///     the last rebuild is **dropped** (not latched), so a host bumping the
    ///     epoch every frame can't drive unbounded device churn — a later frame
    ///     rebuilds to the newest epoch once the window passes.
    ///   - **Older** epoch → [`EpochAction::Drop`]: a late straggler (e.g. a
    ///     FEC-completed pre-resize frame). It must never reach the
    ///     new-resolution decoder. The reassembler keys per epoch and has no
    ///     global watermark, so these can be delivered after an advance.
    ///
    /// The first job is always `Decode` (no prior epoch to compare). On a
    /// `Rebuild` the high-water mark is NOT advanced here — the caller must
    /// call [`Self::confirm_epoch_rebuilt`] *after* the decoder is actually
    /// rebuilt, so a failed rebuild can't strand `current_epoch` ahead of the
    /// decoder and let the next same-epoch frame decode on the wrong context.
    pub fn classify_epoch(&mut self, epoch: u32, now: MonoNanos) -> EpochAction {
        match self.current_epoch {
            None => {
                // First job: establish the baseline. The initial decoder
                // already exists, so this is a Decode, not a Rebuild.
                self.current_epoch = Some(epoch);
                self.last_epoch_rebuild = Some(now);
                EpochAction::Decode
            }
            Some(cur) if epoch == cur => EpochAction::Decode,
            // Wrap-safe forward compare: `epoch` is newer than `cur`.
            Some(cur) if epoch.wrapping_sub(cur) < u32::MAX / 2 => {
                // EPOCH_REBUILD_MIN_INTERVAL is a small constant; nanos fit u64.
                #[allow(clippy::cast_possible_truncation)]
                let min_interval_ns = EPOCH_REBUILD_MIN_INTERVAL.as_nanos() as u64;
                if self
                    .last_epoch_rebuild
                    .is_some_and(|t| now.saturating_sub(t) < min_interval_ns)
                {
                    // Too soon after the last rebuild — drop without latching so
                    // a later frame still rebuilds to the newest epoch.
                    return EpochAction::Drop(EpochDropReason::RebuildRateLimited);
                }
                // Latching deferred to `confirm_epoch_rebuilt` (after success).
                EpochAction::Rebuild
            }
            // `epoch` older than the high-water mark: a late straggler.
            Some(_) => EpochAction::Drop(EpochDropReason::Stale),
        }
    }

    /// Advance the epoch high-water mark after a successful epoch-driven
    /// rebuild (the [`EpochAction::Rebuild`] arm of [`Self::classify_epoch`]).
    /// Kept separate so the mark only moves once the decoder for the new epoch
    /// actually exists — a `build_decoder` failure leaves the mark unchanged.
    pub fn confirm_epoch_rebuilt(&mut self, epoch: u32, now: MonoNanos) {
        self.current_epoch = Some(epoch);
        self.last_epoch_rebuild = Some(now);
    }

    /// Gate the worker on its first IDR before any P-frame is fed.
    ///
    /// A freshly-built decoder has no parameter sets, so the P-frames
    /// that race ahead of the first keyframe on a new connection (all
    /// frames share the datagram channel, and a P-frame can arrive or
    /// FEC-complete before the larger first IDR does) only produce
    /// `PPS id out of range` errors and a destructive rebuild — which
    /// then discards the IDR that's arriving. Starting gated makes the
    /// worker discard those racing
    /// P-frames and cleanly latch the first keyframe, the same way
    /// Moonlight and RustDesk gate a fresh decoder on its first IDR.
    /// `replace_decoder` already gates after a rebuild; this gates the
    /// very first decoder too.
    pub fn gate_on_first_idr(&mut self) {
        self.awaiting_idr = true;
        self.awaiting_idr_discards = 0;
    }

    /// Swap in a freshly-built decoder after a recovery rebuild.
    /// Resets failure counters so the new instance starts with a
    /// clean slate.
    pub fn replace_decoder(&mut self, new_decoder: Box<dyn tether_codec::Decoder>) {
        self.decoder = new_decoder;
        self.consecutive_failures = 0;
        self.watchdog_armed = false;
        self.awaiting_idr = true;
        self.awaiting_idr_discards = 0;
        // Clear the IDR rate-limiter: a fresh decoder must be able to
        // re-request its keyframe immediately. Without this, a rebuild that
        // lands inside the 500 ms window of a prior request (e.g. a resize
        // right after a recovery IDR) would sit gated awaiting an IDR it isn't
        // allowed to re-request yet.
        self.last_idr_request = None;
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
        // After a rebuild, discard P-frames until an IDR arrives.
        // A fresh decoder has no parameter sets; feeding it non-IDR
        // data produces immediate "PPS id out of range" errors that
        // burn through the rebuild budget.
        //
        // Detection: check both the wire-level `keyframe` flag AND
        // the bitstream itself. Some encoders (AMF) don't set
        // AV_PKT_FLAG_KEY on forced IDRs, so the wire flag can be
        // false even when the bitstream IS a self-decodable IDR.
        // drain_encoder prepends VPS/SPS/PPS extradata to every
        // keyframe, so we detect IDRs by looking for Annex-B
        // parameter sets at the start of the body.
        if self.awaiting_idr {
            if job.keyframe || body_starts_with_parameter_sets(&job.body) {
                info!(
                    discarded_p_frames = self.awaiting_idr_discards,
                    "IDR acquired; resuming decode"
                );
                self.awaiting_idr = false;
                self.awaiting_idr_discards = 0;
            } else {
                // No IDR yet — re-request one (rate-limited). The initial
                // ForceIdr can be lost or arrive before the host's
                // encoder is ready to honour it; without re-requesting,
                // this early return bypasses the failure + watchdog paths
                // that would otherwise recover, stranding the stream on a
                // green screen until the user happens to resize (which
                // forces a host IDR). The 500 ms rate limit prevents a
                // keyframe storm while P-frames pour in at frame rate.
                self.awaiting_idr_discards = self.awaiting_idr_discards.saturating_add(1);
                let fired = self.try_fire_idr(now);
                if fired {
                    warn!(
                        discarded_p_frames = self.awaiting_idr_discards,
                        keyframe_flag = job.keyframe,
                        "awaiting IDR; re-requesting keyframe (P-frames discarded while gated)"
                    );
                }
                return DecodeCompletion {
                    idr_request_fired: fired,
                    ..DecodeCompletion::default()
                };
            }
        }

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
            idr_request_fired |= self.try_fire_idr(now);
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
        let (recovery, watchdog_fired_idr) = self.evaluate_recovery(now);
        idr_request_fired |= watchdog_fired_idr;

        DecodeCompletion {
            decode_duration_ns,
            decode_err: decode_err.is_some(),
            soft_failure,
            render_drops,
            idr_request_fired,
            recovery,
        }
    }

    /// Rate-limited `request_idr` invocation. Returns `true` if the
    /// callback actually fired (i.e. the rate-limit window had elapsed
    /// since the last request). Shared by the per-job failure path and
    /// the watchdog so both update `DecodeCompletion::idr_request_fired`
    /// consistently.
    fn try_fire_idr(&mut self, now: MonoNanos) -> bool {
        // IDR_RATE_LIMIT is a small constant Duration; its nanos fit in u64.
        #[allow(clippy::cast_possible_truncation)]
        let rate_limit_ns = IDR_RATE_LIMIT.as_nanos() as u64;
        let fire = self
            .last_idr_request
            .is_none_or(|t| now.saturating_sub(t) > rate_limit_ns);
        if fire {
            (self.request_idr)();
            self.last_idr_request = Some(now);
        }
        fire
    }

    /// Returns `(recovery, watchdog_fired_idr)`. The second element
    /// reflects whether the watchdog arm path actually invoked
    /// `request_idr` this tick — surfaced in `DecodeCompletion` so the
    /// recv-loop's per-second stats account for IDRs from both the
    /// failure and watchdog paths.
    fn evaluate_recovery(&mut self, now: MonoNanos) -> (Option<RecoveryAction>, bool) {
        // Transient threshold crossed → ask the loop to rebuild.
        if self.consecutive_failures >= TRANSIENT_FAILURE_THRESHOLD {
            return (Some(RecoveryAction::Rebuild), false);
        }

        // Watchdog: only meaningful once we've decoded at least one
        // frame and have a baseline. The first IDR after connect
        // hasn't landed yet otherwise; no-output is expected.
        let Some(last) = self.last_successful_decode else {
            return (None, false);
        };
        let elapsed_ns = now.saturating_sub(last);
        // NO_OUTPUT_WATCHDOG is a small constant Duration; its nanos fit in u64.
        #[allow(clippy::cast_possible_truncation)]
        let window_ns = NO_OUTPUT_WATCHDOG.as_nanos() as u64;
        if elapsed_ns < window_ns {
            return (None, false);
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
            let fired = self.try_fire_idr(now);
            tracing::warn!(
                elapsed_ms = elapsed_ns / 1_000_000,
                idr_fired = fired,
                "decoder produced no output for {NO_OUTPUT_WATCHDOG:?}; flushed and requested IDR"
            );
            return (None, fired);
        }

        // Watchdog already armed, still no output one window
        // later → escalate.
        if elapsed_ns >= window_ns.saturating_mul(2) {
            tracing::warn!(
                elapsed_ms = elapsed_ns / 1_000_000,
                "decoder still produced no output after watchdog flush; requesting rebuild"
            );
            return (Some(RecoveryAction::Rebuild), false);
        }
        (None, false)
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
    completion_tx: XbSender<DecodeEvent>,
    frames: LatestFrame,
    request_idr: Arc<dyn Fn() + Send + Sync + 'static>,
    warnings: Arc<dyn Fn() -> u64 + Send + Sync + 'static>,
    ready_tx: tokio::sync::oneshot::Sender<()>,
    gpu_export: bool,
) -> JoinHandle<()> {
    let decoder_init = build_decoder(profile, gpu_export);
    run_thread_with_init(
        profile,
        decoder_init,
        job_rx,
        completion_tx,
        frames,
        request_idr,
        warnings,
        ready_tx,
        gpu_export,
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
    completion_tx: XbSender<DecodeEvent>,
    frames: LatestFrame,
    request_idr: Arc<dyn Fn() + Send + Sync + 'static>,
    warnings: Arc<dyn Fn() -> u64 + Send + Sync + 'static>,
    ready_tx: tokio::sync::oneshot::Sender<()>,
    // Renderer's zero-copy import capability, forwarded to `build_decoder`
    // on rebuild so a rebuilt Windows decoder keeps the same export mode.
    gpu_export: bool,
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
            // Gate on the first IDR: a fresh connection's P-frames can
            // arrive or FEC-complete before the larger first keyframe on
            // the shared datagram channel, and feeding them to a virgin
            // decoder would force a destructive rebuild that loses the
            // inbound IDR. Discard until the keyframe lands instead.
            worker.gate_on_first_idr();
            let mut rebuilds_used: u32 = 0;
            while let Ok(job) = job_rx.recv() {
                let now = MonoNanos::now();
                // Classify by wire epoch BEFORE submitting. An advance means the
                // host changed resolution/codec/hw-context: rebuild so the
                // new-resolution IDR initialises a clean decoder instead of
                // forcing an in-place reconfigure (which corrupts/crashes the
                // D3D11VA AV1 decoder on AMD RDNA 4). Distinct from the
                // failure-driven rebuild below and not subject to its budget —
                // epoch changes are expected (resizes), not faults.
                match worker.classify_epoch(job.stream_epoch, now) {
                    EpochAction::Decode => {}
                    EpochAction::Rebuild => match build_decoder(profile, gpu_export) {
                        Ok(new) => {
                            info!(
                                new_epoch = job.stream_epoch,
                                backend = new.name(),
                                "stream epoch advanced; rebuilt decoder for clean reconfigure"
                            );
                            worker.replace_decoder(new);
                            // Advance the high-water mark only now that the new
                            // decoder exists (see `classify_epoch`).
                            worker.confirm_epoch_rebuilt(job.stream_epoch, now);
                        }
                        Err(e) => {
                            error!(error = %e, "decoder rebuild on epoch change failed; exiting decode thread");
                            break;
                        }
                    },
                    EpochAction::Drop(reason) => {
                        // Old-epoch straggler, or a new-epoch frame inside the
                        // rebuild rate-limit window. Don't feed it to the
                        // decoder and don't emit a DecodeCompletion, because
                        // completions are the avg/min/max decode-latency
                        // divisor. Emit a separate event so live stats still
                        // explain why recv frames did not become decode work.
                        let _ = completion_tx.send(DecodeEvent::EpochDrop {
                            reason,
                            stream_epoch: job.stream_epoch,
                        });
                        continue;
                    }
                }

                let completion = worker.process_job(job, now);
                let recovery = completion.recovery;
                // Send completion. If the recv loop has exited, the
                // receiver is dropped — that's expected at session
                // end; ignore.
                let _ = completion_tx.send(DecodeEvent::Completion(completion));

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
                    match build_decoder(profile, gpu_export) {
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

/// Check if a frame body starts with Annex-B parameter sets, indicating
/// a self-decodable IDR. `drain_encoder` prepends extradata (VPS/SPS/PPS
/// for HEVC, SPS/PPS for H.264) to every keyframe, so the first NALU
/// will be a parameter set. This is more reliable than the wire-level
/// `keyframe` flag, which some encoders (AMF) fail to set on forced IDRs.
fn body_starts_with_parameter_sets(body: &[u8]) -> bool {
    // Find the first NALU type after the Annex-B start code.
    let nalu_byte =
        if body.len() >= 5 && body[0] == 0 && body[1] == 0 && body[2] == 0 && body[3] == 1 {
            Some(body[4])
        } else if body.len() >= 4 && body[0] == 0 && body[1] == 0 && body[2] == 1 {
            Some(body[3])
        } else {
            None
        };
    // No Annex-B start code → not HEVC/H.264; try AV1 OBU framing instead.
    let Some(b) = nalu_byte else {
        return av1_has_sequence_header(body);
    };
    // HEVC: VPS = type 32, first byte = (32 << 1) | 0 = 0x40.
    // NALU type is bits [6:1] of the first byte.
    let hevc_type = (b >> 1) & 0x3F;
    if hevc_type == 32 || hevc_type == 33 || hevc_type == 34 {
        return true;
    }
    // H.264: SPS = type 7, PPS = type 8.
    // NALU type is bits [4:0] of the first byte.
    let h264_type = b & 0x1F;
    h264_type == 7 || h264_type == 8
}

/// True if an AV1 low-overhead bitstream carries a sequence-header OBU (type 1)
/// near the start — the in-band parameter set a keyframe temporal unit always
/// includes. AV1 has no Annex-B start codes, and AMF / VAAPI don't reliably set
/// the wire keyframe flag, so this is the AV1 analog of the HEVC/H.264 NAL check
/// that lets the worker recognise an IDR even when the flag is missing. Bounded
/// scan of the first few OBUs. OBU header: bit7 forbidden, bits6-3 type, bit2
/// extension, bit1 has_size, bit0 reserved; optional 1-byte extension; leb128
/// size.
fn av1_has_sequence_header(body: &[u8]) -> bool {
    let mut i = 0usize;
    // A keyframe TU leads with a temporal delimiter then the sequence header;
    // a handful of OBUs is plenty — don't walk the whole frame.
    for _ in 0..4 {
        let Some(&hdr) = body.get(i) else {
            return false;
        };
        if hdr & 0x80 != 0 {
            return false; // forbidden bit set — not a valid OBU stream
        }
        let obu_type = (hdr >> 3) & 0x0f;
        if obu_type == 1 {
            return true; // OBU_SEQUENCE_HEADER
        }
        let ext = (hdr >> 2) & 1 == 1;
        let has_size = (hdr >> 1) & 1 == 1;
        if !has_size {
            return false; // can't advance without sizes
        }
        let mut p = i + 1 + usize::from(ext);
        // leb128 OBU size.
        let mut size = 0usize;
        let mut shift = 0u32;
        loop {
            let Some(&byte) = body.get(p) else {
                return false;
            };
            p += 1;
            size |= ((byte & 0x7f) as usize) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 28 {
                return false; // malformed / unbounded leb128
            }
        }
        i = p.saturating_add(size);
    }
    false
}
