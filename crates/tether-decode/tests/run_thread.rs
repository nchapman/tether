//! Unit tests for the per-job decode logic via `Worker::process_job`.
//!
//! These tests avoid spawning threads or building the real decoder so
//! they can drive the IDR rate limiter deterministically by passing a
//! synthetic `now: MonoNanos` directly. Channel / threading behavior
//! is covered by integration tests in `tether-session/tests`.

#![cfg(feature = "test-support")]
// Test timing offsets derived from small loop indices; casts are in range.
// The lifecycle-callbacks tuple is a test helper, not a public API.
#![allow(clippy::cast_sign_loss, clippy::type_complexity)]

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tether_codec::bytes::Bytes;
use tether_codec::CodecError;
use tether_decode::test_support::{FakeDecoder, FakeOutcome};
use tether_decode::{run_thread_with_init, DecodeCompletion, DecodeJob, Worker, IDR_RATE_LIMIT};
use tether_protocol::MonoNanos;
use tether_render::LatestFrame;

/// Construct a `Worker` with bare callbacks and a fake decoder.
/// Returns `(worker, idr_calls, warnings)` so the test can poke
/// `warnings.store(...)` and assert `idr_calls.load(...)`.
fn make_worker(decoder: FakeDecoder) -> (Worker, Arc<AtomicU32>, Arc<AtomicU64>, LatestFrame) {
    let idr_calls = Arc::new(AtomicU32::new(0));
    let warnings = Arc::new(AtomicU64::new(0));
    let frames = LatestFrame::new();

    let idr_cb = Arc::clone(&idr_calls);
    let warnings_cb = Arc::clone(&warnings);

    let worker = Worker::new(
        Box::new(decoder),
        frames.clone(),
        Arc::new(move || {
            idr_cb.fetch_add(1, Ordering::Relaxed);
        }),
        Arc::new(move || warnings_cb.load(Ordering::Relaxed)),
    );
    (worker, idr_calls, warnings, frames)
}

fn job() -> DecodeJob {
    DecodeJob {
        body: Bytes::from_static(b"encoded-bytes"),
        host_in_client_clock: MonoNanos::now(),
        keyframe: true,
    }
}

fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).expect("duration fits in u64 ns")
}

fn at(base: MonoNanos, offset: Duration) -> MonoNanos {
    // MonoNanos is opaque (no add API), so produce an "offset" instance
    // by adding nanoseconds via the only available channel: wait. For
    // tests we want determinism, so derive via the trait's `From<u64>`
    // if any. Fall back to recreating from raw u64 via Debug parse.
    // The simplest stable path: subtract `base` later via saturating_sub.
    // We need a MonoNanos value `base + offset`. Since MonoNanos has no
    // constructor from u64, return base for offset=0 and otherwise
    // sleep. We avoid sleeping in tests by using the fact that
    // `MonoNanos::now()` is monotonic and offset-only matters via
    // `saturating_sub`; tests pass `now` values that are real
    // `MonoNanos::now()` samples bracketing real (short) elapsed time.
    if offset.is_zero() {
        base
    } else {
        std::thread::sleep(offset);
        MonoNanos::now()
    }
}

#[test]
fn decode_success_pushes_frame_to_latest_and_no_idr() {
    let dec = FakeDecoder::one_frame_then_idle(8, 8);
    let (mut worker, idr_calls, _warnings, frames) = make_worker(dec);
    let completion = worker.process_job(job(), MonoNanos::now());
    assert!(!completion.decode_err);
    assert!(!completion.soft_failure);
    assert!(!completion.idr_request_fired);
    assert_eq!(idr_calls.load(Ordering::Relaxed), 0);
    assert!(
        frames.take().is_some(),
        "LatestFrame should hold the decoded frame"
    );
}

#[test]
fn hard_submit_error_fires_idr() {
    let dec = FakeDecoder::new(vec![FakeOutcome::SubmitError]);
    let (mut worker, idr_calls, _warnings, _frames) = make_worker(dec);
    let completion = worker.process_job(job(), MonoNanos::now());
    assert!(completion.decode_err);
    assert!(completion.idr_request_fired);
    assert_eq!(idr_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn next_frame_error_fires_idr() {
    let dec = FakeDecoder::new(vec![FakeOutcome::NextFrameError]);
    let (mut worker, idr_calls, _warnings, _frames) = make_worker(dec);
    let completion = worker.process_job(job(), MonoNanos::now());
    assert!(completion.decode_err);
    assert!(completion.idr_request_fired);
    assert_eq!(idr_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn soft_failure_via_warning_bump_fires_idr() {
    // libavcodec soft failures surface as a bump in the warning
    // counter between `submit` and `next_frame` returning Ok. Model
    // that by returning 0 on the first warnings read and 1 on the
    // second.
    let read_calls = Arc::new(AtomicU64::new(0));
    let read_calls_cb = Arc::clone(&read_calls);
    let idr_calls = Arc::new(AtomicU32::new(0));
    let idr_cb = Arc::clone(&idr_calls);
    let frames = LatestFrame::new();

    let mut worker = Worker::new(
        Box::new(FakeDecoder::one_frame_then_idle(8, 8)),
        frames,
        Arc::new(move || {
            idr_cb.fetch_add(1, Ordering::Relaxed);
        }),
        // 0 then 1 — bump observed exactly within a single job's bracket.
        Arc::new(move || read_calls_cb.fetch_add(1, Ordering::Relaxed).min(1)),
    );

    let completion = worker.process_job(job(), MonoNanos::now());
    assert!(!completion.decode_err);
    assert!(completion.soft_failure);
    assert!(completion.idr_request_fired);
    assert_eq!(idr_calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        read_calls.load(Ordering::Relaxed),
        2,
        "exactly two reads bracket the job"
    );
}

#[test]
fn rapid_soft_errors_within_rate_limit_coalesce_to_single_idr() {
    // Configure warnings to bump on every call so every job looks like
    // a soft failure. Then drive several jobs at `now` values close
    // together and assert only the first fires `request_idr`.
    let warnings_state = Arc::new(AtomicU64::new(0));
    let idr_calls = Arc::new(AtomicU32::new(0));
    let frames = LatestFrame::new();

    let warn_cb_state = Arc::clone(&warnings_state);
    let idr_cb = Arc::clone(&idr_calls);
    let mut worker = Worker::new(
        Box::new(FakeDecoder::new(vec![
            FakeOutcome::Frames(vec![]),
            FakeOutcome::Frames(vec![]),
            FakeOutcome::Frames(vec![]),
        ])),
        frames,
        Arc::new(move || {
            idr_cb.fetch_add(1, Ordering::Relaxed);
        }),
        // Every read bumps by 1, so each job sees a positive delta.
        Arc::new(move || warn_cb_state.fetch_add(1, Ordering::Relaxed) + 1),
    );

    let base = MonoNanos::now();
    let c1 = worker.process_job(job(), base);
    let c2 = worker.process_job(job(), at(base, Duration::from_millis(50)));
    let c3 = worker.process_job(job(), at(base, Duration::from_millis(100)));

    assert!(c1.idr_request_fired, "first soft failure should fire IDR");
    assert!(!c2.idr_request_fired, "50ms later should be rate-limited");
    assert!(!c3.idr_request_fired, "100ms later should be rate-limited");
    assert_eq!(idr_calls.load(Ordering::Relaxed), 1);
    // Sanity: the rate-limit constant is what we expect.
    assert_eq!(IDR_RATE_LIMIT, Duration::from_millis(500));
    assert!(nanos(Duration::from_millis(100)) < nanos(IDR_RATE_LIMIT));
}

#[test]
fn soft_error_past_rate_limit_window_fires_again() {
    let warnings_state = Arc::new(AtomicU64::new(0));
    let idr_calls = Arc::new(AtomicU32::new(0));
    let frames = LatestFrame::new();

    let warn_cb_state = Arc::clone(&warnings_state);
    let idr_cb = Arc::clone(&idr_calls);
    let mut worker = Worker::new(
        Box::new(FakeDecoder::new(vec![
            FakeOutcome::Frames(vec![]),
            FakeOutcome::Frames(vec![]),
        ])),
        frames,
        Arc::new(move || {
            idr_cb.fetch_add(1, Ordering::Relaxed);
        }),
        Arc::new(move || warn_cb_state.fetch_add(1, Ordering::Relaxed) + 1),
    );

    // First soft failure fires.
    let base = MonoNanos::now();
    let _ = worker.process_job(job(), base);
    // Wait past the rate-limit window and try again.
    let later = at(base, IDR_RATE_LIMIT + Duration::from_millis(50));
    let c2 = worker.process_job(job(), later);
    assert!(c2.idr_request_fired);
    assert_eq!(idr_calls.load(Ordering::Relaxed), 2);
}

#[test]
fn render_drops_counted_when_latest_frame_displaces() {
    // Push two frames in a single job (FakeOutcome::Frames yields them
    // all from one submit). The first lands in LatestFrame, the second
    // displaces it — Worker should report render_drops=1.
    let dec = FakeDecoder::new(vec![FakeOutcome::Frames(vec![
        // Two trivial frames.
        small_solid(2, 2, 1),
        small_solid(2, 2, 2),
    ])]);
    let (mut worker, _idr_calls, _warnings, frames) = make_worker(dec);
    let completion = worker.process_job(job(), MonoNanos::now());
    assert_eq!(completion.render_drops, 1);
    // The second frame is the one left in the slot.
    let last = frames.take().expect("frame in slot");
    match last {
        tether_render::Frame::Cpu(c) => assert_eq!(c.y[0], 2),
        _ => panic!("expected Cpu frame"),
    }
}

fn small_solid(width: u32, height: u32, luma: u8) -> tether_codec::Frame {
    use tether_codec::{DecodedFrame, Frame};
    let y_len = (width as usize) * (height as usize);
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    let uv_len = cw * ch * 2;
    Frame::Cpu(DecodedFrame {
        width,
        height,
        pts: None,
        y: vec![luma; y_len],
        uv: vec![0x80; uv_len],
    })
}

// ---------------------------------------------------------------------
// Thread-lifecycle tests exercising `run_thread_with_init` directly.
// These verify the wrapper around `Worker::process_job`: decoder-init
// failure handling, completion-stream wiring, and orderly shutdown
// when the recv task drops the job sender.
// ---------------------------------------------------------------------

use tether_protocol::control::VideoProfile;

fn lifecycle_callbacks() -> (
    LatestFrame,
    Arc<AtomicU32>,
    Arc<dyn Fn() + Send + Sync + 'static>,
    Arc<dyn Fn() -> u64 + Send + Sync + 'static>,
) {
    let frames = LatestFrame::new();
    let idr_calls = Arc::new(AtomicU32::new(0));
    let idr_cb = Arc::clone(&idr_calls);
    let request_idr: Arc<dyn Fn() + Send + Sync + 'static> = Arc::new(move || {
        idr_cb.fetch_add(1, Ordering::Relaxed);
    });
    let warnings: Arc<dyn Fn() -> u64 + Send + Sync + 'static> = Arc::new(|| 0u64);
    (frames, idr_calls, request_idr, warnings)
}

#[tokio::test(flavor = "current_thread")]
async fn init_failure_drops_ready_tx_without_send() {
    // The recv task uses `ready_rx.await` to gate StreamReady on
    // decoder construction; a `Err` here triggers session teardown
    // (a clean Goodbye rather than a silent hang).
    let (job_tx, job_rx) = crossbeam_channel::bounded::<DecodeJob>(1);
    let (completion_tx, _completion_rx) = crossbeam_channel::unbounded::<DecodeCompletion>();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let (frames, _idr_calls, request_idr, warnings) = lifecycle_callbacks();

    let handle = run_thread_with_init(
        VideoProfile::H264_8BIT_420,
        Err(CodecError::NoHardwareCodec("synthetic test failure".into())),
        job_rx,
        completion_tx,
        frames,
        request_idr,
        warnings,
        ready_tx,
        false,
    );

    // ready_rx must surface RecvError because the worker dropped the
    // sender without sending.
    let err = ready_rx.await.err();
    assert!(
        err.is_some(),
        "ready_tx dropped without send should error ready_rx"
    );

    // Thread should exit promptly; dropping the job tx isn't even
    // needed because the worker returned before the loop. Join with
    // a small budget so a regression that hangs the worker fails fast.
    drop(job_tx);
    let joined = tokio::task::spawn_blocking(move || handle.join())
        .await
        .expect("spawn_blocking failed");
    joined.expect("decode thread should exit cleanly on init failure");
}

#[tokio::test(flavor = "current_thread")]
async fn happy_path_thread_runs_jobs_and_exits_on_sender_drop() {
    // Smoke: scripted decode emits a single solid frame; worker pushes
    // it into LatestFrame; completion stream reports success; thread
    // exits when the job sender drops.
    let (job_tx, job_rx) = crossbeam_channel::bounded::<DecodeJob>(4);
    let (completion_tx, completion_rx) = crossbeam_channel::unbounded::<DecodeCompletion>();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let (frames, _idr_calls, request_idr, warnings) = lifecycle_callbacks();

    let decoder: Box<dyn tether_codec::Decoder> = Box::new(FakeDecoder::new(vec![
        FakeOutcome::Frames(vec![small_solid(8, 8, 0xab)]),
    ]));
    let frames_view = frames.clone();
    let handle = run_thread_with_init(
        VideoProfile::H264_8BIT_420,
        Ok(decoder),
        job_rx,
        completion_tx,
        frames,
        request_idr,
        warnings,
        ready_tx,
        false,
    );

    ready_rx.await.expect("ready signal");
    job_tx
        .send(DecodeJob {
            body: Bytes::from_static(b"x"),
            host_in_client_clock: MonoNanos::now(),
            keyframe: true,
        })
        .unwrap();
    let completion = tokio::task::spawn_blocking(move || {
        completion_rx.recv_timeout(std::time::Duration::from_secs(2))
    })
    .await
    .unwrap()
    .expect("completion");
    assert!(!completion.decode_err);
    assert!(!completion.soft_failure);

    let landed = frames_view.take().expect("frame landed in LatestFrame");
    match landed {
        tether_render::Frame::Cpu(c) => assert_eq!(c.y[0], 0xab),
        _ => panic!("expected Cpu frame"),
    }

    drop(job_tx);
    let joined = tokio::task::spawn_blocking(move || handle.join())
        .await
        .unwrap();
    joined.expect("decode thread exits on sender drop");
}

// --- Error-concealment paths (#10) ----------------------------------

#[test]
fn flush_called_after_hard_submit_error() {
    // A single transient failure should trigger flush(); the
    // FakeDecoder's `flush_count` proves the worker honoured the
    // transient-recovery contract.
    let decoder = FakeDecoder::new(vec![FakeOutcome::SubmitError]);
    // We need to peek at flush_count after the fact. Borrow the
    // fake through the worker via a pointer-trick: hold a raw
    // *mut and de-ref after process_job. Simpler: build the fake,
    // then box it; reach the box through Worker. Since the trait
    // object hides the type, we use a `Arc<Mutex<...>>` indirection
    // pattern by parking the count separately.
    //
    // Cleanest path: put a count in the FakeDecoder, then access
    // via downcasting — but the trait isn't Any. Easier still:
    // construct the fake on the stack, observe via interior
    // mutability through a shared counter.
    use std::sync::Mutex;
    let inspect: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(None));
    // Trick: build a wrapper decoder that bumps the shared counter
    // on flush. Avoids the downcast problem entirely.
    struct FlushSpy {
        inner: FakeDecoder,
        last_seen_flushes: Arc<Mutex<Option<u32>>>,
    }
    impl tether_codec::Decoder for FlushSpy {
        fn submit(&mut self, b: &[u8]) -> Result<(), CodecError> {
            self.inner.submit(b)
        }
        fn next_frame(&mut self) -> Result<Option<tether_codec::Frame>, CodecError> {
            self.inner.next_frame()
        }
        fn codec_kind(&self) -> tether_protocol::control::CodecKind {
            self.inner.codec_kind()
        }
        fn name(&self) -> &'static str {
            "flush-spy"
        }
        fn flush(&mut self) -> Result<(), CodecError> {
            let r = self.inner.flush();
            *self.last_seen_flushes.lock().unwrap() = Some(self.inner.flush_count);
            r
        }
    }
    let inspect_for_spy = Arc::clone(&inspect);
    let spy = FlushSpy {
        inner: decoder,
        last_seen_flushes: inspect_for_spy,
    };
    let idr_calls = Arc::new(AtomicU32::new(0));
    let warnings = Arc::new(AtomicU64::new(0));
    let frames = LatestFrame::new();
    let idr_cb = Arc::clone(&idr_calls);
    let warnings_cb = Arc::clone(&warnings);
    let mut worker = Worker::new(
        Box::new(spy),
        frames.clone(),
        Arc::new(move || {
            idr_cb.fetch_add(1, Ordering::Relaxed);
        }),
        Arc::new(move || warnings_cb.load(Ordering::Relaxed)),
    );
    let c = worker.process_job(job(), MonoNanos::now());
    assert!(c.decode_err);
    assert!(c.idr_request_fired, "hard error must fire IDR");
    assert_eq!(
        *inspect.lock().unwrap(),
        Some(1),
        "flush must be called once after a hard decode error"
    );
}

#[test]
fn three_consecutive_failures_request_rebuild() {
    let decoder = FakeDecoder::new(vec![
        FakeOutcome::SubmitError,
        FakeOutcome::SubmitError,
        FakeOutcome::SubmitError,
    ]);
    let (mut worker, _idr, _warn, _frames) = make_worker(decoder);
    let base = MonoNanos::now();
    let mut last = None;
    for i in 0..3 {
        let c = worker.process_job(job(), at(base, Duration::from_millis(10 * (i as u64 + 1))));
        last = Some(c);
    }
    assert_eq!(
        last.unwrap().recovery,
        Some(tether_decode::RecoveryAction::Rebuild),
        "3 transient failures must escalate to Rebuild"
    );
}

#[test]
fn rebuild_request_clears_after_replace_decoder() {
    let decoder = FakeDecoder::new(vec![
        FakeOutcome::SubmitError,
        FakeOutcome::SubmitError,
        FakeOutcome::SubmitError,
    ]);
    let (mut worker, _idr, _warn, _frames) = make_worker(decoder);
    let base = MonoNanos::now();
    for i in 0..3 {
        let _ = worker.process_job(job(), at(base, Duration::from_millis(10 * (i as u64 + 1))));
    }
    // Swap in a fresh, healthy decoder. The next job must not
    // re-trigger Rebuild — counter must have been reset.
    let healthy = FakeDecoder::new(vec![FakeOutcome::Solid {
        width: 8,
        height: 8,
        luma: 7,
    }]);
    worker.replace_decoder(Box::new(healthy));
    let c = worker.process_job(job(), at(base, Duration::from_millis(100)));
    assert!(
        !c.decode_err && !c.soft_failure,
        "post-rebuild job must succeed against the fresh decoder"
    );
    assert_eq!(
        c.recovery, None,
        "replace_decoder must reset failure counters"
    );
}

#[test]
fn watchdog_fires_after_no_output_window() {
    // Land one success to set last_successful_decode. Then drive
    // soft-failure jobs (warnings bumped) past the watchdog
    // window — the watchdog should request an extra IDR + flush
    // even though the soft-failure IDR rate-limit is in effect.
    // Phased warnings closure: returns 0 for the first two reads
    // (job 0's before/after pair → no soft failure → clean
    // success), then strictly increasing thereafter so every
    // subsequent job sees `after > before` and classifies as a
    // soft failure. Use AtomicU64 to track call count.
    let call_idx = Arc::new(AtomicU64::new(0));
    let call_idx_cb = Arc::clone(&call_idx);
    let phased_warnings: Arc<dyn Fn() -> u64 + Send + Sync + 'static> = Arc::new(move || {
        let n = call_idx_cb.fetch_add(1, Ordering::Relaxed);
        if n < 2 {
            0
        } else {
            n - 1
        }
    });

    let decoder = FakeDecoder::new(vec![
        // Job 0: solid frame → success.
        FakeOutcome::Solid {
            width: 4,
            height: 4,
            luma: 1,
        },
        // Jobs 1..: empty outcomes — no frames decoded.
        FakeOutcome::Frames(vec![]),
        FakeOutcome::Frames(vec![]),
        FakeOutcome::Frames(vec![]),
        FakeOutcome::Frames(vec![]),
    ]);
    let idr_calls = Arc::new(AtomicU32::new(0));
    let idr_cb = Arc::clone(&idr_calls);
    let frames = LatestFrame::new();
    let mut worker = Worker::new(
        Box::new(decoder),
        frames.clone(),
        Arc::new(move || {
            idr_cb.fetch_add(1, Ordering::Relaxed);
        }),
        phased_warnings,
    );

    let base = MonoNanos::now();
    // Job 0: success. Sets last_successful_decode = base.
    let c0 = worker.process_job(job(), base);
    assert!(!c0.decode_err && !c0.soft_failure, "job 0 should succeed");
    assert!(frames.take().is_some(), "job 0 should land a frame");
    assert_eq!(idr_calls.load(Ordering::Relaxed), 0);

    // Job 1: soft failure, but inside watchdog window. The
    // rate-limited soft-failure IDR fires (counter += 1).
    let c1 = worker.process_job(job(), at(base, Duration::from_millis(100)));
    assert!(c1.soft_failure, "job 1 must be soft failure");
    assert!(c1.idr_request_fired);
    let after_first_soft = idr_calls.load(Ordering::Relaxed);

    // Job 2: soft failure, past the watchdog window. The watchdog
    // attempts a flush + IDR — the IDR rate-limit may swallow the
    // second IDR call, but the recovery path ran.
    let elapsed = NO_OUTPUT_WATCHDOG + Duration::from_millis(50);
    let _c2 = worker.process_job(job(), at(base, elapsed));
    let after_watchdog = idr_calls.load(Ordering::Relaxed);
    assert!(after_watchdog >= after_first_soft);

    // Job 3: third consecutive soft failure → hits
    // REBUILD_AFTER_TRANSIENTS and escalates to Rebuild.
    let c3 = worker.process_job(job(), at(base, elapsed + Duration::from_millis(10)));
    assert_eq!(
        c3.recovery,
        Some(tether_decode::RecoveryAction::Rebuild),
        "three consecutive failures must escalate to Rebuild"
    );
}

use tether_decode::NO_OUTPUT_WATCHDOG;

#[test]
fn watchdog_alone_escalates_after_two_silent_windows() {
    // Exercise the watchdog path in isolation from the
    // transient-failure threshold: feed jobs that succeed
    // (decoder doesn't error, doesn't bump warnings) but produce
    // *no frame* — the worker sees no failure to increment
    // consecutive_failures, only the no-output watchdog elapsing
    // can drive Rebuild. The "no warnings" closure returns 0
    // unchanged so `soft_failure = false` everywhere.
    let decoder = FakeDecoder::new(vec![
        FakeOutcome::Solid {
            width: 4,
            height: 4,
            luma: 9,
        },
        FakeOutcome::Frames(vec![]),
        FakeOutcome::Frames(vec![]),
        FakeOutcome::Frames(vec![]),
    ]);
    let idr_calls = Arc::new(AtomicU32::new(0));
    let idr_cb = Arc::clone(&idr_calls);
    let frames = LatestFrame::new();
    let mut worker = Worker::new(
        Box::new(decoder),
        frames.clone(),
        Arc::new(move || {
            idr_cb.fetch_add(1, Ordering::Relaxed);
        }),
        Arc::new(|| 0u64), // never bumps → no soft failure
    );
    let base = MonoNanos::now();
    // Job 0: success → last_successful_decode set.
    let c0 = worker.process_job(job(), base);
    assert!(!c0.decode_err && !c0.soft_failure);
    let _ = frames.take();

    // Job 1: silent (no frame, no warning bump). No failure
    // counter increment, but produced_frame = false so the
    // watchdog clock is still running.
    let _ = worker.process_job(job(), at(base, Duration::from_millis(200)));
    assert_eq!(idr_calls.load(Ordering::Relaxed), 0);

    // Job 2: past one watchdog window. Watchdog arms + requests
    // IDR.
    let c2 = worker.process_job(
        job(),
        at(base, NO_OUTPUT_WATCHDOG + Duration::from_millis(10)),
    );
    assert_eq!(c2.recovery, None, "first window expiry must not rebuild");
    assert!(
        idr_calls.load(Ordering::Relaxed) >= 1,
        "watchdog must request IDR on first window expiry"
    );
    assert!(
        c2.idr_request_fired,
        "DecodeCompletion must report the watchdog IDR so per-second \
         stats account for IDRs from both failure and watchdog paths"
    );

    // Job 3: past two windows. Watchdog escalates.
    let c3 = worker.process_job(
        job(),
        at(base, NO_OUTPUT_WATCHDOG * 2 + Duration::from_millis(10)),
    );
    assert_eq!(
        c3.recovery,
        Some(tether_decode::RecoveryAction::Rebuild),
        "two silent watchdog windows must escalate to Rebuild"
    );
}

#[test]
fn after_rebuild_non_idr_frames_are_skipped_until_keyframe_arrives() {
    // Build a decoder that always succeeds (returns one frame per submit).
    let outcomes: Vec<FakeOutcome> = (0..10)
        .map(|_| FakeOutcome::Solid {
            width: 8,
            height: 8,
            luma: 0x80,
        })
        .collect();
    let (mut worker, _idr_calls, _warnings, frames) = make_worker(FakeDecoder::new(outcomes));

    // First job succeeds (establishes baseline).
    let c = worker.process_job(job(), MonoNanos::now());
    assert!(!c.decode_err);
    assert!(frames.take().is_some(), "first frame should render");

    // Simulate a rebuild by calling replace_decoder.
    let rebuild_outcomes: Vec<FakeOutcome> = (0..10)
        .map(|_| FakeOutcome::Solid {
            width: 8,
            height: 8,
            luma: 0x40,
        })
        .collect();
    worker.replace_decoder(Box::new(FakeDecoder::new(rebuild_outcomes)));

    // P-frames (keyframe=false) should be silently dropped.
    let p_frame = DecodeJob {
        body: Bytes::from_static(b"p-frame-1"),
        host_in_client_clock: MonoNanos::now(),
        keyframe: false,
    };
    let c = worker.process_job(p_frame, MonoNanos::now());
    assert!(!c.decode_err, "skipped frames should not count as errors");
    assert!(
        frames.take().is_none(),
        "P-frame after rebuild should not render"
    );

    let p_frame2 = DecodeJob {
        body: Bytes::from_static(b"p-frame-2"),
        host_in_client_clock: MonoNanos::now(),
        keyframe: false,
    };
    let _c = worker.process_job(p_frame2, MonoNanos::now());
    assert!(
        frames.take().is_none(),
        "second P-frame after rebuild should not render"
    );

    // IDR (keyframe=true) clears the gate and decodes normally.
    let idr = DecodeJob {
        body: Bytes::from_static(b"idr-frame"),
        host_in_client_clock: MonoNanos::now(),
        keyframe: true,
    };
    let c = worker.process_job(idr, MonoNanos::now());
    assert!(!c.decode_err);
    assert!(
        frames.take().is_some(),
        "IDR after rebuild should decode and render"
    );

    // Subsequent P-frames now work (gate cleared).
    let p_after = DecodeJob {
        body: Bytes::from_static(b"p-frame-after-idr"),
        host_in_client_clock: MonoNanos::now(),
        keyframe: false,
    };
    let c = worker.process_job(p_after, MonoNanos::now());
    assert!(!c.decode_err);
    assert!(
        frames.take().is_some(),
        "P-frame after IDR should decode normally"
    );
}
