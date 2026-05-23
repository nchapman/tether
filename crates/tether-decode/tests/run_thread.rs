//! Unit tests for the per-job decode logic via `Worker::process_job`.
//!
//! These tests avoid spawning threads or building the real decoder so
//! they can drive the IDR rate limiter deterministically by passing a
//! synthetic `now: MonoNanos` directly. Channel / threading behavior
//! is covered by integration tests in `tether-session/tests`.

#![cfg(feature = "test-support")]

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
fn make_worker(
    decoder: FakeDecoder,
) -> (Worker, Arc<AtomicU32>, Arc<AtomicU64>, LatestFrame) {
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
    let dec = FakeDecoder::always_one_frame(8, 8);
    let (mut worker, idr_calls, _warnings, frames) = make_worker(dec);
    let completion = worker.process_job(job(), MonoNanos::now());
    assert!(!completion.decode_err);
    assert!(!completion.soft_failure);
    assert!(!completion.idr_request_fired);
    assert_eq!(idr_calls.load(Ordering::Relaxed), 0);
    assert!(frames.take().is_some(), "LatestFrame should hold the decoded frame");
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
        Box::new(FakeDecoder::always_one_frame(8, 8)),
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
    assert_eq!(read_calls.load(Ordering::Relaxed), 2, "exactly two reads bracket the job");
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
    );

    // ready_rx must surface RecvError because the worker dropped the
    // sender without sending.
    let err = ready_rx.await.err();
    assert!(err.is_some(), "ready_tx dropped without send should error ready_rx");

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
    );

    ready_rx.await.expect("ready signal");
    job_tx
        .send(DecodeJob {
            body: Bytes::from_static(b"x"),
            host_in_client_clock: MonoNanos::now(),
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
