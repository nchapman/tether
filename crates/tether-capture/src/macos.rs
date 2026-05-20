//! macOS screen capture via ScreenCaptureKit.
//!
//! Calling [`start`] discovers the primary display via `SCShareableContent`,
//! builds an NV12 (`YCbCr_420v`) stream, and spawns a delegate that emits
//! one [`CapturedFrame::Gpu`] per `CMSampleBuffer` carrying a live
//! `IOSurface`. The sample-buffer dispatch queue is owned by SCK; we
//! retain each `CMSampleBuffer` in the per-frame `release_guard` so the
//! underlying IOSurface stays valid until the consumer drops the frame.
//!
//! Permission model: the first `start_capture` call triggers the macOS
//! ScreenRecording TCC prompt. There's no synchronous preflight that
//! returns a usable error before the prompt — by design, Apple wants the
//! prompt to surface from the capture attempt itself. If the user
//! refuses, the framework returns an `SCError` we surface as
//! [`CaptureError::Sck`].
//!
//! Shutdown: dropping the returned receiver causes the next
//! `did_output_sample_buffer` to see [`crossbeam_channel::TrySendError::Disconnected`],
//! which signals the dedicated capture thread to `stop_capture` and drop
//! the stream. The thread itself blocks on a sentinel channel until that
//! signal arrives — keeps the SCStream alive across the dispatch-queue
//! callbacks without leaking it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use screencapturekit::cm::CMSampleBufferExt;
use screencapturekit::cv::CVPixelBuffer;
use screencapturekit::prelude::{
    CMSampleBuffer, PixelFormat, SCContentFilter, SCShareableContent, SCStream,
    SCStreamConfiguration, SCStreamOutputTrait, SCStreamOutputType,
};
use tether_protocol::MonoNanos;

use crate::{
    CaptureError, CapturedFrame, CapturedIOSurface, GpuCapturedFrame, GpuCapturedGuard,
    GpuCapturedSource, Result,
};

/// Channel depth matches the Linux path's `CAPTURE_CHANNEL_DEPTH` — small,
/// because the consumer should always be draining and any back-pressure
/// should drop the oldest pending frame rather than queue.
const CAPTURE_CHANNEL_DEPTH: usize = 2;

/// Default capture rate. The encoder runs at the same fps via the host
/// configuration constant; SCK's frame interval is independent and acts
/// as a producer cap, so they don't need to be the same call site —
/// just the same value.
const CAPTURE_FPS: u32 = 60;

/// Start ScreenCaptureKit on the primary display.
///
/// The returned receiver emits one [`CapturedFrame::Gpu`] per delivered
/// sample. Dropping the receiver shuts the capture thread down on the
/// next sample-buffer callback.
pub async fn start() -> Result<Receiver<CapturedFrame>> {
    let (tx, rx) = bounded::<CapturedFrame>(CAPTURE_CHANNEL_DEPTH);
    let (ready_tx, ready_rx) = bounded::<Result<()>>(1);
    let stop = Arc::new(AtomicBool::new(false));

    let stop_thread = Arc::clone(&stop);
    std::thread::Builder::new()
        .name("tether-capture-sck".into())
        .spawn(move || run_capture_thread(tx, ready_tx, stop_thread))?;

    // Block on the readiness signal from the capture thread without
    // tying up the async runtime.
    let ready = tokio::task::spawn_blocking(move || ready_rx.recv())
        .await
        .map_err(|e| CaptureError::Sck(format!("capture thread join: {e}")))?
        .map_err(|_| CaptureError::Sck("capture thread exited before signaling ready".into()))?;
    ready?;

    // Drop wakes the capture thread. The receiver is what the caller
    // observes for live frames; the `Stop` handle lives inside the
    // thread itself until the channel disconnects.
    let _ = stop;
    Ok(rx)
}

/// Drive the SCStream lifecycle on a dedicated thread.
///
/// The thread:
/// 1. Builds + starts the SCStream.
/// 2. Signals readiness (success or error) to the caller.
/// 3. Parks on the stop signal that the frame handler raises when the
///    consumer drops the receiver.
/// 4. Stops the stream and lets it drop.
fn run_capture_thread(
    tx: Sender<CapturedFrame>,
    ready_tx: Sender<Result<()>>,
    stop: Arc<AtomicBool>,
) {
    let stop_handler = Arc::clone(&stop);
    let stream = match build_and_start_stream(tx, stop_handler) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };
    let _ = ready_tx.send(Ok(()));

    // Park until the frame handler signals shutdown. SCK delivers
    // samples on its own dispatch queue, so this thread has nothing to
    // do besides keep the stream alive.
    while !stop.load(Ordering::Acquire) {
        std::thread::park_timeout(std::time::Duration::from_millis(100));
    }

    if let Err(e) = stream.stop_capture() {
        tracing::warn!(error = %e, "SCStream::stop_capture failed during shutdown");
    }
    drop(stream);
}

fn build_and_start_stream(
    tx: Sender<CapturedFrame>,
    stop: Arc<AtomicBool>,
) -> Result<SCStream> {
    let content = SCShareableContent::get()?;
    let primary = content
        .displays()
        .into_iter()
        .next()
        .ok_or_else(|| CaptureError::Sck("no displays reported by SCShareableContent".into()))?;
    let width = primary.width();
    let height = primary.height();
    tracing::info!(
        display_id = primary.display_id(),
        width,
        height,
        fps = CAPTURE_FPS,
        "capture source: macOS (ScreenCaptureKit, primary display)"
    );

    let filter = SCContentFilter::create()
        .with_display(&primary)
        .with_excluding_windows(&[])
        .build();
    let config = SCStreamConfiguration::new()
        // NV12 video range. Matches what `tether-codec` configures the
        // VideoToolbox encoder to consume zero-copy.
        .with_pixel_format(PixelFormat::YCbCr_420v)
        .with_width(width)
        .with_height(height)
        .with_fps(CAPTURE_FPS)
        // Low queue depth keeps capture-side latency tight; the wire
        // path already absorbs jitter via the fragmenter.
        .with_queue_depth(3)
        .with_shows_cursor(true);

    let mut stream = SCStream::new(&filter, &config);
    let thread_handle = std::thread::current();
    let handler = FrameHandler {
        tx,
        stop,
        wake: thread_handle,
        width: width as u32,
        height: height as u32,
    };
    stream.add_output_handler(handler, SCStreamOutputType::Screen);
    stream.start_capture()?;
    Ok(stream)
}

struct FrameHandler {
    tx: Sender<CapturedFrame>,
    /// Signal that the supervisor thread should stop the stream. Raised
    /// when the consumer drops the receiver.
    stop: Arc<AtomicBool>,
    /// Handle to the supervisor thread, so we can wake it from its
    /// `park_timeout` immediately on disconnect rather than waiting for
    /// the next 100 ms tick.
    wake: std::thread::Thread,
    width: u32,
    height: u32,
}

impl SCStreamOutputTrait for FrameHandler {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        if !matches!(of_type, SCStreamOutputType::Screen) {
            return;
        }
        if self.stop.load(Ordering::Relaxed) {
            return;
        }
        let t_capture_userspace = MonoNanos::now();
        let Some(frame) = build_frame(&sample, self.width, self.height, t_capture_userspace)
        else {
            return;
        };
        match self.tx.try_send(frame) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                // try_send on a bounded crossbeam channel drops the
                // *new* frame on overflow (not the oldest). For an
                // interactive workload the consumer should always be
                // draining, so a full queue means we're already behind;
                // dropping the newest sample is acceptable until we
                // have telemetry that says otherwise.
                tracing::trace!("capture queue full; dropping newest frame");
            }
            Err(TrySendError::Disconnected(_)) => {
                self.stop.store(true, Ordering::Release);
                self.wake.unpark();
            }
        }
    }
}

fn build_frame(
    sample: &CMSampleBuffer,
    width: u32,
    height: u32,
    t_capture_userspace: MonoNanos,
) -> Option<CapturedFrame> {
    let pixel_buffer: CVPixelBuffer = sample.image_buffer()?;
    let iosurface = pixel_buffer.io_surface()?;
    let pixel_format = pixel_buffer.pixel_format();
    let surface_ptr = iosurface.as_ptr();

    // SCK's `CMSampleBuffer::presentation_timestamp` rides a CMClock
    // whose epoch is session-relative (not aligned with
    // `MonoNanos::now()`, which is nanoseconds-since-process-start via
    // `std::time::Instant`). Until we have a calibrated offset, both
    // timestamps share the userspace sample — same conservative choice
    // the Linux backend makes today (see `linux.rs::on_process` —
    // `t_capture_kernel: t` where `t` is `MonoNanos::now()`).
    let t_capture_kernel = t_capture_userspace;

    // Park both retains in the guard: the cloned `CMSampleBuffer`
    // (CFRetain) and the `IOSurface` wrapper we just constructed (also
    // a fresh retain via the apple-cf Swift bridge). The CMSampleBuffer
    // alone would transitively keep the IOSurface alive, but holding
    // the IOSurface explicitly makes the lifetime contract legible
    // without relying on the CV→IOSurface internal retain chain.
    let retained_sample = sample.clone();
    let guard = GpuCapturedGuard::new((retained_sample, iosurface));

    Some(CapturedFrame::Gpu(GpuCapturedFrame {
        width,
        height,
        source: GpuCapturedSource::IOSurface(CapturedIOSurface {
            surface: surface_ptr,
            pixel_format,
            width,
            height,
        }),
        t_capture_kernel,
        t_capture_userspace,
        release_guard: guard,
    }))
}

