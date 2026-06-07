//! macOS screen capture via ScreenCaptureKit.
//!
//! Calling [`start`] discovers the primary display via `SCShareableContent`,
//! builds a stream for the requested pixel format (BGRA in the live macOS
//! host path), and spawns a delegate that emits one [`CapturedFrame::Gpu`]
//! per `CMSampleBuffer` carrying a live `IOSurface`. The sample-buffer
//! dispatch queue is owned by SCK; we retain each `CMSampleBuffer` in the
//! per-frame `release_guard` so the underlying IOSurface stays valid until
//! the consumer drops the frame.
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

use std::sync::atomic::AtomicU32;

use crossbeam_channel::{bounded, Sender, TrySendError};
use screencapturekit::cm::{CMSampleBufferExt, CMSampleBufferSCExt, SCFrameStatus};
use screencapturekit::cv::CVPixelBuffer;
use screencapturekit::prelude::{
    CMSampleBuffer, PixelFormat, SCContentFilter, SCShareableContent, SCStream,
    SCStreamConfiguration, SCStreamOutputTrait, SCStreamOutputType,
};
use screencapturekit::stream::configuration::SCCaptureResolutionType;
use screencapturekit::FourCharCode;
use tether_protocol::control::{ChromaSubsampling, VideoProfile};
use tether_protocol::control::{DisplayDescriptor, DisplayId, DisplayMode};
use tether_protocol::MonoNanos;

use crate::{
    damage::NativeDamage, scale_to_ratio, CaptureError, CaptureHandle, CapturedFrame,
    CapturedIOSurface, GpuCapturedFrame, GpuCapturedGuard, GpuCapturedSource, Result,
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

const ENV_SCK_CAPTURE_SIZE: &str = "TETHER_SCK_CAPTURE_SIZE";

/// Start ScreenCaptureKit on the primary display with the requested
/// pixel format.
///
/// The returned receiver emits one [`CapturedFrame::Gpu`] per delivered
/// sample. Dropping the receiver shuts the capture thread down on the
/// next sample-buffer callback.
///
/// The caller is responsible for picking a format the host's encoder
/// can ingest — typically via [`sck_pixel_format_for_profile`] after
/// the handshake's chosen profile is known. See
/// [`probe_capture_pixel_formats`] for the per-Mac acceptance matrix.
///
/// Runs [`probe_capture_pixel_formats`] first and logs the result, so
/// every host startup leaves an empirical "what could this Mac do?"
/// alongside "what are we asking it for?" record.
pub async fn start(pixel_format: PixelFormat) -> Result<CaptureHandle> {
    // The capability probe shares the SCK TCC prompt with the real
    // session — first attempt to start a stream triggers ScreenRecording
    // permission once per process. Whether the prompt fires from a
    // probe or from the live stream is the same UX moment; running the
    // probe first means our log records what's possible before we
    // start consuming frames.
    match probe_capture_pixel_formats().await {
        Ok(caps) => {
            tracing::info!(
                bgra = caps.bgra,
                yuv420_video_range = caps.yuv420_video_range,
                yuv420_full_range = caps.yuv420_full_range,
                yuv444_8bit_video_range = caps.yuv444_8bit_video_range,
                yuv444_8bit_full_range = caps.yuv444_8bit_full_range,
                yuv420_10bit_video_range = caps.yuv420_10bit_video_range,
                yuv420_10bit_full_range = caps.yuv420_10bit_full_range,
                yuv444_10bit_full_range = caps.yuv444_10bit_full_range,
                live_stream_format = %pixel_format,
                "SCK pixel-format probe results"
            );
        }
        Err(e) => {
            // Probe failure is not fatal — fall through to the real
            // session attempt, which will surface the same underlying
            // SCK error with a clearer call site if the issue is
            // structural (missing permission, no displays, etc.).
            tracing::warn!(error = %e, "SCK pixel-format probe failed");
        }
    }

    let (tx, rx) = bounded::<CapturedFrame>(CAPTURE_CHANNEL_DEPTH);
    let (ready_tx, ready_rx) = bounded::<Result<CaptureReady>>(1);
    let stop = Arc::new(AtomicBool::new(false));

    let stop_thread = Arc::clone(&stop);
    std::thread::Builder::new()
        .name("tether-capture-sck".into())
        .spawn(move || {
            run_capture_thread(tx, ready_tx, stop_thread, pixel_format);
        })?;

    // Block on the readiness signal from the capture thread without
    // tying up the async runtime.
    let ready = tokio::task::spawn_blocking(move || ready_rx.recv())
        .await
        .map_err(|e| CaptureError::Sck(format!("capture thread join: {e}")))?
        .map_err(|_| CaptureError::Sck("capture thread exited before signaling ready".into()))??;

    // Drop wakes the capture thread. The receiver is what the caller
    // observes for live frames; the `Stop` handle lives inside the
    // thread itself until the channel disconnects.
    let _ = stop;
    // FPS atomic surfaced for the ABR controller. SCK exposes
    // `SCStreamConfiguration.minimumFrameInterval` as a runtime
    // setter, but plumbing the live update from this atomic to the
    // running SCStream lands per-backend; until then writes are
    // observable via `CaptureHandle::target_fps` but don't change
    // the actual SCK cadence — see `CaptureHandle` docs.
    let target_fps = Arc::new(AtomicU32::new(CAPTURE_FPS));
    let cursor_source = crate::cursor_macos::start(ready.cursor_geom);
    Ok(CaptureHandle::from_parts(rx, target_fps).with_cursor_source(cursor_source))
}

pub fn display_list() -> Result<Vec<DisplayDescriptor>> {
    let content = SCShareableContent::get()?;
    let displays = content.displays();
    if displays.is_empty() {
        return Err(CaptureError::Sck(
            "no displays reported by SCShareableContent".into(),
        ));
    }

    Ok(displays
        .into_iter()
        .enumerate()
        .map(|(idx, display)| {
            let display_id = display.display_id();
            let frame = display.frame();
            let logical_w = display.width().max(1);
            let logical_h = display.height().max(1);
            let (pixel_w, pixel_h, refresh_millihz) =
                active_display_mode(display_id).unwrap_or((logical_w, logical_h, 60_000));
            let mode = DisplayMode::new(pixel_w, pixel_h, refresh_millihz);
            let scale = f64::from(pixel_w) / f64::from(logical_w);
            let (scale_num, scale_den) = scale_to_ratio(scale);
            DisplayDescriptor {
                id: DisplayId(display_id),
                name: format!("display-{display_id}"),
                scale_num,
                scale_den,
                primary: idx == 0,
                position: (
                    round_f64_to_i32(frame.origin.x),
                    round_f64_to_i32(frame.origin.y),
                ),
                current_mode: mode,
                available_modes: vec![mode],
                can_set_mode: false,
            }
        })
        .collect())
}

/// Geometry of the live capture stream, plumbed from the capture
/// thread back to `start` so it can hand the cursor source the
/// point-to-pixel scale.
struct CaptureReady {
    cursor_geom: crate::cursor_macos::CaptureGeometry,
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
    ready_tx: Sender<Result<CaptureReady>>,
    stop: Arc<AtomicBool>,
    pixel_format: PixelFormat,
) {
    let stop_handler = Arc::clone(&stop);
    let (stream, cursor_geom) = match build_and_start_stream(tx, stop_handler, pixel_format) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };
    let _ = ready_tx.send(Ok(CaptureReady { cursor_geom }));

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
    pixel_format: PixelFormat,
) -> Result<(SCStream, crate::cursor_macos::CaptureGeometry)> {
    let content = SCShareableContent::get()?;
    let primary =
        content.displays().into_iter().next().ok_or_else(|| {
            CaptureError::Sck("no displays reported by SCShareableContent".into())
        })?;
    // SCDisplay::width/height report logical *point* dimensions, not
    // physical pixels. On a Retina 2× display that's half the actual
    // resolution, and configuring SCStreamConfiguration with the
    // point dims makes SCK deliver a downsampled stream — the client
    // then upscales it back to viewport dims, producing the blurry
    // output a HiDPI session would otherwise be missing entirely.
    //
    // `CGDisplayPixelsWide`/`High` ALSO return logical-point dims on
    // a Retina display configured with HiDPI scaling enabled (the
    // "active display mode" is the scaled mode, and that mode's
    // `width`/`height` are points). The right API is
    // `CGDisplayCopyDisplayMode` → `CGDisplayModeGetPixelWidth` /
    // `GetPixelHeight`, which return the *backing pixel* count
    // independent of the active scaling mode.
    let display_id = primary.display_id();
    let logical_w = primary.width();
    let logical_h = primary.height();
    // SAFETY: `CGDisplayCopyDisplayMode` returns a +1 retained
    // `CGDisplayModeRef` (CF object) for an attached display, or
    // null if the display dropped off. The display_id was just
    // obtained from SCShareableContent for an attached display.
    let (pixel_w, pixel_h) = unsafe {
        let mode = CGDisplayCopyDisplayMode(display_id);
        if mode.is_null() {
            (0u32, 0u32)
        } else {
            // Pixel dims always fit u32; the 0 fallback rejoins the
            // null-mode path below (→ logical dims) if a query ever overflows.
            let w = u32::try_from(CGDisplayModeGetPixelWidth(mode)).unwrap_or(0);
            let h = u32::try_from(CGDisplayModeGetPixelHeight(mode)).unwrap_or(0);
            // CGDisplayModeRef is a CFType; balance the +1 retain
            // from Copy.
            cf_release(mode);
            (w, h)
        }
    };
    // Fall back to the logical dims if the CG query failed (would
    // happen if the display dropped off between SCShareableContent
    // and the CG query). 0×0 would crash SCStreamConfiguration.
    let (raw_w, raw_h) = if pixel_w > 0 && pixel_h > 0 {
        (pixel_w, pixel_h)
    } else {
        tracing::warn!(
            display_id,
            logical_w,
            logical_h,
            "CGDisplayCopyDisplayMode returned null; falling back to logical dims"
        );
        (logical_w, logical_h)
    };
    let dims = sck_configured_capture_dims(raw_w, raw_h);
    let (width, height) = (dims.width, dims.height);
    tracing::info!(
        display_id,
        logical_w,
        logical_h,
        raw_pixel_w = raw_w,
        raw_pixel_h = raw_h,
        capture_w = width,
        capture_h = height,
        fps = CAPTURE_FPS,
        pixel_format = %pixel_format,
        capture_resolution = %SCCaptureResolutionType::Best,
        capture_size_override = dims.override_label.as_deref(),
        "capture source: macOS (ScreenCaptureKit, primary display)"
    );

    let filter = SCContentFilter::create()
        .with_display(&primary)
        .with_excluding_windows(&[])
        .build();
    // Configure SCK at the display's backing-pixel grid, aligned to the
    // encoder block size. Viewport sizing happens downstream in the host
    // encode path; `TETHER_SCK_CAPTURE_SIZE` remains as an explicit
    // diagnostic override for comparing Apple's native downscale.
    let config = stream_config(pixel_format, width, height);

    let mut stream = SCStream::new(&filter, &config);
    let thread_handle = std::thread::current();
    let cursor_geom = crate::cursor_macos::CaptureGeometry {
        logical_point_w: f64::from(logical_w),
        logical_point_h: f64::from(logical_h),
        capture_pixel_w: width,
        capture_pixel_h: height,
    };
    let handler = FrameHandler {
        tx,
        stop,
        wake: thread_handle,
    };
    stream.add_output_handler(handler, SCStreamOutputType::Screen);
    stream.start_capture()?;
    Ok((stream, cursor_geom))
}

fn stream_config(pixel_format: PixelFormat, width: u32, height: u32) -> SCStreamConfiguration {
    SCStreamConfiguration::new()
        .with_pixel_format(pixel_format)
        .with_width(width)
        .with_height(height)
        // Default `Automatic` may trade source fidelity for bandwidth.
        // We do our own explicit viewport sizing and bitrate control, so
        // ask SCK to sample the source at the best available resolution
        // before delivering our configured output dimensions.
        .with_capture_resolution_type(SCCaptureResolutionType::Best)
        .with_fps(CAPTURE_FPS)
        // Low queue depth keeps capture-side latency tight; the wire
        // path already absorbs jitter via the fragmenter. Previously
        // raised to 5 (commit 8f252b6) to absorb VT-session / cursor-
        // poller stalls, but the deeper queue aged frames by up to
        // ~33 ms whenever the encoder paused, which dominated user-
        // perceived lag. Reverted to 3 pending stage-by-stage latency
        // instrumentation — re-raise only with measured evidence that
        // SCK is the producer-side dropper, not a downstream stall.
        .with_queue_depth(3)
        // Cursor is extracted out-of-band by `cursor_macos` and
        // overlay-rendered on the client at sub-frame latency. Leaving
        // SCK's in-frame cursor on would double-draw it.
        .with_shows_cursor(false)
}

#[derive(Debug)]
struct SckCaptureDims {
    width: u32,
    height: u32,
    override_label: Option<String>,
}

fn sck_configured_capture_dims(raw_w: u32, raw_h: u32) -> SckCaptureDims {
    let (native_w, native_h) = sck_native_capture_dims(raw_w, raw_h);
    let Some(requested) = std::env::var(ENV_SCK_CAPTURE_SIZE)
        .ok()
        .and_then(|v| parse_capture_size(&v))
    else {
        return SckCaptureDims {
            width: native_w,
            height: native_h,
            override_label: None,
        };
    };

    let (width, height) = align_capture_dims(requested.0, requested.1);
    if (width, height) != requested {
        tracing::warn!(
            requested_w = requested.0,
            requested_h = requested.1,
            capture_w = width,
            capture_h = height,
            "TETHER_SCK_CAPTURE_SIZE floored to encoder alignment"
        );
    }
    SckCaptureDims {
        width,
        height,
        override_label: Some(format!("{}x{}", width, height)),
    }
}

fn parse_capture_size(value: &str) -> Option<(u32, u32)> {
    let (w, h) = value
        .trim()
        .split_once('x')
        .or_else(|| value.trim().split_once('X'))?;
    let w = w.trim().parse::<u32>().ok()?;
    let h = h.trim().parse::<u32>().ok()?;
    (w > 0 && h > 0).then_some((w, h))
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
        let Some(frame) = build_frame(&sample, t_capture_userspace) else {
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

fn build_frame(sample: &CMSampleBuffer, t_capture_userspace: MonoNanos) -> Option<CapturedFrame> {
    let pixel_buffer: CVPixelBuffer = sample.image_buffer()?;
    let iosurface = pixel_buffer.io_surface()?;
    let pixel_format = pixel_buffer.pixel_format();
    let width = u32::try_from(pixel_buffer.width()).ok()?;
    let height = u32::try_from(pixel_buffer.height()).ok()?;
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
        native_damage: read_sck_damage(sample),
    }))
}

/// Read `SCStreamFrameInfo.status` from the sample buffer. Split out
/// from the policy fn ([`native_damage_for_frame_status`]) so the
/// policy is unit-testable on the pure `SCFrameStatus` enum without
/// fabricating a `CMSampleBuffer`.
fn read_sck_damage(sample: &CMSampleBuffer) -> Option<NativeDamage> {
    let status = sample.frame_status()?;
    Some(native_damage_for_frame_status(status))
}

/// Map `SCStreamFrameInfo.status` to our backend-agnostic
/// [`NativeDamage`]. SCK delivers periodic re-emits with
/// `Idle`/`Blank`/`Suspended` for liveness; `Complete` and `Started`
/// are real frames. `Stopped` is a terminal signal — we treat it as
/// idle (stream is ending; host shuts down shortly) and log a warn at
/// the call site so a mid-session `Stopped` stays visible.
fn native_damage_for_frame_status(status: SCFrameStatus) -> NativeDamage {
    if matches!(status, SCFrameStatus::Stopped) {
        tracing::warn!("SCStreamFrameInfo.Stopped delivered; stream is ending");
    }
    NativeDamage {
        idle: matches!(
            status,
            SCFrameStatus::Idle
                | SCFrameStatus::Blank
                | SCFrameStatus::Suspended
                | SCFrameStatus::Stopped
        ),
    }
}

/// Diagnostic capture-size override alignment in pixels. Native SCK
/// capture preserves the display's backing-pixel dimensions so capture
/// aspect ratio and cursor point→pixel mapping stay exact; the host
/// encode path handles viewport sizing downstream. The override is an
/// explicit test knob for asking SCK to resize, so we floor only that
/// requested size to the encoder-friendly grid.
const SCK_CAPTURE_ALIGN: u32 = 16;

fn sck_native_capture_dims(raw_w: u32, raw_h: u32) -> (u32, u32) {
    (raw_w.max(1), raw_h.max(1))
}

fn align_capture_dims(width: u32, height: u32) -> (u32, u32) {
    let aligned_w = (width / SCK_CAPTURE_ALIGN) * SCK_CAPTURE_ALIGN;
    let aligned_h = (height / SCK_CAPTURE_ALIGN) * SCK_CAPTURE_ALIGN;
    (
        aligned_w.max(SCK_CAPTURE_ALIGN),
        aligned_h.max(SCK_CAPTURE_ALIGN),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sck_status_complete_and_started_are_changed() {
        assert!(!native_damage_for_frame_status(SCFrameStatus::Complete).idle);
        assert!(!native_damage_for_frame_status(SCFrameStatus::Started).idle);
    }

    #[test]
    fn sck_status_idle_blank_suspended_stopped_are_idle() {
        for s in [
            SCFrameStatus::Idle,
            SCFrameStatus::Blank,
            SCFrameStatus::Suspended,
            SCFrameStatus::Stopped,
        ] {
            assert!(
                native_damage_for_frame_status(s).idle,
                "{s:?} should map to idle"
            );
        }
    }

    #[test]
    fn sck_native_dims_preserve_backing_pixels() {
        assert_eq!(sck_native_capture_dims(3024, 1964), (3024, 1964));
    }

    #[test]
    fn sck_native_dims_never_use_viewport_size() {
        assert_eq!(sck_native_capture_dims(3840, 2160), (3840, 2160));
        assert_eq!(sck_native_capture_dims(1280, 720), (1280, 720));
    }

    #[test]
    fn diagnostic_override_floors_to_alignment() {
        let (w, h) = sck_native_capture_dims(1920, 1080);
        assert_eq!((w, h), (1920, 1080));
        assert_eq!(align_capture_dims(w, h), (1920, 1072));
    }

    #[test]
    fn parses_diagnostic_capture_size_override() {
        assert_eq!(parse_capture_size("1712x1104"), Some((1712, 1104)));
        assert_eq!(parse_capture_size(" 1600 X 900 "), Some((1600, 900)));
        assert_eq!(parse_capture_size("0x900"), None);
        assert_eq!(parse_capture_size("nope"), None);
    }

    #[test]
    fn diagnostic_capture_size_override_aligns_like_native_dims() {
        assert_eq!(align_capture_dims(1600, 900), (1600, 896));
        assert_eq!(align_capture_dims(1712, 1104), (1712, 1104));
    }
}

/// The ScreenCaptureKit pixel format the live stream should use to
/// feed the given encoder [`VideoProfile`]. Mapping mirrors
/// `vt_sw_format` on the encoder side — the encoder's configured
/// `sw_format` and the IOSurface fourcc the capture delivers must
/// agree, or `submit_iosurface` rejects the frame at the
/// `iosurface_fourcc_matches` gate. The 4:2:0 paths pick *video range*
/// (`420v` for 8-bit, `x420` for 10-bit) because VT's HEVC encoder
/// expects limited-range input by default; mixing full-range capture
/// against the encoder's video-range matrix would shift levels at the
/// decoder.
///
/// Returns [`SckCapabilityCheck::Unsupported`] for combos this layer
/// doesn't model — caller must reject or fall through.
pub fn sck_pixel_format_for_profile(profile: VideoProfile) -> SckCapabilityCheck {
    use SckCapabilityCheck::{Supported, Unsupported};
    match (profile.chroma, profile.bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => Supported(PixelFormat::YCbCr_420v),
        (ChromaSubsampling::Yuv420, 10) => {
            Supported(PixelFormat::Unknown(FourCharCode::from_bytes(*b"x420")))
        }
        (ChromaSubsampling::Yuv444, 8) => {
            Supported(PixelFormat::Unknown(FourCharCode::from_bytes(*b"444v")))
        }
        (ChromaSubsampling::Yuv444, 10) => Supported(PixelFormat::xf44),
        _ => Unsupported,
    }
}

/// SCK's packed BGRA capture format. Exposed so the host can request
/// BGRA without depending directly on the ScreenCaptureKit crate.
#[must_use]
pub fn sck_bgra_pixel_format() -> PixelFormat {
    PixelFormat::BGRA
}

/// Outcome of mapping a `VideoProfile` to an SCK `PixelFormat`. Two
/// variants so a caller can match exhaustively without a stringly
/// "unknown" sentinel.
#[derive(Debug, Clone, Copy)]
pub enum SckCapabilityCheck {
    Supported(PixelFormat),
    Unsupported,
}

impl SckCapabilityCheck {
    /// The raw `kCVPixelFormatType_*` fourcc this profile maps to, as
    /// a big-endian `u32`. Exposed so cross-crate consistency tests
    /// can compare against the VT encoder / probe / renderer accept
    /// tables without depending on the `screencapturekit` crate
    /// directly.
    #[must_use]
    pub fn fourcc(self) -> Option<u32> {
        match self {
            Self::Supported(pf) => Some(FourCharCode::from(pf).as_u32()),
            Self::Unsupported => None,
        }
    }
}

impl SckCapabilityCheck {
    /// Check the mapped SCK pixel format against a probed
    /// [`SckCaptureCapability`]. Returns `true` iff:
    ///   1. the profile maps to an SCK format ([`Supported`]), AND
    ///   2. that format was accepted by the SCK probe on this Mac.
    ///
    /// `Unknown(code)` dispatch goes through `u32` fourcc comparison
    /// rather than `code.display()` string match — matches the pattern
    /// in `tether-codec::videotoolbox::probe::expected_iosurface_fourccs`
    /// and avoids a per-call `String` allocation on the
    /// `capture_filtered_encode_profiles` path.
    ///
    /// [`Supported`]: SckCapabilityCheck::Supported
    pub fn is_deliverable(self, caps: &SckCaptureCapability) -> bool {
        const V444V: u32 = u32::from_be_bytes(*b"444v");
        const V444F: u32 = u32::from_be_bytes(*b"444f");
        const VX420: u32 = u32::from_be_bytes(*b"x420");
        const VXF20: u32 = u32::from_be_bytes(*b"xf20");
        match self {
            Self::Supported(PixelFormat::YCbCr_420v) => caps.yuv420_video_range,
            Self::Supported(PixelFormat::YCbCr_420f) => caps.yuv420_full_range,
            Self::Supported(PixelFormat::xf44) => caps.yuv444_10bit_full_range,
            Self::Supported(PixelFormat::Unknown(code)) => match code.as_u32() {
                V444V => caps.yuv444_8bit_video_range,
                V444F => caps.yuv444_8bit_full_range,
                VX420 => caps.yuv420_10bit_video_range,
                VXF20 => caps.yuv420_10bit_full_range,
                _ => false,
            },
            Self::Supported(_) => false,
            Self::Unsupported => false,
        }
    }
}

/// Per-format ScreenCaptureKit capture acceptance, derived from a real
/// `SCStream::start_capture()` attempt at probe-time. Independent bits
/// per format: a Mac may accept 4:2:0 video range but not 4:2:0 full
/// range, or accept `xf44` (10-bit) without accepting `'444v'` (8-bit),
/// etc. The `Unknown(FourCharCode)` SCK formats (`'444v'`/`'444f'`) are
/// in CoreVideo but not in SCK's documented `pixelFormat` enum; whether
/// SCK accepts them via the `Unknown` escape hatch is what the probe
/// answers empirically.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SckCaptureCapability {
    /// `BGRA` — packed 32-bit, SCK's documented default before macOS 26.
    pub bgra: bool,
    /// `420v` — biplanar NV12 video range. The format the live stream
    /// uses today and what `tether-codec`'s VideoToolbox NV12 fast path
    /// is wired to consume.
    pub yuv420_video_range: bool,
    /// `420f` — biplanar NV12 full range.
    pub yuv420_full_range: bool,
    /// `'444v'` (via `PixelFormat::Unknown`) — biplanar NV24 video
    /// range. Not in SCK's documented enum; if accepted, this is the
    /// path that unlocks macOS-host HEVC Main 4:4:4 8-bit encode.
    pub yuv444_8bit_video_range: bool,
    /// `'444f'` (via `PixelFormat::Unknown`) — biplanar NV24 full
    /// range. Same status as `'444v'`.
    pub yuv444_8bit_full_range: bool,
    /// `'x420'` (via `PixelFormat::Unknown`) — biplanar 10-bit 4:2:0
    /// video range (CoreVideo `kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange`).
    /// Not in SCK's documented enum; the path that unlocks macOS-host
    /// HEVC Main 4:2:0 10-bit (Main10) encode in *video range* — the
    /// VT encoder's expected limited-range input format.
    pub yuv420_10bit_video_range: bool,
    /// `'xf20'` (via `PixelFormat::Unknown`) — biplanar 10-bit 4:2:0
    /// full range (CoreVideo `kCVPixelFormatType_420YpCbCr10BiPlanarFullRange`).
    /// Companion of `x420` for full-range pipelines.
    pub yuv420_10bit_full_range: bool,
    /// `xf44` — biplanar 10-bit 4:4:4 full range, the only 10-bit
    /// 4:4:4 SCK format Apple documents. If accepted, this is the
    /// path that unlocks macOS-host HEVC Main 4:4:4 10-bit encode
    /// (which Commit 1's probe confirmed VT can accept).
    pub yuv444_10bit_full_range: bool,
}

/// The set of pixel formats we probe. Order is arbitrary — every probe
/// is independent. Adding a variant here forces every match arm below
/// to be updated (no string-keyed dispatch), so a new format can't get
/// half-plumbed silently.
#[derive(Debug, Clone, Copy)]
enum ProbeFormat {
    Bgra,
    Yuv420v,
    Yuv420f,
    /// Biplanar NV24 video range. Not in the SCK crate's named enum;
    /// passed through the `PixelFormat::Unknown(FourCharCode)` escape
    /// hatch. Whether SCK actually delivers frames in this format vs.
    /// silently downgrading is the question the frame-arrival check
    /// answers — `start_capture` Ok alone is a false-positive risk for
    /// the `Unknown` variants (the catch-all bypasses any named-format
    /// validation SCK does internally).
    Yuv444v,
    /// Biplanar NV24 full range. Same status as `Yuv444v`.
    Yuv444f,
    /// Biplanar 10-bit 4:2:0 video range (`'x420'`). Not in SCK's
    /// named enum; passed via `Unknown`. The HEVC Main10 input format
    /// VT's encoder consumes by default. Same false-positive concern
    /// as the other `Unknown` variants — frame-arrival check required.
    X420,
    /// Biplanar 10-bit 4:2:0 full range (`'xf20'`). Companion of
    /// `X420` for full-range pipelines.
    Xf20,
    /// Apple's documented 10-bit 4:4:4 biplanar full range.
    Xf44,
}

impl ProbeFormat {
    const ALL: &'static [Self] = &[
        Self::Bgra,
        Self::Yuv420v,
        Self::Yuv420f,
        Self::Yuv444v,
        Self::Yuv444f,
        Self::X420,
        Self::Xf20,
        Self::Xf44,
    ];

    fn pixel_format(self) -> PixelFormat {
        match self {
            Self::Bgra => PixelFormat::BGRA,
            Self::Yuv420v => PixelFormat::YCbCr_420v,
            Self::Yuv420f => PixelFormat::YCbCr_420f,
            Self::Yuv444v => PixelFormat::Unknown(FourCharCode::from_bytes(*b"444v")),
            Self::Yuv444f => PixelFormat::Unknown(FourCharCode::from_bytes(*b"444f")),
            Self::X420 => PixelFormat::Unknown(FourCharCode::from_bytes(*b"x420")),
            Self::Xf20 => PixelFormat::Unknown(FourCharCode::from_bytes(*b"xf20")),
            Self::Xf44 => PixelFormat::xf44,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Bgra => "BGRA",
            Self::Yuv420v => "420v",
            Self::Yuv420f => "420f",
            Self::Yuv444v => "444v",
            Self::Yuv444f => "444f",
            Self::X420 => "x420",
            Self::Xf20 => "xf20",
            Self::Xf44 => "xf44",
        }
    }

    /// SCK's named enum variants are Apple-documented as accepted by
    /// `pixelFormat` — `start_capture` Ok is a reliable signal for
    /// those (BGRA, 420v, 420f). The `Unknown(FourCharCode)` escape
    /// hatch bypasses that validation, so `start_capture` may return
    /// Ok for an undocumented code without SCK actually delivering
    /// frames in it; probes for those must verify a delivered
    /// sample's fourcc matches the request. `xf44` is the documented
    /// 10-bit 4:4:4 capture format but support is silicon-generation
    /// dependent — some configurations (pre-M3 with stock macOS)
    /// silently accept the config and deliver zero frames — so it
    /// also gets the frame-arrival check despite being a named
    /// enum variant.
    fn needs_frame_arrival_check(self) -> bool {
        matches!(
            self,
            Self::Yuv444v | Self::Yuv444f | Self::X420 | Self::Xf20 | Self::Xf44
        )
    }

    fn apply(self, caps: &mut SckCaptureCapability, accepted: bool) {
        match self {
            Self::Bgra => caps.bgra = accepted,
            Self::Yuv420v => caps.yuv420_video_range = accepted,
            Self::Yuv420f => caps.yuv420_full_range = accepted,
            Self::Yuv444v => caps.yuv444_8bit_video_range = accepted,
            Self::Yuv444f => caps.yuv444_8bit_full_range = accepted,
            Self::X420 => caps.yuv420_10bit_video_range = accepted,
            Self::Xf20 => caps.yuv420_10bit_full_range = accepted,
            Self::Xf44 => caps.yuv444_10bit_full_range = accepted,
        }
    }
}

/// Per-format outcome of the probe attempt. The outer `Result` covers
/// fatal SCK errors that should abort the probe loop (cleanup failure
/// would leave zombie streams running alongside subsequent probes and
/// the live session); `Ok(false)` is the regular "not supported" answer
/// and continues the loop.
enum ProbeOutcome {
    /// `start_capture` returned Ok and (for `Unknown` formats) a
    /// delivered sample buffer carried the fourcc we asked for.
    Accepted,
    /// `start_capture` returned an error, or `start_capture` returned
    /// Ok but the format-arrival check timed out / saw a wrong fourcc.
    Rejected,
}

/// Try each candidate SCK pixel format against a real, briefly-started
/// stream on the primary display. Two-tier signal:
///
/// - For Apple-documented `PixelFormat` variants (BGRA, 420v, 420f,
///   xf44) we treat `SCStream::start_capture` returning Ok as a
///   sufficient acceptance signal — SCK validates these against its
///   internal table at configuration time, so the rejection path is
///   reliable.
/// - For `PixelFormat::Unknown(FourCharCode)` variants (`'444v'`,
///   `'444f'`) we additionally wait briefly for the first delivered
///   sample buffer and assert `CVPixelBuffer::pixel_format()` matches
///   the requested fourcc. SCK accepts arbitrary FourCharCodes through
///   the escape hatch but may silently downgrade — without the
///   frame-arrival check we'd record false positives that would later
///   manifest as wrong-format IOSurfaces in a negotiated session.
///
/// Probe + live-session UX: SCK's ScreenRecording TCC prompt fires on
/// the first stream attempt per process. Running this probe before
/// the live session means the first prompt the user sees is ours, in
/// the same callsite as the rest of our startup logs — better than
/// the alternative where the probe is the second attempt and the user
/// sees a prompt for "tether host" twice.
pub async fn probe_capture_pixel_formats() -> Result<SckCaptureCapability> {
    // Probes are blocking SCK calls; run them on a blocking thread so an
    // async caller's runtime isn't tied up waiting on SCK's dispatch queue.
    tokio::task::spawn_blocking(probe_capture_pixel_formats_blocking)
        .await
        .map_err(|e| CaptureError::Sck(format!("probe task join: {e}")))?
}

/// Synchronous core of [`probe_capture_pixel_formats`]: query the shareable
/// display, build a filter, and run each candidate format through
/// `probe_one_format`. Blocks the calling thread on SCK's dispatch queue.
///
/// Unlike the async wrapper this needs **no** Tokio runtime, so sync callers
/// (the host capability probe in `tether-probe`, which drives this from inside
/// a sync `ProfileProbe` impl) can call it directly. Bridging the async fn with
/// a non-Tokio executor like `pollster` panicked here — the wrapper's
/// `spawn_blocking` needs an ambient reactor the probe didn't provide.
pub fn probe_capture_pixel_formats_blocking() -> Result<SckCaptureCapability> {
    let content = SCShareableContent::get()?;
    let display =
        content.displays().into_iter().next().ok_or_else(|| {
            CaptureError::Sck("no displays reported by SCShareableContent".into())
        })?;
    let display_width = display.width();
    let display_height = display.height();

    // Build one filter and reuse across probes — it's just a reference
    // to the display, no per-format coupling.
    let filter = SCContentFilter::create()
        .with_display(&display)
        .with_excluding_windows(&[])
        .build();

    let mut caps = SckCaptureCapability::default();
    for &probe in ProbeFormat::ALL {
        match probe_one_format(&filter, probe, display_width, display_height) {
            Ok(outcome) => {
                let accepted = matches!(outcome, ProbeOutcome::Accepted);
                tracing::debug!(format = probe.label(), accepted, "SCK probe result");
                probe.apply(&mut caps, accepted);
            }
            Err(e) => {
                // Fatal probe error (cleanup failure most likely).
                // Abort the loop so we don't leave zombie probe
                // streams running alongside subsequent probes and
                // the live session — SCK allows concurrent streams,
                // but a partial cleanup compounds across iterations.
                tracing::warn!(
                    format = probe.label(),
                    error = %e,
                    "SCK probe aborted; remaining formats reported as not-supported"
                );
                return Ok(caps);
            }
        }
    }
    Ok(caps)
}

/// Configure + start + (briefly verify) + stop an SCStream for one
/// candidate format. Returns the per-format outcome on success, or
/// `Err(CaptureError)` if cleanup failed in a way that would leak a
/// running stream into subsequent probes.
fn probe_one_format(
    filter: &SCContentFilter,
    probe: ProbeFormat,
    width: u32,
    height: u32,
) -> Result<ProbeOutcome> {
    let format = probe.pixel_format();
    let config = SCStreamConfiguration::new()
        .with_pixel_format(format)
        .with_width(width)
        .with_height(height)
        .with_fps(CAPTURE_FPS)
        .with_queue_depth(2)
        .with_shows_cursor(false);
    let mut stream = SCStream::new(filter, &config);

    let (frame_tx, frame_rx) = bounded::<u32>(1);
    let handler = ProbeFrameSink { frame_tx };
    stream.add_output_handler(handler, SCStreamOutputType::Screen);

    if stream.start_capture().is_err() {
        return Ok(ProbeOutcome::Rejected);
    }

    let outcome = if probe.needs_frame_arrival_check() {
        // Wait for one frame and check its delivered fourcc against
        // what we asked for. 250ms is plenty at 60fps (≈4 frame
        // intervals) and bounds the worst-case probe time at 1.5s
        // total if all six Unknown probes hit the timeout.
        let expected: u32 = FourCharCode::from(format).as_u32();
        match frame_rx.recv_timeout(std::time::Duration::from_millis(250)) {
            Ok(actual) if actual == expected => ProbeOutcome::Accepted,
            Ok(actual) => {
                tracing::debug!(
                    format = probe.label(),
                    expected = format_args!("0x{:08x}", expected),
                    actual = format_args!("0x{:08x}", actual),
                    "SCK accepted configuration but delivered a different fourcc"
                );
                ProbeOutcome::Rejected
            }
            Err(_) => ProbeOutcome::Rejected,
        }
    } else {
        ProbeOutcome::Accepted
    };

    if let Err(e) = stream.stop_capture() {
        // Cleanup failed — SCStream may still be running. Returning
        // Err here aborts the probe loop so we don't accumulate
        // running streams across subsequent iterations.
        return Err(CaptureError::Sck(format!(
            "SCStream::stop_capture failed during probe for {label}: {e}",
            label = probe.label()
        )));
    }
    Ok(outcome)
}

/// Probe-only output handler. Reports the first delivered frame's
/// pixel format on a one-shot channel so the probe can verify the
/// delivered fourcc matches the configured one (the
/// `PixelFormat::Unknown` escape hatch makes `start_capture` Ok an
/// unreliable signal on its own).
struct ProbeFrameSink {
    frame_tx: Sender<u32>,
}

impl SCStreamOutputTrait for ProbeFrameSink {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        if !matches!(of_type, SCStreamOutputType::Screen) {
            return;
        }
        // Only the first frame matters — `try_send` against a depth-1
        // bounded channel drops every later sample silently, which is
        // the desired behaviour (we're sampling the format, not the
        // stream).
        if let Some(pixel_buffer) = sample.image_buffer() {
            let fourcc = pixel_buffer.pixel_format();
            let _ = self.frame_tx.try_send(fourcc);
        }
    }
}

#[cfg(test)]
mod sck_tests {
    use super::*;
    use std::time::Duration;

    /// Hardware probe — runs SCK and records what this Mac accepts.
    /// 4:2:0 video range should always come back true; the higher
    /// chroma / 10-bit results are the interesting ones to log.
    /// Triggers the ScreenRecording TCC prompt on first run.
    #[tokio::test]
    #[ignore = "requires macOS + ScreenRecording permission; run with: cargo test -p tether-capture -- --ignored"]
    async fn probe_capture_pixel_formats_reports_baseline() {
        let caps = probe_capture_pixel_formats()
            .await
            .expect("SCK probe should succeed when permission is granted");
        // 420v is the floor — every Mac with SCK accepts it. If this
        // is false, the probe shape itself is broken.
        assert!(
            caps.yuv420_video_range,
            "SCK should always accept 420v (NV12 video range) — \
             probe likely returning false negatives across the board"
        );
        // Print everything else for the operator running the probe to
        // record on a new Mac model.
        eprintln!("SCK probe matrix: {caps:#?}");
    }

    /// Real ScreenCaptureKit smoke test for the production capture start path.
    /// The pixel-format probe above proves SCK accepts configurations; this
    /// verifies `start(BGRA)` actually delivers a GPU IOSurface frame through
    /// `CaptureHandle`.
    #[tokio::test]
    #[ignore = "requires macOS + ScreenRecording permission + active display; run with: cargo test -p tether-capture -- --ignored"]
    async fn sck_capture_start_delivers_one_frame() {
        // Anchor `MonoNanos`' lazily-initialised process-start epoch before
        // capture begins. The epoch is set on the first `now()` call ever made
        // in the process, so without this a capture stamp that happens to be
        // that first call reads exactly 0 — a real value, but indistinguishable
        // from "unpopulated". With the epoch anchored here, SCK startup burns
        // real time and the delivered frame's stamp is reliably later.
        let before = MonoNanos::now();

        let handle = start(sck_bgra_pixel_format())
            .await
            .expect("start ScreenCaptureKit BGRA capture");
        let rx = handle.into_rx();
        let frame = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("receive first ScreenCaptureKit frame within 10s");

        assert!(frame.width() > 0, "captured frame width must be non-zero");
        assert!(frame.height() > 0, "captured frame height must be non-zero");
        let (t_kernel, t_userspace) = frame.timestamps();
        assert!(
            t_userspace >= before,
            "userspace timestamp should be stamped at/after capture start ({before:?})"
        );
        // macOS has no separate kernel capture clock, so the capture path
        // mirrors the userspace stamp into the kernel slot (see `build_frame`).
        assert_eq!(
            t_kernel, t_userspace,
            "macOS capture mirrors the userspace stamp into the kernel slot"
        );

        let CapturedFrame::Gpu(gpu) = frame else {
            panic!("ScreenCaptureKit should deliver GPU IOSurface frames");
        };
        let GpuCapturedSource::IOSurface(surface) = gpu.source;
        assert!(!surface.surface.is_null(), "IOSurfaceRef must be non-null");
        assert_eq!(surface.width, gpu.width);
        assert_eq!(surface.height, gpu.height);
        assert_eq!(
            surface.pixel_format,
            u32::from_be_bytes(*b"BGRA"),
            "BGRA smoke test should receive BGRA IOSurfaces"
        );
        // `native_damage` is intentionally not asserted here: SCK does not
        // attach an `SCStreamFrameInfo.status` to every sample (the first
        // frame frequently arrives without one), so it is legitimately
        // `None` for a delivered frame. The status→damage mapping is unit-
        // tested directly in `native_damage_for_frame_status`'s tests; this
        // smoke test only verifies one real delivered resource shape.
    }

    #[test]
    fn sck_pixel_format_for_profile_covers_every_modeled_combo() {
        use tether_protocol::control::{CodecKind, VideoProfile};
        for (chroma, bit_depth, expected_fourcc) in [
            (ChromaSubsampling::Yuv420, 8, *b"420v"),
            (ChromaSubsampling::Yuv420, 10, *b"x420"),
            (ChromaSubsampling::Yuv444, 8, *b"444v"),
            (ChromaSubsampling::Yuv444, 10, *b"xf44"),
        ] {
            let profile = VideoProfile {
                codec: CodecKind::Hevc,
                chroma,
                bit_depth,
            };
            let mapping = sck_pixel_format_for_profile(profile);
            match mapping {
                SckCapabilityCheck::Supported(pf) => {
                    let actual: FourCharCode = pf.into();
                    assert_eq!(
                        actual.as_u32(),
                        u32::from_be_bytes(expected_fourcc),
                        "({chroma:?}, {bit_depth}) should map to {:?}",
                        std::str::from_utf8(&expected_fourcc).unwrap_or("<bin>")
                    );
                }
                SckCapabilityCheck::Unsupported => {
                    panic!("({chroma:?}, {bit_depth}) should be supported");
                }
            }
        }
    }

    #[test]
    fn sck_pixel_format_for_profile_rejects_unmodeled_bit_depths() {
        use tether_protocol::control::{CodecKind, VideoProfile};
        let bogus = VideoProfile {
            codec: CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 12,
        };
        assert!(matches!(
            sck_pixel_format_for_profile(bogus),
            SckCapabilityCheck::Unsupported
        ));
    }

    #[test]
    fn is_deliverable_returns_true_only_when_caps_accept_mapped_format() {
        use tether_protocol::control::{CodecKind, VideoProfile};
        // Caps: only 420v accepted (the universal floor).
        let caps = SckCaptureCapability {
            yuv420_video_range: true,
            ..Default::default()
        };
        let p_420_8 = VideoProfile {
            codec: CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 8,
        };
        let p_420_10 = VideoProfile {
            codec: CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 10,
        };
        assert!(sck_pixel_format_for_profile(p_420_8).is_deliverable(&caps));
        assert!(!sck_pixel_format_for_profile(p_420_10).is_deliverable(&caps));

        // Caps: x420 also accepted — now 10-bit 4:2:0 is deliverable too.
        let caps10 = SckCaptureCapability {
            yuv420_video_range: true,
            yuv420_10bit_video_range: true,
            ..Default::default()
        };
        assert!(sck_pixel_format_for_profile(p_420_10).is_deliverable(&caps10));
    }
}

// === Core Graphics FFI for physical-pixel display dimensions ===
//
// SCDisplay::width/height (from screencapturekit) and the simpler
// `CGDisplayPixelsWide`/`High` APIs both report the display's
// *active mode* dimensions, which on a Retina display configured
// with HiDPI scaling is the scaled (logical-point) grid — half the
// backing-pixel grid on a 2× display. Capturing at that grid forces
// the wire stream to half resolution and makes the client renderer
// upscale, producing the blurry output a HiDPI session would
// otherwise be missing.
//
// The correct path is `CGDisplayCopyDisplayMode` to get the active
// `CGDisplayModeRef`, then `CGDisplayModeGetPixelWidth` /
// `GetPixelHeight` which return the actual backing-pixel count
// regardless of the active scaling mode. Declared inline here rather
// than pulling in the full `core-graphics` crate just for these
// functions.
//
// Apple docs:
//   https://developer.apple.com/documentation/coregraphics/1454417-cgdisplaycopydisplaymode
//   https://developer.apple.com/documentation/coregraphics/1454205-cgdisplaymodegetpixelwidth
//   https://developer.apple.com/documentation/coregraphics/1454163-cgdisplaymodegetpixelheight

/// `CGDisplayModeRef` is an opaque CF type. We treat it as an opaque
/// pointer.
type CGDisplayModeRef = *mut std::ffi::c_void;

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    /// Returns a `+1` retained `CGDisplayModeRef` for the display's
    /// active mode (or null if the display is invalid).
    fn CGDisplayCopyDisplayMode(display: u32) -> CGDisplayModeRef;
    fn CGDisplayModeGetPixelWidth(mode: CGDisplayModeRef) -> usize;
    fn CGDisplayModeGetPixelHeight(mode: CGDisplayModeRef) -> usize;
    fn CGDisplayModeGetRefreshRate(mode: CGDisplayModeRef) -> f64;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: *const std::ffi::c_void);
}

/// Balance the `+1` retain from `CGDisplayCopyDisplayMode`.
unsafe fn cf_release(mode: CGDisplayModeRef) {
    if !mode.is_null() {
        // SAFETY: caller's contract — the pointer was a +1 retained
        // CF reference and is not used after this call.
        unsafe { CFRelease(mode as *const std::ffi::c_void) };
    }
}

fn active_display_mode(display_id: u32) -> Option<(u32, u32, u32)> {
    // SAFETY: CoreGraphics returns either a +1 retained display mode for an
    // attached display or null. We release non-null results below.
    let mode = unsafe { CGDisplayCopyDisplayMode(display_id) };
    if mode.is_null() {
        return None;
    }

    let width = u32::try_from(unsafe { CGDisplayModeGetPixelWidth(mode) }).ok();
    let height = u32::try_from(unsafe { CGDisplayModeGetPixelHeight(mode) }).ok();
    let refresh = unsafe { CGDisplayModeGetRefreshRate(mode) };
    // SAFETY: balances the +1 retain from CGDisplayCopyDisplayMode.
    unsafe { cf_release(mode) };

    let width = width?;
    let height = height?;
    let refresh_millihz = if refresh.is_finite() && refresh > 0.0 {
        round_f64_to_u32(refresh * 1000.0, 1, u32::MAX)
    } else {
        60_000
    };
    Some((width, height, refresh_millihz))
}

fn round_f64_to_i32(value: f64) -> i32 {
    if !value.is_finite() {
        0
    } else {
        let clamped = value
            .round()
            .clamp(f64::from(i32::MIN), f64::from(i32::MAX));
        #[allow(clippy::cast_possible_truncation)]
        {
            clamped as i32
        }
    }
}

fn round_f64_to_u32(value: f64, min: u32, max: u32) -> u32 {
    if !value.is_finite() {
        return min;
    }
    let clamped = value.round().clamp(f64::from(min), f64::from(max));
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        clamped as u32
    }
}
