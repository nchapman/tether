//! System-output ("what's playing on the host") audio capture.
//!
//! Unlike playback — where one cross-platform crate (cpal) serves every OS —
//! capturing the system's *output* is irreducibly platform-specific (Linux
//! PipeWire monitor source, macOS ScreenCaptureKit, Windows WASAPI loopback),
//! so each backend lives behind a cfg gate, mirroring `tether-capture`. The
//! consumer-facing shape is uniform: [`start`] returns an
//! [`AudioCaptureHandle`] carrying a `Receiver<AudioFrame>`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::Receiver;

use crate::{AudioFrame, OpusConfig};

#[cfg(target_os = "macos")]
pub mod macos;

/// Consumer handle for captured system audio. Holds the PCM receiver plus the
/// stop signal + backend thread; dropping it tells the backend to tear down.
pub struct AudioCaptureHandle {
    /// Captured interleaved-f32 frames, drop-oldest under backpressure.
    pub rx: Receiver<AudioFrame>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl AudioCaptureHandle {
    pub(crate) fn from_parts(
        rx: Receiver<AudioFrame>,
        stop: Arc<AtomicBool>,
        thread: JoinHandle<()>,
    ) -> Self {
        Self {
            rx,
            stop,
            thread: Some(thread),
        }
    }

    /// Stop capture and join the backend thread.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for AudioCaptureHandle {
    fn drop(&mut self) {
        // Best-effort teardown if the caller didn't `stop()` explicitly; the
        // backend notices the flag within its park interval and tears down.
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Errors starting system-audio capture.
#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    /// No capture backend is implemented for this platform yet.
    #[error("system audio capture is not yet supported on this platform")]
    Unsupported,
    /// A platform backend failed to start.
    #[error("audio capture backend: {0}")]
    Backend(String),
}

/// Whether a system-audio capture backend exists for this platform. The host
/// uses this to decide whether to advertise audio at all, so a client never
/// opts into audio a backend-less host can't deliver. Linux/Windows flip to
/// `true` when their backends land.
#[must_use]
pub fn is_supported() -> bool {
    cfg!(target_os = "macos")
}

/// Start capturing system-output audio for the current platform.
///
/// Returns [`CaptureError::Unsupported`] on platforms whose backend isn't wired
/// yet (Linux/Windows land in follow-up changes) so the host can degrade to a
/// silent session rather than fail.
pub fn start(cfg: OpusConfig) -> Result<AudioCaptureHandle, CaptureError> {
    #[cfg(target_os = "macos")]
    {
        macos::start(cfg)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = cfg;
        Err(CaptureError::Unsupported)
    }
}
