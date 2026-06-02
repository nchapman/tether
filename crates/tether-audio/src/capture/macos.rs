//! macOS system-audio capture via ScreenCaptureKit.
//!
//! SCK is the only native path to system *output* audio on macOS, and the
//! video path already runs an SCStream — so audio is the same framework. We
//! run a small standalone audio SCStream here (tiny video config we never
//! read) to keep `tether-audio` self-contained; it could later share the video
//! path's stream. SCK delivers 48 kHz float PCM on its dispatch queue; we
//! interleave it into [`AudioFrame`]s and push them down a bounded channel.
//!
//! Requires Screen Recording permission (SCK audio is gated behind it), same
//! as the video capture path.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::{bounded, Sender, TrySendError};
use screencapturekit::cm::{AudioBufferList, CMSampleBufferExt};
use screencapturekit::prelude::{
    CMSampleBuffer, SCContentFilter, SCShareableContent, SCStream, SCStreamConfiguration,
    SCStreamOutputTrait, SCStreamOutputType,
};

use crate::capture::{AudioCaptureHandle, CaptureError};
use crate::{AudioFrame, OpusConfig};

/// ~80 ms of 10 ms frames. Capture should always be drained promptly; a short
/// queue keeps capture-side latency tight and drops the newest on overflow.
const CAPTURE_CHANNEL_DEPTH: usize = 8;

/// Start an SCK audio stream on a dedicated supervisor thread.
pub fn start(cfg: OpusConfig) -> Result<AudioCaptureHandle, CaptureError> {
    let (tx, rx) = bounded::<AudioFrame>(CAPTURE_CHANNEL_DEPTH);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = Arc::clone(&stop);

    let thread = thread::Builder::new()
        .name("tether-audio-capture".into())
        .spawn(move || {
            if let Err(e) = run(cfg, tx, &stop_thread) {
                tracing::error!(error = %e, "macOS audio capture ended with error");
            }
        })
        .map_err(|e| CaptureError::Backend(format!("spawn capture thread: {e}")))?;

    Ok(AudioCaptureHandle::from_parts(rx, stop, thread))
}

fn run(cfg: OpusConfig, tx: Sender<AudioFrame>, stop: &AtomicBool) -> Result<(), CaptureError> {
    let content = SCShareableContent::get()
        .map_err(|e| CaptureError::Backend(format!("SCShareableContent::get: {e:?}")))?;
    let display = content
        .displays()
        .into_iter()
        .next()
        .ok_or_else(|| CaptureError::Backend("no display to attach audio capture".into()))?;

    let filter = SCContentFilter::create()
        .with_display(&display)
        .with_excluding_windows(&[])
        .build();

    // Audio-only intent: minimal video config we never read, system audio on,
    // and exclude our own process so playback doesn't loop back into capture.
    let config = SCStreamConfiguration::new()
        .with_width(16)
        .with_height(16)
        .with_captures_audio(true)
        .with_sample_rate(i32::try_from(cfg.sample_rate).unwrap_or(48_000))
        .with_channel_count(i32::from(cfg.channels))
        .with_excludes_current_process_audio(true);

    let mut stream = SCStream::new(&filter, &config);
    stream.add_output_handler(
        AudioHandler {
            tx,
            channels: cfg.channels,
            sample_rate: cfg.sample_rate,
            dropped: AtomicBool::new(false),
        },
        SCStreamOutputType::Audio,
    );
    stream
        .start_capture()
        .map_err(|e| CaptureError::Backend(format!("start_capture: {e:?}")))?;
    tracing::info!(
        sample_rate = cfg.sample_rate,
        channels = cfg.channels,
        "macOS system audio capture started (ScreenCaptureKit)"
    );

    // SCK fires callbacks on its own dispatch queue; this thread only needs to
    // keep the stream object alive until asked to stop.
    while !stop.load(Ordering::Relaxed) {
        thread::park_timeout(Duration::from_millis(100));
    }

    if let Err(e) = stream.stop_capture() {
        tracing::warn!(error = ?e, "SCStream::stop_capture failed during shutdown");
    }
    Ok(())
}

struct AudioHandler {
    tx: Sender<AudioFrame>,
    channels: u8,
    sample_rate: u32,
    /// One-shot rate-limit so a full channel logs once, not every callback.
    dropped: AtomicBool,
}

impl SCStreamOutputTrait for AudioHandler {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        if !matches!(of_type, SCStreamOutputType::Audio) {
            return;
        }
        let Some(list) = sample.audio_buffer_list() else {
            return;
        };
        let Some(frame) = frame_from_buffer_list(&list, self.channels, self.sample_rate) else {
            return;
        };
        match self.tx.try_send(frame) {
            Ok(()) => {
                self.dropped.store(false, Ordering::Relaxed);
            }
            Err(TrySendError::Full(_)) => {
                if !self.dropped.swap(true, Ordering::Relaxed) {
                    tracing::warn!("audio capture channel full; dropping frames (consumer behind)");
                }
            }
            Err(TrySendError::Disconnected(_)) => {
                // Consumer gone; the supervisor thread will stop the stream.
            }
        }
    }
}

/// Convert an SCK [`AudioBufferList`] to one interleaved-f32 [`AudioFrame`].
///
/// SCK delivers system audio as deinterleaved (planar) float — one mono buffer
/// per channel — but we also handle a single interleaved buffer defensively.
fn frame_from_buffer_list(
    list: &AudioBufferList,
    channels: u8,
    sample_rate: u32,
) -> Option<AudioFrame> {
    let nb = list.num_buffers();
    if nb == 0 {
        return None;
    }
    let first = list.get(0)?;

    if first.number_channels == 1 && nb >= channels as usize {
        // Planar: interleave the per-channel mono planes.
        let ch = channels as usize;
        let planes: Vec<Vec<f32>> = (0..ch)
            .map(|c| bytes_to_f32(list.get(c).map_or(&[][..], |b| b.data())))
            .collect();
        let n = planes.iter().map(Vec::len).min().unwrap_or(0);
        if n == 0 {
            return None;
        }
        let mut samples = Vec::with_capacity(n * ch);
        for i in 0..n {
            for plane in &planes {
                samples.push(plane[i]);
            }
        }
        Some(AudioFrame::new(sample_rate, channels, samples))
    } else {
        // Already interleaved in one buffer. Trust the configured channel count
        // (SCK is set up with `with_channel_count(channels)`) rather than the
        // buffer metadata, so the frame's declared layout always matches what
        // the encoder was built for; guard against a length that doesn't divide.
        let interleaved = bytes_to_f32(first.data());
        if interleaved.is_empty() || interleaved.len() % channels.max(1) as usize != 0 {
            return None;
        }
        Some(AudioFrame::new(sample_rate, channels, interleaved))
    }
}

/// Reinterpret a native-endian f32 byte buffer (SCK's PCM) as `f32`s. Uses
/// `from_ne_bytes` rather than a pointer cast so we make no alignment
/// assumption about the SCK buffer.
fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
