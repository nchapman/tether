//! Client-side audio output: a cpal stream draining the [`JitterBuffer`].
//!
//! The cpal output callback is the consumer (pulls on the device clock); the
//! Opus decode thread is the producer (pushes via an [`AudioSink`]). cpal's
//! `Stream` is thread-bound (`!Send`), so [`AudioPlayer`] — which owns it —
//! stays on its creating thread, while the cheap, `Send` [`AudioSink`] travels
//! to the decode thread.

pub mod jitter;

use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::{AudioFrame, OpusConfig};
use jitter::JitterBuffer;

/// Default jitter cushion before playback starts. ~40 ms keeps a LAN stream
/// smooth without much added latency (Moonlight's level-2 buffer is in this
/// range).
pub const DEFAULT_TARGET_MS: u32 = 40;
/// Default latency ceiling; beyond this the oldest audio is dropped.
pub const DEFAULT_MAX_MS: u32 = 120;

/// Errors bringing up audio output.
#[derive(Debug, thiserror::Error)]
pub enum PlaybackError {
    #[error("no default audio output device")]
    NoDevice,
    #[error("no output config supports {0} Hz f32 stereo")]
    NoSupportedConfig(u32),
    #[error("query output configs: {0}")]
    Configs(#[from] cpal::SupportedStreamConfigsError),
    #[error("build output stream: {0}")]
    BuildStream(#[from] cpal::BuildStreamError),
    #[error("start output stream: {0}")]
    PlayStream(#[from] cpal::PlayStreamError),
}

/// Producer handle to the playback buffer. Clone it to the decode thread to
/// push PCM; the [`AudioPlayer`] keeps the stream alive elsewhere.
#[derive(Clone)]
pub struct AudioSink {
    jitter: Arc<Mutex<JitterBuffer>>,
}

impl AudioSink {
    /// Push a decoded frame for playback.
    pub fn submit(&self, frame: &AudioFrame) {
        if let Ok(mut j) = self.jitter.lock() {
            j.push(&frame.samples);
        }
    }

    /// `(underruns, overruns, buffered_samples)` — for periodic logging.
    #[must_use]
    pub fn stats(&self) -> (u64, u64, usize) {
        self.jitter
            .lock()
            .map(|j| (j.underruns(), j.overruns(), j.buffered_samples()))
            .unwrap_or_default()
    }
}

/// Owns the live cpal output stream; keep it alive for the session (drop stops
/// playback). `!Send` — create it on the thread that will keep it.
pub struct AudioPlayer {
    _stream: cpal::Stream,
    sink: AudioSink,
}

impl AudioPlayer {
    /// Open the default output device at `cfg`'s rate/channels and start
    /// playing from a fresh jitter buffer.
    pub fn new(cfg: OpusConfig, target_ms: u32, max_ms: u32) -> Result<Self, PlaybackError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or(PlaybackError::NoDevice)?;
        let stream_config = pick_output_config(&device, cfg.sample_rate, cfg.channels)?;

        let jitter = Arc::new(Mutex::new(JitterBuffer::new(
            cfg.channels,
            cfg.sample_rate,
            target_ms,
            max_ms,
        )));
        let sink = AudioSink {
            jitter: Arc::clone(&jitter),
        };

        let cb_jitter = Arc::clone(&jitter);
        let stream = device.build_output_stream(
            &stream_config,
            move |out: &mut [f32], _: &cpal::OutputCallbackInfo| match cb_jitter.lock() {
                Ok(mut j) => {
                    j.pull(out);
                }
                Err(_) => out.fill(0.0),
            },
            move |err| tracing::warn!(error = %err, "audio output stream error"),
            None,
        )?;
        stream.play()?;
        Ok(Self {
            _stream: stream,
            sink,
        })
    }

    /// [`AudioPlayer::new`] with the default cushion/ceiling.
    pub fn with_defaults(cfg: OpusConfig) -> Result<Self, PlaybackError> {
        Self::new(cfg, DEFAULT_TARGET_MS, DEFAULT_MAX_MS)
    }

    /// A producer handle for the decode thread.
    #[must_use]
    pub fn sink(&self) -> AudioSink {
        self.sink.clone()
    }
}

/// Pick an f32 output config with exactly `channels` channels at `sample_rate`.
/// CoreAudio/WASAPI resample to the hardware rate internally, so a default
/// device accepts 48 kHz even when its hardware runs at another rate.
fn pick_output_config(
    device: &cpal::Device,
    sample_rate: u32,
    channels: u8,
) -> Result<cpal::StreamConfig, PlaybackError> {
    // cpal 0.17: SampleRate is a plain u32.
    let target: cpal::SampleRate = sample_rate;
    let chosen = device
        .supported_output_configs()?
        .filter(|c| c.sample_format() == cpal::SampleFormat::F32)
        .filter(|c| c.channels() == u16::from(channels))
        .find(|c| c.min_sample_rate() <= target && target <= c.max_sample_rate())
        .ok_or(PlaybackError::NoSupportedConfig(sample_rate))?;
    Ok(chosen.with_sample_rate(target).config())
}
