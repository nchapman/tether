//! Client-side audio output: a cpal stream draining a lock-free ring.
//!
//! The cpal output callback is the consumer (pulls on the device clock); the
//! Opus decode thread is the producer (pushes via an [`AudioSink`]). They are
//! joined by a lock-free SPSC [`ring`] so the callback — a hard-real-time thread
//! — never takes a lock, and a pure [`PlaybackPolicy`] on the consumer side
//! handles priming, the latency cap, and underruns.
//!
//! cpal's `Stream` is thread-bound (`!Send`), so [`AudioPlayer`] — which owns it
//! — stays on its creating thread, while the `Send` [`AudioSink`] (the ring
//! producer) travels to the decode thread.

mod policy;
mod ring;

use std::sync::atomic::Ordering;
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::{AudioFrame, OpusConfig};
use policy::{PlaybackPolicy, PlaybackStats};

/// Default jitter cushion before playback starts. 30 ms is a tight cushion that
/// keeps the steady-state latency this primes to low; with RED now covering
/// loss, we don't need a fatter buffer for resilience. (Same operating point as
/// Moonlight's SDL renderer, which gates feeding at ~30 ms of queued audio.)
pub const DEFAULT_TARGET_MS: u32 = 30;
/// Default latency ceiling; beyond this the oldest audio is dropped back to the
/// target. 50 ms keeps the absolute latency floor low under clock drift — far
/// tighter than the old 120 ms, so drops are smaller. (Comparable to Moonlight's
/// ~50 ms SDL queue cap.)
pub const DEFAULT_MAX_MS: u32 = 50;

/// Errors bringing up audio output.
#[derive(Debug, thiserror::Error)]
pub enum PlaybackError {
    #[error("no default audio output device")]
    NoDevice,
    #[error("no output config supports {channels}-channel f32 at {rate} Hz")]
    NoSupportedConfig { rate: u32, channels: u8 },
    #[error("query output configs: {0}")]
    Configs(#[from] cpal::SupportedStreamConfigsError),
    #[error("build output stream: {0}")]
    BuildStream(#[from] cpal::BuildStreamError),
    #[error("start output stream: {0}")]
    PlayStream(#[from] cpal::PlayStreamError),
}

/// Producer handle to the playback ring. Held by the decode thread to push PCM;
/// the [`AudioPlayer`] keeps the cpal stream alive elsewhere. Not `Clone` —
/// there is exactly one producer.
pub struct AudioSink {
    producer: ring::Producer,
    stats: Arc<PlaybackStats>,
}

impl AudioSink {
    /// Push a decoded frame for playback. If the ring is momentarily full the
    /// newest samples are dropped (the consumer trims to the latency target
    /// first, so this is rare).
    pub fn submit(&self, frame: &AudioFrame) {
        self.producer.push(&frame.samples);
    }

    /// `(underruns, dropped_samples, buffered_samples)` — for periodic logging.
    #[must_use]
    pub fn stats(&self) -> (u64, u64, usize) {
        (
            self.stats.underruns.load(Ordering::Relaxed),
            self.stats.dropped_samples.load(Ordering::Relaxed),
            self.producer.buffered(),
        )
    }
}

/// Owns the live cpal output stream; keep it alive for the session (drop stops
/// playback). `!Send` — create it on the thread that will keep it.
pub struct AudioPlayer {
    _stream: cpal::Stream,
}

impl AudioPlayer {
    /// Open the default output device at `cfg`'s rate/channels and start playing
    /// from a fresh ring. Returns the player (keep it alive) and the producer
    /// [`AudioSink`] for the decode thread.
    pub fn new(
        cfg: OpusConfig,
        target_ms: u32,
        max_ms: u32,
    ) -> Result<(Self, AudioSink), PlaybackError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or(PlaybackError::NoDevice)?;
        let stream_config = pick_output_config(&device, cfg.sample_rate, cfg.channels)?;

        // Ring capacity comfortably exceeds the latency ceiling so the policy's
        // drop-oldest is the active bound (the producer-side full-ring drop is
        // just a safety valve).
        let per_ms = (cfg.sample_rate as usize * usize::from(cfg.channels.max(1))) / 1000;
        let ring_capacity = (per_ms * max_ms as usize * 2).max(per_ms * 4);
        let (producer, consumer) = ring::channel(ring_capacity);

        let stats = Arc::new(PlaybackStats::default());
        let mut policy = PlaybackPolicy::new(
            cfg.channels,
            cfg.sample_rate,
            target_ms,
            max_ms,
            Arc::clone(&stats),
        );

        let stream = device.build_output_stream(
            &stream_config,
            move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let plan = policy.plan(consumer.buffered(), out.len());
                consumer.skip(plan.skip);
                consumer.pop(&mut out[..plan.copy]);
                out[plan.copy..].fill(0.0);
            },
            move |err| tracing::warn!(error = %err, "audio output stream error"),
            None,
        )?;
        stream.play()?;
        Ok((Self { _stream: stream }, AudioSink { producer, stats }))
    }

    /// [`AudioPlayer::new`] with the default cushion/ceiling.
    pub fn with_defaults(cfg: OpusConfig) -> Result<(Self, AudioSink), PlaybackError> {
        Self::new(cfg, DEFAULT_TARGET_MS, DEFAULT_MAX_MS)
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
        .ok_or(PlaybackError::NoSupportedConfig {
            rate: sample_rate,
            channels,
        })?;
    Ok(chosen.with_sample_rate(target).config())
}
