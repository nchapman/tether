//! Audio pipeline for Tether: system-output capture → Opus encode →
//! (unreliable datagram, owned by `tether-transport`) → Opus decode →
//! jitter-buffered playback.
//!
//! This crate owns the platform-independent pieces — the PCM frame type and
//! the Opus codec (`codec`) — plus, in later layers, per-platform capture and
//! cpal-based playback. The wire shape (`tether_protocol::audio::AudioPacket`,
//! `AudioConfig`) lives in `tether-protocol`; this crate produces and consumes
//! the Opus payloads that ride it.
//!
//! Opus is statically linked through the same LGPL FFmpeg 8.1 as the video
//! codec (libopus, every target). Audio is software-coded end to end —
//! hardware audio codecs aren't warranted at ~128 kbps — so the codec
//! round-trip runs in the default (no-hardware) test set.

pub mod capture;
pub mod codec;
pub mod playback;
pub mod test_pattern;

pub use capture::{AudioCaptureHandle, CaptureError};
pub use codec::{OpusConfig, OpusDecoder, OpusEncoder};
pub use playback::{AudioPlayer, AudioSink, PlaybackError};

/// Session audio constants. 48 kHz stereo is the only configuration we ship
/// in v1; the protocol's `AudioConfig` keeps the door open for surround.
pub const SAMPLE_RATE_HZ: u32 = 48_000;
pub const CHANNELS: u8 = 2;

/// A block of interleaved PCM audio.
///
/// Samples are `f32` in `[-1.0, 1.0]`, interleaved by channel
/// (`L R L R …` for stereo). This is the format cpal hands us on capture and
/// the format the playback callback wants, so the codec speaks it on both
/// sides and the hot path stays copy-light.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioFrame {
    pub sample_rate: u32,
    pub channels: u8,
    /// Interleaved PCM. `len() == frames() * channels`.
    pub samples: Vec<f32>,
}

impl AudioFrame {
    /// Build a frame, asserting the sample count is a whole number of frames.
    pub fn new(sample_rate: u32, channels: u8, samples: Vec<f32>) -> Self {
        debug_assert!(
            channels > 0 && samples.len() % channels as usize == 0,
            "interleaved sample count {} is not a multiple of {channels} channels",
            samples.len()
        );
        Self {
            sample_rate,
            channels,
            samples,
        }
    }

    /// A silent frame of `frames` samples-per-channel — used as v1 loss
    /// concealment (see [`OpusDecoder::conceal`]).
    pub fn silence(sample_rate: u32, channels: u8, frames: usize) -> Self {
        Self {
            sample_rate,
            channels,
            samples: vec![0.0; frames * channels as usize],
        }
    }

    /// Number of samples per channel.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1) as usize
    }

    /// True if every sample is exactly zero.
    #[must_use]
    pub fn is_silent(&self) -> bool {
        self.samples.iter().all(|&s| s == 0.0)
    }
}

/// Errors from the audio codec.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// FFmpeg didn't build the named codec — should never happen with our
    /// pinned static FFmpeg, but surfaced rather than panicked.
    #[error("codec not found in this FFmpeg build: {0}")]
    CodecNotFound(&'static str),

    /// The encoder advertised no sample format we can feed.
    #[error("opus encoder does not accept a supported sample format")]
    NoSupportedSampleFormat,

    /// Channel count outside the mono/stereo range v1 supports. Guards the
    /// decoder's per-channel plane indexing against an out-of-range config.
    #[error("unsupported channel count: {0} (v1 supports 1 or 2)")]
    UnsupportedChannelCount(u8),

    /// A decoded frame came back in a sample format we don't convert.
    #[error("unsupported decoded sample format: {0}")]
    UnsupportedSampleFormat(i32),

    /// Wraps an underlying ffmpeg/rsmpeg error.
    #[error("ffmpeg: {0}")]
    Ffmpeg(#[from] rsmpeg::error::RsmpegError),
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, AudioError>;
