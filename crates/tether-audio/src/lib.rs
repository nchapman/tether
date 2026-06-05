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
//! The codec binds libopus directly (`opus_sys`), linking the standalone
//! `libopus` archive the static FFmpeg package already stages — the same one
//! FFmpeg's `libavcodec` resolves against, so no second copy enters the binary.
//! Going direct (rather than through FFmpeg's avcodec wrapper as the video codec
//! does) is what gives us real `opus_decode(NULL)` packet-loss concealment and
//! live encoder ctl. Audio is software-coded end to end — hardware audio codecs
//! aren't warranted at ~128 kbps — so the codec round-trip runs in the default
//! (no-hardware) test set.

pub mod capture;
pub mod codec;
mod opus_sys;
pub mod playback;
pub mod recovery;
pub mod redundancy;
pub mod test_pattern;

pub use capture::{AudioCaptureHandle, CaptureError};
pub use codec::{OpusConfig, OpusDecoder, OpusEncoder};
pub use playback::{AudioPlayer, AudioSink, PlaybackError};
pub use recovery::{LossRecovery, RecoveryStats};
pub use redundancy::RedundancyBuffer;

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
    /// Channel count outside the mono/stereo range v1 supports. Guards the
    /// codec's per-channel handling against an out-of-range config.
    #[error("unsupported channel count: {0} (v1 supports 1 or 2)")]
    UnsupportedChannelCount(u8),

    /// A libopus call failed; carries the entry point and the library's own
    /// `opus_strerror` message.
    #[error("libopus: {0}")]
    Opus(String),
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, AudioError>;
