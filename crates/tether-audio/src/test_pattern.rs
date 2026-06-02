//! Synthetic PCM for hardware-free tests — a deterministic sine source so the
//! codec and (later) pipeline tests have real signal to push without an audio
//! device. Mirrors `tether-capture`'s `test_pattern` for video.

use crate::AudioFrame;

/// Generate one `frames`-long block of a `freq_hz` sine at amplitude 0.5,
/// interleaved across `channels`. `start_sample` is the absolute sample index
/// of the block's first sample, so consecutive blocks line up into a
/// continuous, phase-coherent tone.
#[must_use]
pub fn sine_frame(
    sample_rate: u32,
    channels: u8,
    freq_hz: f32,
    frames: usize,
    start_sample: usize,
) -> AudioFrame {
    let mut samples = Vec::with_capacity(frames * channels as usize);
    let omega = std::f32::consts::TAU * freq_hz / sample_rate as f32;
    for i in 0..frames {
        // start_sample stays well under 2^24 in tests, so the index is exact
        // in f32 and the tone doesn't drift.
        let t = (start_sample + i) as f32;
        let v = (omega * t).sin() * 0.5;
        for _ in 0..channels {
            samples.push(v);
        }
    }
    AudioFrame::new(sample_rate, channels, samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_frame_has_expected_shape_and_energy() {
        let f = sine_frame(48_000, 2, 440.0, 480, 0);
        assert_eq!(f.channels, 2);
        assert_eq!(f.frames(), 480);
        assert_eq!(f.samples.len(), 960);
        // Left and right carry the identical mono tone (interleaved).
        for pair in f.samples.chunks_exact(2) {
            assert_eq!(pair[0], pair[1]);
        }
        assert!(!f.is_silent());
    }
}
