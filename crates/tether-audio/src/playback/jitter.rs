//! A small interleaved-PCM jitter buffer that sits between the Opus decode
//! thread (producer) and the cpal output callback (consumer).
//!
//! It absorbs network/decode timing jitter by buffering a target amount of
//! audio before playback starts, caps end-to-end latency by dropping the
//! oldest samples when the producer runs ahead, and conceals a starved
//! consumer with silence. This is the "level-2" buffer in Moonlight's
//! two-stage scheme — small (tens of ms), device-clock-driven, no explicit A/V
//! timestamp steering in v1.
//!
//! Pure logic, no audio device — fully unit-tested.

use std::collections::VecDeque;

/// Interleaved-f32 jitter buffer for a fixed channel count.
pub struct JitterBuffer {
    ring: VecDeque<f32>,
    /// Target fill (in interleaved samples) before playback starts or resumes.
    target: usize,
    /// Hard cap (in interleaved samples); beyond this we drop oldest to bound
    /// latency.
    max: usize,
    /// While priming, `pull` emits silence until `target` is buffered.
    priming: bool,
    underruns: u64,
    overruns: u64,
}

impl JitterBuffer {
    /// `target_ms` is the cushion built before audio starts flowing; `max_ms`
    /// is the latency ceiling past which the oldest audio is dropped.
    #[must_use]
    pub fn new(channels: u8, sample_rate: u32, target_ms: u32, max_ms: u32) -> Self {
        let per_ms = (sample_rate as usize * channels.max(1) as usize) / 1000;
        let target = per_ms * target_ms as usize;
        let max = per_ms * max_ms as usize;
        Self {
            ring: VecDeque::with_capacity(max + per_ms),
            target,
            max: max.max(target),
            priming: true,
            underruns: 0,
            overruns: 0,
        }
    }

    /// Push interleaved PCM from the decoder. If the buffer has grown past the
    /// latency ceiling (consumer fell behind, or the network burst), drop the
    /// oldest audio back down to the target cushion.
    pub fn push(&mut self, samples: &[f32]) {
        self.ring.extend(samples.iter().copied());
        if self.ring.len() > self.max {
            let drop = self.ring.len() - self.target;
            self.ring.drain(..drop);
            self.overruns += 1;
        }
        if self.priming && self.ring.len() >= self.target {
            self.priming = false;
        }
    }

    /// Fill `out` with interleaved PCM for the output device. Returns the count
    /// of real (non-silence) samples written. While priming or on a full
    /// drain, the remainder is silence and the buffer re-primes so it rebuilds
    /// a cushion instead of stuttering sample-by-sample.
    pub fn pull(&mut self, out: &mut [f32]) -> usize {
        if self.priming {
            out.fill(0.0);
            return 0;
        }
        let n = out.len().min(self.ring.len());
        for slot in out.iter_mut().take(n) {
            *slot = self.ring.pop_front().expect("n <= ring length");
        }
        if n < out.len() {
            out[n..].fill(0.0);
            self.underruns += 1;
            // A short read means we drained the ring; rebuild the cushion.
            self.priming = true;
        }
        n
    }

    /// Samples currently buffered (interleaved).
    #[must_use]
    pub fn buffered_samples(&self) -> usize {
        self.ring.len()
    }

    /// Times the consumer outran the producer (since construction).
    #[must_use]
    pub fn underruns(&self) -> u64 {
        self.underruns
    }

    /// Times the latency ceiling forced an oldest-sample drop.
    #[must_use]
    pub fn overruns(&self) -> u64 {
        self.overruns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 48 kHz stereo: 96 interleaved samples per ms.
    fn buf() -> JitterBuffer {
        JitterBuffer::new(2, 48_000, 10, 40)
    }

    #[test]
    fn primes_with_silence_until_target_buffered() {
        let mut jb = buf();
        let mut out = [1.0f32; 64];
        // Nothing pushed yet → pure silence, no real samples.
        assert_eq!(jb.pull(&mut out), 0);
        assert!(out.iter().all(|&s| s == 0.0));

        // Push less than the 10 ms target (960 samples) → still priming.
        jb.push(&[0.5; 480]);
        let mut out = [1.0f32; 64];
        assert_eq!(jb.pull(&mut out), 0);
    }

    #[test]
    fn delivers_real_samples_once_primed() {
        let mut jb = buf();
        jb.push(&vec![0.25; 2000]); // > 960-sample target
        let mut out = [0.0f32; 100];
        assert_eq!(jb.pull(&mut out), 100);
        assert!(out.iter().all(|&s| (s - 0.25).abs() < f32::EPSILON));
    }

    #[test]
    fn underrun_fills_silence_counts_and_reprimes() {
        let mut jb = buf();
        jb.push(&vec![0.5; 1000]); // primes (target 960)
        let mut out = [0.0f32; 1200]; // ask for more than buffered
        let real = jb.pull(&mut out);
        assert_eq!(real, 1000);
        assert!(out[1000..].iter().all(|&s| s == 0.0), "tail is silence");
        assert_eq!(jb.underruns(), 1);
        // Re-primed: a follow-up pull before refilling is silence again.
        let mut out2 = [0.0f32; 64];
        assert_eq!(jb.pull(&mut out2), 0);
    }

    #[test]
    fn overrun_drops_oldest_to_bound_latency() {
        let mut jb = buf(); // max 40 ms = 3840 interleaved samples
                            // Push well past the ceiling in one go.
        jb.push(&vec![0.1; 8000]);
        assert_eq!(jb.overruns(), 1);
        // Trimmed back toward the 10 ms target, never above the ceiling.
        assert!(jb.buffered_samples() <= 3840);
        assert!(jb.buffered_samples() >= 960);
    }
}
