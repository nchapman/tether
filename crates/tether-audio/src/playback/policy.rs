//! Playback buffering policy — the brain on the consumer side of the
//! [`ring`](super::ring), run inside the cpal output callback.
//!
//! It implements the latency-cap model that Sunshine/Moonlight converged on
//! (free-running device clock, no resampler, no A/V timestamp steering) as a
//! pure decision function, separated from the lock-free ring I/O so it can be
//! unit-tested deterministically:
//!
//! - **Prime** — play silence until a `target` cushion is buffered, so playback
//!   starts with headroom against jitter (equivalent to Moonlight's startup
//!   drain, minus the dropped packets).
//! - **Latency cap** — when the buffer exceeds `max` (consumer fell behind, or
//!   the producer's clock runs fast), drop the oldest samples back down to
//!   `target`. This is also how ~100 ppm clock drift is absorbed: a bounded
//!   ~10 ms drop every minute or two, rather than an in-app resampler.
//! - **Underrun** — a short read is filled with silence. A *partial* read is a
//!   transient dip and keeps playing (no re-prime — the bug that turned mild
//!   starvation into a per-callback stutter); a *full* starvation is treated as
//!   a cold start and re-primes once to rebuild the cushion.
//!
//! The deferred adaptive resampler (issue #35) would slot in here, slewing the
//! consume rate to hold the buffer at `target` instead of periodically dropping.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Cumulative playback counters, shared with the [`AudioSink`](super::AudioSink)
/// for periodic logging. Written by the callback-side policy, read elsewhere.
#[derive(Debug, Default)]
pub struct PlaybackStats {
    /// Output callbacks that had to insert any silence (producer fell behind).
    pub underruns: AtomicU64,
    /// Samples dropped (oldest-first) to hold latency under the ceiling.
    pub dropped_samples: AtomicU64,
}

/// What a single output callback should do with the ring: discard `skip` oldest
/// samples, then copy `copy` samples to the device; the caller zero-fills the
/// rest of the output buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FillPlan {
    pub skip: usize,
    pub copy: usize,
}

/// Consumer-side buffering policy. All sizes are interleaved-sample counts.
pub struct PlaybackPolicy {
    /// Cushion (samples) built before playback starts or after full starvation.
    target: usize,
    /// Latency ceiling (samples); beyond it, drop oldest back to `target`.
    max: usize,
    /// While priming, the callback emits silence until `target` is buffered.
    priming: bool,
    stats: Arc<PlaybackStats>,
}

impl PlaybackPolicy {
    /// `target_ms` is the cushion built before audio flows; `max_ms` the latency
    /// ceiling past which the oldest audio is dropped.
    pub fn new(
        channels: u8,
        sample_rate: u32,
        target_ms: u32,
        max_ms: u32,
        stats: Arc<PlaybackStats>,
    ) -> Self {
        let per_ms = (sample_rate as usize * usize::from(channels.max(1))) / 1000;
        let target = per_ms * target_ms as usize;
        let max = (per_ms * max_ms as usize).max(target);
        Self {
            target,
            max,
            priming: true,
            stats,
        }
    }

    /// Decide what to do given `buffered` samples in the ring and an output
    /// buffer of `out_len` samples. `buffered` is only read once per callback;
    /// because the producer only ever *adds* samples, the planned `copy` is
    /// always still available when the caller performs it.
    pub fn plan(&mut self, buffered: usize, out_len: usize) -> FillPlan {
        // Prime gate: silence until the cushion is built.
        if self.priming {
            if buffered < self.target {
                return FillPlan { skip: 0, copy: 0 };
            }
            self.priming = false;
        }

        // Latency cap: drop oldest back down to `target` when over the ceiling.
        let mut available = buffered;
        let mut skip = 0;
        if available > self.max {
            skip = available - self.target;
            available -= skip;
            self.stats
                .dropped_samples
                .fetch_add(u64::try_from(skip).unwrap_or(u64::MAX), Ordering::Relaxed);
        }

        let copy = out_len.min(available);
        if copy < out_len {
            // Short read → the caller fills the rest with silence.
            self.stats.underruns.fetch_add(1, Ordering::Relaxed);
            // Full starvation is a cold start: rebuild the cushion. A partial
            // read is a transient dip and keeps playing.
            if copy == 0 {
                self.priming = true;
            }
        }
        FillPlan { skip, copy }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 48 kHz stereo: 96 interleaved samples per ms. target 10 ms = 960, max 40
    // ms = 3840.
    fn policy() -> (PlaybackPolicy, Arc<PlaybackStats>) {
        let stats = Arc::new(PlaybackStats::default());
        (
            PlaybackPolicy::new(2, 48_000, 10, 40, Arc::clone(&stats)),
            stats,
        )
    }

    #[test]
    fn primes_with_silence_until_target_buffered() {
        let (mut p, _s) = policy();
        // Below the 960-sample target → silence (copy 0), still priming.
        assert_eq!(p.plan(0, 64), FillPlan { skip: 0, copy: 0 });
        assert_eq!(p.plan(480, 64), FillPlan { skip: 0, copy: 0 });
        // Reaching target ends priming and delivers real samples.
        assert_eq!(p.plan(960, 64), FillPlan { skip: 0, copy: 64 });
    }

    #[test]
    fn delivers_real_samples_once_primed() {
        let (mut p, _s) = policy();
        assert_eq!(p.plan(2000, 100), FillPlan { skip: 0, copy: 100 });
        assert_eq!(p.plan(2000, 100), FillPlan { skip: 0, copy: 100 });
    }

    #[test]
    fn partial_underrun_keeps_playing_without_reprime() {
        let (mut p, stats) = policy();
        p.plan(2000, 100); // prime

        // Fewer buffered than requested, but not empty: deliver what we have,
        // count an underrun, and DO NOT re-prime (the #3 stutter bug).
        assert_eq!(p.plan(50, 100), FillPlan { skip: 0, copy: 50 });
        assert_eq!(stats.underruns.load(Ordering::Relaxed), 1);
        // Next callback with samples available plays immediately — not silence.
        assert_eq!(p.plan(200, 100), FillPlan { skip: 0, copy: 100 });
    }

    #[test]
    fn full_starvation_reprimes_to_rebuild_cushion() {
        let (mut p, stats) = policy();
        p.plan(2000, 100); // prime

        // Empty ring → all silence, count underrun, re-enter priming.
        assert_eq!(p.plan(0, 100), FillPlan { skip: 0, copy: 0 });
        assert_eq!(stats.underruns.load(Ordering::Relaxed), 1);
        // Re-priming: a trickle below target is still silence until the cushion
        // rebuilds.
        assert_eq!(p.plan(480, 100), FillPlan { skip: 0, copy: 0 });
        assert_eq!(p.plan(960, 100), FillPlan { skip: 0, copy: 100 });
    }

    #[test]
    fn over_ceiling_drops_oldest_back_to_target() {
        let (mut p, stats) = policy();
        p.plan(2000, 100); // prime

        // 8000 buffered, ceiling 3840 → drop down to the 960 target, then copy.
        let plan = p.plan(8000, 100);
        assert_eq!(plan.skip, 8000 - 960);
        assert_eq!(plan.copy, 100);
        assert_eq!(
            stats.dropped_samples.load(Ordering::Relaxed),
            u64::try_from(8000 - 960).unwrap()
        );
    }
}
