//! Frame-age-based drop policy + present-mode preference.
//!
//! The renderer's `RedrawRequested` handler pulls whatever sits in
//! [`crate::LatestFrame`] every vsync. Without an age check, a late-
//! arriving frame (network hiccup, decoder stall) still consumes a
//! refresh slot — wasting it on stale content the user has already
//! visually adjusted to. This module packages the policy:
//!
//! 1. **Frame-age check.** If the frame's host-clock-adjusted capture
//!    timestamp shows it is more than 1.5× the display refresh period
//!    old AND we've seen at least 2 late frames in a row, drop it on
//!    the floor with a per-skip cooldown.
//! 2. **Present-mode preference.** Two profiles — [`PresentPolicy::LowLatency`]
//!    keeps the existing `Immediate > Mailbox > Fifo` ordering;
//!    [`PresentPolicy::Smooth`] flips to `Fifo > Mailbox > Immediate`
//!    for the smoothness-biased case. The drop policy works under all
//!    three modes — Fifo benefits most because every render commits to
//!    a refresh slot.
//!
//! The policy itself is pure (no clock reads, no GPU calls) so it's
//! unit-testable without spinning up winit or wgpu. The renderer
//! threads `now()` and the cached refresh rate in as parameters.

use std::time::Duration;

/// Refresh rate cap for the late-frame threshold. Falls back to this
/// when winit can't report a real value (headless harness, monitor
/// hot-unplug between presents).
pub const REFRESH_RATE_FALLBACK_HZ: u32 = 60;

/// Multiplier on the refresh period used as the "frame is late"
/// threshold. 1.5 matches Moonlight's `800_000_000 / refresh` (~0.8 ×
/// refresh) in spirit: enough slack to absorb normal scheduler jitter,
/// tight enough to actually skip a stale frame within one vsync.
pub const FRAME_AGE_THRESHOLD_FACTOR: u32 = 3; // expressed as N/2; see below

/// Consecutive late frames required before we start skipping. One
/// late frame alone could be a stutter; two in a row signals the
/// stream is actually behind.
pub const LATE_STREAK_THRESHOLD: u32 = 2;

/// After a skip, allow this many redraws to actually present before
/// the next skip can fire. Prevents drop-oscillation when the stream
/// is steadily ~1.5× behind: skipping every other frame is worse than
/// presenting every frame at a stable latency.
pub const SKIP_COOLDOWN_FRAMES: u32 = 4;

/// Present-mode preference profiles.
///
/// `LowLatency` is the current default (since the foundation review).
/// `Smooth` is the new alternative for users who care more about
/// vsync alignment than head-of-line latency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PresentPolicy {
    /// Prefer `Immediate > Mailbox > Fifo`. Today's default.
    #[default]
    LowLatency,
    /// Prefer `Fifo > Mailbox > Immediate`. Every render commits to
    /// a refresh slot; ideal when the network jitter is low enough
    /// that the lowest-latency mode shows tearing as motion.
    Smooth,
}

impl PresentPolicy {
    /// Ordered list of preferred wgpu present modes. The caller
    /// picks the first one the surface advertises.
    #[must_use]
    pub fn preference(self) -> [wgpu::PresentMode; 3] {
        match self {
            Self::LowLatency => [
                wgpu::PresentMode::Immediate,
                wgpu::PresentMode::Mailbox,
                wgpu::PresentMode::Fifo,
            ],
            Self::Smooth => [
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::Mailbox,
                wgpu::PresentMode::Immediate,
            ],
        }
    }
}

/// Per-instance late-frame tracking. The renderer holds one of these
/// next to its `present_stats`. Counters reset on every successful
/// (not-skipped) present; skips advance the cooldown.
#[derive(Default, Debug, Clone, Copy)]
pub struct FrameAgeTracker {
    /// Consecutive frames seen above the late threshold since the
    /// last on-time present. Skip fires when this reaches
    /// [`LATE_STREAK_THRESHOLD`].
    pub late_streak: u32,
    /// Number of presents remaining in the post-skip cooldown.
    /// Decremented on every present (skipped or not); skipping is
    /// gated on this being zero.
    pub cooldown_remaining: u32,
    /// Cumulative skip count for the current log window.
    pub drops_in_window: u32,
}

/// Decision the renderer should take for the current redraw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentDecision {
    /// Apply the frame and present.
    Render,
    /// Skip the frame; don't apply, don't present. The renderer's
    /// next redraw cycle will see a (hopefully fresher) frame.
    Skip,
}

/// Decide whether to render `frame_age_ns` at `refresh_rate_mhz`,
/// given the tracker's prior state. Pure: no clock reads, no I/O.
/// Mutates `tracker` to reflect the post-decision state (late_streak
/// + cooldown).
#[must_use]
pub fn decide_present(
    frame_age_ns: u64,
    refresh_rate_mhz: u32,
    tracker: &mut FrameAgeTracker,
) -> PresentDecision {
    let refresh_period_ns = refresh_period_ns(refresh_rate_mhz);
    // `FRAME_AGE_THRESHOLD_FACTOR / 2` keeps the multiplier
    // expressible as an integer fraction (3/2 = 1.5×). Integer math
    // throughout avoids any float-rounding surprises at the
    // threshold boundary.
    let threshold_ns = refresh_period_ns
        .saturating_mul(u64::from(FRAME_AGE_THRESHOLD_FACTOR))
        / 2;
    let is_late = frame_age_ns > threshold_ns;

    // Always tick down the cooldown so a long stretch of on-time
    // frames eventually re-enables skipping.
    if tracker.cooldown_remaining > 0 {
        tracker.cooldown_remaining -= 1;
    }

    if is_late {
        tracker.late_streak = tracker.late_streak.saturating_add(1);
    } else {
        tracker.late_streak = 0;
        return PresentDecision::Render;
    }

    // Skip only when the streak has cleared the threshold AND we're
    // not in the post-skip cooldown.
    if tracker.late_streak >= LATE_STREAK_THRESHOLD && tracker.cooldown_remaining == 0 {
        tracker.cooldown_remaining = SKIP_COOLDOWN_FRAMES;
        tracker.drops_in_window = tracker.drops_in_window.saturating_add(1);
        // Reset the streak: a single skip is enough; next late
        // frame restarts the count.
        tracker.late_streak = 0;
        PresentDecision::Skip
    } else {
        PresentDecision::Render
    }
}

/// Refresh period in nanoseconds for a millihertz rate. Falls back
/// to [`REFRESH_RATE_FALLBACK_HZ`] when the input is zero.
#[must_use]
pub fn refresh_period_ns(refresh_rate_mhz: u32) -> u64 {
    let rate_mhz = if refresh_rate_mhz == 0 {
        REFRESH_RATE_FALLBACK_HZ.saturating_mul(1000)
    } else {
        refresh_rate_mhz
    };
    // mHz → period: 1e12 ns / mHz.
    1_000_000_000_000u64 / u64::from(rate_mhz)
}

/// Simple EWMA accumulator. α = 0.25 (one of the standard "slow but
/// responsive" choices). Used by the renderer for inter-arrival,
/// decode-to-present, and jitter samples — three instances of this
/// struct, one shape.
#[derive(Default, Debug, Clone, Copy)]
pub struct EwmaNs {
    mean_ns: u64,
    samples: u32,
}

impl EwmaNs {
    pub fn record(&mut self, sample_ns: u64) {
        if self.samples == 0 {
            self.mean_ns = sample_ns;
        } else {
            // 0.25 × sample + 0.75 × mean, integer-only:
            //   (sample + 3 × mean) / 4
            let s = u128::from(sample_ns);
            let m = u128::from(self.mean_ns);
            self.mean_ns = u64::try_from((s + 3 * m) / 4).unwrap_or(u64::MAX);
        }
        self.samples = self.samples.saturating_add(1);
    }

    #[must_use]
    pub fn mean_ns(&self) -> u64 {
        self.mean_ns
    }

    #[must_use]
    pub fn samples(&self) -> u32 {
        self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_at(age_ns: u64, t: &mut FrameAgeTracker) -> PresentDecision {
        decide_present(age_ns, 60_000, t) // 60 Hz
    }

    #[test]
    fn on_time_frames_always_render() {
        let mut t = FrameAgeTracker::default();
        // 60 Hz → 16.67 ms period. 5 ms age is well within threshold.
        for _ in 0..100 {
            assert_eq!(render_at(5_000_000, &mut t), PresentDecision::Render);
        }
        assert_eq!(t.drops_in_window, 0);
        assert_eq!(t.late_streak, 0);
    }

    #[test]
    fn one_late_frame_does_not_skip_yet() {
        let mut t = FrameAgeTracker::default();
        // Threshold = 1.5 × 16.67 ≈ 25 ms. 30 ms is late.
        assert_eq!(render_at(30_000_000, &mut t), PresentDecision::Render);
        assert_eq!(t.late_streak, 1);
        assert_eq!(t.drops_in_window, 0);
    }

    #[test]
    fn two_consecutive_late_frames_skip_the_second() {
        let mut t = FrameAgeTracker::default();
        assert_eq!(render_at(30_000_000, &mut t), PresentDecision::Render);
        assert_eq!(render_at(35_000_000, &mut t), PresentDecision::Skip);
        assert_eq!(t.drops_in_window, 1);
        assert_eq!(t.late_streak, 0, "streak resets after skip");
        assert!(t.cooldown_remaining > 0);
    }

    #[test]
    fn cooldown_blocks_back_to_back_skips() {
        let mut t = FrameAgeTracker::default();
        // Force a skip.
        render_at(30_000_000, &mut t);
        assert_eq!(render_at(30_000_000, &mut t), PresentDecision::Skip);
        // Next late frame must not skip — cooldown active.
        assert_eq!(render_at(40_000_000, &mut t), PresentDecision::Render);
        assert_eq!(t.drops_in_window, 1);
    }

    #[test]
    fn cooldown_eventually_expires() {
        let mut t = FrameAgeTracker::default();
        // Initial skip.
        render_at(30_000_000, &mut t);
        render_at(30_000_000, &mut t);
        // Burn through the cooldown with on-time frames.
        for _ in 0..SKIP_COOLDOWN_FRAMES {
            render_at(5_000_000, &mut t);
        }
        assert_eq!(t.cooldown_remaining, 0);
        // Now two consecutive late frames should skip again.
        render_at(30_000_000, &mut t);
        assert_eq!(render_at(30_000_000, &mut t), PresentDecision::Skip);
        assert_eq!(t.drops_in_window, 2);
    }

    #[test]
    fn refresh_period_zero_falls_back_to_60hz() {
        assert_eq!(refresh_period_ns(0), 1_000_000_000_000 / 60_000);
    }

    #[test]
    fn ewma_initial_sample_sets_mean() {
        let mut e = EwmaNs::default();
        e.record(1000);
        assert_eq!(e.mean_ns(), 1000);
    }

    #[test]
    fn ewma_converges_toward_steady_input() {
        let mut e = EwmaNs::default();
        for _ in 0..50 {
            e.record(10_000);
        }
        // After many samples the mean should track input.
        assert!(e.mean_ns().abs_diff(10_000) < 100);
    }

    #[test]
    fn present_policy_low_latency_orders_immediate_first() {
        let p = PresentPolicy::LowLatency.preference();
        assert_eq!(p[0], wgpu::PresentMode::Immediate);
    }

    #[test]
    fn present_policy_smooth_orders_fifo_first() {
        let p = PresentPolicy::Smooth.preference();
        assert_eq!(p[0], wgpu::PresentMode::Fifo);
    }
}

/// Duration form of [`refresh_period_ns`] for callers that prefer
/// the `Duration` type.
#[must_use]
pub fn refresh_period(refresh_rate_mhz: u32) -> Duration {
    Duration::from_nanos(refresh_period_ns(refresh_rate_mhz))
}
