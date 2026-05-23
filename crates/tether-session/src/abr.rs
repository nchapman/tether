//! Adaptive bitrate (ABR) controller.
//!
//! The host calls [`AbrController::observe`] once per tick (typically the
//! 1 Hz `ClientStats` cadence) with a fresh [`AbrSample`] combining quinn
//! path stats and the client's `ClientStats`. The controller folds the
//! sample into a rolling 3-second window and returns an
//! [`AbrDecision`] — a target bitrate (kbps) and target FPS — that the
//! caller debounces against its current setting before pushing it into
//! the encoder.
//!
//! Two independent state machines drive the decision:
//!
//! - **Bitrate gear.** Reflects quality vs. congestion. Congestion is
//!   inferred from RTT (high-water 150 ms, low-water 60 ms) plus
//!   quinn-side loss (`congestion_events_delta`, `lost_packets_delta`)
//!   and the client's `fragments_lost`. A burst of loss collapses
//!   immediately to the floor (no hysteresis on a fall — keep latency
//!   playable); recovery is gated by N consecutive healthy samples so
//!   we don't oscillate.
//! - **FPS gear.** Reflects the encoder's ability to keep up. Climbs
//!   when the network is healthy and the client isn't dropping frames;
//!   falls when `frames_dropped` rises. Wired into the encoder is
//!   future work; the controller exposes the value today so the
//!   plumbing seam exists when the capture/encoder learn to take an
//!   FPS hint.
//!
//! The controller is pure: no I/O, no clock reads. Callers supply the
//! tick interval. Tests drive deterministic state transitions by
//! constructing samples directly.

use std::time::Duration;

/// Per-tick observation. Built by the host send loop from the latest
/// quinn `PathStats` snapshot and the most recent `ClientStats`.
///
/// All "delta" fields are differences against the previous observation
/// — the host owns the subtraction so the controller stays pure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbrSample {
    /// Connection RTT as quinn currently estimates it.
    pub rtt: Duration,
    /// New congestion events seen by quinn since the last sample
    /// (`PathStats::congestion_events` delta). One or more is a strong
    /// signal that the CC algorithm just backed off.
    pub congestion_events_delta: u64,
    /// New packets quinn marked lost since the last sample
    /// (`PathStats::lost_packets` delta).
    pub lost_packets_delta: u64,
    /// Fragments the client's defragmenter never assembled, per the
    /// most recent `ClientStats::fragments_lost` window.
    pub client_fragments_lost: u32,
    /// Frames the client dropped (decode lag, render queue full), per
    /// the most recent `ClientStats::frames_dropped` window.
    pub client_frames_dropped: u32,
}

impl AbrSample {
    /// `true` if any congestion or loss signal fired on this sample.
    ///
    /// `client_frames_dropped` counts here too: the client only drops
    /// decoded frames when its render queue can't keep up, which on a
    /// pure-network bottleneck is downstream of the same loss/RTT
    /// pressure quinn is reporting — treating it as a loss signal
    /// resets the healthy streak and gates the bitrate step-up.
    fn has_loss(&self) -> bool {
        self.congestion_events_delta > 0
            || self.lost_packets_delta > 0
            || self.client_fragments_lost > 0
            || self.client_frames_dropped > 0
    }
}

/// What the controller wants right now. The host debounces against the
/// last applied decision before reconfiguring the encoder.
///
/// **FPS not yet exposed.** An FPS gear was scaffolded earlier in the
/// session, but no real capture backend honours runtime FPS
/// renegotiation today (PipeWire format-renegotiation and SCK
/// `minimumFrameInterval` updates are per-backend follow-up work).
/// Re-add `target_fps` here once at least one production backend
/// observes [`crate::CaptureHandle::set_target_fps`] writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbrDecision {
    pub target_kbps: u32,
}

/// Tunables. The defaults reflect what RustDesk's `video_qos`
/// converges on — 150 ms RTT high-water, ~3 s windows, ±10% bitrate
/// steps.
#[derive(Debug, Clone, Copy)]
pub struct AbrConfig {
    /// Bitrate the encoder was built with. Acts as both the ceiling
    /// and the starting point.
    pub baseline_kbps: u32,
    /// Floor — never drop below this even on sustained congestion.
    /// Picking a sane floor prevents the controller from starving the
    /// session into unreadability.
    pub floor_kbps: u32,
    /// RTT above which we treat the path as congested.
    pub rtt_high: Duration,
    /// RTT below which we treat the path as healthy.
    pub rtt_low: Duration,
    /// How many consecutive healthy samples are required before we
    /// step up. One alone could be transient.
    pub healthy_samples_for_step_up: u32,
    /// Steady-state bitrate step-up size, expressed as the
    /// numerator of `step / 100` (i.e. `10` = 10%). Applied once
    /// the fast-climb budget is exhausted.
    pub bitrate_step_pct: u32,
    /// Larger step used for the first `fast_climb_steps` step-ups
    /// after a collapse. Climbing from the 1.5 Mbps floor back to
    /// an 8 Mbps baseline at the steady-state 10% takes 16 cool-
    /// down windows (~48 s); the fast-climb phase covers the
    /// first few windows at 25% to absorb most of the gap in ~9 s,
    /// then narrows to `bitrate_step_pct` to avoid overshooting
    /// the path capacity.
    pub bitrate_fast_step_pct: u32,
    /// Number of step-ups taken at `bitrate_fast_step_pct` after a
    /// collapse before switching to the steady-state percentage.
    /// Any loss / step-down resets the budget.
    pub fast_climb_steps: u32,
    /// Cooldown after a step in either direction. Prevents thrash if
    /// samples arrive faster than the encoder can settle.
    pub cooldown: Duration,
}

impl AbrConfig {
    /// Construct with defaults. `baseline_kbps` is the encoder's
    /// initial bitrate and the climb ceiling. FPS gear was scaffolded
    /// earlier in the session but removed when it became clear no
    /// production capture backend honours runtime FPS retunes today
    /// — re-add a `baseline_fps` parameter when the first backend
    /// does.
    #[must_use]
    pub fn new(baseline_kbps: u32) -> Self {
        Self {
            baseline_kbps,
            // 1.5 Mbps is the absolute minimum for a desktop stream
            // that's still legible; below this, text becomes mush
            // even with a long GOP. Clamp to baseline if the host
            // configured a baseline below this — the floor must never
            // exceed the ceiling.
            floor_kbps: 1_500.min(baseline_kbps),
            rtt_high: Duration::from_millis(150),
            rtt_low: Duration::from_millis(60),
            healthy_samples_for_step_up: 3,
            bitrate_step_pct: 10,
            bitrate_fast_step_pct: 25,
            fast_climb_steps: 4,
            cooldown: Duration::from_secs(3),
        }
    }
}

/// State for one of the two gears. Tracks the current target and a
/// last-change timestamp so the cooldown is enforced symmetrically.
#[derive(Debug, Clone, Copy)]
struct Gear {
    current: u32,
    floor: u32,
    ceiling: u32,
    elapsed_since_change: Duration,
}

impl Gear {
    fn new(start: u32, floor: u32, ceiling: u32) -> Self {
        Self {
            current: start,
            floor,
            ceiling,
            // Start fully past the cooldown so the first warranted
            // change isn't artificially delayed.
            elapsed_since_change: Duration::MAX,
        }
    }

    fn step_down_pct(&mut self, pct: u32) -> bool {
        let next = self.current.saturating_sub(self.current * pct / 100);
        let next = next.max(self.floor);
        if next == self.current {
            return false;
        }
        self.current = next;
        self.elapsed_since_change = Duration::ZERO;
        true
    }

    fn step_up_pct(&mut self, pct: u32) -> bool {
        let next = self.current.saturating_add(self.current * pct / 100);
        let next = next.min(self.ceiling);
        if next == self.current {
            return false;
        }
        self.current = next;
        self.elapsed_since_change = Duration::ZERO;
        true
    }

    fn collapse_to_floor(&mut self) -> bool {
        if self.current == self.floor {
            return false;
        }
        self.current = self.floor;
        self.elapsed_since_change = Duration::ZERO;
        true
    }

    fn tick(&mut self, dt: Duration) {
        self.elapsed_since_change = self.elapsed_since_change.saturating_add(dt);
    }
}

/// Trailing run-length of "healthy" samples. Tracked as a counter
/// rather than a time-bounded window because the cooldown gate
/// already handles the wall-clock side — duplicating it here just
/// fights itself at slow tick rates (e.g., 1 Hz `ClientStats` with a
/// 3 s cooldown would leave at most one sample inside any reasonable
/// span). A single un-healthy sample resets the counter; the gear
/// only steps up after `healthy_samples_for_step_up` in a row.
#[derive(Debug, Default)]
struct HealthyRun {
    consecutive: u32,
}

impl HealthyRun {
    fn observe(&mut self, sample: &AbrSample, rtt_low: Duration) {
        if sample.rtt <= rtt_low && !sample.has_loss() {
            self.consecutive = self.consecutive.saturating_add(1);
        } else {
            self.consecutive = 0;
        }
    }
}

/// The controller.
#[derive(Debug)]
pub struct AbrController {
    cfg: AbrConfig,
    bitrate: Gear,
    healthy_run: HealthyRun,
    /// Remaining fast-climb step-ups after a collapse. Decremented
    /// on each step-up; reset to `cfg.fast_climb_steps` on every
    /// step-down or collapse. Lets the climb take wider steps for
    /// the first few healthy windows after a fall, then narrow to
    /// the steady-state rate.
    fast_climb_budget: u32,
}

impl AbrController {
    #[must_use]
    pub fn new(cfg: AbrConfig) -> Self {
        Self {
            bitrate: Gear::new(cfg.baseline_kbps, cfg.floor_kbps, cfg.baseline_kbps),
            healthy_run: HealthyRun::default(),
            fast_climb_budget: 0,
            cfg,
        }
    }

    /// Current decision without observing a new sample. Cheap; safe to
    /// poll.
    #[must_use]
    pub fn current(&self) -> AbrDecision {
        AbrDecision {
            target_kbps: self.bitrate.current,
        }
    }

    /// Fold one observation into the state and return the (possibly
    /// updated) decision.
    ///
    /// `dt` is the wall-clock interval since the previous call. The
    /// caller owns the clock so the controller stays deterministic and
    /// testable.
    pub fn observe(&mut self, dt: Duration, sample: AbrSample) -> AbrDecision {
        self.bitrate.tick(dt);
        // `observe` resets the streak counter to zero on any loss /
        // high-RTT sample, so the counter is inherently correct for
        // the gate below: a streak can only accumulate starting from
        // the first post-fall healthy sample.
        self.healthy_run.observe(&sample, self.cfg.rtt_low);

        let congested = sample.rtt >= self.cfg.rtt_high;
        let healthy = sample.rtt <= self.cfg.rtt_low && !sample.has_loss();
        let cooled = |g: &Gear| g.elapsed_since_change >= self.cfg.cooldown;

        // --- Bitrate gear ---
        if sample.has_loss() && (congested || sample.congestion_events_delta > 0) {
            // Hard signal — quinn's CC backed off or we're already in
            // congested-RTT territory and now seeing loss. Collapse
            // straight to the floor; trust quinn's CC to refill the
            // window before we step up. No cooldown gate on a fall:
            // latency suffers if we hesitate.
            self.bitrate.collapse_to_floor();
            // Refill the fast-climb budget: the next few healthy
            // windows will take wider steps back up.
            self.fast_climb_budget = self.cfg.fast_climb_steps;
        } else if congested && cooled(&self.bitrate) {
            // Sustained high RTT without loss yet — back off gently.
            self.bitrate.step_down_pct(self.cfg.bitrate_step_pct);
            self.fast_climb_budget = self.cfg.fast_climb_steps;
        } else if healthy
            && cooled(&self.bitrate)
            && self.healthy_run.consecutive >= self.cfg.healthy_samples_for_step_up
        {
            // Asymmetric climb: spend the fast-climb budget first,
            // then narrow to the steady-state percentage. The
            // budget is drained one step per healthy cooldown
            // window, so a transient collapse → 1.5 Mbps reaches
            // ~3.66 Mbps after 4 fast steps (~12 s of healthy
            // windows) before throttling back to 10% steps.
            let pct = if self.fast_climb_budget > 0 {
                self.fast_climb_budget = self.fast_climb_budget.saturating_sub(1);
                self.cfg.bitrate_fast_step_pct
            } else {
                self.cfg.bitrate_step_pct
            };
            self.bitrate.step_up_pct(pct);
        }

        self.current()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy() -> AbrSample {
        AbrSample {
            rtt: Duration::from_millis(20),
            congestion_events_delta: 0,
            lost_packets_delta: 0,
            client_fragments_lost: 0,
            client_frames_dropped: 0,
        }
    }

    fn congested() -> AbrSample {
        AbrSample {
            rtt: Duration::from_millis(200),
            congestion_events_delta: 0,
            lost_packets_delta: 0,
            client_fragments_lost: 0,
            client_frames_dropped: 0,
        }
    }

    fn loss_burst() -> AbrSample {
        AbrSample {
            rtt: Duration::from_millis(200),
            congestion_events_delta: 1,
            lost_packets_delta: 5,
            client_fragments_lost: 2,
            client_frames_dropped: 0,
        }
    }

    fn ctl() -> AbrController {
        AbrController::new(AbrConfig::new(8_000))
    }

    #[test]
    fn starts_at_baseline() {
        let c = ctl();
        let d = c.current();
        assert_eq!(d.target_kbps, 8_000);
    }

    #[test]
    fn loss_burst_collapses_bitrate_to_floor_immediately() {
        let mut c = ctl();
        let d = c.observe(Duration::from_secs(1), loss_burst());
        assert_eq!(d.target_kbps, 1_500, "loss burst should snap to floor");
    }

    #[test]
    fn sustained_high_rtt_steps_bitrate_down_gently() {
        let mut c = ctl();
        // First observation — past the cooldown by construction, so
        // step fires immediately at -10%.
        let d = c.observe(Duration::from_secs(1), congested());
        assert_eq!(d.target_kbps, 7_200);
        // Second observation inside the cooldown window — no further
        // step.
        let d = c.observe(Duration::from_millis(500), congested());
        assert_eq!(d.target_kbps, 7_200, "cooldown should suppress thrash");
    }

    #[test]
    fn healthy_streak_steps_bitrate_back_up() {
        let mut c = ctl();
        // Force a fall first so we have headroom.
        c.observe(Duration::from_secs(1), loss_burst());
        assert_eq!(c.current().target_kbps, 1_500);
        // Now stream healthy samples spaced past the cooldown each.
        // Need at least `healthy_samples_for_step_up` (3) consecutive
        // healthy samples in the window AND the cooldown elapsed
        // since the last change.
        c.observe(Duration::from_secs(4), healthy());
        c.observe(Duration::from_millis(500), healthy());
        c.observe(Duration::from_millis(500), healthy());
        // Window now has 3 healthy samples; cooldown elapsed on the
        // first one. Subsequent calls should step.
        let d = c.observe(Duration::from_secs(4), healthy());
        assert!(
            d.target_kbps > 1_500,
            "expected step up, got {}",
            d.target_kbps
        );
    }

    #[test]
    fn fast_climb_takes_wider_steps_after_collapse() {
        let mut c = ctl();
        c.observe(Duration::from_secs(1), loss_burst());
        assert_eq!(c.current().target_kbps, 1_500);

        // Build streak to 3 — the third healthy observe is the
        // first eligible step (cooldown elapsed + streak met).
        // After that, each healthy(4s) observe drives one step.
        // First 4 step-ups at 25%, then 5th at steady-state 10%.
        c.observe(Duration::from_secs(4), healthy()); // streak=1, no step
        c.observe(Duration::from_millis(500), healthy()); // streak=2, no step
        let s1 = c.observe(Duration::from_millis(500), healthy()).target_kbps; // streak=3 → fires
        let s2 = c.observe(Duration::from_secs(4), healthy()).target_kbps;
        let s3 = c.observe(Duration::from_secs(4), healthy()).target_kbps;
        let s4 = c.observe(Duration::from_secs(4), healthy()).target_kbps;
        let s5 = c.observe(Duration::from_secs(4), healthy()).target_kbps;
        assert_eq!(s1, 1_875, "1st step at 25%: 1500 * 1.25");
        assert_eq!(s2, 2_343, "2nd step at 25%");
        assert_eq!(s3, 2_928, "3rd step at 25%");
        assert_eq!(s4, 3_660, "4th step at 25%: budget exhausted");
        assert_eq!(s5, 4_026, "5th step narrows to steady-state 10%");
    }

    #[test]
    fn fast_climb_budget_reset_on_subsequent_collapse() {
        let mut c = ctl();
        c.observe(Duration::from_secs(1), loss_burst());
        // Burn one fast-step.
        c.observe(Duration::from_secs(4), healthy());
        c.observe(Duration::from_millis(500), healthy());
        let after_first_fast = c
            .observe(Duration::from_millis(500), healthy())
            .target_kbps;
        assert!(after_first_fast > 1_500);
        // Second collapse mid-climb.
        c.observe(Duration::from_secs(4), loss_burst());
        assert_eq!(c.current().target_kbps, 1_500);
        // Fast-climb budget should be refilled; next eligible step
        // fires at 25%.
        c.observe(Duration::from_secs(4), healthy());
        c.observe(Duration::from_millis(500), healthy());
        let s1 = c
            .observe(Duration::from_millis(500), healthy())
            .target_kbps;
        assert_eq!(s1, 1_875, "post-second-collapse step should be fast (25%)");
    }

    #[test]
    fn step_up_clamped_to_baseline_ceiling() {
        let mut c = AbrController::new(AbrConfig::new(2_000));
        // Drive a fall, then a sustained healthy stretch. The ceiling
        // is 2000; we should not exceed it no matter how long we run.
        c.observe(Duration::from_secs(1), loss_burst());
        for _ in 0..50 {
            c.observe(Duration::from_secs(4), healthy());
        }
        assert_eq!(c.current().target_kbps, 2_000);
    }

    #[test]
    fn floor_never_exceeds_baseline() {
        // If baseline is below the configured floor default, the
        // floor must clamp to baseline so floor <= ceiling.
        let cfg = AbrConfig::new(1_000);
        assert!(cfg.floor_kbps <= cfg.baseline_kbps);
    }

    #[test]
    fn single_high_rtt_blip_does_not_step_during_cooldown() {
        let mut c = ctl();
        c.observe(Duration::from_secs(1), congested());
        let kbps_after_first = c.current().target_kbps;
        // A healthy sample inside the cooldown shouldn't step back up
        // either — the cooldown is symmetric.
        c.observe(Duration::from_millis(100), healthy());
        assert_eq!(c.current().target_kbps, kbps_after_first);
    }

    #[test]
    fn has_loss_detects_each_signal_independently() {
        let mut s = healthy();
        assert!(!s.has_loss());
        s.congestion_events_delta = 1;
        assert!(s.has_loss());
        s = healthy();
        s.lost_packets_delta = 1;
        assert!(s.has_loss());
        s = healthy();
        s.client_fragments_lost = 1;
        assert!(s.has_loss());
        s = healthy();
        s.client_frames_dropped = 1;
        assert!(s.has_loss());
    }

    #[test]
    fn client_frames_dropped_resets_healthy_streak() {
        let mut c = ctl();
        // Force a fall to leave headroom for a step-up.
        c.observe(Duration::from_secs(1), loss_burst());
        assert_eq!(c.current().target_kbps, 1_500);
        // Two healthy samples build the streak.
        c.observe(Duration::from_secs(4), healthy());
        c.observe(Duration::from_millis(500), healthy());
        // A frames-dropped blip resets the streak; next would-be
        // step-up should NOT fire even though the cooldown has
        // elapsed and rtt is fine.
        let mut blip = healthy();
        blip.client_frames_dropped = 3;
        let after_blip = c.observe(Duration::from_millis(500), blip).target_kbps;
        assert_eq!(
            after_blip, 1_500,
            "client_frames_dropped should suppress the step-up"
        );
    }
}
