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
    fn has_loss(&self) -> bool {
        self.congestion_events_delta > 0
            || self.lost_packets_delta > 0
            || self.client_fragments_lost > 0
    }
}

/// What the controller wants right now. The host debounces against the
/// last applied decision before reconfiguring the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbrDecision {
    pub target_kbps: u32,
    pub target_fps: u32,
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
    /// FPS the capture loop was built with. Ceiling + starting point
    /// for the FPS gear.
    pub baseline_fps: u32,
    /// Floor for the FPS gear.
    pub floor_fps: u32,
    /// RTT above which we treat the path as congested.
    pub rtt_high: Duration,
    /// RTT below which we treat the path as healthy.
    pub rtt_low: Duration,
    /// How many consecutive healthy samples are required before we
    /// step up. One alone could be transient.
    pub healthy_samples_for_step_up: u32,
    /// Bitrate step size, expressed as the numerator of `step / 100`
    /// (i.e. `10` = 10%).
    pub bitrate_step_pct: u32,
    /// FPS step-up size, expressed the same way. Separate from
    /// `bitrate_step_pct` because the FPS fall is a halving (sharp),
    /// so the recovery climb needs to be sharp too — a 10% step from
    /// FPS=15 only adds 1 FPS per healthy streak, which never makes
    /// it back to 60 inside a reasonable session. 25% gets from 15
    /// to baseline 60 in ~7 cooldown windows.
    pub fps_step_up_pct: u32,
    /// Cooldown after a step in either direction. Prevents thrash if
    /// samples arrive faster than the encoder can settle.
    pub cooldown: Duration,
}

impl AbrConfig {
    #[must_use]
    pub fn new(baseline_kbps: u32, baseline_fps: u32) -> Self {
        Self {
            baseline_kbps,
            // 1.5 Mbps is the absolute minimum for a desktop stream
            // that's still legible; below this, text becomes mush
            // even with a long GOP. Clamp to baseline if the host
            // configured a baseline below this — the floor must never
            // exceed the ceiling.
            floor_kbps: 1_500.min(baseline_kbps),
            baseline_fps,
            floor_fps: 15.min(baseline_fps),
            rtt_high: Duration::from_millis(150),
            rtt_low: Duration::from_millis(60),
            healthy_samples_for_step_up: 3,
            bitrate_step_pct: 10,
            fps_step_up_pct: 25,
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

    /// Halve toward floor in one coarse step. Used by the FPS gear on a
    /// client-side dropped-frame signal: a renderer that can't keep up
    /// at 60 isn't going to recover from a 10% nudge.
    fn halve_toward_floor(&mut self) -> bool {
        let next = (self.current / 2).max(self.floor);
        if next == self.current {
            return false;
        }
        self.current = next;
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
    fps: Gear,
    healthy_run: HealthyRun,
}

impl AbrController {
    #[must_use]
    pub fn new(cfg: AbrConfig) -> Self {
        Self {
            bitrate: Gear::new(cfg.baseline_kbps, cfg.floor_kbps, cfg.baseline_kbps),
            fps: Gear::new(cfg.baseline_fps, cfg.floor_fps, cfg.baseline_fps),
            healthy_run: HealthyRun::default(),
            cfg,
        }
    }

    /// Current decision without observing a new sample. Cheap; safe to
    /// poll.
    #[must_use]
    pub fn current(&self) -> AbrDecision {
        AbrDecision {
            target_kbps: self.bitrate.current,
            target_fps: self.fps.current,
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
        self.fps.tick(dt);
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
        } else if congested && cooled(&self.bitrate) {
            // Sustained high RTT without loss yet — back off gently.
            self.bitrate.step_down_pct(self.cfg.bitrate_step_pct);
        } else if healthy
            && cooled(&self.bitrate)
            && self.healthy_run.consecutive >= self.cfg.healthy_samples_for_step_up
        {
            self.bitrate.step_up_pct(self.cfg.bitrate_step_pct);
        }

        // --- FPS gear ---
        // FPS is the renderer-side knob: it tracks the client's
        // ability to keep up with what we're already sending, not
        // network capacity. A network-side fall doesn't immediately
        // drop FPS — we lower bitrate first and only touch FPS if the
        // client is *also* dropping frames, which means encoding
        // faster wouldn't help.
        if sample.client_frames_dropped > 0 && cooled(&self.fps) {
            self.fps.halve_toward_floor();
        } else if healthy
            && cooled(&self.fps)
            && self.healthy_run.consecutive >= self.cfg.healthy_samples_for_step_up
        {
            self.fps.step_up_pct(self.cfg.fps_step_up_pct);
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
        AbrController::new(AbrConfig::new(8_000, 60))
    }

    #[test]
    fn starts_at_baseline() {
        let c = ctl();
        let d = c.current();
        assert_eq!(d.target_kbps, 8_000);
        assert_eq!(d.target_fps, 60);
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
    fn step_up_clamped_to_baseline_ceiling() {
        let mut c = AbrController::new(AbrConfig::new(2_000, 60));
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
        let cfg = AbrConfig::new(1_000, 60);
        assert!(cfg.floor_kbps <= cfg.baseline_kbps);
    }

    #[test]
    fn frames_dropped_halves_fps() {
        let mut c = ctl();
        let mut s = healthy();
        s.client_frames_dropped = 4;
        let d = c.observe(Duration::from_secs(1), s);
        assert_eq!(d.target_fps, 30);
    }

    #[test]
    fn fps_floor_respected() {
        let mut c = ctl();
        let mut s = healthy();
        s.client_frames_dropped = 4;
        // Repeated halvings should stop at the floor (15) — and stay
        // there if frames_dropped keeps firing. Verify the floor
        // exactly, not just `>= floor`: an unbounded saturating_sub
        // would still satisfy `>= floor`.
        for _ in 0..6 {
            c.observe(Duration::from_secs(4), s);
        }
        assert_eq!(c.current().target_fps, 15);
        // One more tick at the floor should be a no-op, not a wrap.
        c.observe(Duration::from_secs(4), s);
        assert_eq!(c.current().target_fps, 15);
    }

    #[test]
    fn fps_recovers_to_baseline_in_bounded_steps() {
        // Drive FPS to the floor (15) via two halvings, then run a
        // long healthy streak. The recovery must reach the baseline
        // (60) in a bounded number of steps — 25% step up means
        // 15 → 18 → 22 → 27 → 33 → 41 → 51 → 63 (clamped 60), so
        // 7 step-up samples past the cooldown. The bitrate gear
        // (10% steps) was the source of the original asymmetry bug;
        // assert the corrected FPS gear actually climbs.
        let mut c = ctl();
        let mut bad = healthy();
        bad.client_frames_dropped = 4;
        c.observe(Duration::from_secs(4), bad);
        c.observe(Duration::from_secs(4), bad);
        assert_eq!(c.current().target_fps, 15);

        // The healthy_run counter resets on the bad samples above. We
        // need 3 healthy samples before the first step fires, then
        // one per cooldown after that.
        for _ in 0..40 {
            c.observe(Duration::from_secs(4), healthy());
            if c.current().target_fps == 60 {
                return;
            }
        }
        panic!("FPS failed to reach baseline within 40 healthy ticks");
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
    }
}
