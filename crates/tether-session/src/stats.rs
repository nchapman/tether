//! Rolling encode-stats window for the host's periodic log line.
//!
//! Accumulates per-frame deltas and emits a structured `tracing::info!`
//! every ~2 s (configurable) with average encode latency, output
//! kbps, and keyframe count. Resets after each emit.

use std::time::{Duration, Instant};

/// One emit window of encoder stats.
///
/// All accumulators saturate on overflow — a session that produces
/// 18.4 EB of encoded bytes can swallow one truncated log line.
#[derive(Debug)]
pub struct EncodeStatsWindow {
    frame_count: u64,
    encode_latency_sum_ns: u64,
    encoded_bytes_sum: u64,
    keyframe_count: u32,
    started: Instant,
    interval: Duration,
}

impl EncodeStatsWindow {
    /// Build a fresh window that emits every `interval`. A 2-second
    /// interval matches what the host has used historically — short
    /// enough that resolution changes / encoder restarts show up
    /// promptly, long enough that the periodic log line doesn't
    /// dominate a casual `tail -f host.log`.
    #[must_use]
    pub fn new(interval: Duration) -> Self {
        Self {
            frame_count: 0,
            encode_latency_sum_ns: 0,
            encoded_bytes_sum: 0,
            keyframe_count: 0,
            started: Instant::now(),
            interval,
        }
    }

    /// Record one encoded-frame outcome. `encode_delta_ns` is from
    /// [`tether_protocol::video::HostFrameTimingBuilder::encode_delta_ns`]
    /// (this crate doesn't depend on tether-protocol to keep the
    /// dependency cone tight).
    pub fn record_frame(&mut self, encode_delta_ns: u64, encoded_bytes: u64, keyframe: bool) {
        self.frame_count = self.frame_count.saturating_add(1);
        self.encode_latency_sum_ns = self.encode_latency_sum_ns.saturating_add(encode_delta_ns);
        self.encoded_bytes_sum = self.encoded_bytes_sum.saturating_add(encoded_bytes);
        if keyframe {
            self.keyframe_count = self.keyframe_count.saturating_add(1);
        }
    }

    /// True if `interval` has elapsed since the window started or last
    /// reset.
    #[must_use]
    pub fn should_emit(&self) -> bool {
        self.started.elapsed() >= self.interval
    }

    /// Compute the snapshot to log, then clear the accumulators and
    /// restart the window. Returns `None` if no frames were recorded
    /// (idle desktop, encoder stalled — emitting "0 fps over 2 s" is
    /// noise, not signal).
    pub fn snapshot_and_reset(&mut self) -> Option<EncodeStats> {
        if self.frame_count == 0 {
            // Still reset the timer so the next window measures a clean
            // 2 s of activity, not 2+N s including the idle gap.
            self.started = Instant::now();
            return None;
        }
        let window_secs = self.started.elapsed().as_secs_f64();
        // Frame counts well under 2^53; the precision-loss lint here
        // would fire on a workload that's already misbehaving.
        #[allow(clippy::cast_precision_loss)]
        let avg_encode_ms =
            (self.encode_latency_sum_ns as f64 / self.frame_count as f64) / 1_000_000.0;
        #[allow(clippy::cast_precision_loss)]
        let kbps_out = if window_secs > 0.0 {
            (self.encoded_bytes_sum as f64 * 8.0 / 1000.0) / window_secs
        } else {
            0.0
        };
        let snapshot = EncodeStats {
            frame_count: self.frame_count,
            avg_encode_ms,
            kbps_out,
            keyframe_count: self.keyframe_count,
            window_secs,
        };
        self.frame_count = 0;
        self.encode_latency_sum_ns = 0;
        self.encoded_bytes_sum = 0;
        self.keyframe_count = 0;
        self.started = Instant::now();
        Some(snapshot)
    }
}

/// Snapshot returned by [`EncodeStatsWindow::snapshot_and_reset`].
#[derive(Clone, Copy, Debug)]
pub struct EncodeStats {
    pub frame_count: u64,
    pub avg_encode_ms: f64,
    pub kbps_out: f64,
    pub keyframe_count: u32,
    pub window_secs: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_after_interval() {
        let mut w = EncodeStatsWindow::new(Duration::from_millis(1));
        w.record_frame(1_000_000, 1024, true);
        std::thread::sleep(Duration::from_millis(5));
        assert!(w.should_emit());
        let snap = w.snapshot_and_reset().expect("had a frame");
        assert_eq!(snap.frame_count, 1);
        assert_eq!(snap.keyframe_count, 1);
        assert!((snap.avg_encode_ms - 1.0).abs() < 0.01);
    }

    #[test]
    fn idle_window_returns_none_but_resets_timer() {
        let mut w = EncodeStatsWindow::new(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        let before = std::time::Instant::now();
        assert!(w.snapshot_and_reset().is_none());
        // Timer reset: should_emit() must be false immediately after.
        assert!(!w.should_emit() || before.elapsed() >= Duration::from_millis(1));
    }

    #[test]
    fn counts_accumulate() {
        let mut w = EncodeStatsWindow::new(Duration::from_secs(60));
        w.record_frame(1_000_000, 100, false);
        w.record_frame(3_000_000, 200, true);
        w.record_frame(2_000_000, 300, false);
        let snap = w.snapshot_and_reset().expect("frames");
        assert_eq!(snap.frame_count, 3);
        assert_eq!(snap.keyframe_count, 1);
        assert!((snap.avg_encode_ms - 2.0).abs() < 0.01);
    }
}
