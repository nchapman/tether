//! Per-frame packet pacing.
//!
//! The host's send loop today bursts every P-frame datagram into
//! `quinn::Connection::send_datagram` in a tight `for` loop — a
//! 25 Mbps 1080p60 stream emits 30–40 datagrams in well under a
//! millisecond and then idles for the rest of the frame interval.
//! Small-buffer LAN switches and consumer-grade NICs respond to that
//! shape with clustered drops, which is exactly the failure mode any
//! future FEC scheme is least equipped to repair (every shard in a
//! group lost together).
//!
//! [`PacedSender`] spreads fragments evenly across the frame
//! interval, sized by the current ABR-target bitrate. The integration
//! is one line at the call site:
//!
//! ```ignore
//! pacer.begin_frame(Instant::now());
//! for packet in fragmenter.fragment(meta, body) {
//!     let bytes = bincode::serialized_size(&packet).unwrap() as usize;
//!     pacer.wait_for_slot(bytes);
//!     conn.send_datagram(&Datagram::Video(packet))?;
//! }
//! ```
//!
//! ## Why sync sleep
//!
//! The send loop is its own dedicated thread (capture → encode →
//! send). Blocking it with `std::thread::sleep` for the brief
//! per-packet gap is the correct cost — switching to a tokio task
//! would mean either futures-aware QUIC sends (the trait is
//! deliberately sync; see `tether_transport::channel`) or shoving
//! every packet through a channel to a tokio task, both of which add
//! latency for no benefit.
//!
//! ## The math
//!
//! For each packet, the "due" instant is
//! `frame_start + bytes_so_far_including_this_packet * 8 / target_bps`.
//! If the wall clock is already past the due time (encoder ran long,
//! we're behind), the pacer does not sleep — we never *delay* a frame
//! to slow down. Pacing only smooths a burst that arrived ahead of
//! schedule.
//!
//! ## Bench escape hatch
//!
//! `target_bitrate_bps = 0` disables pacing entirely. The benchmark
//! cells run with the pacer disabled so per-packet timing is not
//! contaminated by `thread::sleep` calls.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Shared, atomically-updated target bitrate in bits/second. The ABR
/// tick writes this on every bitrate change; the send loop reads it
/// on every packet. `0` disables pacing.
pub type TargetBitrateBps = Arc<AtomicU64>;

/// Per-frame packet pacer. One instance lives in the send loop; the
/// ABR controller mutates `target_bitrate_bps` through a shared
/// `Arc<AtomicU64>` clone.
pub struct PacedSender {
    target_bitrate_bps: TargetBitrateBps,
    frame_start: Option<Instant>,
    bytes_sent_this_frame: u64,
}

impl PacedSender {
    /// Construct a pacer reading from the given shared bitrate slot.
    /// The caller keeps another clone of the `Arc` and updates it
    /// when the ABR controller's decision changes.
    #[must_use]
    pub fn new(target_bitrate_bps: TargetBitrateBps) -> Self {
        Self {
            target_bitrate_bps,
            frame_start: None,
            bytes_sent_this_frame: 0,
        }
    }

    /// Reset per-frame state. Call exactly once before the first
    /// `wait_for_slot` of a new frame.
    ///
    /// `now` is taken explicitly (rather than calling `Instant::now()`
    /// internally) so tests can drive the pacer deterministically.
    ///
    /// **Tail-frame policy:** if the previous frame finished late
    /// (encoder ran long or the pacer was still sleeping when the
    /// next encode completed), this reset *discards* the leftover
    /// schedule. We do not push frame N's tail into frame N+1's
    /// budget — under sustained encode pressure that would grow
    /// monotonic latency. Better to take one frame's worth of
    /// burst at the start of the new frame than to leak the
    /// previous frame's pacing tail forward.
    pub fn begin_frame(&mut self, now: Instant) {
        self.frame_start = Some(now);
        self.bytes_sent_this_frame = 0;
    }

    /// Returns the `Instant` at which `packet_bytes` may be released
    /// to the wire, given the current frame's accounting. `None`
    /// means pacing is disabled (target bitrate is 0) or
    /// `begin_frame` was never called — caller sends immediately.
    ///
    /// Pure: no clock reads, no sleeps. Production wraps this with
    /// [`Self::wait_for_slot`] which compares against `Instant::now()`
    /// and sleeps if needed; tests inspect the returned `Instant`
    /// directly.
    #[must_use]
    pub fn due_time(&self, packet_bytes: usize) -> Option<Instant> {
        let bps = self.target_bitrate_bps.load(Ordering::Relaxed);
        if bps == 0 {
            return None;
        }
        let start = self.frame_start?;
        let bytes_through_this_packet = self
            .bytes_sent_this_frame
            .saturating_add(packet_bytes as u64);
        // Spread evenly: this packet's last byte should be on the
        // wire at `bytes_through_this_packet * 8 / bps` seconds after
        // frame_start. u128 intermediate avoids overflow at the
        // extremes (240 Gbps test inputs, or huge frames).
        let bits = u128::from(bytes_through_this_packet) * 8;
        let nanos = bits
            .saturating_mul(1_000_000_000)
            .checked_div(u128::from(bps))
            .unwrap_or(0);
        let nanos_u64 = u64::try_from(nanos).unwrap_or(u64::MAX);
        Some(start + Duration::from_nanos(nanos_u64))
    }

    /// Block (via `std::thread::sleep`) until this packet is due to
    /// leave the wire, then bookkeep the byte count.
    ///
    /// The caller MUST `send_datagram` *immediately* after this
    /// returns — any work between `wait_for_slot` and the actual send
    /// pushes the next packet's schedule back by that much.
    pub fn wait_for_slot(&mut self, packet_bytes: usize) {
        if let Some(due) = self.due_time(packet_bytes) {
            let now = Instant::now();
            if due > now {
                std::thread::sleep(due - now);
            }
        }
        self.bytes_sent_this_frame = self
            .bytes_sent_this_frame
            .saturating_add(packet_bytes as u64);
    }

    /// Update the target bitrate. Cheap; safe to call from the ABR
    /// tick thread. Pass `0` to disable pacing (bench escape hatch).
    pub fn set_target_bitrate_kbps(&self, kbps: u32) {
        self.target_bitrate_bps
            .store(u64::from(kbps).saturating_mul(1000), Ordering::Relaxed);
    }

    /// Current target bitrate as it sits in the shared slot. Mostly
    /// for diagnostic logs.
    #[must_use]
    pub fn target_bitrate_bps(&self) -> u64 {
        self.target_bitrate_bps.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pacer_at(kbps: u32) -> PacedSender {
        let slot = Arc::new(AtomicU64::new(u64::from(kbps) * 1000));
        PacedSender::new(slot)
    }

    #[test]
    fn disabled_when_bitrate_zero() {
        let mut p = pacer_at(0);
        p.begin_frame(Instant::now());
        assert_eq!(p.due_time(1200), None);
    }

    #[test]
    fn no_due_time_without_begin_frame() {
        let p = pacer_at(25_000);
        assert_eq!(p.due_time(1200), None);
    }

    #[test]
    fn due_time_spreads_packets_across_frame_window() {
        // 25 Mbps, 30 packets × 1200 bytes = 36 000 bytes total.
        // Drain time = 36 000 × 8 / 25 Mbps = 11.52 ms. Span between
        // the 1st and 30th packet's due-time is 29 × 1200 × 8 / 25 Mbps
        // = 11.14 ms — about 67% of a 16.67 ms frame, clearly paced
        // (not a sub-ms burst) and not overflowing the frame.
        let mut p = pacer_at(25_000);
        let t0 = Instant::now();
        p.begin_frame(t0);

        let packet_bytes = 1200usize;
        let mut first_due = None;
        let mut last_due = None;
        let mut due_in_first_ms = 0u32;
        for _ in 0..30 {
            let due = p.due_time(packet_bytes).unwrap();
            if first_due.is_none() {
                first_due = Some(due);
            }
            last_due = Some(due);
            if due - t0 <= Duration::from_millis(1) {
                due_in_first_ms += 1;
            }
            p.wait_for_slot_test(packet_bytes); // bookkeep without sleeping
        }
        let span = last_due.unwrap() - first_due.unwrap();
        let frame_interval = Duration::from_micros(16_667);
        assert!(
            span > Duration::from_millis(5),
            "pacing should spread well beyond a 1 ms burst, got {span:?}"
        );
        assert!(
            span < frame_interval,
            "pacing must not push packets past the next frame boundary, got {span:?}"
        );
        // Without pacing, all 30 would arrive in <1 ms. With pacing
        // at 25 Mbps, only ~3 should land in the first millisecond
        // (1 ms × 25 Mbps / (1200 B × 8) ≈ 2.6).
        assert!(
            due_in_first_ms <= 4,
            "expected ≤ 4 packets due in the first ms, got {due_in_first_ms}"
        );
    }

    #[test]
    fn no_pacing_needed_when_natural_rate_below_target() {
        // 100 Mbps target, 5 packets × 1200 bytes = 6 000 bytes.
        //   6_000 * 8 / 100_000_000 = 0.48 ms total drain.
        // The last packet's due-time should land within ~half a
        // millisecond of frame_start.
        let mut p = pacer_at(100_000);
        let t0 = Instant::now();
        p.begin_frame(t0);

        let mut last_due = None;
        for _ in 0..5 {
            let due = p.due_time(1200).unwrap();
            last_due = Some(due);
            p.wait_for_slot_test(1200);
        }
        let span = last_due.unwrap() - t0;
        assert!(
            span < Duration::from_millis(1),
            "5 packets at 100 Mbps should drain in well under 1 ms, got {span:?}"
        );
    }

    #[test]
    fn first_packet_is_due_immediately_at_or_after_frame_start() {
        // The first packet's due time should be `frame_start +
        // (packet_bytes * 8 / bps)` — strictly after frame_start by
        // a small positive amount, never before.
        let mut p = pacer_at(25_000);
        let t0 = Instant::now();
        p.begin_frame(t0);
        let due = p.due_time(1200).unwrap();
        assert!(due > t0, "first packet must be due at frame_start + ε");
        assert!(
            due - t0 < Duration::from_micros(500),
            "first packet's due should be sub-ms at 25 Mbps with 1200 B"
        );
    }

    #[test]
    fn live_bitrate_update_takes_effect_immediately() {
        let mut p = pacer_at(25_000);
        let t0 = Instant::now();
        p.begin_frame(t0);
        // Two packets at 25 Mbps.
        let due_a = p.due_time(1200).unwrap();
        p.wait_for_slot_test(1200);
        // Now bump to 100 Mbps; next packet's due slope is 4× shallower.
        p.set_target_bitrate_kbps(100_000);
        let due_b = p.due_time(1200).unwrap();
        let span_a = due_a - t0; // first 1200 B at 25 Mbps
        let span_b = due_b - t0; // 2400 B at the *new* rate combined w/ accounting
                                 // due_b reflects 2400 B at 100 Mbps = 192 µs total.
                                 // due_a reflects 1200 B at 25 Mbps = 384 µs total.
        assert!(
            span_b < span_a,
            "post-bump cumulative due should drop below pre-bump first packet"
        );
        assert!(
            span_b < Duration::from_micros(300),
            "2400 B at 100 Mbps = ~192 µs"
        );
    }

    #[test]
    fn begin_frame_resets_byte_accounting() {
        let mut p = pacer_at(25_000);
        p.begin_frame(Instant::now());
        for _ in 0..10 {
            p.wait_for_slot_test(1200);
        }
        let t1 = Instant::now() + Duration::from_millis(50);
        p.begin_frame(t1);
        let due = p.due_time(1200).unwrap();
        // After reset, due is `t1 + (1200 * 8 / 25 Mbps)` = t1 + 384 µs,
        // NOT pushed by the previous frame's 14 400 bytes of accounting.
        assert!(
            due - t1 < Duration::from_micros(500),
            "begin_frame must reset bytes_sent_this_frame; got due-t1={:?}",
            due - t1
        );
    }

    #[test]
    fn wait_for_slot_sleeps_when_due_in_future() {
        // Real wall-clock test: 10 Mbps × 10 KB worth of packets
        // ≈ 8 ms theoretical. Lower bound 5 ms to absorb CI jitter
        // while still proving wait_for_slot did sleep. Kept short
        // to minimise total test-suite wall time.
        let mut p = pacer_at(10_000);
        let t0 = Instant::now();
        p.begin_frame(t0);
        for _ in 0..10 {
            p.wait_for_slot(1000);
        }
        let elapsed = t0.elapsed();
        assert!(
            elapsed >= Duration::from_millis(5),
            "10 × 1000 B at 10 Mbps should take ≥ 5 ms (theoretical 8 ms); got {elapsed:?}"
        );
    }

    // Test helper: bookkeep without sleeping, so the `due_time`
    // spread tests don't run for the full frame interval.
    impl PacedSender {
        fn wait_for_slot_test(&mut self, packet_bytes: usize) {
            self.bytes_sent_this_frame = self
                .bytes_sent_this_frame
                .saturating_add(packet_bytes as u64);
        }
    }
}
