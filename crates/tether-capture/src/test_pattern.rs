//! Synthetic capture backend that produces an animated gradient at a
//! fixed cadence. Used by the walking skeleton and headless tests where
//! a real display isn't available.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{bounded, TrySendError};
use tether_protocol::MonoNanos;

use crate::{CaptureHandle, CapturedFrame, CpuFrame, PixelFormat};

/// Start a test-pattern producer. Returns a receiver bounded to 2 frames.
///
/// Backpressure discipline: if the consumer falls behind, the producer
/// drops the **newest** frame (the one it just generated) and continues.
/// The expert review recommends drop-oldest for real capture, but
/// crossbeam's bounded sender can't pop from its own end of the channel,
/// and the difference at a depth of 2 is small enough not to warrant a
/// custom ring buffer for this synthetic backend.
///
/// Dropping `Receiver` causes the producer thread to exit on the next
/// `try_send`.
pub fn start(width: u32, height: u32, fps: u32) -> CaptureHandle {
    let (tx, rx) = bounded(2);
    let target_fps = Arc::new(AtomicU32::new(fps.max(1)));
    let target_fps_for_thread = Arc::clone(&target_fps);
    std::thread::Builder::new()
        .name("tether-capture-test-pattern".into())
        .spawn(move || {
            let mut t: u32 = 0;
            loop {
                let frame = generate(width, height, t);
                t = t.wrapping_add(1);
                match tx.try_send(frame) {
                    Ok(()) => {}
                    Err(TrySendError::Full(_)) => {
                        tracing::trace!("test-pattern: consumer slow, dropping frame");
                    }
                    Err(TrySendError::Disconnected(_)) => return,
                }
                // Re-read target FPS each iteration so a mid-stream
                // `CaptureHandle::set_target_fps` takes effect on the
                // next produced frame. `Relaxed` is fine — this is a
                // pacing knob, not a synchronization barrier.
                let current_fps = target_fps_for_thread.load(Ordering::Relaxed).max(1);
                let period = Duration::from_secs_f32(1.0 / current_fps as f32);
                std::thread::sleep(period);
            }
        })
        .expect("spawn test-pattern thread");
    CaptureHandle::from_parts(rx, target_fps)
}

// All `as u8` casts in this fn are intentional truncations of math whose
// outputs are bounded to [0, 255] by construction. clippy can't prove it.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn generate(width: u32, height: u32, t: u32) -> CapturedFrame {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    let phase = (t as f32 * 0.05).sin() * 0.5 + 0.5;
    let b_byte = (phase * 255.0) as u8;
    let t_capture = MonoNanos::now();
    for y in 0..height {
        for x in 0..width {
            // BGRA layout matches what most real-capture backends emit
            // (ScreenCaptureKit + DXGI default to BGRA; PipeWire often
            // negotiates BGRA on Wayland compositors).
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            data.extend_from_slice(&[b_byte, g, r, 255]);
        }
    }
    CapturedFrame::Cpu(CpuFrame {
        width,
        height,
        format: PixelFormat::Bgra8,
        data,
        t_capture_kernel: t_capture,
        t_capture_userspace: t_capture,
        native_damage: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn produces_frames_of_expected_shape() {
        let handle = start(64, 48, 60);
        let frame = handle
            .rx()
            .recv_timeout(Duration::from_secs(2))
            .expect("frame from test_pattern");
        let CapturedFrame::Cpu(cpu) = frame else {
            panic!("test_pattern must emit Cpu frames");
        };
        assert_eq!(cpu.width, 64);
        assert_eq!(cpu.height, 48);
        assert_eq!(cpu.format, PixelFormat::Bgra8);
        assert_eq!(cpu.data.len(), 64 * 48 * 4);
        // alpha channel is always 255
        for px in cpu.data.chunks_exact(4) {
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn producer_exits_when_receiver_dropped() {
        let handle = start(8, 8, 120);
        // Consume one frame so we know the producer started, then drop.
        let _ = handle.rx().recv_timeout(Duration::from_secs(2)).unwrap();
        drop(handle);
        // No assertion on thread exit — the test passes if it doesn't
        // hang. The thread observes Disconnected on its next try_send
        // and returns.
    }

    /// Live `set_target_fps` is observed by the producer's per-
    /// iteration re-read of the atomic. The contract is "the *next*
    /// sleep after the retune uses the new period" — an in-flight
    /// sleep at the old period continues to its scheduled wake. To
    /// observe the cadence shift without racing that in-flight sleep,
    /// start at a moderate rate (10 fps → 100 ms periods), retune to
    /// a high rate (200 fps → 5 ms periods), then time the gap
    /// between two consecutive post-retune frames.
    ///
    /// Catches the regression where the producer computes `period`
    /// once outside the loop (pre-refactor behaviour) — that
    /// implementation would still hit the 100 ms cadence after the
    /// retune.
    #[test]
    fn set_target_fps_changes_cadence_mid_stream() {
        let handle = start(8, 8, 10);
        // Drain the initial frame and one more so the producer is
        // settled into its loop with the old period.
        let _ = handle
            .rx()
            .recv_timeout(Duration::from_secs(2))
            .expect("frame 1");
        let _ = handle
            .rx()
            .recv_timeout(Duration::from_secs(2))
            .expect("frame 2");
        handle.set_target_fps(200);
        // Skip one transitional frame whose preceding sleep was at
        // the old 100 ms period.
        let _ = handle
            .rx()
            .recv_timeout(Duration::from_millis(250))
            .expect("frame 3");
        // Now time the gap between two frames whose sleeps were both
        // post-retune. At 200 fps the gap is ~5 ms; with generous
        // jitter budget assert < 50 ms (well below the 100 ms old
        // period — the discriminator).
        let t = std::time::Instant::now();
        let _ = handle
            .rx()
            .recv_timeout(Duration::from_millis(250))
            .expect("frame 4");
        let gap = t.elapsed();
        // Threshold 80 ms = comfortably below the 100 ms old period
        // (the discriminator) with enough margin for scheduler jitter
        // on a loaded CI host. Tighter thresholds (20 ms / 50 ms)
        // were flaky under load.
        assert!(
            gap < Duration::from_millis(80),
            "expected post-retune frame gap < 80 ms (200 fps), got {gap:?}; \
             producer may not be re-reading target_fps each iteration"
        );
    }

    #[test]
    fn set_target_fps_clamps_zero_to_one() {
        let handle = start(8, 8, 60);
        handle.set_target_fps(0);
        assert_eq!(handle.target_fps(), 1, "zero must be clamped to 1");
    }
}
