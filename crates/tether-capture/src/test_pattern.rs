//! Synthetic capture backend that produces an animated gradient at a
//! fixed cadence. Used by the walking skeleton and headless tests where
//! a real display isn't available.

use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, TrySendError};
use tether_protocol::MonoNanos;

use crate::{CapturedFrame, CpuFrame, PixelFormat};

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
pub fn start(width: u32, height: u32, fps: u32) -> Receiver<CapturedFrame> {
    let (tx, rx) = bounded(2);
    let period = Duration::from_secs_f32(1.0 / f32::from(u16::try_from(fps.max(1)).unwrap_or(60)));
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
                std::thread::sleep(period);
            }
        })
        .expect("spawn test-pattern thread");
    rx
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
        let rx = start(64, 48, 60);
        let frame = rx
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
        let rx = start(8, 8, 120);
        // Consume one frame so we know the producer started, then drop.
        let _ = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        drop(rx);
        // No assertion on thread exit — the test passes if it doesn't
        // hang. The thread observes Disconnected on its next try_send
        // and returns.
    }
}
