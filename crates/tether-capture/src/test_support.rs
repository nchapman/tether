//! Fakes for integration tests in downstream crates.
//!
//! Gated behind the `test-support` Cargo feature; production builds
//! never compile this module.
//!
//! **Anti-drift rule**: when a fake produces a `CapturedFrame` it
//! constructs it through the same public constructors and field set
//! production code uses. Don't add convenience wrappers that bypass
//! the public API surface — when fields are added to `CpuFrame` or
//! `GpuCapturedFrame`, the compiler must force every fake to be
//! updated alongside.

#![deny(unused_imports)]

use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, TrySendError};

use crate::CapturedFrame;

/// Scripted producer: emits a precise sequence of `CapturedFrame`s on a
/// scripted timeline. Each event is `(delay_after_previous, frame)`; the
/// producer thread sleeps `delay` and then sends.
///
/// Designed for tests that need to reason about exact frame arrival
/// timing relative to consumer behavior — e.g. "consumer is slow, the
/// drain should keep the freshest of N queued frames" or "epoch bump
/// arrives between fragments". For free-running synthetic frames at a
/// fixed cadence, use [`crate::test_pattern::start`] instead.
///
/// The producer drops the **newest** frame on backpressure (single-slot
/// channel + `try_send`), matching the discipline we want from real
/// capture sources. Dropping the returned `Receiver` exits the thread.
///
/// Returned channel is bounded to 1: combined with the producer's
/// drop-newest behavior, a consumer that doesn't read fast enough loses
/// frames, which is precisely the property tests want to exercise.
pub struct ScriptedSource;

impl ScriptedSource {
    pub fn start(events: Vec<(Duration, CapturedFrame)>) -> Receiver<CapturedFrame> {
        let (tx, rx) = bounded(1);
        std::thread::Builder::new()
            .name("tether-capture-scripted".into())
            .spawn(move || {
                for (delay, frame) in events {
                    if !delay.is_zero() {
                        std::thread::sleep(delay);
                    }
                    match tx.try_send(frame) {
                        Ok(()) => {}
                        Err(TrySendError::Full(_)) => {
                            tracing::trace!("scripted source: consumer slow, dropping frame");
                        }
                        Err(TrySendError::Disconnected(_)) => return,
                    }
                }
            })
            .expect("spawn scripted source thread");
        rx
    }
}

/// Wraps a `Receiver<CapturedFrame>` and stalls every consumer read by
/// a caller-provided `delay()` callback. Lets tests inject controlled
/// consumer lag without depending on a clock — the callback returns a
/// `Duration` that the wrapper sleeps before yielding each frame.
///
/// Use case: assert that an upstream drain keeps the freshest frame
/// when the consumer is artificially slow.
pub struct LagInjectingReceiver<F> {
    inner: Receiver<CapturedFrame>,
    delay: F,
}

impl<F: FnMut() -> Duration> LagInjectingReceiver<F> {
    pub fn new(inner: Receiver<CapturedFrame>, delay: F) -> Self {
        Self { inner, delay }
    }

    /// Blocking receive with injected lag.
    pub fn recv(&mut self) -> Result<CapturedFrame, crossbeam_channel::RecvError> {
        let d = (self.delay)();
        if !d.is_zero() {
            std::thread::sleep(d);
        }
        self.inner.recv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuFrame, PixelFormat};
    use tether_protocol::MonoNanos;

    fn make_frame(tag: u8) -> CapturedFrame {
        let t = MonoNanos::now();
        CapturedFrame::Cpu(CpuFrame {
            width: 4,
            height: 4,
            format: PixelFormat::Bgra8,
            data: vec![tag; 4 * 4 * 4],
            t_capture_kernel: t,
            t_capture_userspace: t,
        })
    }

    #[test]
    fn scripted_emits_frames_in_order() {
        let rx = ScriptedSource::start(vec![
            (Duration::ZERO, make_frame(0xa)),
            (Duration::from_millis(5), make_frame(0xb)),
        ]);
        let f1 = rx.recv_timeout(Duration::from_secs(1)).expect("first frame");
        let CapturedFrame::Cpu(c) = f1 else {
            panic!("expected Cpu frame");
        };
        assert_eq!(c.data[0], 0xa);
        let f2 = rx.recv_timeout(Duration::from_secs(1)).expect("second frame");
        let CapturedFrame::Cpu(c) = f2 else {
            panic!("expected Cpu frame");
        };
        assert_eq!(c.data[0], 0xb);
    }

    #[test]
    fn scripted_thread_exits_on_receiver_drop() {
        let rx = ScriptedSource::start(vec![
            (Duration::ZERO, make_frame(0)),
            (Duration::from_millis(50), make_frame(1)),
            (Duration::from_millis(50), make_frame(2)),
        ]);
        let _ = rx.recv_timeout(Duration::from_secs(1));
        drop(rx);
        // Passes if it doesn't hang. The producer hits try_send →
        // Disconnected on its next iteration and returns.
    }

    #[test]
    fn lag_injecting_receiver_calls_delay_then_yields() {
        use std::sync::{
            atomic::{AtomicU32, Ordering},
            Arc,
        };
        let (tx, rx) = bounded::<CapturedFrame>(2);
        tx.send(make_frame(0x42)).unwrap();
        let calls = Arc::new(AtomicU32::new(0));
        let calls_cb = Arc::clone(&calls);
        let mut lagger = LagInjectingReceiver::new(rx, move || {
            calls_cb.fetch_add(1, Ordering::Relaxed);
            Duration::ZERO
        });
        let f = lagger.recv().expect("frame");
        let CapturedFrame::Cpu(c) = f else {
            panic!("expected Cpu frame");
        };
        assert_eq!(c.data[0], 0x42);
        assert_eq!(calls.load(Ordering::Relaxed), 1, "delay callback fires once per recv");
    }
}
