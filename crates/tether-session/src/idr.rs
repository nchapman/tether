//! Coalesced IDR-request signal.
//!
//! Multiple `ForceIdr` messages arriving from the client between two
//! encode calls should produce one keyframe, not N. The mechanism is a
//! shared `AtomicBool` that the recv side `.raise()`s and the encode
//! side `.take()`s with a single atomic swap.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A shared "force a keyframe on the next encode" flag.
///
/// Clone-cheap; both ends hold an `Arc<AtomicBool>` under the hood.
/// `raise` is idempotent — `N` calls between two `take`s collapse to
/// one keyframe, which is the desired behavior when packet loss
/// triggers several `ForceIdr` requests in quick succession.
#[derive(Clone, Debug, Default)]
pub struct IdrSignal {
    flag: Arc<AtomicBool>,
}

impl IdrSignal {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the flag. Idempotent; cheap.
    ///
    /// `Release` so that any writes the recv side performed before
    /// raising the flag (e.g. updating loss counters that feed into
    /// the IDR decision) are visible to the encode thread once it
    /// observes the raised bit on Apple Silicon / ARM. `Relaxed`
    /// would be safe on x86 alone but not on weakly-ordered ISAs.
    pub fn raise(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// Atomically read-and-clear. Returns `true` if a force was
    /// pending — call this at the top of the encode hot path and pass
    /// the result as `force_keyframe`.
    ///
    /// `Acquire` pairs with [`Self::raise`]'s `Release` so the recv
    /// side's preceding writes are visible here. The write-back of
    /// `false` needs no stronger ordering: the only consumer of the
    /// cleared flag is this same encode thread (subsequent `take` /
    /// `peek` calls), and intra-thread ordering doesn't need a
    /// fence.
    pub fn take(&self) -> bool {
        self.flag.swap(false, Ordering::Acquire)
    }

    /// Non-consuming read. Used by damage-skip gating to decide
    /// whether an unchanged frame *must* still go through the
    /// encoder for a pending IDR. The eventual [`Self::take`] call
    /// in the encode branch is what actually clears the flag, so
    /// peek + take is safe — no IDR is lost between the two reads.
    /// Don't substitute this for `take` in the encode hot path: a
    /// peek that finds `true` and is never followed by a `take`
    /// would coalesce all subsequent requests into the same single
    /// IDR.
    ///
    /// `Acquire` pairs with [`Self::raise`]'s `Release`.
    pub fn peek(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesces_multiple_raises() {
        let s = IdrSignal::new();
        s.raise();
        s.raise();
        s.raise();
        assert!(s.take(), "first take after raise should be true");
        assert!(!s.take(), "second take should be false (coalesced)");
    }

    #[test]
    fn clones_share_state() {
        let s = IdrSignal::new();
        let s2 = s.clone();
        s2.raise();
        assert!(s.take(), "raise via clone is visible to original");
    }

    #[test]
    fn raise_on_one_thread_is_observed_on_another() {
        // Sanity check that a raise on a producer thread is observed
        // by a consumer thread that loops on take. Not a formal proof
        // of the memory model — just confirms the Release/Acquire
        // pairing doesn't drop the signal in practice.
        use std::thread;
        use std::time::{Duration, Instant};

        for _ in 0..32 {
            let s = IdrSignal::new();
            let producer = s.clone();
            let handle = thread::spawn(move || {
                producer.raise();
            });
            let deadline = Instant::now() + Duration::from_secs(1);
            loop {
                if s.take() {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "consumer never observed the raise"
                );
                std::hint::spin_loop();
            }
            handle.join().unwrap();
        }
    }

    #[test]
    fn peek_does_not_consume() {
        // Damage-skip relies on this: a non-consuming read so the
        // encode branch's later `take` still finds the bit.
        let s = IdrSignal::new();
        s.raise();
        assert!(s.peek());
        assert!(s.peek(), "second peek still sees the raise");
        assert!(s.take(), "take finds the bit despite two peeks");
        assert!(!s.peek(), "after take, peek is clean");
    }
}
