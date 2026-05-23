//! Coalesced loss-recovery request signal.
//!
//! Sibling of [`crate::IdrSignal`] for `ControlMessage::RequestRecovery`.
//! Multiple `RequestRecovery` messages arriving between encode ticks
//! collapse to one recovery action — same coalescing pattern, but
//! the carrying datum is the *lowest* `last_known_good_frame_id`
//! seen (most pessimistic view of what the client trusts), not a
//! plain bool.
//!
//! Lowest-wins so a client that gets bursts of loss across multiple
//! reassembly windows produces the most-conservative recovery (the
//! earliest still-good reference). A naive "latest wins" would let
//! a later, smaller drop overwrite an earlier, larger one and roll
//! the host's invalidation forward incorrectly.
//!
//! Phase 1 (this file): the signal exists and is wired through
//! host/client. The encoder side currently treats any pending
//! recovery as "force an IDR" — equivalent to a `ForceIdr`, but
//! triggered by the client's reassembler the moment it sees a stale
//! drop, well before the decoder thread's IDR-on-warning recovery
//! kicks in. Phase 2 will replace the IDR fallback with actual
//! VAAPI LTR re-prediction against the pessimistic id.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Sentinel meaning "no recovery requested." `u32::MAX` would also
/// work but the natural lattice for "lowest wins" needs an absolute
/// upper bound; `u64::MAX` as the sentinel keeps the encoded
/// pessimistic-merge logic uncluttered.
const NO_REQUEST: u64 = u64::MAX;

/// Shared "client lost some fragments; lowest still-good frame_id
/// is N" flag.
///
/// Clone-cheap; both ends hold an `Arc<AtomicU64>`. `raise(N)`
/// merges via `min` so the most-pessimistic id wins; `take()`
/// returns the pending id (or `None`) and resets the slot.
#[derive(Clone, Debug, Default)]
pub struct RecoverySignal {
    pessimistic: Arc<AtomicU64>,
}

impl RecoverySignal {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pessimistic: Arc::new(AtomicU64::new(NO_REQUEST)),
        }
    }

    /// Merge a new "last-known-good = id" report. Lowest wins.
    pub fn raise(&self, last_known_good_frame_id: u32) {
        let new = u64::from(last_known_good_frame_id);
        let mut cur = self.pessimistic.load(Ordering::Relaxed);
        loop {
            if new >= cur {
                // Existing report is at least as pessimistic; nothing
                // to do.
                return;
            }
            match self.pessimistic.compare_exchange_weak(
                cur,
                new,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Atomically read-and-reset. Returns the pessimistic
    /// `last_known_good_frame_id` if a recovery was pending,
    /// `None` otherwise.
    pub fn take(&self) -> Option<u32> {
        let v = self.pessimistic.swap(NO_REQUEST, Ordering::Relaxed);
        if v == NO_REQUEST {
            None
        } else {
            // `raise` only stores values from u32::from(...), so v
            // is provably in u32 range here; debug_assert keeps the
            // invariant visible.
            debug_assert!(v <= u64::from(u32::MAX));
            Some(v as u32)
        }
    }

    /// Non-consuming probe — for callers that need to know "is a
    /// recovery pending" without taking it. Encode-loop damage-skip
    /// uses this so an idle frame still goes through encode when
    /// loss recovery is pending, just like the `IdrSignal::peek`
    /// pattern.
    pub fn peek(&self) -> Option<u32> {
        let v = self.pessimistic.load(Ordering::Relaxed);
        if v == NO_REQUEST {
            None
        } else {
            debug_assert!(v <= u64::from(u32::MAX));
            Some(v as u32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_take_is_none() {
        let s = RecoverySignal::new();
        assert_eq!(s.take(), None);
    }

    #[test]
    fn single_raise_take() {
        let s = RecoverySignal::new();
        s.raise(42);
        assert_eq!(s.take(), Some(42));
        assert_eq!(s.take(), None, "take must reset");
    }

    #[test]
    fn lowest_wins() {
        let s = RecoverySignal::new();
        s.raise(100);
        s.raise(50);
        s.raise(75);
        assert_eq!(s.take(), Some(50), "merge keeps the lowest id");
    }

    #[test]
    fn raise_above_pending_does_not_clobber() {
        let s = RecoverySignal::new();
        s.raise(10);
        s.raise(99);
        assert_eq!(s.take(), Some(10));
    }

    #[test]
    fn peek_does_not_consume() {
        let s = RecoverySignal::new();
        s.raise(7);
        assert_eq!(s.peek(), Some(7));
        assert_eq!(s.take(), Some(7));
    }

    #[test]
    fn clones_share_state() {
        let s = RecoverySignal::new();
        let s2 = s.clone();
        s2.raise(3);
        assert_eq!(s.take(), Some(3));
    }
}
