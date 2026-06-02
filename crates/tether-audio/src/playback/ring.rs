//! Lock-free single-producer / single-consumer ring of interleaved f32 PCM.
//!
//! This is the seam between the Opus decode thread (producer) and the cpal
//! output callback (consumer, a hard-real-time thread). The callback must never
//! block, so the ring takes no locks: the producer owns `tail`, the consumer
//! owns `head`, and a Release/Acquire pair on those two atomics publishes each
//! side's sample writes/reads to the other. `head`/`tail` are monotonic sample
//! counts (wrapping at `usize::MAX`); the live range is always `tail - head`,
//! never more than the capacity, so plain wrapping arithmetic is exact.
//!
//! Capacity is a power of two so indexing is a mask. The ring carries raw
//! samples only — priming, latency-cap drop, and underrun handling live in the
//! pure [`PlaybackPolicy`](super::policy::PlaybackPolicy) on the consumer side.
//! The ring is channel-agnostic; left/right (frame) alignment is preserved by
//! the producer pushing whole frames and the consumer advancing `head` by
//! frame-aligned amounts (enforced in the policy).

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct Inner {
    buf: Box<[UnsafeCell<f32>]>,
    /// `capacity - 1`; capacity is a power of two so `index & mask` wraps.
    mask: usize,
    /// Total samples consumed (consumer-owned writer). Monotonic, wrapping.
    head: AtomicUsize,
    /// Total samples produced (producer-owned writer). Monotonic, wrapping.
    tail: AtomicUsize,
}

// SAFETY: `Inner` is shared between exactly one producer thread and one consumer
// thread via `Arc`. The producer writes only cells in `[tail, tail+n)` and is
// the only writer of `tail`; the consumer reads only cells in `[head, head+n)`
// (with `head + n <= tail`) and is the only writer of `head`. The head/tail
// discipline keeps those index ranges disjoint, so no two threads ever touch the
// same cell concurrently. The Release store of `tail`/`head` paired with the
// Acquire load on the other side publishes each thread's cell accesses before
// the index that exposes them becomes visible.
unsafe impl Sync for Inner {}
unsafe impl Send for Inner {}

impl Inner {
    fn buffered(&self) -> usize {
        self.tail
            .load(Ordering::Acquire)
            .wrapping_sub(self.head.load(Ordering::Acquire))
    }
}

/// Producer half — held by the [`AudioSink`](super::AudioSink), written from the
/// decode thread. Not `Clone`: exactly one producer.
pub struct Producer {
    inner: Arc<Inner>,
}

/// Consumer half — owned by the cpal output callback. Not `Clone`: exactly one
/// consumer.
pub struct Consumer {
    inner: Arc<Inner>,
}

/// Build a ring holding at least `min_capacity` samples (rounded up to a power
/// of two), split into its producer and consumer halves.
pub fn channel(min_capacity: usize) -> (Producer, Consumer) {
    let cap = min_capacity.max(2).next_power_of_two();
    let buf = (0..cap)
        .map(|_| UnsafeCell::new(0.0))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let inner = Arc::new(Inner {
        buf,
        mask: cap - 1,
        head: AtomicUsize::new(0),
        tail: AtomicUsize::new(0),
    });
    (
        Producer {
            inner: Arc::clone(&inner),
        },
        Consumer { inner },
    )
}

/// Test-only: build a ring with `head`/`tail` pre-seeded to `start`, so a test
/// can drive the monotonic counters across the `usize::MAX` wrap boundary
/// (unreachable in a normal run — it'd take millions of years at 48 kHz).
#[cfg(test)]
fn channel_at(min_capacity: usize, start: usize) -> (Producer, Consumer) {
    let (p, c) = channel(min_capacity);
    p.inner.head.store(start, Ordering::Relaxed);
    p.inner.tail.store(start, Ordering::Relaxed);
    (p, c)
}

impl Producer {
    /// Push up to `samples.len()` interleaved samples. Returns the number
    /// written; fewer than requested only when the ring is physically full, in
    /// which case the newest samples are dropped (a safety valve — the consumer
    /// trims to the latency target first, so this is rare).
    pub fn push(&self, samples: &[f32]) -> usize {
        let cap = self.inner.mask + 1;
        // Producer owns `tail`; only the consumer's `head` needs an Acquire load
        // (to see how much it has freed).
        let tail = self.inner.tail.load(Ordering::Relaxed);
        let head = self.inner.head.load(Ordering::Acquire);
        let free = cap - tail.wrapping_sub(head);
        let n = samples.len().min(free);
        for (i, &s) in samples.iter().take(n).enumerate() {
            let idx = tail.wrapping_add(i) & self.inner.mask;
            // SAFETY: `idx` is in the producer's exclusive `[tail, tail+n)` range;
            // no consumer reads it until the Release store of `tail` below.
            unsafe { *self.inner.buf[idx].get() = s };
        }
        self.inner
            .tail
            .store(tail.wrapping_add(n), Ordering::Release);
        n
    }

    /// Samples currently buffered (a snapshot, for logging).
    pub fn buffered(&self) -> usize {
        self.inner.buffered()
    }
}

impl Consumer {
    /// Copy up to `out.len()` samples into `out`; returns the number copied
    /// (fewer when the ring is starved — the caller zero-fills the remainder).
    pub fn pop(&self, out: &mut [f32]) -> usize {
        // Consumer owns `head`; Acquire-load `tail` to see produced samples.
        let head = self.inner.head.load(Ordering::Relaxed);
        let tail = self.inner.tail.load(Ordering::Acquire);
        let n = out.len().min(tail.wrapping_sub(head));
        for (i, slot) in out.iter_mut().take(n).enumerate() {
            let idx = head.wrapping_add(i) & self.inner.mask;
            // SAFETY: `idx` is in the consumer's exclusive `[head, head+n)` range
            // with `head + n <= tail`, so the producer is not writing it.
            *slot = unsafe { *self.inner.buf[idx].get() };
        }
        self.inner
            .head
            .store(head.wrapping_add(n), Ordering::Release);
        n
    }

    /// Discard up to `count` of the oldest samples (latency-cap drop). Returns
    /// the number discarded.
    pub fn skip(&self, count: usize) -> usize {
        let head = self.inner.head.load(Ordering::Relaxed);
        let tail = self.inner.tail.load(Ordering::Acquire);
        let n = count.min(tail.wrapping_sub(head));
        self.inner
            .head
            .store(head.wrapping_add(n), Ordering::Release);
        n
    }

    /// Samples currently buffered.
    pub fn buffered(&self) -> usize {
        self.inner.buffered()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_then_pop_round_trips_in_order() {
        let (p, c) = channel(16);
        assert_eq!(p.push(&[1.0, 2.0, 3.0]), 3);
        assert_eq!(c.buffered(), 3);
        let mut out = [0.0f32; 4];
        assert_eq!(c.pop(&mut out), 3);
        assert_eq!(&out, &[1.0, 2.0, 3.0, 0.0]);
        assert_eq!(c.buffered(), 0);
    }

    #[test]
    fn push_caps_at_capacity_dropping_newest() {
        let (p, c) = channel(8); // capacity 8
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        assert_eq!(p.push(&data), 8, "only capacity samples accepted");
        assert_eq!(c.buffered(), 8);
    }

    #[test]
    fn pop_caps_at_available() {
        let (p, c) = channel(16);
        p.push(&[1.0, 2.0]);
        let mut out = [0.0f32; 8];
        assert_eq!(c.pop(&mut out), 2);
        assert!(out[2..].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn skip_drops_oldest_and_bounds_at_available() {
        let (p, c) = channel(16);
        p.push(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(c.skip(2), 2, "skip the two oldest");
        assert_eq!(c.skip(10), 2, "skip can't exceed what's buffered");
        let mut out = [0.0f32; 4];
        assert_eq!(c.pop(&mut out), 0, "everything was skipped");
    }

    #[test]
    fn monotonic_counters_wrap_across_usize_max() {
        // Seed head=tail just below usize::MAX so a few push/pop rounds carry
        // the wrapping_add/wrapping_sub arithmetic across the 2^usize boundary.
        let (p, c) = channel_at(8, usize::MAX - 3);
        let mut out = [0.0f32; 8];
        for round in 0..4u32 {
            let data: Vec<f32> = (0..6u32).map(|i| (round * 10 + i) as f32).collect();
            assert_eq!(p.push(&data), 6);
            assert_eq!(c.pop(&mut out), 6);
            assert_eq!(&out[..6], data.as_slice());
            assert_eq!(c.buffered(), 0, "buffered stays correct across the wrap");
        }
    }

    #[test]
    fn wraps_around_capacity_boundary() {
        let (p, c) = channel(8); // capacity 8
        let mut out = [0.0f32; 8];
        // Push/pop 6 to move head/tail near the end, then straddle the wrap.
        p.push(&[0.0; 6]);
        c.pop(&mut out[..6]);
        let data: Vec<f32> = (1..=6).map(|i| i as f32).collect();
        assert_eq!(p.push(&data), 6, "spans the index wrap");
        assert_eq!(c.pop(&mut out), 6);
        assert_eq!(&out[..6], data.as_slice());
    }

    /// Cross-thread stress: a producer streams a contiguous integer ramp in
    /// varied bursts while a consumer drains it. The consumer must observe every
    /// value exactly once, in order — no loss, duplication, or reordering. This
    /// exercises the Release/Acquire pairing under real concurrency.
    #[test]
    fn spsc_concurrent_stream_preserves_every_sample_in_order() {
        use std::thread;
        const N: u64 = 1_000_000;
        let (p, c) = channel(1024);

        let producer = thread::spawn(move || {
            let mut next: u64 = 0;
            while next < N {
                let burst = next % 97 + 1;
                let end = (next + burst).min(N);
                let chunk: Vec<f32> = (next..end).map(|v| v as f32).collect();
                let mut off = 0;
                while off < chunk.len() {
                    let wrote = p.push(&chunk[off..]);
                    if wrote == 0 {
                        thread::yield_now();
                    }
                    off += wrote;
                }
                next = end;
            }
        });

        let mut received: u64 = 0;
        let mut out = [0.0f32; 64];
        while received < N {
            let n = c.pop(&mut out);
            if n == 0 {
                thread::yield_now();
                continue;
            }
            for &s in &out[..n] {
                // Values are exact integers in f32 (N < 2^24); a 0.5 tolerance
                // uniquely identifies the expected sample without a float-eq lint.
                assert!(
                    (s - received as f32).abs() < 0.5,
                    "out-of-order/lost at {received}: got {s}"
                );
                received += 1;
            }
        }
        producer.join().unwrap();
        assert_eq!(received, N);
    }
}
