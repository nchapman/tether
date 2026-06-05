//! RFC-2198-style RED redundancy for the audio send path.
//!
//! Audio rides unreliable QUIC datagrams with no transport FEC, and Opus
//! in-band FEC doesn't fit our config: LBRR is SILK-only, our
//! `RESTRICTED_LOWDELAY` mode forces CELT-only, and even in
//! `OPUS_APPLICATION_AUDIO` the 128 kbps / 10 ms fullband-music operating point
//! stays in CELT — so LBRR would emit little. So instead the host attaches a
//! small redundancy tail to every datagram: the previous
//! `depth` Opus payloads, newest-first. When a datagram is lost but a later one
//! arrives, the client recovers the missing frame from the redundant copy and
//! decodes it in order — no concealment click. The copies are tiny (~80 B per
//! frame at the default 5 ms / 128 kbps), so a `depth` of 1–2 barely moves the
//! bandwidth and stays far under the per-datagram MTU. A `depth` of 0 disables
//! redundancy (the tail is always empty).
//!
//! Recovery span scales with depth: depth 1 recovers any isolated single-frame
//! loss (the common Wi-Fi case); depth N recovers a burst of up to N
//! consecutive lost frames, as long as the first datagram *after* the burst
//! arrives.

use std::collections::VecDeque;

use bytes::Bytes;

/// Builds the RED redundancy tail for each outgoing Opus frame.
///
/// Feed every emitted payload through [`Self::next_tail`] in send order; it
/// returns the tail to ship alongside that payload and records the payload as a
/// redundancy candidate for the frames that follow.
pub struct RedundancyBuffer {
    /// How many previous payloads to attach to each frame.
    depth: usize,
    /// The most recent `depth` payloads, oldest-first.
    history: VecDeque<Bytes>,
}

impl RedundancyBuffer {
    /// Build a buffer that attaches the previous `depth` payloads to each frame.
    /// `depth == 0` disables redundancy.
    #[must_use]
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            history: VecDeque::with_capacity(depth),
        }
    }

    /// The redundancy tail to send with `payload`: the previous `depth`
    /// payloads, **newest-first** (so element `k` is the frame `k + 1` before
    /// this one). Then records `payload` for subsequent frames. Returns an
    /// empty tail when `depth == 0` or fewer than one prior frame exists.
    pub fn next_tail(&mut self, payload: &Bytes) -> Vec<Bytes> {
        // Newest-first: the receiver indexes `redundant[k]` as `frame_seq-(k+1)`.
        let tail: Vec<Bytes> = self.history.iter().rev().cloned().collect();
        if self.depth > 0 {
            self.history.push_back(payload.clone());
            while self.history.len() > self.depth {
                self.history.pop_front();
            }
        }
        tail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(byte: u8) -> Bytes {
        Bytes::from(vec![byte; 4])
    }

    #[test]
    fn depth_zero_never_attaches_redundancy() {
        let mut r = RedundancyBuffer::new(0);
        assert!(r.next_tail(&b(1)).is_empty());
        assert!(r.next_tail(&b(2)).is_empty());
        assert!(r.next_tail(&b(3)).is_empty());
    }

    #[test]
    fn first_frame_has_no_predecessor() {
        let mut r = RedundancyBuffer::new(2);
        assert!(
            r.next_tail(&b(1)).is_empty(),
            "nothing precedes the first frame"
        );
    }

    #[test]
    fn tail_is_previous_payloads_newest_first() {
        let mut r = RedundancyBuffer::new(2);
        r.next_tail(&b(1)); // frame 1
                            // Frame 2 carries [frame1].
        assert_eq!(r.next_tail(&b(2)), vec![b(1)]);
        // Frame 3 carries [frame2, frame1] — newest-first.
        assert_eq!(r.next_tail(&b(3)), vec![b(2), b(1)]);
        // Frame 4 drops frame1 past the depth-2 window: [frame3, frame2].
        assert_eq!(r.next_tail(&b(4)), vec![b(3), b(2)]);
    }

    #[test]
    fn depth_one_recovers_a_single_predecessor() {
        let mut r = RedundancyBuffer::new(1);
        r.next_tail(&b(1));
        assert_eq!(r.next_tail(&b(2)), vec![b(1)]);
        assert_eq!(
            r.next_tail(&b(3)),
            vec![b(2)],
            "only the immediate predecessor"
        );
    }
}
