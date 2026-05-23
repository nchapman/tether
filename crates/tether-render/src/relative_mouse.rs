//! Relative-mouse delta accumulator.
//!
//! winit's `DeviceEvent::MouseMotion { delta: (f64, f64) }` reports
//! pointer motion as a high-resolution `f64` pair — the OS already
//! pre-scaled by the user's pointer acceleration curve, so on a
//! trackpad a slow swipe can produce many sub-pixel deltas in a row.
//! The wire form is `i16` integer pixels. Truncating each `f64` to
//! `i16` individually would silently lose sub-pixel motion forever;
//! a steady 0.4-px/event stream rounds to zero every time.
//!
//! [`SubPixelAccum`] tracks the fractional residue per axis and
//! folds it back into the next sample, so the sum of emitted
//! integer deltas converges to the sum of the raw `f64` deltas
//! (modulo the per-event `i16` saturation cap, which is a separate
//! ceiling for huge gaming-mouse flicks). Mirror image of the
//! standard "DDA line drawing" idea applied to mouse motion.

/// Per-instance sub-pixel residue. One lives in the client's
/// renderer state alongside `age_tracker` etc.; reset on cursor-
/// mode toggle so a pre-toggle residue can't leak into the first
/// post-toggle delta.
#[derive(Default, Debug, Clone, Copy)]
pub struct SubPixelAccum {
    residue_x: f64,
    residue_y: f64,
}

impl SubPixelAccum {
    /// Fold one `(dx, dy)` raw motion sample. Returns `Some((dx,
    /// dy))` clamped to `i16` if the combined value (accum + sample)
    /// has at least one whole pixel of motion on either axis;
    /// `None` for pure sub-pixel that we'll roll into the next call.
    ///
    /// The clamp uses `i16::MIN..=i16::MAX` saturation: a one-frame
    /// 40_000-pixel gaming-mouse flick is rare but not impossible,
    /// and the wire shape gives us 32 767 max — capping is the
    /// right "deliver what fits, drop the rest" choice over an
    /// alternative panic or wraparound.
    pub fn record(&mut self, dx: f64, dy: f64) -> Option<(i16, i16)> {
        self.residue_x += dx;
        self.residue_y += dy;
        let whole_x = self.residue_x.trunc();
        let whole_y = self.residue_y.trunc();
        if whole_x == 0.0 && whole_y == 0.0 {
            return None;
        }
        self.residue_x -= whole_x;
        self.residue_y -= whole_y;
        Some((clamp_i16(whole_x), clamp_i16(whole_y)))
    }

    /// Drop any pending residue. Called on cursor-mode toggle so
    /// stale fractional motion from the prior session of relative
    /// mode doesn't get folded into the first post-toggle delta.
    pub fn reset(&mut self) {
        self.residue_x = 0.0;
        self.residue_y = 0.0;
    }
}

#[allow(clippy::cast_possible_truncation)]
fn clamp_i16(v: f64) -> i16 {
    if v >= f64::from(i16::MAX) {
        i16::MAX
    } else if v <= f64::from(i16::MIN) {
        i16::MIN
    } else {
        v as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_pixel_passes_through() {
        let mut a = SubPixelAccum::default();
        assert_eq!(a.record(5.0, -3.0), Some((5, -3)));
    }

    #[test]
    fn sub_pixel_alone_is_held() {
        let mut a = SubPixelAccum::default();
        assert_eq!(a.record(0.4, 0.4), None);
        assert_eq!(a.record(0.4, 0.4), None);
        // 1.2 px accumulated; whole-px portion fires, residue stays.
        let (dx, dy) = a.record(0.4, 0.4).unwrap();
        assert_eq!((dx, dy), (1, 1));
    }

    #[test]
    fn long_run_of_sub_pixel_converges_to_correct_total() {
        // Feeding 100 × 0.4-px deltas should report a cumulative
        // dx of 40 px (with at most 1-px residue left over).
        let mut a = SubPixelAccum::default();
        let mut total_x: i32 = 0;
        let mut total_y: i32 = 0;
        for _ in 0..100 {
            if let Some((dx, dy)) = a.record(0.4, 0.4) {
                total_x += i32::from(dx);
                total_y += i32::from(dy);
            }
        }
        assert_eq!(total_x, 40);
        assert_eq!(total_y, 40);
    }

    #[test]
    fn mixed_sign_residue_does_not_stall() {
        let mut a = SubPixelAccum::default();
        a.record(-0.4, 0.4); // residue (-0.4, 0.4)
        a.record(0.4, -0.4); // residue (0.0, 0.0) — back to zero
        a.record(0.4, -0.4); // residue (0.4, -0.4) — still sub
        let (dx, dy) = a.record(0.7, -0.7).unwrap_or_default();
        assert_eq!((dx, dy), (1, -1));
    }

    #[test]
    fn extreme_delta_saturates_to_i16_range() {
        let mut a = SubPixelAccum::default();
        let (dx, dy) = a.record(1e9, -1e9).unwrap();
        assert_eq!(dx, i16::MAX);
        assert_eq!(dy, i16::MIN);
    }

    #[test]
    fn reset_clears_residue() {
        let mut a = SubPixelAccum::default();
        a.record(0.7, 0.7);
        a.reset();
        // After reset, a 0.5-px delta alone should still be sub.
        assert_eq!(a.record(0.5, 0.5), None);
    }
}
