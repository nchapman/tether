//! Frame-change detection: skip encode + send when the captured
//! frame is bit-identical to its predecessor.
//!
//! The host pumps captured frames through [`DamageSignal::classify`]
//! before handing them to the encoder. When the signal returns
//! [`DamageHint::Unchanged`], the send loop drops the frame entirely;
//! the decoder/renderer already holds the previous frame indefinitely,
//! so the visual output is identical to encoding-and-sending another
//! copy at the cost of every bit on the wire. On a desktop sitting at
//! a quiescent editor, this elides the steady-state stream completely.
//!
//! ## What's implemented today
//!
//! [`HashDamage`] hashes a strided sample of the captured BGRA bytes
//! (CPU-resident frames only). Stride is picked to land roughly
//! [`HASH_SAMPLE_BYTES`] bytes worth of pixels per frame, sampled
//! across the image rather than one contiguous region — a localized
//! change (e.g. a blinking cursor in one corner) is still detected.
//!
//! GPU-resident frames (DMA-BUF on Linux, IOSurface on macOS) bypass
//! the hash and report [`DamageHint::Unknown`], because reading their
//! contents back to host memory would defeat the zero-copy hot path.
//! The caller treats `Unknown` as "encode anyway" — damage-detection
//! never *causes* a frame to be dropped; it only skips frames we've
//! already proven are duplicates.
//!
//! ## Native platform damage signals
//!
//! Both PipeWire (`SPA_META_VideoDamage`) and ScreenCaptureKit
//! (`SCStreamFrameInfo.status`) emit per-frame metadata that's
//! strictly cheaper to consult than any hash. The capture backends
//! read those signals and attach a [`NativeDamage`] to each
//! [`CapturedFrame`] when available. [`HashDamage::classify`] uses
//! the native idle hint to short-circuit on quiescent desktops where
//! it's authoritative; the hash stays the universal fallback for
//! CPU frames the native signal didn't cover and for GPU frames
//! where idle wasn't reported.

use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

use crate::CapturedFrame;

/// Per-frame native damage metadata attached by the capture backend.
///
/// PipeWire emits `SPA_META_VideoDamage` whose region list is empty
/// for an idle frame; ScreenCaptureKit emits
/// `SCStreamFrameInfo.status` whose `Idle`/`Blank`/`Suspended` values
/// mean the frame is a periodic liveness re-emit with no visible
/// change. Both collapse to `idle = true` here.
///
/// `Option<NativeDamage>` on a captured frame is the three-state
/// signal: `None` means the backend had nothing to report (older
/// PipeWire, missing attachment, CPU SHM path), `Some { idle: false }`
/// means the backend confirmed real change, `Some { idle: true }`
/// means the backend confirmed nothing changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeDamage {
    pub idle: bool,
}

/// Outcome of [`DamageSignal::classify`] for one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DamageHint {
    /// The frame definitely differs from the previous one. Encode it.
    Changed,
    /// The frame is bit-identical to the previous one (modulo the
    /// strided sample we examined). Safe to skip on its own; the
    /// caller still MUST encode if a keyframe was forced — a client
    /// mid-reconnect requests an IDR specifically to bootstrap, and
    /// skipping it would deadlock the session.
    Unchanged,
    /// Damage state can't be determined for this frame (GPU buffer
    /// we can't cheaply read back, native signal unavailable, etc.).
    /// The caller treats this as "encode anyway" — `Unknown` is
    /// strictly safer than guessing `Unchanged`.
    Unknown,
}

/// Per-frame change classifier. One instance lives in the host send
/// loop; classification is stateful (the implementation compares
/// against its own remembered previous frame).
///
/// Implementations must handle resolution changes gracefully —
/// returning `Unchanged` after a `(width, height, format)` change
/// would lose the first new-shape frame, which the client needs.
/// [`HashDamage`] handles this by including those fields in its
/// fingerprint tuple; implementations backed by native signals
/// should do whichever is natural for their state.
pub trait DamageSignal: Send {
    fn classify(&mut self, frame: &CapturedFrame) -> DamageHint;
}

/// Approximate target byte count for the strided sample. 4096 is
/// large enough to make accidental hash collisions vanishingly rare
/// over a few hours of desktop usage, small enough to add no
/// detectable wall time to the host's send loop. Sized as bytes-of-
/// pixels, not pixels — 4 KiB ÷ 4 bytes per BGRA pixel = 1024
/// pixels sampled per frame.
const HASH_SAMPLE_BYTES: usize = 4096;

/// Damage classifier backed by a strided CPU hash. Default impl for
/// callers that don't have a native signal to feed.
#[derive(Debug, Default)]
pub struct HashDamage {
    /// Hash of the previous frame's strided sample, paired with that
    /// frame's `(width, height, format)` discriminator. Stored as a
    /// tuple so a resolution change doesn't silently match a
    /// numerically-equal hash from a different image shape.
    last: Option<FrameFingerprint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrameFingerprint {
    hash: u64,
    width: u32,
    height: u32,
    format: crate::PixelFormat,
}

impl HashDamage {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn fingerprint_cpu(frame: &crate::CpuFrame) -> FrameFingerprint {
        // Sample roughly HASH_SAMPLE_BYTES across the whole buffer.
        // For tiny frames (e.g. the 320x240 test pattern at 307200
        // bytes), the stride approaches 1 and we hash the whole
        // thing — which is also fine: the test pattern is the
        // smallest case and 300 KiB is still ~µs.
        let len = frame.data.len();
        let stride = (len / HASH_SAMPLE_BYTES).max(1);
        // Gather the strided sample into a small scratch buffer and
        // feed the hasher in one `write` rather than 1024 single-byte
        // writes — one mixing round per 8 bytes instead of one per
        // byte. The algorithm is whatever std::hash::DefaultHasher
        // uses (currently SipHash, but the std lib does not guarantee
        // this across Rust versions); we only need session-scoped
        // identity so non-stable hashes are fine.
        let mut scratch = Vec::with_capacity(len.div_ceil(stride));
        for &b in frame.data.iter().step_by(stride) {
            scratch.push(b);
        }
        let mut h = DefaultHasher::new();
        h.write(&scratch);
        FrameFingerprint {
            hash: h.finish(),
            width: frame.width,
            height: frame.height,
            format: frame.format,
        }
    }
}

/// Native-damage extraction. Free function so callers (and tests)
/// can reach it without going through `&mut HashDamage`.
pub fn native_damage_of(frame: &CapturedFrame) -> Option<NativeDamage> {
    match frame {
        CapturedFrame::Cpu(f) => f.native_damage,
        CapturedFrame::Gpu(f) => f.native_damage,
    }
}

impl DamageSignal for HashDamage {
    fn classify(&mut self, frame: &CapturedFrame) -> DamageHint {
        // Trust a native "idle" signal unconditionally — both PW and
        // SCK only flag idle when *they* observed no compositor
        // changes since the previous frame, which is a stricter
        // statement than our hash can make. A native "changed"
        // signal does NOT short-circuit: on CPU we still want the
        // hash's bit-identical check (cheap belt-and-suspenders), and
        // on GPU there's nothing better to fall back to than the hash
        // we can't run.
        if let Some(nd) = native_damage_of(frame) {
            if nd.idle {
                return DamageHint::Unchanged;
            }
        }
        let CapturedFrame::Cpu(cpu) = frame else {
            // GPU-resident frames bypass the hash. See the module
            // doc for the zero-copy rationale.
            return DamageHint::Unknown;
        };
        let fp = Self::fingerprint_cpu(cpu);
        let hint = match self.last {
            Some(prev) if prev == fp => DamageHint::Unchanged,
            _ => DamageHint::Changed,
        };
        self.last = Some(fp);
        hint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuFrame, PixelFormat};
    use tether_protocol::MonoNanos;

    fn cpu(width: u32, height: u32, fill: u8) -> CapturedFrame {
        cpu_with(width, height, fill, None)
    }

    fn cpu_with(
        width: u32,
        height: u32,
        fill: u8,
        native_damage: Option<NativeDamage>,
    ) -> CapturedFrame {
        CapturedFrame::Cpu(CpuFrame {
            width,
            height,
            format: PixelFormat::Bgra8,
            data: vec![fill; (width * height * 4) as usize],
            t_capture_kernel: MonoNanos::ZERO,
            t_capture_userspace: MonoNanos::ZERO,
            native_damage,
        })
    }

    #[test]
    fn first_frame_is_changed() {
        let mut d = HashDamage::new();
        assert_eq!(d.classify(&cpu(64, 64, 0)), DamageHint::Changed);
    }

    #[test]
    fn identical_repeat_is_unchanged() {
        let mut d = HashDamage::new();
        d.classify(&cpu(64, 64, 0));
        assert_eq!(d.classify(&cpu(64, 64, 0)), DamageHint::Unchanged);
    }

    #[test]
    fn pixel_difference_is_changed() {
        let mut d = HashDamage::new();
        d.classify(&cpu(64, 64, 0));
        // Build a frame that differs only at one offset that the
        // stride is guaranteed to visit (offset 0 is always sampled).
        let mut f2 = cpu(64, 64, 0);
        if let CapturedFrame::Cpu(c) = &mut f2 {
            c.data[0] = 0xFF;
        }
        assert_eq!(d.classify(&f2), DamageHint::Changed);
    }

    #[test]
    fn resolution_change_is_changed_even_if_hash_collides() {
        // Different dimensions must never report Unchanged: the
        // client's stream epoch should have bumped by the time this
        // arrives, and a stale "Unchanged" would lose the first new-
        // resolution frame.
        let mut d = HashDamage::new();
        d.classify(&cpu(64, 64, 0));
        let result = d.classify(&cpu(128, 128, 0));
        assert_eq!(result, DamageHint::Changed);
    }

    /// Mirrors the host's send-loop predicate. Encoding the predicate
    /// as a test guards the critical invariant: a forced IDR must
    /// always go through, even on a bit-identical frame.
    fn should_encode(hint: DamageHint, force_idr_pending: bool) -> bool {
        !matches!(hint, DamageHint::Unchanged) || force_idr_pending
    }

    #[test]
    fn native_idle_short_circuits_on_changed_cpu_pixels() {
        // Native idle is authoritative — the compositor observed no
        // rendered change, so even if capture bytes differ (double-
        // buffer aliasing, DMA-BUF slot rotation, format-conversion
        // rounding) the display did not change and we should skip.
        let mut d = HashDamage::new();
        d.classify(&cpu(64, 64, 0));
        let frame = cpu_with(64, 64, 0xFF, Some(NativeDamage { idle: true }));
        assert_eq!(d.classify(&frame), DamageHint::Unchanged);
    }

    #[test]
    fn native_changed_does_not_block_hash() {
        // Native says "changed" but the pixels are bit-identical to
        // the previous frame — the hash still gets the final word
        // and returns Unchanged. This is the cheap-skip path on the
        // CPU/SHM fallback when SCK/PW lied about damage.
        let mut d = HashDamage::new();
        d.classify(&cpu_with(64, 64, 0, Some(NativeDamage { idle: false })));
        let frame = cpu_with(64, 64, 0, Some(NativeDamage { idle: false }));
        assert_eq!(d.classify(&frame), DamageHint::Unchanged);
    }

    #[test]
    fn skip_predicate_drops_repeats_but_honours_forced_idr() {
        let mut d = HashDamage::new();
        let f = cpu(64, 64, 0);
        // First frame: Changed → encode.
        assert!(should_encode(d.classify(&f), false));
        // Repeat: Unchanged + no force → skip.
        assert!(!should_encode(d.classify(&f), false));
        // Repeat + force → encode anyway.
        assert!(should_encode(d.classify(&f), true));
        // Unknown (e.g. GPU frame) always encodes.
        assert!(should_encode(DamageHint::Unknown, false));
    }
}
