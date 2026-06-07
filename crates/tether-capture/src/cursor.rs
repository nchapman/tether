//! Cursor sprite capture seam.
//!
//! The host originates two kinds of cursor information:
//!
//! - **Position**, on the unreliable `HostCursorPacket::Position`
//!   datagram channel — high-rate, latest-wins.
//! - **Shape** (the pointer sprite itself: pixels + hotspot), on the
//!   reliable control stream via `ControlMessage::CursorShape` and
//!   `ControlMessage::CursorUseShape`. Clients cache shapes by id so
//!   recurring cursors (text-beam, hand, arrow) ride the wire once.
//!
//! This module owns the trait that produces those events. The host
//! send-side pump polls a [`CursorSource`] each tick and forwards
//! events onto the wire. Backends are swapped per platform via
//! compile-time `#[cfg]` selection — no runtime "discover the
//! cursor API" step.

// Cursor sprite geometry: hotspot/dimension casts (f64 screen coords → u32,
// u32 dims → u16/u8 wire fields) are intentional and range-bounded.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use tether_protocol::control::MAX_CURSOR_SHAPE_BYTES;
use tether_protocol::cursor::CursorPixelFormat;

/// One change to the host's cursor sprite. `id` is a stable hash of
/// the pixel bytes so a repeated cursor (text-beam after editing
/// text, hand after hovering a link) doesn't re-send the same pixels
/// — the client cache key matches by `id` and the host emits
/// `CursorUseShape { id }` instead. Position is **not** carried here;
/// it rides the existing unreliable cursor datagram channel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CursorShapeEvent {
    pub id: u64,
    pub width: u16,
    pub height: u16,
    pub hotspot: (u16, u16),
    pub format: CursorPixelFormat,
    pub pixels: Vec<u8>,
}

/// One observation a [`CursorSource`] can produce.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CursorEvent {
    /// New sprite (pixels + hotspot). Caller sends
    /// `ControlMessage::CursorShape` if the id is unseen, else
    /// `ControlMessage::CursorUseShape`.
    Shape(CursorShapeEvent),
    /// Source has no more events to produce right now. Callers may
    /// poll again later.
    Idle,
}

/// Latest pointer position snapshot the host's send pump reads on each
/// tick. `(x, y)` are captured-frame pixels in the same coordinate
/// space as the encoded video. `visible = false` means the pointer is
/// off-screen / hidden and the client should not draw the overlay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CursorPosition {
    pub x: i32,
    pub y: i32,
    pub visible: bool,
}

/// Per-platform cursor source. Implementations are typically I/O-
/// driven (PipeWire buffer callbacks, SCK delegate, X11 event loop)
/// and own their own state — the trait stays minimal so a future
/// impl can do whatever bookkeeping fits its API surface.
pub trait CursorSource: Send {
    /// Returns the next pending shape / lifecycle event without
    /// blocking. Backends that have nothing buffered yield
    /// [`CursorEvent::Idle`] — that means "try again later," not "done
    /// forever." Named `next_event` rather than `poll` to keep it out
    /// of the `Future::poll` idiom space, which carries different
    /// non-blocking semantics.
    fn next_event(&mut self) -> CursorEvent;

    /// Snapshot of the latest known pointer position, if the backend
    /// can produce one. Latest-wins: the host pump reads this once
    /// per send tick and emits a single `HostCursorPacket::Position`
    /// datagram, so backends that update at sub-frame rate must
    /// collapse internally rather than emit a queue here.
    ///
    /// Default returns `None`, which the placeholder + any future
    /// position-less source can take unchanged.
    fn poll_position(&mut self) -> Option<CursorPosition> {
        None
    }
}

/// Stub source matching the pre-trait inline behaviour: one
/// 16×16 checkerboard at startup, then [`CursorEvent::Idle`] forever.
/// Hosts swap in a real platform impl when one is wired.
#[derive(Debug, Default)]
pub struct PlaceholderCursorSource {
    emitted_initial: bool,
}

impl PlaceholderCursorSource {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl CursorSource for PlaceholderCursorSource {
    fn next_event(&mut self) -> CursorEvent {
        if self.emitted_initial {
            return CursorEvent::Idle;
        }
        self.emitted_initial = true;
        let pixels: Vec<u8> = (0..16 * 16)
            .flat_map(|i: i32| {
                let on = ((i / 16) + (i % 16)) % 2 == 0;
                let v: u8 = if on { 0xFF } else { 0x00 };
                [v, v, v, 0xFF]
            })
            .collect();
        CursorEvent::Shape(CursorShapeEvent {
            // Stable id for the placeholder. Real impls use a hash of
            // pixel bytes so an identical sprite reuses an id across
            // reconnects.
            id: 0,
            width: 16,
            height: 16,
            hotspot: (0, 0),
            format: CursorPixelFormat::Rgba8,
            pixels,
        })
    }
}

/// Resample a [`CursorShapeEvent`] from its source coordinate frame
/// (capture pixels) into a different frame (e.g. encode pixels) using
/// a box filter, preserving aspect ratio. Both dimensions scale by
/// the smaller of the two axis ratios so the rendered sprite keeps
/// its proportions. Hotspot and id are rescaled / recomputed
/// accordingly.
///
/// The host pump calls this before serializing `CursorShape` /
/// `Position` so the wire payload is already in the coordinate frame
/// the client's overlay renders against (the client only knows the
/// decoded-video pixel dims, not the host's capture dims). Skipping
/// this on a macOS Retina host that downscales 3024×1952 → 1104×720
/// at the encoder produces a sprite drawn 2.74× too large.
///
/// `(src_w, src_h)` and `(dst_w, dst_h)` are the source and target
/// frame dims. The returned shape's width/height are the input dims
/// scaled by `dst/src` (per-axis, then unified to the smaller ratio
/// to preserve aspect). Returns `shape` unchanged if `src == dst` or
/// if any dim is zero.
#[must_use]
pub fn rescale_shape_to_frame(
    shape: CursorShapeEvent,
    src: (u32, u32),
    dst: (u32, u32),
) -> CursorShapeEvent {
    if src == dst
        || src.0 == 0
        || src.1 == 0
        || dst.0 == 0
        || dst.1 == 0
        || shape.width == 0
        || shape.height == 0
    {
        return shape;
    }
    let Some(expected_len) = expected_rgba_len(shape.width, shape.height) else {
        return shape;
    };
    if shape.pixels.len() != expected_len {
        return shape;
    }

    let ratio_x = f64::from(dst.0) / f64::from(src.0);
    let ratio_y = f64::from(dst.1) / f64::from(src.1);
    // Use the smaller ratio so the sprite stays in the displayed
    // letterbox rect on both axes — Apple cursors aren't square.
    let ratio = ratio_x.min(ratio_y);
    if (ratio - 1.0).abs() < 1e-6 {
        return shape;
    }
    let (new_w, new_h) = scaled_shape_dimensions(shape.width, shape.height, ratio);
    let new_pixels =
        crate::cursor::box_downsample_rgba(shape.width, shape.height, &shape.pixels, new_w, new_h);
    let new_hot = (
        scale_hotspot(shape.hotspot.0, f64::from(new_w) / f64::from(shape.width)),
        scale_hotspot(shape.hotspot.1, f64::from(new_h) / f64::from(shape.height)),
    );
    // Recompute the id from the rescaled pixels: the wire id must
    // uniquely identify the *exact bitmap* the client will cache. If
    // we kept the source id, a mid-session viewport change that
    // rebuilds the encoder at new dims would re-rescale the same
    // source cursor to a different bitmap — but the pump would see
    // the same id, fail its dedup check, and only send
    // `CursorUseShape{id}` referencing the stale cached bitmap.
    CursorShapeEvent {
        id: fnv1a_64(new_w, new_h, &new_pixels),
        width: new_w,
        height: new_h,
        hotspot: new_hot,
        format: shape.format,
        pixels: new_pixels,
    }
}

fn expected_rgba_len(width: u16, height: u16) -> Option<usize> {
    usize::from(width)
        .checked_mul(usize::from(height))?
        .checked_mul(4)
}

fn scale_dimension(value: u16, ratio: f64) -> u16 {
    if !ratio.is_finite() || ratio <= 0.0 {
        return 1;
    }
    let scaled = (f64::from(value) * ratio)
        .round()
        .clamp(1.0, f64::from(u16::MAX));
    scaled as u16
}

fn scaled_shape_dimensions(width: u16, height: u16, ratio: f64) -> (u16, u16) {
    let mut width = scale_dimension(width, ratio);
    let mut height = scale_dimension(height, ratio);
    if expected_rgba_len(width, height).is_some_and(|len| len <= MAX_CURSOR_SHAPE_BYTES) {
        return (width, height);
    }

    let max_pixels = MAX_CURSOR_SHAPE_BYTES / 4;
    let pixels = f64::from(width) * f64::from(height);
    let cap_ratio = ((max_pixels as f64) / pixels).sqrt().min(1.0);
    width = scale_dimension(width, cap_ratio);
    height = scale_dimension(height, cap_ratio);

    while expected_rgba_len(width, height).is_none_or(|len| len > MAX_CURSOR_SHAPE_BYTES) {
        if width >= height && width > 1 {
            width -= 1;
        } else if height > 1 {
            height -= 1;
        } else {
            break;
        }
    }

    (width, height)
}

fn scale_hotspot(value: u16, ratio: f64) -> u16 {
    if !ratio.is_finite() || ratio <= 0.0 {
        return 0;
    }
    let scaled = (f64::from(value) * ratio)
        .round()
        .clamp(0.0, f64::from(u16::MAX));
    scaled as u16
}

/// FNV-1a 64-bit over `(width, height, pixel bytes)`. Wire-stable
/// (independent of `std::hash::Hasher`'s explicitly non-stable
/// output) so the same bitmap maps to the same id across reconnects
/// and across host builds. Used both by the platform cursor sources
/// (initial hash of extracted pixels) and by [`rescale_shape_to_frame`]
/// (re-hash after dim conversion). Crate-local: every wire id must
/// flow through this hasher with the same byte order + dims prefix.
#[must_use]
pub(crate) fn fnv1a_64(width: u16, height: u16, pixels: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in width
        .to_le_bytes()
        .iter()
        .chain(height.to_le_bytes().iter())
        .chain(pixels.iter())
    {
        h ^= u64::from(*byte);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Simple area-average box downsampler for RGBA8 with premultiplied
/// averaging so a transparent edge doesn't bleed colour from the
/// underlying RGB. Used by both the cursor source backends (sanity
/// cap on bitmap dims) and the host pump (capture-to-encode rescale).
/// Crate-local — external callers should go through
/// [`rescale_shape_to_frame`].
#[must_use]
pub(crate) fn box_downsample_rgba(
    src_w: u16,
    src_h: u16,
    src: &[u8],
    dst_w: u16,
    dst_h: u16,
) -> Vec<u8> {
    let src_w = src_w as usize;
    let src_h = src_h as usize;
    let dst_w_us = dst_w as usize;
    let dst_h_us = dst_h as usize;
    let mut out = vec![0u8; dst_w_us * dst_h_us * 4];
    if src_w == 0 || src_h == 0 || dst_w_us == 0 || dst_h_us == 0 {
        return out;
    }
    let Some(expected_src_len) = src_w
        .checked_mul(src_h)
        .and_then(|pixels| pixels.checked_mul(4))
    else {
        return out;
    };
    if src.len() < expected_src_len {
        return out;
    }
    for dy in 0..dst_h_us {
        let sy0 = dy * src_h / dst_h_us;
        let sy1 = ((dy + 1) * src_h / dst_h_us).max(sy0 + 1).min(src_h);
        for dx in 0..dst_w_us {
            let sx0 = dx * src_w / dst_w_us;
            let sx1 = ((dx + 1) * src_w / dst_w_us).max(sx0 + 1).min(src_w);
            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;
            let mut a_sum = 0u32;
            let mut n = 0u32;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    let i = (sy * src_w + sx) * 4;
                    let a = u32::from(src[i + 3]);
                    r_sum += u32::from(src[i]) * a;
                    g_sum += u32::from(src[i + 1]) * a;
                    b_sum += u32::from(src[i + 2]) * a;
                    a_sum += a;
                    n += 1;
                }
            }
            let o = (dy * dst_w_us + dx) * 4;
            if a_sum == 0 || n == 0 {
                out[o] = 0;
                out[o + 1] = 0;
                out[o + 2] = 0;
                out[o + 3] = 0;
            } else {
                out[o] = (r_sum / a_sum) as u8;
                out[o + 1] = (g_sum / a_sum) as u8;
                out[o + 2] = (b_sum / a_sum) as u8;
                out[o + 3] = (a_sum / n) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_emits_one_shape_then_idle() {
        let mut s = PlaceholderCursorSource::new();
        match s.next_event() {
            CursorEvent::Shape(shape) => {
                assert_eq!(shape.width, 16);
                assert_eq!(shape.height, 16);
                assert_eq!(shape.pixels.len(), 16 * 16 * 4);
            }
            CursorEvent::Idle => panic!("first call must emit Shape"),
        }
        assert_eq!(s.next_event(), CursorEvent::Idle);
        assert_eq!(
            s.next_event(),
            CursorEvent::Idle,
            "Idle is sticky once emitted"
        );
    }

    #[test]
    fn rescale_shape_with_malformed_pixels_returns_unchanged() {
        let shape = CursorShapeEvent {
            id: 7,
            width: 4,
            height: 4,
            hotspot: (1, 1),
            format: CursorPixelFormat::Rgba8,
            pixels: vec![0; 4 * 4 * 4 - 1],
        };

        let out = rescale_shape_to_frame(shape.clone(), (100, 100), (50, 50));

        assert_eq!(out, shape);
    }

    #[test]
    fn rescale_shape_with_zero_shape_dimension_returns_unchanged() {
        let shape = CursorShapeEvent {
            id: 7,
            width: 0,
            height: 4,
            hotspot: (0, 0),
            format: CursorPixelFormat::Rgba8,
            pixels: Vec::new(),
        };

        let out = rescale_shape_to_frame(shape.clone(), (100, 100), (50, 50));

        assert_eq!(out, shape);
    }

    #[test]
    fn rescale_shape_caps_oversized_payload() {
        let shape = CursorShapeEvent {
            id: 7,
            width: 2,
            height: 2,
            hotspot: (2, 2),
            format: CursorPixelFormat::Rgba8,
            pixels: vec![0xFF; 2 * 2 * 4],
        };

        let out = rescale_shape_to_frame(shape, (1, 1), (100_000, 100_000));

        assert_eq!(out.width, 128);
        assert_eq!(out.height, 128);
        assert_eq!(out.hotspot, (128, 128));
        assert_eq!(out.pixels.len(), MAX_CURSOR_SHAPE_BYTES);
    }
}
