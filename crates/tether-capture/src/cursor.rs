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
        assert_eq!(s.next_event(), CursorEvent::Idle, "Idle is sticky once emitted");
    }
}
