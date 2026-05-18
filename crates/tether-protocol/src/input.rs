//! Input events flowing client → host on the reliable input stream.

use crate::MonoNanos;
use serde::{Deserialize, Serialize};

/// HID Usage page + usage ID packed as a single u32 (`(page << 16) | usage`).
/// We prefer HID over OS-native scancodes so the wire format is platform
/// independent and the host's input injector maps to its native scheme.
///
/// See <https://usb.org/document-library/hid-usage-tables>.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HidUsage(pub u32);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    X1,
    X2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScrollKind {
    /// Discrete notched scroll wheel.
    Line,
    /// Continuous high-resolution scroll (trackpad / Magic Mouse).
    Pixel,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InputEventKind {
    KeyDown { key: HidUsage, modifiers: Modifiers },
    KeyUp { key: HidUsage, modifiers: Modifiers },
    /// Absolute mouse position normalised to `[0.0, 1.0]` along each axis of
    /// the client's rendered video surface. The host scales to its own
    /// display resolution before injection.
    MousePosition { x: f32, y: f32 },
    MouseButton { button: MouseButton, pressed: bool },
    MouseScroll { dx: f32, dy: f32, kind: ScrollKind },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputEvent {
    /// Monotonically increasing per connection. Echoed by the host via
    /// [`crate::video::InputEchoBatch`] in the next video frame whose
    /// captured content reflects this event.
    pub event_id: u64,
    /// Client-monotonic time at which the OS delivered the event.
    pub t_client: MonoNanos,
    pub kind: InputEventKind,
}
