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

/// Modifier-key state at the time the event was captured.
///
/// Sent on every key event (rather than inferred from key down/up of
/// modifier keys) so the host doesn't have to reconstruct modifier state
/// from the event stream. Belt-and-braces against dropped reorders on the
/// input stream and against the client's modifier state diverging from the
/// host's after a focus change or hotkey escape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    /// Layout-/IME-resolved character(s) to type as a unit, complementing
    /// the HID path. The client emits this when it has the actual text
    /// the user produced — IME composition results, dead-key sequences,
    /// AltGr letters, and ordinary typing where a layout-aware host
    /// `type-text` operation is more correct than reconstructing the
    /// string from physical keycodes. Hosts apply the current keymap;
    /// shortcuts (Ctrl+C etc.) keep going through `KeyDown`/`KeyUp`.
    Text { utf8: String },
    /// Absolute mouse position normalised to `[0.0, 1.0]` along each axis of
    /// the client's rendered video surface (the area showing host pixels,
    /// excluding any letterbox bars introduced by an aspect-ratio mismatch
    /// between the client window and the host display). The host scales
    /// these normalised coordinates to its own display resolution before
    /// injection. Clients must clamp to `[0.0, 1.0]` and suppress events
    /// whose pointer falls outside the video region. `display_idx`
    /// addresses a specific display in a multi-monitor host setup;
    /// single-display hosts and clients that don't know about the
    /// distinction send `0` (primary).
    MousePosition {
        display_idx: u8,
        x: f32,
        y: f32,
    },
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
