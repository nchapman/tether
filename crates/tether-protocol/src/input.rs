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
    KeyDown {
        key: HidUsage,
        modifiers: Modifiers,
    },
    KeyUp {
        key: HidUsage,
        modifiers: Modifiers,
    },
    /// Layout-/IME-resolved character(s) to type as a unit, complementing
    /// the HID path. The client emits this when it has the actual text
    /// the user produced — IME composition results, dead-key sequences,
    /// AltGr letters, and ordinary typing where a layout-aware host
    /// `type-text` operation is more correct than reconstructing the
    /// string from physical keycodes. Hosts apply the current keymap;
    /// shortcuts (Ctrl+C etc.) keep going through `KeyDown`/`KeyUp`.
    Text {
        utf8: String,
    },
    // Mouse-position events live on the cursor datagram channel
    // (`tether_protocol::cursor::ClientCursorPacket`), not here, so a
    // queue of keystrokes can't head-of-line-block the pointer. The
    // input stream stays reliable + ordered for things that *must*
    // arrive (key state, button clicks, scroll deltas).
    //
    // Mouse events carry a `modifiers` snapshot for the same reason
    // key events do: lets the host reconcile against whatever it
    // thinks is held and avoid drift after a dropped event or a
    // focus-loss-induced state reset. Shift-click selection-extend,
    // Ctrl-scroll zoom, and Cmd-click on macOS all rely on this.
    MouseButton {
        button: MouseButton,
        pressed: bool,
        modifiers: Modifiers,
    },
    MouseScroll {
        dx: f32,
        dy: f32,
        kind: ScrollKind,
        modifiers: Modifiers,
    },
    /// Device-level mouse delta in pixels. Rides on the reliable
    /// input stream (not the cursor datagram channel) so deltas
    /// can't be dropped or reordered relative to button events,
    /// which breaks every recenter-loop / raw-input game otherwise.
    ///
    /// The client emits these only while
    /// [`crate::control::CursorMode::Relative`] is active; the
    /// host dispatches them via `enigo::move_mouse(_, _, Rel)`
    /// which routes through the OS's native delta API (uinput
    /// EV_REL on Linux, CGEvent delta fields on macOS,
    /// SendInput-without-ABSOLUTE on Windows).
    RelativeMouseMove {
        dx: i16,
        dy: i16,
        modifiers: Modifiers,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputEvent {
    /// Monotonically increasing per connection. Echoed by the host via
    /// [`crate::video::InputEchoBatch`] in the next video frame whose
    /// captured content reflects this event.
    pub event_id: u64,
    /// Client-monotonic time at which the OS delivered the event.
    pub t_client: MonoNanos,
    /// Identifies which input device the event came from. `0` is the
    /// implicit "primary" keyboard/mouse pair, which is what every
    /// `KeyDown`/`MouseScroll`/etc. event uses today. Reserved here so
    /// future device kinds (a second gamepad, pen, touchpoint) can ride
    /// the same wire shape — gamepad rumble going back to the client
    /// also keys off this id (see `tether.gamepad-rumble` extension in
    /// `control.rs`).
    pub device_id: u8,
    pub kind: InputEventKind,
}
