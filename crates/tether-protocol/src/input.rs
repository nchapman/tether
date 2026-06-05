//! Input events flowing client → host on the reliable input stream.

use crate::{pb, CodecError, MonoNanos, ReliableMessage};
use prost::Message as _;
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

fn mono_to_pb(value: MonoNanos) -> pb::MonoNanos {
    pb::MonoNanos { value: value.0 }
}

fn mono_from_pb(value: Option<pb::MonoNanos>) -> Result<MonoNanos, CodecError> {
    Ok(MonoNanos(
        value
            .ok_or(CodecError::Wire("missing input timestamp"))?
            .value,
    ))
}

fn modifiers_to_pb(value: Modifiers) -> pb::Modifiers {
    pb::Modifiers {
        shift: value.shift,
        ctrl: value.ctrl,
        alt: value.alt,
        meta: value.meta,
    }
}

fn modifiers_from_pb(value: Option<pb::Modifiers>) -> Modifiers {
    value.map_or_else(Modifiers::default, |v| Modifiers {
        shift: v.shift,
        ctrl: v.ctrl,
        alt: v.alt,
        meta: v.meta,
    })
}

fn key_event_to_pb(key: HidUsage, modifiers: Modifiers) -> pb::KeyEvent {
    pb::KeyEvent {
        key: Some(pb::HidUsage { value: key.0 }),
        modifiers: Some(modifiers_to_pb(modifiers)),
    }
}

fn key_event_from_pb(value: pb::KeyEvent) -> Result<(HidUsage, Modifiers), CodecError> {
    Ok((
        HidUsage(
            value
                .key
                .ok_or(CodecError::Wire("missing HID usage"))?
                .value,
        ),
        modifiers_from_pb(value.modifiers),
    ))
}

fn mouse_button_to_pb(value: MouseButton) -> i32 {
    match value {
        MouseButton::Left => 1,
        MouseButton::Right => 2,
        MouseButton::Middle => 3,
        MouseButton::X1 => 4,
        MouseButton::X2 => 5,
    }
}

fn mouse_button_from_pb(value: i32) -> Result<MouseButton, CodecError> {
    match value {
        1 => Ok(MouseButton::Left),
        2 => Ok(MouseButton::Right),
        3 => Ok(MouseButton::Middle),
        4 => Ok(MouseButton::X1),
        5 => Ok(MouseButton::X2),
        _ => Err(CodecError::Wire("unknown MouseButton")),
    }
}

fn scroll_kind_to_pb(value: ScrollKind) -> i32 {
    match value {
        ScrollKind::Line => 1,
        ScrollKind::Pixel => 2,
    }
}

fn scroll_kind_from_pb(value: i32) -> Result<ScrollKind, CodecError> {
    match value {
        1 => Ok(ScrollKind::Line),
        2 => Ok(ScrollKind::Pixel),
        _ => Err(CodecError::Wire("unknown ScrollKind")),
    }
}

impl ReliableMessage for InputEvent {
    fn encode_reliable(&self) -> Vec<u8> {
        use pb::input_event::Kind;
        let kind = match self.kind.clone() {
            InputEventKind::KeyDown { key, modifiers } => {
                Kind::KeyDown(key_event_to_pb(key, modifiers))
            }
            InputEventKind::KeyUp { key, modifiers } => {
                Kind::KeyUp(key_event_to_pb(key, modifiers))
            }
            InputEventKind::Text { utf8 } => Kind::Text(pb::TextInput { utf8 }),
            InputEventKind::MouseButton {
                button,
                pressed,
                modifiers,
            } => Kind::MouseButton(pb::MouseButtonEvent {
                button: mouse_button_to_pb(button),
                pressed,
                modifiers: Some(modifiers_to_pb(modifiers)),
            }),
            InputEventKind::MouseScroll {
                dx,
                dy,
                kind,
                modifiers,
            } => Kind::MouseScroll(pb::MouseScroll {
                dx,
                dy,
                kind: scroll_kind_to_pb(kind),
                modifiers: Some(modifiers_to_pb(modifiers)),
            }),
            InputEventKind::RelativeMouseMove { dx, dy, modifiers } => {
                Kind::RelativeMouseMove(pb::RelativeMouseMove {
                    dx: i32::from(dx),
                    dy: i32::from(dy),
                    modifiers: Some(modifiers_to_pb(modifiers)),
                })
            }
        };
        pb::InputEvent {
            event_id: self.event_id,
            t_client: Some(mono_to_pb(self.t_client)),
            device_id: u32::from(self.device_id),
            kind: Some(kind),
        }
        .encode_to_vec()
    }

    fn decode_reliable(bytes: &[u8]) -> Result<Self, CodecError> {
        use pb::input_event::Kind;
        let value = pb::InputEvent::decode(bytes)?;
        let kind = value
            .kind
            .ok_or(CodecError::Wire("missing InputEvent kind"))?;
        let kind = match kind {
            Kind::KeyDown(v) => {
                let (key, modifiers) = key_event_from_pb(v)?;
                InputEventKind::KeyDown { key, modifiers }
            }
            Kind::KeyUp(v) => {
                let (key, modifiers) = key_event_from_pb(v)?;
                InputEventKind::KeyUp { key, modifiers }
            }
            Kind::Text(v) => InputEventKind::Text { utf8: v.utf8 },
            Kind::MouseButton(v) => InputEventKind::MouseButton {
                button: mouse_button_from_pb(v.button)?,
                pressed: v.pressed,
                modifiers: modifiers_from_pb(v.modifiers),
            },
            Kind::MouseScroll(v) => InputEventKind::MouseScroll {
                dx: v.dx,
                dy: v.dy,
                kind: scroll_kind_from_pb(v.kind)?,
                modifiers: modifiers_from_pb(v.modifiers),
            },
            Kind::RelativeMouseMove(v) => InputEventKind::RelativeMouseMove {
                dx: i16::try_from(v.dx).map_err(|_| CodecError::Wire("relative dx > i16"))?,
                dy: i16::try_from(v.dy).map_err(|_| CodecError::Wire("relative dy > i16"))?,
                modifiers: modifiers_from_pb(v.modifiers),
            },
        };
        Ok(InputEvent {
            event_id: value.event_id,
            t_client: mono_from_pb(value.t_client)?,
            device_id: u8::try_from(value.device_id)
                .map_err(|_| CodecError::Wire("input device_id > u8"))?,
            kind,
        })
    }
}
