//! Input plumbing: translate render-layer window events into wire-level
//! [`InputEvent`]s on the client (`WinitTranslator`), and inject those
//! events into the host's native input system (`inject::Injector`).
//!
//! Backends are picked per-target by [`inject::default_injector`]; today
//! that's libei on Linux (portal-mediated, Wayland-native) and a noop
//! everywhere else. Adding macOS / Windows means another backend behind
//! the same trait.

pub mod inject;

pub use inject::{InjectError, Injector, NoopInjector};

use tether_protocol::cursor::ClientCursorPacket;
use tether_protocol::input::{
    HidUsage, InputEvent, InputEventKind, Modifiers, MouseButton, ScrollKind,
};
use tether_protocol::MonoNanos;
use tether_render::{KeyCode, ModifiersState, RenderEvent};

/// One translated event ready to ship over the wire. The two arms map
/// to the two distinct wire channels they ride: `Input` goes on the
/// reliable input stream, `Cursor` goes on the unreliable cursor
/// datagram channel. Callers must respect the split — sending a
/// `Cursor` over the input stream would defeat its whole purpose.
#[derive(Clone, Debug)]
pub enum WireEvent {
    Input(InputEvent),
    Cursor(ClientCursorPacket),
}

/// HID Usage Page 0x07 is "Keyboard / Keypad". Bit-packed with the usage
/// id into a single `u32` per `HidUsage`'s contract: `(page << 16) | usage`.
const HID_PAGE_KEYBOARD: u32 = 0x07;

fn hid_keyboard(usage: u32) -> HidUsage {
    HidUsage((HID_PAGE_KEYBOARD << 16) | usage)
}

/// Stateful translator from [`RenderEvent`]s (window-layer) to
/// [`InputEvent`]s (wire-layer). Tracks modifier state and an incrementing
/// event id so the host can echo per-event timing back via
/// [`tether_protocol::video::InputEchoBatch`].
///
/// One translator per connection. Not Sync — single owner consumes the
/// render event stream sequentially.
pub struct WinitTranslator {
    modifiers: Modifiers,
    next_event_id: u64,
    /// Per-connection cursor sequence number. Datagrams reorder
    /// freely; the host uses this to drop late-arriving positions
    /// (a stale x,y after a fresher one would visibly snap the
    /// pointer backwards).
    next_cursor_seq: u32,
    /// Last cursor position inside the video region, normalised to
    /// `[0,1]^2`. We remember it across mouse-button and mouse-wheel
    /// events because winit delivers button/scroll events without
    /// coordinates — the host needs the most recent position to know
    /// *where* a click landed.
    last_cursor: Option<(f32, f32)>,
}

impl Default for WinitTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl WinitTranslator {
    pub fn new() -> Self {
        Self {
            modifiers: Modifiers::default(),
            next_event_id: 0,
            next_cursor_seq: 0,
            last_cursor: None,
        }
    }

    /// Consume one render event and produce zero or more wire events.
    /// Most render events translate 1:1; modifier/focus updates are
    /// pure internal state changes and return an empty Vec. Cursor
    /// motion produces `WireEvent::Cursor` (datagram path); everything
    /// else produces `WireEvent::Input` (reliable input stream).
    pub fn translate(&mut self, event: RenderEvent) -> Vec<WireEvent> {
        let kinds = match event {
            RenderEvent::Modifiers(state) => {
                self.modifiers = modifiers_from_winit(state);
                return Vec::new();
            }
            RenderEvent::Focused(focused) => {
                if !focused {
                    // Forget cursor + modifier state on focus loss so
                    // the host doesn't end up with a sticky modifier
                    // or a click that landed on whatever was focused
                    // when we re-enter the window. Tell the host the
                    // pointer left the video region too (visible=false)
                    // so its synthetic cursor disappears.
                    self.modifiers = Modifiers::default();
                    let pkt = self.next_cursor_packet(0.0, 0.0, false);
                    self.last_cursor = None;
                    return vec![WireEvent::Cursor(pkt)];
                }
                return Vec::new();
            }
            RenderEvent::Cursor { video_normalized } => {
                self.last_cursor = video_normalized;
                let pkt = match video_normalized {
                    Some((x, y)) => self.next_cursor_packet(x, y, true),
                    None => self.next_cursor_packet(0.0, 0.0, false),
                };
                return vec![WireEvent::Cursor(pkt)];
            }
            RenderEvent::Key {
                code,
                pressed,
                repeat,
                text,
            } => {
                if repeat {
                    // The host's OS will generate its own auto-repeats
                    // from a single keydown; forwarding both ends up
                    // looking like double-typing.
                    return Vec::new();
                }
                // Text path: when the OS resolved this keypress to a
                // string and no shortcut-style modifier is held, send
                // the text and skip the HID emit. This is what lets
                // IME, dead keys, AltGr, and layout-mismatched typing
                // round-trip correctly. We still emit on key-up via the
                // HID path so the host can release any held HID keys
                // that the operator typed before switching to a text-
                // producing keystroke (rare, but cheap).
                if pressed {
                    if let Some(utf8) = text {
                        if !utf8.is_empty() && !uses_shortcut_modifier(self.modifiers) {
                            return self.wrap(vec![InputEventKind::Text { utf8 }]);
                        }
                    }
                }
                match keycode_to_hid(code) {
                    Some(key) => vec![if pressed {
                        InputEventKind::KeyDown {
                            key,
                            modifiers: self.modifiers,
                        }
                    } else {
                        InputEventKind::KeyUp {
                            key,
                            modifiers: self.modifiers,
                        }
                    }],
                    None => {
                        tracing::trace!(?code, "no HID mapping for KeyCode; dropping");
                        return Vec::new();
                    }
                }
            }
            RenderEvent::MouseButton { button, pressed } => {
                let button = match map_mouse_button(button) {
                    Some(b) => b,
                    None => {
                        tracing::trace!(?button, "unrecognised mouse button; dropping");
                        return Vec::new();
                    }
                };
                vec![InputEventKind::MouseButton {
                    button,
                    pressed,
                    modifiers: self.modifiers,
                }]
            }
            RenderEvent::Scroll { dx, dy, by_line } => {
                vec![InputEventKind::MouseScroll {
                    dx,
                    dy,
                    kind: if by_line {
                        ScrollKind::Line
                    } else {
                        ScrollKind::Pixel
                    },
                    modifiers: self.modifiers,
                }]
            }
        };
        self.wrap(kinds)
    }

    fn wrap(&mut self, kinds: Vec<InputEventKind>) -> Vec<WireEvent> {
        kinds
            .into_iter()
            .map(|kind| {
                let evt = InputEvent {
                    event_id: self.next_event_id,
                    t_client: MonoNanos::now(),
                    device_id: 0,
                    kind,
                };
                self.next_event_id = self.next_event_id.wrapping_add(1);
                WireEvent::Input(evt)
            })
            .collect()
    }

    fn next_cursor_packet(&mut self, x: f32, y: f32, visible: bool) -> ClientCursorPacket {
        let pkt = ClientCursorPacket {
            seq: self.next_cursor_seq,
            display_idx: 0,
            x,
            y,
            visible,
            t_client: MonoNanos::now(),
        };
        self.next_cursor_seq = self.next_cursor_seq.wrapping_add(1);
        pkt
    }
}

/// True when a "shortcut" modifier is held — Ctrl, Alt (without Shift's
/// usual layout overlap), or Meta. We deliberately exclude bare Shift
/// because Shift+letter is just capitalisation, which the text path
/// handles correctly via the OS-resolved string.
fn uses_shortcut_modifier(m: Modifiers) -> bool {
    m.ctrl || m.alt || m.meta
}

fn modifiers_from_winit(state: ModifiersState) -> Modifiers {
    Modifiers {
        shift: state.shift_key(),
        ctrl: state.control_key(),
        alt: state.alt_key(),
        meta: state.super_key(),
    }
}

fn map_mouse_button(b: tether_render::MouseButton) -> Option<MouseButton> {
    use tether_render::MouseButton as Wb;
    Some(match b {
        Wb::Left => MouseButton::Left,
        Wb::Right => MouseButton::Right,
        Wb::Middle => MouseButton::Middle,
        Wb::Back => MouseButton::X1,
        Wb::Forward => MouseButton::X2,
        Wb::Other(_) => return None,
    })
}

/// Map winit physical `KeyCode` to HID Keyboard usage. Coverage is
/// intentionally partial: enough to type English text, navigate, and
/// hit common shortcuts (modifiers + arrows + function keys). Unmapped
/// keys are dropped with a trace-level log so the long tail (media keys,
/// browser shortcut keys, numpad operators) can be added on demand.
///
/// Reference: USB HID Usage Tables, section 10 (Keyboard/Keypad).
fn keycode_to_hid(code: KeyCode) -> Option<HidUsage> {
    let u: u32 = match code {
        // Letters
        KeyCode::KeyA => 0x04,
        KeyCode::KeyB => 0x05,
        KeyCode::KeyC => 0x06,
        KeyCode::KeyD => 0x07,
        KeyCode::KeyE => 0x08,
        KeyCode::KeyF => 0x09,
        KeyCode::KeyG => 0x0A,
        KeyCode::KeyH => 0x0B,
        KeyCode::KeyI => 0x0C,
        KeyCode::KeyJ => 0x0D,
        KeyCode::KeyK => 0x0E,
        KeyCode::KeyL => 0x0F,
        KeyCode::KeyM => 0x10,
        KeyCode::KeyN => 0x11,
        KeyCode::KeyO => 0x12,
        KeyCode::KeyP => 0x13,
        KeyCode::KeyQ => 0x14,
        KeyCode::KeyR => 0x15,
        KeyCode::KeyS => 0x16,
        KeyCode::KeyT => 0x17,
        KeyCode::KeyU => 0x18,
        KeyCode::KeyV => 0x19,
        KeyCode::KeyW => 0x1A,
        KeyCode::KeyX => 0x1B,
        KeyCode::KeyY => 0x1C,
        KeyCode::KeyZ => 0x1D,
        // Digits — HID orders them 1..9, 0 (yes, in that order).
        KeyCode::Digit1 => 0x1E,
        KeyCode::Digit2 => 0x1F,
        KeyCode::Digit3 => 0x20,
        KeyCode::Digit4 => 0x21,
        KeyCode::Digit5 => 0x22,
        KeyCode::Digit6 => 0x23,
        KeyCode::Digit7 => 0x24,
        KeyCode::Digit8 => 0x25,
        KeyCode::Digit9 => 0x26,
        KeyCode::Digit0 => 0x27,
        // Named keys
        KeyCode::Enter => 0x28,
        KeyCode::Escape => 0x29,
        KeyCode::Backspace => 0x2A,
        KeyCode::Tab => 0x2B,
        KeyCode::Space => 0x2C,
        KeyCode::Minus => 0x2D,
        KeyCode::Equal => 0x2E,
        KeyCode::BracketLeft => 0x2F,
        KeyCode::BracketRight => 0x30,
        KeyCode::Backslash => 0x31,
        KeyCode::Semicolon => 0x33,
        KeyCode::Quote => 0x34,
        KeyCode::Backquote => 0x35,
        KeyCode::Comma => 0x36,
        KeyCode::Period => 0x37,
        KeyCode::Slash => 0x38,
        KeyCode::CapsLock => 0x39,
        // Function row
        KeyCode::F1 => 0x3A,
        KeyCode::F2 => 0x3B,
        KeyCode::F3 => 0x3C,
        KeyCode::F4 => 0x3D,
        KeyCode::F5 => 0x3E,
        KeyCode::F6 => 0x3F,
        KeyCode::F7 => 0x40,
        KeyCode::F8 => 0x41,
        KeyCode::F9 => 0x42,
        KeyCode::F10 => 0x43,
        KeyCode::F11 => 0x44,
        KeyCode::F12 => 0x45,
        // Navigation
        KeyCode::PrintScreen => 0x46,
        KeyCode::ScrollLock => 0x47,
        KeyCode::Pause => 0x48,
        KeyCode::Insert => 0x49,
        KeyCode::Home => 0x4A,
        KeyCode::PageUp => 0x4B,
        KeyCode::Delete => 0x4C,
        KeyCode::End => 0x4D,
        KeyCode::PageDown => 0x4E,
        KeyCode::ArrowRight => 0x4F,
        KeyCode::ArrowLeft => 0x50,
        KeyCode::ArrowDown => 0x51,
        KeyCode::ArrowUp => 0x52,
        // Numpad / keypad cluster
        KeyCode::NumLock => 0x53,
        KeyCode::NumpadDivide => 0x54,
        KeyCode::NumpadMultiply => 0x55,
        KeyCode::NumpadSubtract => 0x56,
        KeyCode::NumpadAdd => 0x57,
        KeyCode::NumpadEnter => 0x58,
        KeyCode::Numpad1 => 0x59,
        KeyCode::Numpad2 => 0x5A,
        KeyCode::Numpad3 => 0x5B,
        KeyCode::Numpad4 => 0x5C,
        KeyCode::Numpad5 => 0x5D,
        KeyCode::Numpad6 => 0x5E,
        KeyCode::Numpad7 => 0x5F,
        KeyCode::Numpad8 => 0x60,
        KeyCode::Numpad9 => 0x61,
        KeyCode::Numpad0 => 0x62,
        KeyCode::NumpadDecimal => 0x63,
        // Intl-102 (non-US backslash, key between LShift and Z on
        // European ISO keyboards)
        KeyCode::IntlBackslash => 0x64,
        // "Application" / context menu key
        KeyCode::ContextMenu => 0x65,
        KeyCode::NumpadEqual => 0x67,
        // Extended function row (Mac extended keyboards, IBM Model M)
        KeyCode::F13 => 0x68,
        KeyCode::F14 => 0x69,
        KeyCode::F15 => 0x6A,
        KeyCode::F16 => 0x6B,
        KeyCode::F17 => 0x6C,
        KeyCode::F18 => 0x6D,
        KeyCode::F19 => 0x6E,
        KeyCode::F20 => 0x6F,
        KeyCode::F21 => 0x70,
        KeyCode::F22 => 0x71,
        KeyCode::F23 => 0x72,
        KeyCode::F24 => 0x73,
        KeyCode::NumpadComma => 0x85,
        // Japanese / international layouts
        KeyCode::IntlRo => 0x87,
        KeyCode::IntlYen => 0x89,
        // Modifiers (HID page 0x07, usages 0xE0..=0xE7)
        KeyCode::ControlLeft => 0xE0,
        KeyCode::ShiftLeft => 0xE1,
        KeyCode::AltLeft => 0xE2,
        KeyCode::SuperLeft => 0xE3,
        KeyCode::ControlRight => 0xE4,
        KeyCode::ShiftRight => 0xE5,
        KeyCode::AltRight => 0xE6,
        KeyCode::SuperRight => 0xE7,
        _ => return None,
    };
    Some(hid_keyboard(u))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expect_input(w: &WireEvent) -> &InputEvent {
        match w {
            WireEvent::Input(e) => e,
            other => panic!("expected Input, got {other:?}"),
        }
    }

    #[test]
    fn keydown_carries_current_modifiers() {
        let mut t = WinitTranslator::new();
        // Press shift.
        t.translate(RenderEvent::Modifiers({
            let mut m = ModifiersState::default();
            m |= ModifiersState::SHIFT;
            m
        }));
        // Then press 'a'.
        let evts = t.translate(RenderEvent::Key {
            code: KeyCode::KeyA,
            pressed: true,
            repeat: false,
            text: None,
        });
        assert_eq!(evts.len(), 1);
        match &expect_input(&evts[0]).kind {
            InputEventKind::KeyDown { key, modifiers } => {
                assert_eq!(*key, hid_keyboard(0x04));
                assert!(modifiers.shift);
                assert!(!modifiers.ctrl);
            }
            other => panic!("expected KeyDown, got {other:?}"),
        }
    }

    #[test]
    fn key_repeat_is_dropped() {
        let mut t = WinitTranslator::new();
        let evts = t.translate(RenderEvent::Key {
            code: KeyCode::KeyA,
            pressed: true,
            repeat: true,
            text: None,
        });
        assert!(evts.is_empty());
    }

    #[test]
    fn cursor_outside_video_emits_invisible_packet() {
        let mut t = WinitTranslator::new();
        let evts = t.translate(RenderEvent::Cursor {
            video_normalized: None,
        });
        assert_eq!(evts.len(), 1);
        match &evts[0] {
            WireEvent::Cursor(c) => assert!(!c.visible),
            other => panic!("expected Cursor, got {other:?}"),
        }
    }

    #[test]
    fn focus_loss_clears_modifiers() {
        let mut t = WinitTranslator::new();
        t.translate(RenderEvent::Modifiers({
            let mut m = ModifiersState::default();
            m |= ModifiersState::CONTROL;
            m
        }));
        t.translate(RenderEvent::Focused(false));
        // Now press 'a'; modifiers should be clear.
        let evts = t.translate(RenderEvent::Key {
            code: KeyCode::KeyA,
            pressed: true,
            repeat: false,
            text: None,
        });
        match &expect_input(&evts[0]).kind {
            InputEventKind::KeyDown { modifiers, .. } => {
                assert!(!modifiers.ctrl);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn event_ids_increment() {
        let mut t = WinitTranslator::new();
        let a = expect_input(
            &t.translate(RenderEvent::Key {
                code: KeyCode::KeyA,
                pressed: true,
                repeat: false,
                text: None,
            })[0],
        )
        .event_id;
        let b = expect_input(
            &t.translate(RenderEvent::Key {
                code: KeyCode::KeyB,
                pressed: true,
                repeat: false,
                text: None,
            })[0],
        )
        .event_id;
        assert_eq!(b, a + 1);
    }

    #[test]
    fn unmodified_text_keypress_takes_the_text_path() {
        let mut t = WinitTranslator::new();
        let evts = t.translate(RenderEvent::Key {
            code: KeyCode::KeyA,
            pressed: true,
            repeat: false,
            text: Some("a".into()),
        });
        assert_eq!(evts.len(), 1);
        match &expect_input(&evts[0]).kind {
            InputEventKind::Text { utf8 } => assert_eq!(utf8, "a"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn ctrl_held_uses_hid_path_for_shortcut() {
        let mut t = WinitTranslator::new();
        t.translate(RenderEvent::Modifiers({
            let mut m = ModifiersState::default();
            m |= ModifiersState::CONTROL;
            m
        }));
        let evts = t.translate(RenderEvent::Key {
            code: KeyCode::KeyC,
            pressed: true,
            repeat: false,
            text: Some("c".into()),
        });
        assert_eq!(evts.len(), 1);
        match &expect_input(&evts[0]).kind {
            InputEventKind::KeyDown { modifiers, .. } => assert!(modifiers.ctrl),
            other => panic!("expected KeyDown, got {other:?}"),
        }
    }

    #[test]
    fn shift_does_not_force_hid_path() {
        let mut t = WinitTranslator::new();
        t.translate(RenderEvent::Modifiers({
            let mut m = ModifiersState::default();
            m |= ModifiersState::SHIFT;
            m
        }));
        let evts = t.translate(RenderEvent::Key {
            code: KeyCode::KeyA,
            pressed: true,
            repeat: false,
            text: Some("A".into()),
        });
        match &expect_input(&evts[0]).kind {
            InputEventKind::Text { utf8 } => assert_eq!(utf8, "A"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn cursor_emits_packet_with_incrementing_seq() {
        let mut t = WinitTranslator::new();
        let evts1 = t.translate(RenderEvent::Cursor {
            video_normalized: Some((0.5, 0.5)),
        });
        let evts2 = t.translate(RenderEvent::Cursor {
            video_normalized: Some((0.6, 0.6)),
        });
        let (s1, s2) = match (&evts1[0], &evts2[0]) {
            (WireEvent::Cursor(a), WireEvent::Cursor(b)) => (a.seq, b.seq),
            _ => panic!("expected Cursor packets"),
        };
        assert_eq!(s2, s1 + 1);
    }
}
