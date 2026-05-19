//! Input plumbing: translate render-layer window events into wire-level
//! [`InputEvent`]s on the client, and inject those events into the host's
//! native input system. v0 covers the client side only — injection lands
//! next, behind a per-platform backend.

use tether_protocol::input::{
    HidUsage, InputEvent, InputEventKind, Modifiers, MouseButton, ScrollKind,
};
use tether_protocol::MonoNanos;
use tether_render::{KeyCode, ModifiersState, RenderEvent};

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
            last_cursor: None,
        }
    }

    /// Consume one render event and produce zero or more wire-level input
    /// events. Most render events translate 1:1, but a couple are
    /// pure state updates (modifiers, focus) and return an empty Vec.
    pub fn translate(&mut self, event: RenderEvent) -> Vec<InputEvent> {
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
                    // when we re-enter the window.
                    self.modifiers = Modifiers::default();
                    self.last_cursor = None;
                }
                return Vec::new();
            }
            RenderEvent::Cursor { video_normalized } => {
                self.last_cursor = video_normalized;
                match video_normalized {
                    Some((x, y)) => vec![InputEventKind::MousePosition { x, y }],
                    None => Vec::new(),
                }
            }
            RenderEvent::Key {
                code,
                pressed,
                repeat,
            } => {
                if repeat {
                    // The host's OS will generate its own auto-repeats
                    // from a single keydown; forwarding both ends up
                    // looking like double-typing.
                    return Vec::new();
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
                vec![InputEventKind::MouseButton { button, pressed }]
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
                }]
            }
        };
        kinds
            .into_iter()
            .map(|kind| {
                let evt = InputEvent {
                    event_id: self.next_event_id,
                    t_client: MonoNanos::now(),
                    kind,
                };
                self.next_event_id = self.next_event_id.wrapping_add(1);
                evt
            })
            .collect()
    }
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
        });
        assert_eq!(evts.len(), 1);
        match &evts[0].kind {
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
        });
        assert!(evts.is_empty());
    }

    #[test]
    fn cursor_outside_video_emits_nothing() {
        let mut t = WinitTranslator::new();
        let evts = t.translate(RenderEvent::Cursor {
            video_normalized: None,
        });
        assert!(evts.is_empty());
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
        });
        match &evts[0].kind {
            InputEventKind::KeyDown { modifiers, .. } => {
                assert!(!modifiers.ctrl);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn event_ids_increment() {
        let mut t = WinitTranslator::new();
        let a = &t.translate(RenderEvent::Key {
            code: KeyCode::KeyA,
            pressed: true,
            repeat: false,
        })[0];
        let id_a = a.event_id;
        let b = &t.translate(RenderEvent::Key {
            code: KeyCode::KeyB,
            pressed: true,
            repeat: false,
        })[0];
        assert_eq!(b.event_id, id_a + 1);
    }
}
