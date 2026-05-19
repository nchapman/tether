//! Linux input injection via libei (Wayland-native, portal-mediated).
//!
//! The `enigo` crate with the `libei_tokio` feature handles the portal
//! handshake against the Remote Desktop interface; the user sees a
//! permission prompt analogous to the screen-capture one. After that,
//! events go straight over the EI Unix socket to the compositor's input
//! sink.
//!
//! v0 limitations:
//!   - Display size is sampled once at connect time and never refreshed,
//!     so a monitor hot-plug or scale change between sessions will
//!     misplace the cursor until the operator reconnects.
//!   - Scroll is forwarded as integer ticks; the precise pixel-delta
//!     mode used by trackpads is rounded.

use enigo::{Axis, Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings};

use tether_protocol::input::{
    HidUsage, InputEvent, InputEventKind, MouseButton as ProtoButton, ScrollKind,
};

use super::{InjectError, Injector, Result};

pub struct LibeiInjector {
    enigo: Enigo,
    display: (i32, i32),
}

impl LibeiInjector {
    /// Establish the libei connection. Runs the portal handshake on a
    /// blocking thread so it doesn't stall the tokio worker that called
    /// us — enigo's constructor uses its own internal block_on, which
    /// deadlocks if invoked from inside an active tokio context.
    pub async fn connect() -> Result<Self> {
        let enigo = tokio::task::spawn_blocking(|| Enigo::new(&Settings::default()))
            .await
            .map_err(|e| InjectError::Init(format!("spawn_blocking join: {e}")))?
            .map_err(|e| InjectError::Init(format!("enigo new: {e:?}")))?;
        let display = enigo
            .main_display()
            .map_err(|e| InjectError::Init(format!("main_display: {e:?}")))?;
        Ok(Self { enigo, display })
    }
}

impl Injector for LibeiInjector {
    // All float->int casts here either clamp (mouse coords) or round and
    // saturate (scroll ticks); none can produce a value outside the
    // sensible i32 pixel/tick range for any monitor that exists in 2026.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn inject(&mut self, evt: &InputEvent) -> Result<()> {
        match &evt.kind {
            InputEventKind::KeyDown { key, .. } => {
                if let Some(k) = hid_to_enigo(*key) {
                    self.enigo
                        .key(k, Direction::Press)
                        .map_err(|e| InjectError::Inject(format!("key down: {e:?}")))?;
                }
                Ok(())
            }
            InputEventKind::KeyUp { key, .. } => {
                if let Some(k) = hid_to_enigo(*key) {
                    self.enigo
                        .key(k, Direction::Release)
                        .map_err(|e| InjectError::Inject(format!("key up: {e:?}")))?;
                }
                Ok(())
            }
            InputEventKind::MousePosition { x, y } => {
                // [0,1] -> absolute pixel within the primary display.
                // Clamp because the client *should* already have done so
                // but a misbehaving client could still send 1.5, and
                // enigo doesn't bound-check.
                let px = (x.clamp(0.0, 1.0) * self.display.0 as f32) as i32;
                let py = (y.clamp(0.0, 1.0) * self.display.1 as f32) as i32;
                self.enigo
                    .move_mouse(px, py, Coordinate::Abs)
                    .map_err(|e| InjectError::Inject(format!("move_mouse: {e:?}")))?;
                Ok(())
            }
            InputEventKind::MouseButton { button, pressed } => {
                let b = proto_button_to_enigo(*button);
                self.enigo
                    .button(
                        b,
                        if *pressed {
                            Direction::Press
                        } else {
                            Direction::Release
                        },
                    )
                    .map_err(|e| InjectError::Inject(format!("button: {e:?}")))?;
                Ok(())
            }
            InputEventKind::MouseScroll { dx, dy, kind } => {
                // libei wants discrete ticks. For pixel-mode trackpads
                // we lose sub-tick precision — acceptable for v0.
                let scale = match kind {
                    ScrollKind::Line => 1.0,
                    ScrollKind::Pixel => 1.0 / 16.0,
                };
                let vert = (dy * scale).round() as i32;
                let horiz = (dx * scale).round() as i32;
                if vert != 0 {
                    self.enigo
                        .scroll(-vert, Axis::Vertical)
                        .map_err(|e| InjectError::Inject(format!("scroll v: {e:?}")))?;
                }
                if horiz != 0 {
                    self.enigo
                        .scroll(horiz, Axis::Horizontal)
                        .map_err(|e| InjectError::Inject(format!("scroll h: {e:?}")))?;
                }
                Ok(())
            }
        }
    }
}

fn proto_button_to_enigo(b: ProtoButton) -> Button {
    match b {
        ProtoButton::Left => Button::Left,
        ProtoButton::Right => Button::Right,
        ProtoButton::Middle => Button::Middle,
        ProtoButton::X1 => Button::Back,
        ProtoButton::X2 => Button::Forward,
    }
}

/// HID Keyboard usage (Page 0x07) -> enigo `Key`. Inverse of the
/// `keycode_to_hid` table in the client-side translator; if these
/// diverge, keys round-trip to nothing on the host. Coverage is the
/// same subset (letters, digits, named keys, F-row, nav, modifiers).
// All `as u8` casts below are inside match arms whose ranges guarantee
// the value fits in u8 (max HID usage handled is 0xE7).
#[allow(clippy::cast_possible_truncation)]
fn hid_to_enigo(usage: HidUsage) -> Option<Key> {
    let page = (usage.0 >> 16) & 0xFFFF;
    let id = usage.0 & 0xFFFF;
    if page != 0x07 {
        tracing::trace!(?usage, "HID page outside Keyboard; dropping");
        return None;
    }
    let key = match id {
        // Letters: HID 0x04..=0x1D => 'a'..='z'. enigo uses Unicode for
        // letters; the kernel/compositor will apply current keymap.
        0x04..=0x1D => {
            let c = char::from(b'a' + (id - 0x04) as u8);
            Key::Unicode(c)
        }
        // Digits: HID lists 1..9 then 0.
        0x1E..=0x26 => Key::Unicode(char::from(b'1' + (id - 0x1E) as u8)),
        0x27 => Key::Unicode('0'),
        0x28 => Key::Return,
        0x29 => Key::Escape,
        0x2A => Key::Backspace,
        0x2B => Key::Tab,
        0x2C => Key::Space,
        0x2D => Key::Unicode('-'),
        0x2E => Key::Unicode('='),
        0x2F => Key::Unicode('['),
        0x30 => Key::Unicode(']'),
        0x31 => Key::Unicode('\\'),
        0x33 => Key::Unicode(';'),
        0x34 => Key::Unicode('\''),
        0x35 => Key::Unicode('`'),
        0x36 => Key::Unicode(','),
        0x37 => Key::Unicode('.'),
        0x38 => Key::Unicode('/'),
        0x39 => Key::CapsLock,
        0x3A => Key::F1,
        0x3B => Key::F2,
        0x3C => Key::F3,
        0x3D => Key::F4,
        0x3E => Key::F5,
        0x3F => Key::F6,
        0x40 => Key::F7,
        0x41 => Key::F8,
        0x42 => Key::F9,
        0x43 => Key::F10,
        0x44 => Key::F11,
        0x45 => Key::F12,
        0x49 => Key::Insert,
        0x4A => Key::Home,
        0x4B => Key::PageUp,
        0x4C => Key::Delete,
        0x4D => Key::End,
        0x4E => Key::PageDown,
        0x4F => Key::RightArrow,
        0x50 => Key::LeftArrow,
        0x51 => Key::DownArrow,
        0x52 => Key::UpArrow,
        0xE0 => Key::LControl,
        0xE1 => Key::LShift,
        0xE2 => Key::Alt,
        0xE3 => Key::Meta,
        0xE4 => Key::RControl,
        0xE5 => Key::RShift,
        0xE6 => Key::Alt,
        0xE7 => Key::Meta,
        _ => {
            tracing::trace!(?usage, "no enigo mapping for HID id");
            return None;
        }
    };
    Some(key)
}
