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

use std::collections::HashSet;

use enigo::{Axis, Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings};

use tether_protocol::cursor::ClientCursorPacket;
use tether_protocol::input::{
    HidUsage, InputEvent, InputEventKind, Modifiers, MouseButton as ProtoButton, ScrollKind,
};

// HID Page 0x07 + usage IDs for the eight modifier keys. The trailing
// 4 are the right-side variants (e.g. RCtrl); held_keys is a flat
// HashSet so a "shift held" check has to look up both LShift and RShift.
const HID_LCTRL: HidUsage = HidUsage((0x07 << 16) | 0xE0);
const HID_LSHIFT: HidUsage = HidUsage((0x07 << 16) | 0xE1);
const HID_LALT: HidUsage = HidUsage((0x07 << 16) | 0xE2);
const HID_LMETA: HidUsage = HidUsage((0x07 << 16) | 0xE3);
const HID_RCTRL: HidUsage = HidUsage((0x07 << 16) | 0xE4);
const HID_RSHIFT: HidUsage = HidUsage((0x07 << 16) | 0xE5);
const HID_RALT: HidUsage = HidUsage((0x07 << 16) | 0xE6);
const HID_RMETA: HidUsage = HidUsage((0x07 << 16) | 0xE7);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ModBit {
    Shift,
    Ctrl,
    Alt,
    Meta,
}

/// Identify which modifier bit a HID usage controls (or `None` for
/// non-modifier keys). Used to skip reconciliation for the very
/// modifier we're about to press / release, which would otherwise
/// double-press.
fn modifier_bit_of(hid: HidUsage) -> Option<ModBit> {
    match hid {
        HID_LSHIFT | HID_RSHIFT => Some(ModBit::Shift),
        HID_LCTRL | HID_RCTRL => Some(ModBit::Ctrl),
        HID_LALT | HID_RALT => Some(ModBit::Alt),
        HID_LMETA | HID_RMETA => Some(ModBit::Meta),
        _ => None,
    }
}

use super::{InjectError, Injector, Result};

/// Conservative fallback when neither enigo nor the capture path has
/// told us a real display size. 1080p is the median desktop today; if
/// we end up here AND the host display is much larger, the cursor will
/// be confined to the top-left 1920×1080 region until `set_display_size`
/// arrives. Loud trace logs at first inject_cursor catch this.
const FALLBACK_DISPLAY: (i32, i32) = (1920, 1080);

pub struct LibeiInjector {
    enigo: Enigo,
    /// Host display pixel dimensions used to convert normalised cursor
    /// coords to absolute pixels. Sourced from the capture path via
    /// `set_display_size`; falls back to [`FALLBACK_DISPLAY`] until the
    /// first capture frame arrives.
    display: (i32, i32),
    display_is_authoritative: bool,
    /// Keys we've sent a press for that haven't been released yet.
    /// Walked on Drop so a sudden client disconnect doesn't leave the
    /// host stuck holding Ctrl, Cmd, or whatever the operator was
    /// chording. Mirrors RustDesk's `release_pressed_modifiers` and
    /// Sunshine's `input::reset`.
    held_keys: HashSet<HidUsage>,
    held_buttons: HashSet<ProtoButton>,
    /// Highest cursor seq we've applied. Cursor datagrams reorder; we
    /// drop any incoming packet whose seq is older. `None` until the
    /// first cursor arrives. u32 wraps; a 4-billion-event reset is
    /// effectively never in v0.
    last_cursor_seq: Option<u32>,
}

impl LibeiInjector {
    /// Establish the libei session via the Remote Desktop portal.
    /// enigo is built without its x11rb feature (see the crate Cargo.toml
    /// for rationale), so libei is the only backend on the table — no
    /// runtime probe needed and no X11 connection attempt to surface as
    /// an error in the host's log.
    ///
    /// Runs on a blocking thread because enigo's constructor uses its
    /// own internal block_on (deadlocks inside an active tokio context).
    ///
    /// Display dimensions are intentionally NOT sourced from enigo:
    /// `Enigo::main_display()` is unimplemented on libei. The capture
    /// path calls `set_display_size` with the real numbers from the
    /// PipeWire-negotiated stream format.
    ///
    /// Three lines of `Start emulating` on stdout during this call are
    /// from a raw `println!` in enigo 0.6.1's libei backend — one per
    /// virtual device the portal exposes (keyboard, pointer, scroll). It
    /// isn't gated by a logger, so we can't filter it from our side
    /// without dup2'ing fd 1 across the call. Live with it until enigo
    /// upstream demotes that line.
    pub async fn connect() -> Result<Self> {
        let settings = Settings::default();
        let enigo = tokio::task::spawn_blocking(move || Enigo::new(&settings))
            .await
            .map_err(|e| InjectError::Init(format!("spawn_blocking join: {e}")))?
            .map_err(|e| InjectError::Init(format!("enigo new: {e:?}")))?;
        tracing::info!(
            session = "wayland (libei)",
            "linux input backend selected"
        );
        Ok(Self {
            enigo,
            display: FALLBACK_DISPLAY,
            display_is_authoritative: false,
            held_keys: HashSet::new(),
            held_buttons: HashSet::new(),
            last_cursor_seq: None,
        })
    }
}

impl Drop for LibeiInjector {
    fn drop(&mut self) {
        for hid in self.held_keys.drain().collect::<Vec<_>>() {
            if let Some(k) = hid_to_enigo(hid) {
                let _ = self.enigo.key(k, Direction::Release);
            }
        }
        for btn in self.held_buttons.drain().collect::<Vec<_>>() {
            let _ = self
                .enigo
                .button(proto_button_to_enigo(btn), Direction::Release);
        }
    }
}

impl LibeiInjector {
    /// What we currently believe is held on the host, derived from
    /// `held_keys`. Cheap (8 lookups in a small HashSet).
    fn host_modifiers(&self) -> Modifiers {
        Modifiers {
            shift: self.held_keys.contains(&HID_LSHIFT) || self.held_keys.contains(&HID_RSHIFT),
            ctrl: self.held_keys.contains(&HID_LCTRL) || self.held_keys.contains(&HID_RCTRL),
            alt: self.held_keys.contains(&HID_LALT) || self.held_keys.contains(&HID_RALT),
            meta: self.held_keys.contains(&HID_LMETA) || self.held_keys.contains(&HID_RMETA),
        }
    }

    /// Bring host modifier state in line with what the wire says is
    /// held, by synthesising press/release pairs for any bit that
    /// disagrees. `skip` lets the caller exclude the bit they're
    /// about to press or release themselves (would otherwise
    /// double-press).
    ///
    /// Mirrors Sunshine's `send_key_and_modifiers` and RustDesk's
    /// pre-mouse-down reconciliation. The "left" variant is always
    /// synthesised — we don't try to remember which side the user
    /// originally pressed, since the wire's `Modifiers` bitmask
    /// doesn't carry that distinction.
    fn reconcile_modifiers(&mut self, wire: Modifiers, skip: Option<ModBit>) -> Result<()> {
        let host = self.host_modifiers();
        for (bit, wire_bit, host_bit, hid) in [
            (ModBit::Shift, wire.shift, host.shift, HID_LSHIFT),
            (ModBit::Ctrl, wire.ctrl, host.ctrl, HID_LCTRL),
            (ModBit::Alt, wire.alt, host.alt, HID_LALT),
            (ModBit::Meta, wire.meta, host.meta, HID_LMETA),
        ] {
            if Some(bit) == skip || wire_bit == host_bit {
                continue;
            }
            let direction = if wire_bit {
                Direction::Press
            } else {
                Direction::Release
            };
            if let Some(k) = hid_to_enigo(hid) {
                self.enigo
                    .key(k, direction)
                    .map_err(|e| InjectError::Inject(format!("reconcile modifier: {e:?}")))?;
                if wire_bit {
                    self.held_keys.insert(hid);
                } else {
                    self.held_keys.remove(&hid);
                    // Also clear the right-hand variant if it happened
                    // to be in our set — a single wire bit covers both.
                    self.held_keys
                        .remove(&modifier_right_variant(hid).unwrap_or(hid));
                }
                tracing::trace!(
                    ?bit,
                    ?direction,
                    "synthesised modifier to match wire state"
                );
            }
        }
        Ok(())
    }
}

/// Map a left modifier HID to its right-side variant (and vice versa)
/// so the release path can clear both sides of a single wire bit.
fn modifier_right_variant(hid: HidUsage) -> Option<HidUsage> {
    match hid {
        HID_LSHIFT => Some(HID_RSHIFT),
        HID_LCTRL => Some(HID_RCTRL),
        HID_LALT => Some(HID_RALT),
        HID_LMETA => Some(HID_RMETA),
        _ => None,
    }
}

impl Injector for LibeiInjector {
    // All float->int casts here either clamp (mouse coords) or round and
    // saturate (scroll ticks); none can produce a value outside the
    // sensible i32 pixel/tick range for any monitor that exists in 2026.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn inject(&mut self, evt: &InputEvent) -> Result<()> {
        match &evt.kind {
            InputEventKind::KeyDown { key, modifiers } => {
                self.reconcile_modifiers(*modifiers, modifier_bit_of(*key))?;
                if let Some(k) = hid_to_enigo(*key) {
                    self.enigo
                        .key(k, Direction::Press)
                        .map_err(|e| InjectError::Inject(format!("key down: {e:?}")))?;
                    self.held_keys.insert(*key);
                }
                Ok(())
            }
            InputEventKind::KeyUp { key, modifiers } => {
                self.reconcile_modifiers(*modifiers, modifier_bit_of(*key))?;
                if let Some(k) = hid_to_enigo(*key) {
                    self.enigo
                        .key(k, Direction::Release)
                        .map_err(|e| InjectError::Inject(format!("key up: {e:?}")))?;
                    self.held_keys.remove(key);
                }
                Ok(())
            }
            InputEventKind::Text { utf8 } => {
                // enigo's text() picks the fastest available path
                // (xdotool CHARS, Wayland virtual_keyboard text, etc.)
                // and falls back to per-codepoint Key::Unicode entry.
                // This is the path that handles IME / dead keys / AltGr
                // / non-ASCII typing correctly without depending on the
                // host's keymap matching the client's.
                self.enigo
                    .text(utf8)
                    .map_err(|e| InjectError::Inject(format!("text: {e:?}")))?;
                Ok(())
            }
            InputEventKind::MouseButton {
                button,
                pressed,
                modifiers,
            } => {
                self.reconcile_modifiers(*modifiers, None)?;
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
                if *pressed {
                    self.held_buttons.insert(*button);
                } else {
                    self.held_buttons.remove(button);
                }
                Ok(())
            }
            InputEventKind::MouseScroll {
                dx,
                dy,
                kind,
                modifiers,
            } => {
                self.reconcile_modifiers(*modifiers, None)?;
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

    fn set_display_size(&mut self, width: u32, height: u32) {
        // i32 cast: any display whose pixel dim exceeds i32::MAX
        // (2.1 billion) is not real. Clamp defensively anyway.
        let w = i32::try_from(width).unwrap_or(i32::MAX);
        let h = i32::try_from(height).unwrap_or(i32::MAX);
        if w <= 0 || h <= 0 {
            tracing::warn!(width, height, "ignoring zero/negative display size");
            return;
        }
        let new_dims = (w, h);
        if new_dims != self.display {
            tracing::info!(
                old = ?self.display,
                new = ?new_dims,
                authoritative = self.display_is_authoritative,
                "injector display size updated"
            );
            self.display = new_dims;
        }
        self.display_is_authoritative = true;
    }

    // All float->int casts here clamp first and project into pixel
    // coords within a monitor that fits in i32; safe for any display
    // that exists in 2026.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn inject_cursor(&mut self, cursor: &ClientCursorPacket) -> Result<()> {
        // Drop reordered datagrams. The wrapping comparison is the
        // standard "is `new` strictly newer than `last`?" test for an
        // unsigned counter that wraps at u32::MAX.
        if let Some(last) = self.last_cursor_seq {
            if cursor.seq.wrapping_sub(last) >= (u32::MAX / 2) {
                tracing::trace!(seq = cursor.seq, last, "out-of-order cursor; dropping");
                return Ok(());
            }
        }
        self.last_cursor_seq = Some(cursor.seq);

        if !cursor.visible {
            // No-op for now — there's no separate "hide host cursor"
            // verb in enigo, and the host's own OS cursor renders
            // independently. When we ship a host-cursor sprite, this
            // is where the "stop drawing" signal goes.
            return Ok(());
        }
        if !self.display_is_authoritative {
            tracing::trace!(
                fallback = ?self.display,
                "injecting cursor with fallback display size; capture path hasn't reported real dims yet"
            );
        }
        if cursor.display_idx != 0 {
            tracing::trace!(
                display_idx = cursor.display_idx,
                "non-zero display_idx; pinning to primary"
            );
        }
        let px = (cursor.x.clamp(0.0, 1.0) * self.display.0 as f32) as i32;
        let py = (cursor.y.clamp(0.0, 1.0) * self.display.1 as f32) as i32;
        self.enigo
            .move_mouse(px, py, Coordinate::Abs)
            .map_err(|e| InjectError::Inject(format!("move_mouse: {e:?}")))?;
        Ok(())
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
        0x46 => Key::PrintScr,
        0x47 => Key::ScrollLock,
        0x48 => Key::Pause,
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
        // Numpad / keypad. enigo distinguishes Numpad0..9 from main-
        // row digits; the arithmetic keys reuse the non-numpad symbol
        // variants since enigo doesn't have NumpadAdd etc.
        0x53 => Key::Numlock,
        0x54 => Key::Divide,
        0x55 => Key::Multiply,
        0x56 => Key::Subtract,
        0x57 => Key::Add,
        0x58 => Key::Return,
        0x59 => Key::Numpad1,
        0x5A => Key::Numpad2,
        0x5B => Key::Numpad3,
        0x5C => Key::Numpad4,
        0x5D => Key::Numpad5,
        0x5E => Key::Numpad6,
        0x5F => Key::Numpad7,
        0x60 => Key::Numpad8,
        0x61 => Key::Numpad9,
        0x62 => Key::Numpad0,
        0x63 => Key::Decimal,
        // No clean enigo mapping for IntlBackslash or the context-menu
        // "Application" key on Linux (enigo's `Apps` is Windows-only).
        // European ISO users typing "<>" and anyone hitting Menu will
        // see those round-trip to nothing for now; pick fallbacks
        // when a real workload needs them.
        0x67 => Key::Unicode('='),
        0x68 => Key::F13,
        0x69 => Key::F14,
        0x6A => Key::F15,
        0x6B => Key::F16,
        0x6C => Key::F17,
        0x6D => Key::F18,
        0x6E => Key::F19,
        0x6F => Key::F20,
        0x70 => Key::F21,
        0x71 => Key::F22,
        0x72 => Key::F23,
        0x73 => Key::F24,
        0x85 => Key::Unicode(','),
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
