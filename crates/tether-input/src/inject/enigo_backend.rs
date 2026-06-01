//! Shared `enigo`-based injector state + logic for the Linux (libei)
//! and macOS (CGEvent) backends. The two platforms differ only in how
//! they construct the underlying [`Enigo`] (portal handshake for libei
//! vs. plain construction for macOS); everything downstream — modifier
//! reconciliation, the HID→`Key` table, cursor coord conversion, the
//! Drop-time held-key release — is identical.
//!
//! Per-platform files own only their `connect()`. They wrap an
//! [`EnigoBackend`] and forward the `Injector` trait calls to it.

use std::collections::HashSet;

use enigo::{Axis, Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse};

use tether_protocol::cursor::ClientCursorPacket;
use tether_protocol::input::{
    HidUsage, InputEvent, InputEventKind, Modifiers, MouseButton as ProtoButton, ScrollKind,
};

use super::{InjectError, Result};

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

/// Conservative fallback when neither enigo nor the capture path has
/// told us a real display size. 1080p is the median desktop today; if
/// we end up here AND the host display is much larger, the cursor will
/// be confined to the top-left 1920×1080 region until `set_display_size`
/// arrives. Loud trace logs at first inject_cursor catch this.
pub(super) const FALLBACK_DISPLAY: (i32, i32) = (1920, 1080);

/// Shared injector state. Constructed by per-platform `connect()`
/// helpers from a `connect`-built [`Enigo`].
pub(super) struct EnigoBackend {
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
    /// first cursor arrives. u32 wraps; at 1 kHz cursor packets it
    /// takes ~50 days of continuous session to roll over, so plain
    /// `<` comparison is fine in practice.
    last_cursor_seq: Option<u32>,
}

impl EnigoBackend {
    pub(super) fn new(enigo: Enigo) -> Self {
        Self {
            enigo,
            display: FALLBACK_DISPLAY,
            display_is_authoritative: false,
            held_keys: HashSet::new(),
            held_buttons: HashSet::new(),
            last_cursor_seq: None,
        }
    }

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
    /// disagrees. `skip` is a single [`ModBit`] that covers BOTH the
    /// left and right variant of the modifier the caller is about to
    /// press or release itself (the wire's `Modifiers` bitmask
    /// doesn't distinguish sides); without it we'd double-press the
    /// modifier on its own keydown event.
    ///
    /// Mirrors Sunshine's `send_key_and_modifiers` and RustDesk's
    /// pre-mouse-down reconciliation. The "left" variant is always
    /// synthesised — we don't try to remember which side the user
    /// originally pressed.
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
                tracing::trace!(?bit, ?direction, "synthesised modifier to match wire state");
            }
        }
        Ok(())
    }

    // All float->int casts here either clamp (mouse coords) or round and
    // saturate (scroll ticks); none can produce a value outside the
    // sensible i32 pixel/tick range for any monitor that exists in 2026.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub(super) fn inject(&mut self, evt: &InputEvent) -> Result<()> {
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
            InputEventKind::RelativeMouseMove { dx, dy, modifiers } => {
                self.reconcile_modifiers(*modifiers, None)?;
                // Per-event sanity clamp. i16 wire values are
                // bounded to ±32_767; without clamping, a malicious
                // client at high event rate could move the host
                // cursor across multiple displays per event. macOS's
                // CGEvent path applies large deltas before the
                // screen-edge guard, so values like i16::MAX can
                // reach hot corners / Mission Control / Dock /
                // Terminal windows the user never intended to
                // target. ±1000 px per event covers any legitimate
                // gaming-mouse flick (a 1600 DPI mouse moved 1
                // inch in 1 frame is ~1600 dots, but the OS scales
                // for pointer-acceleration before delivery so the
                // post-scale value is well under this cap).
                let (dx_clamped, dy_clamped) = clamp_relative_delta(*dx, *dy);
                // Coordinate::Rel routes through the OS's native
                // delta injection path: EV_REL/REL_X+REL_Y on Linux
                // uinput, CGEventCreateMouseEvent's delta fields on
                // macOS, SendInput-without-ABSOLUTE on Windows. Only
                // events from that path are observable to raw-input
                // games — synthetic SetCursorPos-style warps are
                // filtered out by every OS by design.
                self.enigo
                    .move_mouse(dx_clamped, dy_clamped, Coordinate::Rel)
                    .map_err(|e| InjectError::Inject(format!("relative move: {e:?}")))?;
                Ok(())
            }
            InputEventKind::MouseScroll {
                dx,
                dy,
                kind,
                modifiers,
            } => {
                self.reconcile_modifiers(*modifiers, None)?;
                // libei wants discrete ticks. Pixel-mode trackpad
                // deltas get quantised to whole ticks via the
                // 1/16-px scale below; smoother sub-tick scroll
                // would need libei's high-resolution-scroll axis,
                // which our wire format doesn't carry yet.
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

    pub(super) fn set_display_size(&mut self, width: u32, height: u32) {
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
    pub(super) fn inject_cursor(&mut self, cursor: &ClientCursorPacket) -> Result<()> {
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
        // macOS note: CGEvent expects screen points (Retina-scaled),
        // not raw pixels. enigo's macOS backend already does the
        // points-vs-pixels conversion internally via NSScreen scale,
        // so the absolute pixel coordinates we pass here are correct
        // for both libei (linux) and CGEvent (macos).
        self.enigo
            .move_mouse(px, py, Coordinate::Abs)
            .map_err(|e| InjectError::Inject(format!("move_mouse: {e:?}")))?;
        Ok(())
    }
}

impl Drop for EnigoBackend {
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
        // PrintScreen / ScrollLock / Pause / Insert / Numlock / F21–F24
        // are gated to Linux-only in enigo's `Key` enum (no equivalent
        // on macOS; on Windows enigo 0.6 doesn't expose ScrollLock et al
        // as Key variants). Drop silently on other platforms.
        #[cfg(target_os = "linux")]
        0x46 => Key::PrintScr,
        #[cfg(target_os = "linux")]
        0x47 => Key::ScrollLock,
        #[cfg(target_os = "linux")]
        0x48 => Key::Pause,
        #[cfg(target_os = "linux")]
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
        // variants since enigo doesn't have NumpadAdd etc. Numlock is
        // Linux/Windows-only (no NumLock on Mac keyboards).
        #[cfg(target_os = "linux")]
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
        // F21–F24 — enigo's macOS Key enum stops at F20 (CGEvent
        // doesn't have key codes past F20); Windows enigo 0.6 also
        // doesn't expose F21-F24 variants. Linux-only.
        #[cfg(target_os = "linux")]
        0x70 => Key::F21,
        #[cfg(target_os = "linux")]
        0x71 => Key::F22,
        #[cfg(target_os = "linux")]
        0x72 => Key::F23,
        #[cfg(target_os = "linux")]
        0x73 => Key::F24,
        0x85 => Key::Unicode(','),
        0xE0 => Key::LControl,
        0xE1 => Key::LShift,
        0xE2 => Key::Alt,
        0xE3 => Key::Meta,
        0xE4 => Key::RControl,
        0xE5 => Key::RShift,
        // Right-Alt. On macOS enigo distinguishes left/right Option;
        // on Linux/Windows enigo's `Key` has no right-Alt variant, so
        // both sides collapse to the same wire bit (the modifier-bit
        // reconciliation in `modifier_bit_of` already keeps held-key
        // state consistent across the collapse).
        #[cfg(target_os = "macos")]
        0xE6 => Key::ROption,
        #[cfg(not(target_os = "macos"))]
        0xE6 => Key::Alt,
        0xE7 => Key::Meta,
        _ => {
            tracing::trace!(?usage, "no enigo mapping for HID id");
            return None;
        }
    };
    Some(key)
}

/// Clamp i16 wire deltas to ±1000 px per event before handing them
/// to enigo's relative-move path. See the call site in
/// `RelativeMouseMove` for the motivation; pure helper extracted so
/// the bounds are unit-testable without needing a live display.
const MAX_REL_DELTA_PX: i32 = 1000;
fn clamp_relative_delta(dx: i16, dy: i16) -> (i32, i32) {
    (
        i32::from(dx).clamp(-MAX_REL_DELTA_PX, MAX_REL_DELTA_PX),
        i32::from(dy).clamp(-MAX_REL_DELTA_PX, MAX_REL_DELTA_PX),
    )
}

#[cfg(test)]
mod clamp_tests {
    use super::*;

    #[test]
    fn small_deltas_pass_through() {
        assert_eq!(clamp_relative_delta(5, -3), (5, -3));
        assert_eq!(clamp_relative_delta(1000, -1000), (1000, -1000));
    }

    #[test]
    fn extreme_positive_saturates() {
        assert_eq!(clamp_relative_delta(i16::MAX, i16::MAX), (1000, 1000));
    }

    #[test]
    fn extreme_negative_saturates() {
        assert_eq!(clamp_relative_delta(i16::MIN, i16::MIN), (-1000, -1000));
    }

    #[test]
    fn asymmetric_clamp_per_axis() {
        assert_eq!(clamp_relative_delta(i16::MAX, 50), (1000, 50));
        assert_eq!(clamp_relative_delta(-50, i16::MIN), (-50, -1000));
    }
}
