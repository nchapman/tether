//! Linux input injection via `/dev/uinput` (Wayland-native, portal-free).
//!
//! Unlike the screen-capture path, input does **not** go through the
//! RemoteDesktop portal. That portal re-prompts the user on every
//! session and its restore-token support is unreliable across
//! compositors — unacceptable for unattended remote access. uinput
//! creates virtual input devices the kernel exposes to the compositor's
//! libinput seat directly: once `/dev/uinput` is writable (see
//! `tether-host --setup-input`) there is no prompt, ever. This mirrors
//! what the mature Wayland hosts (Sunshine, rustdesk server mode) ship.
//!
//! Three virtual devices are created, following Sunshine's split (a
//! single device mixing `EV_REL` and `EV_ABS` confuses libinput):
//!   - a **keyboard** (`EV_KEY`),
//!   - a **relative pointer** (`EV_REL` REL_X/REL_Y + buttons + wheels),
//!   - an **absolute pointer** (`EV_ABS` ABS_X/ABS_Y + buttons + wheels,
//!     `INPUT_PROP_POINTER`).
//!
//! Known gaps (acceptable for the current input contract):
//!   - **Text is keymap-dependent.** The client sends most ordinary
//!     typing as [`InputEventKind::Text`]; uinput can only emit physical
//!     keycodes, not Unicode codepoints, so we reverse-map characters to
//!     a US-QWERTY keystroke. Non-US host layouts and non-ASCII
//!     codepoints (accents, CJK, emoji) can't round-trip this way and
//!     are logged + dropped. A portal/libei text channel would be needed
//!     to fix this, which would reintroduce the prompt we're removing.
//!   - **Scroll is quantised to whole detents.** Sub-detent trackpad
//!     precision is lost, same as the previous libei backend.

use std::collections::{HashMap, HashSet};

use evdev::{
    uinput::VirtualDevice, AbsInfo, AbsoluteAxisCode, AbsoluteAxisEvent, AttributeSet, BusType,
    InputEvent, InputId, KeyCode, KeyEvent, PropType, RelativeAxisCode, RelativeAxisEvent,
    SynchronizationCode, SynchronizationEvent, UinputAbsSetup,
};

use tether_protocol::cursor::ClientCursorPacket;
use tether_protocol::input::{
    HidUsage, InputEvent as ProtoInputEvent, InputEventKind, Modifiers, MouseButton as ProtoButton,
    ScrollKind,
};

use super::{InjectError, Injector, Result};

// HID Page 0x07 + usage IDs for the eight modifier keys (mirrors the
// enigo backend's constants). `held_keys` is a flat HID set, so a
// "shift held" check looks up both LShift and RShift.
const HID_LCTRL: HidUsage = HidUsage((0x07 << 16) | 0xE0);
const HID_LSHIFT: HidUsage = HidUsage((0x07 << 16) | 0xE1);
const HID_LALT: HidUsage = HidUsage((0x07 << 16) | 0xE2);
const HID_LMETA: HidUsage = HidUsage((0x07 << 16) | 0xE3);
const HID_RCTRL: HidUsage = HidUsage((0x07 << 16) | 0xE4);
const HID_RSHIFT: HidUsage = HidUsage((0x07 << 16) | 0xE5);
const HID_RALT: HidUsage = HidUsage((0x07 << 16) | 0xE6);
const HID_RMETA: HidUsage = HidUsage((0x07 << 16) | 0xE7);

/// Full-scale value for the absolute pointer axes. Client cursor coords
/// arrive normalised (0.0–1.0), so we map them onto a fixed logical
/// range independent of the host's pixel resolution; the compositor
/// rescales to the actual screen. 0xFFFF gives sub-pixel precision on
/// any real display.
const ABS_MAX: i32 = 0xFFFF;

/// Clamp i16 wire deltas to ±1000 px per relative-move event — see the
/// shared rationale in `enigo_backend` (a malicious client at high
/// event rate could otherwise fling the pointer across displays).
const MAX_REL_DELTA_PX: i32 = 1000;

/// High-resolution wheel units per notch/detent (kernel convention for
/// `REL_WHEEL_HI_RES` / `REL_HWHEEL_HI_RES`).
const WHEEL_HI_RES_PER_DETENT: i32 = 120;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ModBit {
    Shift,
    Ctrl,
    Alt,
    Meta,
}

fn modifier_bit_of(hid: HidUsage) -> Option<ModBit> {
    match hid {
        HID_LSHIFT | HID_RSHIFT => Some(ModBit::Shift),
        HID_LCTRL | HID_RCTRL => Some(ModBit::Ctrl),
        HID_LALT | HID_RALT => Some(ModBit::Alt),
        HID_LMETA | HID_RMETA => Some(ModBit::Meta),
        _ => None,
    }
}

fn modifier_right_variant(hid: HidUsage) -> Option<HidUsage> {
    match hid {
        HID_LSHIFT => Some(HID_RSHIFT),
        HID_LCTRL => Some(HID_RCTRL),
        HID_LALT => Some(HID_RALT),
        HID_LMETA => Some(HID_RMETA),
        _ => None,
    }
}

/// HID Keyboard usage (Page 0x07) → evdev physical key. HID usages are
/// themselves physical-key based, so this is the correct identity
/// mapping; the compositor then applies the host's keymap to produce a
/// keysym. Inverse of the client's `keycode_to_hid`. Covers the same
/// subset as the enigo backend's `hid_to_enigo`.
fn hid_to_evdev(usage: HidUsage) -> Option<KeyCode> {
    let page = (usage.0 >> 16) & 0xFFFF;
    let id = usage.0 & 0xFFFF;
    if page != 0x07 {
        tracing::trace!(?usage, "HID page outside Keyboard; dropping");
        return None;
    }
    let key = match id {
        0x04 => KeyCode::KEY_A,
        0x05 => KeyCode::KEY_B,
        0x06 => KeyCode::KEY_C,
        0x07 => KeyCode::KEY_D,
        0x08 => KeyCode::KEY_E,
        0x09 => KeyCode::KEY_F,
        0x0A => KeyCode::KEY_G,
        0x0B => KeyCode::KEY_H,
        0x0C => KeyCode::KEY_I,
        0x0D => KeyCode::KEY_J,
        0x0E => KeyCode::KEY_K,
        0x0F => KeyCode::KEY_L,
        0x10 => KeyCode::KEY_M,
        0x11 => KeyCode::KEY_N,
        0x12 => KeyCode::KEY_O,
        0x13 => KeyCode::KEY_P,
        0x14 => KeyCode::KEY_Q,
        0x15 => KeyCode::KEY_R,
        0x16 => KeyCode::KEY_S,
        0x17 => KeyCode::KEY_T,
        0x18 => KeyCode::KEY_U,
        0x19 => KeyCode::KEY_V,
        0x1A => KeyCode::KEY_W,
        0x1B => KeyCode::KEY_X,
        0x1C => KeyCode::KEY_Y,
        0x1D => KeyCode::KEY_Z,
        0x1E => KeyCode::KEY_1,
        0x1F => KeyCode::KEY_2,
        0x20 => KeyCode::KEY_3,
        0x21 => KeyCode::KEY_4,
        0x22 => KeyCode::KEY_5,
        0x23 => KeyCode::KEY_6,
        0x24 => KeyCode::KEY_7,
        0x25 => KeyCode::KEY_8,
        0x26 => KeyCode::KEY_9,
        0x27 => KeyCode::KEY_0,
        0x28 => KeyCode::KEY_ENTER,
        0x29 => KeyCode::KEY_ESC,
        0x2A => KeyCode::KEY_BACKSPACE,
        0x2B => KeyCode::KEY_TAB,
        0x2C => KeyCode::KEY_SPACE,
        0x2D => KeyCode::KEY_MINUS,
        0x2E => KeyCode::KEY_EQUAL,
        0x2F => KeyCode::KEY_LEFTBRACE,
        0x30 => KeyCode::KEY_RIGHTBRACE,
        0x31 => KeyCode::KEY_BACKSLASH,
        0x33 => KeyCode::KEY_SEMICOLON,
        0x34 => KeyCode::KEY_APOSTROPHE,
        0x35 => KeyCode::KEY_GRAVE,
        0x36 => KeyCode::KEY_COMMA,
        0x37 => KeyCode::KEY_DOT,
        0x38 => KeyCode::KEY_SLASH,
        0x39 => KeyCode::KEY_CAPSLOCK,
        0x3A => KeyCode::KEY_F1,
        0x3B => KeyCode::KEY_F2,
        0x3C => KeyCode::KEY_F3,
        0x3D => KeyCode::KEY_F4,
        0x3E => KeyCode::KEY_F5,
        0x3F => KeyCode::KEY_F6,
        0x40 => KeyCode::KEY_F7,
        0x41 => KeyCode::KEY_F8,
        0x42 => KeyCode::KEY_F9,
        0x43 => KeyCode::KEY_F10,
        0x44 => KeyCode::KEY_F11,
        0x45 => KeyCode::KEY_F12,
        0x46 => KeyCode::KEY_SYSRQ, // PrintScreen
        0x47 => KeyCode::KEY_SCROLLLOCK,
        0x48 => KeyCode::KEY_PAUSE,
        0x49 => KeyCode::KEY_INSERT,
        0x4A => KeyCode::KEY_HOME,
        0x4B => KeyCode::KEY_PAGEUP,
        0x4C => KeyCode::KEY_DELETE,
        0x4D => KeyCode::KEY_END,
        0x4E => KeyCode::KEY_PAGEDOWN,
        0x4F => KeyCode::KEY_RIGHT,
        0x50 => KeyCode::KEY_LEFT,
        0x51 => KeyCode::KEY_DOWN,
        0x52 => KeyCode::KEY_UP,
        0x53 => KeyCode::KEY_NUMLOCK,
        0x54 => KeyCode::KEY_KPSLASH,
        0x55 => KeyCode::KEY_KPASTERISK,
        0x56 => KeyCode::KEY_KPMINUS,
        0x57 => KeyCode::KEY_KPPLUS,
        0x58 => KeyCode::KEY_KPENTER,
        0x59 => KeyCode::KEY_KP1,
        0x5A => KeyCode::KEY_KP2,
        0x5B => KeyCode::KEY_KP3,
        0x5C => KeyCode::KEY_KP4,
        0x5D => KeyCode::KEY_KP5,
        0x5E => KeyCode::KEY_KP6,
        0x5F => KeyCode::KEY_KP7,
        0x60 => KeyCode::KEY_KP8,
        0x61 => KeyCode::KEY_KP9,
        0x62 => KeyCode::KEY_KP0,
        0x63 => KeyCode::KEY_KPDOT,
        0x64 => KeyCode::KEY_102ND,   // IntlBackslash (ISO "<>" key)
        0x65 => KeyCode::KEY_COMPOSE, // Application / context menu
        0x67 => KeyCode::KEY_KPEQUAL,
        0x68 => KeyCode::KEY_F13,
        0x69 => KeyCode::KEY_F14,
        0x6A => KeyCode::KEY_F15,
        0x6B => KeyCode::KEY_F16,
        0x6C => KeyCode::KEY_F17,
        0x6D => KeyCode::KEY_F18,
        0x6E => KeyCode::KEY_F19,
        0x6F => KeyCode::KEY_F20,
        0x70 => KeyCode::KEY_F21,
        0x71 => KeyCode::KEY_F22,
        0x72 => KeyCode::KEY_F23,
        0x73 => KeyCode::KEY_F24,
        0x85 => KeyCode::KEY_KPCOMMA,
        0xE0 => KeyCode::KEY_LEFTCTRL,
        0xE1 => KeyCode::KEY_LEFTSHIFT,
        0xE2 => KeyCode::KEY_LEFTALT,
        0xE3 => KeyCode::KEY_LEFTMETA,
        0xE4 => KeyCode::KEY_RIGHTCTRL,
        0xE5 => KeyCode::KEY_RIGHTSHIFT,
        0xE6 => KeyCode::KEY_RIGHTALT,
        0xE7 => KeyCode::KEY_RIGHTMETA,
        _ => {
            tracing::trace!(?usage, "no evdev mapping for HID id");
            return None;
        }
    };
    Some(key)
}

/// A US-QWERTY keystroke that produces an ASCII character: the physical
/// key plus whether Shift is required.
struct KeyStroke {
    code: KeyCode,
    shift: bool,
}

/// Reverse-map an ASCII character to the US-QWERTY keystroke that types
/// it. Returns `None` for anything outside printable ASCII (plus tab /
/// newline, handled defensively) — those can't be expressed as physical
/// keycodes without the host's keymap and are dropped by the caller.
fn char_to_keystroke(c: char) -> Option<KeyStroke> {
    let (code, shift) = match c {
        'a'..='z' => (letter_keycode(c), false),
        'A'..='Z' => (letter_keycode(c.to_ascii_lowercase()), true),
        '1' => (KeyCode::KEY_1, false),
        '2' => (KeyCode::KEY_2, false),
        '3' => (KeyCode::KEY_3, false),
        '4' => (KeyCode::KEY_4, false),
        '5' => (KeyCode::KEY_5, false),
        '6' => (KeyCode::KEY_6, false),
        '7' => (KeyCode::KEY_7, false),
        '8' => (KeyCode::KEY_8, false),
        '9' => (KeyCode::KEY_9, false),
        '0' => (KeyCode::KEY_0, false),
        '!' => (KeyCode::KEY_1, true),
        '@' => (KeyCode::KEY_2, true),
        '#' => (KeyCode::KEY_3, true),
        '$' => (KeyCode::KEY_4, true),
        '%' => (KeyCode::KEY_5, true),
        '^' => (KeyCode::KEY_6, true),
        '&' => (KeyCode::KEY_7, true),
        '*' => (KeyCode::KEY_8, true),
        '(' => (KeyCode::KEY_9, true),
        ')' => (KeyCode::KEY_0, true),
        ' ' => (KeyCode::KEY_SPACE, false),
        '\t' => (KeyCode::KEY_TAB, false),
        '\n' | '\r' => (KeyCode::KEY_ENTER, false),
        '-' => (KeyCode::KEY_MINUS, false),
        '_' => (KeyCode::KEY_MINUS, true),
        '=' => (KeyCode::KEY_EQUAL, false),
        '+' => (KeyCode::KEY_EQUAL, true),
        '[' => (KeyCode::KEY_LEFTBRACE, false),
        '{' => (KeyCode::KEY_LEFTBRACE, true),
        ']' => (KeyCode::KEY_RIGHTBRACE, false),
        '}' => (KeyCode::KEY_RIGHTBRACE, true),
        '\\' => (KeyCode::KEY_BACKSLASH, false),
        '|' => (KeyCode::KEY_BACKSLASH, true),
        ';' => (KeyCode::KEY_SEMICOLON, false),
        ':' => (KeyCode::KEY_SEMICOLON, true),
        '\'' => (KeyCode::KEY_APOSTROPHE, false),
        '"' => (KeyCode::KEY_APOSTROPHE, true),
        '`' => (KeyCode::KEY_GRAVE, false),
        '~' => (KeyCode::KEY_GRAVE, true),
        ',' => (KeyCode::KEY_COMMA, false),
        '<' => (KeyCode::KEY_COMMA, true),
        '.' => (KeyCode::KEY_DOT, false),
        '>' => (KeyCode::KEY_DOT, true),
        '/' => (KeyCode::KEY_SLASH, false),
        '?' => (KeyCode::KEY_SLASH, true),
        _ => return None,
    };
    Some(KeyStroke { code, shift })
}

/// Map a lowercase ASCII letter to its evdev keycode. Letters aren't
/// contiguous in the evdev numbering, so route through the (correct,
/// explicit) HID table rather than arithmetic.
fn letter_keycode(c: char) -> KeyCode {
    // 'a' is HID 0x04; offset within the contiguous HID letter block.
    let hid = 0x04 + (c as u32 - 'a' as u32);
    hid_to_evdev(HidUsage((0x07 << 16) | hid)).unwrap_or(KeyCode::KEY_RESERVED)
}

fn proto_button_to_btn(b: ProtoButton) -> KeyCode {
    match b {
        ProtoButton::Left => KeyCode::BTN_LEFT,
        ProtoButton::Right => KeyCode::BTN_RIGHT,
        ProtoButton::Middle => KeyCode::BTN_MIDDLE,
        ProtoButton::X1 => KeyCode::BTN_SIDE,
        ProtoButton::X2 => KeyCode::BTN_EXTRA,
    }
}

/// Every evdev key our keyboard device must advertise: derived directly
/// from `hid_to_evdev` so the capability set can't drift from the events
/// we actually emit.
fn keyboard_keycodes() -> AttributeSet<KeyCode> {
    let mut set = AttributeSet::<KeyCode>::new();
    for id in 0x00u32..=0xE7 {
        if let Some(code) = hid_to_evdev(HidUsage((0x07 << 16) | id)) {
            set.insert(code);
        }
    }
    set
}

fn pointer_buttons() -> AttributeSet<KeyCode> {
    let mut set = AttributeSet::<KeyCode>::new();
    for b in [
        KeyCode::BTN_LEFT,
        KeyCode::BTN_RIGHT,
        KeyCode::BTN_MIDDLE,
        KeyCode::BTN_SIDE,
        KeyCode::BTN_EXTRA,
    ] {
        set.insert(b);
    }
    set
}

fn key_ev(code: KeyCode, value: i32) -> InputEvent {
    KeyEvent::new(code, value).into()
}

fn rel_ev(code: RelativeAxisCode, value: i32) -> InputEvent {
    RelativeAxisEvent::new(code, value).into()
}

fn abs_ev(code: AbsoluteAxisCode, value: i32) -> InputEvent {
    AbsoluteAxisEvent::new(code, value).into()
}

fn syn() -> InputEvent {
    SynchronizationEvent::new(SynchronizationCode::SYN_REPORT, 0).into()
}

pub struct UinputInjector {
    keyboard: VirtualDevice,
    rel_pointer: VirtualDevice,
    abs_pointer: VirtualDevice,
    /// Whether the most recent pointer motion was absolute. Buttons and
    /// scroll are emitted on whichever pointer device last moved so the
    /// click lands at the right place (mirrors Sunshine's
    /// `active_mouse_device`).
    last_motion_absolute: bool,
    held_keys: HashSet<HidUsage>,
    /// Buttons currently held, mapped to the device the press was emitted
    /// on (`true` = absolute pointer). The release MUST go to the same
    /// device: if a mode toggle or a stray cursor datagram flips
    /// `last_motion_absolute` between a button's down and up, routing the
    /// up by *current* motion would press one device and release the
    /// other — leaving a button stuck down on the host.
    held_buttons: HashMap<ProtoButton, bool>,
    last_cursor_seq: Option<u32>,
}

impl UinputInjector {
    /// Create the three virtual devices. Fails with [`InjectError::Init`]
    /// if `/dev/uinput` isn't writable — the host then logs an actionable
    /// hint (`tether-host --setup-input`) and falls back to no-op.
    pub async fn connect() -> Result<Self> {
        let keys = keyboard_keycodes();
        let buttons = pointer_buttons();

        let keyboard = VirtualDevice::builder()
            .map_err(|e| InjectError::Init(open_uinput_error(&e)))?
            .name("Tether Virtual Keyboard")
            .input_id(InputId::new(BusType::BUS_USB, 0x1209, 0x7e70, 1))
            .with_keys(&keys)
            .map_err(|e| InjectError::Init(format!("keyboard keys: {e}")))?
            .build()
            .map_err(|e| InjectError::Init(format!("keyboard build: {e}")))?;

        let mut rel_axes = AttributeSet::<RelativeAxisCode>::new();
        for axis in [
            RelativeAxisCode::REL_X,
            RelativeAxisCode::REL_Y,
            RelativeAxisCode::REL_WHEEL,
            RelativeAxisCode::REL_HWHEEL,
            RelativeAxisCode::REL_WHEEL_HI_RES,
            RelativeAxisCode::REL_HWHEEL_HI_RES,
        ] {
            rel_axes.insert(axis);
        }
        let rel_pointer = VirtualDevice::builder()
            .map_err(|e| InjectError::Init(open_uinput_error(&e)))?
            .name("Tether Virtual Pointer")
            .input_id(InputId::new(BusType::BUS_USB, 0x1209, 0x7e71, 1))
            .with_keys(&buttons)
            .map_err(|e| InjectError::Init(format!("rel pointer keys: {e}")))?
            .with_relative_axes(&rel_axes)
            .map_err(|e| InjectError::Init(format!("rel pointer axes: {e}")))?
            .build()
            .map_err(|e| InjectError::Init(format!("rel pointer build: {e}")))?;

        // Wheels on the absolute device too, so scroll works while the
        // pointer is in absolute mode.
        let mut abs_wheels = AttributeSet::<RelativeAxisCode>::new();
        for axis in [
            RelativeAxisCode::REL_WHEEL,
            RelativeAxisCode::REL_HWHEEL,
            RelativeAxisCode::REL_WHEEL_HI_RES,
            RelativeAxisCode::REL_HWHEEL_HI_RES,
        ] {
            abs_wheels.insert(axis);
        }
        let mut props = AttributeSet::<PropType>::new();
        props.insert(PropType::POINTER);
        let abs_x = UinputAbsSetup::new(
            AbsoluteAxisCode::ABS_X,
            AbsInfo::new(0, 0, ABS_MAX, 0, 0, 0),
        );
        let abs_y = UinputAbsSetup::new(
            AbsoluteAxisCode::ABS_Y,
            AbsInfo::new(0, 0, ABS_MAX, 0, 0, 0),
        );
        let abs_pointer = VirtualDevice::builder()
            .map_err(|e| InjectError::Init(open_uinput_error(&e)))?
            .name("Tether Virtual Pointer (absolute)")
            .input_id(InputId::new(BusType::BUS_USB, 0x1209, 0x7e72, 1))
            .with_properties(&props)
            .map_err(|e| InjectError::Init(format!("abs pointer props: {e}")))?
            .with_keys(&buttons)
            .map_err(|e| InjectError::Init(format!("abs pointer keys: {e}")))?
            .with_relative_axes(&abs_wheels)
            .map_err(|e| InjectError::Init(format!("abs pointer wheels: {e}")))?
            .with_absolute_axis(&abs_x)
            .map_err(|e| InjectError::Init(format!("abs pointer ABS_X: {e}")))?
            .with_absolute_axis(&abs_y)
            .map_err(|e| InjectError::Init(format!("abs pointer ABS_Y: {e}")))?
            .build()
            .map_err(|e| InjectError::Init(format!("abs pointer build: {e}")))?;

        tracing::info!(session = "wayland (uinput)", "linux input backend selected");
        Ok(Self {
            keyboard,
            rel_pointer,
            abs_pointer,
            last_motion_absolute: false,
            held_keys: HashSet::new(),
            held_buttons: HashMap::new(),
            last_cursor_seq: None,
        })
    }

    fn host_modifiers(&self) -> Modifiers {
        Modifiers {
            shift: self.held_keys.contains(&HID_LSHIFT) || self.held_keys.contains(&HID_RSHIFT),
            ctrl: self.held_keys.contains(&HID_LCTRL) || self.held_keys.contains(&HID_RCTRL),
            alt: self.held_keys.contains(&HID_LALT) || self.held_keys.contains(&HID_RALT),
            meta: self.held_keys.contains(&HID_LMETA) || self.held_keys.contains(&HID_RMETA),
        }
    }

    /// Bring host modifier state in line with the wire's `Modifiers`
    /// snapshot by synthesising press/release for any disagreeing bit.
    /// `skip` covers both sides of the modifier the caller is about to
    /// press/release itself, avoiding a double-press. Mirrors the enigo
    /// backend's `reconcile_modifiers`.
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
            if let Some(code) = hid_to_evdev(hid) {
                let value = i32::from(wire_bit);
                self.emit_keyboard(&[key_ev(code, value), syn()])?;
                if wire_bit {
                    self.held_keys.insert(hid);
                } else {
                    self.held_keys.remove(&hid);
                    self.held_keys
                        .remove(&modifier_right_variant(hid).unwrap_or(hid));
                }
                tracing::trace!(?bit, press = wire_bit, "synthesised modifier to match wire");
            }
        }
        Ok(())
    }

    fn emit_keyboard(&mut self, events: &[InputEvent]) -> Result<()> {
        self.keyboard
            .emit(events)
            .map_err(|e| InjectError::Inject(format!("keyboard emit: {e}")))
    }

    /// The pointer device that should carry buttons / scroll right now:
    /// whichever one last moved the cursor.
    fn active_pointer(&mut self) -> &mut VirtualDevice {
        if self.last_motion_absolute {
            &mut self.abs_pointer
        } else {
            &mut self.rel_pointer
        }
    }

    fn emit_active_pointer(&mut self, events: &[InputEvent]) -> Result<()> {
        self.active_pointer()
            .emit(events)
            .map_err(|e| InjectError::Inject(format!("pointer emit: {e}")))
    }

    /// Type a Unicode string by reverse-mapping each character to a
    /// US-QWERTY keystroke. Characters with no physical-key equivalent
    /// (non-ASCII, host non-US layout) can't be expressed through uinput
    /// and are dropped with a warning — see the module docs.
    fn inject_text(&mut self, text: &str) -> Result<()> {
        for c in text.chars() {
            let Some(ks) = char_to_keystroke(c) else {
                tracing::warn!(
                    char = %c,
                    "uinput backend can't type this character (non-ASCII / non-US layout); dropping"
                );
                continue;
            };
            // Add a transient Shift only if the character needs it and
            // the operator isn't already holding one. We never release a
            // user-held Shift here, so tracked modifier state is intact.
            let temp_shift = ks.shift && !self.host_modifiers().shift;
            // The key-down and key-up must land in *separate* SYN frames:
            // a press+release of the same key in one frame is ambiguous to
            // libinput/xkb (it can collapse to a single event). Shift-down
            // rides the press frame, shift-up the release frame — exactly
            // how a physical shifted keystroke reports.
            let mut events: Vec<InputEvent> = Vec::with_capacity(6);
            if temp_shift {
                events.push(key_ev(KeyCode::KEY_LEFTSHIFT, 1));
            }
            events.push(key_ev(ks.code, 1));
            events.push(syn());
            events.push(key_ev(ks.code, 0));
            if temp_shift {
                events.push(key_ev(KeyCode::KEY_LEFTSHIFT, 0));
            }
            events.push(syn());
            self.emit_keyboard(&events)?;
        }
        Ok(())
    }
}

impl Injector for UinputInjector {
    fn inject(&mut self, evt: &ProtoInputEvent) -> Result<()> {
        match &evt.kind {
            InputEventKind::KeyDown { key, modifiers } => {
                self.reconcile_modifiers(*modifiers, modifier_bit_of(*key))?;
                if let Some(code) = hid_to_evdev(*key) {
                    self.emit_keyboard(&[key_ev(code, 1), syn()])?;
                    self.held_keys.insert(*key);
                }
                Ok(())
            }
            InputEventKind::KeyUp { key, modifiers } => {
                self.reconcile_modifiers(*modifiers, modifier_bit_of(*key))?;
                if let Some(code) = hid_to_evdev(*key) {
                    self.emit_keyboard(&[key_ev(code, 0), syn()])?;
                    self.held_keys.remove(key);
                }
                Ok(())
            }
            InputEventKind::Text { utf8 } => self.inject_text(utf8),
            InputEventKind::MouseButton {
                button,
                pressed,
                modifiers,
            } => {
                self.reconcile_modifiers(*modifiers, None)?;
                let btn = proto_button_to_btn(*button);
                // Down goes to the device matching current motion; up goes
                // to the device the matching down used, so a press/release
                // pair always targets one device even if motion mode
                // flipped in between.
                let on_abs = if *pressed {
                    let on_abs = self.last_motion_absolute;
                    self.held_buttons.insert(*button, on_abs);
                    on_abs
                } else {
                    self.held_buttons
                        .remove(button)
                        .unwrap_or(self.last_motion_absolute)
                };
                let device = if on_abs {
                    &mut self.abs_pointer
                } else {
                    &mut self.rel_pointer
                };
                device
                    .emit(&[key_ev(btn, i32::from(*pressed)), syn()])
                    .map_err(|e| InjectError::Inject(format!("button emit: {e}")))?;
                Ok(())
            }
            InputEventKind::RelativeMouseMove { dx, dy, modifiers } => {
                self.reconcile_modifiers(*modifiers, None)?;
                let dx = i32::from(*dx).clamp(-MAX_REL_DELTA_PX, MAX_REL_DELTA_PX);
                let dy = i32::from(*dy).clamp(-MAX_REL_DELTA_PX, MAX_REL_DELTA_PX);
                self.last_motion_absolute = false;
                self.rel_pointer
                    .emit(&[
                        rel_ev(RelativeAxisCode::REL_X, dx),
                        rel_ev(RelativeAxisCode::REL_Y, dy),
                        syn(),
                    ])
                    .map_err(|e| InjectError::Inject(format!("relative move emit: {e}")))
            }
            InputEventKind::MouseScroll {
                dx,
                dy,
                kind,
                modifiers,
            } => {
                self.reconcile_modifiers(*modifiers, None)?;
                let (vert, horiz) = scroll_detents(*dx, *dy, *kind);
                let mut events: Vec<InputEvent> = Vec::with_capacity(5);
                if vert != 0 {
                    events.push(rel_ev(RelativeAxisCode::REL_WHEEL, vert));
                    events.push(rel_ev(
                        RelativeAxisCode::REL_WHEEL_HI_RES,
                        vert * WHEEL_HI_RES_PER_DETENT,
                    ));
                }
                if horiz != 0 {
                    events.push(rel_ev(RelativeAxisCode::REL_HWHEEL, horiz));
                    events.push(rel_ev(
                        RelativeAxisCode::REL_HWHEEL_HI_RES,
                        horiz * WHEEL_HI_RES_PER_DETENT,
                    ));
                }
                if events.is_empty() {
                    return Ok(());
                }
                events.push(syn());
                self.emit_active_pointer(&events)
            }
        }
    }

    fn inject_cursor(&mut self, cursor: &ClientCursorPacket) -> Result<()> {
        // Drop reordered datagrams (standard wrapping "is newer?" test).
        if let Some(last) = self.last_cursor_seq {
            if cursor.seq.wrapping_sub(last) >= (u32::MAX / 2) {
                tracing::trace!(seq = cursor.seq, last, "out-of-order cursor; dropping");
                return Ok(());
            }
        }
        self.last_cursor_seq = Some(cursor.seq);

        if !cursor.visible {
            // No "hide host cursor" verb yet; the host renders its own
            // OS cursor independently.
            return Ok(());
        }
        if cursor.display_idx != 0 {
            tracing::trace!(
                display_idx = cursor.display_idx,
                "non-zero display_idx; pinning to primary"
            );
        }
        let x = normalised_to_abs(cursor.x);
        let y = normalised_to_abs(cursor.y);
        self.last_motion_absolute = true;
        self.abs_pointer
            .emit(&[
                abs_ev(AbsoluteAxisCode::ABS_X, x),
                abs_ev(AbsoluteAxisCode::ABS_Y, y),
                syn(),
            ])
            .map_err(|e| InjectError::Inject(format!("cursor emit: {e}")))
    }
}

impl Drop for UinputInjector {
    fn drop(&mut self) {
        // Release anything still held so a sudden disconnect doesn't
        // leave the host stuck on Ctrl / a mouse button.
        for hid in self.held_keys.drain().collect::<Vec<_>>() {
            if let Some(code) = hid_to_evdev(hid) {
                let _ = self.keyboard.emit(&[key_ev(code, 0), syn()]);
            }
        }
        for (btn, on_abs) in self.held_buttons.drain().collect::<Vec<_>>() {
            let code = proto_button_to_btn(btn);
            // Release on the same device the press used.
            let device = if on_abs {
                &mut self.abs_pointer
            } else {
                &mut self.rel_pointer
            };
            let _ = device.emit(&[key_ev(code, 0), syn()]);
        }
    }
}

/// Convert a normalised (0.0–1.0) coordinate to an absolute axis value.
fn normalised_to_abs(v: f32) -> i32 {
    let clamped = v.clamp(0.0, 1.0);
    // f32 → i32: clamped * ABS_MAX is in [0, 65535], well within range.
    #[allow(clippy::cast_possible_truncation)]
    {
        (clamped * ABS_MAX as f32).round() as i32
    }
}

/// Upper bound on detents per scroll event. Far beyond any real wheel
/// flick, but small enough that `detents * WHEEL_HI_RES_PER_DETENT`
/// can't overflow `i32` — a hostile client could otherwise send
/// `dy = f32::INFINITY`, which saturates the cast to `i32::MAX` and then
/// panics (debug) or wraps (release) on the hi-res multiply.
const MAX_SCROLL_DETENTS: i32 = i32::MAX / WHEEL_HI_RES_PER_DETENT;

/// Quantise wire scroll deltas to whole wheel detents, matching the
/// previous libei backend's behaviour. Returns `(vertical, horizontal)`
/// detents with kernel sign convention (REL_WHEEL +ve = up, REL_HWHEEL
/// +ve = right). winit's `dy` is +ve when scrolling up, so no negation.
fn scroll_detents(dx: f32, dy: f32, kind: ScrollKind) -> (i32, i32) {
    let scale = match kind {
        ScrollKind::Line => 1.0,
        // Pixel deltas (trackpad) are far larger per event; quantise to
        // detents with the same 1/16 factor the libei backend used.
        ScrollKind::Pixel => 1.0 / 16.0,
    };
    // The `as i32` cast saturates NaN→0 and ±inf→i32::MIN/MAX; the clamp
    // then bounds the result so the caller's hi-res multiply is safe.
    #[allow(clippy::cast_possible_truncation)]
    let to_detents =
        |d: f32| ((d * scale).round() as i32).clamp(-MAX_SCROLL_DETENTS, MAX_SCROLL_DETENTS);
    (to_detents(dy), to_detents(dx))
}

/// Friendly message for the common `/dev/uinput` permission failure.
fn open_uinput_error(e: &std::io::Error) -> String {
    if e.kind() == std::io::ErrorKind::PermissionDenied {
        format!(
            "cannot open /dev/uinput ({e}); run `tether-host --setup-input` once to grant access"
        )
    } else {
        format!("open /dev/uinput: {e}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hid_letters_map_to_correct_evdev_keys() {
        assert_eq!(
            hid_to_evdev(HidUsage((0x07 << 16) | 0x04)),
            Some(KeyCode::KEY_A)
        );
        assert_eq!(
            hid_to_evdev(HidUsage((0x07 << 16) | 0x1D)),
            Some(KeyCode::KEY_Z)
        );
    }

    #[test]
    fn hid_digits_and_named_keys() {
        assert_eq!(
            hid_to_evdev(HidUsage((0x07 << 16) | 0x1E)),
            Some(KeyCode::KEY_1)
        );
        assert_eq!(
            hid_to_evdev(HidUsage((0x07 << 16) | 0x27)),
            Some(KeyCode::KEY_0)
        );
        assert_eq!(
            hid_to_evdev(HidUsage((0x07 << 16) | 0x28)),
            Some(KeyCode::KEY_ENTER)
        );
        assert_eq!(
            hid_to_evdev(HidUsage((0x07 << 16) | 0x2B)),
            Some(KeyCode::KEY_TAB)
        );
        assert_eq!(
            hid_to_evdev(HidUsage((0x07 << 16) | 0x29)),
            Some(KeyCode::KEY_ESC)
        );
    }

    #[test]
    fn hid_arrows_and_modifiers() {
        assert_eq!(
            hid_to_evdev(HidUsage((0x07 << 16) | 0x4F)),
            Some(KeyCode::KEY_RIGHT)
        );
        assert_eq!(
            hid_to_evdev(HidUsage((0x07 << 16) | 0x52)),
            Some(KeyCode::KEY_UP)
        );
        assert_eq!(hid_to_evdev(HID_LCTRL), Some(KeyCode::KEY_LEFTCTRL));
        assert_eq!(hid_to_evdev(HID_RMETA), Some(KeyCode::KEY_RIGHTMETA));
    }

    #[test]
    fn non_keyboard_hid_page_is_none() {
        // Page 0x0C (Consumer) is not handled by the keyboard table.
        assert_eq!(hid_to_evdev(HidUsage((0x0C << 16) | 0xE9)), None);
    }

    #[test]
    fn char_to_keystroke_basic_ascii() {
        let a = char_to_keystroke('a').unwrap();
        assert_eq!(a.code, KeyCode::KEY_A);
        assert!(!a.shift);

        let cap = char_to_keystroke('A').unwrap();
        assert_eq!(cap.code, KeyCode::KEY_A);
        assert!(cap.shift);

        let bang = char_to_keystroke('!').unwrap();
        assert_eq!(bang.code, KeyCode::KEY_1);
        assert!(bang.shift);

        let space = char_to_keystroke(' ').unwrap();
        assert_eq!(space.code, KeyCode::KEY_SPACE);
        assert!(!space.shift);
    }

    #[test]
    fn char_to_keystroke_rejects_non_ascii() {
        assert!(char_to_keystroke('é').is_none());
        assert!(char_to_keystroke('λ').is_none());
        assert!(char_to_keystroke('🎉').is_none());
    }

    #[test]
    fn letter_keycode_roundtrips_alphabet() {
        for c in 'a'..='z' {
            let code = letter_keycode(c);
            assert_ne!(code, KeyCode::KEY_RESERVED, "missing keycode for {c}");
        }
    }

    #[test]
    fn keyboard_keycodes_includes_letters_and_modifiers() {
        let set = keyboard_keycodes();
        assert!(set.contains(KeyCode::KEY_A));
        assert!(set.contains(KeyCode::KEY_LEFTSHIFT));
        assert!(set.contains(KeyCode::KEY_F1));
    }

    #[test]
    fn scroll_line_is_not_negated() {
        // Positive wire dy (scroll up) must stay positive for REL_WHEEL.
        let (vert, _) = scroll_detents(0.0, 3.0, ScrollKind::Line);
        assert_eq!(vert, 3);
        let (vert_down, _) = scroll_detents(0.0, -2.0, ScrollKind::Line);
        assert_eq!(vert_down, -2);
    }

    #[test]
    fn scroll_extreme_deltas_cannot_overflow_hi_res_multiply() {
        // A hostile client sending dy = INFINITY / f32::MAX must not be
        // able to make `detents * WHEEL_HI_RES_PER_DETENT` overflow.
        for d in [f32::INFINITY, f32::MAX, f32::NEG_INFINITY, f32::MIN] {
            let (vert, horiz) = scroll_detents(d, d, ScrollKind::Line);
            // Bounded so the hi-res multiply stays in range.
            assert!(vert.abs() <= MAX_SCROLL_DETENTS);
            assert!(horiz.abs() <= MAX_SCROLL_DETENTS);
            assert!(vert.checked_mul(WHEEL_HI_RES_PER_DETENT).is_some());
            assert!(horiz.checked_mul(WHEEL_HI_RES_PER_DETENT).is_some());
        }
        // NaN saturates to 0, not a panic.
        assert_eq!(scroll_detents(f32::NAN, f32::NAN, ScrollKind::Line), (0, 0));
    }

    #[test]
    fn scroll_pixel_quantises_to_detents() {
        // 32 px at 1/16 → 2 detents.
        let (vert, _) = scroll_detents(0.0, 32.0, ScrollKind::Pixel);
        assert_eq!(vert, 2);
        // Sub-detent pixel motion rounds to zero (caller skips it).
        let (small, _) = scroll_detents(0.0, 4.0, ScrollKind::Pixel);
        assert_eq!(small, 0);
    }

    #[test]
    fn normalised_to_abs_maps_edges_and_centre() {
        assert_eq!(normalised_to_abs(0.0), 0);
        assert_eq!(normalised_to_abs(1.0), ABS_MAX);
        // 0.5 → mid-scale (0xFFFF * 0.5 = 32767.5, rounds to 32768).
        assert_eq!(normalised_to_abs(0.5), 32768);
        // Out-of-range clamps.
        assert_eq!(normalised_to_abs(-1.0), 0);
        assert_eq!(normalised_to_abs(2.0), ABS_MAX);
    }
}
