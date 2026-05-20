//! macOS input injection via `CGEventPost` (Quartz Event Services).
//!
//! enigo's macOS backend wraps `CGEventCreateMouseEvent` /
//! `CGEventCreateKeyboardEvent` / `CGEventCreateScrollWheelEvent` and
//! posts via `CGEventPost(kCGHIDEventTap, …)`. All the input-shaping
//! logic — modifier reconciliation, HID→`Key` mapping, held-key
//! cleanup on disconnect — is shared with the Linux (libei) backend
//! through [`super::enigo_backend::EnigoBackend`].
//!
//! Permission: the first event injection triggers the macOS
//! Accessibility TCC prompt. The host must be granted "Accessibility"
//! (not "Input Monitoring" — that controls *reading*, not posting).
//! enigo doesn't preflight; if the user denies, posts silently no-op
//! until accessibility is granted and the host is restarted. Same
//! first-run shape as the screen-capture prompt.

use enigo::{Enigo, Settings};

use tether_protocol::cursor::ClientCursorPacket;
use tether_protocol::input::InputEvent;

use super::enigo_backend::EnigoBackend;
use super::{InjectError, Injector, Result};

pub struct MacOsInjector {
    inner: EnigoBackend,
}

impl MacOsInjector {
    /// Construct the CGEvent-backed injector. Unlike the libei path
    /// there's no portal handshake — enigo opens the system event
    /// source synchronously. We still go through `spawn_blocking` for
    /// symmetry and to keep the async signature.
    pub async fn connect() -> Result<Self> {
        let settings = Settings::default();
        let enigo = tokio::task::spawn_blocking(move || Enigo::new(&settings))
            .await
            .map_err(|e| InjectError::Init(format!("spawn_blocking join: {e}")))?
            .map_err(|e| InjectError::Init(format!("enigo new: {e:?}")))?;
        // CGEvent construction succeeds regardless of the Accessibility
        // TCC grant — `CGEventPost` silently no-ops when the grant is
        // missing rather than returning an error, so an operator who
        // forgot to grant Accessibility sees a happy `connect()` and
        // events that go nowhere. Make the grant a load-bearing
        // post-condition by logging it at info every time, not just on
        // failure.
        tracing::info!(
            session = "macos (CGEvent)",
            "macOS input backend selected. If injected events do not appear, \
             verify the host has Accessibility access in \
             System Settings → Privacy & Security → Accessibility."
        );
        Ok(Self {
            inner: EnigoBackend::new(enigo),
        })
    }
}

impl Injector for MacOsInjector {
    fn inject(&mut self, evt: &InputEvent) -> Result<()> {
        self.inner.inject(evt)
    }
    fn inject_cursor(&mut self, cursor: &ClientCursorPacket) -> Result<()> {
        self.inner.inject_cursor(cursor)
    }
    fn set_display_size(&mut self, width: u32, height: u32) {
        self.inner.set_display_size(width, height);
    }
}
