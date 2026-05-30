//! Windows input injection via `SendInput` (Win32 User Input API).
//!
//! enigo's Windows backend wraps `SendInput` for keyboard and mouse
//! events. All the input-shaping logic — modifier reconciliation,
//! HID→`Key` mapping, held-key cleanup on disconnect — is shared with
//! the Linux (libei) and macOS (CGEvent) backends through
//! [`super::enigo_backend::EnigoBackend`].
//!
//! Permission: `SendInput` works without elevation for normal desktop
//! windows. Injecting into elevated (UAC) windows requires the host
//! process to run elevated or have a UIAccess manifest.

use enigo::{Enigo, Settings};

use tether_protocol::cursor::ClientCursorPacket;
use tether_protocol::input::InputEvent;

use super::enigo_backend::EnigoBackend;
use super::{InjectError, Injector, Result};

pub struct WindowsInjector {
    inner: EnigoBackend,
}

impl WindowsInjector {
    pub async fn connect() -> Result<Self> {
        let settings = Settings::default();
        let enigo = tokio::task::spawn_blocking(move || Enigo::new(&settings))
            .await
            .map_err(|e| InjectError::Init(format!("spawn_blocking join: {e}")))?
            .map_err(|e| InjectError::Init(format!("enigo new: {e:?}")))?;
        tracing::info!(
            session = "windows (SendInput)",
            "Windows input backend selected."
        );
        Ok(Self {
            inner: EnigoBackend::new(enigo),
        })
    }
}

impl Injector for WindowsInjector {
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
