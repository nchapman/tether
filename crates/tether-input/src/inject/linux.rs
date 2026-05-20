//! Linux input injection via libei (Wayland-native, portal-mediated).
//!
//! The `enigo` crate with the `libei_tokio` feature handles the portal
//! handshake against the Remote Desktop interface; the user sees a
//! permission prompt analogous to the screen-capture one. After that,
//! events go straight over the EI Unix socket to the compositor's input
//! sink. All the actual input-shaping logic — modifier reconciliation,
//! HID→`Key` mapping, held-key cleanup on disconnect — lives in
//! [`super::enigo_backend::EnigoBackend`] and is shared with the
//! macOS (CGEvent) backend.
//!
//! v0 limitations:
//!   - Display size is sampled once at connect time and never refreshed,
//!     so a monitor hot-plug or scale change between sessions will
//!     misplace the cursor until the operator reconnects.
//!   - Scroll is forwarded as integer ticks; the precise pixel-delta
//!     mode used by trackpads is rounded.

use std::ffi::CString;
use std::os::fd::RawFd;

use enigo::{Enigo, Settings};

use tether_protocol::cursor::ClientCursorPacket;
use tether_protocol::input::InputEvent;

use super::enigo_backend::EnigoBackend;
use super::{InjectError, Injector, Result};

pub struct LibeiInjector {
    inner: EnigoBackend,
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
    /// `enigo` 0.6.1's libei backend emits three raw `println!("Start
    /// emulating")` lines during construction — one per virtual device
    /// the portal exposes (keyboard, pointer, scroll). They're not
    /// gated by a logger, so we silence them by redirecting fd 1 to
    /// `/dev/null` for the duration of the call. Tracing output goes
    /// to fd 2 and is unaffected. Remove this wrapper once enigo
    /// upstream demotes those lines.
    pub async fn connect() -> Result<Self> {
        let settings = Settings::default();
        let enigo = tokio::task::spawn_blocking(move || {
            let _silence = SilenceStdout::new();
            Enigo::new(&settings)
        })
        .await
        .map_err(|e| InjectError::Init(format!("spawn_blocking join: {e}")))?
        .map_err(|e| InjectError::Init(format!("enigo new: {e:?}")))?;
        tracing::info!(
            session = "wayland (libei)",
            "linux input backend selected"
        );
        Ok(Self {
            inner: EnigoBackend::new(enigo),
        })
    }
}

impl Injector for LibeiInjector {
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

/// RAII guard that redirects process stdout to `/dev/null` for the
/// duration of its lifetime. Used exclusively to muzzle enigo 0.6.1's
/// raw `println!("Start emulating")` lines on libei session init —
/// see [`LibeiInjector::connect`].
///
/// fd 1 is process-global, so anything else trying to write to stdout
/// during this guard's lifetime also goes to `/dev/null`. tracing
/// writes to fd 2 by default, so logs are unaffected. The guard is
/// constructed on the `spawn_blocking` thread enigo runs on, which
/// keeps the window short and predictable.
///
/// If any `libc` call fails, the guard quietly becomes a no-op rather
/// than panicking — losing the muzzle is harmless; aborting startup
/// over a cosmetic log line would not be.
struct SilenceStdout {
    saved_fd: Option<RawFd>,
}

impl SilenceStdout {
    fn new() -> Self {
        // SAFETY: dup/open/dup2 on fd 1 are sound; we own the saved fd
        // and the devnull fd. On any error we leave fd 1 untouched and
        // return a no-op guard.
        unsafe {
            let saved = libc::dup(1);
            if saved < 0 {
                return Self { saved_fd: None };
            }
            let Ok(path) = CString::new("/dev/null") else {
                libc::close(saved);
                return Self { saved_fd: None };
            };
            let devnull = libc::open(path.as_ptr(), libc::O_WRONLY);
            if devnull < 0 {
                libc::close(saved);
                return Self { saved_fd: None };
            }
            if libc::dup2(devnull, 1) < 0 {
                libc::close(devnull);
                libc::close(saved);
                return Self { saved_fd: None };
            }
            libc::close(devnull);
            Self { saved_fd: Some(saved) }
        }
    }
}

impl Drop for SilenceStdout {
    fn drop(&mut self) {
        // SAFETY: saved_fd was produced by libc::dup on fd 1; restoring
        // it via dup2 is sound. Errors here are unrecoverable (we'd
        // leak the saved fd at worst) so we ignore them.
        if let Some(saved) = self.saved_fd.take() {
            unsafe {
                libc::dup2(saved, 1);
                libc::close(saved);
            }
        }
    }
}
