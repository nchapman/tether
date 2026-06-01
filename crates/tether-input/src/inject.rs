//! Host-side input injection. Cross-platform `Injector` trait + a
//! per-platform backend selected at compile time. On Linux we drive
//! `/dev/uinput` directly (no portal — see [`uinput`]); on macOS / Windows
//! we drive `enigo` (CGEvent / SendInput). Targets without a real backend
//! fall through to `NoopInjector`, which just logs each event and returns
//! success.

use tether_protocol::cursor::ClientCursorPacket;
use tether_protocol::input::InputEvent;

#[derive(Debug, thiserror::Error)]
pub enum InjectError {
    /// The backend's input device can't be opened for use — on Linux
    /// `/dev/uinput` is either missing (the `uinput` module isn't loaded)
    /// or present but not writable (no udev rule yet). Distinct from
    /// [`InjectError::Init`] because it's *fixable by a one-time privileged
    /// setup*: the host reacts by requesting that grant through the GUI
    /// rather than treating it as a hard failure.
    #[error("input device unavailable: {0}")]
    DeviceUnavailable(String),
    #[error("backend init: {0}")]
    Init(String),
    #[error("backend inject: {0}")]
    Inject(String),
}

pub type Result<T> = std::result::Result<T, InjectError>;

/// Apply wire-level input to the host's local input system. Split into
/// two methods because the two paths arrive on independent wire
/// channels (reliable input stream vs. unreliable cursor datagram) and
/// often need different rate-limiting / coalescing strategies on the
/// host. Implementations are `Send` so the host can park them inside a
/// tokio task; they are NOT `Sync` since backends keep mutable state
/// (held keys, last cursor seq, last-known mouse position) per connection.
pub trait Injector: Send {
    fn inject(&mut self, evt: &InputEvent) -> Result<()>;

    /// Apply a cursor position from the client → host datagram channel.
    /// Backends should track the highest `seq` they've seen and drop
    /// any packet whose seq is older — datagrams reorder freely and a
    /// stale position after a fresher one would visibly snap the
    /// pointer backwards.
    fn inject_cursor(&mut self, cursor: &ClientCursorPacket) -> Result<()>;

    /// Tell the injector the host display's pixel dimensions so it can
    /// convert normalised cursor coords to absolute pixels. The capture
    /// path is the authoritative source — enigo's `main_display()` is
    /// `Not implemented yet` on libei/Wayland, so we feed dims from
    /// PipeWire/ScreenCaptureKit instead. Default impl is a no-op; only
    /// backends that need it (real pointer movers) override.
    fn set_display_size(&mut self, _width: u32, _height: u32) {}
}

/// Last-resort backend: just log the event at debug level and pretend
/// it succeeded. Useful in CI, in tests, and as the fallback when no
/// real backend compiled in for the current target.
pub struct NoopInjector;

impl Injector for NoopInjector {
    fn inject(&mut self, evt: &InputEvent) -> Result<()> {
        tracing::debug!(
            event_id = evt.event_id,
            kind = ?evt.kind,
            "noop injector: event discarded"
        );
        Ok(())
    }

    fn inject_cursor(&mut self, cursor: &ClientCursorPacket) -> Result<()> {
        tracing::trace!(
            seq = cursor.seq,
            x = cursor.x,
            y = cursor.y,
            visible = cursor.visible,
            "noop injector: cursor discarded"
        );
        Ok(())
    }
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
mod enigo_backend;

#[cfg(target_os = "linux")]
mod uinput;
#[cfg(target_os = "linux")]
pub use uinput::UinputInjector;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::MacOsInjector;

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::WindowsInjector;

/// Pick the best available backend for the current target. On Linux
/// [`UinputInjector`] drives `/dev/uinput` directly (no portal — the
/// `/dev/uinput` permission is the equivalent gate); on macOS
/// [`MacOsInjector`] drives CGEvent (the Accessibility TCC grant is the
/// gate). On unsupported targets we fall back to [`NoopInjector`] with a
/// single warn-level log so the operator notices.
///
/// The Linux host does *not* call this — it builds the injector through a
/// path that turns [`InjectError::DeviceUnavailable`] into a GUI
/// authorization prompt and retry (see `tether-host`'s `setup_input`).
/// This default stays the simple connect-or-noop for non-host callers.
pub async fn default_injector() -> Box<dyn Injector> {
    #[cfg(target_os = "linux")]
    {
        match UinputInjector::connect().await {
            Ok(inj) => {
                return Box::new(inj);
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "linux injector unavailable; falling back to noop. \
                     Input events will be logged but not applied. \
                     Most likely /dev/uinput is not writable — \
                     run `tether-host --setup-input` once to grant access."
                );
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        match MacOsInjector::connect().await {
            Ok(inj) => {
                return Box::new(inj);
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "macOS injector unavailable; falling back to noop. \
                     Input events will be logged but not applied. \
                     Most likely missing Accessibility TCC grant — \
                     System Settings → Privacy & Security → Accessibility."
                );
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        match WindowsInjector::connect().await {
            Ok(inj) => {
                return Box::new(inj);
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Windows injector unavailable; falling back to noop. \
                     Input events will be logged but not applied."
                );
            }
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    tracing::warn!("no input injection backend compiled for this target; using noop");
    Box::new(NoopInjector)
}
