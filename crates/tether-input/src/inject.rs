//! Host-side input injection. Cross-platform `Injector` trait + a
//! per-platform backend selected at compile time. On Linux we drive
//! `libei` via the `enigo` crate (`libei_tokio` feature), which performs
//! a Remote Desktop portal handshake — same model as our screen capture.
//! Targets without a real backend fall through to `NoopInjector`, which
//! just logs each event and returns success.

use tether_protocol::input::InputEvent;

#[derive(Debug, thiserror::Error)]
pub enum InjectError {
    #[error("backend init: {0}")]
    Init(String),
    #[error("backend inject: {0}")]
    Inject(String),
}

pub type Result<T> = std::result::Result<T, InjectError>;

/// Apply a wire-level [`InputEvent`] to the host's local input system.
/// Implementations are `Send` so the host can park them inside a tokio
/// task; they are NOT `Sync` since most backends keep mutable state
/// (held keys, last-known mouse position) per connection.
pub trait Injector: Send {
    fn inject(&mut self, evt: &InputEvent) -> Result<()>;
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
}

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::LibeiInjector;

/// Pick the best available backend for the current target. On Linux we
/// try `LibeiInjector` first (real injection via the portal); on
/// failure or on unsupported targets we fall back to `NoopInjector`
/// with a single warn-level log so the operator notices.
pub async fn default_injector() -> Box<dyn Injector> {
    #[cfg(target_os = "linux")]
    {
        match LibeiInjector::connect().await {
            Ok(inj) => {
                tracing::info!("input injector: libei (Wayland portal)");
                return Box::new(inj);
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "libei injector unavailable; falling back to noop. \
                     Input events will be logged but not applied."
                );
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    tracing::warn!(
        "no input injection backend compiled for this target; using noop"
    );
    Box::new(NoopInjector)
}
