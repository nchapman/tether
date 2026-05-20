//! Opaque "hold these refs alive" container for GPU resources.
//!
//! Both capture and codec backends need to keep platform-specific handles
//! alive while a consumer is reading a GPU buffer — PipeWire's
//! producer-side buffer (capture), an `AVFrame` whose `Drop` returns a
//! VAAPI surface to the pool (decode), ScreenCaptureKit's
//! `CMSampleBuffer` retain (future), etc.
//!
//! Same shape in every crate: a `Box<dyn Send + 'static>` behind a
//! sealed trait so consumers can't downcast or inspect. Defined here in
//! `tether-protocol` (already the de-facto common dep) so the two
//! producer crates that need it (capture, codec) share one definition
//! rather than two copies; `tether-render` consumes it transitively
//! via `tether-codec`'s re-export.

/// Sealed Drop carrier. Construct with [`GpuResourceGuard::new`]; the
/// stashed value is dropped when the guard is. Consumers can't inspect
/// or downcast.
pub struct GpuResourceGuard {
    // Held purely for its Drop.
    #[allow(dead_code)]
    inner: Box<dyn GuardPayload>,
}

impl GpuResourceGuard {
    /// Wrap any `Send + 'static` value so it's dropped when the guard
    /// is. Backends stash whatever per-frame handles they need to keep
    /// alive.
    pub fn new<T: Send + 'static>(payload: T) -> Self {
        Self {
            inner: Box::new(payload),
        }
    }
}

impl std::fmt::Debug for GpuResourceGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuResourceGuard").finish_non_exhaustive()
    }
}

/// Sealed marker for anything a backend wants to keep alive for the
/// lifetime of a GPU frame. Blanket-impl'd for every `Send + 'static`
/// type; the trait is private so no one outside this crate can
/// implement it or downcast through it.
trait GuardPayload: Send + 'static {}
impl<T: Send + 'static> GuardPayload for T {}
