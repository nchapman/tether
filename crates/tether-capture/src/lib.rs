//! Screen capture for Tether.
//!
//! Each backend is a free function returning a
//! [`crossbeam_channel::Receiver`] of [`CapturedFrame`]s. Dropping the
//! receiver shuts the producer down. A `Capturer` trait will land once
//! we have multiple real backends that need runtime selection.
//!
//! Backends:
//! - [`linux`] — PipeWire + xdg-desktop-portal; advertises DMA-BUF and
//!   SHM alternatives and produces zero-copy DMA-BUF frames when the
//!   compositor agrees.
//! - [`macos`] — ScreenCaptureKit producing zero-copy IOSurface frames.
//! - [`test_pattern`] is always available and produces synthetic frames
//!   at a fixed cadence so the walking skeleton and headless tests can
//!   exercise the pipeline without a display server.

pub mod cursor;
pub mod damage;
pub mod test_pattern;

#[cfg(feature = "test-support")]
pub mod test_support;

pub use cursor::{
    rescale_shape_to_frame, CursorEvent, CursorPosition, CursorShapeEvent, CursorSource,
    PlaceholderCursorSource,
};
pub use damage::{DamageHint, DamageSignal, HashDamage, NativeDamage};

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use crossbeam_channel::{Receiver, RecvTimeoutError};

/// Capture-source handle returned by every backend's `start()`. Bundles
/// the [`CapturedFrame`] receiver with a runtime-mutable target-FPS
/// atomic so the ABR / quality-tier controller can throttle capture
/// without rebuilding the backend.
///
/// **Per-backend honouring is uneven.** [`test_pattern::start`] reads
/// the atomic on every produced frame and adjusts its sleep period
/// immediately. The Linux PipeWire and macOS ScreenCaptureKit backends
/// currently *accept* the atomic so the seam is end-to-end live, but
/// renegotiating the stream's actual frame interval requires backend-
/// specific work (PipeWire format renegotiation; SCK
/// `SCStreamConfiguration.minimumFrameInterval` update) that lands
/// per backend. Until then, `set_target_fps` updates the atomic
/// silently — readers see the new value but the produced cadence
/// does not change. The honouring matrix lands per-backend; check
/// the backend's module docs before assuming a write took effect.
pub struct CaptureHandle {
    /// Frame receiver. Take ownership via [`Self::into_rx`] or borrow
    /// via [`Self::rx`].
    rx: Receiver<CapturedFrame>,
    /// Target FPS the backend should aim for. `Arc` so the ABR
    /// controller can hold a clone and update it from a different
    /// thread.
    target_fps: Arc<AtomicU32>,
    /// Optional per-backend cursor source. Wayland/PipeWire fills
    /// this with a `SPA_META_Cursor` parser; macOS will fill it with
    /// an `NSCursor` poller; the test pattern leaves it `None` and
    /// the host falls back to [`PlaceholderCursorSource`].
    cursor_source: Option<Box<dyn CursorSource>>,
    /// Consumer-liveness token. Moves into the [`FrameReceiver`] on
    /// [`Self::into_rx`] so its strong count tracks whether the consumer
    /// still holds the receiver. A backend producer that keeps its own
    /// receiver clone (Windows drop-oldest eviction masks channel
    /// disconnection) watches [`Self::liveness`] to know when to stop;
    /// backends that rely on channel `Disconnected` ignore it.
    alive: Arc<()>,
}

impl CaptureHandle {
    /// Build a handle owning the receiver and seeded at `initial_fps`.
    /// Backends construct the atomic up front (so they can also read
    /// it from their producer thread) and pass it through.
    #[must_use]
    pub fn from_parts(rx: Receiver<CapturedFrame>, target_fps: Arc<AtomicU32>) -> Self {
        Self {
            rx,
            target_fps,
            cursor_source: None,
            alive: Arc::new(()),
        }
    }

    /// A weak handle to the consumer-liveness token. A backend producer
    /// holds this to detect when the consumer has dropped its
    /// [`FrameReceiver`] — necessary when the producer keeps its own
    /// receiver clone (which would otherwise mask channel disconnection).
    /// `strong_count() == 0` means the consumer is gone.
    #[must_use]
    pub fn liveness(&self) -> Weak<()> {
        Arc::downgrade(&self.alive)
    }

    /// Attach a cursor source to this handle. Called by the backend
    /// after `from_parts` once the producer thread is spawned.
    #[must_use]
    pub fn with_cursor_source(mut self, src: Box<dyn CursorSource>) -> Self {
        self.cursor_source = Some(src);
        self
    }

    /// Take the cursor source out, leaving `None`. The host pump
    /// consumes this once at session start.
    #[must_use]
    pub fn take_cursor_source(&mut self) -> Option<Box<dyn CursorSource>> {
        self.cursor_source.take()
    }

    /// Borrow the frame receiver — for callers that need to interleave
    /// receiving with shutdown checks.
    #[must_use]
    pub fn rx(&self) -> &Receiver<CapturedFrame> {
        &self.rx
    }

    /// Consume the handle, returning the [`FrameReceiver`]. The FPS atomic
    /// is dropped from the handle's side; clone via [`Self::fps_handle`]
    /// before calling this if the ABR controller needs to keep writing.
    /// The returned receiver carries the consumer-liveness token (see
    /// [`Self::liveness`]); hold it for as long as frames are wanted.
    #[must_use]
    pub fn into_rx(self) -> FrameReceiver {
        FrameReceiver {
            rx: self.rx,
            _alive: self.alive,
        }
    }

    /// Current target FPS as observed by the backend's producer
    /// thread. Reads are `Relaxed` — the value is advisory, not a
    /// synchronization primitive.
    #[must_use]
    pub fn target_fps(&self) -> u32 {
        self.target_fps.load(Ordering::Relaxed)
    }

    /// Update the target FPS. Backends pick up the new value on their
    /// next loop iteration (test_pattern) or via renegotiation
    /// (real backends — currently unimplemented; see struct docs).
    /// Values < 1 are clamped to 1.
    pub fn set_target_fps(&self, fps: u32) {
        self.target_fps.store(fps.max(1), Ordering::Relaxed);
    }

    /// Cheap clone of the atomic for the ABR controller to retain
    /// across the `into_rx` boundary.
    #[must_use]
    pub fn fps_handle(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.target_fps)
    }
}

/// The consuming end of a capture stream, returned by
/// [`CaptureHandle::into_rx`]. Wraps the [`CapturedFrame`] receiver and
/// carries the consumer-liveness token: while this is held, the backend
/// producer's [`CaptureHandle::liveness`] weak handle reports a non-zero
/// strong count. Dropping it both disconnects the channel and drops the
/// token, so producers can detect consumer shutdown either way.
pub struct FrameReceiver {
    rx: Receiver<CapturedFrame>,
    _alive: Arc<()>,
}

impl FrameReceiver {
    /// Wait up to `timeout` for the next frame. Mirrors
    /// [`crossbeam_channel::Receiver::recv_timeout`] exactly.
    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> std::result::Result<CapturedFrame, RecvTimeoutError> {
        self.rx.recv_timeout(timeout)
    }
}

#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "macos")]
pub mod cursor_macos;

#[cfg(target_os = "windows")]
pub mod cursor_windows;

#[cfg(target_os = "windows")]
pub mod windows;

use tether_protocol::MonoNanos;

/// Re-export of [`tether_protocol::GpuResourceGuard`]. Producers (the
/// capture backend) stash whatever they need to keep alive while the
/// consumer reads the buffer; consumers can't downcast or inspect.
pub use tether_protocol::GpuResourceGuard as GpuCapturedGuard;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// 8-bit-per-channel BGRA, little-endian per pixel (B, G, R, A).
    /// What ScreenCaptureKit and DXGI emit by default.
    Bgra8,
    /// 8-bit-per-channel RGBA. Used by `tether-render`'s passthrough.
    Rgba8,
    /// 8-bit Y plane followed by interleaved 8-bit Cb/Cr at half
    /// resolution in each axis. What the HEVC/H.264 hardware encoders
    /// want as input.
    Nv12,
}

/// A single captured frame from the host's display.
///
/// Two shapes: CPU-side owned bytes (the SHM fallback path), and a
/// platform-specific GPU handle (DMA-BUF on Linux, IOSurface on macOS
/// later, D3D11 shared handle on Windows later). The host's encode
/// path pattern-matches: GPU frames go through the gpuconvert + VAAPI
/// zero-copy pipeline; CPU frames fall through to `encode_bgra`.
///
/// Shape mirrors [`tether_codec::Frame`] / [`tether_codec::GpuFrame`]
/// for consistency — the producer and consumer end of the same
/// architectural split.
pub enum CapturedFrame {
    Cpu(CpuFrame),
    Gpu(GpuCapturedFrame),
}

/// CPU-resident captured frame (BGRA / RGBA / NV12 bytes). The SHM
/// fallback path and the test pattern produce these.
pub struct CpuFrame {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub data: Vec<u8>,
    /// Source timestamp from the capture API (e.g. `CVTimeStamp`, PipeWire
    /// `pts`). For backends that don't expose this, falls back to the
    /// userspace timestamp.
    pub t_capture_kernel: MonoNanos,
    /// Monotonic time at which our userspace code first observed the
    /// frame. Always populated.
    pub t_capture_userspace: MonoNanos,
    /// Backend-supplied damage hint, when the capture API exposes one.
    /// `None` means the backend has no opinion; the consumer falls back
    /// to the hash classifier. See [`damage::NativeDamage`].
    pub native_damage: Option<damage::NativeDamage>,
}

/// GPU-resident captured frame. The descriptor varies per platform via
/// [`GpuCapturedSource`]; a release guard keeps the producer's backing
/// buffer alive until the consumer is done with it (PipeWire's buffer
/// must stay queued back to the stream; ScreenCaptureKit's
/// `IOSurface` needs its `CMSampleBuffer` retained; etc.).
pub struct GpuCapturedFrame {
    pub width: u32,
    pub height: u32,
    pub source: GpuCapturedSource,
    pub t_capture_kernel: MonoNanos,
    pub t_capture_userspace: MonoNanos,
    /// Opaque "hold this alive while the consumer reads the buffer"
    /// container. Dropped by the consumer once it has either copied
    /// the data into encoder-owned memory (VAAPI dups the dma-buf
    /// internally during `vaCreateSurfaces`) or otherwise no longer
    /// needs the source.
    pub release_guard: GpuCapturedGuard,
    /// Backend-supplied damage hint; mirrors [`CpuFrame::native_damage`].
    pub native_damage: Option<damage::NativeDamage>,
}

/// Per-platform GPU buffer descriptor. Gated on `target_os` so the
/// host's match is exhaustive on each platform without a catch-all
/// that silently swallows future variants — same pattern as
/// [`tether_codec::GpuFrameSource`].
pub enum GpuCapturedSource {
    /// Linux DMA-BUF (typically from PipeWire DMA-BUF buffer-type).
    /// Single-plane BGRx/BGRA from the compositor; multi-plane
    /// negotiation will be a future addition when there's a
    /// compositor known to produce multi-plane capture buffers.
    #[cfg(target_os = "linux")]
    DmaBuf(CapturedDmaBuf),
    /// macOS IOSurface from ScreenCaptureKit's `CMSampleBuffer`. The
    /// real CFRetain on the underlying `IOSurfaceRef` (and the
    /// `CMSampleBuffer` keeping it alive) lives in the parent
    /// [`GpuCapturedFrame::release_guard`] — the pointer here is a
    /// non-owning view, valid until the guard is dropped.
    #[cfg(target_os = "macos")]
    IOSurface(CapturedIOSurface),
    /// Windows D3D11 texture from DXGI Desktop Duplication. The
    /// texture is an owned pool copy (the duplication surface is
    /// released immediately after `CopyResource`). Carries a
    /// reference to the shared device so downstream consumers can
    /// operate without cross-device copies.
    #[cfg(target_os = "windows")]
    D3D11Texture(windows::CapturedD3D11Texture),
}

/// Linux DMA-BUF descriptor for a captured frame. Mirrors what
/// `tether_codec::DmaBufObject + DmaBufLayer` carry for a single-plane
/// surface; kept separate so `tether-capture` doesn't depend on
/// `tether-codec` (capture/encode stay decoupled).
#[cfg(target_os = "linux")]
pub struct CapturedDmaBuf {
    /// DRM fourcc of the source plane (typically `XR24`/`AR24`/`XB24`
    /// etc.) as supplied by PipeWire's negotiated format.
    pub fourcc: u32,
    pub fd: std::os::fd::OwnedFd,
    pub stride: u64,
    pub offset: u64,
    pub modifier: u64,
}

/// macOS IOSurface descriptor for a captured frame. Mirrors the shape
/// `tether_codec::IOSurfaceFrame` carries for the encoder side; kept
/// separate so `tether-capture` doesn't depend on `tether-codec` (same
/// rationale as [`CapturedDmaBuf`] vs `DmaBufFrame`).
///
/// The pointer is a non-owning view; lifetime is the parent
/// [`GpuCapturedFrame::release_guard`], which retains the
/// `CMSampleBuffer` (and transitively the IOSurface). Dropping the
/// guard releases both.
#[cfg(target_os = "macos")]
pub struct CapturedIOSurface {
    /// `IOSurfaceRef` — opaque Apple type, valid until the parent
    /// `release_guard` is dropped.
    pub surface: *mut std::ffi::c_void,
    /// `kCVPixelFormatType_*` fourcc as returned by
    /// `IOSurfaceGetPixelFormat`. Typically NV12
    /// (`420YpCbCr8BiPlanarVideoRange` = `'420v'`).
    pub pixel_format: u32,
    pub width: u32,
    pub height: u32,
}

// No `&mut` access is possible to the IOSurface through this raw
// pointer from Rust; all mutation goes through Apple's IOSurface C
// API, which is itself thread-safe (CF-style refcounted, kernel
// surface thread-shareable). The struct carries no Rust state that
// would conflict with crossing a thread boundary.
#[cfg(target_os = "macos")]
unsafe impl Send for CapturedIOSurface {}

impl CapturedFrame {
    #[must_use]
    pub fn width(&self) -> u32 {
        match self {
            Self::Cpu(f) => f.width,
            Self::Gpu(f) => f.width,
        }
    }
    #[must_use]
    pub fn height(&self) -> u32 {
        match self {
            Self::Cpu(f) => f.height,
            Self::Gpu(f) => f.height,
        }
    }
    /// `(t_capture_kernel, t_capture_userspace)` — populated for both
    /// variants. The host's timing-metric path doesn't care which
    /// shape produced the frame.
    #[must_use]
    pub fn timestamps(&self) -> (MonoNanos, MonoNanos) {
        match self {
            Self::Cpu(f) => (f.t_capture_kernel, f.t_capture_userspace),
            Self::Gpu(f) => (f.t_capture_kernel, f.t_capture_userspace),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("capture backend not available on this platform")]
    Unsupported,
    #[error("portal: {0}")]
    Portal(String),
    #[error("pipewire: {0}")]
    PipeWire(String),
    /// ScreenCaptureKit (macOS) error — typically a permission denial
    /// (`NSScreenCaptureUsageDescription` missing or TCC denied),
    /// `SCShareableContent::get` failure, or `start_capture` rejection.
    /// Carries the framework error's display form.
    #[error("ScreenCaptureKit: {0}")]
    Sck(String),
    /// DXGI Desktop Duplication (Windows) error — typically
    /// `DuplicateOutput` failure, access lost (driver reset / mode
    /// change), or D3D11 device creation failure.
    #[cfg(target_os = "windows")]
    #[error("DXGI: {0}")]
    Dxgi(String),
}

pub type Result<T> = std::result::Result<T, CaptureError>;

#[cfg(target_os = "linux")]
impl From<ashpd::Error> for CaptureError {
    fn from(e: ashpd::Error) -> Self {
        Self::Portal(e.to_string())
    }
}

#[cfg(target_os = "linux")]
impl From<pipewire::Error> for CaptureError {
    fn from(e: pipewire::Error) -> Self {
        Self::PipeWire(e.to_string())
    }
}

#[cfg(target_os = "macos")]
impl From<screencapturekit::error::SCError> for CaptureError {
    fn from(e: screencapturekit::error::SCError) -> Self {
        Self::Sck(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::bounded;

    // The Windows drop-oldest producer keeps its own receiver clone to
    // evict the stale mailbox frame, which masks channel `Disconnected`.
    // It therefore shuts down off the liveness token instead. These
    // tests pin the exact contract that producer relies on.

    #[test]
    fn liveness_tracks_frame_receiver_lifetime() {
        let (_tx, rx) = bounded::<CapturedFrame>(1);
        let handle = CaptureHandle::from_parts(rx, Arc::new(AtomicU32::new(60)));

        let weak = handle.liveness();
        assert_eq!(weak.strong_count(), 1, "handle holds the liveness token");

        let frames = handle.into_rx();
        assert_eq!(
            weak.strong_count(),
            1,
            "token moves into the FrameReceiver — still a live consumer"
        );

        drop(frames);
        assert_eq!(
            weak.strong_count(),
            0,
            "dropping the receiver signals the consumer is gone"
        );
    }

    #[test]
    fn liveness_drops_when_handle_discarded_without_into_rx() {
        let (_tx, rx) = bounded::<CapturedFrame>(1);
        let handle = CaptureHandle::from_parts(rx, Arc::new(AtomicU32::new(60)));
        let weak = handle.liveness();

        drop(handle);
        assert_eq!(
            weak.strong_count(),
            0,
            "a handle dropped without into_rx means no consumer ever materialised"
        );
    }
}
