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

pub use cursor::{CursorEvent, CursorShapeEvent, CursorSource, PlaceholderCursorSource};
pub use damage::{DamageHint, DamageSignal, HashDamage, NativeDamage};

#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_os = "macos")]
pub mod macos;

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
