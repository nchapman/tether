//! Screen capture for Tether.
//!
//! Each backend is a free function returning a
//! [`crossbeam_channel::Receiver`] of [`CapturedFrame`]s. Dropping the
//! receiver shuts the producer down. A `Capturer` trait will land once
//! we have multiple real backends that need runtime selection.
//!
//! Backends planned but not yet implemented:
//! - `linux` — PipeWire + xdg-desktop-portal (DMA-BUF zero-copy aim)
//! - `macos` — ScreenCaptureKit (NV12 zero-copy via CVPixelBuffer)
//!
//! [`test_pattern`] is always available and produces synthetic frames
//! at a fixed cadence. It exists so the walking skeleton and headless
//! tests can exercise the pipeline without a display server.

pub mod test_pattern;

use tether_protocol::MonoNanos;

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
/// v0 carries CPU-side owned bytes (`data: Vec<u8>`). Later iterations
/// will expose zero-copy variants (DMA-BUF fd on Linux, IOSurface on
/// macOS) so the encoder can ingest GPU-resident buffers without a
/// readback.
pub struct CapturedFrame {
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
}

pub type Result<T> = std::result::Result<T, CaptureError>;
