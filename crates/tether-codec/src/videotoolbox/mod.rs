//! VideoToolbox hardware video encode for macOS (Apple Silicon + Intel).
//!
//! Wraps FFmpeg's `h264_videotoolbox` / `hevc_videotoolbox` encoders
//! through rsmpeg's safe `AVHWDeviceContext` / `AVHWFramesContext` /
//! `hwframe_transfer_data` bindings. Same trait shape as the VAAPI
//! sibling — only the device-context type and the per-input-format
//! wrapping differ.
//!
//! Encoder pipeline:
//!
//!   BGRA &[u8] -> bgra AVFrame -> swscale NV12 sw_frame
//!                              -> av_hwframe_transfer_data -> VT surface
//!                              -> h264_videotoolbox encode -> H.264 packet
//!
//!   IOSurface  -> CVPixelBufferCreateWithIOSurface
//!              -> AVFrame::data[3] = CVPixelBufferRef (held by buf[0])
//!              -> h264_videotoolbox encode -> H.264 packet
//!
//! The IOSurface path matches what ScreenCaptureKit hands us in
//! `tether-capture::macos`: NV12 video-range (`'420v'`), so the
//! encoder's `sw_format` lines up without any conversion.
//!
//! Decoder mirrors the same shape: FFmpeg's `h264` / `hevc` decoders
//! with the `AV_HWDEVICE_TYPE_VIDEOTOOLBOX` hwaccel selected via the
//! `get_format` callback. Decoded output is an `AVFrame` whose
//! `data[3]` is a `CVPixelBufferRef` — the reverse of the encoder's
//! IOSurface→CVPixelBuffer wrap — and we hand the underlying IOSurface
//! straight to the renderer.

mod decoder;
pub mod encoder;
mod ffi;
pub mod probe;

#[cfg(test)]
mod tests;

pub use decoder::VideoToolboxDecoder;
pub use encoder::VideoToolboxEncoder;

/// Surfaces beyond what the decoder needs for its own reference
/// picture list. Same rationale as the VAAPI sibling: the renderer's
/// `LatestFrame` cell holds the previous decoded `IOSurface` (via the
/// `AVFrame` guard) until the next one lands, so the hwframes pool
/// needs spare slots or `receive_frame` stalls.
///
/// 8 is sized for HEVC headroom: HEVC L5.1 has a DPB up to 6 reference
/// pictures, and although the host disables B-frames, a remote peer
/// running a different encoder could legally send more references than
/// our own encoder produces. Cost is ~8 × (W×H×1.5) bytes of wired
/// memory per session — trivial. Matches Sunshine's setting for the
/// same reason.
pub(crate) const DECODE_EXTRA_HW_FRAMES: i32 = 8;
