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
//! Decoder lands with the macOS client work; this module is encoder-only
//! today.

mod encoder;
mod ffi;

#[cfg(test)]
mod tests;

pub use encoder::VideoToolboxEncoder;
