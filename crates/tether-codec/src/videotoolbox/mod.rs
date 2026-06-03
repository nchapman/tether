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

#[cfg(test)]
mod tests;

pub use decoder::VideoToolboxDecoder;
pub use encoder::VideoToolboxEncoder;

use tether_protocol::control::{ChromaSubsampling, VideoProfile};

/// The set of IOSurface fourccs a VT decoder may emit for `profile`'s
/// bitstream, when the encoder *actually* produced that profile's
/// chroma + bit-depth (vs. silently downsampled).
///
/// Restricted to the **video-range** fourcc of each family. The
/// renderer shader is hardcoded BT.709 *limited* range, so it only
/// imports video-range surfaces (`macos_interop::accepts_iosurface_fourcc`),
/// and every tether host (VAAPI / Windows / macOS) tags its bitstream
/// video-range, so VT decode of a cross-platform stream lands in the
/// video-range label. A full-range surface (`'420f'` / `'xf20'` /
/// `'444f'`) therefore can't arise in a tether↔tether session and
/// wouldn't be displayable if it did, so listing it here would only
/// mask a real renderer-accept gap rather than describe a reachable
/// decode. The 4:4:4 10-bit family's video-range label is `'x444'`
/// (what a Linux host's stream decodes to); `'xf44'` / `'P410'`
/// remain listed for the probe-gated macOS encode path, which
/// produces the full-range label.
///
/// Exposed at `pub` so cross-crate consistency tests can confirm the
/// renderer's IOSurface accept set (`tether-render::gpu::metal`) is a
/// superset of this — a subset mismatch is what bit us in commit
/// `621badc` (renderer rejected `'x420'` even though the probe listed
/// it).
#[must_use]
pub fn expected_iosurface_fourccs(profile: VideoProfile) -> &'static [u32] {
    const NV12_VIDEO: u32 = u32::from_be_bytes(*b"420v");
    const NV24_VIDEO: u32 = u32::from_be_bytes(*b"444v");
    const P010: u32 = u32::from_be_bytes(*b"P010");
    const X420: u32 = u32::from_be_bytes(*b"x420");
    const X444: u32 = u32::from_be_bytes(*b"x444");
    const XF44: u32 = u32::from_be_bytes(*b"xf44");
    const P410: u32 = u32::from_be_bytes(*b"P410");
    match (profile.chroma, profile.bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => &[NV12_VIDEO],
        (ChromaSubsampling::Yuv420, 10) => &[P010, X420],
        (ChromaSubsampling::Yuv444, 8) => &[NV24_VIDEO],
        (ChromaSubsampling::Yuv444, 10) => &[X444, XF44, P410],
        _ => &[],
    }
}

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
