//! Bits of encoder logic shared between every hardware backend
//! (VAAPI on Linux, VideoToolbox on macOS, future NVENC/QSV/…).
//!
//! Right now the only resident is `drain_encoder`, which yields all
//! packets currently buffered in an `AVCodecContext` and prepends the
//! encoder's `extradata` (Annex-B SPS/PPS/VPS) to every keyframe so
//! every IDR is self-decodable. This is the streaming-friendly contract
//! Tether's raw wire format requires: a client that joins mid-session,
//! rebuilds its decoder (resume, resolution change), or loses the
//! session's first IDR has no recovery path otherwise.
//!
//! Cost: one allocation per keyframe of `extradata.len()` bytes
//! (~25 bytes H.264, ~50 bytes HEVC). P-frames pass through untouched.

use std::slice;

use bytes::{Bytes, BytesMut};
use rsmpeg::avcodec::AVCodecContext;
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;

use crate::{CodecError, EncodedPacket, Result};

#[allow(clippy::cast_sign_loss)]
pub(crate) fn drain_encoder(
    encoder: &mut AVCodecContext,
    extradata: &[u8],
) -> Result<Vec<EncodedPacket>> {
    let mut out = Vec::new();
    loop {
        let packet = match encoder.receive_packet() {
            Ok(p) => p,
            Err(RsmpegError::EncoderDrainError | RsmpegError::EncoderFlushedError) => break,
            Err(e) => return Err(CodecError::Ffmpeg(e)),
        };
        let size = packet.size as usize;
        // SAFETY: packet.data points to packet.size valid bytes owned
        // by the AVPacket; we copy them before drop.
        let raw = unsafe { slice::from_raw_parts(packet.data, size) };
        let keyframe = (packet.flags & ffi::AV_PKT_FLAG_KEY as i32) != 0;
        let data: Bytes = if keyframe && !extradata.is_empty() {
            let mut buf = BytesMut::with_capacity(extradata.len() + raw.len());
            buf.extend_from_slice(extradata);
            buf.extend_from_slice(raw);
            buf.freeze()
        } else {
            Bytes::copy_from_slice(raw)
        };
        let pts_out = if packet.pts == ffi::AV_NOPTS_VALUE {
            None
        } else {
            Some(packet.pts)
        };
        out.push(EncodedPacket {
            data,
            pts: pts_out,
            keyframe,
        });
    }
    Ok(out)
}

/// Snapshot `AVCodecContext::extradata` into an owned `Vec<u8>`. Call
/// this once immediately after `encoder.open()` has succeeded with
/// `AV_CODEC_FLAG_GLOBAL_HEADER` set. Returns
/// `Err(CodecError::NoHardwareCodec(..))` if libavcodec did not populate
/// extradata — without it, keyframes won't carry SPS/PPS and any client
/// that loses the first IDR or rebuilds its decoder mid-session is
/// permanently stuck. That breaks Tether's self-decodable-IDR invariant,
/// so we fail loudly at encoder construction rather than silently
/// continuing.
///
/// SAFETY: libavcodec populates `extradata` inside `open()` when
/// `AV_CODEC_FLAG_GLOBAL_HEADER` is set, and does not mutate it
/// mid-stream for the fixed-resolution encoders we ship (the hardware
/// encoders rebuild from scratch on resolution change). Copying into an
/// owned buffer immediately prevents any subsequent encoder operation
/// from racing with our read.
#[allow(clippy::cast_sign_loss)]
pub(crate) fn snapshot_extradata(encoder: &AVCodecContext, codec_name: &str) -> Result<Vec<u8>> {
    let extradata = unsafe {
        let raw = encoder.extradata;
        let size = encoder.extradata_size;
        if raw.is_null() || size <= 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(raw, size as usize).to_vec()
        }
    };
    if extradata.is_empty() {
        return Err(CodecError::NoHardwareCodec(format!(
            "{codec_name}: encoder.extradata was empty after open() despite \
             AV_CODEC_FLAG_GLOBAL_HEADER. Keyframes would not carry SPS/PPS, \
             so any client that loses the first IDR or rebuilds its decoder \
             mid-session would be stuck. Verify the FFmpeg build honours \
             AV_CODEC_FLAG_GLOBAL_HEADER for this codec."
        )));
    }
    Ok(extradata)
}

/// Static debug label for `CodecError::ScalerInit`, keyed off the
/// destination `AVPixelFormat`. Shared between every hardware backend
/// so the table stays in one place — VAAPI's `vaapi_sw_format` and
/// VideoToolbox's `vt_sw_format` both feed their outputs here. Falls
/// back to a generic label for a future `*_sw_format` addition that
/// hasn't reached this table yet (the fallback keeps `ScalerInit` a
/// `&'static str` instead of a heap-allocating per-call format).
pub(crate) fn pix_fmt_scaler_label(sw_format: i32) -> &'static str {
    match sw_format {
        x if x == ffi::AV_PIX_FMT_NV12 => "BGRA -> NV12",
        x if x == ffi::AV_PIX_FMT_P010LE => "BGRA -> P010",
        x if x == ffi::AV_PIX_FMT_NV24 => "BGRA -> NV24",
        x if x == ffi::AV_PIX_FMT_P410LE => "BGRA -> P410",
        x if x == ffi::AV_PIX_FMT_VUYX => "BGRA -> VUYX",
        x if x == ffi::AV_PIX_FMT_XV30LE => "BGRA -> XV30",
        _ => "BGRA -> sw_format",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pix_fmt_scaler_label_covers_every_sw_format_in_use() {
        // Every format any `*_sw_format` function in the codec backends
        // returns must produce a specific label — not the generic
        // fallback. If you add a new sw_format to a backend, add the
        // case here too.
        for (fmt, expected) in [
            (ffi::AV_PIX_FMT_NV12, "BGRA -> NV12"),
            (ffi::AV_PIX_FMT_P010LE, "BGRA -> P010"),
            (ffi::AV_PIX_FMT_NV24, "BGRA -> NV24"),
            (ffi::AV_PIX_FMT_P410LE, "BGRA -> P410"),
            (ffi::AV_PIX_FMT_VUYX, "BGRA -> VUYX"),
            (ffi::AV_PIX_FMT_XV30LE, "BGRA -> XV30"),
        ] {
            assert_eq!(pix_fmt_scaler_label(fmt), expected);
        }
    }

    #[test]
    fn pix_fmt_scaler_label_falls_back_for_unknown_format() {
        assert_eq!(pix_fmt_scaler_label(-1), "BGRA -> sw_format");
    }
}
