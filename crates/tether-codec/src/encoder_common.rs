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
        let data = if keyframe && !extradata.is_empty() {
            let mut buf = Vec::with_capacity(extradata.len() + raw.len());
            buf.extend_from_slice(extradata);
            buf.extend_from_slice(raw);
            buf
        } else {
            raw.to_vec()
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
/// `AV_CODEC_FLAG_GLOBAL_HEADER` set. Returns an empty vec if libavcodec
/// did not populate extradata (a misconfiguration we surface as a
/// warning at the call site).
///
/// SAFETY: libavcodec populates `extradata` inside `open()` when
/// `AV_CODEC_FLAG_GLOBAL_HEADER` is set, and does not mutate it
/// mid-stream for the fixed-resolution encoders we ship (the hardware
/// encoders rebuild from scratch on resolution change). Copying into an
/// owned buffer immediately prevents any subsequent encoder operation
/// from racing with our read.
#[allow(clippy::cast_sign_loss)]
pub(crate) fn snapshot_extradata(encoder: &AVCodecContext) -> Vec<u8> {
    unsafe {
        let raw = encoder.extradata;
        let size = encoder.extradata_size;
        if raw.is_null() || size <= 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(raw, size as usize).to_vec()
        }
    }
}
