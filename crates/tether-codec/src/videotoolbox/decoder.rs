//! VideoToolbox hardware video decoder. Codec-parameterized via
//! [`CodecKind`]: H.264 selects the `h264` decoder with VideoToolbox
//! hwaccel, HEVC selects the `hevc` decoder. Both paths share the
//! CVPixelBuffer → IOSurface export → renderer pipeline; only the
//! AVCodecID differs.
//!
//! Mirrors the shape of the VAAPI decoder so the client's decode loop
//! stays backend-agnostic. The interesting differences vs VAAPI:
//!
//! - No explicit surface export: VT hands back an `AVFrame` whose
//!   `data[3]` is already a `CVPixelBufferRef` (the reverse of the
//!   encoder's IOSurface→CVPixelBuffer wrapping). We pull the
//!   IOSurface out with `CVPixelBufferGetIOSurface` and pass the raw
//!   pointer through to the renderer.
//! - No explicit sync call: the FFmpeg VT wrapper blocks inside
//!   `receive_frame` until the surface is ready, so by the time we
//!   hand the IOSurface out the GPU side is done writing.
//! - Lifetime: the `AVFrame` itself is the release guard — its
//!   `Drop` calls `av_frame_unref` which drops the +1 retain on the
//!   `CVPixelBufferRef` in `buf[0]`, which in turn releases the
//!   IOSurface.

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{AVFrame, AVHWDeviceContext};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use rsmpeg::UnsafeDerefMut;
use tracing::warn;

use tether_protocol::control::CodecKind;

use crate::h264::packet_from_bytes;
use crate::{
    init_ffmpeg, CodecError, Decoder, Frame, GpuFrame, GpuFrameSource, IOSurfaceFrame, Result,
};

use super::ffi::{
    CVPixelBufferGetHeight, CVPixelBufferGetIOSurface, CVPixelBufferGetPixelFormatType,
    CVPixelBufferGetWidth, CVPixelBufferRef,
};
use super::DECODE_EXTRA_HW_FRAMES;

/// VideoToolbox-accelerated video decoder. Uses ffmpeg's generic
/// `h264` or `hevc` decoder with VideoToolbox hwaccel selected via the
/// `get_format` callback. Decoded output rides through as a
/// `Frame::Gpu(IOSurface(..))`; the CPU NV12 path is treated as an
/// error to keep parity with VAAPI's "no SW fallback" contract.
pub struct VideoToolboxDecoder {
    kind: CodecKind,
    decoder: AVCodecContext,
    // Keeping our own ref documents the lifetime relationship; the
    // decoder also holds a cloned ref via `set_hw_device_ctx`. Drop
    // order: `decoder` drops first, releasing all surfaces, then the
    // device context goes.
    _hw_device: AVHWDeviceContext,
}

// SAFETY: same move-fine / share-bad rationale as `VideoToolboxEncoder`
// and `VaapiDecoder`. AVCodecContext is move-safe but `&mut`-only
// across threads; we never share a reference, only ownership.
unsafe impl Send for VideoToolboxDecoder {}

impl VideoToolboxDecoder {
    /// Construct a VideoToolbox decoder for the given codec.
    /// `Err(CodecError::CodecNotFound)` if the decoder isn't compiled
    /// in or this build's decoder doesn't advertise VideoToolbox
    /// hwaccel support for that codec.
    pub fn new(kind: CodecKind) -> Result<Self> {
        init_ffmpeg();

        let codec_id = vt_av_codec_id(kind)?;
        let codec = AVCodec::find_decoder(codec_id)
            .ok_or(CodecError::CodecNotFound(vt_decoder_name(kind)))?;

        // Walk the decoder's HW config table to confirm VideoToolbox is
        // actually supported in this ffmpeg build. Without this probe an
        // unsupported config would fail at `decoder.open()` with a less
        // actionable error. Same shape as the VAAPI sibling.
        let mut vt_supported = false;
        for i in 0.. {
            let Some(config) = codec.hw_config(i) else {
                break;
            };
            #[allow(clippy::cast_possible_wrap)] // single-bit constant
            let supports_device_ctx =
                config.methods & ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32 != 0;
            if supports_device_ctx && config.device_type == ffi::AV_HWDEVICE_TYPE_VIDEOTOOLBOX {
                vt_supported = true;
                break;
            }
        }
        if !vt_supported {
            return Err(CodecError::CodecNotFound(vt_decoder_name(kind)));
        }

        let hw_device =
            AVHWDeviceContext::create(ffi::AV_HWDEVICE_TYPE_VIDEOTOOLBOX, None, None, 0)?;

        let mut decoder = AVCodecContext::new(&codec);
        decoder.set_hw_device_ctx(hw_device.clone());
        decoder.set_get_format(Some(get_videotoolbox_format));
        // Size the internal hwframes pool with headroom so the
        // decoder doesn't stall while the renderer's LatestFrame cell
        // is still holding the previous IOSurface guard.
        // SAFETY: extra_hw_frames must be written before
        // avcodec_open2 — that's the libavcodec ordering invariant
        // the unsafe here attests to.
        unsafe {
            decoder.deref_mut().extra_hw_frames = DECODE_EXTRA_HW_FRAMES;
        }
        decoder.open(None)?;

        Ok(Self {
            kind,
            decoder,
            _hw_device: hw_device,
        })
    }

    /// Pull the IOSurface backing `frame.data[3]` and wrap the whole
    /// `AVFrame` as the release guard. Lifetime contract: the AVFrame
    /// owns a +1 retain on the `CVPixelBufferRef` in `buf[0]`; the
    /// CVPixelBuffer in turn keeps the IOSurface alive. Dropping the
    /// `AVFrame` releases both.
    #[allow(clippy::cast_sign_loss)] // i32 width/height from ffmpeg are non-negative
    fn export_videotoolbox_frame(&self, frame: AVFrame) -> Result<Frame> {
        let pts_out = if frame.pts == ffi::AV_NOPTS_VALUE {
            None
        } else {
            Some(frame.pts)
        };

        // libavutil's VideoToolbox hwaccel parks the `CVPixelBufferRef`
        // in `data[3]` (same channel as the VAAPI surface ID, just a
        // different opaque type). A null pointer here would mean ffmpeg
        // either violated its own contract or fell into a format
        // renegotiation path; surface as a recoverable error so the
        // decode thread doesn't panic on adversarial input.
        let pixbuf = frame.data[3] as CVPixelBufferRef;
        if pixbuf.is_null() {
            warn!("VideoToolbox frame missing CVPixelBufferRef in data[3]");
            return Err(CodecError::UnsupportedInputFormat);
        }

        // SAFETY: pixbuf is non-null (checked above) and was just
        // produced by libavcodec; it stays valid until the AVFrame's
        // buf[0] releases it, which happens when the guard drops.
        let (surface, pixel_format, width, height) = unsafe {
            let surface = CVPixelBufferGetIOSurface(pixbuf);
            let fourcc = CVPixelBufferGetPixelFormatType(pixbuf);
            let w = CVPixelBufferGetWidth(pixbuf);
            let h = CVPixelBufferGetHeight(pixbuf);
            (surface, fourcc, w, h)
        };
        // A non-IOSurface-backed CVPixelBuffer would be a CPU buffer —
        // contradicts the hwaccel contract but again, treat as a
        // recoverable format error rather than aborting.
        if surface.is_null() {
            warn!("VideoToolbox CVPixelBuffer is not IOSurface-backed");
            return Err(CodecError::UnsupportedInputFormat);
        }

        let iosurface = IOSurfaceFrame {
            surface,
            pixel_format,
            width: u32::try_from(width).expect("width fits in u32"),
            height: u32::try_from(height).expect("height fits in u32"),
        };

        // The AVFrame is the release guard. rsmpeg doesn't mark
        // AVFrame Send (raw ptr fields); same move-fine wrapper as the
        // VAAPI sibling. We move ownership across the channel; the
        // renderer's only interaction with the guard is to drop it.
        #[allow(dead_code)]
        struct SendFrame(AVFrame);
        // SAFETY: ownership is moved across the channel boundary, not
        // shared. av_frame_unref's reentrancy means it's safe to call
        // from whichever thread ends up dropping the box.
        unsafe impl Send for SendFrame {}

        Ok(Frame::Gpu(GpuFrame::new(
            iosurface.width,
            iosurface.height,
            pts_out,
            GpuFrameSource::IOSurface(iosurface),
            SendFrame(frame),
        )))
    }
}

impl VideoToolboxDecoder {
    /// Signal end-of-stream to libavcodec so any frames the decoder
    /// was holding internally get drained on the next `next_frame`
    /// call. Used by the capability probe path to extract a frame
    /// from a single-IDR fixture submit (VT's wrapper otherwise
    /// requires either another packet or EOF before emitting).
    ///
    /// `pub` rather than `pub(crate)` for the same reason
    /// `VideoToolboxEncoder::flush` is `pub`: cross-crate hardware
    /// tests need to drain a short stream, even though the production
    /// `Decoder` trait deliberately omits EOF (the live recv loop
    /// runs until disconnect, never EOFs).
    pub fn signal_eof(&mut self) -> Result<()> {
        self.decoder.send_packet(None)?;
        Ok(())
    }
}

impl Decoder for VideoToolboxDecoder {
    fn submit(&mut self, encoded: &[u8]) -> Result<()> {
        if encoded.is_empty() {
            return Ok(());
        }
        let packet = packet_from_bytes(encoded)?;
        self.decoder.send_packet(Some(&packet))?;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        let frame = match self.decoder.receive_frame() {
            Ok(f) => f,
            Err(RsmpegError::DecoderDrainError | RsmpegError::DecoderFlushedError) => {
                return Ok(None)
            }
            Err(e) => return Err(CodecError::Ffmpeg(e)),
        };

        if frame.format == ffi::AV_PIX_FMT_VIDEOTOOLBOX {
            return self.export_videotoolbox_frame(frame).map(Some);
        }

        // FFmpeg fell back to a software pixel format despite our
        // `get_format` callback insisting on VideoToolbox. Same
        // contract as the VAAPI decoder: refuse rather than silently
        // running CPU decode at 4K. The av_log bridge will have
        // logged the underlying reason.
        warn!(
            format = frame.format,
            "VideoToolboxDecoder produced a software frame; refusing per HW-only contract"
        );
        Err(CodecError::UnsupportedInputFormat)
    }

    fn codec_kind(&self) -> CodecKind {
        self.kind
    }

    fn is_hardware(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        vt_decoder_name(self.kind)
    }

    fn flush(&mut self) -> Result<()> {
        // Same contract as VAAPI: drop reorder queue + pending
        // output, keep the codec context alive. The VT
        // CVPixelBufferPool stays allocated; refs to its surfaces
        // are released by the flush.
        self.decoder.flush_buffers();
        Ok(())
    }
}

/// FFmpeg `AVCodecID` for the given codec kind. Errors for codecs we
/// don't have a VideoToolbox decode path for yet.
fn vt_av_codec_id(kind: CodecKind) -> Result<ffi::AVCodecID> {
    match kind {
        CodecKind::H264 => Ok(ffi::AV_CODEC_ID_H264),
        CodecKind::Hevc => Ok(ffi::AV_CODEC_ID_HEVC),
        // AV1 is held out of the macOS path on both sides until we
        // can hardware-verify decode end-to-end. M3 and M4 generation
        // silicon ships AV1 hardware decode and FFmpeg has the
        // `videotoolbox_av1` hwaccel (`libavcodec/videotoolbox_av1.c`)
        // to drive it, but tether ships no hardware test for the
        // path — advertising decode-only would negotiate AV1 against
        // a Linux encoder host and route a codepath we haven't
        // validated. Return `CodecNotFound` here so the probe marks
        // AV1 as `Decode`-stage Unsupported on macOS; pair with the
        // matching encoder.rs arm so AV1 disappears from both
        // `host_decode_profiles()` and `host_encode_profiles()` on
        // this platform.
        CodecKind::Av1 => Err(CodecError::CodecNotFound(
            "av1 VideoToolbox decode (held out pending hardware verification)",
        )),
    }
}

/// Human-readable decoder name for logs / `Decoder::name`.
fn vt_decoder_name(kind: CodecKind) -> &'static str {
    match kind {
        CodecKind::H264 => "h264 (VideoToolbox hw)",
        CodecKind::Hevc => "hevc (VideoToolbox hw)",
        // Unreachable: `vt_av_codec_id(Av1)` errors before this is
        // consulted. Kept exhaustive so a future re-enable only
        // needs the codec-id arm + a hardware test.
        CodecKind::Av1 => "av1 (VideoToolbox hw — disabled)",
    }
}

/// `get_format` callback invoked once the bitstream's parameter sets
/// reveal the format options. The candidate array is null-terminated
/// and includes `AV_PIX_FMT_VIDEOTOOLBOX` because we set
/// `hw_device_ctx`; we pick it (signalling "yes, allocate
/// IOSurface-backed CVPixelBuffers"). Returning `AV_PIX_FMT_NONE`
/// causes ffmpeg to fall back to its software pixel format selection;
/// the `next_frame` SW-frame guard then rejects those with
/// `UnsupportedInputFormat`, preserving the HW-only contract.
unsafe extern "C" fn get_videotoolbox_format(
    _ctx: *mut ffi::AVCodecContext,
    pix_fmts: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    let fmts = unsafe { rsmpeg::build_array(pix_fmts, ffi::AV_PIX_FMT_NONE) }.unwrap_or_default();
    for &fmt in fmts {
        if fmt == ffi::AV_PIX_FMT_VIDEOTOOLBOX {
            return fmt;
        }
    }
    ffi::AV_PIX_FMT_NONE
}
