//! D3D11VA hardware video decoder. Uses FFmpeg's generic `h264` or
//! `hevc` decoder with D3D11VA hwaccel for GPU-accelerated decode.
//!
//! Phase 1: decoded surfaces are downloaded to CPU NV12 via
//! `av_hwframe_transfer_data` and returned as `Frame::Cpu`. This
//! validates the full pipeline without needing D3D11→wgpu texture
//! import on the renderer side.
//!
//! Phase 2: return `Frame::Gpu` with a D3D11 texture handle and let
//! the renderer import it zero-copy.

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{AVFrame, AVHWDeviceContext};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use rsmpeg::UnsafeDerefMut;

use tether_protocol::control::CodecKind;

use crate::h264::packet_from_bytes;
use crate::{init_ffmpeg, CodecError, DecodedFrame, Decoder, Frame, Result};

const DECODE_EXTRA_HW_FRAMES: i32 = 4;

pub struct D3D11Decoder {
    kind: CodecKind,
    decoder: AVCodecContext,
    _hw_device: AVHWDeviceContext,
}

unsafe impl Send for D3D11Decoder {}

impl D3D11Decoder {
    pub fn new(kind: CodecKind) -> Result<Self> {
        init_ffmpeg();

        let codec_id = d3d11_av_codec_id(kind)?;
        let codec = AVCodec::find_decoder(codec_id)
            .ok_or(CodecError::CodecNotFound(d3d11_decoder_name(kind)))?;

        // Verify D3D11VA hwaccel is supported for this codec.
        let mut d3d11va_supported = false;
        for i in 0.. {
            let Some(config) = codec.hw_config(i) else { break };
            #[allow(clippy::cast_possible_wrap)]
            let supports_device_ctx =
                config.methods & ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32 != 0;
            if supports_device_ctx && config.device_type == ffi::AV_HWDEVICE_TYPE_D3D11VA {
                d3d11va_supported = true;
                break;
            }
        }
        if !d3d11va_supported {
            return Err(CodecError::CodecNotFound(d3d11_decoder_name(kind)));
        }

        let hw_device =
            AVHWDeviceContext::create(ffi::AV_HWDEVICE_TYPE_D3D11VA, None, None, 0)?;

        let mut decoder = AVCodecContext::new(&codec);
        decoder.set_hw_device_ctx(hw_device.clone());
        decoder.set_get_format(Some(get_d3d11va_format));
        unsafe {
            decoder.deref_mut().extra_hw_frames = DECODE_EXTRA_HW_FRAMES;
        }
        decoder.open(None)?;

        tracing::info!(
            codec = d3d11_decoder_name(kind),
            "D3D11VA decoder opened"
        );

        Ok(Self {
            kind,
            decoder,
            _hw_device: hw_device,
        })
    }

    /// Transfer a D3D11VA surface to CPU NV12 bytes (tight-packed).
    #[allow(clippy::cast_sign_loss)]
    fn download_frame(&self, hw_frame: &AVFrame) -> Result<Frame> {
        let mut sw_frame = AVFrame::new();
        sw_frame.set_format(ffi::AV_PIX_FMT_NV12);

        let rc = unsafe {
            ffi::av_hwframe_transfer_data(sw_frame.as_mut_ptr(), hw_frame.as_ptr(), 0)
        };
        if rc < 0 {
            return Err(CodecError::Ffmpeg(RsmpegError::AVError(rc)));
        }

        let width = sw_frame.width as u32;
        let height = sw_frame.height as u32;
        let y_stride = unsafe { (*sw_frame.as_ptr()).linesize[0] as usize };
        let uv_stride = unsafe { (*sw_frame.as_ptr()).linesize[1] as usize };
        let y_ptr = sw_frame.data[0];
        let uv_ptr = sw_frame.data[1];

        let w = width as usize;
        let h = height as usize;
        let chroma_h = (h + 1) / 2;
        let chroma_w = (w + 1) / 2;

        // Copy Y plane, removing stride padding if present.
        let mut y = Vec::with_capacity(w * h);
        for row in 0..h {
            let src = unsafe { y_ptr.add(row * y_stride) };
            y.extend_from_slice(unsafe { std::slice::from_raw_parts(src, w) });
        }

        // Copy UV plane (interleaved Cb/Cr, width = chroma_w * 2).
        let uv_row_bytes = chroma_w * 2;
        let mut uv = Vec::with_capacity(uv_row_bytes * chroma_h);
        for row in 0..chroma_h {
            let src = unsafe { uv_ptr.add(row * uv_stride) };
            uv.extend_from_slice(unsafe { std::slice::from_raw_parts(src, uv_row_bytes) });
        }

        let pts = if sw_frame.pts == ffi::AV_NOPTS_VALUE {
            None
        } else {
            Some(sw_frame.pts)
        };

        Ok(Frame::Cpu(DecodedFrame {
            width,
            height,
            y,
            uv,
            pts,
        }))
    }
}

impl Decoder for D3D11Decoder {
    fn submit(&mut self, encoded: &[u8]) -> Result<()> {
        let packet = packet_from_bytes(encoded)?;
        match self.decoder.send_packet(Some(&packet)) {
            Ok(()) => Ok(()),
            Err(RsmpegError::DecoderFlushedError) => Ok(()),
            Err(e) => Err(CodecError::Ffmpeg(e)),
        }
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        match self.decoder.receive_frame() {
            Ok(frame) => {
                let decoded = self.download_frame(&frame)?;
                Ok(Some(decoded))
            }
            Err(RsmpegError::DecoderDrainError) => Ok(None),
            Err(e) => Err(CodecError::Ffmpeg(e)),
        }
    }

    fn codec_kind(&self) -> CodecKind {
        self.kind
    }

    fn is_hardware(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        match self.kind {
            CodecKind::H264 => "h264 (d3d11va)",
            CodecKind::Hevc => "hevc (d3d11va)",
            CodecKind::Av1 => "av1 (d3d11va)",
        }
    }
}

fn d3d11_av_codec_id(kind: CodecKind) -> Result<ffi::AVCodecID> {
    Ok(match kind {
        CodecKind::H264 => ffi::AV_CODEC_ID_H264,
        CodecKind::Hevc => ffi::AV_CODEC_ID_HEVC,
        CodecKind::Av1 => {
            return Err(CodecError::CodecNotFound("av1 d3d11va decoder"));
        }
    })
}

fn d3d11_decoder_name(kind: CodecKind) -> &'static str {
    match kind {
        CodecKind::H264 => "h264 (d3d11va)",
        CodecKind::Hevc => "hevc (d3d11va)",
        CodecKind::Av1 => "av1 (d3d11va)",
    }
}

/// FFmpeg get_format callback: select D3D11VA when offered.
unsafe extern "C" fn get_d3d11va_format(
    _ctx: *mut ffi::AVCodecContext,
    pix_fmts: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    let mut p = pix_fmts;
    unsafe {
        while *p != ffi::AV_PIX_FMT_NONE {
            if *p == ffi::AV_PIX_FMT_D3D11 {
                return ffi::AV_PIX_FMT_D3D11;
            }
            p = p.add(1);
        }
    }
    ffi::AV_PIX_FMT_NONE
}
