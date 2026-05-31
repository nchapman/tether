//! VAAPI hardware video decoder. Codec-parameterized via
//! [`CodecKind`]: H.264 selects the `h264` decoder with VAAPI
//! hwaccel, HEVC selects the `hevc` decoder. Both paths share the
//! VAAPI surface export → DMA-BUF → renderer pipeline; only the
//! AVCodecID differs.

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{AVFrame, AVHWDeviceContext};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use rsmpeg::UnsafeDerefMut;
use tracing::warn;

use tether_protocol::control::CodecKind;

use crate::h264::packet_from_bytes;
use crate::{
    init_ffmpeg, CodecError, Decoder, DmaBufFrame, DmaBufLayer, DmaBufObject, Frame, GpuFrame,
    GpuFrameSource, Result,
};

use super::ffi::AVVAAPIDeviceContext;
use super::DECODE_EXTRA_HW_FRAMES;

/// VAAPI-accelerated video decoder. Uses ffmpeg's generic `h264` or
/// `hevc` decoder with VAAPI hwaccel selected via `get_format`.
/// Output VAAPI surfaces are exported as DMA-BUFs (zero copy) and
/// handed straight to the renderer; the CPU NV12 path is a safety
/// net for unexpected SW fallback.
pub struct VaapiDecoder {
    kind: CodecKind,
    decoder: AVCodecContext,
    // Decoder holds a cloned ref to the device internally
    // (set_hw_device_ctx); keeping our own ref documents the
    // lifetime relationship and matches what hw_decode.c does.
    // Drop order: `decoder` drops first (declaration order), so all
    // surfaces allocated against the device are released before the
    // device context itself goes.
    _hw_device: AVHWDeviceContext,
}

// SAFETY: same move-fine / share-bad rationale as VaapiEncoder.
unsafe impl Send for VaapiDecoder {}

impl VaapiDecoder {
    /// Construct a VAAPI decoder for the given codec.
    /// `Err(CodecError::CodecNotFound)` if the decoder isn't compiled
    /// in or this build's decoder doesn't advertise VAAPI hwaccel
    /// support for that codec. Any other failure (device open,
    /// get_format callback rejected) comes through as
    /// `Err(CodecError::Ffmpeg)`.
    pub fn new(kind: CodecKind) -> Result<Self> {
        init_ffmpeg();

        let codec_id = vaapi_av_codec_id(kind)?;
        let codec = AVCodec::find_decoder(codec_id)
            .ok_or(CodecError::CodecNotFound(vaapi_decoder_name(kind)))?;

        // Walk the decoder's HW config table to confirm VAAPI is
        // actually supported in this ffmpeg build. Without this probe
        // an unsupported config would fail at decoder.open() with a
        // less actionable error.
        let mut vaapi_supported = false;
        for i in 0.. {
            let Some(config) = codec.hw_config(i) else { break };
            #[allow(clippy::cast_possible_wrap)] // single-bit constant
            let supports_device_ctx =
                config.methods & ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32 != 0;
            if supports_device_ctx && config.device_type == ffi::AV_HWDEVICE_TYPE_VAAPI {
                vaapi_supported = true;
                break;
            }
        }
        if !vaapi_supported {
            return Err(CodecError::CodecNotFound(vaapi_decoder_name(kind)));
        }

        let hw_device =
            AVHWDeviceContext::create(ffi::AV_HWDEVICE_TYPE_VAAPI, None, None, 0)?;

        let mut decoder = AVCodecContext::new(&codec);
        // Cloning an AVHWDeviceContext is an av_buffer_ref under the
        // hood (see rsmpeg/src/avutil/buffer.rs); the decoder gets a
        // fresh ref-counted handle while we keep our own owner.
        decoder.set_hw_device_ctx(hw_device.clone());
        decoder.set_get_format(Some(get_vaapi_format));
        // Tell ffmpeg to size its internal hwframes pool with room for
        // surfaces we hand to the renderer. rsmpeg doesn't wrap this
        // field; AVCodecContext derefs to ffi::AVCodecContext so we
        // poke it directly.
        // SAFETY: extra_hw_frames must be written before avcodec_open2
        // — that's the libavcodec ordering invariant the unsafe here
        // attests to.
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

    /// Export the VAAPI surface backing `frame` as a DMA-BUF and wrap
    /// it in a `Frame::Gpu`. The AVFrame is moved into the
    /// `GpuFrameGuard`: when the renderer drops the `GpuFrame`, the
    /// AVFrame's ref-count hits zero and the surface returns to the
    /// hwframes pool. `frame.data[3]` carries the `VASurfaceID`
    /// (libavutil convention); `frame.hw_frames_ctx -> data ->
    /// device_ctx -> hwctx` is the `AVVAAPIDeviceContext` whose
    /// `display` field is the `VADisplay` we need to pass to libva.
    #[allow(clippy::cast_sign_loss)] // i32 width/height from ffmpeg are non-negative
    fn export_vaapi_frame(&self, frame: AVFrame) -> Result<Frame> {
        let width = frame.width;
        let height = frame.height;
        let pts_out = if frame.pts == ffi::AV_NOPTS_VALUE {
            None
        } else {
            Some(frame.pts)
        };

        // SAFETY: frame is a freshly received decoded frame; for any
        // VAAPI surface ffmpeg sets hw_frames_ctx to the pool's
        // buffer ref, and data[3] holds the VASurfaceID as an integer
        // sentinel (cast through usize). Both invariants are part of
        // libavutil's hwaccel contract. Null fields would mean ffmpeg
        // violated that contract — treat as unreachable rather than a
        // user-facing format mismatch.
        let (display, surface_id) = unsafe {
            let hwf_ref = frame.hw_frames_ctx;
            assert!(!hwf_ref.is_null(), "VAAPI frame missing hw_frames_ctx");
            let hwf = (*hwf_ref).data as *const ffi::AVHWFramesContext;
            // `device_ctx` on AVHWFramesContext is the unwrapped
            // pointer (sibling `device_ref` is the AVBufferRef*),
            // so we deref it directly without another `.data` step.
            let dev = (*hwf).device_ctx;
            assert!(!dev.is_null(), "VAAPI frame's hwframes ctx missing device_ctx");
            let vactx = (*dev).hwctx as *const AVVAAPIDeviceContext;
            let display = (*vactx).display;
            // VASurfaceID is unsigned int; libavutil stores it in
            // data[3] by casting through uintptr_t.
            let surface_id = frame.data[3] as usize as tether_vaapi::VASurfaceID;
            (display, surface_id)
        };

        // Wait for the decode to actually finish before handing the
        // surface out. dma-buf implicit sync via the reservation object
        // only works when the producer attaches a write fence, and
        // not every libva backend does. Cost is microseconds when the
        // surface is already done (the common case at 30 fps where
        // decode lags submit by 0).
        // SAFETY: display + surface_id come from the same AVFrame
        // whose buffer ref keeps both alive.
        if let Err(e) = unsafe { tether_vaapi::sync_surface(display, surface_id) } {
            warn!(error = %e, "vaSyncSurface failed; surface export may race decode");
            return Err(CodecError::SurfaceExportFailed(e));
        }

        // READ_ONLY because we're a *consumer* of the decoded surface;
        // the encoder side (Sunshine's reference) uses WRITE_ONLY for
        // the opposite reason. SEPARATE_LAYERS gives one DRM layer per
        // plane (Y, UV) which is what wgpu's external-memory import
        // wants for multi-planar formats.
        // SAFETY: display and surface_id are derived from the
        // currently-live AVFrame above; both stay valid until that
        // frame's buffer ref is dropped, which happens after this fn
        // returns and after the GpuFrameGuard releases.
        let prime = unsafe {
            tether_vaapi::export_surface_handle(
                display,
                surface_id,
                tether_vaapi::VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                tether_vaapi::VA_EXPORT_SURFACE_READ_ONLY
                    | tether_vaapi::VA_EXPORT_SURFACE_SEPARATE_LAYERS,
            )
        }
        .map_err(|e| {
            // The AVFrame Drops on the `?` below, releasing the
            // surface back to the pool — so we lose this frame but
            // the decoder stays healthy.
            warn!(error = %e, "vaExportSurfaceHandle failed; dropping this frame");
            CodecError::SurfaceExportFailed(e)
        })?;

        let objects = prime
            .objects
            .into_iter()
            .map(|o| DmaBufObject {
                fd: o.fd,
                size: u64::from(o.size),
                drm_format_modifier: o.drm_format_modifier,
            })
            .collect();
        let layers = prime
            .layers
            .into_iter()
            .map(|l| DmaBufLayer {
                drm_format: l.drm_format,
                num_planes: l.num_planes,
                object_index: l.object_index,
                offset: l.offset,
                pitch: l.pitch,
            })
            .collect();
        let dmabuf = DmaBufFrame {
            fourcc: prime.fourcc,
            objects,
            layers,
        };

        // rsmpeg doesn't mark AVFrame Send (it has raw ptr fields).
        // The renderer thread takes ownership of this guard via the
        // channel and the only thing it ever does is drop it — never
        // shares a reference, never accesses fields. Same move-fine /
        // share-bad rationale as the existing `unsafe impl Send for
        // VaapiDecoder`. The `dead_code` allow is correct: this
        // struct exists for its Drop side-effect (av_frame_unref
        // releasing the surface).
        #[allow(dead_code)]
        struct SendFrame(AVFrame);
        // SAFETY: ownership is moved across the channel boundary, not
        // shared. av_frame_unref's reentrancy means it's safe to call
        // from whichever thread ends up dropping the box.
        unsafe impl Send for SendFrame {}

        Ok(Frame::Gpu(GpuFrame::new(
            width as u32,
            height as u32,
            pts_out,
            GpuFrameSource::DmaBuf(dmabuf),
            SendFrame(frame),
        )))
    }

    /// Signal end-of-stream so the decoder emits any buffered frames.
    /// ffmpeg's HEVC (and AV1) decoders hold the first frame in the
    /// reorder DPB until a subsequent packet or EOF arrives; in a
    /// continuous stream the next packet flushes it, but a lone IDR
    /// (e.g. the capability probe) needs this explicit drain to emit.
    /// Matches `D3D11Decoder::signal_eof` / VideoToolbox. After this,
    /// `next_frame` drains until it returns `None`.
    pub fn signal_eof(&mut self) -> Result<()> {
        self.decoder.send_packet(None)?;
        Ok(())
    }
}

impl Decoder for VaapiDecoder {
    fn submit(&mut self, encoded: &[u8]) -> Result<()> {
        if encoded.is_empty() {
            return Ok(());
        }
        let packet = packet_from_bytes(encoded)?;
        self.decoder.send_packet(Some(&packet))?;
        Ok(())
    }

    // ffmpeg's i32 ABI fields (width, height, linesize) are
    // non-negative on allocated decoded frames; cast sites are at
    // the FFI boundary and follow that invariant.
    #[allow(clippy::cast_sign_loss)]
    fn next_frame(&mut self) -> Result<Option<Frame>> {
        let frame = match self.decoder.receive_frame() {
            Ok(f) => f,
            Err(RsmpegError::DecoderDrainError | RsmpegError::DecoderFlushedError) => {
                return Ok(None)
            }
            Err(e) => return Err(CodecError::Ffmpeg(e)),
        };

        // VAAPI surface: export as DMA-BUF and hand the renderer a
        // GPU-resident frame. The exported fds borrow from the
        // surface; the AVFrame keeps the surface alive in the pool,
        // so we move it into the GpuFrame as a release guard.
        if frame.format == ffi::AV_PIX_FMT_VAAPI {
            return self.export_vaapi_frame(frame).map(Some);
        }

        // Reaching here means ffmpeg decoded into system memory
        // despite our get_format callback insisting on VAAPI. The
        // probe in `new()` already verified the build advertises
        // VAAPI hwaccel for the codec, so this only happens if the
        // driver bailed mid-stream (HW context loss, OOM,
        // unsupported profile slipping through). Silently doing CPU
        // decode at 4K would spike CPU with no obvious cause and
        // contradicts CLAUDE.md's "Hard requirement on hardware
        // codecs" contract. Refuse — auto-IDR in the client will log
        // it loudly via the av_log bridge and the user sees a clean
        // failure mode rather than a mysterious slowdown.
        warn!(
            format = frame.format,
            "VaapiDecoder produced a software frame; refusing per HW-only contract"
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
        vaapi_decoder_name(self.kind)
    }

    fn flush(&mut self) -> Result<()> {
        // avcodec_flush_buffers drops the reorder queue, releases
        // pending output, and resets internal state without tearing
        // the codec context down — the decoder is ready to accept
        // a fresh IDR + subsequent packets after this call. VAAPI's
        // surface pool stays allocated; only references are
        // released. Cheap (~µs) and idempotent.
        self.decoder.flush_buffers();
        Ok(())
    }
}

/// FFmpeg `AVCodecID` for the given codec kind. Errors for codecs we
/// don't have a VAAPI decode path for yet.
fn vaapi_av_codec_id(kind: CodecKind) -> Result<ffi::AVCodecID> {
    match kind {
        CodecKind::H264 => Ok(ffi::AV_CODEC_ID_H264),
        CodecKind::Hevc => Ok(ffi::AV_CODEC_ID_HEVC),
        CodecKind::Av1 => Ok(ffi::AV_CODEC_ID_AV1),
    }
}

/// Human-readable decoder name for logs / `Decoder::name`.
fn vaapi_decoder_name(kind: CodecKind) -> &'static str {
    match kind {
        CodecKind::H264 => "h264 (VAAPI hw)",
        CodecKind::Hevc => "hevc (VAAPI hw)",
        CodecKind::Av1 => "av1 (VAAPI hw)",
    }
}

/// get_format callback the decoder invokes once the bitstream's
/// SPS/PPS reveal the format options. The decoder's `pix_fmts` arg is
/// a null-terminated array of `AVPixelFormat` candidates including
/// `AV_PIX_FMT_VAAPI` (because we set `hw_device_ctx`). We pick
/// `AV_PIX_FMT_VAAPI` if present, telling ffmpeg "yes, allocate
/// VAAPI surfaces for output"; otherwise return `AV_PIX_FMT_NONE`
/// which makes the decoder bail.
unsafe extern "C" fn get_vaapi_format(
    _ctx: *mut ffi::AVCodecContext,
    pix_fmts: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    let fmts =
        unsafe { rsmpeg::build_array(pix_fmts, ffi::AV_PIX_FMT_NONE) }.unwrap_or_default();
    for &fmt in fmts {
        if fmt == ffi::AV_PIX_FMT_VAAPI {
            return fmt;
        }
    }
    ffi::AV_PIX_FMT_NONE
}
