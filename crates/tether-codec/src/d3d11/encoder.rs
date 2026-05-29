//! D3D11VA hardware encoder implementation.
//!
//! Zero-copy path: DXGI capture produces a BGRA `ID3D11Texture2D` on
//! a shared device. This encoder:
//! 1. Configures FFmpeg's `d3d11va` `AVHWDeviceContext` with the same
//!    D3D11 device (no device-creation overhead, no cross-device copy).
//! 2. Allocates an `AVHWFramesContext` pool of NV12 textures.
//! 3. On each `encode_gpu` call, copies the BGRA source texture into
//!    the pool's NV12 texture via D3D11 Video Processor (hardware
//!    color-space conversion, no CPU readback).
//! 4. Sends the hw frame to the encoder.
//!
//! Encoder backend is selected by GPU vendor (DXGI VendorId):
//! - Intel (0x8086): `hevc_qsv` / `h264_qsv` via derived QSV device
//! - AMD   (0x1002): `hevc_amf` / `h264_amf` via D3D11VA device
//! - NVIDIA(0x10de): `hevc_nvenc` / `h264_nvenc` via D3D11VA device
//! - Fallback:       `hevc_mf` / `h264_mf` (Media Foundation)
//!
//! QSV requires an FFmpeg build with a working oneVPL-over-D3D11 path
//! (see [`backends_for_vendor`]).

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{ra, AVDictionary, AVFrame, AVHWDeviceContext};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use rsmpeg::swscale::SwsContext;
use rsmpeg::UnsafeDerefMut;

use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

use crate::encoder_common::{drain_encoder, snapshot_extradata};
use crate::h264::frame_plane_mut;
use crate::{
    init_ffmpeg, CodecError, D3D11TextureFrame, Encoder, EncodedPacket, GpuEncoderFrame, Result,
    GOP_SECONDS,
};

use super::video_processor::VideoProcessorState;

/// Matches FFmpeg's `AVD3D11VAFramesContext` from `hwcontext_d3d11va.h`.
/// Matches FFmpeg 8.x `AVD3D11VAFramesContext` from hwcontext_d3d11va.h.
/// Size assertion guards against layout drift across FFmpeg versions.
#[repr(C)]
struct AvD3D11VAFramesContext {
    texture: *mut std::ffi::c_void,
    bind_flags: u32,
    misc_flags: u32,
    texture_infos: *mut std::ffi::c_void,
}
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<AvD3D11VAFramesContext>() == 24);

/// Matches FFmpeg's `AVD3D11VADeviceContext` from hwcontext_d3d11va.h.
/// Full struct declared to prevent offset drift; we only write `device`
/// and `device_context` (FFmpeg's `init` derives the rest).
#[repr(C)]
pub(crate) struct AvD3D11VADeviceContext {
    pub(crate) device: *mut std::ffi::c_void,
    pub(crate) device_context: *mut std::ffi::c_void,
    pub(crate) video_device: *mut std::ffi::c_void,
    pub(crate) video_context: *mut std::ffi::c_void,
    pub(crate) lock: *mut std::ffi::c_void,
    pub(crate) unlock: *mut std::ffi::c_void,
    pub(crate) lock_ctx: *mut std::ffi::c_void,
}
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<AvD3D11VADeviceContext>() == 56);

const D3D11_BIND_RENDER_TARGET: u32 = 0x20;

/// QSV hw_frames pool size. MUST be 1: the Intel D3D11 driver rejects
/// multi-slice NV12 texture arrays (DXGI_ERROR_INVALID_CALL on
/// CreateTexture2D), so the pool can only hold one surface. We then
/// follow Apollo's pattern — allocate that surface ONCE and reuse it
/// every frame (re-blit + re-send, async_depth=1 keeps the encoder
/// synchronous so the surface is free before the next send). Calling
/// `av_hwframe_get_buffer` per frame against a pool of 1 instead
/// exhausts it (the encoder holds the surface) → AVERROR(ENOMEM).
const QSV_POOL_SIZE: i32 = 1;

const VENDOR_INTEL: u32 = 0x8086;
const VENDOR_AMD: u32 = 0x1002;
const VENDOR_NVIDIA: u32 = 0x10de;

/// Return the encoder backends to try for a given codec and GPU vendor.
/// Vendor-native backend first, Media Foundation as universal fallback.
///
/// QSV (Intel) derives an `AV_HWDEVICE_TYPE_QSV` context from D3D11VA;
/// this depends on the oneVPL runtime in the linked FFmpeg build. Builds
/// with a broken QSV-over-D3D11 path (notably gyan.dev's
/// `full_build-shared`) fail child-frames-context creation AND hang in
/// `MFXClose` during teardown, so a failed QSV attempt is unrecoverable.
/// Link an FFmpeg build whose oneVPL works (e.g. BtbN's) — see
/// `docs/CODEC_CAPABILITIES.md`.
fn backends_for_vendor(kind: CodecKind, vendor_id: u32) -> &'static [&'static str] {
    match (kind, vendor_id) {
        (CodecKind::Hevc, VENDOR_INTEL) => &["hevc_qsv", "hevc_mf"],
        (CodecKind::Hevc, VENDOR_AMD) => &["hevc_amf", "hevc_mf"],
        (CodecKind::Hevc, VENDOR_NVIDIA) => &["hevc_nvenc", "hevc_mf"],
        (CodecKind::Hevc, _) => &["hevc_mf", "hevc_amf", "hevc_nvenc"],
        (CodecKind::H264, VENDOR_INTEL) => &["h264_qsv", "h264_mf"],
        (CodecKind::H264, VENDOR_AMD) => &["h264_amf", "h264_mf"],
        (CodecKind::H264, VENDOR_NVIDIA) => &["h264_nvenc", "h264_mf"],
        (CodecKind::H264, _) => &["h264_mf", "h264_amf", "h264_nvenc"],
        (CodecKind::Av1, _) => &[],
    }
}

fn is_qsv_backend(name: &str) -> bool {
    name.contains("qsv")
}

pub struct D3D11Encoder {
    kind: CodecKind,
    encoder: AVCodecContext,
    hw_device: AVHWDeviceContext,
    bgra_to_nv12: SwsContext,
    sw_frame: AVFrame,
    bgra_frame: AVFrame,
    extradata: Vec<u8>,
    encoder_name: &'static str,
    width: u32,
    height: u32,
    bgra_row_bytes: usize,
    vp_state: Option<VideoProcessorState>,
    /// QSV frames use `AV_PIX_FMT_QSV` and need `av_hwframe_map` to
    /// expose the underlying D3D11 texture for VP blit.
    is_qsv: bool,
    /// QSV's media-driver rejects a first frame whose `FrameType` is
    /// UNKNOWN (`Invalid FrameType:0`) — the stream must open on an IDR.
    /// Other backends auto-promote frame 0, but forcing it is correct
    /// for our protocol regardless (the client needs an initial IDR).
    first_frame: bool,
    /// QSV-only: the single hw surface (pool size 1) allocated once and
    /// reused every frame (Apollo's pattern — see [`QSV_POOL_SIZE`]).
    /// `qsv_mapped` is its persistent `AV_PIX_FMT_D3D11` mapping;
    /// `qsv_dst_texture`/`qsv_dst_index` are the VP blit target read from
    /// that mapping. All `None`/null until the first GPU frame.
    qsv_frame: Option<AVFrame>,
    qsv_mapped: Option<AVFrame>,
    qsv_dst_texture: *mut std::ffi::c_void,
    qsv_dst_index: u32,
}

unsafe impl Send for D3D11Encoder {}

impl D3D11Encoder {
    /// Construct a D3D11-accelerated encoder. Selects the backend based
    /// on `vendor_id` (PCI vendor from DXGI adapter) and tries each
    /// candidate until one successfully opens.
    ///
    /// `device_ptr` and `device_ctx_ptr` are raw COM pointers to the
    /// shared `ID3D11Device` / `ID3D11DeviceContext` from capture.
    pub fn new(
        profile: VideoProfile,
        width: u32,
        height: u32,
        fps: u32,
        bitrate_kbps: u32,
        device_ptr: *mut std::ffi::c_void,
        device_ctx_ptr: *mut std::ffi::c_void,
        vendor_id: u32,
    ) -> Result<Self> {
        init_ffmpeg();

        let kind = profile.codec;

        // The D3D11 Video Processor BGRA→YUV blit only produces 4:2:0
        // surfaces (NV12 / P010); there is no 4:4:4 output path. Reject
        // 4:4:4 here so the host probe records it Unsupported at the
        // Construct stage and host-authoritative negotiation never picks
        // a profile we would silently downsample to 4:2:0 (the trap in
        // `d3d11_sw_format`'s NV12 fallback). A real 4:4:4 path needs
        // custom HLSL conversion (AYUV/Y410, like Apollo) — a separate
        // future capability, not a silent no-op.
        if profile.chroma == ChromaSubsampling::Yuv444 {
            return Err(CodecError::CodecNotFound(
                "D3D11 Video Processor has no 4:4:4 encode path (only NV12/P010 4:2:0)",
            ));
        }

        let backends = backends_for_vendor(kind, vendor_id);
        if backends.is_empty() {
            return Err(CodecError::CodecNotFound("av1 d3d11 (not yet supported)"));
        }

        let mut last_err = CodecError::CodecNotFound("d3d11 encoder");
        for &backend_name in backends {
            match Self::try_open(
                profile,
                backend_name,
                width,
                height,
                fps,
                bitrate_kbps,
                device_ptr,
                device_ctx_ptr,
            ) {
                Ok(enc) => {
                    tracing::info!(
                        encoder = backend_name,
                        vendor_id = format_args!("0x{vendor_id:04x}"),
                        width,
                        height,
                        bitrate_kbps,
                        "D3D11 encoder opened"
                    );
                    return Ok(enc);
                }
                Err(e) => {
                    tracing::warn!(
                        encoder = backend_name,
                        error = %e,
                        "D3D11 encoder backend unavailable, trying next"
                    );
                    last_err = e;
                }
            }
        }

        Err(last_err)
    }

    fn try_open(
        profile: VideoProfile,
        backend_name: &'static str,
        width: u32,
        height: u32,
        fps: u32,
        bitrate_kbps: u32,
        device_ptr: *mut std::ffi::c_void,
        device_ctx_ptr: *mut std::ffi::c_void,
    ) -> Result<Self> {
        let kind = profile.codec;
        let codec_cname = std::ffi::CString::new(backend_name)
            .map_err(|_| CodecError::CodecNotFound(backend_name))?;
        let codec = AVCodec::find_encoder_by_name(&codec_cname)
            .ok_or(CodecError::CodecNotFound(backend_name))?;

        let d3d11va_device = create_d3d11va_hw_device(device_ptr, device_ctx_ptr)?;

        let is_qsv = is_qsv_backend(backend_name);

        // QSV requires a derived QSV device context on top of D3D11VA.
        // SetMultithreadProtected is already called by tether-capture.
        let (encoder_device, pix_fmt) = if is_qsv {
            let qsv_device = d3d11va_device
                .create_derived(ffi::AV_HWDEVICE_TYPE_QSV)
                .map_err(|e| {
                    tracing::debug!("QSV device derivation failed: {e}");
                    CodecError::Ffmpeg(e)
                })?;
            (qsv_device, ffi::AV_PIX_FMT_QSV)
        } else {
            (d3d11va_device.clone(), ffi::AV_PIX_FMT_D3D11)
        };

        let width_i32 = i32::try_from(width).expect("width fits i32");
        let height_i32 = i32::try_from(height).expect("height fits i32");
        let fps_i32 = i32::try_from(fps.max(1)).unwrap_or(60);

        let mut encoder = AVCodecContext::new(&codec);
        encoder.set_width(width_i32);
        encoder.set_height(height_i32);
        encoder.set_pix_fmt(pix_fmt);
        encoder.set_time_base(ra(1, fps_i32));
        encoder.set_framerate(ra(fps_i32, 1));
        encoder.set_bit_rate(i64::from(bitrate_kbps) * 1000);
        let gop_frames = fps_i32
            .saturating_mul(i32::try_from(GOP_SECONDS).expect("GOP_SECONDS fits i32"));
        encoder.set_gop_size(gop_frames);
        encoder.set_max_b_frames(0);

        #[allow(clippy::cast_possible_wrap)]
        encoder.set_flags(encoder.flags | ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32);

        let sw_format = d3d11_sw_format(profile);

        // Provide the hw_frames pool. Non-QSV (AMF/MF/NVENC): D3D11VA
        // device, format = D3D11, pool 0 = dynamic. QSV: QSV device,
        // format = QSV, FIXED pool (libmfx can't grow). The pool must be
        // big enough for the async pipeline + DPB + our in-flight frame,
        // but this Intel driver rejects large multi-slice NV12 arrays
        // (DXGI_ERROR_INVALID_CALL) — QSV_POOL_SIZE is the tuned bound.
        let mut hw_frames_ref = encoder_device.hwframe_ctx_alloc();
        hw_frames_ref.data().format = pix_fmt;
        hw_frames_ref.data().sw_format = sw_format;
        hw_frames_ref.data().width = width_i32;
        hw_frames_ref.data().height = height_i32;
        hw_frames_ref.data().initial_pool_size = if is_qsv { QSV_POOL_SIZE } else { 0 };

        if !is_qsv {
            unsafe {
                let hwctx = hw_frames_ref.data().hwctx as *mut AvD3D11VAFramesContext;
                (*hwctx).bind_flags = D3D11_BIND_RENDER_TARGET;
                (*hwctx).misc_flags = 0;
            }
        }

        hw_frames_ref.init()?;
        encoder.set_hw_device_ctx(encoder_device.clone());
        encoder.set_hw_frames_ctx(hw_frames_ref);

        unsafe {
            let raw = encoder.deref_mut();
            raw.color_primaries = ffi::AVCOL_PRI_BT709;
            raw.color_trc = ffi::AVCOL_TRC_BT709;
            raw.colorspace = ffi::AVCOL_SPC_BT709;
            raw.color_range = ffi::AVCOL_RANGE_MPEG;
        }

        let dict = if backend_name.contains("amf") {
            // `async_depth=1` is critical: amfenc defaults it to 16
            // ("Higher values increase output latency" per its AVOption
            // help), so without this the AMD path runs with 16 frames of
            // encode queue. One frame in flight matches QSV/NVENC and the
            // single-frame handoff. `usage=ultralowlatency` + `latency=1`
            // select AMF's low-latency RC path (B-frames are already off
            // via `max_b_frames=0` above).
            Some(AVDictionary::new(c"usage", c"ultralowlatency", 0)
                .set(c"quality", c"speed", 0)
                .set(c"latency", c"1", 0)
                .set(c"async_depth", c"1", 0)
                .set(c"forced_idr", c"1", 0)
                .set(c"gops_per_idr", c"1", 0))
        } else if backend_name.contains("nvenc") {
            // `delay=0` is critical: nvenc defaults it to INT_MAX (output
            // is held indefinitely waiting to reorder). `zerolatency=1`
            // removes the reordering delay, `tune=ull` is the ultra-low-
            // latency tuning, `surfaces=1` bounds the pool to one frame.
            Some(AVDictionary::new(c"delay", c"0", 0)
                .set(c"forced-idr", c"1", 0)
                .set(c"zerolatency", c"1", 0)
                .set(c"tune", c"ull", 0)
                .set(c"rc", c"cbr", 0)
                .set(c"surfaces", c"1", 0))
        } else if backend_name.contains("qsv") {
            // `async_depth=1` keeps latency low (one frame in flight) and
            // bounds the surface pool. `forced_idr` honours ForceIdr.
            // Do NOT add `low_delay_brc` / `low_power`: both put the QSV
            // media-driver into a mode that demands an explicit per-frame
            // `mfxEncodeCtrl.FrameType`, which our `pict_type` doesn't
            // supply — every frame is rejected `Invalid FrameType:0`
            // (AVERROR_INVALIDDATA). Verified by
            // `d3d11_qsv_gpu_encode_decode_roundtrip`.
            Some(AVDictionary::new(c"forced_idr", c"1", 0)
                .set(c"async_depth", c"1", 0))
        } else if backend_name.contains("mf") {
            Some(AVDictionary::new(c"hw_encoding", c"1", 0))
        } else {
            None
        };
        let _leftover = encoder.open(dict)?;

        // Prepare software fallback path (BGRA → NV12 via swscale).
        // Used by encode_bgra; the zero-copy GPU path bypasses this.
        let mut bgra_to_nv12 = SwsContext::get_context(
            width_i32,
            height_i32,
            ffi::AV_PIX_FMT_BGRA,
            width_i32,
            height_i32,
            sw_format,
            ffi::SWS_FAST_BILINEAR,
            None,
            None,
            None,
        )
        .ok_or(CodecError::ScalerInit("bgra→nv12/p010 for d3d11 encoder"))?;

        unsafe {
            let coeffs = ffi::sws_getCoefficients(ffi::SWS_CS_ITU709 as i32);
            let _ = ffi::sws_setColorspaceDetails(
                bgra_to_nv12.as_mut_ptr(),
                coeffs,
                1, // src full range
                coeffs,
                0, // dst video range
                0,
                65536,
                65536,
            );
        }

        let mut bgra_frame = AVFrame::new();
        bgra_frame.set_format(ffi::AV_PIX_FMT_BGRA);
        bgra_frame.set_width(width_i32);
        bgra_frame.set_height(height_i32);
        bgra_frame.alloc_buffer()?;

        let mut sw_frame = AVFrame::new();
        sw_frame.set_format(sw_format);
        sw_frame.set_width(width_i32);
        sw_frame.set_height(height_i32);
        sw_frame.alloc_buffer()?;

        let extradata = snapshot_extradata(&encoder, backend_name, kind)?;
        let bgra_row_bytes = (width as usize) * 4;

        // Keep the D3D11VA device alive — for QSV the derived device
        // holds a reference to it, but we own it explicitly too.
        let hw_device = d3d11va_device;

        Ok(Self {
            kind,
            encoder,
            hw_device,
            bgra_to_nv12,
            sw_frame,
            bgra_frame,
            extradata,
            encoder_name: backend_name,
            width,
            height,
            bgra_row_bytes,
            vp_state: None,
            is_qsv,
            first_frame: true,
            qsv_frame: None,
            qsv_mapped: None,
            qsv_dst_texture: std::ptr::null_mut(),
            qsv_dst_index: 0,
        })
    }

    /// Submit a D3D11 texture for zero-copy GPU encode.
    ///
    /// The texture must be a BGRA `ID3D11Texture2D` on the same device
    /// the encoder was constructed with. Internally we use the D3D11
    /// Video Processor to convert BGRA→NV12 into an hw_frames pool
    /// surface, then submit that to the encoder.
    pub fn submit_d3d11_texture(
        &mut self,
        frame: &D3D11TextureFrame,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>> {
        if frame.texture.is_null() || frame.device.is_null() {
            return Err(CodecError::UnsupportedInputFormat);
        }

        // Guard: if the VP was built for different input dims (e.g.
        // capture resolution changed after a DXGI reconnect), reject
        // the frame. The encoder slot should be rebuilt upstream.
        if let Some(vp) = &self.vp_state {
            if vp.input_width() != frame.width || vp.input_height() != frame.height {
                tracing::error!(
                    vp_input_w = vp.input_width(),
                    vp_input_h = vp.input_height(),
                    frame_w = frame.width,
                    frame_h = frame.height,
                    "VP input dims mismatch; encoder slot should rebuild"
                );
                return Err(CodecError::UnsupportedInputFormat);
            }
        }

        // Lazily initialize the Video Processor on first GPU frame.
        // The VP handles both color conversion (BGRA→NV12) and scaling
        // when the input (capture) dimensions differ from the output
        // (encode) dimensions.
        if self.vp_state.is_none() {
            match VideoProcessorState::new(
                frame.device,
                frame.device_context,
                frame.width,
                frame.height,
                self.width,
                self.height,
            ) {
                Ok(vp) => self.vp_state = Some(vp),
                Err(e) => {
                    tracing::warn!(error = %e, "Video Processor init failed; GPU encode unavailable");
                    return Err(CodecError::UnsupportedInputFormat);
                }
            }
        }

        let force_idr = force_keyframe || self.first_frame;

        if self.is_qsv {
            // QSV: one surface, allocated once and reused every frame
            // (the Intel driver caps the pool at 1 — see QSV_POOL_SIZE).
            // Lazily get the buffer and its persistent D3D11 mapping; the
            // mapping aliases the QSV surface so the VP blit writes the
            // bytes the encoder reads.
            if self.qsv_frame.is_none() {
                let mut hw = AVFrame::new();
                let rc = unsafe {
                    ffi::av_hwframe_get_buffer(
                        self.encoder.deref_mut().hw_frames_ctx,
                        hw.as_mut_ptr(),
                        0,
                    )
                };
                if rc < 0 {
                    return Err(CodecError::Ffmpeg(RsmpegError::AVError(rc)));
                }
                let mut mapped = AVFrame::new();
                unsafe {
                    (*mapped.as_mut_ptr()).format = ffi::AV_PIX_FMT_D3D11;
                }
                let rc = unsafe {
                    ffi::av_hwframe_map(
                        mapped.as_mut_ptr(),
                        hw.as_ptr(),
                        (ffi::AV_HWFRAME_MAP_WRITE | ffi::AV_HWFRAME_MAP_OVERWRITE) as i32,
                    )
                };
                if rc < 0 {
                    return Err(CodecError::Ffmpeg(RsmpegError::AVError(rc)));
                }
                self.qsv_dst_texture =
                    unsafe { (*mapped.as_ptr()).data[0] as *mut std::ffi::c_void };
                self.qsv_dst_index = unsafe { (*mapped.as_ptr()).data[1] as usize as u32 };
                self.qsv_mapped = Some(mapped);
                self.qsv_frame = Some(hw);
            }

            // Re-blit into the reused surface.
            let dst_texture = self.qsv_dst_texture;
            let dst_index = self.qsv_dst_index;
            self.vp_state
                .as_ref()
                .unwrap()
                .blit(frame.texture, dst_texture, dst_index)
                .map_err(|e| {
                    tracing::error!(error = %e, "Video Processor blit failed");
                    CodecError::UnsupportedInputFormat
                })?;

            let hw_frame = self.qsv_frame.as_mut().unwrap();
            hw_frame.set_pts(pts);
            unsafe {
                (*hw_frame.as_mut_ptr()).pict_type = if force_idr {
                    ffi::AV_PICTURE_TYPE_I
                } else {
                    ffi::AV_PICTURE_TYPE_NONE
                };
            }
            self.first_frame = false;
            let hw_ptr: *const AVFrame = &*hw_frame;
            self.encoder.send_frame(Some(unsafe { &*hw_ptr }))?;
            return drain_encoder(&mut self.encoder, &self.extradata);
        }

        // Non-QSV (AMF/MF/NVENC): pull a fresh D3D11 frame from the pool
        // each call (these backends honour a dynamically-grown pool).
        let mut hw_frame = AVFrame::new();
        let rc = unsafe {
            ffi::av_hwframe_get_buffer(
                self.encoder.deref_mut().hw_frames_ctx,
                hw_frame.as_mut_ptr(),
                0,
            )
        };
        if rc < 0 {
            return Err(CodecError::Ffmpeg(RsmpegError::AVError(rc)));
        }

        let dst_texture = unsafe { (*hw_frame.as_ptr()).data[0] as *mut std::ffi::c_void };
        let dst_index = unsafe { (*hw_frame.as_ptr()).data[1] as usize };

        let vp = self.vp_state.as_ref().unwrap();
        vp.blit(frame.texture, dst_texture, dst_index as u32)
            .map_err(|e| {
                tracing::error!(error = %e, "Video Processor blit failed");
                CodecError::UnsupportedInputFormat
            })?;

        hw_frame.set_pts(pts);
        if force_idr {
            unsafe {
                (*hw_frame.as_mut_ptr()).pict_type = ffi::AV_PICTURE_TYPE_I;
            }
        }
        self.first_frame = false;

        self.encoder.send_frame(Some(&hw_frame))?;
        drain_encoder(&mut self.encoder, &self.extradata)
    }
}

impl Encoder for D3D11Encoder {
    fn encode_bgra(
        &mut self,
        bgra: &[u8],
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>> {
        let expected = self.bgra_row_bytes * (self.height as usize);
        if bgra.len() < expected {
            return Err(CodecError::UnsupportedInputFormat);
        }

        // Copy BGRA into the AVFrame.
        let stride = unsafe { (*self.bgra_frame.as_ptr()).linesize[0] as usize };
        let dst_plane = frame_plane_mut(&mut self.bgra_frame, 0, self.height as usize);
        for row in 0..self.height as usize {
            let src_offset = row * self.bgra_row_bytes;
            let dst_offset = row * stride;
            dst_plane[dst_offset..dst_offset + self.bgra_row_bytes]
                .copy_from_slice(&bgra[src_offset..src_offset + self.bgra_row_bytes]);
        }
        self.bgra_frame.set_pts(pts);

        // BGRA → NV12 via swscale.
        self.bgra_to_nv12.scale_frame(&self.bgra_frame, 0, self.height as i32, &mut self.sw_frame)?;
        self.sw_frame.set_pts(pts);

        // Upload NV12 sw_frame into a hw_frame from the pool.
        let mut hw_frame = AVFrame::new();
        let rc = unsafe {
            ffi::av_hwframe_get_buffer(
                self.encoder.deref_mut().hw_frames_ctx,
                hw_frame.as_mut_ptr(),
                0,
            )
        };
        if rc < 0 {
            return Err(CodecError::Ffmpeg(RsmpegError::AVError(rc)));
        }

        // Transfer sw_frame → hw_frame.
        let rc = unsafe {
            ffi::av_hwframe_transfer_data(hw_frame.as_mut_ptr(), self.sw_frame.as_ptr(), 0)
        };
        if rc < 0 {
            return Err(CodecError::Ffmpeg(RsmpegError::AVError(rc)));
        }
        hw_frame.set_pts(pts);

        if force_keyframe || self.first_frame {
            unsafe {
                (*hw_frame.as_mut_ptr()).pict_type = ffi::AV_PICTURE_TYPE_I;
            }
        }
        self.first_frame = false;

        self.encoder.send_frame(Some(&hw_frame))?;
        drain_encoder(&mut self.encoder, &self.extradata)
    }

    fn encode_gpu(
        &mut self,
        frame: GpuEncoderFrame<'_>,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>> {
        match frame {
            GpuEncoderFrame::D3D11Texture(tex) => {
                self.submit_d3d11_texture(tex, pts, force_keyframe)
            }
            _ => Err(CodecError::UnsupportedInputFormat),
        }
    }

    fn supports_changing_bitrate(&self) -> bool {
        false
    }

    fn is_hardware(&self) -> bool {
        true
    }

    fn codec_kind(&self) -> CodecKind {
        self.kind
    }

    fn name(&self) -> &'static str {
        self.encoder_name
    }
}

impl D3D11Encoder {
    /// Flush the encoder so AMF's hardware session is idle before the
    /// context is destroyed. Without this, AMF's async session cleanup
    /// may cause subsequent encoder construction to fail with a
    /// single-session-limit error.
    pub fn shutdown(&mut self) {
        self.encoder.flush_buffers();
    }

    #[cfg(test)]
    pub fn extradata(&self) -> &[u8] {
        &self.extradata
    }
}

/// Create an FFmpeg `d3d11va` hardware device context.
///
/// When `device_ptr` is non-null, injects the capture-created device
/// into FFmpeg's hwctx so both capture and encode share a single
/// D3D11 device (no cross-device texture copies in the VP blit).
/// When null (probe path), lets FFmpeg create its own device.
fn create_d3d11va_hw_device(
    device_ptr: *mut std::ffi::c_void,
    device_ctx_ptr: *mut std::ffi::c_void,
) -> Result<AVHWDeviceContext> {
    if device_ptr.is_null() {
        let hw_device =
            AVHWDeviceContext::create(ffi::AV_HWDEVICE_TYPE_D3D11VA, None, None, 0)?;
        return Ok(hw_device);
    }

    // Inject the externally-provided device into FFmpeg's hwctx.
    // AddRef both COM objects — FFmpeg calls Release on cleanup.
    let mut hw_device = AVHWDeviceContext::alloc(ffi::AV_HWDEVICE_TYPE_D3D11VA);
    unsafe {
        com_addref(device_ptr);
        if !device_ctx_ptr.is_null() {
            com_addref(device_ctx_ptr);
        }

        let buf_ptr = hw_device.as_mut_ptr();
        let data = &mut *((*buf_ptr).data as *mut ffi::AVHWDeviceContext);
        let hwctx = data.hwctx as *mut AvD3D11VADeviceContext;
        (*hwctx).device = device_ptr;
        (*hwctx).device_context = device_ctx_ptr;
        // Install no-op lock/unlock: the device is already
        // multithread-protected by the capture layer
        // (SetMultithreadProtected). Without this, FFmpeg installs its
        // own ID3D11Multithread-based lock, which conflicts with QSV's
        // session creation on a derived device (MFX_ERR_NOT_FOUND).
        // lock_ctx must be non-null or FFmpeg falls back to its default.
        (*hwctx).lock = d3d11_no_op_lock as *mut std::ffi::c_void;
        (*hwctx).unlock = d3d11_no_op_lock as *mut std::ffi::c_void;
        (*hwctx).lock_ctx = 1 as *mut std::ffi::c_void;
    }
    if let Err(e) = hw_device.init() {
        // init failed — FFmpeg won't release the injected refs, so we must.
        unsafe {
            com_release(device_ptr);
            if !device_ctx_ptr.is_null() {
                com_release(device_ctx_ptr);
            }
        }
        return Err(CodecError::Ffmpeg(e));
    }
    Ok(hw_device)
}

/// Map a VideoProfile to the FFmpeg sw_format for the hw_frames pool.
///
/// `D3D11Encoder::new` rejects 4:4:4 up front, so only 4:2:0 profiles
/// reach here. The NV12 fallback therefore only ever covers 4:2:0 with
/// an unexpected bit depth — it is NOT a silent 4:4:4→4:2:0 downsample
/// (that path is closed at construction).
fn d3d11_sw_format(profile: VideoProfile) -> ffi::AVPixelFormat {
    match (profile.chroma, profile.bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => ffi::AV_PIX_FMT_NV12,
        (ChromaSubsampling::Yuv420, 10) => ffi::AV_PIX_FMT_P010LE,
        _ => ffi::AV_PIX_FMT_NV12,
    }
}

/// No-op lock callback for FFmpeg's `AVD3D11VADeviceContext`. The
/// shared D3D11 device is externally synchronized via
/// `SetMultithreadProtected`, so FFmpeg need not lock around its use.
unsafe extern "C" fn d3d11_no_op_lock(_lock_ctx: *mut std::ffi::c_void) {}

/// Raw COM IUnknown vtable layout.
#[repr(C)]
struct IUnknownVtbl {
    query_interface: *const std::ffi::c_void,
    add_ref: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
    release: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
}

/// Call IUnknown::AddRef on a raw COM pointer via vtable.
unsafe fn com_addref(ptr: *mut std::ffi::c_void) {
    unsafe {
        let vtbl_ptr = *(ptr as *const *const IUnknownVtbl);
        ((*vtbl_ptr).add_ref)(ptr);
    }
}

/// Call IUnknown::Release on a raw COM pointer via vtable.
unsafe fn com_release(ptr: *mut std::ffi::c_void) {
    unsafe {
        let vtbl_ptr = *(ptr as *const *const IUnknownVtbl);
        ((*vtbl_ptr).release)(ptr);
    }
}
