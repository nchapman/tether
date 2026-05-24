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
//! Encoder selection follows a preference order:
//! - `hevc_mf` / `h264_mf` (Media Foundation — Intel/AMD/NVIDIA)
//! - `hevc_nvenc` / `h264_nvenc` (NVIDIA-specific, higher quality)
//! - `hevc_amf` / `h264_amf` (AMD-specific)
//!
//! The first encoder that successfully opens with the given profile
//! wins. The probe layer surfaces this empirically.

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{ra, AVFrame, AVHWDeviceContext};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use rsmpeg::swscale::SwsContext;
use rsmpeg::UnsafeDerefMut;

use tether_protocol::control::{CodecKind, VideoProfile};

use crate::encoder_common::{drain_encoder, snapshot_extradata};
use crate::h264::frame_plane_mut;
use crate::{
    init_ffmpeg, CodecError, D3D11TextureFrame, Encoder, EncodedPacket, GpuEncoderFrame, Result,
    GOP_SECONDS,
};

use super::video_processor::VideoProcessorState;

/// Matches FFmpeg's `AVD3D11VAFramesContext` from `hwcontext_d3d11va.h`.
/// Only the fields we need to set before `av_hwframe_ctx_init` are
/// included; the struct is `repr(C)` so field offsets match the C layout.
#[repr(C)]
struct AvD3D11VAFramesContext {
    texture: *mut std::ffi::c_void,
    bind_flags: u32,
    misc_flags: u32,
    texture_infos: *mut std::ffi::c_void,
}

/// Matches FFmpeg's `AVD3D11VADeviceContext` from hwcontext_d3d11va.h.
/// Full struct declared to prevent offset drift; we only write `device`
/// and `device_context` (FFmpeg's `init` derives the rest).
#[repr(C)]
struct AvD3D11VADeviceContext {
    device: *mut std::ffi::c_void,
    device_context: *mut std::ffi::c_void,
    video_device: *mut std::ffi::c_void,
    video_context: *mut std::ffi::c_void,
    lock: *mut std::ffi::c_void,
    unlock: *mut std::ffi::c_void,
    lock_ctx: *mut std::ffi::c_void,
}
const _: () = assert!(std::mem::size_of::<AvD3D11VADeviceContext>() == 56);

const D3D11_BIND_RENDER_TARGET: u32 = 0x20;

/// Encoder backend names to try in preference order for each codec.
/// AMF first: on AMD hardware, native AMF handles d3d11 hw_frames
/// correctly; MF's wrapper sometimes fails at send_frame for H.264.
const HEVC_BACKENDS: &[&str] = &["hevc_amf", "hevc_mf", "hevc_nvenc"];
const H264_BACKENDS: &[&str] = &["h264_amf", "h264_mf", "h264_nvenc"];

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
}

unsafe impl Send for D3D11Encoder {}

impl D3D11Encoder {
    /// Construct a D3D11-accelerated encoder. Tries multiple FFmpeg
    /// backend encoders in preference order and returns the first that
    /// successfully opens.
    ///
    /// `device_ptr` and `device_ctx_ptr` are raw COM pointers to the
    /// shared `ID3D11Device` / `ID3D11DeviceContext` from capture.
    /// The encoder configures FFmpeg's `d3d11va` hw_device_ctx to use
    /// this device — no new device is created, and textures produced by
    /// capture are directly usable without cross-device copies.
    pub fn new(
        profile: VideoProfile,
        width: u32,
        height: u32,
        fps: u32,
        bitrate_kbps: u32,
        device_ptr: *mut std::ffi::c_void,
        device_ctx_ptr: *mut std::ffi::c_void,
    ) -> Result<Self> {
        init_ffmpeg();

        let kind = profile.codec;
        let backends = match kind {
            CodecKind::Hevc => HEVC_BACKENDS,
            CodecKind::H264 => H264_BACKENDS,
            CodecKind::Av1 => {
                return Err(CodecError::CodecNotFound("av1 d3d11 (not yet supported)"));
            }
        };

        let mut last_err = CodecError::CodecNotFound("d3d11 encoder");
        for &backend_name in backends {
            match Self::try_open(
                kind,
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
                        width,
                        height,
                        bitrate_kbps,
                        "D3D11 encoder opened"
                    );
                    return Ok(enc);
                }
                Err(e) => {
                    tracing::debug!(
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
        kind: CodecKind,
        backend_name: &'static str,
        width: u32,
        height: u32,
        fps: u32,
        bitrate_kbps: u32,
        device_ptr: *mut std::ffi::c_void,
        device_ctx_ptr: *mut std::ffi::c_void,
    ) -> Result<Self> {
        let codec_cname = std::ffi::CString::new(backend_name)
            .map_err(|_| CodecError::CodecNotFound(backend_name))?;
        let codec = AVCodec::find_encoder_by_name(&codec_cname)
            .ok_or(CodecError::CodecNotFound(backend_name))?;

        let hw_device = create_d3d11va_hw_device(device_ptr, device_ctx_ptr)?;

        let width_i32 = i32::try_from(width).expect("width fits i32");
        let height_i32 = i32::try_from(height).expect("height fits i32");
        let fps_i32 = i32::try_from(fps.max(1)).unwrap_or(60);

        let mut encoder = AVCodecContext::new(&codec);
        encoder.set_width(width_i32);
        encoder.set_height(height_i32);
        encoder.set_pix_fmt(ffi::AV_PIX_FMT_D3D11);
        encoder.set_time_base(ra(1, fps_i32));
        encoder.set_framerate(ra(fps_i32, 1));
        encoder.set_bit_rate(i64::from(bitrate_kbps) * 1000);
        let gop_frames = fps_i32
            .saturating_mul(i32::try_from(GOP_SECONDS).expect("GOP_SECONDS fits i32"));
        encoder.set_gop_size(gop_frames);
        encoder.set_max_b_frames(0);

        #[allow(clippy::cast_possible_wrap)]
        encoder.set_flags(encoder.flags | ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32);

        // Configure hw_frames_ctx: NV12 pool on the shared device.
        let mut hw_frames_ref = hw_device.hwframe_ctx_alloc();
        hw_frames_ref.data().format = ffi::AV_PIX_FMT_D3D11;
        hw_frames_ref.data().sw_format = ffi::AV_PIX_FMT_NV12;
        hw_frames_ref.data().width = width_i32;
        hw_frames_ref.data().height = height_i32;
        hw_frames_ref.data().initial_pool_size = 0;

        // AMF/MF require D3D11_BIND_RENDER_TARGET on pool textures,
        // otherwise avcodec_open2 fails with AVERROR_UNKNOWN. Access
        // the backend-specific AVD3D11VAFramesContext and set flags.
        unsafe {
            let hwctx = hw_frames_ref.data().hwctx as *mut AvD3D11VAFramesContext;
            (*hwctx).bind_flags = D3D11_BIND_RENDER_TARGET;
            (*hwctx).misc_flags = 0;
        }

        hw_frames_ref.init()?;

        // Set hw_device_ctx on the encoder as well — some backends
        // (h264_mf) read it separately from hw_frames_ctx.
        encoder.set_hw_device_ctx(hw_device.clone());
        encoder.set_hw_frames_ctx(hw_frames_ref);

        unsafe {
            let raw = encoder.deref_mut();
            raw.color_primaries = ffi::AVCOL_PRI_BT709;
            raw.color_trc = ffi::AVCOL_TRC_BT709;
            raw.colorspace = ffi::AVCOL_SPC_BT709;
            raw.color_range = ffi::AVCOL_RANGE_MPEG;
        }

        let _leftover = encoder.open(None)?;

        // Prepare software fallback path (BGRA → NV12 via swscale).
        // Used by encode_bgra; the zero-copy GPU path bypasses this.
        let mut bgra_to_nv12 = SwsContext::get_context(
            width_i32,
            height_i32,
            ffi::AV_PIX_FMT_BGRA,
            width_i32,
            height_i32,
            ffi::AV_PIX_FMT_NV12,
            ffi::SWS_FAST_BILINEAR,
            None,
            None,
            None,
        )
        .ok_or(CodecError::ScalerInit("bgra→nv12 for d3d11 encoder"))?;

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
        sw_frame.set_format(ffi::AV_PIX_FMT_NV12);
        sw_frame.set_width(width_i32);
        sw_frame.set_height(height_i32);
        sw_frame.alloc_buffer()?;

        let extradata = snapshot_extradata(&encoder, backend_name)?;
        let bgra_row_bytes = (width as usize) * 4;

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

        // Get a hardware frame from the pool.
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

        // BGRA→NV12 via D3D11 Video Processor. data[0] is the pool's
        // ID3D11Texture2D (NV12 array texture), data[1] is the slice
        // index within that array.
        let dst_texture = unsafe { (*hw_frame.as_mut_ptr()).data[0] as *mut std::ffi::c_void };
        let dst_index = unsafe { (*hw_frame.as_mut_ptr()).data[1] as usize };

        let vp = self.vp_state.as_ref().unwrap();
        vp.blit(frame.texture, dst_texture, dst_index as u32)
            .map_err(|e| {
                tracing::error!(error = %e, "Video Processor blit failed");
                CodecError::UnsupportedInputFormat
            })?;

        hw_frame.set_pts(pts);
        if force_keyframe {
            unsafe {
                (*hw_frame.as_mut_ptr()).pict_type = ffi::AV_PICTURE_TYPE_I;
            }
        }

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

        if force_keyframe {
            unsafe {
                (*hw_frame.as_mut_ptr()).pict_type = ffi::AV_PICTURE_TYPE_I;
            }
        }

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
