//! VAAPI hardware H.264 encode + decode for Linux (Intel Arc / AMD /
//! NVIDIA-via-translation).
//!
//! Wraps ffmpeg's `h264_vaapi` encoder and the `h264` decoder's VAAPI
//! hwaccel through rsmpeg's safe `AVHWDeviceContext` /
//! `AVHWFramesContext` / `hwframe_transfer_data` bindings.
//!
//! Encoder pipeline (capture-side, still pays CPU swscale + upload):
//!
//!   BGRA &[u8] -> bgra AVFrame -> swscale NV12 sw_frame
//!                              -> av_hwframe_transfer_data -> VAAPI surface
//!                              -> h264_vaapi encode -> H.264 packet
//!
//! Decoder pipeline (client-side, still pays CPU readback):
//!
//!   H.264 bytes -> AVPacket -> h264 (VAAPI hwaccel) -> VAAPI surface
//!                                                   -> av_hwframe_transfer_data
//!                                                   -> NV12 sw frame
//!                                                   -> Y + UV planes (NV12 layout) for renderer
//!
//! Both pipelines have an obvious-but-deferred next optimisation: zero-
//! copy GPU handoff. Encoder side wants DMA-BUF from PipeWire capture
//! straight into a VAAPI surface; decoder side wants the VAAPI surface
//! to land in a wgpu texture without a CPU detour. Both need real
//! EGL/Vulkan interop work and live as separate modules when they ship.

use std::slice;

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{ra, AVDictionary, AVFrame, AVHWDeviceContext};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use rsmpeg::swscale::SwsContext;
use rsmpeg::UnsafeDerefMut;
use tracing::warn;

use tether_protocol::control::CodecKind;

use crate::h264::{frame_plane, frame_plane_mut, interleave_uv, pack_plane, packet_from_bytes};
use crate::{
    init_ffmpeg, CodecError, DecodedFrame, Decoder, DmaBufFrame, DmaBufLayer, DmaBufObject,
    Encoder, EncodedPacket, Frame, GpuFrame, GpuFrameSource, Result, GOP_SECONDS,
};

/// `AVVAAPIDeviceContext` from `<libavutil/hwcontext_vaapi.h>`. rusty-ffmpeg
/// doesn't bind it (the header requires libva at bindgen time, which the
/// FFI crate doesn't pull in), so we declare it locally. Two fields,
/// stable since libavutil 56.x — if FFmpeg ever extends it, the new
/// fields land *after* `driver_quirks` so our access of `display`
/// stays correct.
#[repr(C)]
struct AVVAAPIDeviceContext {
    display: tether_vaapi::VADisplay,
    driver_quirks: std::os::raw::c_uint,
}

/// Surfaces beyond what the decoder needs for its own reference picture
/// list. We hand each decoded surface to the renderer and only release
/// the AVFrame ref when the renderer drops the `GpuFrame`. With a
/// 1-deep render channel that's at most 2 surfaces held outside the
/// decoder (one rendering, one in flight in the channel); 4 gives
/// headroom for the brief window where the renderer is sampling the
/// previous surface while the next one lands. Surfaces are cheap to
/// allocate but not free — a 2880×1920 NV12 surface is ~8 MiB of GPU
/// memory, so don't over-provision.
const DECODE_EXTRA_HW_FRAMES: i32 = 4;

/// Number of VAAPI surfaces in the hwframes pool. With `async_depth=1`
/// the encoder reports a packet synchronously after each `send_frame`,
/// so we never have more than one surface in flight. The pool also
/// holds reference frames the encoder needs internally (1 short-term
/// ref with `max_b_frames=0`) plus headroom for the brief windows
/// where VAAPI hardware feedback transiently holds an extra surface
/// before releasing it. Eight is comfortable; tighter values risk
/// `EAGAIN` on `hwframe_ctx_alloc::get_buffer`.
const VAAPI_POOL_SIZE: i32 = 8;

pub struct VaapiEncoder {
    encoder: AVCodecContext,
    bgra_to_nv12: SwsContext,
    sw_frame: AVFrame,
    bgra_frame: AVFrame,
    // Keep the device context alive for the encoder's lifetime. The
    // encoder's `hw_frames_ctx` holds an internal ref-counted handle
    // to the device, so dropping this field early wouldn't free VAAPI
    // resources prematurely — but keeping the explicit owner here
    // documents the lifetime relationship and is cheap.
    //
    // Drop order: struct fields drop in declaration order, so `encoder`
    // (and its internal hw_frames_ctx) tear down before `_hw_device`,
    // which is the correct order — surfaces must be freed before the
    // device that allocated them.
    _hw_device: AVHWDeviceContext,
    height: u32,
    bgra_row_bytes: usize,
}

// SAFETY: ffmpeg HW codec context, VAAPI device, and per-encoder frames
// are safe to MOVE between threads but unsafe to SHARE. We only expose
// `&mut self` methods so the borrow checker serialises access within a
// single thread; the manual `unsafe impl Send` matches the move-fine /
// share-bad contract that all our other encoders document.
unsafe impl Send for VaapiEncoder {}

impl VaapiEncoder {
    /// Construct an `h264_vaapi` encoder for BGRA input at the given
    /// dimensions. Returns `Err(CodecError::CodecNotFound)` if the
    /// installed FFmpeg wasn't built with VAAPI support, and any
    /// `RsmpegError` if the VAAPI device can't be opened (no
    /// `/dev/dri/renderD*` accessible, driver mismatch, etc.) — the
    /// caller (the probe in `crate::probe`) treats either as "no
    /// hardware path available" and falls through to libx264.
    ///
    /// The low-power VAAPI encode entrypoint
    /// (`VAEntrypointEncSliceLP`) would shave more latency on Intel
    /// hardware that exposes it, but we don't currently probe whether
    /// LP is supported for the (codec, profile) combo. Asking for LP
    /// when the device doesn't expose it produces a partially-init'd
    /// FFmpeg encoder context that segfaults on Drop instead of
    /// returning a clean error — and Meteor Lake Arc is one of those
    /// devices (no LP for H.264). A safe LP path needs an explicit
    /// libva capability query before encoder.open(); deferring until
    /// the win is worth the FFI surface.
    pub fn new_bgra(width: u32, height: u32, fps: u32, bitrate_kbps: u32) -> Result<Self> {
        init_ffmpeg();

        let codec = AVCodec::find_encoder_by_name(c"h264_vaapi")
            .ok_or(CodecError::CodecNotFound("h264_vaapi"))?;

        // Default VAAPI device (typically /dev/dri/renderD128). None
        // lets FFmpeg pick — explicit device strings only matter on
        // multi-GPU systems, which we'll handle when a user with that
        // setup hits us up.
        let hw_device =
            AVHWDeviceContext::create(ffi::AV_HWDEVICE_TYPE_VAAPI, None, None, 0)?;

        let width_i32 = i32::try_from(width).expect("width fits in i32");
        let height_i32 = i32::try_from(height).expect("height fits in i32");
        let fps_i32 = i32::try_from(fps.max(1)).unwrap_or(60);

        let mut encoder = AVCodecContext::new(&codec);
        encoder.set_width(width_i32);
        encoder.set_height(height_i32);
        // pix_fmt is VAAPI: pixels live on the GPU; the actual storage
        // layout is in `sw_format` on the hwframes context below.
        encoder.set_pix_fmt(ffi::AV_PIX_FMT_VAAPI);
        encoder.set_time_base(ra(1, fps_i32));
        encoder.set_framerate(ra(fps_i32, 1));
        encoder.set_bit_rate(i64::from(bitrate_kbps) * 1000);
        // GOP cadence matches the libx264 fallback so the on-wire
        // worst-case "garbled until next IDR" window is the same
        // regardless of which encoder the probe picked.
        let gop_frames = fps_i32
            .saturating_mul(i32::try_from(GOP_SECONDS).expect("GOP_SECONDS fits in i32"));
        encoder.set_gop_size(gop_frames);
        encoder.set_max_b_frames(0);

        // Build + attach the hwframes context. NV12 is the canonical
        // VAAPI surface layout — Intel/AMD/NVIDIA all natively encode
        // from NV12. The encoder mutates the pool over its lifetime,
        // so once set_hw_frames_ctx moves ownership in, we go through
        // encoder.hw_frames_ctx_mut() to allocate frames.
        let mut hw_frames_ref = hw_device.hwframe_ctx_alloc();
        hw_frames_ref.data().format = ffi::AV_PIX_FMT_VAAPI;
        hw_frames_ref.data().sw_format = ffi::AV_PIX_FMT_NV12;
        hw_frames_ref.data().width = width_i32;
        hw_frames_ref.data().height = height_i32;
        hw_frames_ref.data().initial_pool_size = VAAPI_POOL_SIZE;
        hw_frames_ref.init()?;
        encoder.set_hw_frames_ctx(hw_frames_ref);

        // h264_vaapi private options. The defaults are tuned for
        // file-based transcoding throughput, not realtime; we
        // override the knobs that cost us the most:
        //   profile=main — Main is the safest broadly-supported
        //     profile across Intel/AMD/NVIDIA VAAPI drivers. The
        //     encoder defaults to High, which fails to open on
        //     hardware that exposes only Main for the chosen
        //     entrypoint. Every realistic H.264 decoder we'll talk
        //     to handles Main.
        //   async_depth=1 — synchronous mode. Default is 4, which
        //     buys throughput at the cost of three extra frames of
        //     latency before the first packet emerges. We need the
        //     opposite trade.
        //   rc_mode=VBR — VBR matches Intel's recommended low-latency
        //     mode (CBR-style buffering also works but introduces a
        //     visible bitrate floor on static content).
        let dict = AVDictionary::new(c"profile", c"main", 0)
            .set(c"async_depth", c"1", 0)
            .set(c"rc_mode", c"VBR", 0);
        let leftover = encoder.open(Some(dict))?;
        if let Some(unused) = leftover {
            // Driver/encoder didn't recognise one or more opts. Not
            // fatal — the encoder will still work, just with the
            // unrecognised setting at its default. Surfacing the keys
            // helps diagnose "why is latency higher than expected?"
            let mut unused_keys: Vec<String> = Vec::new();
            for entry in unused.iter() {
                unused_keys.push(format!(
                    "{}={}",
                    entry.key().to_string_lossy(),
                    entry.value().to_string_lossy()
                ));
            }
            if !unused_keys.is_empty() {
                warn!(
                    unused = ?unused_keys,
                    "h264_vaapi ignored some private options; latency knobs may not be applied"
                );
            }
        }

        let bgra_to_nv12 = SwsContext::get_context(
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
        .ok_or(CodecError::ScalerInit("BGRA -> NV12"))?;

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

        let bgra_row_bytes = (width as usize) * 4;

        Ok(Self {
            encoder,
            bgra_to_nv12,
            sw_frame,
            bgra_frame,
            _hw_device: hw_device,
            height,
            bgra_row_bytes,
        })
    }
}

impl Encoder for VaapiEncoder {
    // ffmpeg's i32 ABI fields (linesize, packet.size) are non-negative
    // on allocated frames / valid packets. Same rationale as h264.rs.
    #[allow(clippy::cast_sign_loss)]
    fn encode_bgra(
        &mut self,
        bgra: &[u8],
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>> {
        let height = self.height as usize;
        let expected = self.bgra_row_bytes * height;
        if bgra.len() != expected {
            return Err(CodecError::BufferSizeMismatch {
                got: bgra.len(),
                expected,
            });
        }

        // 1. Copy BGRA bytes into the encoder-side BGRA AVFrame,
        // stride-aware in case the row alignment differs from
        // width*4.
        {
            let stride = self.bgra_frame.linesize[0] as usize;
            let plane = frame_plane_mut(&mut self.bgra_frame, 0, height);
            if stride == self.bgra_row_bytes {
                plane[..expected].copy_from_slice(bgra);
            } else {
                for row in 0..height {
                    let src = row * self.bgra_row_bytes;
                    let dst = row * stride;
                    plane[dst..dst + self.bgra_row_bytes]
                        .copy_from_slice(&bgra[src..src + self.bgra_row_bytes]);
                }
            }
        }

        // 2. swscale BGRA -> NV12 into the CPU-side sw_frame. This is
        // the CPU pixel-format conversion the DMA-BUF zero-copy path
        // will eventually eliminate by handing us a GPU surface in
        // the native format.
        self.bgra_to_nv12.scale_frame(
            &self.bgra_frame,
            0,
            i32::try_from(height).expect("height fits in i32"),
            &mut self.sw_frame,
        )?;

        // 3. Allocate a VAAPI surface from the encoder's hwframes pool
        // and upload the NV12 bytes via av_hwframe_transfer_data
        // (CPU memcpy into GPU memory). The pool blocks if all
        // surfaces are still in flight at the encoder, which with
        // async_depth=1 + 8 surfaces basically never happens.
        let mut hw_frame = AVFrame::new();
        self.encoder
            .hw_frames_ctx_mut()
            .expect("hw_frames_ctx set in new_bgra")
            .get_buffer(&mut hw_frame)?;
        hw_frame.hwframe_transfer_data(&self.sw_frame)?;
        hw_frame.set_pts(pts);
        hw_frame.set_pict_type(if force_keyframe {
            ffi::AV_PICTURE_TYPE_I
        } else {
            ffi::AV_PICTURE_TYPE_NONE
        });

        // 4. Submit + drain. With async_depth=1 the receive_packet
        // immediately yields a packet on the first call and EAGAIN on
        // the second; the loop generalises gracefully if a future
        // async_depth bump emits multiple packets per submit.
        self.encoder.send_frame(Some(&hw_frame))?;

        let mut out = Vec::new();
        loop {
            let packet = match self.encoder.receive_packet() {
                Ok(p) => p,
                Err(RsmpegError::EncoderDrainError | RsmpegError::EncoderFlushedError) => break,
                Err(e) => return Err(CodecError::Ffmpeg(e)),
            };
            let size = packet.size as usize;
            // SAFETY: `packet.data` points to `packet.size` valid bytes
            // owned by the AVPacket; we copy them out before the
            // packet is dropped at end-of-iteration.
            let data = unsafe { slice::from_raw_parts(packet.data, size) }.to_vec();
            let keyframe = (packet.flags & ffi::AV_PKT_FLAG_KEY as i32) != 0;
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

    fn is_hardware(&self) -> bool {
        true
    }

    fn codec_kind(&self) -> CodecKind {
        CodecKind::H264
    }

    fn name(&self) -> &'static str {
        // Could be refined to "h264_vaapi (Intel)" / "(AMD)" / "(NVIDIA)"
        // by querying the VA driver string from the hw_device. Not
        // worth the extra FFI on first pass — the log already shows
        // is_hardware=true alongside the name.
        "h264_vaapi"
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

/// VAAPI-accelerated H.264 decoder. Uses ffmpeg's generic `h264`
/// decoder with VAAPI hwaccel selected via `get_format`. Output VAAPI
/// surfaces are downloaded to CPU NV12 via `hwframe_transfer_data`
/// and handed straight to the renderer in that shape — no NV12→YUV420P
/// swscale on the decode hot path. The renderer was migrated to a
/// two-texture NV12 layout to match.
pub struct VaapiDecoder {
    decoder: AVCodecContext,
    // Decoder holds a cloned ref to the device internally
    // (set_hw_device_ctx); keeping our own ref documents the
    // lifetime relationship and matches what hw_decode.c does.
    // Drop order: `decoder` drops first (declaration order), so all
    // surfaces allocated against the device are released before the
    // device context itself goes.
    _hw_device: AVHWDeviceContext,
}

// SAFETY: same move-fine / share-bad rationale as VaapiEncoder above.
unsafe impl Send for VaapiDecoder {}

impl VaapiDecoder {
    /// Construct an h264 decoder bound to a VAAPI device.
    /// `Err(CodecError::CodecNotFound)` if either the h264 decoder
    /// isn't compiled in (effectively impossible — every ffmpeg ships
    /// it) or this build's h264 decoder doesn't advertise VAAPI
    /// hwaccel support. Any other failure (device open, get_format
    /// callback rejected) comes through as `Err(CodecError::Ffmpeg)`.
    /// The probe treats either as "no HW path" and falls through to
    /// libavcodec software decode.
    pub fn new() -> Result<Self> {
        init_ffmpeg();

        let codec = AVCodec::find_decoder(ffi::AV_CODEC_ID_H264)
            .ok_or(CodecError::CodecNotFound("h264 (for VAAPI decode)"))?;

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
            return Err(CodecError::CodecNotFound("h264 VAAPI hwaccel"));
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
        // VAAPI hwaccel for h264, so this only happens if the driver
        // bailed mid-stream — silently doing CPU decode at 4K would
        // make CPU usage spike with no obvious cause. Refuse loudly
        // so the failure mode is observable. The remaining
        // CPU-packing logic stays below in case a future SW backend
        // wants to reuse it via a different decoder type.
        warn!(
            format = frame.format,
            "VaapiDecoder fell back to software decode unexpectedly"
        );
        let sw_frame = frame;

        let width = sw_frame.width;
        let height = sw_frame.height;
        let w = width as usize;
        let h = height as usize;
        let chroma_w = w.div_ceil(2);
        let chroma_h = h.div_ceil(2);

        // Two formats we accept from the system-memory side:
        //  - NV12 (canonical VAAPI sw_format) — already the
        //    renderer's preferred layout, just pack the Y and UV
        //    planes tight (strip any compositor stride padding).
        //  - YUV420P (SW fallback emitted this directly) — interleave
        //    U and V into NV12 layout before handing off. Pure byte
        //    permutation; no resampling.
        // Anything else (notably NV21, which is V-first instead of
        // U-first) lands in the error arm rather than silently
        // shipping inverted colors. If a future hwaccel surfaces
        // NV21 we'd need either a dedicated arm or a U/V swap in
        // the packing step.
        let fmt = sw_frame.format;
        let (y, uv) = if fmt == ffi::AV_PIX_FMT_NV12 {
            let y = pack_plane(
                frame_plane(&sw_frame, 0, h),
                sw_frame.linesize[0] as usize,
                w,
                h,
            );
            // NV12 plane 1: interleaved UV at half resolution. Each
            // "chroma sample" is two bytes (U, V) so a row carries
            // `chroma_w * 2` bytes; `pack_plane` strips any extra
            // padding the driver may have added.
            let uv = pack_plane(
                frame_plane(&sw_frame, 1, chroma_h),
                sw_frame.linesize[1] as usize,
                chroma_w * 2,
                chroma_h,
            );
            (y, uv)
        } else if fmt == ffi::AV_PIX_FMT_YUV420P {
            let y = pack_plane(
                frame_plane(&sw_frame, 0, h),
                sw_frame.linesize[0] as usize,
                w,
                h,
            );
            let uv = interleave_uv(
                frame_plane(&sw_frame, 1, chroma_h),
                sw_frame.linesize[1] as usize,
                frame_plane(&sw_frame, 2, chroma_h),
                sw_frame.linesize[2] as usize,
                chroma_w,
                chroma_h,
            );
            (y, uv)
        } else {
            return Err(CodecError::UnsupportedInputFormat);
        };

        let pts_out = if sw_frame.pts == ffi::AV_NOPTS_VALUE {
            None
        } else {
            Some(sw_frame.pts)
        };
        Ok(Some(Frame::Cpu(DecodedFrame {
            width: width as u32,
            height: height as u32,
            pts: pts_out,
            y,
            uv,
        })))
    }

    fn codec_kind(&self) -> CodecKind {
        CodecKind::H264
    }

    fn is_hardware(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "h264 (VAAPI hw)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::H264Encoder;

    #[test]
    #[ignore = "requires a working VAAPI device (run on hardware with: cargo test -p tether-codec --ignored vaapi)"]
    fn vaapi_encoder_smoke() {
        let w = 640;
        let h = 480;
        let mut enc = VaapiEncoder::new_bgra(w, h, 30, 4_000).expect("VAAPI encoder");
        let bgra = vec![0x80u8; (w * h * 4) as usize];
        let packets = enc.encode_bgra(&bgra, 0, true).expect("encode");
        // First frame may produce 0 packets (encoder warm-up) or 1+
        // packets carrying SPS/PPS + the IDR slice. Either way it
        // shouldn't error.
        for p in packets {
            assert!(!p.data.is_empty());
        }
    }

    fn make_test_bgra(width: u32, height: u32, t: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let r: u8 = if (x / 64 + t / 4) % 2 == 0 { 200 } else { 50 };
                let g: u8 = if (y / 64) % 2 == 0 { 200 } else { 50 };
                let b: u8 = 128;
                data.extend_from_slice(&[b, g, r, 255]);
            }
        }
        data
    }

    #[test]
    #[ignore = "requires a working VAAPI device (run on hardware with: cargo test -p tether-codec --ignored vaapi)"]
    fn vaapi_decoder_smoke() {
        // Encode a few frames with the software encoder so we have a
        // valid Annex-B bitstream to decode, then verify the VAAPI
        // decoder produces a frame at the expected dimensions.
        let w = 320;
        let h = 240;
        let mut enc = H264Encoder::new_bgra(w, h, 30, 2_000).expect("sw encoder");
        let mut dec = VaapiDecoder::new().expect("VAAPI decoder");

        let mut got: Option<GpuFrame> = None;
        for t in 0..6i64 {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let bgra = make_test_bgra(w, h, t as u32);
            let packets = enc.encode_bgra(&bgra, t, t == 0).expect("encode");
            for p in packets {
                dec.submit(&p.data).expect("vaapi submit");
                while let Some(f) = dec.next_frame().expect("vaapi next_frame") {
                    let Frame::Gpu(g) = f else {
                        panic!("VaapiDecoder must emit DMA-BUF Gpu frames");
                    };
                    got = Some(g);
                }
            }
        }
        let frame = got.expect("decoder produced a frame within six input frames");
        assert_eq!(frame.width, w);
        assert_eq!(frame.height, h);
        let GpuFrameSource::DmaBuf(dmabuf) = frame.source else {
            panic!("expected DmaBuf source on Linux VAAPI");
        };
        // SEPARATE_LAYERS gives NV12 two planes (Y, UV). Each layer
        // points at one object (might be the same fd with different
        // offsets, or two distinct fds) so we expect 2 layers and at
        // least 1 object.
        assert_eq!(dmabuf.layers.len(), 2, "NV12 export should yield 2 layers");
        assert!(!dmabuf.objects.is_empty(), "at least one DMA-BUF object");
        // Surface-level fourcc is NV12.
        assert_eq!(
            dmabuf.fourcc,
            u32::from_le_bytes(*b"NV12"),
            "expected NV12 fourcc"
        );
        // Per-layer DRM fourccs: R8 for the Y plane, GR88 for the
        // interleaved UV plane. Asserting these catches a driver
        // silently falling back to COMPOSED_LAYERS (which would emit
        // a single NV12-fourcc layer) — under that mode our wgpu
        // import would do the wrong thing and we'd rather fail in
        // the test than in production.
        assert_eq!(
            dmabuf.layers[0].drm_format,
            u32::from_le_bytes(*b"R8  "),
            "Y plane should be DRM_FORMAT_R8"
        );
        assert_eq!(
            dmabuf.layers[1].drm_format,
            u32::from_le_bytes(*b"GR88"),
            "UV plane should be DRM_FORMAT_GR88"
        );
    }
}
