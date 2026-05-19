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
use tracing::warn;

use tether_protocol::control::CodecKind;

use crate::h264::{frame_plane, frame_plane_mut, interleave_uv, pack_plane, packet_from_bytes};
use crate::{
    init_ffmpeg, CodecError, DecodedFrame, Decoder, Encoder, EncodedPacket, Frame, Result,
    GOP_SECONDS,
};

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
        decoder.open(None)?;

        Ok(Self {
            decoder,
            _hw_device: hw_device,
        })
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

        // If get_format returned VAAPI, frame is a GPU surface and
        // needs the transfer dance. If ffmpeg fell back to a
        // software path (rare in our config; usually a build with
        // hwaccel disabled silently emits SW frames), the frame is
        // already in system memory and we use it directly. The
        // transfer here will be replaced by a DMA-BUF export +
        // `Frame::Gpu` once the renderer can import VAAPI surfaces.
        let sw_frame = if frame.format == ffi::AV_PIX_FMT_VAAPI {
            let mut sw = AVFrame::new();
            sw.hwframe_transfer_data(&frame)?;
            sw
        } else {
            frame
        };

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

        let mut got = None;
        for t in 0..6i64 {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let bgra = make_test_bgra(w, h, t as u32);
            let packets = enc.encode_bgra(&bgra, t, t == 0).expect("encode");
            for p in packets {
                dec.submit(&p.data).expect("vaapi submit");
                while let Some(f) = dec.next_frame().expect("vaapi next_frame") {
                    let Frame::Cpu(f) = f else {
                        panic!("VaapiDecoder is in CPU-readback mode; \
                                update this test when DMA-BUF export ships");
                    };
                    got = Some(f);
                }
            }
        }
        let frame = got.expect("decoder produced a frame within six input frames");
        assert_eq!(frame.width, w);
        assert_eq!(frame.height, h);
        let (cw, ch) = frame.chroma_dims();
        assert_eq!(frame.y.len(), (w * h) as usize);
        // NV12 layout: each chroma sample carries 2 bytes (U then V).
        assert_eq!(frame.uv.len(), (cw * ch * 2) as usize);
    }
}
