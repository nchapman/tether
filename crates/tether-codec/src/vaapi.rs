//! VAAPI hardware H.264 encoder for Linux (Intel Arc / AMD / NVIDIA-
//! via-translation).
//!
//! Wraps ffmpeg's `h264_vaapi` encoder through rsmpeg's safe
//! `AVHWDeviceContext` / `AVHWFramesContext` / `hwframe_transfer_data`
//! bindings. The capture path still produces BGRA frames in system
//! memory, so this encoder does:
//!
//!   BGRA &[u8] -> bgra AVFrame -> swscale NV12 sw_frame
//!                              -> av_hwframe_transfer_data -> VAAPI surface
//!                              -> h264_vaapi encode -> H.264 packet
//!
//! The CPU swscale + CPU→GPU transfer are still on the critical path.
//! The next step is the DMA-BUF capture handoff that hands us a GPU
//! surface directly from PipeWire, removes both, and lets us encode
//! straight off the captured surface. That's a separate workstream
//! and lives in another module when it lands.

use std::slice;

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{ra, AVDictionary, AVFrame, AVHWDeviceContext};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use rsmpeg::swscale::SwsContext;
use tracing::warn;

use tether_protocol::control::CodecKind;

use crate::h264::frame_plane_mut;
use crate::{init_ffmpeg, CodecError, EncodedPacket, Encoder, Result};

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
        encoder.set_gop_size(fps_i32);
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
