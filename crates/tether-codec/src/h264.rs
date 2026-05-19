//! Software H.264 encoder + decoder via libx264 / ffmpeg's `h264` decoder.
//!
//! v0 goals: get bytes-on-the-wire down from ~240 MB/s (raw 1080p BGRA)
//! to ~5 MB/s (zerolatency H.264). Quality tuning, B-frame strategies,
//! and adaptive bitrate land later — for now we hardcode the lowest-
//! latency preset and trust libx264's defaults beyond that.
//!
//! Hardware encoders (VAAPI on Linux, VideoToolbox on macOS) will land
//! as siblings of this module under the same `Encoder` / `Decoder`
//! traits.

use std::ffi::CString;
use std::slice;

use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVPacket};
use rsmpeg::avutil::{ra, AVDictionary, AVFrame};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use rsmpeg::swscale::SwsContext;

use tether_protocol::control::CodecKind;

use crate::{init_ffmpeg, CodecError, DecodedFrame, Decoder, Encoder, EncodedPacket, Result, GOP_SECONDS};

/// Pack a planar AVFrame plane into a tight `Vec<u8>`, stripping any
/// stride padding so downstream consumers can upload row-major without
/// per-row math.
pub(crate) fn pack_plane(plane: &[u8], stride: usize, row_bytes: usize, height: usize) -> Vec<u8> {
    let mut packed = Vec::with_capacity(row_bytes * height);
    if stride == row_bytes {
        packed.extend_from_slice(&plane[..row_bytes * height]);
    } else {
        for row in 0..height {
            let start = row * stride;
            packed.extend_from_slice(&plane[start..start + row_bytes]);
        }
    }
    packed
}

/// Interleave two single-channel chroma planes (YUV420P layout) into
/// an NV12-shaped UV buffer (U₀ V₀ U₁ V₁ ...). Output is tight (no
/// stride padding) and sized `chroma_w * chroma_h * 2` bytes — twice
/// the input plane size, since each chroma sample contributes one byte
/// of U and one of V. Replaces the per-plane swscale we used to do
/// after the VAAPI decoder; for the software path this is a fast
/// memory-bound shuffle (no resampling).
pub(crate) fn interleave_uv(
    u: &[u8],
    u_stride: usize,
    v: &[u8],
    v_stride: usize,
    chroma_w: usize,
    chroma_h: usize,
) -> Vec<u8> {
    let mut uv = Vec::with_capacity(chroma_w * chroma_h * 2);
    for row in 0..chroma_h {
        let u_row = &u[row * u_stride..row * u_stride + chroma_w];
        let v_row = &v[row * v_stride..row * v_stride + chroma_w];
        for col in 0..chroma_w {
            uv.push(u_row[col]);
            uv.push(v_row[col]);
        }
    }
    uv
}

/// Borrow plane `idx` of an allocated `AVFrame` as `&mut [u8]` covering
/// `linesize[idx] * height` bytes. Encapsulates the one unsafe slice
/// construction we need against rsmpeg's raw-pointer frame layout —
/// the mutable borrow of `frame` keeps the underlying buffer alive for
/// the slice's lifetime.
#[allow(clippy::cast_sign_loss)] // ffmpeg linesize is i32 but non-negative for allocated frames
pub(crate) fn frame_plane_mut(frame: &mut AVFrame, idx: usize, height: usize) -> &mut [u8] {
    let stride = frame.linesize[idx] as usize;
    let ptr = frame.data_mut()[idx];
    // SAFETY: caller asserts `frame` was constructed with a successful
    // alloc_buffer() at width/height matching `linesize[idx]` / `height`,
    // so plane `idx` is backed by at least `stride * height` valid bytes.
    // The &mut borrow on `frame` is held for the slice's lifetime,
    // preventing reallocation underneath us.
    unsafe { slice::from_raw_parts_mut(ptr, stride * height) }
}

/// Borrow plane `idx` of an allocated `AVFrame` as a read-only slice.
#[allow(clippy::cast_sign_loss)] // ffmpeg linesize is i32 but non-negative for allocated frames
pub(crate) fn frame_plane(frame: &AVFrame, idx: usize, height: usize) -> &[u8] {
    let stride = frame.linesize[idx] as usize;
    let ptr = frame.data[idx];
    // SAFETY: same rationale as `frame_plane_mut`; the &AVFrame borrow
    // keeps the buffer alive.
    unsafe { slice::from_raw_parts(ptr, stride * height) }
}

/// Wrap a slice of encoded bytes in an `AVPacket` owned by ffmpeg so it
/// can be fed to a decoder. rsmpeg doesn't expose a safe helper for
/// this — we allocate via `av_new_packet` (so the packet owns its
/// buffer and frees it on Drop) and memcpy our bytes in.
pub(crate) fn packet_from_bytes(bytes: &[u8]) -> Result<AVPacket> {
    let mut packet = AVPacket::new();
    let size = i32::try_from(bytes.len()).map_err(|_| {
        CodecError::Ffmpeg(RsmpegError::AVError(ffi::AVERROR_INVALIDDATA))
    })?;
    // SAFETY: `packet` was just allocated by AVPacket::new(). av_new_packet
    // allocates `size + AV_INPUT_BUFFER_PADDING_SIZE` bytes, fills the
    // padding with zeroes, and sets packet.data + packet.size. Ownership
    // of the buffer transfers to the packet, freed by AVPacket's Drop
    // impl via av_packet_free.
    let ret = unsafe { ffi::av_new_packet(packet.as_mut_ptr(), size) };
    if ret < 0 {
        return Err(CodecError::Ffmpeg(RsmpegError::AVError(ret)));
    }
    // SAFETY: packet.data now points to an owned buffer of exactly
    // `size` writable bytes (plus padding we don't touch).
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), packet.data, bytes.len());
    }
    Ok(packet)
}

pub struct H264Encoder {
    encoder: AVCodecContext,
    bgra_to_yuv: SwsContext,
    yuv_frame: AVFrame,
    bgra_frame: AVFrame,
    height: u32,
    bgra_row_bytes: usize,
}

// SAFETY: ffmpeg's encoder context + swscale context + AVFrame are safe to
// MOVE between threads (FFmpeg internally uses thread-local state per
// codec instance) but unsafe to SHARE. We expose only `&mut self` methods,
// which the borrow checker already serialises within a single thread, and
// we never hand out aliasing references to the inner pointers. The Send
// marker matches the "move is fine, share is not" contract.
unsafe impl Send for H264Encoder {}

impl H264Encoder {
    /// Construct a libx264 encoder configured for low-latency BGRA input.
    /// `fps` is the *target* rate (sets time_base); we don't enforce it on
    /// the caller. `bitrate_kbps` is a soft target; libx264 with
    /// `tune=zerolatency` doesn't strictly cap.
    pub fn new_bgra(width: u32, height: u32, fps: u32, bitrate_kbps: u32) -> Result<Self> {
        init_ffmpeg();
        let codec = AVCodec::find_encoder(ffi::AV_CODEC_ID_H264)
            .ok_or(CodecError::CodecNotFound("h264"))?;
        let mut encoder = AVCodecContext::new(&codec);
        let fps_i32 = i32::try_from(fps.max(1)).unwrap_or(60);
        let width_i32 = i32::try_from(width).expect("width fits in i32");
        let height_i32 = i32::try_from(height).expect("height fits in i32");

        encoder.set_width(width_i32);
        encoder.set_height(height_i32);
        encoder.set_pix_fmt(ffi::AV_PIX_FMT_YUV420P);
        encoder.set_time_base(ra(1, fps_i32));
        encoder.set_framerate(ra(fps_i32, 1));
        encoder.set_bit_rate(i64::from(bitrate_kbps) * 1000);
        let gop_frames = fps_i32
            .saturating_mul(i32::try_from(GOP_SECONDS).expect("GOP_SECONDS fits in i32"));
        encoder.set_gop_size(gop_frames);
        encoder.set_max_b_frames(0);

        // Latency flag at the codec-context level. AV_CODEC_FLAG_LOW_DELAY
        // tells ffmpeg "emit a packet per input frame, do not buffer for
        // throughput." Most ffmpeg encoders respect this; libx264 with
        // tune=zerolatency already implies it, but setting it
        // unconditionally is cheap belt-and-braces and matches what
        // Sunshine does (src/video.cpp:1727).
        #[allow(clippy::cast_possible_wrap)] // AV_CODEC_FLAG_LOW_DELAY is a single-bit constant
        let low_delay_flag = ffi::AV_CODEC_FLAG_LOW_DELAY as i32;
        encoder.set_flags(encoder.flags | low_delay_flag);

        // libx264 private options.
        //
        // tune=zerolatency already sets rc-lookahead=0 +
        // sync-lookahead=0; pinning min-keyint=keyint forces *exactly*
        // periodic IDRs (no early IDRs from scene-cut detection) so
        // the GOP cadence matches what `set_gop_size` promised.
        // baseline profile = no B-frames, no CABAC — belt-and-braces
        // for predictability across libx264 versions.
        //
        // threads + thread_type=slice is the load-bearing latency knob
        // for libx264. ffmpeg's default thread_type=frame buffers
        // `threads` frames before emitting any output, costing N-1
        // frames of pipeline latency at fps. thread_type=slice
        // partitions the *current* frame across threads instead, so
        // each frame still emerges in its own encode step. Sunshine
        // sets FF_THREAD_SLICE the same way (src/video.cpp:1808-1809).
        // 4 slices is a reasonable balance — more slices means more
        // parallelism but worse compression efficiency (each slice has
        // its own header + can't predict across slice boundaries).
        let x264_params =
            CString::new(format!("keyint={gop_frames}:min-keyint={gop_frames}:scenecut=0"))
                .expect("static format yields no nul bytes");
        let dict = AVDictionary::new(c"preset", c"ultrafast", 0)
            .set(c"tune", c"zerolatency", 0)
            .set(c"profile", c"baseline", 0)
            .set(c"threads", c"4", 0)
            .set(c"thread_type", c"slice", 0)
            .set(c"x264-params", &x264_params, 0);
        encoder.open(Some(dict))?;

        let bgra_to_yuv = SwsContext::get_context(
            width_i32,
            height_i32,
            ffi::AV_PIX_FMT_BGRA,
            width_i32,
            height_i32,
            ffi::AV_PIX_FMT_YUV420P,
            ffi::SWS_FAST_BILINEAR,
            None,
            None,
            None,
        )
        .ok_or(CodecError::ScalerInit("BGRA -> YUV420P"))?;

        let mut bgra_frame = AVFrame::new();
        bgra_frame.set_format(ffi::AV_PIX_FMT_BGRA);
        bgra_frame.set_width(width_i32);
        bgra_frame.set_height(height_i32);
        bgra_frame.alloc_buffer()?;

        let mut yuv_frame = AVFrame::new();
        yuv_frame.set_format(ffi::AV_PIX_FMT_YUV420P);
        yuv_frame.set_width(width_i32);
        yuv_frame.set_height(height_i32);
        yuv_frame.alloc_buffer()?;

        let bgra_row_bytes = (width as usize) * 4;

        Ok(Self {
            encoder,
            bgra_to_yuv,
            yuv_frame,
            bgra_frame,
            height,
            bgra_row_bytes,
        })
    }
}

impl Encoder for H264Encoder {
    // ffmpeg's i32 ABI fields (linesize, packet.size) are non-negative
    // in practice; cast sites are documented at the boundary.
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

        // Copy BGRA bytes into the encoder's input frame, row-by-row in
        // case stride > width*4 (AVFrame buffers are usually 32-byte
        // aligned, so 1920*4 = 7680 bytes/row is already a clean stride
        // — but smaller widths may have padding).
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

        self.bgra_to_yuv.scale_frame(
            &self.bgra_frame,
            0,
            i32::try_from(height).expect("height fits in i32"),
            &mut self.yuv_frame,
        )?;
        self.yuv_frame.set_pts(pts);
        self.yuv_frame.set_pict_type(if force_keyframe {
            ffi::AV_PICTURE_TYPE_I
        } else {
            ffi::AV_PICTURE_TYPE_NONE
        });

        self.encoder.send_frame(Some(&self.yuv_frame))?;

        let mut out = Vec::new();
        loop {
            // Mirror the explicit-match style used in the decoder: only
            // the drain/flushed sentinels mean "no more output for now";
            // every other error is a real codec failure the caller must
            // see.
            let packet = match self.encoder.receive_packet() {
                Ok(p) => p,
                Err(RsmpegError::EncoderDrainError | RsmpegError::EncoderFlushedError) => break,
                Err(e) => return Err(CodecError::Ffmpeg(e)),
            };
            let size = packet.size as usize;
            // SAFETY: `packet.data` points to `packet.size` valid bytes
            // owned by the AVPacket; we copy them out before the packet
            // is dropped at end-of-iteration.
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

    fn codec_kind(&self) -> CodecKind {
        CodecKind::H264
    }

    fn name(&self) -> &'static str {
        "libx264 sw"
    }
}

pub struct H264Decoder {
    decoder: AVCodecContext,
}

// SAFETY: same rationale as H264Encoder above.
unsafe impl Send for H264Decoder {}

impl H264Decoder {
    pub fn new() -> Result<Self> {
        init_ffmpeg();
        let codec = AVCodec::find_decoder(ffi::AV_CODEC_ID_H264)
            .ok_or(CodecError::CodecNotFound("h264"))?;
        let mut decoder = AVCodecContext::new(&codec);
        decoder.open(None)?;
        Ok(Self { decoder })
    }
}

impl Decoder for H264Decoder {
    // Same rationale as encode_bgra: ffmpeg's i32 width/height/linesize
    // on an allocated decoder output frame are non-negative.
    #[allow(clippy::cast_sign_loss)]
    fn decode(&mut self, encoded: &[u8]) -> Result<Vec<DecodedFrame>> {
        if encoded.is_empty() {
            return Ok(Vec::new());
        }
        let packet = packet_from_bytes(encoded)?;
        self.decoder.send_packet(Some(&packet))?;

        let mut out = Vec::new();
        loop {
            let yuv = match self.decoder.receive_frame() {
                Ok(f) => f,
                Err(RsmpegError::DecoderDrainError | RsmpegError::DecoderFlushedError) => break,
                Err(e) => return Err(CodecError::Ffmpeg(e)),
            };
            // H264 in our configuration always emits YUV420P. If a
            // future capture path negotiates something else (NV12,
            // YUV422) the renderer's YUV→RGB shader will need
            // companion shaders / upload paths; for now reject loudly
            // so the failure is observable instead of a silent garble.
            if yuv.format != ffi::AV_PIX_FMT_YUV420P {
                return Err(CodecError::UnsupportedInputFormat);
            }
            let w = yuv.width as usize;
            let h = yuv.height as usize;
            let chroma_w = w.div_ceil(2);
            let chroma_h = h.div_ceil(2);
            let y_plane = pack_plane(frame_plane(&yuv, 0, h), yuv.linesize[0] as usize, w, h);
            // Interleave U + V into NV12-layout UV plane so the rest of
            // the pipeline (including the renderer) sees the same shape
            // VAAPI hardware decode produces natively. The interleave is
            // a byte permutation — no resampling, ~chroma_w*chroma_h
            // memory writes — and replaces the older path that uploaded
            // U and V as separate textures.
            let uv_plane = interleave_uv(
                frame_plane(&yuv, 1, chroma_h),
                yuv.linesize[1] as usize,
                frame_plane(&yuv, 2, chroma_h),
                yuv.linesize[2] as usize,
                chroma_w,
                chroma_h,
            );
            let pts_out = if yuv.pts == ffi::AV_NOPTS_VALUE {
                None
            } else {
                Some(yuv.pts)
            };
            out.push(DecodedFrame {
                width: yuv.width as u32,
                height: yuv.height as u32,
                pts: pts_out,
                y: y_plane,
                uv: uv_plane,
            });
        }
        Ok(out)
    }

    fn codec_kind(&self) -> CodecKind {
        CodecKind::H264
    }

    fn name(&self) -> &'static str {
        "libavcodec h264 sw"
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
mod tests {
    use super::*;

    /// Verifies `interleave_uv` produces the correct NV12 byte order
    /// (U₀ V₀ U₁ V₁ ...) and correctly strips stride padding. The
    /// stride case matters because ffmpeg-allocated YUV420P frames
    /// typically have linesize > chroma_w, and a regression here
    /// would silently shift chroma samples (visible as wrong colors
    /// per row) instead of erroring.
    #[test]
    fn interleave_uv_packs_nv12_and_strips_stride() {
        // 4x2 chroma plane with stride 6 (2 bytes of trailing padding
        // per row). U byte values encode (row << 4) | col so any
        // mis-indexing is obvious.
        let chroma_w = 4;
        let chroma_h = 2;
        let stride = 6;
        let mut u_plane = vec![0u8; stride * chroma_h];
        let mut v_plane = vec![0u8; stride * chroma_h];
        for row in 0..chroma_h {
            for col in 0..chroma_w {
                u_plane[row * stride + col] = ((row << 4) | col) as u8;
                v_plane[row * stride + col] = 0xF0 | (((row << 4) | col) & 0x0F) as u8;
            }
            // Sentinel bytes in the stride padding region — should
            // never appear in the output.
            u_plane[row * stride + 4] = 0xAA;
            u_plane[row * stride + 5] = 0xAA;
            v_plane[row * stride + 4] = 0xAA;
            v_plane[row * stride + 5] = 0xAA;
        }

        let uv = interleave_uv(&u_plane, stride, &v_plane, stride, chroma_w, chroma_h);

        assert_eq!(uv.len(), chroma_w * chroma_h * 2);
        // Row 0: U(0,0) V(0,0) U(0,1) V(0,1) U(0,2) V(0,2) U(0,3) V(0,3)
        // Row 1: U(1,0) V(1,0) U(1,1) V(1,1) ...
        let expected: Vec<u8> = vec![
            0x00, 0xF0, 0x01, 0xF1, 0x02, 0xF2, 0x03, 0xF3,
            0x10, 0xF0, 0x11, 0xF1, 0x12, 0xF2, 0x13, 0xF3,
        ];
        assert_eq!(uv, expected);
        // No stride-padding sentinels leaked into the output.
        assert!(!uv.contains(&0xAA), "stride padding leaked into UV plane");
    }

    fn make_test_bgra(width: u32, height: u32, t: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                // Coarse colour bands so lossy compression still produces
                // something recognisable.
                let r: u8 = if (x / 64 + t / 4) % 2 == 0 { 200 } else { 50 };
                let g: u8 = if (y / 64) % 2 == 0 { 200 } else { 50 };
                let b: u8 = 128;
                data.extend_from_slice(&[b, g, r, 255]);
            }
        }
        data
    }

    #[test]
    fn encode_decode_round_trip_produces_a_decoded_frame() {
        let w = 320;
        let h = 240;
        let mut enc = H264Encoder::new_bgra(w, h, 30, 2_000).expect("encoder");
        let mut dec = H264Decoder::new().expect("decoder");

        let mut got = None;
        // Feed several frames so the decoder definitely has enough
        // bitstream to emit one (SPS/PPS + IDR + a slice).
        for t in 0..4i64 {
            let bgra = make_test_bgra(w, h, t as u32);
            let packets = enc.encode_bgra(&bgra, t, t == 0).expect("encode");
            for p in packets {
                let frames = dec.decode(&p.data).expect("decode");
                if let Some(f) = frames.into_iter().next() {
                    got = Some(f);
                }
            }
        }
        let frame = got.expect("decoder produced a frame within four input frames");
        assert_eq!(frame.width, w);
        assert_eq!(frame.height, h);
        // NV12 plane sizes: Y is full res; UV is quarter-resolution
        // chroma but with two bytes per sample (interleaved U + V), so
        // the byte total matches the old U+V combined.
        let (cw, ch) = frame.chroma_dims();
        assert_eq!(frame.y.len(), (w * h) as usize);
        assert_eq!(frame.uv.len(), (cw * ch * 2) as usize);
    }
}
