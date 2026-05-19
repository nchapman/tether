//! Software H.264 encoder + decoder via libx264 / ffmpeg's `h264` decoder.
//!
//! v0 goals: get bytes-on-the-wire down from ~240 MB/s (raw 1080p BGRA)
//! to ~5 MB/s (zerolatency H.264). Quality tuning, B-frame strategies,
//! and adaptive bitrate land later — for now we hardcode the lowest-
//! latency preset and trust libx264's defaults beyond that.

use ffmpeg_next::{
    codec, decoder, encoder, format::Pixel, frame, picture, software::scaling,
    util::dictionary::Owned as Dictionary, Packet, Rational,
};

use tether_protocol::control::CodecKind;

use crate::{init_ffmpeg, CodecError, DecodedFrame, Decoder, Encoder, EncodedPacket, Result};

pub struct H264Encoder {
    encoder: encoder::Video,
    bgra_to_yuv: scaling::Context,
    yuv_frame: frame::Video,
    bgra_frame: frame::Video,
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
        let codec = encoder::find(codec::Id::H264)
            .ok_or(CodecError::Ffmpeg(ffmpeg_next::Error::EncoderNotFound))?;
        let mut ctx = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()?;
        ctx.set_width(width);
        ctx.set_height(height);
        ctx.set_format(Pixel::YUV420P);
        let fps_i32 = i32::try_from(fps.max(1)).unwrap_or(60);
        ctx.set_time_base(Rational(1, fps_i32));
        ctx.set_frame_rate(Some(Rational(fps_i32, 1)));
        ctx.set_bit_rate(bitrate_kbps as usize * 1000);
        // GOP = 1 second of frames. The MVP datagram path has no FEC and
        // no on-demand IDR signalling yet, so any lost fragment corrupts
        // the P-frame chain until the next IDR. A 1-second IDR cadence
        // caps the worst-case "garbled text" window without spending
        // catastrophic bandwidth on keyframes. Drop this back to a long
        // GOP once `ControlMessage::ForceIdr` is wired up end-to-end.
        ctx.set_gop(fps.max(1));
        ctx.set_max_b_frames(0);

        let mut opts = Dictionary::new();
        opts.set("preset", "ultrafast");
        opts.set("tune", "zerolatency");
        // baseline profile = no B-frames, no CABAC entropy coding. ultrafast
        // preset already drops most of what would slow us down; this is
        // belt-and-braces for predictability across libx264 versions.
        opts.set("profile", "baseline");
        // tune=zerolatency already sets rc-lookahead=0 + sync-lookahead=0,
        // but pinning min-keyint=keyint forces *exactly* periodic IDRs
        // (no early IDRs from scene-cut detection) so the GOP cadence
        // matches what `set_gop` promised.
        let gop = fps.max(1);
        opts.set(
            "x264-params",
            &format!("keyint={gop}:min-keyint={gop}:scenecut=0"),
        );

        let encoder = ctx.open_with(opts)?;

        let bgra_to_yuv = scaling::Context::get(
            Pixel::BGRA,
            width,
            height,
            Pixel::YUV420P,
            width,
            height,
            scaling::Flags::FAST_BILINEAR,
        )?;

        let yuv_frame = frame::Video::new(Pixel::YUV420P, width, height);
        let bgra_frame = frame::Video::new(Pixel::BGRA, width, height);
        let bgra_row_bytes = (width * 4) as usize;

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
    fn encode_bgra(
        &mut self,
        bgra: &[u8],
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>> {
        let expected = self.bgra_row_bytes * self.height as usize;
        if bgra.len() != expected {
            return Err(CodecError::BufferTooSmall {
                got: bgra.len(),
                expected,
            });
        }

        // Copy BGRA bytes into the encoder's input frame, row-by-row in
        // case stride > width*4 (AVFrame buffers are usually 32-byte
        // aligned, so 1920*4=7680 bytes/row is already a clean stride —
        // but smaller widths may have padding).
        let stride = self.bgra_frame.stride(0);
        let plane = self.bgra_frame.data_mut(0);
        if stride == self.bgra_row_bytes {
            plane[..expected].copy_from_slice(bgra);
        } else {
            for row in 0..self.height as usize {
                let src = row * self.bgra_row_bytes;
                let dst = row * stride;
                plane[dst..dst + self.bgra_row_bytes]
                    .copy_from_slice(&bgra[src..src + self.bgra_row_bytes]);
            }
        }

        self.bgra_to_yuv
            .run(&self.bgra_frame, &mut self.yuv_frame)?;
        self.yuv_frame.set_pts(Some(pts));
        self.yuv_frame.set_kind(if force_keyframe {
            picture::Type::I
        } else {
            picture::Type::None
        });

        self.encoder.send_frame(&self.yuv_frame)?;

        let mut out = Vec::new();
        let mut packet = Packet::empty();
        while self.encoder.receive_packet(&mut packet).is_ok() {
            if let Some(data) = packet.data() {
                out.push(EncodedPacket {
                    data: data.to_vec(),
                    keyframe: packet.is_key(),
                });
            }
        }
        Ok(out)
    }

    fn codec_kind(&self) -> CodecKind {
        CodecKind::H264
    }
}

pub struct H264Decoder {
    decoder: decoder::Video,
    yuv_to_bgra: Option<scaling::Context>,
    bgra_frame: frame::Video,
    bgra_dims: (u32, u32),
}

// SAFETY: same rationale as H264Encoder above.
unsafe impl Send for H264Decoder {}

impl H264Decoder {
    pub fn new() -> Result<Self> {
        init_ffmpeg();
        let codec = decoder::find(codec::Id::H264)
            .ok_or(CodecError::Ffmpeg(ffmpeg_next::Error::DecoderNotFound))?;
        let ctx = codec::context::Context::new_with_codec(codec);
        let decoder = ctx.decoder().video()?;
        Ok(Self {
            decoder,
            yuv_to_bgra: None,
            bgra_frame: frame::Video::empty(),
            bgra_dims: (0, 0),
        })
    }

    fn ensure_scaler(&mut self, src_format: Pixel, w: u32, h: u32) -> Result<()> {
        if self.bgra_dims == (w, h) && self.yuv_to_bgra.is_some() {
            return Ok(());
        }
        self.yuv_to_bgra = Some(scaling::Context::get(
            src_format,
            w,
            h,
            Pixel::BGRA,
            w,
            h,
            scaling::Flags::FAST_BILINEAR,
        )?);
        self.bgra_frame = frame::Video::new(Pixel::BGRA, w, h);
        self.bgra_dims = (w, h);
        Ok(())
    }
}

impl Decoder for H264Decoder {
    fn decode(&mut self, encoded: &[u8]) -> Result<Vec<DecodedFrame>> {
        if encoded.is_empty() {
            return Ok(Vec::new());
        }
        let packet = Packet::copy(encoded);
        self.decoder.send_packet(&packet)?;

        let mut out = Vec::new();
        let mut yuv = frame::Video::empty();
        loop {
            match self.decoder.receive_frame(&mut yuv) {
                Ok(()) => {}
                Err(ffmpeg_next::Error::Other { errno })
                    if errno == ffmpeg_next::error::EAGAIN =>
                {
                    break;
                }
                Err(ffmpeg_next::Error::Eof) => break,
                Err(e) => return Err(CodecError::Ffmpeg(e)),
            }
            let w = yuv.width();
            let h = yuv.height();
            self.ensure_scaler(yuv.format(), w, h)?;
            let sws = self
                .yuv_to_bgra
                .as_mut()
                .expect("scaler installed by ensure_scaler");
            sws.run(&yuv, &mut self.bgra_frame)?;

            // Pack rows tightly into the output Vec — sws output stride
            // may exceed width*4.
            let stride = self.bgra_frame.stride(0);
            let row_bytes = (w * 4) as usize;
            let plane = self.bgra_frame.data(0);
            let mut packed = Vec::with_capacity(row_bytes * h as usize);
            if stride == row_bytes {
                packed.extend_from_slice(&plane[..row_bytes * h as usize]);
            } else {
                for row in 0..h as usize {
                    let start = row * stride;
                    packed.extend_from_slice(&plane[start..start + row_bytes]);
                }
            }
            out.push(DecodedFrame {
                width: w,
                height: h,
                data: packed,
            });
        }
        Ok(out)
    }

    fn codec_kind(&self) -> CodecKind {
        CodecKind::H264
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
        assert_eq!(frame.data.len(), (w * h * 4) as usize);
        // alpha is always 255 because BGRA conversion fills it in
        for px in frame.data.chunks_exact(4) {
            assert_eq!(px[3], 255);
        }
    }
}
