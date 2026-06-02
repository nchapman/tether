//! Opus encode/decode over the statically-linked FFmpeg libopus.
//!
//! The encoder buffers interleaved f32 PCM and emits one Opus packet per
//! whole `frame_size` (e.g. 480 samples/channel = 10 ms at 48 kHz), so callers
//! can push arbitrary capture chunk sizes. The decoder turns each packet back
//! into interleaved f32 and exposes [`OpusDecoder::conceal`] for the
//! jitter-buffer's loss path.
//!
//! ## Loss concealment (v1)
//!
//! Real Opus PLC is driven by calling `opus_decode(NULL)` on the raw libopus
//! API; FFmpeg's avcodec wrapper doesn't surface that, so v1 conceals a lost
//! frame with silence ([`OpusDecoder::conceal`]). At the 1% isolated-loss
//! target this is barely perceptible; if bursty-loss testing shows otherwise,
//! the decode side can move to direct libopus (already in our static build)
//! without touching the wire or the encoder.

use std::ffi::CString;
use std::slice;

use bytes::Bytes;
use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVPacket};
use rsmpeg::avutil::{ra, AVChannelLayout, AVDictionary, AVFrame};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;

use crate::{AudioError, AudioFrame, Result};

/// Opus session parameters. The defaults are the v1 shipping config: 48 kHz
/// stereo, 128 kbps CBR, 10 ms frames, restricted-lowdelay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpusConfig {
    pub sample_rate: u32,
    pub channels: u8,
    pub bitrate_bps: u32,
    pub frame_duration_ms: u32,
}

impl Default for OpusConfig {
    fn default() -> Self {
        Self {
            sample_rate: crate::SAMPLE_RATE_HZ,
            channels: crate::CHANNELS,
            bitrate_bps: 128_000,
            frame_duration_ms: 10,
        }
    }
}

impl OpusConfig {
    /// Samples per channel in one Opus frame (e.g. 480 at 48 kHz / 10 ms).
    #[must_use]
    pub fn frame_size(&self) -> usize {
        (self.sample_rate as usize * self.frame_duration_ms as usize) / 1000
    }

    /// The wire [`AudioConfig`](tether_protocol::audio::AudioConfig) the host
    /// advertises for this codec config. Stereo maps to one coupled Opus
    /// stream (RFC 7845 family 0); mono to a single uncoupled stream.
    #[must_use]
    pub fn wire_config(&self) -> tether_protocol::audio::AudioConfig {
        tether_protocol::audio::AudioConfig {
            sample_rate_hz: self.sample_rate,
            channels: self.channels,
            streams: 1,
            coupled_streams: u8::from(self.channels == 2),
            channel_mapping: (0..self.channels).collect(),
        }
    }
}

/// Wrap encoded bytes in an `AVPacket` ffmpeg owns, so it can be fed to a
/// decoder. Mirrors `tether_codec`'s private helper (rsmpeg has no safe
/// equivalent): allocate via `av_new_packet` so the packet owns and frees its
/// buffer, then memcpy our bytes in.
fn packet_from_bytes(bytes: &[u8]) -> Result<AVPacket> {
    let mut packet = AVPacket::new();
    let size = i32::try_from(bytes.len())
        .map_err(|_| AudioError::Ffmpeg(RsmpegError::AVError(ffi::AVERROR_INVALIDDATA)))?;
    // SAFETY: `packet` was just allocated by AVPacket::new(). av_new_packet
    // allocates `size + AV_INPUT_BUFFER_PADDING_SIZE` bytes, zeroes the
    // padding, and sets packet.data + packet.size; ownership transfers to the
    // packet, freed by its Drop via av_packet_free.
    let ret = unsafe { ffi::av_new_packet(packet.as_mut_ptr(), size) };
    if ret < 0 {
        return Err(AudioError::Ffmpeg(RsmpegError::AVError(ret)));
    }
    // SAFETY: packet.data now points to an owned buffer of exactly `size`
    // writable bytes (plus padding we don't touch).
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), packet.data, bytes.len());
    }
    Ok(packet)
}

/// Opus encoder: interleaved f32 PCM in, Opus packets out.
pub struct OpusEncoder {
    ctx: AVCodecContext,
    channels: u8,
    sample_rate: u32,
    sample_fmt: i32,
    /// Samples per channel per Opus frame (from the opened context).
    frame_size: usize,
    /// Interleaved PCM not yet aligned to a whole frame.
    accum: Vec<f32>,
    /// Running pts in samples (time_base 1/sample_rate).
    next_pts: i64,
}

// SAFETY: an ffmpeg codec context is safe to MOVE between threads but not to
// SHARE. We expose only `&mut self` methods, so the borrow checker serialises
// access within a thread and the inner pointer is never aliased.
unsafe impl Send for OpusEncoder {}

impl OpusEncoder {
    /// Build and open a libopus encoder for `cfg`.
    pub fn new(cfg: OpusConfig) -> Result<Self> {
        if !(1..=2).contains(&cfg.channels) {
            return Err(AudioError::UnsupportedChannelCount(cfg.channels));
        }
        let codec = AVCodec::find_encoder_by_name(c"libopus")
            .ok_or(AudioError::CodecNotFound("libopus"))?;
        let mut ctx = AVCodecContext::new(&codec);

        // libopus accepts interleaved float (FLT) or s16; we speak FLT so the
        // hot path is a single memcpy from the cpal-native f32 buffer.
        let sample_fmt = ffi::AV_SAMPLE_FMT_FLT;
        let supported = codec
            .sample_fmts()
            .ok_or(AudioError::NoSupportedSampleFormat)?;
        if !supported.iter().copied().any(|f| f == sample_fmt) {
            return Err(AudioError::NoSupportedSampleFormat);
        }

        ctx.set_sample_rate(cfg.sample_rate as i32);
        ctx.set_ch_layout(AVChannelLayout::from_nb_channels(i32::from(cfg.channels)).into_inner());
        ctx.set_sample_fmt(sample_fmt);
        ctx.set_bit_rate(i64::from(cfg.bitrate_bps));
        ctx.set_time_base(ra(1, cfg.sample_rate as i32));

        // Private libopus options: restricted-lowdelay for interactivity,
        // hard CBR for predictable bandwidth, and the frame duration that sets
        // the encoder's Opus frame size.
        let frame_dur = CString::new(cfg.frame_duration_ms.to_string())
            .expect("integer string has no interior NUL");
        let opts = AVDictionary::new(c"application", c"lowdelay", 0)
            .set(c"vbr", c"off", 0)
            .set(c"frame_duration", &frame_dur, 0);
        ctx.open(Some(opts))?;

        // After open the context reports the Opus frame size in samples; fall
        // back to the configured size if it's unset or nonsensical.
        let frame_size = usize::try_from(ctx.frame_size)
            .ok()
            .filter(|&n| n > 0)
            .unwrap_or_else(|| cfg.frame_size());

        Ok(Self {
            ctx,
            channels: cfg.channels,
            sample_rate: cfg.sample_rate,
            sample_fmt,
            frame_size,
            accum: Vec::new(),
            next_pts: 0,
        })
    }

    /// Samples per channel per emitted Opus packet.
    #[must_use]
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Feed interleaved f32 PCM. Returns zero or more Opus packets — one per
    /// whole frame the accumulated input now covers. Leftover samples are
    /// buffered for the next call.
    pub fn encode(&mut self, interleaved: &[f32]) -> Result<Vec<Bytes>> {
        self.accum.extend_from_slice(interleaved);
        let chunk = self.frame_size * self.channels as usize;
        let mut out = Vec::new();
        let mut consumed = 0;
        while self.accum.len() - consumed >= chunk {
            let frame = self.build_frame(&self.accum[consumed..consumed + chunk], self.next_pts)?;
            self.next_pts += self.frame_size as i64;
            consumed += chunk;
            self.ctx.send_frame(Some(&frame))?;
            self.drain_packets(&mut out)?;
        }
        if consumed > 0 {
            self.accum.drain(..consumed);
        }
        Ok(out)
    }

    /// Flush the encoder (e.g. on shutdown), returning any trailing packets.
    /// Does not pad a partial trailing frame — Opus needs whole frames, so a
    /// sub-frame remainder is dropped.
    pub fn flush(&mut self) -> Result<Vec<Bytes>> {
        let mut out = Vec::new();
        self.ctx.send_frame(None)?;
        self.drain_packets(&mut out)?;
        Ok(out)
    }

    fn build_frame(&self, chunk: &[f32], pts: i64) -> Result<AVFrame> {
        let mut frame = AVFrame::new();
        frame.set_nb_samples(i32::try_from(self.frame_size).expect("opus frame_size fits in i32"));
        frame.set_ch_layout(
            AVChannelLayout::from_nb_channels(i32::from(self.channels)).into_inner(),
        );
        frame.set_format(self.sample_fmt);
        frame.set_sample_rate(self.sample_rate as i32);
        frame.set_pts(pts);
        frame.alloc_buffer()?;
        // SAFETY: alloc_buffer gave us a packed FLT buffer of
        // frame_size * channels f32s in data[0]; `chunk` is exactly that many.
        unsafe {
            std::ptr::copy_nonoverlapping(
                chunk.as_ptr().cast::<u8>(),
                frame.data_mut()[0],
                std::mem::size_of_val(chunk),
            );
        }
        Ok(frame)
    }

    fn drain_packets(&mut self, out: &mut Vec<Bytes>) -> Result<()> {
        loop {
            match self.ctx.receive_packet() {
                Ok(pkt) => {
                    // SAFETY: a received packet's data points to pkt.size valid
                    // bytes owned by the packet; we copy them out before drop.
                    let len = usize::try_from(pkt.size).unwrap_or(0);
                    let data = unsafe { slice::from_raw_parts(pkt.data, len) };
                    out.push(Bytes::copy_from_slice(data));
                }
                Err(RsmpegError::EncoderDrainError | RsmpegError::EncoderFlushedError) => break,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }
}

/// Opus decoder: Opus packets in, interleaved f32 PCM out.
pub struct OpusDecoder {
    ctx: AVCodecContext,
    channels: u8,
    sample_rate: u32,
    frame_size: usize,
}

// SAFETY: see `OpusEncoder` — move-but-not-share, enforced by `&mut self`.
unsafe impl Send for OpusDecoder {}

impl OpusDecoder {
    /// Build and open a libopus decoder for `cfg`.
    ///
    /// We set channel layout + sample rate on the context rather than relying
    /// on OpusHead extradata: our packets are raw Opus frames off the wire with
    /// no container, and both ends share `cfg`, so the config is authoritative.
    pub fn new(cfg: OpusConfig) -> Result<Self> {
        // Bounds the per-channel plane indexing in `append_interleaved` against
        // an out-of-range (untrusted) channel count.
        if !(1..=2).contains(&cfg.channels) {
            return Err(AudioError::UnsupportedChannelCount(cfg.channels));
        }
        let codec = AVCodec::find_decoder_by_name(c"libopus")
            .ok_or(AudioError::CodecNotFound("libopus"))?;
        let mut ctx = AVCodecContext::new(&codec);
        ctx.set_sample_rate(cfg.sample_rate as i32);
        ctx.set_ch_layout(AVChannelLayout::from_nb_channels(i32::from(cfg.channels)).into_inner());
        ctx.set_pkt_timebase(ra(1, cfg.sample_rate as i32));
        ctx.open(None)?;
        Ok(Self {
            ctx,
            channels: cfg.channels,
            sample_rate: cfg.sample_rate,
            frame_size: cfg.frame_size(),
        })
    }

    /// Decode one Opus packet into interleaved f32 PCM.
    pub fn decode(&mut self, payload: &[u8]) -> Result<AudioFrame> {
        let pkt = packet_from_bytes(payload)?;
        self.ctx.send_packet(Some(&pkt))?;
        let mut samples = Vec::new();
        loop {
            match self.ctx.receive_frame() {
                Ok(frame) => self.append_interleaved(&frame, &mut samples)?,
                Err(RsmpegError::DecoderDrainError | RsmpegError::DecoderFlushedError) => break,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(AudioFrame::new(self.sample_rate, self.channels, samples))
    }

    /// Produce one frame of concealment for a lost packet. v1: silence (see
    /// the module-level note on PLC).
    #[must_use]
    pub fn conceal(&self) -> AudioFrame {
        AudioFrame::silence(self.sample_rate, self.channels, self.frame_size)
    }

    /// Read a decoded `AVFrame` into the interleaved f32 accumulator,
    /// converting whatever sample format the decoder produced.
    #[allow(clippy::cast_sign_loss)] // ffmpeg nb_samples / linesize are non-negative on a decoded frame
    fn append_interleaved(&self, frame: &AVFrame, out: &mut Vec<f32>) -> Result<()> {
        let n = frame.nb_samples as usize;
        let ch = self.channels as usize;
        match frame.format {
            // Packed float: already interleaved — straight copy.
            ffi::AV_SAMPLE_FMT_FLT => {
                let src = unsafe { slice::from_raw_parts(frame.data[0].cast::<f32>(), n * ch) };
                out.extend_from_slice(src);
            }
            // Planar float: one plane per channel; interleave.
            ffi::AV_SAMPLE_FMT_FLTP => {
                let planes: Vec<&[f32]> = (0..ch)
                    .map(|c| unsafe { slice::from_raw_parts(frame.data[c].cast::<f32>(), n) })
                    .collect();
                for i in 0..n {
                    for plane in &planes {
                        out.push(plane[i]);
                    }
                }
            }
            // Packed s16: interleaved i16 → f32.
            ffi::AV_SAMPLE_FMT_S16 => {
                let src = unsafe { slice::from_raw_parts(frame.data[0].cast::<i16>(), n * ch) };
                out.extend(src.iter().map(|&s| f32::from(s) / 32768.0));
            }
            // Planar s16: one i16 plane per channel → interleaved f32.
            ffi::AV_SAMPLE_FMT_S16P => {
                let planes: Vec<&[i16]> = (0..ch)
                    .map(|c| unsafe { slice::from_raw_parts(frame.data[c].cast::<i16>(), n) })
                    .collect();
                for i in 0..n {
                    for plane in &planes {
                        out.push(f32::from(plane[i]) / 32768.0);
                    }
                }
            }
            other => return Err(AudioError::UnsupportedSampleFormat(other)),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_pattern::sine_frame;

    /// A 440 Hz sine, encoded then decoded, comes back the right length and
    /// carrying real signal energy (Opus is lossy, so we check shape + energy,
    /// not sample equality).
    #[test]
    fn opus_round_trip_preserves_length_and_energy() {
        let cfg = OpusConfig::default();
        let mut enc = OpusEncoder::new(cfg).unwrap();
        let mut dec = OpusDecoder::new(cfg).unwrap();

        // Feed 200 ms of sine in 10 ms-aligned chunks.
        let fs = enc.frame_size();
        let mut decoded_frames = 0usize;
        let mut peak = 0.0f32;
        for i in 0..20 {
            let frame = sine_frame(cfg.sample_rate, cfg.channels, 440.0, fs, i * fs);
            for pkt in enc.encode(&frame.samples).unwrap() {
                assert!(!pkt.is_empty(), "opus packet must be non-empty");
                let out = dec.decode(&pkt).unwrap();
                assert_eq!(out.channels, cfg.channels);
                assert_eq!(out.frames(), fs, "decoded frame size must match encoder");
                decoded_frames += 1;
                peak = peak.max(out.samples.iter().fold(0.0, |a, &s| a.max(s.abs())));
            }
        }
        assert!(
            decoded_frames >= 18,
            "expected ~20 frames, got {decoded_frames}"
        );
        assert!(
            peak > 0.1,
            "decoded audio should carry signal energy, peak={peak}"
        );
    }

    /// Concealment yields exactly one silent frame of the configured size —
    /// the jitter buffer drops this in for a missing packet to keep the
    /// playback clock advancing.
    #[test]
    fn conceal_yields_one_silent_frame() {
        let cfg = OpusConfig::default();
        let dec = OpusDecoder::new(cfg).unwrap();
        let frame = dec.conceal();
        assert_eq!(frame.frames(), cfg.frame_size());
        assert_eq!(frame.channels, cfg.channels);
        assert!(frame.is_silent());
    }

    /// A decode-after-gap (packet N+1 with N dropped) still decodes cleanly —
    /// the FFmpeg decoder doesn't wedge on a missing predecessor.
    #[test]
    fn decode_survives_a_dropped_packet() {
        let cfg = OpusConfig::default();
        let mut enc = OpusEncoder::new(cfg).unwrap();
        let mut dec = OpusDecoder::new(cfg).unwrap();
        let fs = enc.frame_size();

        let mut packets = Vec::new();
        for i in 0..6 {
            let frame = sine_frame(cfg.sample_rate, cfg.channels, 440.0, fs, i * fs);
            packets.extend(enc.encode(&frame.samples).unwrap());
        }
        assert!(packets.len() >= 5);

        // Decode all but skip packet index 2 (simulating loss). Every
        // delivered packet must still decode to a full frame.
        for (i, pkt) in packets.iter().enumerate() {
            if i == 2 {
                let _ = dec.conceal(); // jitter buffer would insert concealment here
                continue;
            }
            let out = dec.decode(pkt).unwrap();
            assert_eq!(out.frames(), fs);
        }
    }

    /// The `frame_duration` AVOption can silently fall back to libopus's 20 ms
    /// default; pin that the opened encoder actually adopted our 10 ms frame
    /// (480 samples at 48 kHz), so a silent no-op becomes a test failure (per
    /// the repo convention for knobs that may silently no-op). The decoder's
    /// conceal-frame size must agree, or concealment frames would be the wrong
    /// length.
    #[test]
    fn encoder_and_decoder_agree_on_configured_frame_size() {
        let cfg = OpusConfig::default();
        let enc = OpusEncoder::new(cfg).unwrap();
        let dec = OpusDecoder::new(cfg).unwrap();
        assert_eq!(cfg.frame_size(), 480, "48 kHz / 10 ms");
        assert_eq!(enc.frame_size(), cfg.frame_size(), "encoder adopted 10 ms");
        assert_eq!(
            dec.conceal().frames(),
            cfg.frame_size(),
            "conceal size matches"
        );
    }

    /// An out-of-range channel count is rejected before the unsafe per-channel
    /// plane indexing in `append_interleaved` can be reached.
    #[test]
    fn constructors_reject_unsupported_channel_count() {
        for ch in [0u8, 3, 8] {
            let cfg = OpusConfig {
                channels: ch,
                ..OpusConfig::default()
            };
            assert!(
                matches!(OpusEncoder::new(cfg), Err(AudioError::UnsupportedChannelCount(c)) if c == ch)
            );
            assert!(
                matches!(OpusDecoder::new(cfg), Err(AudioError::UnsupportedChannelCount(c)) if c == ch)
            );
        }
    }
}
