//! Opus encode/decode bound directly to libopus (see [`crate::opus_sys`]).
//!
//! The encoder buffers interleaved f32 PCM and emits one Opus packet per whole
//! `frame_size` (e.g. 480 samples/channel = 10 ms at 48 kHz), so callers can
//! push arbitrary capture chunk sizes. The decoder turns each packet back into
//! interleaved f32 and exposes [`OpusDecoder::conceal`] for the jitter buffer's
//! loss path.
//!
//! ## Loss concealment
//!
//! [`OpusDecoder::conceal`] is **real Opus PLC**: `opus_decode_float(NULL, …)`
//! reconstructs the missing frame from decoder state (extrapolating the last
//! good frame), rather than splicing in digital silence. This is what makes a
//! dropped packet a soft, brief artifact instead of a broadband click. PLC
//! advances decoder state, so it takes `&mut self` and the missing frame must be
//! exactly one `frame_size` long (a libopus requirement for staying in the right
//! state for the next packet).

use std::ffi::CStr;
use std::os::raw::c_int;
use std::ptr;

use bytes::Bytes;

use crate::opus_sys;
use crate::{AudioError, AudioFrame, Result};

/// Recommended safe upper bound for a single encoded Opus packet (libopus docs).
const MAX_PACKET_BYTES: usize = 4000;

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

/// Build an [`AudioError::Opus`] from a libopus return code, attaching the
/// library's own human-readable message.
fn opus_err(ctx: &'static str, code: c_int) -> AudioError {
    // SAFETY: opus_strerror returns a static NUL-terminated string (or null).
    let msg = unsafe {
        let p = opus_sys::opus_strerror(code);
        if p.is_null() {
            "unknown error".to_owned()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    AudioError::Opus(format!("{ctx}: {msg}"))
}

/// Opus encoder: interleaved f32 PCM in, Opus packets out.
pub struct OpusEncoder {
    enc: *mut opus_sys::OpusEncoder,
    channels: u8,
    frame_size: usize,
    /// Interleaved PCM not yet aligned to a whole frame.
    accum: Vec<f32>,
}

// SAFETY: a libopus encoder is safe to MOVE between threads but not to SHARE.
// We expose only `&mut self` methods, so the borrow checker serialises access
// within a thread and the inner pointer is never aliased.
unsafe impl Send for OpusEncoder {}

impl OpusEncoder {
    /// Build a libopus encoder for `cfg` in restricted-lowdelay, hard-CBR mode.
    pub fn new(cfg: OpusConfig) -> Result<Self> {
        if !(1..=2).contains(&cfg.channels) {
            return Err(AudioError::UnsupportedChannelCount(cfg.channels));
        }
        let fs = c_int::try_from(cfg.sample_rate)
            .map_err(|_| AudioError::Opus("sample_rate out of range".to_owned()))?;
        let mut err: c_int = opus_sys::OPUS_OK;
        // SAFETY: opus_encoder_create returns a heap encoder or null with `err`
        // set; we check both before use. Freed in Drop.
        let enc = unsafe {
            opus_sys::opus_encoder_create(
                fs,
                c_int::from(cfg.channels),
                opus_sys::OPUS_APPLICATION_RESTRICTED_LOWDELAY,
                &mut err,
            )
        };
        if enc.is_null() || err != opus_sys::OPUS_OK {
            return Err(opus_err("opus_encoder_create", err));
        }

        let mut this = Self {
            enc,
            channels: cfg.channels,
            frame_size: cfg.frame_size(),
            accum: Vec::new(),
        };
        // Hard CBR for predictable bandwidth, the configured target bitrate, and
        // max complexity (cheap at audio rates, best quality).
        this.set_ctl(opus_sys::OPUS_SET_VBR_REQUEST, 0)?;
        let bitrate = c_int::try_from(cfg.bitrate_bps)
            .map_err(|_| AudioError::Opus("bitrate out of range".to_owned()))?;
        this.set_ctl(opus_sys::OPUS_SET_BITRATE_REQUEST, bitrate)?;
        this.set_ctl(opus_sys::OPUS_SET_COMPLEXITY_REQUEST, 10)?;
        Ok(this)
    }

    /// Issue an encoder CTL that takes a single `opus_int32` value.
    fn set_ctl(&mut self, request: c_int, value: c_int) -> Result<()> {
        // SAFETY: every `request` we pass is a SET taking one opus_int32 arg,
        // matching the variadic C signature.
        let ret = unsafe { opus_sys::opus_encoder_ctl(self.enc, request, value) };
        if ret != opus_sys::OPUS_OK {
            return Err(opus_err("opus_encoder_ctl", ret));
        }
        Ok(())
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
        let chunk = self.frame_size * usize::from(self.channels);
        let frame_size = c_int::try_from(self.frame_size)
            .map_err(|_| AudioError::Opus("frame_size out of range".to_owned()))?;

        let mut out = Vec::new();
        let mut consumed = 0;
        while self.accum.len() - consumed >= chunk {
            let pcm = &self.accum[consumed..consumed + chunk];
            let mut buf = [0u8; MAX_PACKET_BYTES];
            let cap = c_int::try_from(buf.len()).expect("MAX_PACKET_BYTES fits in c_int");
            // SAFETY: `pcm` holds exactly frame_size*channels f32 (one Opus
            // frame); `buf` has `cap` writable bytes. opus_encode_float reads the
            // former and writes at most `cap` bytes.
            let n = unsafe {
                opus_sys::opus_encode_float(
                    self.enc,
                    pcm.as_ptr(),
                    frame_size,
                    buf.as_mut_ptr(),
                    cap,
                )
            };
            if n < 0 {
                return Err(opus_err("opus_encode_float", n));
            }
            let n = usize::try_from(n).expect("non-negative after the check above");
            out.push(Bytes::copy_from_slice(&buf[..n]));
            consumed += chunk;
        }
        if consumed > 0 {
            self.accum.drain(..consumed);
        }
        Ok(out)
    }

    /// Flush on shutdown. libopus buffers nothing internally (each whole frame is
    /// emitted by [`encode`](Self::encode)), so any sub-frame remainder is
    /// dropped and there are no trailing packets. Kept for API symmetry.
    #[allow(clippy::unused_self, clippy::missing_const_for_fn)]
    pub fn flush(&mut self) -> Result<Vec<Bytes>> {
        Ok(Vec::new())
    }
}

impl Drop for OpusEncoder {
    fn drop(&mut self) {
        // SAFETY: `enc` came from opus_encoder_create and is destroyed exactly
        // once, here.
        unsafe { opus_sys::opus_encoder_destroy(self.enc) };
    }
}

/// Opus decoder: Opus packets in, interleaved f32 PCM out.
pub struct OpusDecoder {
    dec: *mut opus_sys::OpusDecoder,
    channels: u8,
    sample_rate: u32,
    frame_size: usize,
}

// SAFETY: see `OpusEncoder` — move-but-not-share, enforced by `&mut self`.
unsafe impl Send for OpusDecoder {}

impl OpusDecoder {
    /// Build a libopus decoder for `cfg`. Both ends share `cfg`, so the rate /
    /// channel count is authoritative (our packets are raw Opus frames off the
    /// wire, no OpusHead container).
    pub fn new(cfg: OpusConfig) -> Result<Self> {
        if !(1..=2).contains(&cfg.channels) {
            return Err(AudioError::UnsupportedChannelCount(cfg.channels));
        }
        let fs = c_int::try_from(cfg.sample_rate)
            .map_err(|_| AudioError::Opus("sample_rate out of range".to_owned()))?;
        let mut err: c_int = opus_sys::OPUS_OK;
        // SAFETY: returns a heap decoder or null with `err` set; both checked.
        // Freed in Drop.
        let dec = unsafe { opus_sys::opus_decoder_create(fs, c_int::from(cfg.channels), &mut err) };
        if dec.is_null() || err != opus_sys::OPUS_OK {
            return Err(opus_err("opus_decoder_create", err));
        }
        Ok(Self {
            dec,
            channels: cfg.channels,
            sample_rate: cfg.sample_rate,
            frame_size: cfg.frame_size(),
        })
    }

    /// Decode one Opus packet into interleaved f32 PCM.
    pub fn decode(&mut self, payload: &[u8]) -> Result<AudioFrame> {
        let len = c_int::try_from(payload.len())
            .map_err(|_| AudioError::Opus("packet too large".to_owned()))?;
        self.decode_into(payload.as_ptr(), len)
    }

    /// Conceal one lost frame with real Opus PLC (a null-packet decode). On the
    /// (unexpected) error path, falls back to silence so the playback clock
    /// still advances.
    pub fn conceal(&mut self) -> AudioFrame {
        self.decode_into(ptr::null(), 0).unwrap_or_else(|_| {
            AudioFrame::silence(self.sample_rate, self.channels, self.frame_size)
        })
    }

    /// Shared decode body. `data` is either a valid `len`-byte packet or null
    /// (PLC); libopus writes up to `frame_size` samples/channel into `pcm`.
    fn decode_into(&mut self, data: *const u8, len: c_int) -> Result<AudioFrame> {
        let ch = usize::from(self.channels);
        let mut pcm = vec![0.0f32; self.frame_size * ch];
        let frame_size = c_int::try_from(self.frame_size)
            .map_err(|_| AudioError::Opus("frame_size out of range".to_owned()))?;
        // SAFETY: `pcm` has frame_size*channels capacity; `data` is null (PLC) or
        // points to `len` readable bytes. decode_fec is 0 — we don't use FEC.
        let n = unsafe {
            opus_sys::opus_decode_float(self.dec, data, len, pcm.as_mut_ptr(), frame_size, 0)
        };
        if n < 0 {
            return Err(opus_err("opus_decode_float", n));
        }
        let n = usize::try_from(n).expect("non-negative after the check above");
        pcm.truncate(n * ch);
        Ok(AudioFrame::new(self.sample_rate, self.channels, pcm))
    }
}

impl Drop for OpusDecoder {
    fn drop(&mut self) {
        // SAFETY: `dec` came from opus_decoder_create and is destroyed exactly
        // once, here.
        unsafe { opus_sys::opus_decoder_destroy(self.dec) };
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

    /// Real PLC, not silence: after priming the decoder with a steady tone,
    /// `conceal()` extrapolates the waveform — the concealed frame is the right
    /// size and carries signal energy rather than digital silence. This is the
    /// behaviour the direct-libopus migration buys over the old avcodec path
    /// (which could only splice in zeros).
    #[test]
    fn plc_after_tone_carries_energy_not_silence() {
        let cfg = OpusConfig::default();
        let mut enc = OpusEncoder::new(cfg).unwrap();
        let mut dec = OpusDecoder::new(cfg).unwrap();
        let fs = enc.frame_size();

        for i in 0..10 {
            let frame = sine_frame(cfg.sample_rate, cfg.channels, 440.0, fs, i * fs);
            for pkt in enc.encode(&frame.samples).unwrap() {
                dec.decode(&pkt).unwrap();
            }
        }

        let concealed = dec.conceal();
        assert_eq!(concealed.frames(), fs, "PLC frame is one full frame");
        assert_eq!(concealed.channels, cfg.channels);
        let peak = concealed
            .samples
            .iter()
            .fold(0.0f32, |a, &s| a.max(s.abs()));
        assert!(
            peak > 0.01,
            "PLC should extrapolate signal energy, got peak={peak}"
        );
        assert!(
            !concealed.is_silent(),
            "PLC output must not be digital silence"
        );
    }

    /// A decode-after-gap (packet N+1 with N dropped) still decodes cleanly —
    /// the decoder doesn't wedge on a missing predecessor, and the concealment
    /// call in the gap keeps it in the right state.
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

        // Decode all but skip packet index 2 (simulating loss). Every delivered
        // packet must still decode to a full frame.
        for (i, pkt) in packets.iter().enumerate() {
            if i == 2 {
                let _ = dec.conceal(); // jitter buffer would insert concealment here
                continue;
            }
            let out = dec.decode(pkt).unwrap();
            assert_eq!(out.frames(), fs);
        }
    }

    /// The encoder and decoder agree on the configured 10 ms frame size (480
    /// samples at 48 kHz), and concealment frames are that same length — so a
    /// concealed gap is exactly one frame on the playback clock.
    #[test]
    fn encoder_and_decoder_agree_on_configured_frame_size() {
        let cfg = OpusConfig::default();
        let enc = OpusEncoder::new(cfg).unwrap();
        let mut dec = OpusDecoder::new(cfg).unwrap();
        assert_eq!(cfg.frame_size(), 480, "48 kHz / 10 ms");
        assert_eq!(enc.frame_size(), cfg.frame_size(), "encoder frame size");
        assert_eq!(
            dec.conceal().frames(),
            cfg.frame_size(),
            "conceal size matches"
        );
    }

    /// An out-of-range channel count is rejected before any libopus call.
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
