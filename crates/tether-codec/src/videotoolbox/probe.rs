//! VideoToolbox implementation of the [`ProfileProbe`] contract.
//!
//! Encode probe: real `VideoToolboxEncoder::new` + one BGRA frame at
//! the requested profile. No pre-filter — the encoder constructor sets
//! `sw_format` from `(chroma, bit_depth)` (see `encoder::vt_sw_format`)
//! and lets `encoder.open()` be the authority on whether the
//! VideoToolbox wrapper actually accepts that combination. An unsupported
//! combo surfaces as a real FFmpeg error rather than a hand-maintained
//! capability table.
//!
//! Decode probe: submit the fixture IDR and require a `Frame::Gpu`
//! back. The interesting failure mode this catches is VT's silent
//! software fallback — ffmpeg's `hevc_videotoolbox` wrapper happily
//! constructs for Rext input but the underlying VT session may route
//! through the native (SW) HEVC decoder on profiles the hardware
//! decoder block doesn't implement, emitting `Frame::Cpu` which our
//! decoder layer classifies as `UnsupportedInputFormat`. On M-series
//! silicon HEVC Main 4:4:4 8-bit *does* decode in hardware to a
//! `'444v'` IOSurface — see `docs/CODEC_CAPABILITIES.md` for the
//! per-profile expectation.

use tether_protocol::control::VideoProfile;

use crate::profile_probe::ProfileProbe;
use crate::{CodecError, Decoder, Encoder, Frame, Result};

use super::{VideoToolboxDecoder, VideoToolboxEncoder};

pub(crate) struct VideoToolboxProbe;

const PROBE_DIM: u32 = 128;
const PROBE_FPS: u32 = 30;
const PROBE_BITRATE_KBPS: u32 = 1_000;

impl ProfileProbe for VideoToolboxProbe {
    fn probe_encode(profile: VideoProfile) -> Result<()> {
        let mut enc = VideoToolboxEncoder::new(
            profile,
            PROBE_DIM,
            PROBE_DIM,
            PROBE_FPS,
            PROBE_BITRATE_KBPS,
        )?;
        let bytes = vec![0x80u8; (PROBE_DIM * PROBE_DIM * 4) as usize];
        let _ = enc.encode_bgra(&bytes, 0, true)?;
        Ok(())
    }

    fn probe_decode(profile: VideoProfile, fixture: &[u8]) -> Result<()> {
        let mut dec = VideoToolboxDecoder::new(profile.codec)?;
        dec.submit(fixture)?;
        // VT's wrapper buffers the first packet pending either a
        // second packet or EOF before emitting; the probe only has
        // one IDR, so signal EOF to force the drain.
        dec.signal_eof()?;
        loop {
            match dec.next_frame()? {
                Some(Frame::Gpu(_)) => return Ok(()),
                Some(Frame::Cpu(_)) => return Err(CodecError::UnsupportedInputFormat),
                None => return Err(CodecError::UnsupportedInputFormat),
            }
        }
    }
}
