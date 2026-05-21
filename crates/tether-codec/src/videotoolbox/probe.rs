//! VideoToolbox implementation of the [`ProfileProbe`] contract.
//!
//! Encode probe: real `VideoToolboxEncoder::new` + one BGRA frame.
//! VT's encoder path doesn't accept 4:4:4 inputs on any current Apple
//! Silicon generation; `VideoToolboxEncoder::new` returns
//! `CodecError::CodecNotFound` (Apple's Main444 profile isn't surfaced)
//! and that becomes `encode=false` in the resulting capability.
//!
//! Decode probe: submit the fixture IDR and require a `Frame::Gpu`
//! back. The interesting failure mode this catches is VT's silent
//! software fallback — ffmpeg's `hevc_videotoolbox` wrapper happily
//! constructs for Rext input but then routes through the native (SW)
//! HEVC decoder, emitting `Frame::Cpu` which our decoder layer
//! already classifies as `UnsupportedInputFormat`. Empirical: on
//! current M-series silicon ffmpeg can't decode HEVC 4:4:4 through VT,
//! so this probe will correctly report `decode=false` there.

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
        // VT only handles Yuv420 8-bit; the encoder constructor itself
        // doesn't take a VideoProfile so we short-circuit here. If a
        // future macOS HEVC Main444 encoder ever ships, the gate
        // relaxes in the encoder, not here.
        use tether_protocol::control::ChromaSubsampling;
        if profile.chroma != ChromaSubsampling::Yuv420 || profile.bit_depth != 8 {
            return Err(CodecError::UnsupportedInputFormat);
        }
        let mut enc = VideoToolboxEncoder::new(
            profile.codec,
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
