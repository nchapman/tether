//! VAAPI implementation of the [`ProfileProbe`] contract.
//!
//! Both halves are real round trips against the live driver:
//!   - `probe_encode` constructs `VaapiEncoder` at 128×128 and
//!     encodes one BGRA frame. Driver-side rejection (which on some
//!     Intel paths only surfaces at `send_frame`, not `open()`) is
//!     caught here rather than at first-frame time in production.
//!   - `probe_decode` constructs `VaapiDecoder` and submits the
//!     fixture IDR; success is "produced a `Frame::Gpu` back" — a
//!     `Frame::Cpu` would mean the hwaccel silently fell back to
//!     software, which we treat as "not supported."

use tether_protocol::control::VideoProfile;

use crate::profile_probe::ProfileProbe;
use crate::{CodecError, Decoder, Frame, Result};

use super::{VaapiDecoder, VaapiEncoder};

pub(crate) struct VaapiProbe;

/// Probe canvas. 128×128 satisfies HEVC's minimum-block constraint on
/// Intel; H.264 accepts smaller but using the same dims keeps the
/// probe a single config across codecs and matches the fixture dims.
const PROBE_DIM: u32 = 128;
const PROBE_FPS: u32 = 30;
const PROBE_BITRATE_KBPS: u32 = 1_000;

impl ProfileProbe for VaapiProbe {
    fn probe_encode(profile: VideoProfile) -> Result<()> {
        let mut enc =
            VaapiEncoder::new(profile, PROBE_DIM, PROBE_DIM, PROBE_FPS, PROBE_BITRATE_KBPS)?;
        // Drive a single BGRA frame through the encoder so any driver
        // rejection that only fires at submit time (not construction)
        // surfaces here. A solid-grey buffer is the smallest "real"
        // input; the encoder doesn't care about content for capability
        // purposes.
        let bytes = vec![0x80u8; (PROBE_DIM * PROBE_DIM * 4) as usize];
        use crate::Encoder;
        let packets = enc.encode_bgra(&bytes, 0, true)?;
        // Some encoders buffer the first frame and don't return a packet
        // until the second. The contract we care about is "no error" —
        // the empty-packet case still means the encoder accepted the
        // input. (Real session use exercises the drain loop properly.)
        let _ = packets;
        Ok(())
    }

    fn probe_decode(profile: VideoProfile, fixture: &[u8]) -> Result<()> {
        let mut dec = VaapiDecoder::new(profile.codec)?;
        dec.submit(fixture)?;
        // Loop a few times — VAAPI sometimes needs a couple of
        // `receive_frame` polls before a single-IDR submit emits.
        for _ in 0..4 {
            match dec.next_frame()? {
                Some(Frame::Gpu(_)) => return Ok(()),
                Some(Frame::Cpu(_)) => return Err(CodecError::UnsupportedInputFormat),
                None => continue,
            }
        }
        Err(CodecError::UnsupportedInputFormat)
    }
}
