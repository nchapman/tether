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
        // Temporary gate: the Linux gpuconvert bridges (Nv12DmaBuf /
        // Yuv444DmaBuf) are 8-bit only today. The matching 10-bit
        // shaders (bgra_to_p010.wgsl / bgra_to_p410.wgsl) exist as
        // scaffolding but aren't plumbed into the production
        // GpuConvertBridge variants yet. If a driver returns
        // encode=true for a 10-bit profile here, the host's
        // negotiator could pick it and then feed 8-bit dma-bufs to
        // the 10-bit encoder — a silent format mismatch. Gate at the
        // probe layer so the profile stays out of
        // `supported_encode_profiles()` until the bridge is wired.
        if profile.bit_depth != 8 {
            return Err(crate::CodecError::NoHardwareCodec(format!(
                "VAAPI 10-bit encode ({:?} {}-bit) is gated pending the \
                 gpuconvert P010/P410 bridge wiring; remove this guard once \
                 Nv12DmaBuf / Yuv444DmaBuf grow 10-bit variants",
                profile.chroma, profile.bit_depth
            )));
        }
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
