//! VAAPI implementation of the [`ProfileProbe`] contract.
//!
//! Both halves are real round trips against the live driver:
//!   - `probe_encode` constructs `VaapiEncoder` at 128×128 and
//!     encodes one BGRA frame for 8-bit profiles. Driver-side
//!     rejection (which on some Intel paths only surfaces at
//!     `send_frame`, not `open()`) is caught here rather than at
//!     first-frame time in production.
//!   - `probe_decode` constructs `VaapiDecoder` and submits the
//!     fixture IDR; success is "produced a `Frame::Gpu` back" — a
//!     `Frame::Cpu` would mean the hwaccel silently fell back to
//!     software, which we treat as "not supported."
//!
//! Step 2 of the migration: this is a verbatim move from
//! `tether-codec::vaapi::probe`. Step 3 extends `probe_encode` for
//! 10-bit profiles to also run a real `Bgra2P010DmaBuf` →
//! `submit_dmabuf` round trip, closing the gap that the host's
//! separate `warm_gpuconvert_capability_cache` used to fill.

use tether_codec::vaapi::{VaapiDecoder, VaapiEncoder};
use tether_codec::{CodecError, Decoder, Encoder, Frame, Result};
use tether_protocol::control::VideoProfile;

use crate::profile_probe::ProfileProbe;

pub(crate) struct VaapiProbe;

/// Probe canvas. 128×128 satisfies HEVC's minimum-block constraint on
/// Intel; H.264 accepts smaller but using the same dims keeps the
/// probe a single config across codecs and matches the fixture dims.
const PROBE_DIM: u32 = 128;
const PROBE_FPS: u32 = 30;
const PROBE_BITRATE_KBPS: u32 = 1_000;

impl ProfileProbe for VaapiProbe {
    fn probe_encode(profile: VideoProfile) -> Result<()> {
        // Narrow gate for HEVC 4:4:4 10-bit: VAAPI's encode side
        // takes packed XV30 input (vaapi_drm_format_map has no P410
        // entry), and no gpuconvert bridge produces XV30 today. If a
        // driver constructs the encoder for (Yuv444, 10) successfully
        // and we report encode=true, the host's send loop would crash
        // trying to build a bridge for XV30. Until an XV30 bridge
        // lands, gate this combination explicitly. The 4:2:0 10-bit
        // and 4:4:4 8-bit paths are both fully wired and not gated.
        if profile.chroma == tether_protocol::control::ChromaSubsampling::Yuv444
            && profile.bit_depth == 10
        {
            return Err(CodecError::NoHardwareCodec(
                "HEVC 4:4:4 10-bit (XV30 input) — VAAPI accepts packed XV30 \
                 input but no gpuconvert bridge produces it yet. Profile \
                 will become available when Bgra2Xv30DmaBuf ships."
                    .into(),
            ));
        }

        let mut enc =
            VaapiEncoder::new(profile, PROBE_DIM, PROBE_DIM, PROBE_FPS, PROBE_BITRATE_KBPS)?;
        if profile.bit_depth == 8 {
            // Drive a single BGRA frame through the encoder so any
            // driver rejection that only fires at submit time (not
            // construction) surfaces here. The CPU-upload path is
            // 8-bit-only by design — see `encode_bgra`'s comment.
            let bytes = vec![0x80u8; (PROBE_DIM * PROBE_DIM * 4) as usize];
            let packets = enc.encode_bgra(&bytes, 0, true)?;
            // Some encoders buffer the first frame and don't return a
            // packet until the second. The contract we care about is
            // "no error" — the empty-packet case still means the
            // encoder accepted the input.
            let _ = packets;
        } else {
            // 10-bit probe is construction-only **in this step**.
            // Step 3 of the migration grows a real submit_dmabuf
            // round trip here via `Bgra2P010DmaBuf` — that's the
            // architectural reason this probe lives in tether-probe
            // now (so it can pull in gpuconvert) instead of in
            // tether-codec.
            let _ = enc;
        }
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
