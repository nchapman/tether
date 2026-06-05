//! NVENC implementation of the host encode probe.
//!
//! Only the encode half lives here. Decode on an NVIDIA host goes through
//! `NvdecDecoder`, probed by [`super::nvdec::NvdecProbe`] (see
//! `host::probe_decode`) — NOT VAAPI, whose `nvidia-vaapi-driver` decoder
//! SIGSEGVs. `host::probe_encode` tries this first on an NVIDIA host and falls
//! back to VAAPI, mirroring the live `build_encoder` dispatch so the advertised
//! capability set matches what the session encoder will actually pick.
//!
//! The probe is a real round trip against the live driver, not a
//! construction-only check: `NvencEncoder::new` opening the codec is not
//! sufficient evidence the CUDA upload + encode actually runs, so the 8-bit
//! path encodes one frame. The 10-bit (and, later, 4:4:4) paths need the
//! DMA-BUF → CUDA submit, which lands with that encoder method in a later
//! milestone; until then they report unsupported here and the candidate
//! dispatch falls back to VAAPI rather than advertising a half-probed profile.

use tether_codec::nvenc::NvencEncoder;
use tether_codec::Encoder;
use tether_protocol::control::VideoProfile;

use crate::profile_probe::{ProbeError, Result};
use crate::PipelineStage;

pub(crate) struct NvencProbe;

// NvencProbe deliberately does NOT implement the `ProfileProbe` trait (unlike
// VaapiProbe): it has only an encode half. Decode on an NVIDIA host is probed
// by `NvdecProbe` (wired through `host::probe_decode`), so a trait-required
// `probe_decode` here would be a never-called delegating stub — the kind of
// dead code the project avoids. `host::probe_encode` calls the inherent method
// directly.

/// Probe canvas. 128×128 matches the VAAPI probe and the fixture dims, and
/// satisfies HEVC's minimum-block constraint.
// 256, not the 128 the VAAPI probe uses: constructing a 128×128 HEVC Main10
// (P010) NVENC encoder SIGSEGVs *inside the NVENC runtime* — Main10 has a
// minimum encode dimension that 128px violates, and FFmpeg's wrapper faults
// rather than erroring. 256 is the smallest dimension verified to construct
// every advertised profile (the M3 P010 round-trip runs at 256). Real
// sessions are always far larger; this is a probe-only floor.
const PROBE_DIM: u32 = 256;
const PROBE_FPS: u32 = 30;
const PROBE_BITRATE_KBPS: u32 = 1_000;

impl NvencProbe {
    /// Probe whether this NVIDIA host can encode `profile` through NVENC.
    /// `Ok(())` means a real frame went through the encoder; `Err` carries
    /// the [`PipelineStage`] that rejected it, and the caller
    /// (`host::probe_encode`) falls back to VAAPI.
    pub(crate) fn probe_encode(profile: VideoProfile) -> Result<()> {
        tether_codec::av_log::with_probe_suppression(|| probe_encode_inner(profile))
    }
}

fn probe_encode_inner(profile: VideoProfile) -> Result<()> {
    // Construction catches "driver/codec can't do this profile at all":
    // codec not built (no `--enable-nvenc`), no NVIDIA device / `libcuda`,
    // AV1 on a pre-Ada card, or 4:4:4 (no planar bridge yet → mapped
    // UnsupportedInputFormat in `nvenc_sw_format`).
    let mut enc = NvencEncoder::new(profile, PROBE_DIM, PROBE_DIM, PROBE_FPS, PROBE_BITRATE_KBPS)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Construct, e))?;

    match profile.bit_depth {
        // 8-bit: CPU BGRA upload (swscale → NV12 → host→CUDA transfer →
        // encode) exercises the full encode path, not just `open()`.
        8 => {
            let bytes = vec![0x80u8; (PROBE_DIM * PROBE_DIM * 4) as usize];
            enc.encode_bgra(&bytes, 0, true)
                .map_err(|e| ProbeError::from_codec(PipelineStage::Submit, e))?;
        }
        // 10-bit 4:2:0 constructs fine (P010 pool) but proving it needs the
        // DMA-BUF → CUDA `submit_dmabuf` round trip — construction alone is
        // the half-credit answer the probe architecture rejects. That encoder
        // method lands in a later milestone. Tag this `Construct` (not
        // `Submit`): we never reach a submit, so a `Submit` tag would send an
        // operator debugging DMA-BUF import for what is a not-yet-wired path.
        // Reporting unsupported makes the candidate dispatch fall back to VAAPI.
        _ => {
            return Err(ProbeError::new(
                PipelineStage::Construct,
                "NVENC 10-bit encode not yet probed (DMA-BUF → CUDA submit path \
                 lands in a later milestone); falling back to VAAPI",
            ));
        }
    }
    Ok(())
}
