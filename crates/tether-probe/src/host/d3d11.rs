//! D3D11 (Windows) implementation of the [`ProfileProbe`] contract.
//!
//! Mirrors the VAAPI probe structure exactly:
//!   - `probe_encode`: construct encoder at 128×128, encode one BGRA
//!     frame, verify a packet comes back. That's it — no decoder, no
//!     round-trip. Encoder is dropped at function return.
//!   - `probe_decode`: construct decoder, submit fixture IDR, poll for
//!     a frame back.

use tether_codec::d3d11::{D3D11Decoder, D3D11Encoder};
use tether_codec::{Decoder, Encoder};
use tether_protocol::control::VideoProfile;

use crate::profile_probe::{ProbeError, ProfileProbe, Result};
use crate::PipelineStage;

pub(crate) struct D3D11Probe;

const PROBE_DIM: u32 = 128;
const PROBE_FPS: u32 = 30;
const PROBE_BITRATE_KBPS: u32 = 1_000;

impl ProfileProbe for D3D11Probe {
    fn probe_encode(profile: VideoProfile) -> Result<()> {
        tether_codec::av_log::with_probe_suppression(|| probe_encode_inner(profile))
    }

    fn probe_decode(profile: VideoProfile, fixture: &[u8]) -> Result<()> {
        tether_codec::av_log::with_probe_suppression(|| probe_decode_inner(profile, fixture))
    }
}

fn probe_encode_inner(profile: VideoProfile) -> Result<()> {
    let mut enc = D3D11Encoder::new(
        profile,
        PROBE_DIM,
        PROBE_DIM,
        PROBE_FPS,
        PROBE_BITRATE_KBPS,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    )
    .map_err(|e| ProbeError::from_codec(PipelineStage::Construct, e))?;

    let bgra = vec![0x80u8; (PROBE_DIM * PROBE_DIM * 4) as usize];
    let mut success = false;
    for pts in 0i64..8 {
        let pkts = enc
            .encode_bgra(&bgra, pts, pts == 0)
            .map_err(|e| ProbeError::from_codec(PipelineStage::Submit, e))?;
        if !pkts.is_empty() {
            success = true;
            break;
        }
    }
    // Drain the encoder so AMF's hardware session is fully released
    // before the next profile's probe tries to construct one.
    enc.shutdown();
    if success {
        Ok(())
    } else {
        Err(ProbeError::new(
            PipelineStage::Submit,
            "encoder produced no packets after 8 frames",
        ))
    }
}

fn probe_decode_inner(profile: VideoProfile, fixture: &[u8]) -> Result<()> {
    let mut dec =
        D3D11Decoder::new(profile.codec).map_err(|e| ProbeError::from_codec(PipelineStage::Construct, e))?;
    dec.submit(fixture)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
    // D3D11VA buffers a single IDR until either another packet or EOF
    // arrives (same as VideoToolbox). Signal EOF to drain.
    dec.signal_eof()
        .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
    // Accept Frame::Cpu: D3D11Decoder::next_frame currently always
    // downloads to CPU (GPU export pending Vulkan external-memory
    // support). Tighten to require Frame::Gpu when GPU export lands,
    // matching the VAAPI/VideoToolbox probe contract.
    for _ in 0..4 {
        match dec
            .next_frame()
            .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?
        {
            Some(_) => return Ok(()),
            None => continue,
        }
    }
    Err(ProbeError::new(
        PipelineStage::Decode,
        "decoder produced no frames after submit + eof + 4 polls",
    ))
}
