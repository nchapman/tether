//! D3D11 (Windows) implementation of the [`ProfileProbe`] contract.
//!
//! Windows implements only `probe_decode` — `construct decoder, submit
//! fixture IDR, poll for a frame back`. There is no `probe_encode`: the
//! AMF single-session limit makes a destructive per-profile encode probe
//! unsafe, so `probe_host` reports encode support statically and the
//! trait drops the method on Windows.

use tether_codec::d3d11::D3D11Decoder;
use tether_codec::Decoder;
use tether_protocol::control::VideoProfile;

use crate::profile_probe::{ProbeError, ProfileProbe, Result};
use crate::PipelineStage;

pub(crate) struct D3D11Probe;

impl ProfileProbe for D3D11Probe {
    fn probe_decode(profile: VideoProfile, fixture: &[u8]) -> Result<()> {
        tether_codec::av_log::with_probe_suppression(|| probe_decode_inner(profile, fixture))
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
