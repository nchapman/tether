//! NVDEC implementation of the decode probe.
//!
//! On an NVIDIA host the decode path is native NVDEC, not VAAPI — the
//! default VAAPI device there is the decode-only `nvidia-vaapi-driver`,
//! whose `VaapiDecoder` SIGSEGVs (the same hazard that makes the encode
//! side NVENC-only). `host::probe_decode` routes here when an NVIDIA GPU is
//! present and does NOT fall back to VAAPI.
//!
//! A real round trip against the live driver: construct `NvdecDecoder`,
//! submit the fixture IDR, drain it, and confirm a `Frame::Gpu` comes back
//! whose exported dma-buf fourcc the renderer's import path accepts (the
//! same L2 gate the VAAPI probe applies — a decoded surface the renderer
//! can't import is excluded at negotiation, not black-screened at runtime).

use tether_codec::nvdec::NvdecDecoder;
use tether_codec::vaapi_interop::{accepts_dmabuf_fourcc, dmabuf_fourcc_expected_label};
use tether_codec::{Decoder, Frame, GpuFrameSource};
use tether_protocol::control::VideoProfile;

use crate::profile_probe::{ProbeError, Result};
use crate::PipelineStage;

pub(crate) struct NvdecProbe;

impl NvdecProbe {
    pub(crate) fn probe_decode(profile: VideoProfile, fixture: &[u8]) -> Result<()> {
        tether_codec::av_log::with_probe_suppression(|| probe_decode_inner(profile, fixture))
    }
}

fn probe_decode_inner(profile: VideoProfile, fixture: &[u8]) -> Result<()> {
    let mut dec = NvdecDecoder::new(profile.codec)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Construct, e))?;
    dec.submit(fixture)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
    // A lone IDR sits in the reorder DPB until EOF; signal it so the frame
    // surfaces (mirrors the VAAPI decode probe).
    dec.signal_eof()
        .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
    for _ in 0..4 {
        match dec
            .next_frame()
            .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?
        {
            Some(Frame::Gpu(gpu)) => {
                // L2: the exported surface's dma-buf fourcc must be one the
                // renderer's import path accepts, or every frame of a
                // negotiated session would drop after decode.
                let GpuFrameSource::DmaBuf(dmabuf) = gpu.source;
                let observed = dmabuf.fourcc;
                if accepts_dmabuf_fourcc(profile.chroma, profile.bit_depth, observed) {
                    return Ok(());
                }
                return Err(ProbeError::new(
                    PipelineStage::Decode,
                    format!(
                        "NVDEC decoded surface fourcc 0x{observed:08x} is not render-accepted \
                         for {:?} {}-bit (expected {}); excluding this profile",
                        profile.chroma,
                        profile.bit_depth,
                        dmabuf_fourcc_expected_label(profile.chroma, profile.bit_depth),
                    ),
                ));
            }
            Some(Frame::Cpu(_)) => {
                return Err(ProbeError::new(
                    PipelineStage::Decode,
                    "NVDEC decoder produced a software (Frame::Cpu) frame — \
                     zero-copy GPU decode unavailable for this profile",
                ))
            }
            None => continue,
        }
    }
    Err(ProbeError::new(
        PipelineStage::Decode,
        "NVDEC decoder produced no frames after submit + eof + 4 polls",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile_probe::fixture_for;

    /// The bundled HEVC 8-bit 4:2:0 probe fixture must decode through the live
    /// NVDEC probe. This is a regression guard for the fixture coded size:
    /// NVDEC's HEVC decoder has a 144×144 minimum, so a fixture below that
    /// (the original 128×128 set) fails `send_packet` with a bare `-1` and the
    /// host silently drops HEVC to the H.264 floor. H.264's minimum is far
    /// smaller, so its fixture is fine at any of these sizes — HEVC is the one
    /// that breaks. `#[ignore]` — needs an NVIDIA GPU + an NVDEC FFmpeg.
    #[test]
    #[ignore = "requires NVIDIA GPU + NVDEC FFmpeg + Vulkan dma-buf"]
    fn hevc_main_fixture_decodes_via_nvdec_probe() {
        for profile in [VideoProfile::H264_8BIT_420, VideoProfile::HEVC_8BIT_420] {
            let fixture = fixture_for(profile).expect("bundled probe fixture");
            NvdecProbe::probe_decode(profile, fixture)
                .unwrap_or_else(|e| panic!("{profile:?} should decode via NVDEC, got {e:?}"));
        }
    }

    /// AV1 decode is advertised on NVDEC (unlike AV1 *encode*, which is
    /// Ada-only — NVDEC AV1 works on Ampere). The host advertises it, so per the
    /// "probe-pass is half-credit" rule it gets a real round-trip too: both the
    /// 8-bit (NV12) and 10-bit (P010) AV1 4:2:0 fixtures must decode through the
    /// live NVDEC probe. `#[ignore]` — needs an NVIDIA GPU + an NVDEC FFmpeg.
    #[test]
    #[ignore = "requires NVIDIA GPU + NVDEC FFmpeg + Vulkan dma-buf"]
    fn av1_fixtures_decode_via_nvdec_probe() {
        for profile in [VideoProfile::AV1_8BIT_420, VideoProfile::AV1_10BIT_420] {
            let fixture = fixture_for(profile).expect("bundled probe fixture");
            NvdecProbe::probe_decode(profile, fixture)
                .unwrap_or_else(|e| panic!("{profile:?} should decode via NVDEC, got {e:?}"));
        }
    }
}
