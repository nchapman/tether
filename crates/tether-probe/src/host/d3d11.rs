//! D3D11 (Windows) implementation of the [`ProfileProbe`] contract.
//!
//! Encode probe: build a `D3D11Encoder` at 128×128 (null device ptr —
//! lets FFmpeg create its own), encode grey BGRA frames until a keyframe
//! packet appears, validate the extradata is parseable Annex-B, then
//! submit that packet to a `D3D11Decoder` and verify a frame comes back.
//!
//! Decode probe: build `D3D11Decoder`, submit the fixture IDR, verify
//! `next_frame()` produces a frame. Unlike Linux, we don't reject
//! `Frame::Cpu` — the Windows decode path currently uses staging
//! textures for plane export which returns CPU frames.

use tether_codec::bitstream_sps::parse_sps_chroma_bit_depth;
use tether_codec::d3d11::{D3D11Decoder, D3D11Encoder};
use tether_codec::{Decoder, Encoder};
use tether_protocol::control::VideoProfile;

use crate::profile_probe::{ProbeError, ProfileProbe, Result};
use crate::PipelineStage;

pub(crate) struct D3D11Probe;

const PROBE_DIM: u32 = 128;
const PROBE_FPS: u32 = 30;
const PROBE_BITRATE_KBPS: u32 = 1_000;
const MAX_WARMUP_FRAMES: i64 = 30;

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

    // Encode grey frames until we get a keyframe packet.
    // AMF/MF encoders have an async pipeline — may need several inputs.
    let bgra = vec![0x80u8; (PROBE_DIM * PROBE_DIM * 4) as usize];
    let mut keyframe_packet = None;
    for pts in 0..MAX_WARMUP_FRAMES {
        let packets = enc
            .encode_bgra(&bgra, pts, pts == 0)
            .map_err(|e| ProbeError::from_codec(PipelineStage::Submit, e))?;
        for pkt in packets {
            if pkt.keyframe {
                keyframe_packet = Some(pkt);
                break;
            }
        }
        if keyframe_packet.is_some() {
            break;
        }
    }
    let kf = keyframe_packet.ok_or_else(|| {
        ProbeError::new(
            PipelineStage::Submit,
            "encoder produced no keyframe after 30 frames",
        )
    })?;

    // Validate extradata is valid Annex-B (SPS parser succeeds).
    if parse_sps_chroma_bit_depth(&kf.data, profile.codec).is_none() {
        return Err(ProbeError::new(
            PipelineStage::Submit,
            "keyframe bitstream has no parseable SPS — extradata may still \
             be in hvcC format (conversion failed?)",
        ));
    }

    // Round-trip: submit to decoder, verify a frame comes back.
    let mut dec =
        D3D11Decoder::new(profile.codec).map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
    dec.submit(&kf.data)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;

    // Decoder may need more input to flush its pipeline.
    for pts in MAX_WARMUP_FRAMES..(MAX_WARMUP_FRAMES + 30) {
        match dec
            .next_frame()
            .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?
        {
            Some(_) => return Ok(()),
            None => {
                let extra_pkts = enc
                    .encode_bgra(&bgra, pts, false)
                    .map_err(|e| ProbeError::from_codec(PipelineStage::Submit, e))?;
                for pkt in &extra_pkts {
                    dec.submit(&pkt.data)
                        .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
                }
            }
        }
    }
    Err(ProbeError::new(
        PipelineStage::Decode,
        "decoder produced no frames after encode round-trip",
    ))
}

fn probe_decode_inner(profile: VideoProfile, fixture: &[u8]) -> Result<()> {
    let mut dec =
        D3D11Decoder::new(profile.codec).map_err(|e| ProbeError::from_codec(PipelineStage::Construct, e))?;
    dec.submit(fixture)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
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
        "decoder produced no frames after submit + 4 polls",
    ))
}
