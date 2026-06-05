//! VideoToolbox implementation of the [`ProfileProbe`] contract.
//!
//! Encode probe: real encode → decode round-trip. The encoder is
//! constructed for `profile` (which picks an FFmpeg pix_fmt via the
//! encoder's internal `vt_sw_format`), one BGRA frame with
//! high-frequency chroma detail is fed through, and the resulting
//! Annex-B packet(s) are submitted to a fresh VideoToolbox *decoder*.
//! The probe succeeds only if (a) the decoded IOSurface fourcc lands
//! in the set of fourccs we'd accept for the *requested* `(chroma,
//! bit_depth)` and (b) the emitted SPS independently agrees with the
//! requested chroma/bit-depth. This is what catches VT's known
//! silent-downsample behaviour.
//!
//! Decode probe: submit the checked-in fixture IDR and require a
//! `Frame::Gpu` back. Failure modes this catches are silent SW
//! fallback (`hevc_videotoolbox` accepting the codec context but
//! routing through a native SW decoder for unsupported profiles) and
//! genuine hardware-decoder absence.
//!
//! The probe owns the macOS host capability answer end to end, including
//! SCK BGRA availability, Metal bridge construction, VideoToolbox encode,
//! VideoToolbox decode, and renderer IOSurface acceptance.

use std::sync::OnceLock;

use tether_capture::macos::{probe_capture_pixel_formats_blocking, SckCaptureCapability};
use tether_codec::bitstream_sps::parse_sps_chroma_bit_depth;
use tether_codec::macos_interop::{
    accepts_iosurface_fourcc, iosurface_fourcc_expected_label, NV12_VIDEO_RANGE_FOURCC,
    NV24_VIDEO_RANGE_FOURCC, X420_FOURCC, X444_FOURCC,
};
use tether_codec::videotoolbox::{
    expected_iosurface_fourccs, VideoToolboxDecoder, VideoToolboxEncoder,
};
use tether_codec::{Decoder, Encoder, Frame, GpuFrameSource};
use tether_gpuconvert::nv12_iosurface::BridgeDeviceCapabilities;
use tether_protocol::control::{ChromaSubsampling, VideoProfile};

use crate::profile_probe::{ProbeError, ProfileProbe, Result};
use crate::PipelineStage;

pub(crate) struct VideoToolboxProbe;

const PROBE_DIM: u32 = 128;
const PROBE_FPS: u32 = 30;
const PROBE_BITRATE_KBPS: u32 = 1_000;

/// Process-lifetime cache of the SCK probe result. Folds in what was
/// previously a separate cache (`SCK_CAPS_CACHE` +
/// `warm_sck_capture_capability_cache`) in `tether-host`. Same
/// single-shot per-process semantics: the answer doesn't change at
/// runtime, so we pay the probe cost once.
///
/// Conservative fallback: yuv420_video_range=true on probe error, so
/// a Mac with a broken SCK setup still negotiates H.264/HEVC 4:2:0
/// 8-bit (the universal floor) rather than refusing all profiles.
///
fn sck_capability() -> &'static SckCaptureCapability {
    static CACHED: OnceLock<SckCaptureCapability> = OnceLock::new();
    // Drive the blocking probe core directly — no Tokio runtime required.
    // (The async `probe_capture_pixel_formats` wrapper uses `spawn_blocking`,
    // which panics without an ambient reactor; this sync probe has none.)
    CACHED.get_or_init(|| match probe_capture_pixel_formats_blocking() {
        Ok(c) => {
            tracing::info!(?c, "SCK capture capability probed");
            c
        }
        Err(e) => {
            tracing::error!(error = %e, "SCK capability probe failed; \
                    falling back to yuv420 video-range only");
            SckCaptureCapability {
                yuv420_video_range: true,
                ..Default::default()
            }
        }
    })
}

fn bridge_device() -> Result<(wgpu::Device, wgpu::Queue, BridgeDeviceCapabilities)> {
    static CACHED: OnceLock<
        std::result::Result<(wgpu::Device, wgpu::Queue, BridgeDeviceCapabilities), String>,
    > = OnceLock::new();
    match CACHED.get_or_init(|| {
        pollster::block_on(tether_gpuconvert::nv12_iosurface::build_bridge_device())
            .map_err(|e| e.to_string())
    }) {
        Ok((device, queue, caps)) => Ok((device.clone(), queue.clone(), *caps)),
        Err(e) => Err(ProbeError::new(
            PipelineStage::Capture,
            format!("macOS BGRA IOSurface bridge device init failed: {e}"),
        )),
    }
}

impl ProfileProbe for VideoToolboxProbe {
    fn probe_encode(profile: VideoProfile) -> Result<()> {
        gate_capture_and_bridge(profile)?;

        let mut enc =
            VideoToolboxEncoder::new(profile, PROBE_DIM, PROBE_DIM, PROBE_FPS, PROBE_BITRATE_KBPS)
                .map_err(|e| ProbeError::from_codec(PipelineStage::Construct, e))?;
        // Use a non-uniform BGRA pattern so VT has actual chroma
        // content to encode — a flat grey buffer would produce a
        // bitstream where 4:4:4 vs 4:2:0 are indistinguishable at the
        // decoder, defeating the round-trip check.
        let bytes = probe_bgra_with_chroma_detail();
        let mut packets = enc
            .encode_bgra(&bytes, 0, true)
            .map_err(|e| ProbeError::from_codec(PipelineStage::Submit, e))?;
        // VT typically buffers the first frame; flush to make sure we
        // have at least one packet before round-tripping.
        if packets.is_empty() {
            packets.extend(
                enc.flush()
                    .map_err(|e| ProbeError::from_codec(PipelineStage::Submit, e))?,
            );
        }
        if packets.is_empty() {
            return Err(ProbeError::new(
                PipelineStage::Submit,
                "encoder produced no packets after encode + flush",
            ));
        }

        // Second signal: parse the SPS NAL the encoder emitted and
        // confirm `chroma_format_idc` + `bit_depth_luma_minus8` agree
        // with the profile we requested. The IOSurface fourcc check
        // below reflects what the *decoder* produced; the SPS reflects
        // what the *encoder* declared. Two independent signals catch
        // an edge the round-trip alone could miss.
        let keyframe_bitstream = packets
            .iter()
            .find(|p| p.keyframe)
            .map_or(&*packets[0].data, |p| &p.data);
        if let Some(parsed) = parse_sps_chroma_bit_depth(keyframe_bitstream, profile.codec)
            .and_then(|s| s.to_profile_chroma_bit_depth())
        {
            let expected = (profile.chroma, profile.bit_depth);
            if parsed != expected {
                tracing::debug!(
                    ?profile,
                    sps_chroma = ?parsed.0,
                    sps_bit_depth = parsed.1,
                    "VT encoder emitted an SPS declaring a different chroma / \
                     bit-depth than the profile requested — silent transform"
                );
                return Err(ProbeError::new(
                    PipelineStage::Submit,
                    format!(
                        "VT encoder silently transformed the bitstream: \
                         requested {:?} {}-bit, SPS declares {:?} {}-bit",
                        expected.0, expected.1, parsed.0, parsed.1
                    ),
                ));
            }
        }

        // Round-trip through a fresh VT decoder and check the output
        // IOSurface fourcc actually matches what we asked the encoder
        // for. This is the "did the encoder silently downsample?" gate.
        let mut dec = VideoToolboxDecoder::new(profile.codec)
            .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
        for packet in &packets {
            dec.submit(&packet.data)
                .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
        }
        dec.signal_eof()
            .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
        // One IDR in, EOF signalled — the very next frame is the result.
        let observed = match dec
            .next_frame()
            .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?
        {
            Some(Frame::Gpu(gpu)) => {
                let GpuFrameSource::IOSurface(io) = gpu.source;
                io.pixel_format
            }
            Some(Frame::Cpu(_)) => {
                return Err(ProbeError::new(
                    PipelineStage::Decode,
                    "VT decoder fell back to software (Frame::Cpu) — \
                     silent SW fallback",
                ))
            }
            None => {
                return Err(ProbeError::new(
                    PipelineStage::Decode,
                    "VT decoder produced no frames after EOF signal",
                ))
            }
        };
        let expected = expected_iosurface_fourccs(profile);
        if !expected.contains(&observed) {
            tracing::debug!(
                ?profile,
                observed = format_args!("0x{:08x}", observed),
                expected_count = expected.len(),
                "VT encode produced a bitstream whose decoded IOSurface fourcc \
                 doesn't match the requested chroma/bit-depth — likely silent \
                 downsample"
            );
            return Err(ProbeError::new(
                PipelineStage::Submit,
                format!(
                    "VT encode produced an unexpected IOSurface fourcc \
                     (observed 0x{observed:08x}, expected one of {expected:?}) \
                     — likely silent downsample"
                ),
            ));
        }
        Ok(())
    }

    fn probe_decode(profile: VideoProfile, fixture: &[u8]) -> Result<()> {
        let mut dec = VideoToolboxDecoder::new(profile.codec)
            .map_err(|e| ProbeError::from_codec(PipelineStage::Construct, e))?;
        dec.submit(fixture)
            .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
        // VT's wrapper buffers the first packet pending either a
        // second packet or EOF before emitting; the probe only has
        // one IDR, so signal EOF to force the drain.
        dec.signal_eof()
            .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
        match dec
            .next_frame()
            .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?
        {
            Some(Frame::Gpu(gpu)) => {
                // L2: gate the decoded IOSurface fourcc through the SAME
                // accept set the renderer's IOSurface import uses, so a
                // profile that VT decodes to a fourcc the renderer rejects
                // is excluded at negotiation instead of black-screening at
                // first frame. This is the check that would have caught
                // the 'x444' bug (a Linux-host 4:4:4 10-bit stream decoding
                // to the video-range 'x444' while the accept set listed
                // only the full-range 'xf44').
                let GpuFrameSource::IOSurface(io) = gpu.source;
                let observed = io.pixel_format;
                if accepts_iosurface_fourcc(profile.chroma, profile.bit_depth, observed) {
                    Ok(())
                } else {
                    Err(ProbeError::new(
                        PipelineStage::Decode,
                        format!(
                            "VT decoded IOSurface fourcc 0x{observed:08x} is not render-accepted \
                             for {:?} {}-bit (expected {}) — the renderer's import path would \
                             drop every frame; excluding this profile from negotiation",
                            profile.chroma,
                            profile.bit_depth,
                            iosurface_fourcc_expected_label(profile.chroma, profile.bit_depth),
                        ),
                    ))
                }
            }
            Some(Frame::Cpu(_)) => Err(ProbeError::new(
                PipelineStage::Decode,
                "VT decoder fell back to software (Frame::Cpu)",
            )),
            None => Err(ProbeError::new(
                PipelineStage::Decode,
                "VT decoder produced no frames after EOF signal",
            )),
        }
    }
}

fn gate_capture_and_bridge(profile: VideoProfile) -> Result<()> {
    let caps = sck_capability();
    if !caps.bgra {
        return Err(ProbeError::new(
            PipelineStage::Capture,
            format!(
                "ScreenCaptureKit cannot deliver BGRA for the macOS \
                 host-side Metal bridge ({:?} {:?} {}-bit)",
                profile.codec, profile.chroma, profile.bit_depth
            ),
        ));
    }
    let dst_fourcc = match (profile.chroma, profile.bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => NV12_VIDEO_RANGE_FOURCC,
        (ChromaSubsampling::Yuv420, 10) => X420_FOURCC,
        (ChromaSubsampling::Yuv444, 8) => NV24_VIDEO_RANGE_FOURCC,
        (ChromaSubsampling::Yuv444, 10) => X444_FOURCC,
        _ => {
            return Err(ProbeError::new(
                PipelineStage::Capture,
                format!(
                    "no BGRA IOSurface bridge output fourcc for {:?} {}-bit",
                    profile.chroma, profile.bit_depth
                ),
            ));
        }
    };
    let (device, queue, _caps) = bridge_device()?;
    tether_gpuconvert::nv12_iosurface::BgraIOSurfaceBridge::new(
        device,
        queue,
        (PROBE_DIM * 2, PROBE_DIM * 2),
        (PROBE_DIM, PROBE_DIM),
        dst_fourcc,
    )
    .map_err(|e| {
        ProbeError::new(
            PipelineStage::Capture,
            format!("macOS BGRA IOSurface bridge construction failed: {e}"),
        )
    })?;
    Ok(())
}

/// Produce a `PROBE_DIM × PROBE_DIM` BGRA buffer with high-frequency
/// chroma content — a vertical red/green stripe pattern at 1-pixel
/// granularity. Any encoder that silently downsamples 4:4:4 → 4:2:0
/// loses the stripe (chroma resolution halved in X collapses
/// alternating-red-green columns into a uniform yellow-ish bar), so
/// the resulting bitstream's `chroma_format_idc` reflects the
/// downsample and the round-trip's decoded IOSurface fourcc lands in
/// the 4:2:0 family.
fn probe_bgra_with_chroma_detail() -> Vec<u8> {
    let mut data = Vec::with_capacity((PROBE_DIM * PROBE_DIM * 4) as usize);
    for _y in 0..PROBE_DIM {
        for x in 0..PROBE_DIM {
            if x % 2 == 0 {
                data.extend_from_slice(&[0, 0, 255, 255]); // red
            } else {
                data.extend_from_slice(&[0, 255, 0, 255]); // green
            }
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile_probe::fixture_for;

    /// L1 (pure logic, no hardware): for every negotiable profile, every
    /// IOSurface fourcc the VT decoder is declared to emit
    /// (`expected_iosurface_fourccs`) must be one the renderer's import
    /// path accepts (`accepts_iosurface_fourcc`). The macOS sibling of the
    /// Windows `decoder_output_is_subset_of_renderer_accept` and the Linux
    /// VAAPI version — this is the cheap CI guard that would have caught
    /// the missing `'x444'` accept entry without any hardware.
    #[test]
    fn decoder_output_is_subset_of_renderer_accept() {
        let mut covered = 0;
        for profile in crate::PROFILE_PREFERENCE {
            for &fourcc in expected_iosurface_fourccs(*profile) {
                assert!(
                    accepts_iosurface_fourcc(profile.chroma, profile.bit_depth, fourcc),
                    "VT decode emits IOSurface fourcc 0x{fourcc:08x} for {profile:?}, but the \
                     renderer's accepts_iosurface_fourcc rejects it — every frame of a negotiated \
                     session would be dropped at import (the 'x444' bug shape)."
                );
                covered += 1;
            }
        }
        // Vacuity guard: H.264/HEVC plus AV1 4:2:0 entries all declare
        // at least one decode fourcc, so coverage comfortably clears
        // the four surface families.
        assert!(
            covered >= 4,
            "expected ≥4 declared decode fourccs across PROFILE_PREFERENCE, got {covered}"
        );
    }

    /// Decode every committed probe fixture through VideoToolbox and
    /// assert the IOSurface fourcc VT actually emits is one the
    /// renderer would import (`accepts_iosurface_fourcc`).
    ///
    /// This is the closed-loop guard for the bug class that kept
    /// recurring: the renderer accept set was hand-maintained against
    /// what *macOS encode* produces, while a real session decodes
    /// streams from a Linux/Windows host. The fixtures here are
    /// x264/x265-encoded (i.e. **not** VideoToolbox), so they exercise
    /// exactly the cross-platform decode path. With the renderer
    /// limited to BT.709, the surface a video-range stream lands in is
    /// the video-range fourcc of its family — e.g. HEVC 4:4:4 10-bit
    /// decodes to `'x444'`, which the accept set rejected until this
    /// test pinned it.
    ///
    /// Hardware: needs an Apple-Silicon VT decoder, with M3+ for AV1.
    /// A profile whose fixture this machine genuinely can't decode is
    /// reported as SKIP rather than failing, but the 4:4:4 10-bit
    /// regression target is required to land — every Apple Silicon Mac
    /// decodes it.
    #[test]
    #[ignore = "requires macOS + VideoToolbox"]
    fn decoded_fixture_fourcc_is_renderer_accepted() {
        // The macOS-negotiated decode set that ships a fixture.
        let profiles = [
            VideoProfile::AV1_10BIT_420,
            VideoProfile::AV1_8BIT_420,
            VideoProfile::H264_8BIT_420,
            VideoProfile::HEVC_8BIT_420,
            VideoProfile::HEVC_10BIT_420,
            VideoProfile::HEVC_8BIT_444,
            VideoProfile::HEVC_10BIT_444,
        ];

        let mut decoded_444_10bit = false;
        for profile in profiles {
            let Some(fixture) = fixture_for(profile) else {
                panic!("{profile:?} has no committed decode fixture");
            };
            match decode_fixture_fourcc(profile, fixture) {
                Ok(fourcc) => {
                    assert!(
                        accepts_iosurface_fourcc(profile.chroma, profile.bit_depth, fourcc),
                        "{profile:?} decoded to IOSurface fourcc 0x{fourcc:08x}, which the \
                         renderer's accepts_iosurface_fourcc rejects — the import path would \
                         drop every frame with the surface intact (the 'x444' bug)"
                    );
                    eprintln!("decode-accept: {profile:?} OK (IOSurface 0x{fourcc:08x} accepted)");
                    if profile == VideoProfile::HEVC_10BIT_444 {
                        decoded_444_10bit = true;
                    }
                }
                Err(reason) => {
                    eprintln!("decode-accept: {profile:?} SKIP ({reason})");
                }
            }
        }

        assert!(
            decoded_444_10bit,
            "HEVC 4:4:4 10-bit fixture failed to decode — this is the regression target and \
             every Apple Silicon VT decoder handles it; a failure here means the decode path \
             itself broke, not just the accept set"
        );
    }

    /// Decode a single-IDR fixture and return the observed IOSurface
    /// fourcc, or a reason string if any step failed (treated as a
    /// hardware SKIP by the caller).
    fn decode_fixture_fourcc(
        profile: VideoProfile,
        fixture: &[u8],
    ) -> std::result::Result<u32, String> {
        let mut dec = VideoToolboxDecoder::new(profile.codec)
            .map_err(|e| format!("decoder construction: {e:?}"))?;
        dec.submit(fixture)
            .map_err(|e| format!("decoder submit: {e:?}"))?;
        dec.signal_eof()
            .map_err(|e| format!("decoder signal_eof: {e:?}"))?;
        match dec
            .next_frame()
            .map_err(|e| format!("decoder next_frame: {e:?}"))?
        {
            Some(Frame::Gpu(gpu)) => {
                let GpuFrameSource::IOSurface(io) = gpu.source;
                Ok(io.pixel_format)
            }
            Some(Frame::Cpu(_)) => Err("VT fell back to software (Frame::Cpu)".into()),
            None => Err("no frame produced after EOF".into()),
        }
    }
}
