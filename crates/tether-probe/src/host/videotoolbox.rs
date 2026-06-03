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
//! Step 2 of the migration: this is a verbatim move from
//! `tether-codec::videotoolbox::probe`. Step 3 folds in the SCK
//! capture capability check (today done at the host layer as a
//! separate cache).

use std::sync::OnceLock;

use tether_capture::macos::{
    probe_capture_pixel_formats, sck_pixel_format_for_profile, SckCaptureCapability,
};
use tether_codec::bitstream_sps::parse_sps_chroma_bit_depth;
use tether_codec::videotoolbox::{
    expected_iosurface_fourccs, VideoToolboxDecoder, VideoToolboxEncoder,
};
use tether_codec::{Decoder, Encoder, Frame, GpuFrameSource};
use tether_protocol::control::VideoProfile;

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
/// TODO(probe-migration step 5): the tether-host `SCK_CAPS_CACHE` and
/// `warm_sck_capture_capability_cache` are still alive during the
/// migration window. After step 5 cuts the host over to call
/// `tether_probe::host_supported_profiles()` directly, those become
/// dead code and get deleted; this OnceLock is then the only cache
/// of the SCK probe result.
fn sck_capability() -> &'static SckCaptureCapability {
    static CACHED: OnceLock<SckCaptureCapability> = OnceLock::new();
    CACHED.get_or_init(|| match pollster::block_on(probe_capture_pixel_formats()) {
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

impl ProfileProbe for VideoToolboxProbe {
    fn probe_encode(profile: VideoProfile) -> Result<()> {
        // Capture-stage gate: if SCK can't deliver the matching pixel
        // format on this Mac, the host couldn't deliver real frames to
        // the encoder regardless of whether VT itself accepts the
        // profile. Folds in the SCK_CAPS_CACHE check that used to live
        // in tether-host as a separate filter layer.
        if !sck_pixel_format_for_profile(profile).is_deliverable(sck_capability()) {
            return Err(ProbeError::new(
                PipelineStage::Capture,
                format!(
                    "ScreenCaptureKit cannot deliver the pixel format for \
                     {:?} {:?} {}-bit on this Mac",
                    profile.codec, profile.chroma, profile.bit_depth
                ),
            ));
        }

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
        // `profile` is otherwise unused on this path — the codec is
        // encoded in the fixture's NAL header and that's the only
        // thing the decoder construction needs. Keep the full
        // `profile` in the signature for symmetry with `probe_encode`
        // and so log lines have the requested profile to anchor to.
        let _ = profile;
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
            Some(Frame::Gpu(_)) => Ok(()),
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
