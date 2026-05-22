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

use tether_codec::bitstream_sps::parse_sps_chroma_bit_depth;
use tether_codec::videotoolbox::{
    expected_iosurface_fourccs, VideoToolboxDecoder, VideoToolboxEncoder,
};
use tether_codec::{CodecError, Decoder, Encoder, Frame, GpuFrameSource, Result};
use tether_protocol::control::VideoProfile;

use crate::profile_probe::ProfileProbe;

pub(crate) struct VideoToolboxProbe;

const PROBE_DIM: u32 = 128;
const PROBE_FPS: u32 = 30;
const PROBE_BITRATE_KBPS: u32 = 1_000;

impl ProfileProbe for VideoToolboxProbe {
    fn probe_encode(profile: VideoProfile) -> Result<()> {
        let mut enc = VideoToolboxEncoder::new(
            profile,
            PROBE_DIM,
            PROBE_DIM,
            PROBE_FPS,
            PROBE_BITRATE_KBPS,
        )?;
        // Use a non-uniform BGRA pattern so VT has actual chroma
        // content to encode — a flat grey buffer would produce a
        // bitstream where 4:4:4 vs 4:2:0 are indistinguishable at the
        // decoder, defeating the round-trip check.
        let bytes = probe_bgra_with_chroma_detail();
        let mut packets = enc.encode_bgra(&bytes, 0, true)?;
        // VT typically buffers the first frame; flush to make sure we
        // have at least one packet before round-tripping.
        if packets.is_empty() {
            packets.extend(enc.flush()?);
        }
        if packets.is_empty() {
            return Err(CodecError::UnsupportedInputFormat);
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
                return Err(CodecError::UnsupportedInputFormat);
            }
        }

        // Round-trip through a fresh VT decoder and check the output
        // IOSurface fourcc actually matches what we asked the encoder
        // for. This is the "did the encoder silently downsample?" gate.
        let mut dec = VideoToolboxDecoder::new(profile.codec)?;
        for packet in &packets {
            dec.submit(&packet.data)?;
        }
        dec.signal_eof()?;
        let observed = loop {
            match dec.next_frame()? {
                Some(Frame::Gpu(gpu)) => {
                    let GpuFrameSource::IOSurface(io) = gpu.source;
                    break io.pixel_format;
                }
                Some(Frame::Cpu(_)) => return Err(CodecError::UnsupportedInputFormat),
                None => return Err(CodecError::UnsupportedInputFormat),
            }
        };
        let expected = expected_iosurface_fourccs(profile);
        if !expected.iter().any(|f| *f == observed) {
            tracing::debug!(
                ?profile,
                observed = format_args!("0x{:08x}", observed),
                expected_count = expected.len(),
                "VT encode produced a bitstream whose decoded IOSurface fourcc \
                 doesn't match the requested chroma/bit-depth — likely silent \
                 downsample"
            );
            return Err(CodecError::UnsupportedInputFormat);
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
        let mut dec = VideoToolboxDecoder::new(profile.codec)?;
        dec.submit(fixture)?;
        // VT's wrapper buffers the first packet pending either a
        // second packet or EOF before emitting; the probe only has
        // one IDR, so signal EOF to force the drain.
        dec.signal_eof()?;
        loop {
            match dec.next_frame()? {
                Some(Frame::Gpu(_)) => return Ok(()),
                Some(Frame::Cpu(_)) => return Err(CodecError::UnsupportedInputFormat),
                None => return Err(CodecError::UnsupportedInputFormat),
            }
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
