//! VideoToolbox implementation of the [`ProfileProbe`] contract.
//!
//! Encode probe: real encode → decode round-trip. The encoder is
//! constructed for `profile` (which picks an FFmpeg pix_fmt via
//! `encoder::vt_sw_format`), one BGRA frame is fed through, and the
//! resulting Annex-B packet(s) are submitted to a fresh VideoToolbox
//! *decoder*. The probe succeeds only if the decoded IOSurface fourcc
//! lands in the set of fourccs we'd accept for the *requested*
//! `(chroma, bit_depth)`. This is what catches VT's known
//! silent-downsample behaviour: the encoder wrapper may accept NV24
//! / P410 input and emit a packet, but produce a 4:2:0-shaped
//! bitstream — the decoder then reports a `'420v'` IOSurface, not a
//! `'444v'` one, and the probe fails. Without the round-trip, the
//! probe would claim encode=true for a profile that ships
//! downsampled chroma to peers.
//!
//! Decode probe: submit the checked-in fixture IDR and require a
//! `Frame::Gpu` back. Failure modes this catches are silent SW
//! fallback (the `hevc_videotoolbox` wrapper accepts the codec
//! context but the underlying VT session routes through a native
//! SW decoder for unsupported profiles, emitting `Frame::Cpu`
//! which the decoder layer maps to `UnsupportedInputFormat`) and
//! genuine hardware-decoder absence on a given profile.

use tether_protocol::control::{ChromaSubsampling, VideoProfile};

use crate::bitstream_sps::parse_sps_chroma_bit_depth;
use crate::profile_probe::ProfileProbe;
use crate::{CodecError, Decoder, Encoder, Frame, GpuFrameSource, Result};

use super::{VideoToolboxDecoder, VideoToolboxEncoder};

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
        // have at least one packet before round-tripping. Test-only
        // `flush()` is the same hook the existing hardware tests use.
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
        // an edge the round-trip alone could miss — e.g. if a future
        // decoder rendered a downsampled bitstream into the "expected"
        // fourcc family, the fourcc check would silently pass.
        //
        // SPS parser doesn't model every profile (4:2:2, 12-bit, AV1
        // OBU framing); when it returns `None` we accept the absence
        // and fall back to fourcc-only. False negatives here (parser
        // can't read this SPS) should not gate a profile that the
        // round-trip otherwise confirms; only an *explicit*
        // disagreement is fatal.
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
        // for. This is the "did the encoder silently downsample?" gate
        // — VT has documented history of accepting input it then
        // transcodes internally without erroring (e.g. H.264 4:2:2 →
        // 4:2:0). Without this gate we'd advertise `encode=true` for
        // profiles whose emitted bitstreams don't carry the chroma
        // resolution we promised peers.
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
        let _ = profile; // codec is encoded in the fixture; profile is for logging only
        Self::decode_one_fixture(profile, fixture)
    }
}

impl VideoToolboxProbe {
    /// Helper extracted so the encode probe can reuse the decoder
    /// driver (it round-trips its own emitted packets) without
    /// duplicating the "submit + signal EOF + pull one Gpu frame"
    /// dance.
    fn decode_one_fixture(profile: VideoProfile, fixture: &[u8]) -> Result<()> {
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
/// the 4:2:0 family. A flat-grey buffer wouldn't trigger this — the
/// signal is in the chroma transitions.
fn probe_bgra_with_chroma_detail() -> Vec<u8> {
    let mut data = Vec::with_capacity((PROBE_DIM * PROBE_DIM * 4) as usize);
    for _y in 0..PROBE_DIM {
        for x in 0..PROBE_DIM {
            // Alternating saturated red / saturated green at full
            // chroma swing. BGRA byte order: [B, G, R, A].
            if x % 2 == 0 {
                data.extend_from_slice(&[0, 0, 255, 255]); // red
            } else {
                data.extend_from_slice(&[0, 255, 0, 255]); // green
            }
        }
    }
    data
}

/// The set of IOSurface fourccs a VT decoder may emit for `profile`'s
/// bitstream, when the encoder *actually* produced that profile's
/// chroma + bit-depth (vs. silently downsampled). Per-profile range
/// variants (`'420v'` vs `'420f'`, etc.) both count — range is a VUI
/// signal, not a chroma-resolution one, and we deliberately don't
/// gate on it.
///
/// Exposed at `pub` so cross-crate consistency tests can confirm the
/// renderer's IOSurface accept set (`tether-render::gpu::metal`) is a
/// superset of this — i.e. that the renderer can import everything
/// the probe expects the decoder to produce. A subset mismatch is
/// what bit us in commit `621badc` (renderer rejected `'x420'` even
/// though the probe correctly listed it).
#[must_use]
pub fn expected_iosurface_fourccs(profile: VideoProfile) -> &'static [u32] {
    // Apple's `kCVPixelFormatType_*` values, big-endian fourccs.
    const NV12_VIDEO: u32 = u32::from_be_bytes(*b"420v");
    const NV12_FULL: u32 = u32::from_be_bytes(*b"420f");
    const NV24_VIDEO: u32 = u32::from_be_bytes(*b"444v");
    const NV24_FULL: u32 = u32::from_be_bytes(*b"444f");
    // 10-bit 4:2:0 biplanar — `kCVPixelFormatType_420YpCbCr10BiPlanarFullRange`
    // (`'xf20'`) or video-range / P010 variants. VT decode lands here
    // for HEVC Main10 input.
    const P010: u32 = u32::from_be_bytes(*b"P010");
    const XF20: u32 = u32::from_be_bytes(*b"xf20");
    const X420: u32 = u32::from_be_bytes(*b"x420");
    // 10-bit 4:4:4 biplanar — `'xf44'` (full range) or `'P410'`.
    const XF44: u32 = u32::from_be_bytes(*b"xf44");
    const P410: u32 = u32::from_be_bytes(*b"P410");
    match (profile.chroma, profile.bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => &[NV12_VIDEO, NV12_FULL],
        (ChromaSubsampling::Yuv420, 10) => &[P010, XF20, X420],
        (ChromaSubsampling::Yuv444, 8) => &[NV24_VIDEO, NV24_FULL],
        (ChromaSubsampling::Yuv444, 10) => &[XF44, P410],
        // Anything else: empty set — round-trip always fails. We
        // shouldn't reach here for any profile currently in
        // PROFILE_PREFERENCE; if we do, that's a new combo that needs
        // a fourcc entry added here before it can probe true.
        _ => &[],
    }
}
