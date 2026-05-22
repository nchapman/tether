//! Minimal HEVC / H.264 SPS parser for probe-time cross-checks.
//!
//! The encode probe in [`crate::videotoolbox::probe`] decodes its own
//! output and checks the resulting IOSurface fourcc against the
//! requested `(chroma, bit_depth)`. The fourcc reflects what the
//! *decoder* produced, not what the *encoder* declared in the
//! bitstream. To catch the unlikely case where a future decoder maps
//! a downsampled bitstream into the "wrong" fourcc family — or vice
//! versa — we add a second signal: parse the encoder's SPS NAL and
//! confirm `chroma_format_idc` + `bit_depth_luma_minus8` match the
//! profile we asked for. If the two signals agree, the probe is
//! confident; if they disagree, something silently transformed the
//! data between encoder declaration and decoder rendering, and we
//! report the profile as unsupported.
//!
//! This is the *minimum* SPS parse needed for the cross-check —
//! `chroma_format_idc` (u8 ≤ 3) and `bit_depth_luma_minus8` (u8 ≤ 6
//! per spec). We skip every other field. The parser keeps a single
//! invariant: never return a guess. If anything is short, malformed,
//! or out of range we return `None` and the caller treats the SPS
//! signal as absent (the IOSurface fourcc check is still authoritative
//! on its own).

use tether_protocol::control::{ChromaSubsampling, CodecKind};

/// Result of parsing an SPS NAL. Both fields directly reflect the
/// bitstream — no decoder-side rendering choices muddied in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpsChromaAndBitDepth {
    pub(crate) chroma_format_idc: u8,
    pub(crate) bit_depth_luma: u8,
}

impl SpsChromaAndBitDepth {
    /// Map the parsed SPS to the protocol-level `(ChromaSubsampling,
    /// bit_depth)` so callers can compare against a `VideoProfile`
    /// directly. Returns `None` for combinations Tether doesn't model
    /// today (e.g. 4:2:2, 12-bit) — the probe layer treats those as
    /// "second signal couldn't agree" and falls back to fourcc-only.
    pub fn to_profile_chroma_bit_depth(self) -> Option<(ChromaSubsampling, u8)> {
        let chroma = match self.chroma_format_idc {
            1 => ChromaSubsampling::Yuv420,
            3 => ChromaSubsampling::Yuv444,
            _ => return None,
        };
        if self.bit_depth_luma != 8 && self.bit_depth_luma != 10 {
            return None;
        }
        Some((chroma, self.bit_depth_luma))
    }
}

/// Top-level entry: scan an Annex-B bitstream for the first SPS NAL
/// of the given codec and parse out `(chroma_format_idc, bit_depth)`.
/// Returns `None` if no SPS is found, the SPS is truncated, or any
/// field is outside the values this parser models.
pub fn parse_sps_chroma_bit_depth(
    annexb: &[u8],
    codec: CodecKind,
) -> Option<SpsChromaAndBitDepth> {
    // AV1 isn't an Annex-B / NAL-framed codec — OBU framing is wholly
    // different. The probe layer would have to call a different
    // parser; today we return `None` so AV1 (when wired) falls back to
    // fourcc-only signal until the OBU parser lands.
    let (sps_nut, nal_header_size): (u8, usize) = match codec {
        CodecKind::H264 => (7, 1),
        CodecKind::Hevc => (33, 2),
        CodecKind::Av1 => return None,
    };
    for nal_payload in iter_nal_payloads(annexb) {
        if nal_payload.is_empty() {
            continue;
        }
        let nal_unit_type = match codec {
            CodecKind::H264 => nal_payload[0] & 0x1F,
            CodecKind::Hevc => (nal_payload[0] >> 1) & 0x3F,
            CodecKind::Av1 => return None,
        };
        if nal_unit_type != sps_nut {
            continue;
        }
        // Strip emulation prevention bytes (`0x00 0x00 0x03` → `0x00
        // 0x00`) from the RBSP before bit-reading; otherwise the
        // exp-Golomb codes downstream can mis-align.
        let rbsp = strip_emulation_prevention(&nal_payload[nal_header_size..]);
        return match codec {
            CodecKind::H264 => parse_h264_sps(&rbsp),
            CodecKind::Hevc => parse_hevc_sps(&rbsp),
            CodecKind::Av1 => None,
        };
    }
    None
}

/// Iterate over Annex-B NAL units, yielding each NAL's payload (after
/// the start code). Both 3-byte (`00 00 01`) and 4-byte (`00 00 00 01`)
/// start codes are recognised; the yielded slices do not include the
/// start code or trailing zeros from the next NAL.
fn iter_nal_payloads(annexb: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut starts: Vec<usize> = Vec::new();
    let mut i = 0;
    while i + 3 <= annexb.len() {
        if annexb[i] == 0 && annexb[i + 1] == 0 {
            if annexb[i + 2] == 1 {
                starts.push(i + 3);
                i += 3;
                continue;
            }
            if i + 4 <= annexb.len() && annexb[i + 2] == 0 && annexb[i + 3] == 1 {
                starts.push(i + 4);
                i += 4;
                continue;
            }
        }
        i += 1;
    }
    let len = annexb.len();
    starts
        .clone()
        .into_iter()
        .enumerate()
        .map(move |(idx, start)| {
            let end = starts.get(idx + 1).copied().unwrap_or(len);
            // Strip the trailing zero bytes that precede the next start
            // code (typical encoder output has a single 0x00 byte before
            // the next 00 00 01).
            let mut e = end;
            while e > start && annexb[e - 1] == 0 {
                e -= 1;
            }
            &annexb[start..e]
        })
        .collect::<Vec<_>>()
        .into_iter()
}

fn strip_emulation_prevention(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len());
    let mut i = 0;
    while i < rbsp.len() {
        if i + 2 < rbsp.len() && rbsp[i] == 0 && rbsp[i + 1] == 0 && rbsp[i + 2] == 0x03 {
            out.push(0);
            out.push(0);
            i += 3;
        } else {
            out.push(rbsp[i]);
            i += 1;
        }
    }
    out
}

/// Minimal MSB-first bit reader. Exp-Golomb (`ue`) decoding included.
struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    fn read_bits(&mut self, n: usize) -> Option<u64> {
        if n > 64 {
            return None;
        }
        let mut acc: u64 = 0;
        for _ in 0..n {
            let byte = *self.bytes.get(self.bit_pos / 8)?;
            let bit = (byte >> (7 - (self.bit_pos % 8))) & 1;
            acc = (acc << 1) | u64::from(bit);
            self.bit_pos += 1;
        }
        Some(acc)
    }

    /// Unsigned exp-Golomb. Returns `None` on bitstream exhaustion or
    /// codes with 32+ leading zeros — the resulting value would not
    /// fit in our `u32` storage without truncation, and a code that
    /// long is well outside anything sane in an SPS (`chroma_format_idc`
    /// is bounded to 3, `sps_seq_parameter_set_id` to 15).
    fn read_ue(&mut self) -> Option<u32> {
        let mut leading_zeros = 0;
        loop {
            let bit = self.read_bits(1)?;
            if bit == 1 {
                break;
            }
            leading_zeros += 1;
            // Strictly less than 32 — `leading_zeros == 32` would
            // make the `(1<<32) - 1 + suffix` calculation exceed u32
            // for any non-zero suffix and truncate silently on the
            // `as u32` cast. Reject at the boundary, not past it.
            if leading_zeros > 31 {
                return None;
            }
        }
        if leading_zeros == 0 {
            return Some(0);
        }
        let suffix = self.read_bits(leading_zeros)?;
        // Cannot overflow u32: leading_zeros ≤ 31, so the leading
        // term is at most `(1<<31) - 1 = 0x7FFF_FFFF`, and `suffix`
        // is bounded by `(1<<31) - 1` since we read exactly 31 bits.
        // Sum fits in u32.
        let value = (1u64 << leading_zeros) - 1 + suffix;
        u32::try_from(value).ok()
    }
}

/// Parse just enough of an HEVC SPS RBSP to extract
/// `chroma_format_idc` and `bit_depth_luma_minus8`.
fn parse_hevc_sps(rbsp: &[u8]) -> Option<SpsChromaAndBitDepth> {
    let mut r = BitReader::new(rbsp);
    let _sps_vps_id = r.read_bits(4)?;
    let sps_max_sub_layers_minus1 = r.read_bits(3)? as usize;
    let _sps_temporal_id_nesting_flag = r.read_bits(1)?;
    // profile_tier_level(1, sps_max_sub_layers_minus1): the "1" means
    // include the general PTL. We skip the bits without interpreting.
    skip_hevc_ptl(&mut r, sps_max_sub_layers_minus1)?;
    let _sps_seq_parameter_set_id = r.read_ue()?;
    let chroma_format_idc = r.read_ue()? as u8;
    if chroma_format_idc == 3 {
        let _separate_colour_plane_flag = r.read_bits(1)?;
    }
    let _pic_width_in_luma_samples = r.read_ue()?;
    let _pic_height_in_luma_samples = r.read_ue()?;
    let conformance_window_flag = r.read_bits(1)?;
    if conformance_window_flag == 1 {
        let _ = r.read_ue()?;
        let _ = r.read_ue()?;
        let _ = r.read_ue()?;
        let _ = r.read_ue()?;
    }
    let bit_depth_luma_minus8 = r.read_ue()? as u8;
    // We don't enforce luma == chroma; chroma bit depth is read by spec
    // immediately after, but for our cross-check the luma value is the
    // one that pairs with `profile.bit_depth`.
    Some(SpsChromaAndBitDepth {
        chroma_format_idc,
        bit_depth_luma: bit_depth_luma_minus8.checked_add(8)?,
    })
}

/// Skip the `profile_tier_level(1, max_sub_layers_minus1)` structure
/// in an HEVC SPS. We don't need its contents — just bit-accurate
/// advancement past it.
fn skip_hevc_ptl(r: &mut BitReader<'_>, max_sub_layers_minus1: usize) -> Option<()> {
    // General profile / tier / level: 2 + 1 + 5 + 32 + 4 + 43 + 1 + 8
    // = 96 bits. The penultimate 1-bit slot is `general_inbld_flag` or
    // `general_reserved_zero_bit` depending on profile_idc; either way
    // it sits between the 43 bits of constraint flags and the 8-bit
    // level_idc, and we don't interpret it.
    r.read_bits(2)?; // general_profile_space
    r.read_bits(1)?; // general_tier_flag
    r.read_bits(5)?; // general_profile_idc
    r.read_bits(32)?; // general_profile_compatibility_flag[32]
    r.read_bits(4)?; // progressive/interlaced/non-packed/frame-only
    r.read_bits(43)?; // constraint flags + reserved (total 43, all branches)
    r.read_bits(1)?; // inbld_flag / reserved_zero_bit
    r.read_bits(8)?; // general_level_idc
    // Sub-layer presence flags: 2 bits per sublayer.
    let mut profile_present = Vec::with_capacity(max_sub_layers_minus1);
    let mut level_present = Vec::with_capacity(max_sub_layers_minus1);
    for _ in 0..max_sub_layers_minus1 {
        profile_present.push(r.read_bits(1)?);
        level_present.push(r.read_bits(1)?);
    }
    if max_sub_layers_minus1 > 0 {
        // Reserved alignment: 2 bits per missing sublayer up to 8 slots.
        for _ in max_sub_layers_minus1..8 {
            r.read_bits(2)?;
        }
    }
    for i in 0..max_sub_layers_minus1 {
        if profile_present[i] == 1 {
            // Same shape as the general PTL above, minus the level byte.
            r.read_bits(2)?;
            r.read_bits(1)?;
            r.read_bits(5)?;
            r.read_bits(32)?;
            r.read_bits(4)?;
            r.read_bits(43)?;
            r.read_bits(1)?;
        }
        if level_present[i] == 1 {
            r.read_bits(8)?;
        }
    }
    Some(())
}

/// Parse just enough of an H.264 SPS RBSP to extract `chroma_format_idc`
/// and `bit_depth_luma_minus8`. For "baseline-family" profiles that
/// don't carry chroma_format / bit_depth fields, returns the spec
/// default (4:2:0, 8-bit).
fn parse_h264_sps(rbsp: &[u8]) -> Option<SpsChromaAndBitDepth> {
    let mut r = BitReader::new(rbsp);
    let profile_idc = r.read_bits(8)? as u8;
    let _constraint_set_flags = r.read_bits(8)?;
    let _level_idc = r.read_bits(8)?;
    let _seq_parameter_set_id = r.read_ue()?;
    // The High-family profiles carry chroma_format_idc + bit_depth
    // explicitly. Anything else is implicitly 4:2:0 8-bit per the
    // spec (Annex E).
    const HIGH_FAMILY: [u8; 13] = [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135];
    if HIGH_FAMILY.contains(&profile_idc) {
        let chroma_format_idc = r.read_ue()? as u8;
        if chroma_format_idc == 3 {
            let _separate_colour_plane_flag = r.read_bits(1)?;
        }
        let bit_depth_luma_minus8 = r.read_ue()? as u8;
        Some(SpsChromaAndBitDepth {
            chroma_format_idc,
            bit_depth_luma: bit_depth_luma_minus8.checked_add(8)?,
        })
    } else {
        Some(SpsChromaAndBitDepth {
            chroma_format_idc: 1, // 4:2:0
            bit_depth_luma: 8,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hevc_yuv420_8bit_fixture_parses_as_420_8() {
        let bytes = include_bytes!("../../tether-probe/fixtures/probe/hevc_yuv420_8bit.idr");
        let sps = parse_sps_chroma_bit_depth(bytes, CodecKind::Hevc)
            .expect("hevc 4:2:0 8-bit fixture must contain a parseable SPS");
        assert_eq!(sps.chroma_format_idc, 1, "chroma_format_idc 1 = 4:2:0");
        assert_eq!(sps.bit_depth_luma, 8);
    }

    #[test]
    fn hevc_yuv444_8bit_fixture_parses_as_444_8() {
        let bytes = include_bytes!("../../tether-probe/fixtures/probe/hevc_yuv444_8bit.idr");
        let sps = parse_sps_chroma_bit_depth(bytes, CodecKind::Hevc)
            .expect("hevc 4:4:4 8-bit fixture must contain a parseable SPS");
        assert_eq!(sps.chroma_format_idc, 3, "chroma_format_idc 3 = 4:4:4");
        assert_eq!(sps.bit_depth_luma, 8);
    }

    #[test]
    fn hevc_yuv420_10bit_fixture_parses_as_420_10() {
        let bytes = include_bytes!("../../tether-probe/fixtures/probe/hevc_yuv420_10bit.idr");
        let sps = parse_sps_chroma_bit_depth(bytes, CodecKind::Hevc)
            .expect("hevc 4:2:0 10-bit fixture must contain a parseable SPS");
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.bit_depth_luma, 10);
    }

    #[test]
    fn hevc_yuv444_10bit_fixture_parses_as_444_10() {
        let bytes = include_bytes!("../../tether-probe/fixtures/probe/hevc_yuv444_10bit.idr");
        let sps = parse_sps_chroma_bit_depth(bytes, CodecKind::Hevc)
            .expect("hevc 4:4:4 10-bit fixture must contain a parseable SPS");
        assert_eq!(sps.chroma_format_idc, 3);
        assert_eq!(sps.bit_depth_luma, 10);
    }

    #[test]
    fn h264_yuv420_8bit_fixture_parses_as_420_8() {
        let bytes = include_bytes!("../../tether-probe/fixtures/probe/h264_yuv420_8bit.idr");
        let sps = parse_sps_chroma_bit_depth(bytes, CodecKind::H264)
            .expect("h264 4:2:0 8-bit fixture must contain a parseable SPS");
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.bit_depth_luma, 8);
    }

    #[test]
    fn read_ue_rejects_32_leading_zeros_rather_than_truncating() {
        // 32 leading zeros + a `1` is the boundary case the silent-
        // truncation fix targets. We craft the buffer manually:
        // four bytes of all-zero, then a byte starting with `1`.
        let buf = vec![0x00, 0x00, 0x00, 0x00, 0x80];
        let mut r = BitReader::new(&buf);
        assert_eq!(
            r.read_ue(),
            None,
            "32-leading-zeros code must reject, not truncate to a corrupted u32"
        );
    }

    #[test]
    fn empty_bitstream_returns_none() {
        assert!(parse_sps_chroma_bit_depth(&[], CodecKind::Hevc).is_none());
        assert!(parse_sps_chroma_bit_depth(&[], CodecKind::H264).is_none());
    }

    #[test]
    fn bitstream_without_sps_returns_none() {
        // Annex-B with a single NAL whose type is *not* SPS. For HEVC
        // we pick AUD (35); for H.264 we pick filler data (12).
        let no_sps_hevc = [0x00, 0x00, 0x00, 0x01, (35 << 1), 0x00];
        assert!(parse_sps_chroma_bit_depth(&no_sps_hevc, CodecKind::Hevc).is_none());
        let no_sps_h264 = [0x00, 0x00, 0x00, 0x01, 12];
        assert!(parse_sps_chroma_bit_depth(&no_sps_h264, CodecKind::H264).is_none());
    }

    #[test]
    fn emulation_prevention_bytes_are_stripped() {
        // A 3-byte input `00 00 03` is the emulation prevention
        // pattern; stripping must produce a 2-byte `00 00` output.
        assert_eq!(strip_emulation_prevention(&[0x00, 0x00, 0x03]), vec![0x00, 0x00]);
        // The 0x03 after a non-`00 00` prefix must be preserved
        // verbatim — it's a real payload byte there.
        assert_eq!(
            strip_emulation_prevention(&[0x01, 0x00, 0x03]),
            vec![0x01, 0x00, 0x03]
        );
    }

    #[test]
    fn to_profile_chroma_bit_depth_round_trips_modeled_combos() {
        for (idc, bd, expected_chroma) in [
            (1, 8, ChromaSubsampling::Yuv420),
            (1, 10, ChromaSubsampling::Yuv420),
            (3, 8, ChromaSubsampling::Yuv444),
            (3, 10, ChromaSubsampling::Yuv444),
        ] {
            let s = SpsChromaAndBitDepth {
                chroma_format_idc: idc,
                bit_depth_luma: bd,
            };
            assert_eq!(s.to_profile_chroma_bit_depth(), Some((expected_chroma, bd)));
        }
    }

    #[test]
    fn to_profile_chroma_bit_depth_rejects_unmodeled_combos() {
        // 4:2:2 is not in Tether's profile matrix.
        assert!(SpsChromaAndBitDepth {
            chroma_format_idc: 2,
            bit_depth_luma: 8,
        }
        .to_profile_chroma_bit_depth()
        .is_none());
        // 12-bit is out of range for hot-path remote-desktop.
        assert!(SpsChromaAndBitDepth {
            chroma_format_idc: 1,
            bit_depth_luma: 12,
        }
        .to_profile_chroma_bit_depth()
        .is_none());
    }

    #[test]
    fn bit_reader_ue_decodes_canonical_values() {
        // Spec exp-Golomb codes (k_zeros leading 0s, then `1`, then
        // k_zeros bits of suffix):
        //   0 → `1`        (1 bit)
        //   1 → `010`      (3 bits)
        //   2 → `011`      (3 bits)
        //   3 → `00100`    (5 bits)
        //   4 → `00101`    (5 bits)
        //   5 → `00110`    (5 bits)
        for (bits_msb_first, expected) in [
            (vec![0b1000_0000], 0u32),
            (vec![0b0100_0000], 1),
            (vec![0b0110_0000], 2),
            (vec![0b0010_0000], 3),
            (vec![0b0010_1000], 4),
            (vec![0b0011_0000], 5),
        ] {
            let mut r = BitReader::new(&bits_msb_first);
            assert_eq!(
                r.read_ue(),
                Some(expected),
                "bits {:08b} should decode to {}",
                bits_msb_first[0],
                expected
            );
        }
    }
}
