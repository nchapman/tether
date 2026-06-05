//! Bits of encoder logic shared between every hardware backend
//! (VAAPI on Linux, VideoToolbox on macOS, D3D11VA on Windows).
//!
//! Right now the only resident is `drain_encoder`, which yields all
//! packets currently buffered in an `AVCodecContext` and prepends the
//! encoder's `extradata` (Annex-B SPS/PPS/VPS) to every keyframe so
//! every IDR is self-decodable. This is the streaming-friendly contract
//! Tether's raw wire format requires: a client that joins mid-session,
//! rebuilds its decoder (resume, resolution change), or loses the
//! session's first IDR has no recovery path otherwise.
//!
//! Cost: one allocation per keyframe of `extradata.len()` bytes
//! (~25 bytes H.264, ~50 bytes HEVC). P-frames pass through untouched.

// hvcc/avcc box construction writes NALU and array lengths into the
// container's fixed-width size fields (u16 length prefixes, u8 counts).
// These narrowing casts are intentional and bounded by realistic
// parameter-set sizes.
#![allow(clippy::cast_possible_truncation)]

use std::slice;

use bytes::{Bytes, BytesMut};
use rsmpeg::avcodec::AVCodecContext;
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use tether_protocol::control::CodecKind;

use crate::{CodecError, EncodedPacket, Result};

#[allow(clippy::cast_sign_loss)]
pub(crate) fn drain_encoder(
    encoder: &mut AVCodecContext,
    extradata: &[u8],
) -> Result<Vec<EncodedPacket>> {
    let mut out = Vec::new();
    loop {
        let packet = match encoder.receive_packet() {
            Ok(p) => p,
            Err(RsmpegError::EncoderDrainError | RsmpegError::EncoderFlushedError) => break,
            Err(e) => return Err(CodecError::Ffmpeg(e)),
        };
        let size = packet.size as usize;
        // SAFETY: packet.data points to packet.size valid bytes owned
        // by the AVPacket; we copy them before drop.
        let raw = unsafe { slice::from_raw_parts(packet.data, size) };
        let keyframe = (packet.flags & ffi::AV_PKT_FLAG_KEY as i32) != 0;
        let data: Bytes = if keyframe && !extradata.is_empty() {
            let mut buf = BytesMut::with_capacity(extradata.len() + raw.len());
            buf.extend_from_slice(extradata);
            buf.extend_from_slice(raw);
            buf.freeze()
        } else {
            Bytes::copy_from_slice(raw)
        };
        let pts_out = if packet.pts == ffi::AV_NOPTS_VALUE {
            None
        } else {
            Some(packet.pts)
        };
        out.push(EncodedPacket {
            data,
            pts: pts_out,
            keyframe,
        });
    }
    Ok(out)
}

/// Snapshot `AVCodecContext::extradata` into an owned `Vec<u8>`. Call
/// this once immediately after `encoder.open()` has succeeded with
/// `AV_CODEC_FLAG_GLOBAL_HEADER` set.
///
/// Most hardware encoders (VAAPI, VideoToolbox, the vendor D3D11
/// backends QSV/AMF/NVENC) populate `extradata` at `open()` and emit
/// keyframe slices *without* in-band parameter sets; `drain_encoder`
/// prepends this snapshot to every keyframe so each IDR is
/// self-decodable. For those, an empty snapshot is fatal — without
/// SPS/PPS any client that loses the first IDR or rebuilds its decoder
/// mid-session is permanently stuck — so `inband_parameter_sets = false`
/// fails loudly at construction.
///
/// FFmpeg's Media Foundation encoders (`*_mf`) are the exception: they
/// never populate `extradata` but emit VPS/SPS/PPS (HEVC) or SPS/PPS
/// (H.264) in-band on *every* keyframe, so each IDR is already
/// self-decodable on its own. Callers pass `inband_parameter_sets = true`
/// for those; an empty snapshot is then expected and `drain_encoder`
/// correctly skips the prepend (prepending would duplicate the in-band
/// sets). Verified by `d3d11_mf_keyframes_carry_inband_parameter_sets`.
///
/// SAFETY: libavcodec populates `extradata` inside `open()` when
/// `AV_CODEC_FLAG_GLOBAL_HEADER` is set, and does not mutate it
/// mid-stream for the fixed-resolution encoders we ship (the hardware
/// encoders rebuild from scratch on resolution change). Copying into an
/// owned buffer immediately prevents any subsequent encoder operation
/// from racing with our read.
#[allow(clippy::cast_sign_loss)]
pub(crate) fn snapshot_extradata(
    encoder: &AVCodecContext,
    codec_name: &str,
    codec_kind: CodecKind,
    inband_parameter_sets: bool,
) -> Result<Vec<u8>> {
    let extradata = unsafe {
        let raw = encoder.extradata;
        let size = encoder.extradata_size;
        if raw.is_null() || size <= 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(raw, size as usize).to_vec()
        }
    };
    if extradata.is_empty() {
        if inband_parameter_sets {
            // Expected for Media Foundation: parameter sets ride in-band on
            // every keyframe, so there is nothing to prepend.
            tracing::debug!(
                codec = codec_name,
                "encoder leaves extradata empty; relying on in-band keyframe \
                 parameter sets for self-decodable IDRs"
            );
            return Ok(Vec::new());
        }
        return Err(CodecError::NoHardwareCodec(format!(
            "{codec_name}: encoder.extradata was empty after open() despite \
             AV_CODEC_FLAG_GLOBAL_HEADER. Keyframes would not carry SPS/PPS, \
             so any client that loses the first IDR or rebuilds its decoder \
             mid-session would be stuck. Verify the FFmpeg build honours \
             AV_CODEC_FLAG_GLOBAL_HEADER for this codec."
        )));
    }
    Ok(ensure_annexb_extradata(extradata, codec_kind))
}

/// Normalize codec extradata to Annex-B format. Some encoders (notably
/// D3D11VA/AMF on Windows) emit extradata in container format — hvcC for
/// HEVC, avcC for H.264 (both ISO 14496-15, length-prefixed NALUs) —
/// rather than Annex-B. This function detects container format and
/// converts to Annex-B so downstream code (SPS parser, decoder) gets a
/// consistent format.
///
/// AV1 extradata uses OBU framing (not handled here) and is passed
/// through unchanged. Already-Annex-B data is also passed through.
pub(crate) fn ensure_annexb_extradata(extradata: Vec<u8>, codec_kind: CodecKind) -> Vec<u8> {
    if codec_kind == CodecKind::Av1 {
        return extradata;
    }
    let annexb = if is_annexb(&extradata) {
        extradata
    } else {
        let converted = match codec_kind {
            CodecKind::Hevc => hvcc_to_annexb(&extradata),
            CodecKind::H264 => avcc_to_annexb(&extradata),
            CodecKind::Av1 => None,
        };
        match converted {
            Some(v) => v,
            None => {
                tracing::warn!(
                    codec = ?codec_kind,
                    len = extradata.len(),
                    "extradata is neither Annex-B nor recognisable container format; \
                     passing through unchanged — decoder may reject it"
                );
                return extradata;
            }
        }
    };
    // HEVC decoders require VPS before SPS before PPS. Some encoders
    // (AMF) emit them in SPS→PPS→VPS order. Reorder if needed.
    if codec_kind == CodecKind::Hevc {
        reorder_hevc_parameter_sets(annexb)
    } else {
        annexb
    }
}

/// Reorder HEVC Annex-B NALUs so VPS (32) comes before SPS (33) before
/// PPS (34). Some encoders (AMF) emit SPS→PPS→VPS; decoders require
/// VPS first. Non-parameter-set NALUs keep their relative order.
fn reorder_hevc_parameter_sets(data: Vec<u8>) -> Vec<u8> {
    let nalus = split_annexb_nalus(&data);
    if nalus.len() < 2 {
        return data;
    }
    // Check if reordering is needed by finding the first VPS and SPS.
    let first_vps = nalus.iter().position(|n| hevc_nalu_type(n) == 32);
    let first_sps = nalus.iter().position(|n| hevc_nalu_type(n) == 33);
    let needs_reorder = match (first_vps, first_sps) {
        (Some(v), Some(s)) => v > s,
        _ => false,
    };
    if !needs_reorder {
        return data;
    }
    let mut sorted = nalus;
    sorted.sort_by_key(|n| match hevc_nalu_type(n) {
        32 => 0, // VPS first
        33 => 1, // SPS second
        34 => 2, // PPS third
        _ => 3,  // everything else after
    });
    let mut out = Vec::with_capacity(data.len());
    for nalu in &sorted {
        out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        out.extend_from_slice(nalu);
    }
    out
}

/// Split Annex-B byte stream into individual NALUs (without start codes).
fn split_annexb_nalus(data: &[u8]) -> Vec<&[u8]> {
    let mut nalus = Vec::new();
    let mut i = 0;
    while i < data.len() {
        // Find start code (3 or 4 bytes).
        let sc_len = if i + 3 < data.len()
            && data[i] == 0
            && data[i + 1] == 0
            && data[i + 2] == 0
            && data[i + 3] == 1
        {
            4
        } else if i + 2 < data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            3
        } else {
            i += 1;
            continue;
        };
        let nalu_start = i + sc_len;
        // Find next start code or end of data.
        let mut end = nalu_start;
        while end < data.len() {
            if end + 3 <= data.len()
                && data[end] == 0
                && data[end + 1] == 0
                && (data[end + 2] == 1
                    || (end + 3 < data.len() && data[end + 2] == 0 && data[end + 3] == 1))
            {
                break;
            }
            end += 1;
        }
        if end > nalu_start {
            nalus.push(&data[nalu_start..end]);
        }
        i = end;
    }
    nalus
}

/// Extract the HEVC NAL unit type from the first byte of a NALU body.
fn hevc_nalu_type(nalu: &[u8]) -> u8 {
    if nalu.is_empty() {
        return 0;
    }
    (nalu[0] >> 1) & 0x3F
}

/// Check if data starts with an Annex-B start code.
fn is_annexb(data: &[u8]) -> bool {
    (data.len() >= 4 && data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1)
        || (data.len() >= 3 && data[0] == 0 && data[1] == 0 && data[2] == 1)
}

/// Parse an HEVCDecoderConfigurationRecord (hvcC) and convert to Annex-B.
/// Returns None if the data doesn't look like valid hvcC.
///
/// AMF's hvcC may order arrays as SPS→PPS→VPS, but the decoder needs
/// VPS before SPS. We sort arrays by NAL type (VPS=32 < SPS=33 < PPS=34)
/// so the output is always VPS→SPS→PPS regardless of the hvcC order.
fn hvcc_to_annexb(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 23 {
        return None;
    }
    if data[0] != 1 {
        return None;
    }
    let num_arrays = data[22] as usize;
    let mut nalus: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut pos = 23;
    for _ in 0..num_arrays {
        if pos + 3 > data.len() {
            return None;
        }
        let nal_type = data[pos] & 0x3F;
        pos += 1;
        let num_nalus_in_array = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        for _ in 0..num_nalus_in_array {
            if pos + 2 > data.len() {
                return None;
            }
            let nalu_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;
            if pos + nalu_len > data.len() {
                return None;
            }
            if nalu_len > 0 {
                nalus.push((nal_type, data[pos..pos + nalu_len].to_vec()));
            }
            pos += nalu_len;
        }
    }
    if nalus.is_empty() {
        return None;
    }
    // Sort by NAL type so VPS (32) comes before SPS (33) before PPS (34).
    nalus.sort_by_key(|(t, _)| *t);
    let mut out = Vec::with_capacity(data.len());
    for (_, nalu) in &nalus {
        out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        out.extend_from_slice(nalu);
    }
    Some(out)
}

/// Parse an AVCDecoderConfigurationRecord (avcC) and convert to Annex-B.
/// Returns None if the data doesn't look like valid avcC.
/// ISO 14496-15 §5.3.3.1.
fn avcc_to_annexb(data: &[u8]) -> Option<Vec<u8>> {
    // Minimum avcC: 6 bytes header + at least one SPS entry.
    if data.len() < 7 {
        return None;
    }
    // configurationVersion must be 1.
    if data[0] != 1 {
        return None;
    }
    // Byte 5 lower 5 bits = numOfSequenceParameterSets.
    let num_sps = (data[5] & 0x1F) as usize;
    let mut out = Vec::with_capacity(data.len());
    let mut pos = 6;
    // Read SPS entries.
    for _ in 0..num_sps {
        if pos + 2 > data.len() {
            return None;
        }
        let nalu_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if pos + nalu_len > data.len() {
            return None;
        }
        if nalu_len > 0 {
            out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            out.extend_from_slice(&data[pos..pos + nalu_len]);
        }
        pos += nalu_len;
    }
    // numOfPictureParameterSets — truncated before PPS count means
    // incomplete record (decoder needs both SPS and PPS).
    if pos >= data.len() {
        return None;
    }
    let num_pps = data[pos] as usize;
    pos += 1;
    for _ in 0..num_pps {
        if pos + 2 > data.len() {
            return None;
        }
        let nalu_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if pos + nalu_len > data.len() {
            return None;
        }
        if nalu_len > 0 {
            out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            out.extend_from_slice(&data[pos..pos + nalu_len]);
        }
        pos += nalu_len;
    }
    if out.is_empty() {
        return None;
    }
    Some(out)
}

/// Static debug label for `CodecError::ScalerInit`, keyed off the
/// destination `AVPixelFormat`. Shared between every hardware backend
/// so the table stays in one place — VAAPI's `vaapi_sw_format` and
/// VideoToolbox's `vt_sw_format` both feed their outputs here. Falls
/// back to a generic label for a future `*_sw_format` addition that
/// hasn't reached this table yet (the fallback keeps `ScalerInit` a
/// `&'static str` instead of a heap-allocating per-call format).
///
/// Linux/macOS only: the Windows D3D11 encoder converts via the
/// `ID3D11VideoProcessor` and uses a literal `ScalerInit` label, so it
/// never feeds this table.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) fn pix_fmt_scaler_label(sw_format: i32) -> &'static str {
    match sw_format {
        x if x == ffi::AV_PIX_FMT_NV12 => "BGRA -> NV12",
        x if x == ffi::AV_PIX_FMT_P010LE => "BGRA -> P010",
        x if x == ffi::AV_PIX_FMT_NV24 => "BGRA -> NV24",
        x if x == ffi::AV_PIX_FMT_P410LE => "BGRA -> P410",
        x if x == ffi::AV_PIX_FMT_VUYX => "BGRA -> VUYX",
        x if x == ffi::AV_PIX_FMT_XV30LE => "BGRA -> XV30",
        _ => "BGRA -> sw_format",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn pix_fmt_scaler_label_covers_every_sw_format_in_use() {
        for (fmt, expected) in [
            (ffi::AV_PIX_FMT_NV12, "BGRA -> NV12"),
            (ffi::AV_PIX_FMT_P010LE, "BGRA -> P010"),
            (ffi::AV_PIX_FMT_NV24, "BGRA -> NV24"),
            (ffi::AV_PIX_FMT_P410LE, "BGRA -> P410"),
            (ffi::AV_PIX_FMT_VUYX, "BGRA -> VUYX"),
            (ffi::AV_PIX_FMT_XV30LE, "BGRA -> XV30"),
        ] {
            assert_eq!(pix_fmt_scaler_label(fmt), expected);
        }
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn pix_fmt_scaler_label_falls_back_for_unknown_format() {
        assert_eq!(pix_fmt_scaler_label(-1), "BGRA -> sw_format");
    }

    #[test]
    fn is_annexb_detects_4byte_start_code() {
        assert!(is_annexb(&[0x00, 0x00, 0x00, 0x01, 0x67]));
    }

    #[test]
    fn is_annexb_detects_3byte_start_code() {
        assert!(is_annexb(&[0x00, 0x00, 0x01, 0x67]));
    }

    #[test]
    fn is_annexb_rejects_hvcc() {
        // hvcC starts with configurationVersion=1, not a start code
        let mut hvcc = vec![0u8; 30];
        hvcc[0] = 1; // configurationVersion
        assert!(!is_annexb(&hvcc));
    }

    #[test]
    fn is_annexb_rejects_empty() {
        assert!(!is_annexb(&[]));
        assert!(!is_annexb(&[0x00, 0x00]));
    }

    #[test]
    fn hvcc_to_annexb_converts_known_record() {
        // Construct a synthetic hvcC with one array containing two NALUs
        // (simulating VPS + SPS in one array, which some encoders do).
        let vps_nalu = [0x40, 0x01, 0x0C, 0x01]; // NAL type 32 (VPS)
        let sps_nalu = [0x42, 0x01, 0x01, 0x02, 0x20]; // NAL type 33 (SPS)

        let mut hvcc = vec![0u8; 23];
        hvcc[0] = 1; // configurationVersion
        hvcc[22] = 1; // numOfArrays = 1

        // Array 0: type VPS (32), 2 NALUs
        hvcc.push(0x20); // completeness=0, nal_unit_type=32
        hvcc.extend_from_slice(&(2u16).to_be_bytes()); // numNalus = 2
                                                       // NALU 1 (VPS)
        hvcc.extend_from_slice(&(vps_nalu.len() as u16).to_be_bytes());
        hvcc.extend_from_slice(&vps_nalu);
        // NALU 2 (SPS)
        hvcc.extend_from_slice(&(sps_nalu.len() as u16).to_be_bytes());
        hvcc.extend_from_slice(&sps_nalu);

        let annexb = hvcc_to_annexb(&hvcc).expect("should parse valid hvcC");

        // Expected: start_code + VPS + start_code + SPS
        let mut expected = Vec::new();
        expected.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        expected.extend_from_slice(&vps_nalu);
        expected.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        expected.extend_from_slice(&sps_nalu);

        assert_eq!(annexb, expected);
    }

    #[test]
    fn hvcc_to_annexb_handles_multiple_arrays() {
        // Three arrays: VPS, SPS, PPS (common hvcC layout)
        let vps = [0x40, 0x01, 0x0C];
        let sps = [0x42, 0x01, 0x01, 0x02];
        let pps = [0x44, 0x01, 0xC0];

        let mut hvcc = vec![0u8; 23];
        hvcc[0] = 1;
        hvcc[22] = 3; // numOfArrays = 3

        // Array 0: VPS
        hvcc.push(0x20);
        hvcc.extend_from_slice(&1u16.to_be_bytes());
        hvcc.extend_from_slice(&(vps.len() as u16).to_be_bytes());
        hvcc.extend_from_slice(&vps);
        // Array 1: SPS
        hvcc.push(0x21);
        hvcc.extend_from_slice(&1u16.to_be_bytes());
        hvcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        hvcc.extend_from_slice(&sps);
        // Array 2: PPS
        hvcc.push(0x22);
        hvcc.extend_from_slice(&1u16.to_be_bytes());
        hvcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        hvcc.extend_from_slice(&pps);

        let annexb = hvcc_to_annexb(&hvcc).expect("should parse valid hvcC");

        let mut expected = Vec::new();
        for nalu in [&vps[..], &sps[..], &pps[..]] {
            expected.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            expected.extend_from_slice(nalu);
        }
        assert_eq!(annexb, expected);
    }

    #[test]
    fn hvcc_to_annexb_rejects_truncated_record() {
        assert_eq!(hvcc_to_annexb(&[1; 10]), None); // too short for header
                                                    // Header says 1 array but no array data follows
        let mut hvcc = vec![0u8; 23];
        hvcc[0] = 1;
        hvcc[22] = 1;
        assert_eq!(hvcc_to_annexb(&hvcc), None);
    }

    #[test]
    fn hvcc_to_annexb_rejects_wrong_version() {
        let mut data = vec![0u8; 30];
        data[0] = 2; // wrong version
        assert_eq!(hvcc_to_annexb(&data), None);
    }

    #[test]
    fn ensure_annexb_passthrough_for_already_annexb_h264() {
        let data = vec![0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e];
        let result = ensure_annexb_extradata(data.clone(), CodecKind::H264);
        assert_eq!(result, data);
    }

    #[test]
    fn avcc_to_annexb_converts_known_record() {
        // Synthetic avcC: version=1, profile=66, compat=0, level=30,
        // lengthSizeMinusOne=3, 1 SPS, 1 PPS
        let sps = [0x67, 0x42, 0x00, 0x1e, 0xab]; // NAL type 7 (SPS)
        let pps = [0x68, 0xce, 0x38, 0x80]; // NAL type 8 (PPS)

        let mut avcc = vec![
            1,    // configurationVersion
            66,   // AVCProfileIndication (Baseline)
            0,    // profile_compatibility
            30,   // AVCLevelIndication
            0xFF, // lengthSizeMinusOne=3 (lower 2 bits) + reserved
            0xE1, // numSPS=1 (lower 5 bits) + reserved
        ];
        // SPS
        avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&sps);
        // numPPS
        avcc.push(1);
        // PPS
        avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&pps);

        let annexb = avcc_to_annexb(&avcc).expect("should parse valid avcC");

        let mut expected = Vec::new();
        expected.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        expected.extend_from_slice(&sps);
        expected.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        expected.extend_from_slice(&pps);
        assert_eq!(annexb, expected);
    }

    #[test]
    fn avcc_to_annexb_rejects_truncated() {
        assert_eq!(avcc_to_annexb(&[1, 66, 0, 30, 0xFF]), None); // too short
    }

    #[test]
    fn ensure_annexb_converts_avcc_h264() {
        // Build a minimal avcC with one SPS
        let sps = [0x67, 0x42, 0x00, 0x1e];
        let mut avcc = vec![1, 66, 0, 30, 0xFF, 0xE1];
        avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&sps);
        avcc.push(0); // 0 PPS

        let result = ensure_annexb_extradata(avcc, CodecKind::H264);
        assert!(is_annexb(&result), "result should be Annex-B");
        assert_eq!(&result[4..], &sps);
    }

    #[test]
    fn ensure_annexb_passthrough_for_already_annexb_hevc() {
        let data = vec![0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0C];
        let result = ensure_annexb_extradata(data.clone(), CodecKind::Hevc);
        assert_eq!(result, data);
    }

    #[test]
    fn ensure_annexb_converts_hvcc_hevc() {
        // Build a minimal hvcC with one VPS NALU
        let vps = [0x40, 0x01, 0x0C, 0x01, 0xFF];
        let mut hvcc = vec![0u8; 23];
        hvcc[0] = 1;
        hvcc[22] = 1;
        hvcc.push(0x20);
        hvcc.extend_from_slice(&1u16.to_be_bytes());
        hvcc.extend_from_slice(&(vps.len() as u16).to_be_bytes());
        hvcc.extend_from_slice(&vps);

        let result = ensure_annexb_extradata(hvcc, CodecKind::Hevc);
        assert!(is_annexb(&result), "result should be Annex-B");
        // Should contain start code + VPS bytes
        assert_eq!(&result[0..4], &[0x00, 0x00, 0x00, 0x01]);
        assert_eq!(&result[4..], &vps);
    }

    #[test]
    fn converted_hvcc_is_parseable_by_sps_parser() {
        // Use the real HEVC fixture to build a synthetic hvcC, then verify
        // the converted output still parses. We'll grab the NALUs from the
        // fixture (which is already Annex-B) and wrap them in hvcC format.
        let fixture = include_bytes!("../../tether-probe/fixtures/probe/hevc_yuv420_8bit.idr");

        // Extract NALUs from the Annex-B fixture
        let mut nalus: Vec<&[u8]> = Vec::new();
        let mut i = 0;
        let mut starts: Vec<usize> = Vec::new();
        while i + 3 <= fixture.len() {
            if fixture[i] == 0 && fixture[i + 1] == 0 {
                if fixture[i + 2] == 1 {
                    starts.push(i + 3);
                    i += 3;
                    continue;
                }
                if i + 4 <= fixture.len() && fixture[i + 2] == 0 && fixture[i + 3] == 1 {
                    starts.push(i + 4);
                    i += 4;
                    continue;
                }
            }
            i += 1;
        }
        for (idx, &start) in starts.iter().enumerate() {
            let end = starts
                .get(idx + 1)
                .map(|&s| {
                    // back up past trailing zeros before next start code
                    let mut e = s;
                    while e > start && fixture[e - 1] == 0 {
                        e -= 1;
                    }
                    e
                })
                .unwrap_or(fixture.len());
            nalus.push(&fixture[start..end]);
        }

        // Build hvcC from these NALUs (one array per NALU for simplicity)
        let mut hvcc = vec![0u8; 23];
        hvcc[0] = 1; // configurationVersion
        hvcc[22] = nalus.len() as u8; // numOfArrays
        for nalu in &nalus {
            let nal_type = if !nalu.is_empty() {
                (nalu[0] >> 1) & 0x3F
            } else {
                0
            };
            hvcc.push(nal_type);
            hvcc.extend_from_slice(&1u16.to_be_bytes());
            hvcc.extend_from_slice(&(nalu.len() as u16).to_be_bytes());
            hvcc.extend_from_slice(nalu);
        }

        let converted = ensure_annexb_extradata(hvcc, CodecKind::Hevc);
        assert!(is_annexb(&converted));

        // The SPS parser should find the SPS and extract chroma+bit_depth
        let parsed = crate::bitstream_sps::parse_sps_chroma_bit_depth(&converted, CodecKind::Hevc);
        assert!(
            parsed.is_some(),
            "converted hvcC should be parseable as Annex-B HEVC"
        );
        let sps = parsed.unwrap();
        assert_eq!(sps.chroma_format_idc, 1); // 4:2:0
        assert_eq!(sps.bit_depth_luma, 8);
    }
}
