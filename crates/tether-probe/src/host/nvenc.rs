//! NVENC implementation of the host encode probe.
//!
//! Only the encode half lives here. Decode on an NVIDIA host goes through
//! `NvdecDecoder`, probed by [`super::nvdec::NvdecProbe`] (see
//! `host::probe_decode`) — NOT VAAPI, whose `nvidia-vaapi-driver` decoder
//! SIGSEGVs. `host::probe_encode` tries this first on an NVIDIA host and falls
//! back to VAAPI, mirroring the live `build_encoder` dispatch so the advertised
//! capability set matches what the session encoder will actually pick.
//!
//! The probe is a real round trip against the live driver, not a
//! construction-only check: `NvencEncoder::new` opening the codec is not
//! sufficient evidence the CUDA upload + encode actually runs. The 8-bit path
//! encodes one CPU-uploaded frame; the 10-bit (Main10) path drives a real
//! `Bgra2P010DmaBuf` → `submit_dmabuf` round trip through the production
//! zero-copy import, the same way the VAAPI probe proves its P010 path. NVENC
//! 4:4:4 is rejected at construction (no planar bridge yet), so it never
//! reaches the bit-depth branch.

use tether_codec::nvenc::NvencEncoder;
use tether_codec::{build_p010_dmabuf_frame, Encoder};
use tether_gpuconvert::Bgra2P010DmaBuf;
use tether_protocol::control::{ChromaSubsampling, VideoProfile};

use crate::profile_probe::{ProbeError, Result};
use crate::PipelineStage;

pub(crate) struct NvencProbe;

// NvencProbe deliberately does NOT implement the `ProfileProbe` trait (unlike
// VaapiProbe): it has only an encode half. Decode on an NVIDIA host is probed
// by `NvdecProbe` (wired through `host::probe_decode`), so a trait-required
// `probe_decode` here would be a never-called delegating stub — the kind of
// dead code the project avoids. `host::probe_encode` calls the inherent method
// directly.

/// Probe canvas. 128×128 matches the VAAPI probe and the fixture dims, and
/// satisfies HEVC's minimum-block constraint.
// 256, not the 128 the VAAPI probe uses: constructing a 128×128 HEVC Main10
// (P010) NVENC encoder SIGSEGVs *inside the NVENC runtime* — Main10 has a
// minimum encode dimension that 128px violates, and FFmpeg's wrapper faults
// rather than erroring. 256 is the smallest dimension verified to construct
// every advertised profile (the M3 P010 round-trip runs at 256). Real
// sessions are always far larger; this is a probe-only floor.
const PROBE_DIM: u32 = 256;
const PROBE_FPS: u32 = 30;
const PROBE_BITRATE_KBPS: u32 = 1_000;

impl NvencProbe {
    /// Probe whether this NVIDIA host can encode `profile` through NVENC.
    /// `Ok(())` means a real frame went through the encoder; `Err` carries
    /// the [`PipelineStage`] that rejected it, and the caller
    /// (`host::probe_encode`) falls back to VAAPI.
    pub(crate) fn probe_encode(profile: VideoProfile) -> Result<()> {
        tether_codec::av_log::with_probe_suppression(|| probe_encode_inner(profile))
    }
}

fn probe_encode_inner(profile: VideoProfile) -> Result<()> {
    // Construction catches "driver/codec can't do this profile at all":
    // codec not built (no `--enable-nvenc`), no NVIDIA device / `libcuda`,
    // AV1 on a pre-Ada card, or 4:4:4 (no planar bridge yet → mapped
    // UnsupportedInputFormat in `nvenc_sw_format`).
    let mut enc = NvencEncoder::new(profile, PROBE_DIM, PROBE_DIM, PROBE_FPS, PROBE_BITRATE_KBPS)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Construct, e))?;

    match (profile.chroma, profile.bit_depth) {
        // 8-bit: CPU BGRA upload (swscale → NV12 → host→CUDA transfer →
        // encode) exercises the full encode path, not just `open()`.
        (_, 8) => {
            let bytes = vec![0x80u8; (PROBE_DIM * PROBE_DIM * 4) as usize];
            enc.encode_bgra(&bytes, 0, true)
                .map_err(|e| ProbeError::from_codec(PipelineStage::Submit, e))?;
        }
        // 10-bit 4:2:0 (Main10): real submit_dmabuf round trip via the
        // production `Bgra2P010DmaBuf` bridge — the encoder's only 10-bit input
        // path (encode_bgra has no 10-bit branch). Proves NVENC accepts the
        // P010 dma-buf through the EGL→CUDA import, not just that it opens.
        (ChromaSubsampling::Yuv420, 10) => probe_p010_submit(&mut enc)?,
        // 4:4:4 never reaches here (rejected at NvencEncoder::new); any other
        // bit depth is unmapped.
        (_, bd) => {
            return Err(ProbeError::new(
                PipelineStage::Construct,
                format!("unsupported bit depth {bd} for NVENC encode probe"),
            ));
        }
    }
    Ok(())
}

/// Build a P010 dma-buf through the production `Bgra2P010DmaBuf` bridge and
/// feed it to NVENC's `submit_dmabuf`. Mirrors the VAAPI probe's
/// `probe_p010_submit`. Catches the three failure modes a construction-only
/// check would miss: bridge construction (no R16/Rg16 LINEAR storage), bridge
/// convert, and the EGL→CUDA import + NVENC P010 submit itself.
fn probe_p010_submit(enc: &mut NvencEncoder) -> Result<()> {
    let bridge = pollster::block_on(Bgra2P010DmaBuf::new(PROBE_DIM, PROBE_DIM)).map_err(|e| {
        ProbeError::new(
            PipelineStage::Capture,
            format!(
                "Bgra2P010DmaBuf::new failed — driver likely lacks R16/Rg16 \
                 storage support on DRM_FORMAT_MOD_LINEAR: {e}"
            ),
        )
    })?;
    let probe_bytes = vec![0x80u8; (PROBE_DIM * PROBE_DIM * 4) as usize];
    let p010 = bridge.convert_bgra_bytes(&probe_bytes).map_err(|e| {
        ProbeError::new(PipelineStage::Capture, format!("P010 bridge convert: {e}"))
    })?;
    let codec_frame = build_p010_dmabuf_frame(
        p010.fd,
        p010.size,
        p010.modifier,
        p010.y_offset,
        p010.y_stride,
        p010.uv_offset,
        p010.uv_stride,
    );
    enc.submit_dmabuf(&codec_frame, 0, true)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Submit, e))?;
    Ok(())
}
