//! NVENC implementation of the host encode probe.
//!
//! Only the encode half lives here. Decode on an NVIDIA host goes through
//! `NvdecDecoder`, probed by [`super::nvdec::NvdecProbe`] (see
//! `host::probe_decode`) — NOT VAAPI, whose `nvidia-vaapi-driver` decoder
//! SIGSEGVs. `host::probe_encode` routes NVIDIA hosts here exclusively,
//! mirroring the live `build_encoder` dispatch so the advertised capability
//! set matches what the session encoder will actually pick.
//!
//! The probe is a real round trip against the live driver, not a
//! construction-only check: `NvencEncoder::new` opening the codec is not
//! sufficient evidence the encode actually runs. Both bit depths drive a real
//! `submit_dmabuf` round trip through the production zero-copy import — 8-bit
//! 4:2:0 via `Nv12DmaBuf`, 10-bit Main10 via `Bgra2P010DmaBuf`, and 8-bit
//! 4:4:4 via `Yuv444pDmaBuf`. This is
//! deliberately stronger than the VAAPI probe's 8-bit path (which only does
//! `encode_bgra`): the live Linux capture loop always feeds 8-bit GPU frames
//! through `encode_gpu(DmaBuf) → submit_dmabuf` (the EGL→CUDA import), and a
//! `submit_dmabuf` failure there is handled as a dropped frame, not a CPU-upload
//! fallback. Probing only `encode_bgra` would let a host whose EGLImage import
//! is broken advertise H.264/HEVC 8-bit and then drop every captured frame. The
//! CPU-upload path stays covered by the `encode_bgra` hardware unit tests.

use tether_codec::nvenc::NvencEncoder;
use tether_codec::{build_nv12_dmabuf_frame, build_p010_dmabuf_frame, build_yuv444p_dmabuf_frame};
use tether_gpuconvert::{Bgra2P010DmaBuf, Nv12DmaBuf, Yuv444pDmaBuf};
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

/// Probe canvas. 256, not the 128 the VAAPI probe uses: constructing a 128×128 HEVC Main10
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
    /// (`host::probe_encode`) records the profile unsupported for this
    /// NVIDIA host.
    pub(crate) fn probe_encode(profile: VideoProfile) -> Result<()> {
        tether_codec::av_log::with_probe_suppression(|| probe_encode_inner(profile))
    }
}

fn probe_encode_inner(profile: VideoProfile) -> Result<()> {
    // Construction catches "driver/codec can't do this profile at all":
    // codec not built (no `--enable-nvenc`), no NVIDIA device / `libcuda`,
    // AV1 on a pre-Ada card, or an unsupported chroma/depth tuple.
    let mut enc = NvencEncoder::new(profile, PROBE_DIM, PROBE_DIM, PROBE_FPS, PROBE_BITRATE_KBPS)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Construct, e))?;

    match (profile.chroma, profile.bit_depth) {
        // 8-bit 4:2:0 (Main / H.264): real submit_dmabuf round trip via the
        // production `Nv12DmaBuf` bridge — the path the live capture loop uses
        // for every 8-bit GPU frame. Proves the EGL→CUDA NV12 import works, not
        // just that the codec opens and CPU upload runs. (encode_bgra, the CPU
        // path, would pass even when this zero-copy path is broken.)
        (ChromaSubsampling::Yuv420, 8) => probe_nv12_submit(&mut enc)?,
        // 10-bit 4:2:0 (Main10): same, via the `Bgra2P010DmaBuf` bridge — the
        // encoder's only 10-bit input path (encode_bgra has no 10-bit branch).
        (ChromaSubsampling::Yuv420, 10) => probe_p010_submit(&mut enc)?,
        // 8-bit 4:4:4: planar YUV444P bridge, the only 4:4:4 layout NVENC
        // accepts through FFmpeg. Tested NVIDIA EGL stacks currently reject
        // the YU24 dma-buf at eglCreateImage, so this probe normally records
        // the profile unsupported rather than advertising it.
        (ChromaSubsampling::Yuv444, 8) => probe_yuv444p_submit(&mut enc)?,
        // 10-bit 4:4:4 is deliberately deferred: NVENC/NVDEC can represent
        // planar YUV444P16, and DRM has Q410/S410 candidates, but the current
        // NVIDIA EGL dma-buf import path has not accepted planar 4:4:4 formats.
        (chroma, bd) => {
            return Err(ProbeError::new(
                PipelineStage::Construct,
                format!("unsupported {chroma:?} {bd}-bit for NVENC encode probe"),
            ));
        }
    }
    Ok(())
}

fn probe_yuv444p_submit(enc: &mut NvencEncoder) -> Result<()> {
    let bridge = pollster::block_on(Yuv444pDmaBuf::new(PROBE_DIM, PROBE_DIM)).map_err(|e| {
        ProbeError::new(
            PipelineStage::Capture,
            format!("Yuv444pDmaBuf::new failed (zero-copy YUV444P capture unavailable): {e}"),
        )
    })?;
    let probe_bytes = vec![0x80u8; (PROBE_DIM * PROBE_DIM * 4) as usize];
    let yuv = bridge.convert_bgra_bytes(&probe_bytes).map_err(|e| {
        ProbeError::new(
            PipelineStage::Capture,
            format!("YUV444P bridge convert: {e}"),
        )
    })?;
    let codec_frame = build_yuv444p_dmabuf_frame(
        yuv.fd,
        yuv.size,
        yuv.modifier,
        yuv.y_offset,
        yuv.y_stride,
        yuv.u_offset,
        yuv.u_stride,
        yuv.v_offset,
        yuv.v_stride,
    );
    enc.submit_dmabuf(&codec_frame, 0, true)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Submit, e))?;
    Ok(())
}

/// Build an NV12 dma-buf through the production `Nv12DmaBuf` bridge and feed it
/// to NVENC's `submit_dmabuf` — the exact path the live capture loop uses for
/// 8-bit GPU frames. Catches what a construction-only or `encode_bgra` check
/// would miss: bridge construction, the convert, and the EGL→CUDA NV12 import +
/// NVENC's acceptance of the imported surface.
fn probe_nv12_submit(enc: &mut NvencEncoder) -> Result<()> {
    let bridge = pollster::block_on(Nv12DmaBuf::new(PROBE_DIM, PROBE_DIM)).map_err(|e| {
        ProbeError::new(
            PipelineStage::Capture,
            format!("Nv12DmaBuf::new failed (zero-copy NV12 capture unavailable): {e}"),
        )
    })?;
    let probe_bytes = vec![0x80u8; (PROBE_DIM * PROBE_DIM * 4) as usize];
    let nv12 = bridge.convert_bgra_bytes(&probe_bytes).map_err(|e| {
        ProbeError::new(PipelineStage::Capture, format!("NV12 bridge convert: {e}"))
    })?;
    let codec_frame = build_nv12_dmabuf_frame(
        nv12.fd,
        nv12.size,
        nv12.modifier,
        nv12.y_offset,
        nv12.y_stride,
        nv12.uv_offset,
        nv12.uv_stride,
    );
    enc.submit_dmabuf(&codec_frame, 0, true)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Submit, e))?;
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
