//! VAAPI implementation of the [`ProfileProbe`] contract.
//!
//! Both halves are real round trips against the live driver:
//!   - `probe_encode` constructs `VaapiEncoder` at 128×128 and
//!     encodes one BGRA frame for 8-bit profiles. Driver-side
//!     rejection (which on some Intel paths only surfaces at
//!     `send_frame`, not `open()`) is caught here rather than at
//!     first-frame time in production.
//!   - `probe_decode` constructs `VaapiDecoder` and submits the
//!     fixture IDR; success is "produced a `Frame::Gpu` back" — a
//!     `Frame::Cpu` would mean the hwaccel silently fell back to
//!     software, which we treat as "not supported."
//!
//! Step 2 of the migration: this is a verbatim move from
//! `tether-codec::vaapi::probe`. Step 3 extends `probe_encode` for
//! 10-bit profiles to also run a real `Bgra2P010DmaBuf` →
//! `submit_dmabuf` round trip, closing the gap that the host's
//! separate `warm_gpuconvert_capability_cache` used to fill.

use tether_codec::vaapi::{VaapiDecoder, VaapiEncoder};
use tether_codec::{build_p010_dmabuf_frame, build_xv30_dmabuf_frame, Decoder, Encoder, Frame};
use tether_gpuconvert::{Bgra2P010DmaBuf, Bgra2Xv30DmaBuf};
use tether_protocol::control::{ChromaSubsampling, VideoProfile};

use crate::profile_probe::{ProbeError, ProfileProbe, Result};
use crate::PipelineStage;

pub(crate) struct VaapiProbe;

/// Probe canvas. 128×128 satisfies HEVC's minimum-block constraint on
/// Intel; H.264 accepts smaller but using the same dims keeps the
/// probe a single config across codecs and matches the fixture dims.
const PROBE_DIM: u32 = 128;
const PROBE_FPS: u32 = 30;
const PROBE_BITRATE_KBPS: u32 = 1_000;

impl ProfileProbe for VaapiProbe {
    fn probe_encode(profile: VideoProfile) -> Result<()> {
        tether_codec::av_log::with_probe_suppression(|| probe_encode_inner(profile))
    }

    fn probe_decode(profile: VideoProfile, fixture: &[u8]) -> Result<()> {
        tether_codec::av_log::with_probe_suppression(|| probe_decode_inner(profile, fixture))
    }
}

fn probe_encode_inner(profile: VideoProfile) -> Result<()> {
    let mut enc = VaapiEncoder::new(profile, PROBE_DIM, PROBE_DIM, PROBE_FPS, PROBE_BITRATE_KBPS)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Construct, e))?;
    match (profile.chroma, profile.bit_depth) {
        // 8-bit: CPU BGRA upload exercises encoder submit-time
        // rejection (some Intel paths only fail here, not at
        // `VaapiEncoder::new`).
        (_, 8) => {
            let bytes = vec![0x80u8; (PROBE_DIM * PROBE_DIM * 4) as usize];
            let _ = enc
                .encode_bgra(&bytes, 0, true)
                .map_err(|e| ProbeError::from_codec(PipelineStage::Submit, e))?;
        }
        // 10-bit 4:2:0: real submit_dmabuf round trip via the
        // production `Bgra2P010DmaBuf` bridge. Closes the gap the
        // codec-internal probe couldn't (no gpuconvert dep) and
        // catches the Intel iHD + Mesa + FFmpeg 8.1 case where
        // `VaapiEncoder::new` accepts Main10 + P010LE but
        // `av_hwframe_map(DRM_PRIME → VAAPI)` rejects the dma-buf at
        // submit time.
        (ChromaSubsampling::Yuv420, 10) => probe_p010_submit(&mut enc)?,
        // 10-bit 4:4:4: same shape via the packed XV30 bridge.
        // VAAPI's vaapi_drm_format_map has no biplanar P410 entry,
        // so XV30 is the only viable 4:4:4 10-bit input format.
        (ChromaSubsampling::Yuv444, 10) => probe_xv30_submit(&mut enc)?,
        (_, bd) => {
            return Err(ProbeError::new(
                PipelineStage::Construct,
                format!("unsupported bit depth {bd} for VAAPI encode probe"),
            ));
        }
    }
    Ok(())
}

fn probe_decode_inner(profile: VideoProfile, fixture: &[u8]) -> Result<()> {
    let mut dec = VaapiDecoder::new(profile.codec)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Construct, e))?;
    dec.submit(fixture)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
    // Signal EOF to drain: ffmpeg's HEVC/AV1 decoders buffer the first
    // frame in the reorder DPB until a subsequent packet or EOF arrives.
    // The probe submits a lone IDR, so without this drain HEVC never
    // emits (H.264 happens to flush immediately). Matches the D3D11 and
    // VideoToolbox decode probes.
    dec.signal_eof()
        .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
    // Loop a few times — VAAPI sometimes needs a couple of
    // `receive_frame` polls before the drained frame surfaces.
    for _ in 0..4 {
        match dec
            .next_frame()
            .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?
        {
            Some(Frame::Gpu(_)) => return Ok(()),
            Some(Frame::Cpu(_)) => {
                return Err(ProbeError::new(
                    PipelineStage::Decode,
                    "decoder fell back to software (Frame::Cpu) — \
                     hardware decode unavailable for this profile",
                ))
            }
            None => continue,
        }
    }
    Err(ProbeError::new(
        PipelineStage::Decode,
        "decoder produced no frames after submit + eof + 4 polls",
    ))
}

/// Build a `Bgra2P010DmaBuf`, convert a flat-grey BGRA buffer through
/// it, and feed the resulting dma-buf to the encoder. Returns Err
/// (mapped to `CodecError::NoHardwareCodec`) for the three failures
/// this round trip is supposed to catch:
///   * Bridge construction fails (driver doesn't advertise R16/Rg16
///     storage modifiers on `DRM_FORMAT_MOD_LINEAR`).
///   * Bridge convert fails (transient GPU-side error).
///   * `submit_dmabuf` fails (the Intel iHD `av_hwframe_map` rejection
///     case).
fn probe_p010_submit(enc: &mut VaapiEncoder) -> Result<()> {
    // Bridge construction failure → Capture stage. The driver
    // doesn't expose R16/Rg16 storage modifiers on DRM_FORMAT_MOD_LINEAR,
    // i.e. the producer side can't make a P010 dma-buf at all.
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
    // Convert failure → Capture stage (the producer pipeline broke).
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
    // Submit failure → Submit stage. This is the Intel iHD case: the
    // encoder constructed fine but `av_hwframe_map(DRM_PRIME → VAAPI)`
    // rejects the P010 descriptor at submit time. We're already inside
    // `with_probe_suppression` via the impl wrapper, so the expected
    // driver-side rejection (both our own encoder warn and ffmpeg's
    // native error line) is downgraded — the typed `ProbeError` we
    // return is the load-bearing signal, not the log line.
    enc.submit_dmabuf(&codec_frame, 0, true)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Submit, e))?;
    Ok(())
}

/// XV30 (HEVC Main 4:4:4 10-bit) sibling of [`probe_p010_submit`].
/// Same three failure modes:
///   * Bridge construction fails — driver doesn't advertise
///     `STORAGE_IMAGE` on `DRM_FORMAT_XV30`
///     (`VK_FORMAT_A2B10G10R10_UNORM_PACK32`) with `LINEAR`. The
///     bridge probes this up front and returns
///     `Xv30DmaBufError::StorageUnsupported` cleanly.
///   * Bridge convert fails — transient GPU-side error.
///   * `submit_dmabuf` fails — `av_hwframe_map(DRM_PRIME → VAAPI)`
///     rejects the XV30 descriptor at submit time (the 4:4:4 10-bit
///     analogue of the Intel iHD P010 rejection).
fn probe_xv30_submit(enc: &mut VaapiEncoder) -> Result<()> {
    let bridge = pollster::block_on(Bgra2Xv30DmaBuf::new(PROBE_DIM, PROBE_DIM)).map_err(|e| {
        ProbeError::new(
            PipelineStage::Capture,
            format!(
                "Bgra2Xv30DmaBuf::new failed — driver likely lacks Rgb10a2Unorm \
                 storage support on DRM_FORMAT_MOD_LINEAR: {e}"
            ),
        )
    })?;
    let probe_bytes = vec![0x80u8; (PROBE_DIM * PROBE_DIM * 4) as usize];
    let xv30 = bridge.convert_bgra_bytes(&probe_bytes).map_err(|e| {
        ProbeError::new(PipelineStage::Capture, format!("XV30 bridge convert: {e}"))
    })?;
    let codec_frame = build_xv30_dmabuf_frame(
        xv30.fd,
        xv30.size,
        xv30.modifier,
        xv30.offset,
        xv30.stride,
    );
    enc.submit_dmabuf(&codec_frame, 0, true)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Submit, e))?;
    Ok(())
}
