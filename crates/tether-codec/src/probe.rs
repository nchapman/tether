//! Encoder selection policy.
//!
//! `probe_encoder_bgra` is the single construction site for the host's
//! video encoder. It walks a hardcoded preference list — hardware
//! backends first, software libx264 as the last-resort fallback — and
//! returns the first one whose constructor succeeds wrapped in a
//! `Box<dyn Encoder>`. Each backend's `new_bgra(...)` failure is
//! interpreted as "not available on this system" and falls through
//! quietly; only a SW-fallback failure is propagated, since that means
//! the FFmpeg build is broken and the host can't function.
//!
//! Resolution changes recreate the encoder via this same function, so
//! probe cost is paid per resize, not per frame. We don't cache the
//! chosen-backend identity — re-probing on resize is fast for
//! software and gives hardware backends a clean retry if a previous
//! attempt failed transiently (e.g. VAAPI session not yet established).

use crate::{Decoder, Encoder, H264Decoder, H264Encoder, Result};

/// Probe + construct the best available H.264 encoder for the given
/// dimensions. Hardware backends are tried first; falls through to
/// libx264 software encode if no hardware path constructs successfully.
///
/// `fps` is the *target* rate (sets the encoder's time_base). `bitrate_kbps`
/// is a soft target — libx264 with `tune=zerolatency` doesn't strictly
/// cap, and VAAPI's rate-control mode will treat it as a VBR target.
pub fn probe_encoder_bgra(
    width: u32,
    height: u32,
    fps: u32,
    bitrate_kbps: u32,
) -> Result<Box<dyn Encoder>> {
    // Hardware backends try-in-order. Each candidate is "try construct,
    // fall through quietly on failure" — failure usually means "not
    // available on this system" (no VAAPI device, NVIDIA but no
    // CUDA, etc.), which is expected and shouldn't propagate.
    //
    // The debug log on failure is intentional: a user investigating
    // "why is my host using libx264 instead of VAAPI" can flip the
    // tether-codec log level to debug and see the constructor error
    // without having to attach a debugger.

    #[cfg(target_os = "linux")]
    {
        match crate::vaapi::VaapiEncoder::new_bgra(width, height, fps, bitrate_kbps) {
            Ok(enc) => return Ok(Box::new(enc)),
            Err(e) => {
                tracing::debug!(
                    backend = "h264_vaapi",
                    error = %e,
                    "hardware encoder unavailable; trying next candidate"
                );
            }
        }
    }

    // NVENC / VideoToolbox / AMF candidates slot in here as additional
    // platform-gated try blocks.

    // Software fallback. libx264 is bundled with every reasonable
    // FFmpeg build — if this fails the host can't encode at all, so
    // propagate the error rather than silently going dark.
    let enc = H264Encoder::new_bgra(width, height, fps, bitrate_kbps)?;
    Ok(Box::new(enc))
}

/// Probe + construct the best available H.264 decoder. Same priority
/// pattern as `probe_encoder_bgra`: hardware backends try first, the
/// libavcodec software decoder is the last-resort fallback.
///
/// Hardware decode is the symmetric optimisation to hardware encode —
/// removes CPU cycles from the client's hot path so it can keep up
/// at high resolution without dropping frames or stretching present
/// latency.
pub fn probe_decoder() -> Result<Box<dyn Decoder>> {
    #[cfg(target_os = "linux")]
    {
        match crate::vaapi::VaapiDecoder::new() {
            Ok(dec) => return Ok(Box::new(dec)),
            Err(e) => {
                tracing::debug!(
                    backend = "h264 vaapi",
                    error = %e,
                    "hardware decoder unavailable; trying next candidate"
                );
            }
        }
    }

    // NVDEC / VideoToolbox / D3D11VA candidates slot in here as
    // additional platform-gated try blocks.

    // Software fallback.
    let dec = H264Decoder::new()?;
    Ok(Box::new(dec))
}
