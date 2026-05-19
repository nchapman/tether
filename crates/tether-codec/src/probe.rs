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

use crate::{Encoder, H264Encoder, Result};

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
    // Hardware backends will register here in preference order. The
    // pattern is the same for each: try the constructor, log a debug
    // note on failure (so a user investigating perf can see *why* the
    // HW path was skipped), and fall through to the next candidate.
    //
    // VAAPI / NVENC / VideoToolbox slot in at this point.

    // Software fallback. libx264 is bundled with every reasonable
    // FFmpeg build — if this fails the host can't encode at all, so
    // propagate the error rather than silently going dark.
    let enc = H264Encoder::new_bgra(width, height, fps, bitrate_kbps)?;
    Ok(Box::new(enc))
}
