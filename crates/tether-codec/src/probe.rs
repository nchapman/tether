//! Per-session codec construction + handshake-time validation.
//!
//! Tether hard-requires GPU acceleration on both ends — no software
//! fallback path. The motivation: software H.264 at 4K30 burns ~2-3
//! cores on the capture side and the same on decode; that budget
//! needs to be free for capture, encode-side rate control, and (on
//! the client) a responsive UI thread + sample-accurate input
//! forwarding.
//!
//! Capability discovery has moved to the `tether-probe` crate, which
//! round-trips the full production chain (capture → bridge → encoder
//! → decoder) per profile and caches the result process-wide. Call
//! sites that used to ask `tether_codec::supported_*_profiles()` now
//! call `tether_probe::host_encode_profiles()` /
//! `tether_probe::client_decode_profiles()`.
//!
//! What stays in tether-codec:
//!   * [`build_encoder`] / [`build_decoder`] — per-session construction
//!     helpers. They build the real session objects at real dimensions
//!     and return them for the host send loop / client decode thread.
//!   * [`validate_chosen_profile`] — handshake-time check that the
//!     host's chosen profile is one this client actually advertised.
//!     Pure validation, no probing.

use crate::{CodecError, Decoder, Encoder, Result};
#[cfg_attr(target_os = "macos", allow(unused_imports))]
use tether_protocol::control::{CodecKind, VideoProfile};
// Only the Linux `no_hw_encoder` VAAPI-profile hint matches on chroma.
#[cfg(target_os = "linux")]
use tether_protocol::control::ChromaSubsampling;

/// Verify that a host's chosen `VideoProfile` is one this client
/// actually advertised it could decode. Returns `Ok(())` if `chosen`
/// is in `advertised`, `Err` with an actionable message otherwise.
///
/// The host's negotiator picks from the intersection of host-encode
/// and client-decode sets — a chosen profile outside `advertised` is
/// either a buggy host or a hostile peer trying to push the client
/// onto a code path it never opted into. Either way the right
/// response is a session-fatal bail at handshake, not silent best-
/// effort rendering with the wrong pipeline.
pub fn validate_chosen_profile(
    chosen: VideoProfile,
    advertised: &[VideoProfile],
) -> Result<()> {
    if advertised.contains(&chosen) {
        return Ok(());
    }
    Err(CodecError::NoHardwareCodec(format!(
        "host chose profile {chosen:?} which this client did not advertise \
         ({} entries in supported_decode_profiles)",
        advertised.len()
    )))
}

/// Construct an encoder for the negotiated session. Builds the real
/// session encoder at real dimensions and returns it for the host
/// send loop. Errors with a diagnostics-friendly message if
/// construction fails.
pub fn build_encoder(
    profile: VideoProfile,
    width: u32,
    height: u32,
    fps: u32,
    bitrate_kbps: u32,
) -> Result<(VideoProfile, Box<dyn Encoder>)> {
    #[cfg(target_os = "linux")]
    {
        match crate::vaapi::VaapiEncoder::new(profile, width, height, fps, bitrate_kbps) {
            Ok(enc) => Ok((profile, Box::new(enc))),
            Err(e) => {
                tracing::warn!(
                    backend = "vaapi",
                    codec = ?profile.codec,
                    chroma = ?profile.chroma,
                    bit_depth = profile.bit_depth,
                    error = %e,
                    "VAAPI encoder construction failed"
                );
                Err(no_hw_encoder(profile, e))
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        match crate::videotoolbox::VideoToolboxEncoder::new(
            profile,
            width,
            height,
            fps,
            bitrate_kbps,
        ) {
            Ok(enc) => Ok((profile, Box::new(enc))),
            Err(e) => {
                tracing::warn!(
                    backend = "videotoolbox",
                    codec = ?profile.codec,
                    chroma = ?profile.chroma,
                    bit_depth = profile.bit_depth,
                    error = %e,
                    "VideoToolbox encoder construction failed"
                );
                Err(no_hw_encoder_vt(profile.codec, e))
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        build_encoder_d3d11(profile, width, height, fps, bitrate_kbps, std::ptr::null_mut(), std::ptr::null_mut(), 0)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        let _ = (profile, width, height, fps, bitrate_kbps);
        Err(no_hw_encoder_for_platform())
    }
}

/// Windows-specific encoder construction that accepts a shared D3D11
/// device from the capture layer. When `device_ptr` is non-null, the
/// encoder reuses the capture device (zero-copy texture sharing);
/// when null, FFmpeg creates its own device (probe path, or fallback).
#[cfg(target_os = "windows")]
pub fn build_encoder_d3d11(
    profile: VideoProfile,
    width: u32,
    height: u32,
    fps: u32,
    bitrate_kbps: u32,
    device_ptr: *mut std::ffi::c_void,
    device_ctx_ptr: *mut std::ffi::c_void,
    vendor_id: u32,
) -> Result<(VideoProfile, Box<dyn Encoder>)> {
    match crate::d3d11::D3D11Encoder::new(
        profile,
        width,
        height,
        fps,
        bitrate_kbps,
        device_ptr,
        device_ctx_ptr,
        vendor_id,
    ) {
        Ok(enc) => Ok((profile, Box::new(enc))),
        Err(e) => {
            tracing::warn!(
                backend = "d3d11",
                codec = ?profile.codec,
                chroma = ?profile.chroma,
                bit_depth = profile.bit_depth,
                error = %e,
                "D3D11 encoder construction failed"
            );
            Err(no_hw_encoder_d3d11(profile.codec, e))
        }
    }
}

/// Construct the decoder for the host's negotiated `VideoProfile`.
/// Takes the full profile (not just `CodecKind`) for symmetry with
/// [`build_encoder`] and the bridge/renderer constructors: chroma and
/// bit-depth are part of session identity even when a particular
/// backend (today: FFmpeg HEVC) reads them from the bitstream and
/// doesn't need them at construction. Future chroma-aware decoder
/// backends won't require a signature change.
///
/// `gpu_export` is the renderer's zero-copy import capability, consumed
/// only by the Windows D3D11 decoder (other backends always export
/// GPU-resident frames). See [`crate::d3d11::D3D11Decoder::new`].
///
/// Errors if no GPU decoder is available for `profile.codec` on this
/// client.
#[cfg_attr(not(target_os = "windows"), allow(unused_variables))]
pub fn build_decoder(profile: VideoProfile, gpu_export: bool) -> Result<Box<dyn Decoder>> {
    let kind = profile.codec;
    #[cfg(target_os = "linux")]
    {
        match crate::vaapi::VaapiDecoder::new(kind) {
            Ok(dec) => return Ok(Box::new(dec)),
            Err(e) => {
                tracing::error!(
                    backend = "vaapi",
                    codec = ?kind,
                    chroma = ?profile.chroma,
                    bit_depth = profile.bit_depth,
                    error = %e,
                    "VAAPI decoder construction failed"
                );
                return Err(no_hw_decoder(kind, e));
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        match crate::videotoolbox::VideoToolboxDecoder::new(kind) {
            Ok(dec) => return Ok(Box::new(dec)),
            Err(e) => {
                tracing::error!(
                    backend = "videotoolbox",
                    codec = ?kind,
                    chroma = ?profile.chroma,
                    bit_depth = profile.bit_depth,
                    error = %e,
                    "VideoToolbox decoder construction failed"
                );
                return Err(no_hw_decoder_vt(kind, e));
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        match crate::d3d11::D3D11Decoder::new(kind, gpu_export) {
            Ok(dec) => return Ok(Box::new(dec)),
            Err(e) => {
                tracing::error!(
                    backend = "d3d11va",
                    codec = ?kind,
                    chroma = ?profile.chroma,
                    bit_depth = profile.bit_depth,
                    error = %e,
                    "D3D11VA decoder construction failed"
                );
                return Err(no_hw_decoder_d3d11(kind, e));
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        let _ = profile;
        Err(no_hw_decoder_for_platform())
    }
}

#[cfg(target_os = "linux")]
fn no_hw_encoder(profile: VideoProfile, source: CodecError) -> CodecError {
    let profile_hint = match (profile.codec, profile.chroma) {
        (CodecKind::H264, _) => "VAProfileH264{ConstrainedBaseline,Main,High}",
        (CodecKind::Hevc, ChromaSubsampling::Yuv420) => "VAProfileHEVCMain",
        (CodecKind::Hevc, ChromaSubsampling::Yuv444) => "VAProfileHEVCMain444",
        (CodecKind::Av1, _) => "VAProfileAV1Profile0",
    };
    CodecError::NoHardwareCodec(format!(
        "VAAPI encoder unavailable for {:?} {:?} {}-bit ({source}). \
         Check that /dev/dri/renderD128 is present and readable, and that `vainfo` \
         lists {profile_hint} with VAEntrypointEnc*. \
         Tether requires GPU encode — there is no software fallback.",
        profile.codec, profile.chroma, profile.bit_depth
    ))
}

#[cfg(target_os = "macos")]
fn no_hw_encoder_vt(kind: CodecKind, source: CodecError) -> CodecError {
    CodecError::NoHardwareCodec(format!(
        "VideoToolbox encoder unavailable for {kind:?} ({source}). \
         Check that `ffmpeg -hide_banner -encoders | grep videotoolbox` lists \
         h264_videotoolbox / hevc_videotoolbox — Homebrew's `ffmpeg` formula \
         enables `--enable-videotoolbox` by default; a custom build may not. \
         Tether requires GPU encode — there is no software fallback."
    ))
}

#[cfg(target_os = "linux")]
fn no_hw_decoder(kind: CodecKind, source: CodecError) -> CodecError {
    CodecError::NoHardwareCodec(format!(
        "VAAPI decoder unavailable for {kind:?} ({source}). \
         Check `vainfo` lists VAEntrypointVLD for the chosen codec, and that the \
         kernel + libva versions match (Mesa 24+ on a 6.x kernel is the verified \
         path). Tether requires GPU decode — there is no software fallback."
    ))
}

#[cfg(target_os = "macos")]
fn no_hw_decoder_vt(kind: CodecKind, source: CodecError) -> CodecError {
    CodecError::NoHardwareCodec(format!(
        "VideoToolbox decoder unavailable for {kind:?} ({source}). \
         Check that `ffmpeg -hide_banner -decoders | grep videotoolbox` reports \
         hwaccel support for the codec — Homebrew's `ffmpeg` formula enables \
         `--enable-videotoolbox` by default; a custom build may not. \
         Tether requires GPU decode — there is no software fallback."
    ))
}

#[cfg(target_os = "windows")]
fn no_hw_encoder_d3d11(kind: CodecKind, source: CodecError) -> CodecError {
    CodecError::NoHardwareCodec(format!(
        "D3D11 encoder unavailable for {kind:?} ({source}). \
         Check that FFmpeg was built with Media Foundation, QSV, NVENC, or AMF support \
         (`ffmpeg -encoders | grep -E 'mf|nvenc|amf|qsv'`). \
         Tether requires GPU encode — there is no software fallback."
    ))
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn no_hw_encoder_for_platform() -> CodecError {
    CodecError::NoHardwareCodec(
        "Tether currently supports hardware encode on Linux (VAAPI), \
         macOS (VideoToolbox), and Windows (D3D11VA). No backend is \
         available for this platform."
            .to_string(),
    )
}

#[cfg(target_os = "windows")]
fn no_hw_decoder_d3d11(kind: CodecKind, source: CodecError) -> CodecError {
    CodecError::NoHardwareCodec(format!(
        "D3D11VA decoder unavailable for {kind:?} ({source}). \
         Check that FFmpeg was built with d3d11va support \
         (`ffmpeg -hwaccels | grep d3d11va`). \
         Tether requires GPU decode — there is no software fallback."
    ))
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn no_hw_decoder_for_platform() -> CodecError {
    CodecError::NoHardwareCodec(
        "Tether currently supports hardware decode on Linux (VAAPI), \
         macOS (VideoToolbox), and Windows (D3D11VA). No backend is \
         available for this platform."
            .to_string(),
    )
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn validate_chosen_profile_accepts_advertised() {
        let advertised = vec![
            VideoProfile::HEVC_8BIT_444,
            VideoProfile::HEVC_8BIT_420,
            VideoProfile::H264_8BIT_420,
        ];
        assert!(
            validate_chosen_profile(VideoProfile::HEVC_8BIT_444, &advertised).is_ok()
        );
        assert!(
            validate_chosen_profile(VideoProfile::H264_8BIT_420, &advertised).is_ok()
        );
    }

    #[test]
    fn validate_chosen_profile_rejects_unadvertised() {
        let advertised = vec![VideoProfile::H264_8BIT_420];
        let err = validate_chosen_profile(VideoProfile::HEVC_8BIT_444, &advertised)
            .expect_err("4:4:4 should be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("HEVC_8BIT_444") || msg.contains("Yuv444"),
            "error should name the offending profile; got: {msg}"
        );
    }

    #[test]
    fn validate_chosen_profile_rejects_empty_advertised() {
        assert!(
            validate_chosen_profile(VideoProfile::H264_8BIT_420, &[]).is_err()
        );
    }
}
