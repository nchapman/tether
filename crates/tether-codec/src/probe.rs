//! Encoder / decoder selection policy. Tether hard-requires GPU
//! acceleration on both ends — no software fallback path.
//!
//! The motivation: software H.264 at 4K30 burns ~2-3 cores on the
//! capture side and the same on decode; that budget needs to be free
//! for capture, encode-side rate control, and (on the client) a
//! responsive UI thread + sample-accurate input forwarding. The
//! zero-copy DMA-BUF decode path (decoder surface -> wgpu Vulkan
//! import, no CPU readback) is also only reachable through the VAAPI
//! decoder — the SW path would feed CPU planes back through the
//! upload road we deliberately walked off of. Committing to "GPU or
//! nothing" makes the rest of the pipeline simpler (one render path,
//! one stat surface) and surfaces driver problems immediately
//! instead of hiding them behind a slower-and-different fallback.
//!
//! Resolution changes recreate the encoder via this same function, so
//! probe cost is paid per resize, not per frame.

use crate::{CodecError, Decoder, Encoder, Result};
#[cfg(target_os = "linux")]
use tether_protocol::control::CodecKind;

/// Probe + construct the H.264 encoder for the given dimensions.
/// Errors with a diagnostics-friendly message if no GPU encoder is
/// available on this system.
///
/// `fps` sets the encoder's time_base. `bitrate_kbps` is a soft VBR
/// target.
pub fn probe_encoder_bgra(
    width: u32,
    height: u32,
    fps: u32,
    bitrate_kbps: u32,
) -> Result<Box<dyn Encoder>> {
    #[cfg(target_os = "linux")]
    {
        match crate::vaapi::VaapiEncoder::new(CodecKind::H264, width, height, fps, bitrate_kbps) {
            Ok(enc) => return Ok(Box::new(enc)),
            Err(e) => {
                tracing::error!(
                    backend = "h264_vaapi",
                    error = %e,
                    "VAAPI encoder construction failed"
                );
                return Err(no_hw_encoder(e));
            }
        }
    }

    // NVENC / VideoToolbox / AMF slot in here when their backends
    // land. Until then, non-Linux hosts can't run.
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (width, height, fps, bitrate_kbps);
        Err(no_hw_encoder_for_platform())
    }
}

/// Probe + construct the H.264 decoder. Errors if no GPU decoder is
/// available.
///
/// Hard-required because the zero-copy decode-to-render pipeline
/// (VAAPI surface -> DMA-BUF -> wgpu Vulkan import) only exists on
/// the GPU path; the SW decoder would produce CPU NV12 that the
/// renderer then has to upload, defeating the work.
pub fn probe_decoder() -> Result<Box<dyn Decoder>> {
    #[cfg(target_os = "linux")]
    {
        match crate::vaapi::VaapiDecoder::new(CodecKind::H264) {
            Ok(dec) => return Ok(Box::new(dec)),
            Err(e) => {
                tracing::error!(
                    backend = "h264 vaapi",
                    error = %e,
                    "VAAPI decoder construction failed"
                );
                return Err(no_hw_decoder(e));
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        Err(no_hw_decoder_for_platform())
    }
}

#[cfg(target_os = "linux")]
fn no_hw_encoder(source: CodecError) -> CodecError {
    CodecError::NoHardwareCodec(format!(
        "VAAPI H.264 encoder unavailable ({source}). \
         Check that /dev/dri/renderD128 is present and readable, and that `vainfo` \
         lists VAProfileH264{{ConstrainedBaseline,Main,High}} with VAEntrypointEnc*. \
         Tether requires GPU encode — there is no software fallback."
    ))
}

#[cfg(target_os = "linux")]
fn no_hw_decoder(source: CodecError) -> CodecError {
    CodecError::NoHardwareCodec(format!(
        "VAAPI H.264 decoder unavailable ({source}). \
         Check `vainfo` lists VAEntrypointVLD for H.264, and that the kernel + \
         libva versions match (Mesa 24+ on a 6.x kernel is the verified path). \
         Tether requires GPU decode — there is no software fallback."
    ))
}

#[cfg(not(target_os = "linux"))]
fn no_hw_encoder_for_platform() -> CodecError {
    CodecError::NoHardwareCodec(
        "Tether currently supports hardware encode only on Linux (VAAPI). \
         macOS/VideoToolpath, Windows/NVENC, and Windows/AMF backends are not \
         yet implemented."
            .to_string(),
    )
}

#[cfg(not(target_os = "linux"))]
fn no_hw_decoder_for_platform() -> CodecError {
    CodecError::NoHardwareCodec(
        "Tether currently supports hardware decode only on Linux (VAAPI). \
         macOS/VideoToolbox, Windows/NVDEC, and Windows/D3D11VA backends are \
         not yet implemented."
            .to_string(),
    )
}
