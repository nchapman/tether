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
//!
//! Codec negotiation: [`probe_encoder`] walks the client's
//! preferred-codec list and returns the first `(CodecKind, encoder)`
//! pair we can actually construct. The shape is registry-of-backends:
//! today only VAAPI on Linux, but VideoToolbox / Media Foundation slot
//! into the same iteration when their backends land. Trait-based
//! abstraction is deferred until a second concrete backend exists to
//! shape it against.

use crate::{CodecError, Decoder, Encoder, Result};
use tether_protocol::control::CodecKind;

/// Probe + construct an encoder for the first codec in `preferred`
/// that this host can actually build. Returns the chosen
/// [`CodecKind`] alongside the constructed encoder so the caller
/// (host send loop) can echo it back to the client through
/// [`tether_protocol::control::ServerHelloV1::chosen_codec`].
///
/// `fps` sets the encoder's time_base. `bitrate_kbps` is a soft VBR
/// target.
///
/// Errors with a diagnostics-friendly message if no codec in
/// `preferred` constructs successfully. An empty preference list is
/// also an error — the caller is expected to have validated this
/// upstream (the client's `preferred_codecs` defaults to a non-empty
/// list).
pub fn probe_encoder(
    preferred: &[CodecKind],
    width: u32,
    height: u32,
    fps: u32,
    bitrate_kbps: u32,
) -> Result<(CodecKind, Box<dyn Encoder>)> {
    if preferred.is_empty() {
        return Err(CodecError::NoHardwareCodec(
            "client preferred_codecs list was empty".to_string(),
        ));
    }

    #[cfg(target_os = "linux")]
    {
        let mut last_err: Option<(CodecKind, CodecError)> = None;
        for kind in preferred {
            match crate::vaapi::VaapiEncoder::new(*kind, width, height, fps, bitrate_kbps) {
                Ok(enc) => return Ok((*kind, Box::new(enc))),
                Err(e) => {
                    tracing::warn!(
                        backend = "vaapi",
                        codec = ?kind,
                        error = %e,
                        "VAAPI encoder construction failed for codec; trying next"
                    );
                    last_err = Some((*kind, e));
                }
            }
        }
        let (kind, src) = last_err.expect("loop entered with non-empty preferred");
        return Err(no_hw_encoder(kind, src));
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (preferred, width, height, fps, bitrate_kbps);
        Err(no_hw_encoder_for_platform())
    }
}

/// Lightweight handshake-time capability check: is `kind` a codec we
/// can *currently* build on this host? Implemented as a tiny
/// construction probe at 128×128 — captures real driver state (not
/// just FFmpeg build-time support), at the cost of a one-time VAAPI
/// device open. Caller is expected to invoke this once per session
/// during handshake, not per frame.
///
/// The 128×128 floor satisfies HEVC's minimum-block constraint on
/// Intel hardware (which rejects anything under 128×128 with EINVAL).
/// H.264 accepts smaller, but using the same dims keeps the probe a
/// single config across codecs.
///
/// Driver-portability caveat: rsmpeg's `AVCodecContext` exposes a
/// known failure mode where `encoder.open()` returning an error
/// leaves the context partially-initialized, and the subsequent Drop
/// segfaults (see the comment in `vaapi/encoder.rs` about the LP
/// entrypoint). We've validated this probe at 128×128 against H.264
/// and HEVC on Intel Arc (Meteor Lake). AMD and NVIDIA-via-VAAPI may
/// have different minimum block sizes for HEVC; if the probe ever
/// SIGSEGVs on a new driver, the principled fix is to add the
/// `vaQueryConfigProfiles` libva probe before construction. Today
/// we accept the risk because we don't have the test hardware.
pub fn probe_encoder_kind(kind: CodecKind) -> bool {
    #[cfg(target_os = "linux")]
    {
        crate::vaapi::VaapiEncoder::new(kind, 128, 128, 30, 1_000).is_ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = kind;
        false
    }
}

/// Probe + construct the decoder for the codec the host chose in its
/// [`ServerHelloV1`](tether_protocol::control::ServerHelloV1::chosen_codec).
/// Errors if no GPU decoder is available for that codec on this client.
pub fn probe_decoder(kind: CodecKind) -> Result<Box<dyn Decoder>> {
    #[cfg(target_os = "linux")]
    {
        match crate::vaapi::VaapiDecoder::new(kind) {
            Ok(dec) => return Ok(Box::new(dec)),
            Err(e) => {
                tracing::error!(
                    backend = "vaapi",
                    codec = ?kind,
                    error = %e,
                    "VAAPI decoder construction failed"
                );
                return Err(no_hw_decoder(kind, e));
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = kind;
        Err(no_hw_decoder_for_platform())
    }
}

#[cfg(target_os = "linux")]
fn no_hw_encoder(kind: CodecKind, source: CodecError) -> CodecError {
    let profile_hint = match kind {
        CodecKind::H264 => "VAProfileH264{ConstrainedBaseline,Main,High}",
        CodecKind::Hevc => "VAProfileHEVCMain",
        CodecKind::Av1 => "VAProfileAV1Profile0",
    };
    CodecError::NoHardwareCodec(format!(
        "VAAPI encoder unavailable for {kind:?} ({source}). \
         Check that /dev/dri/renderD128 is present and readable, and that `vainfo` \
         lists {profile_hint} with VAEntrypointEnc*. \
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

#[cfg(not(target_os = "linux"))]
fn no_hw_encoder_for_platform() -> CodecError {
    CodecError::NoHardwareCodec(
        "Tether currently supports hardware encode only on Linux (VAAPI). \
         macOS/VideoToolbox, Windows/NVENC, and Windows/AMF backends are not \
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
