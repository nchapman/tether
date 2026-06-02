//! Minimal FFI to libopus, linked from the static FFmpeg package's standalone
//! `libopus.a` (see `build.rs`). We bind libopus directly — rather than going
//! through FFmpeg's `avcodec` wrapper as the video codec does — because the two
//! things the real-time audio path needs aren't reachable through avcodec:
//!
//!   * **Packet-loss concealment** — `opus_decode_float(NULL, …)` reconstructs a
//!     lost frame from decoder state; avcodec never exposes the null-packet call.
//!   * **Live encoder ctl** — `opus_encoder_ctl` sets bitrate/CBR after open;
//!     avcodec freezes rate-control at `init()`.
//!
//! (In-band FEC/LBRR is intentionally *not* used: it's a SILK feature, and our
//! `RESTRICTED_LOWDELAY` mode runs CELT-only, where it's a silent no-op.)
//!
//! FFmpeg's `libavcodec.a` already carries *undefined* `opus_*` references that
//! resolve from this same `libopus.a` at final link, so calling it ourselves
//! adds no second copy of the library.
//!
//! Only the handful of entry points and the (frozen, public-ABI) request
//! constants the codec uses are declared here; values are quoted from
//! `opus_defines.h` in the staged headers.

use std::os::raw::{c_char, c_int};

/// Opaque libopus encoder state. Created/destroyed only through the FFI below;
/// never constructed or dereferenced in Rust.
#[repr(C)]
pub struct OpusEncoder {
    _private: [u8; 0],
}

/// Opaque libopus decoder state.
#[repr(C)]
pub struct OpusDecoder {
    _private: [u8; 0],
}

// --- opus_defines.h: status codes ---
/// No error.
pub const OPUS_OK: c_int = 0;

// --- opus_defines.h: encoder CTL request constants (encoder_ctl varargs) ---
/// Configures the bitrate in bits per second.
pub const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
/// Enables/disables variable bitrate. We pass `0` for hard CBR.
pub const OPUS_SET_VBR_REQUEST: c_int = 4006;
/// Configures the encoder's computational complexity (0..=10).
pub const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;

// --- opus_defines.h: coding modes (opus_encoder_create `application`) ---
/// Lowest-latency mode (disables the speech/music detector); best for a remote
/// desktop where interactivity dominates.
pub const OPUS_APPLICATION_RESTRICTED_LOWDELAY: c_int = 2051;

extern "C" {
    pub fn opus_encoder_create(
        fs: c_int,
        channels: c_int,
        application: c_int,
        error: *mut c_int,
    ) -> *mut OpusEncoder;

    pub fn opus_encode_float(
        st: *mut OpusEncoder,
        pcm: *const f32,
        frame_size: c_int,
        data: *mut u8,
        max_data_bytes: c_int,
    ) -> c_int;

    /// Variadic in C; every request we issue takes a single `opus_int32` value.
    pub fn opus_encoder_ctl(st: *mut OpusEncoder, request: c_int, ...) -> c_int;

    pub fn opus_encoder_destroy(st: *mut OpusEncoder);

    pub fn opus_decoder_create(fs: c_int, channels: c_int, error: *mut c_int) -> *mut OpusDecoder;

    pub fn opus_decode_float(
        st: *mut OpusDecoder,
        data: *const u8,
        len: c_int,
        pcm: *mut f32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;

    pub fn opus_decoder_destroy(st: *mut OpusDecoder);

    /// Returns a static, NUL-terminated human-readable string for an error code.
    pub fn opus_strerror(error: c_int) -> *const c_char;
}
