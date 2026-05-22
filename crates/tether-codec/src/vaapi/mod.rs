//! VAAPI hardware H.264 encode + decode for Linux (Intel Arc / AMD /
//! NVIDIA-via-translation).
//!
//! Wraps ffmpeg's `h264_vaapi` encoder and the `h264` decoder's VAAPI
//! hwaccel through rsmpeg's safe `AVHWDeviceContext` /
//! `AVHWFramesContext` / `hwframe_transfer_data` bindings.
//!
//! Encoder pipeline (capture-side, CPU path still pays swscale +
//! upload; the DMA-BUF zero-copy path goes through
//! [`encoder::VaapiEncoder::submit_dmabuf`]):
//!
//!   BGRA &[u8] -> bgra AVFrame -> swscale NV12 sw_frame
//!                              -> av_hwframe_transfer_data -> VAAPI surface
//!                              -> h264_vaapi encode -> H.264 packet
//!
//!   DMA-BUF (NV12) -> DRM_PRIME AVFrame -> av_hwframe_map(DIRECT)
//!                                       -> VAAPI surface in encoder pool
//!                                       -> h264_vaapi encode -> H.264 packet
//!
//! Decoder pipeline (client-side, hot path emits a DMA-BUF straight
//! to the renderer; CPU path is a safety net for unexpected SW
//! fallback):
//!
//!   H.264 bytes -> AVPacket -> h264 (VAAPI hwaccel) -> VAAPI surface
//!                                                   -> vaExportSurfaceHandle
//!                                                   -> DRM_PRIME DmaBufFrame
//!
//! ---
//!
//! Spike result, kept here so the next person doesn't redo the experiment:
//! `h264_vaapi` does *not* accept a `sw_format=BGR0` hwframes pool. We
//! tried that hoping the driver would do implicit BGRA→NV12 at encode
//! time, but on Intel iHD / Meteor Lake / Mesa 26.1.5, `encoder.open()`
//! returns EINVAL with "No usable encoding profile found"; the H.264
//! encoder entrypoint is YUV-only across Intel, AMD, and NVIDIA's
//! nvidia-vaapi-driver shim (NVENC is YUV-only under the hood). The
//! explicit BGRA→NV12 wgpu compute pass in `tether-gpuconvert` is the
//! resulting design: it writes Y (R8) + UV (Rg8) into a shared DMA-BUF
//! that `submit_dmabuf` re-imports via DRM_PRIME → VAAPI.

mod decoder;
mod encoder;
mod ffi;

#[cfg(test)]
mod bench;
#[cfg(test)]
mod tests;

pub use decoder::VaapiDecoder;
pub use encoder::{expected_dmabuf_fourcc, VaapiEncoder};

/// Surfaces beyond what the decoder needs for its own reference picture
/// list. We hand each decoded surface to the renderer and only release
/// the AVFrame ref when the renderer drops the `GpuFrame`. With a
/// 1-deep render channel that's at most 2 surfaces held outside the
/// decoder (one rendering, one in flight in the channel); 4 gives
/// headroom for the brief window where the renderer is sampling the
/// previous surface while the next one lands. Surfaces are cheap to
/// allocate but not free — a 2880×1920 NV12 surface is ~8 MiB of GPU
/// memory, so don't over-provision.
pub(crate) const DECODE_EXTRA_HW_FRAMES: i32 = 4;

/// Number of VAAPI surfaces in the hwframes pool. With `async_depth=1`
/// the encoder reports a packet synchronously after each `send_frame`,
/// so we never have more than one surface in flight. The pool also
/// holds reference frames the encoder needs internally (1 short-term
/// ref with `max_b_frames=0`) plus headroom for the brief windows
/// where VAAPI hardware feedback transiently holds an extra surface
/// before releasing it. Eight is comfortable; tighter values risk
/// `EAGAIN` on `hwframe_ctx_alloc::get_buffer`.
pub(crate) const VAAPI_POOL_SIZE: i32 = 8;
