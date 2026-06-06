//! NVDEC hardware video decoder (Linux / NVIDIA).
//!
//! The decode-side twin of [`crate::nvenc`], parallel to [`crate::vaapi`]'s
//! `VaapiDecoder`. Selects FFmpeg's generic `h264` / `hevc` / `av1` decoder
//! with a CUDA `AVHWDeviceContext` attached and a `get_format` callback that
//! pins `AV_PIX_FMT_CUDA` — that combination engages the NVDEC hwaccel
//! (FFmpeg routes through `hwaccels[].pix_fmt == AV_PIX_FMT_CUDA`), so no
//! `*_cuvid` decoder name is needed. The encoder side uses the same
//! "regular decoder + hw_device_ctx" idiom in reverse.
//!
//! Milestone D2 (GitHub #16): zero-copy. NVDEC decodes to an
//! `AV_PIX_FMT_CUDA` surface; the decoder then EGL-imports a pooled dma-buf
//! surface (NV12 for 8-bit 4:2:0, P010 for 10-bit 4:2:0, diagnostic YUV444P
//! for 8-bit 4:4:4) into CUDA and `cuMemcpy2D`s the decoded planes into it
//! device→device — the exact reverse of the encoder's [`crate::nvenc`]
//! DMA-BUF→CUDA copy, reusing the same EGL→CUDA importer
//! ([`crate::nvenc::ffi`]). The renderer imports the resulting dma-buf
//! directly ([`crate::GpuFrameSource::DmaBuf`]), the same handoff the VAAPI
//! decoder uses — no host-memory round-trip. (D1 downloaded the CUDA surface
//! to a [`crate::Frame::Cpu`]; that interim is gone.)
//!
//! Requires an FFmpeg built with NVDEC/cuvid support (the `8.1.0-tether.5`
//! artifact has it) and the runtime `libnvcuvid.so` / `libcuda.so` present,
//! plus a Vulkan driver exposing `VK_EXT_external_memory_dma_buf` +
//! `VK_EXT_image_drm_format_modifier` on the NVIDIA GPU to back the surface
//! pool (allocated with raw `ash` — see [`surface_pool`] for why it can't
//! reuse `tether-gpuconvert`'s exporter). Construction returns a clean
//! [`crate::CodecError`] when any is missing.

pub use decoder::NvdecDecoder;

mod decoder;
mod surface_pool;
