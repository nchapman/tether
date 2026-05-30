//! Windows D3D11 video encode + decode via FFmpeg's `d3d11va` hardware
//! device. The encoder backend is selected by GPU vendor (Intel → QSV,
//! AMD → AMF, NVIDIA → NVENC, unknown → Media Foundation fallback); see
//! `encoder::backends_for_vendor`. The decoder uses the generic `h264` /
//! `hevc` D3D11VA hwaccel. Encode takes zero-copy texture input from DXGI
//! Desktop Duplication; decode exports shared NT handles to the native
//! D3D11 renderer.

pub mod decoder;
pub mod encoder;
#[cfg(test)]
mod tests;
mod video_processor;

pub use decoder::{expected_decode_dxgi_format, D3D11Decoder};
pub use encoder::D3D11Encoder;
