//! Windows D3D11-accelerated video encoder via FFmpeg's `d3d11va`
//! hardware device. Supports multiple backend encoders (Media
//! Foundation `h264_mf`/`hevc_mf`, NVENC `h264_nvenc`/`hevc_nvenc`,
//! AMF `h264_amf`/`hevc_amf`) — the first one that successfully
//! opens wins. All share the same D3D11 device for zero-copy texture
//! input from DXGI Desktop Duplication.

pub mod decoder;
pub mod encoder;
#[cfg(test)]
mod tests;
mod video_processor;

pub use decoder::D3D11Decoder;
pub use encoder::D3D11Encoder;
