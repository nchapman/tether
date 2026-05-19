//! Codec trait + ffmpeg-backed encoders/decoders.
//!
//! v0 ships software H.264 only (libx264 via ffmpeg-next). HW backends
//! (VideoToolbox / VAAPI / NVENC) and additional codecs (HEVC, AV1) will
//! land as additional [`Encoder`] / [`Decoder`] impls — the trait shape
//! is cribbed from RustDesk's `EncoderApi` (`libs/scrap/src/common/codec.rs:60`)
//! so the introspection surface (latency hints, bitrate control, HW vs SW
//! detection) is right from day one.

pub mod h264;

pub use h264::{H264Decoder, H264Encoder};

use std::sync::Once;

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("ffmpeg: {0}")]
    Ffmpeg(#[from] ffmpeg_next::Error),
    #[error("encoder not configured for input format")]
    UnsupportedInputFormat,
    #[error("buffer size mismatch: got {got} bytes, expected {expected}")]
    BufferSizeMismatch { got: usize, expected: usize },
}

pub type Result<T> = std::result::Result<T, CodecError>;

/// One encoded video packet (in our case a sequence of one or more
/// concatenated Annex-B-framed NAL units). The wire layer carries this
/// in `VideoPacket::*::payload`.
#[derive(Clone, Debug)]
pub struct EncodedPacket {
    pub data: Vec<u8>,
    /// The encoder's output PTS for this packet, in the time_base it was
    /// configured with. `None` when the encoder didn't set one (rare; mostly
    /// SPS/PPS-only packets before the first frame is fully buffered).
    /// Jitter buffer logic on the receive side relies on this — keep it
    /// plumbed end-to-end.
    pub pts: Option<i64>,
    pub keyframe: bool,
}

/// A decoded video frame in BGRA8 layout. Dimensions come from the codec
/// (decoded SPS), not from the wire — this lets the decoder tell us if
/// the stream silently changed resolution.
#[derive(Clone, Debug)]
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    /// Decoder-reported PTS in the codec's time_base; `None` if the
    /// upstream packet didn't carry one.
    pub pts: Option<i64>,
    pub data: Vec<u8>,
}

pub trait Encoder: Send {
    fn encode_bgra(
        &mut self,
        bgra: &[u8],
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>>;

    /// Whether this encoder can change bitrate at runtime.
    fn supports_changing_bitrate(&self) -> bool {
        false
    }
    fn set_bitrate_kbps(&mut self, _kbps: u32) -> Result<()> {
        Ok(())
    }
    /// True if encoding runs on dedicated HW (VideoToolbox, NVENC, VAAPI).
    fn is_hardware(&self) -> bool {
        false
    }
    fn codec_kind(&self) -> tether_protocol::control::CodecKind;
}

pub trait Decoder: Send {
    fn decode(&mut self, encoded: &[u8]) -> Result<Vec<DecodedFrame>>;
    fn codec_kind(&self) -> tether_protocol::control::CodecKind;
}

/// Run ffmpeg's one-shot initialiser the first time any encoder/decoder
/// is constructed. Idempotent.
pub(crate) fn init_ffmpeg() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // ffmpeg-next 8.x: ffmpeg::init() registers codecs/formats and is
        // safe to call repeatedly, but we still gate it for clarity.
        ffmpeg_next::init().expect("ffmpeg init");
    });
}
