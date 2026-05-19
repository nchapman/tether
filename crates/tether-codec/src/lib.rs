//! Codec trait + ffmpeg-backed encoders/decoders.
//!
//! v0 ships software H.264 only (libx264 via rsmpeg). HW backends
//! (VideoToolbox / VAAPI / NVENC) and additional codecs (HEVC, AV1) will
//! land as additional [`Encoder`] / [`Decoder`] impls — the trait shape
//! is cribbed from RustDesk's `EncoderApi` (`libs/scrap/src/common/codec.rs:60`)
//! so the introspection surface (latency hints, bitrate control, HW vs SW
//! detection) is right from day one.

pub mod h264;
pub mod probe;

pub use h264::{H264Decoder, H264Encoder};
pub use probe::probe_encoder_bgra;

use std::sync::Once;

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("ffmpeg: {0}")]
    Ffmpeg(#[from] rsmpeg::error::RsmpegError),
    #[error("encoder not configured for input format")]
    UnsupportedInputFormat,
    #[error("buffer size mismatch: got {got} bytes, expected {expected}")]
    BufferSizeMismatch { got: usize, expected: usize },
    /// The requested ffmpeg codec wasn't compiled into the linked
    /// FFmpeg build (e.g. asking for `h264_vaapi` on a system whose
    /// FFmpeg was built without VAAPI support).
    #[error("ffmpeg codec '{0}' not available in this build")]
    CodecNotFound(&'static str),
    /// SwsContext construction returned NULL — almost always means an
    /// unsupported source/destination pixel-format pair.
    #[error("ffmpeg swscale init failed ({0})")]
    ScalerInit(&'static str),
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

/// A decoded video frame in YUV 4:2:0 planar (I420) layout. Three tight
/// planes — Y at full resolution, U and V at quarter resolution
/// (subsampled 2:1 on each axis). The renderer uploads each plane as
/// its own `R8Unorm` texture and converts to RGB in the fragment
/// shader, skipping the per-frame CPU YUV→BGRA→RGBA bounce we used
/// to do.
///
/// Dimensions come from the codec (decoded SPS), not from the wire,
/// so the decoder is authoritative about resolution changes.
#[derive(Clone, Debug)]
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    /// Decoder-reported PTS in the codec's time_base; `None` if the
    /// upstream packet didn't carry one.
    pub pts: Option<i64>,
    /// Tight Y plane, `width * height` bytes.
    pub y: Vec<u8>,
    /// Tight U plane, `chroma_width * chroma_height` bytes where
    /// `chroma_width = (width + 1) / 2` (and same for height).
    pub u: Vec<u8>,
    /// Tight V plane, same layout as U.
    pub v: Vec<u8>,
}

impl DecodedFrame {
    /// Chroma plane dimensions for the 4:2:0 subsampling we assume.
    #[must_use]
    pub fn chroma_dims(&self) -> (u32, u32) {
        (self.width.div_ceil(2), self.height.div_ceil(2))
    }
}

/// Pluggable video-encoder backend. One impl per (codec, backend) pair
/// — `libx264` software, `h264_vaapi`, `h264_videotoolbox`, etc. The
/// host probes available backends at startup, picks the best one that
/// constructs successfully, and stores the result as `Box<dyn Encoder>`
/// so the encode loop is backend-agnostic.
///
/// Shape is patterned after RustDesk's `EncoderApi` in
/// `libs/scrap/src/common/codec.rs:60-83`. The introspection methods
/// — `is_hardware`, `supports_changing_bitrate`, `name` — feed the
/// probe and adaptive-bitrate controller without forcing the caller
/// to know the concrete type.
pub trait Encoder: Send {
    /// Encode one BGRA frame at the negotiated width/height. May emit
    /// zero or more packets — zero on the very first frame while
    /// libx264 buffers SPS/PPS, multiple on an IDR that emits SPS/PPS
    /// + the IDR slice as separate packets.
    fn encode_bgra(
        &mut self,
        bgra: &[u8],
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>>;

    /// Whether this encoder can change bitrate at runtime via
    /// `set_bitrate_kbps`. Defaults to `false` — most ffmpeg encoders
    /// need a full restart to retune.
    fn supports_changing_bitrate(&self) -> bool {
        false
    }

    fn set_bitrate_kbps(&mut self, _kbps: u32) -> Result<()> {
        Ok(())
    }

    /// `true` if encoding runs on dedicated silicon (VideoToolbox,
    /// NVENC, VAAPI, AMF). Used by the probe to prefer HW backends.
    fn is_hardware(&self) -> bool {
        false
    }

    fn codec_kind(&self) -> tether_protocol::control::CodecKind;

    /// Short human-readable identifier suitable for logs and the
    /// `ServerHello` descriptor field — e.g. `"libx264 sw"`,
    /// `"h264_vaapi (Intel)"`, `"h264_videotoolbox"`. Distinct from
    /// `codec_kind` (which only carries the on-wire codec id) so the
    /// client can show which backend the host actually chose.
    fn name(&self) -> &'static str;
}

pub trait Decoder: Send {
    fn decode(&mut self, encoded: &[u8]) -> Result<Vec<DecodedFrame>>;
    fn codec_kind(&self) -> tether_protocol::control::CodecKind;
}

/// First-call hook for any cross-crate ffmpeg setup we want to run
/// exactly once per process. rsmpeg-style bindings don't require a
/// separate `av_register_all()` (deprecated in FFmpeg 4.x, removed
/// later) so this is currently a no-op — kept for the call sites and
/// for future hooks (logging callback, lock manager, etc.).
pub(crate) fn init_ffmpeg() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // Intentionally empty for now. rsmpeg handles codec/format
        // registration lazily on first use.
    });
}
