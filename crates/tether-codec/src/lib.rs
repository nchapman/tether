//! Codec trait + ffmpeg-backed encoders/decoders.
//!
//! Current backends: software H.264 (libx264 via rsmpeg) and hardware
//! H.264 via VAAPI on Linux. VideoToolbox / NVENC and additional codecs
//! (HEVC, AV1) land as additional [`Encoder`] / [`Decoder`] impls.
//! Trait shape is cribbed from RustDesk's `EncoderApi`
//! (`libs/scrap/src/common/codec.rs:60`) so the introspection surface
//! (latency hints, bitrate control, HW vs SW detection) is right from
//! day one.

pub mod h264;
pub mod probe;

#[cfg(target_os = "linux")]
pub mod vaapi;

pub use h264::{H264Decoder, H264Encoder};
pub use probe::{probe_decoder, probe_encoder_bgra};

use std::sync::Once;

/// GOP length used by every H.264 encoder we ship. Long enough that
/// keyframes don't dominate the bitrate envelope (1 IDR every ~240
/// frames at 30 fps), short enough that a client joining mid-stream
/// after our handshake's eager-IDR request never has to wait more
/// than this for a periodic recovery point. Loss recovery between
/// IDRs is handled by the client's on-demand `ForceIdr` plumbing,
/// not by GOP cadence.
pub(crate) const GOP_SECONDS: u32 = 8;

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
    /// `vaExportSurfaceHandle` failed. Most commonly: the driver doesn't
    /// support PRIME_2 export for the surface's tiling modifier (rare on
    /// Intel iGPU since Skylake, more common on edge-case AMD configs).
    /// Distinct from `UnsupportedInputFormat` so the host log can
    /// distinguish "decoder produced a frame in a shape we can't read"
    /// from "the kernel/driver refused to share this specific surface."
    #[cfg(target_os = "linux")]
    #[error("vaExportSurfaceHandle failed: {0}")]
    SurfaceExportFailed(#[from] tether_vaapi::VaError),
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

/// A decoded video frame in NV12 (Y plus interleaved UV) layout. Two
/// tight planes — Y at full resolution, UV at half resolution in both
/// dimensions with U and V byte-interleaved (U₀ V₀ U₁ V₁ ...). NV12 is
/// the native output of every hardware H.264 decoder we care about, so
/// the VAAPI path lands its surface straight into this shape with zero
/// CPU pixel-format conversion. The software path does a cheap
/// YUV420P→NV12 interleave (byte permutation, no resampling) so the
/// renderer only has to know about one format.
///
/// The renderer uploads `y` as an `R8Unorm` texture and `uv` as a
/// half-res `Rg8Unorm` texture, then does the limited-range BT.709
/// matrix in the fragment shader. Two texture binds replace the old
/// three; bandwidth-equivalent but one fewer sampler binding and no
/// CPU NV12→YUV420P swscale on the decode hot path.
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
    /// Tight UV plane in NV12 layout, `chroma_width * chroma_height * 2`
    /// bytes where `chroma_width = (width + 1) / 2` (and same for
    /// height). Each chroma sample is two bytes — U byte first, V byte
    /// second — so a single `Rg8` texture sample on the GPU yields
    /// both channels in one read.
    pub uv: Vec<u8>,
}

impl DecodedFrame {
    /// Chroma plane dimensions (in chroma samples) for the 4:2:0
    /// subsampling we assume. The `uv` buffer is twice this wide in
    /// bytes because each sample carries U and V interleaved.
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

/// Output of one decoded frame, either CPU-resident NV12 planes or a
/// GPU-resident surface handle. SW backends always emit `Cpu`. HW
/// backends emit `Gpu` when their renderer-side import path is ready
/// and `Cpu` (via `av_hwframe_transfer_data`) otherwise. The renderer
/// matches on the variant and either uploads (Cpu) or imports (Gpu).
#[derive(Debug)]
pub enum Frame {
    Cpu(DecodedFrame),
    Gpu(GpuFrame),
}

impl Frame {
    pub fn width(&self) -> u32 {
        match self {
            Frame::Cpu(f) => f.width,
            Frame::Gpu(f) => f.width,
        }
    }
    pub fn height(&self) -> u32 {
        match self {
            Frame::Cpu(f) => f.height,
            Frame::Gpu(f) => f.height,
        }
    }
    pub fn pts(&self) -> Option<i64> {
        match self {
            Frame::Cpu(f) => f.pts,
            Frame::Gpu(f) => f.pts,
        }
    }
}

/// GPU-resident decoded frame. Storage shape is in `source`; the
/// platform-tagged variant tells the renderer how to import. The
/// `_guard` field holds whatever ref-counted handles the decoder needs
/// alive while the renderer is reading the surface — for VAAPI that's
/// the `AVFrame` whose `Drop` calls `av_frame_unref` and returns the
/// surface to the pool. The renderer never inspects the guard; it
/// just drops the `GpuFrame` when done.
#[derive(Debug)]
pub struct GpuFrame {
    pub width: u32,
    pub height: u32,
    pub pts: Option<i64>,
    pub source: GpuFrameSource,
    _guard: GpuFrameGuard,
}

impl GpuFrame {
    /// Build a `GpuFrame`. `guard` is anything `Send + 'static` whose
    /// `Drop` releases the decoder-side resources backing `source` —
    /// most commonly an owned `AVFrame`.
    pub fn new<G: Send + 'static>(
        width: u32,
        height: u32,
        pts: Option<i64>,
        source: GpuFrameSource,
        guard: G,
    ) -> Self {
        Self {
            width,
            height,
            pts,
            source,
            _guard: GpuFrameGuard { inner: Box::new(guard) },
        }
    }
}

#[derive(Debug)]
pub enum GpuFrameSource {
    /// Linux DMA-BUF export of a VAAPI surface. Per-platform variants
    /// (VideoToolbox `CVPixelBuffer`, D3D11 texture) will land alongside
    /// their backends. The variant is gated on `target_os` so the
    /// renderer's match is exhaustive on each platform without a
    /// catch-all that silently swallows future variants.
    #[cfg(target_os = "linux")]
    DmaBuf(DmaBufFrame),
}

/// DMA-BUF descriptor as returned by `vaExportSurfaceHandle`. Mirrors
/// `tether_vaapi::DrmPrimeSurface` but is owned by `tether-codec` so
/// downstream crates that don't otherwise want a libva dep (notably
/// `tether-render`'s wgpu import path) can stay decoupled from
/// `tether-vaapi`. The fds are `OwnedFd` so close-exactly-once is
/// type-enforced.
///
/// Synchronisation model: implicit. VAAPI's `vaExportSurfaceHandle`
/// returns surfaces whose dma-buf reservation object carries the
/// decoder's writes, so an importer that respects implicit sync (the
/// Vulkan ext that backs wgpu's external-memory path does) sees a
/// fully-decoded surface without a separate fence import. If a future
/// backend needs explicit sync (e.g. NVDEC, which prefers timeline
/// semaphores), this struct will need a sync_file fd alongside.
#[cfg(target_os = "linux")]
#[derive(Debug)]
pub struct DmaBufFrame {
    pub fourcc: u32,
    pub objects: Vec<DmaBufObject>,
    pub layers: Vec<DmaBufLayer>,
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
pub struct DmaBufObject {
    pub fd: std::os::fd::OwnedFd,
    /// Object size in bytes. `vaExportSurfaceHandle` returns this as
    /// `uint32_t`, but Vulkan's `VkMemoryAllocateInfo::allocationSize`
    /// is `VkDeviceSize` (u64). We widen here so swapping to a backend
    /// or surface format that exceeds 4 GiB doesn't break the type.
    pub size: u64,
    pub drm_format_modifier: u64,
}

#[cfg(target_os = "linux")]
#[derive(Debug, Copy, Clone)]
pub struct DmaBufLayer {
    pub drm_format: u32,
    pub num_planes: u32,
    pub object_index: [u32; 4],
    pub offset: [u32; 4],
    pub pitch: [u32; 4],
}

/// Opaque "hold these refs alive while the renderer reads the surface"
/// container. The inner box is a sealed trait object so different
/// decoders can stash backend-specific lifetimes (e.g. an `AVFrame`)
/// without leaking their crate's types through `Decoder`'s public API
/// — and without advertising `Any`'s downcasting capability, which
/// would invite consumers to depend on the concrete type by convention.
pub struct GpuFrameGuard {
    // Held purely for its Drop.
    #[allow(dead_code)]
    inner: Box<dyn GuardPayload>,
}

/// Sealed marker for anything a backend wants to keep alive for the
/// lifetime of a `GpuFrame`. Blanket-impl'd for every `Send + 'static`
/// type; the trait itself is private so no one outside this crate can
/// implement it or downcast through it.
trait GuardPayload: Send + 'static {}
impl<T: Send + 'static> GuardPayload for T {}

impl std::fmt::Debug for GpuFrameGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuFrameGuard").finish_non_exhaustive()
    }
}

/// Pluggable video-decoder backend. Same probe pattern as `Encoder` —
/// the client probes available backends at startup, picks the best
/// one whose `new()` succeeds, and stores the result as
/// `Box<dyn Decoder>` so the decode loop doesn't care which backend
/// is underneath.
///
/// The split between `submit` and `next_frame` mirrors ffmpeg's
/// `send_packet` / `receive_frame` and lets the renderer drain the
/// B-frame reorder buffer at its own cadence — and, more importantly,
/// lets a future zero-copy backend hand out GPU surfaces one at a time
/// without forcing the decoder to hold the whole batch in a `Vec`.
pub trait Decoder: Send {
    /// Submit one encoded buffer. Does not produce frames directly —
    /// call `next_frame` until it returns `None` to drain anything the
    /// decoder produced.
    fn submit(&mut self, encoded: &[u8]) -> Result<()>;

    /// Pull the next decoded frame, or `Ok(None)` if the decoder needs
    /// more input. Backends may emit zero frames per submit (warming
    /// up on SPS/PPS) or several (B-frame reorder drain on a
    /// configuration that has B-frames — we don't, but the API stays
    /// honest about it).
    fn next_frame(&mut self) -> Result<Option<Frame>>;

    fn codec_kind(&self) -> tether_protocol::control::CodecKind;

    /// `true` if decoding runs on dedicated silicon (VideoToolbox,
    /// NVDEC, VAAPI). Used by the probe to prefer HW backends.
    fn is_hardware(&self) -> bool {
        false
    }

    /// Short human-readable identifier for logs — e.g.
    /// `"libavcodec h264 sw"`, `"h264 (VAAPI hw)"`. Distinct from
    /// `codec_kind` so the client log can show which backend it
    /// actually picked.
    fn name(&self) -> &'static str;
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
