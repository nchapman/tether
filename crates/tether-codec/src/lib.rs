//! Codec trait + ffmpeg-backed encoders/decoders.
//!
//! Current backends are hardware-only: VAAPI on Linux, VideoToolbox on
//! macOS, D3D11 (QSV/AMF/NVENC/MF) on Windows, across H.264, HEVC, and AV1.
//! There is no software *encoder* — tether links its own LGPL FFmpeg, which
//! has no libx264/libx265; a `#[cfg(test)]` software *decoder* lives in
//! [`h264`] only to exercise decode-path tests against a committed fixture.
//! New backends land as additional [`Encoder`] / [`Decoder`] impls.
//! Trait shape is cribbed from RustDesk's `EncoderApi`
//! (`libs/scrap/src/common/codec.rs:60`) so the introspection surface
//! (latency hints, bitrate control, HW vs SW detection) is right from
//! day one.

pub mod av_log;
pub mod h264;
pub mod probe;

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
pub mod bitstream_sps;
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
mod encoder_common;

#[cfg(target_os = "linux")]
pub mod vaapi;

#[cfg(target_os = "macos")]
pub mod videotoolbox;

#[cfg(target_os = "macos")]
pub mod macos_interop;

#[cfg(target_os = "linux")]
pub mod vaapi_interop;

#[cfg(target_os = "windows")]
pub mod d3d11;

#[cfg(target_os = "windows")]
pub use probe::build_encoder_d3d11;
pub use probe::{build_decoder, build_encoder, validate_chosen_profile};

// Re-exported so downstream crates can name the payload type on
// [`EncodedPacket::data`] without independently depending on `bytes`
// (and without having to track our pinned version).
pub use bytes;

/// Re-export of [`tether_protocol::GpuResourceGuard`]. The decoder
/// stashes whatever ref-counted handles it needs alive while the
/// renderer reads the surface (typically an `AVFrame` whose `Drop`
/// returns the VAAPI surface to the pool); the renderer can't
/// downcast or inspect.
pub use tether_protocol::GpuResourceGuard as GpuFrameGuard;

/// GOP length used by every H.264 encoder we ship. Long enough that
/// GOP length in seconds. Safety-net cadence for decoder recovery —
/// even if on-demand `ForceIdr` fails (some AMF driver versions
/// silently ignore forced IDR), a natural keyframe arrives within
/// this interval. Two seconds balances recovery latency against
/// keyframe bitrate overhead on mostly-static desktop content.
pub(crate) const GOP_SECONDS: u32 = 2;

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
    /// No GPU codec available — Tether hard-requires hardware
    /// encode/decode. Returned from [`build_encoder`] /
    /// [`build_decoder`] when no hardware backend constructs
    /// successfully. The string carries the user-actionable
    /// diagnostics (which call to `vainfo` to run, which kernel
    /// module to check, etc.).
    #[error("no hardware codec: {0}")]
    NoHardwareCodec(String),
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
///
/// `data` is `bytes::Bytes` so the encoder's output can flow through
/// the fragmenter and into QUIC without per-fragment copies — the
/// fragmenter produces fragment payloads via `Bytes::slice` (refcount
/// bump only). Allocation cost is still one per packet inside
/// `drain_encoder` (libavcodec's `AVPacket::data` is not directly
/// re-exposable as a `Bytes`), but downstream copies are eliminated.
#[derive(Clone, Debug)]
pub struct EncodedPacket {
    pub data: bytes::Bytes,
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
/// — `h264_vaapi`, `h264_videotoolbox`, `hevc_qsv`, etc. (all hardware;
/// the LGPL FFmpeg ships no software video encoder). The
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
    /// zero or more packets — zero on the very first frame while the
    /// encoder buffers headers, multiple on an IDR that emits SPS/PPS
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
    /// `ServerHello` descriptor field — e.g. `"h264_vaapi (Intel)"`,
    /// `"h264_videotoolbox"`, `"hevc_qsv"`. Distinct from
    /// `codec_kind` (which only carries the on-wire codec id) so the
    /// client can show which backend the host actually chose.
    fn name(&self) -> &'static str;

    /// Encode one GPU-resident frame zero-copy. Only HW backends that
    /// can import the platform's external GPU buffer implement this;
    /// the default returns [`CodecError::UnsupportedInputFormat`] so
    /// SW encoders and HW backends without an import path stay honest
    /// about not handling GPU frames.
    ///
    /// The input variant is platform-tagged inside [`GpuEncoderFrame`]
    /// so the trait itself stays cross-platform — macOS (IOSurface) and
    /// Windows (D3D11 keyed-mutex) variants land additively in the enum
    /// without changing the trait shape or the host's dispatch site.
    ///
    /// The host's send loop calls this for `CapturedFrame::Gpu`
    /// variants and falls back to `encode_bgra` (with a CPU readback)
    /// when this returns [`CodecError::UnsupportedInputFormat`].
    fn encode_gpu(
        &mut self,
        _frame: GpuEncoderFrame<'_>,
        _pts: i64,
        _force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>> {
        Err(CodecError::UnsupportedInputFormat)
    }
}

/// Platform-tagged GPU-resident frame input for [`Encoder::encode_gpu`].
/// Variants are cfg-gated internally so the enum is cross-platform but
/// each platform compiles with exactly the variants its capture backends
/// can produce — no catch-all that silently swallows future variants.
/// On a platform with no GPU input backends the enum is uninhabited, so
/// the host can't accidentally construct one and the default trait body
/// is the only callable path.
#[derive(Debug)]
pub enum GpuEncoderFrame<'a> {
    /// DRM_PRIME / DMA-BUF input. Today consumed by [`crate::vaapi`]
    /// via `av_hwframe_map` into a VAAPI surface; the variant is
    /// deliberately not named after the consumer because the same
    /// DMA-BUF can be imported by any Linux GPU-encoder backend
    /// (NVENC via FFmpeg's `hwcontext_cuda` + `cuMemImportFromFd`,
    /// AMF, future Vulkan-video encoders). Capture stays one backend
    /// per OS; the encoder picks how to import.
    #[cfg(target_os = "linux")]
    DmaBuf(&'a DmaBufFrame),
    /// macOS IOSurface from a ScreenCaptureKit-captured `CMSampleBuffer`,
    /// consumed zero-copy by VideoToolbox via FFmpeg's `videotoolbox_enc`
    /// (`AVFrame::data[3]` holds the `CVPixelBufferRef` wrapping the
    /// IOSurface).
    #[cfg(target_os = "macos")]
    IOSurface(&'a IOSurfaceFrame),
    /// Windows D3D11 texture from DXGI Desktop Duplication. Consumed
    /// zero-copy by the encoder via FFmpeg's `d3d11va` hwaccel
    /// (`AVFrame` mapped from the shared `ID3D11Texture2D`).
    #[cfg(target_os = "windows")]
    D3D11Texture(&'a D3D11TextureFrame),
    // Keeps the enum inhabited on platforms where no cfg branch above
    // fires. On platforms that have a real variant this is unreachable
    // by safe code; the encoder's `encode_gpu` default body never
    // matches it.
    #[doc(hidden)]
    _Phantom(std::marker::PhantomData<&'a ()>),
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
            _guard: GpuFrameGuard::new(guard),
        }
    }

    /// Decompose into its fields so a renderer can take the source and
    /// the guard separately. The guard must outlive any GPU work that
    /// reads the surface; the renderer typically parks it next to the
    /// imported wgpu textures and drops both together when the next
    /// frame arrives.
    pub fn into_parts(self) -> (u32, u32, Option<i64>, GpuFrameSource, GpuFrameGuard) {
        (self.width, self.height, self.pts, self.source, self._guard)
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
    /// macOS VideoToolbox-decoded `CVPixelBuffer` (typically with an
    /// IOSurface backing). The renderer imports the IOSurface as a
    /// Metal texture via wgpu's Metal HAL `texture_from_raw` path.
    #[cfg(target_os = "macos")]
    IOSurface(IOSurfaceFrame),
    /// Windows D3D11VA decoded biplanar texture (NV12 8-bit or P010
    /// 10-bit) with a shared NT handle for cross-device import into the
    /// native D3D11 renderer, which opens it on its own device. The
    /// texture is a GPU-side copy from the decoder's pool surface into a
    /// shared-handle-enabled staging texture.
    #[cfg(target_os = "windows")]
    D3D11Texture(D3D11DecodedTexture),
}

/// Decoded D3D11 biplanar frame exported as a single shared NT handle
/// to a staging texture. The consumer opens it and creates two SRVs over
/// the one texture, one per plane (the SRV format selects the plane):
/// `DXGI_FORMAT_NV12` → `R8_UNORM` luma + `R8G8_UNORM` chroma (8-bit);
/// `DXGI_FORMAT_P010` → `R16_UNORM` luma + `R16G16_UNORM` chroma (10-bit
/// MSB-aligned in 16-bit cells).
///
/// A single biplanar texture (not two split-plane textures) is required
/// because the decode→staging copy is `CopySubresourceRegion` per plane,
/// and D3D11 only permits that between same-format subresources — an
/// NV12-plane → R8 copy is silently dropped, leaving the export blank.
/// The handle is owned by `CreateSharedHandle` and stays valid as long
/// as the backing staging texture (held by the decoder) exists.
#[cfg(target_os = "windows")]
#[derive(Debug)]
pub struct D3D11DecodedTexture {
    /// Shared NT handle to the staging texture.
    pub handle: *mut std::ffi::c_void,
    pub width: u32,
    pub height: u32,
    /// `DXGI_FORMAT` of the staging texture (`NV12` for 8-bit, `P010`
    /// for 10-bit). The renderer picks its plane-SRV formats from this
    /// so the import can't drift from what the decoder actually copied.
    pub format: u32,
}

#[cfg(target_os = "windows")]
unsafe impl Send for D3D11DecodedTexture {}

/// DMA-BUF descriptor as returned by `vaExportSurfaceHandle`. Mirrors
/// `tether_vaapi::DrmPrimeSurface` but is owned by `tether-codec` so
/// downstream crates that don't otherwise want a libva dep (notably
/// `tether-render`'s wgpu import path) can stay decoupled from
/// `tether-vaapi`. The fds are `OwnedFd` so close-exactly-once is
/// type-enforced.
///
/// Synchronisation model: the decoder calls `vaSyncSurface` before
/// exporting, so the consumer is guaranteed to see a fully-decoded
/// surface. We don't rely on dma-buf implicit sync via the reservation
/// object — not every libva backend attaches the producer write fence
/// reliably, and the cost of an unconditional `vaSyncSurface` is
/// microseconds when the surface is already done. The Vulkan import on
/// the other side likewise doesn't need an explicit fence; a future
/// backend that wants pipelined GPU-to-GPU sync (e.g. NVDEC over
/// CUDA/Vulkan interop) can add a sync_file fd alongside.
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

/// Build a `DmaBufFrame` matching the P010 (HEVC Main10 / 4:2:0
/// 10-bit) descriptor shape FFmpeg's `vaapi_drm_format_map` expects:
/// one DRM object, two layers — Y as R16 and UV as GR32 (R16G16),
/// both pointing at `object_index=0` with their offsets within the
/// shared allocation.
///
/// Single source of truth shared by the production send loop in
/// `tether-host` and the startup probe in `tether-probe`. A previous
/// version had two copies of this helper; cross-table consistency
/// tests in `tether-render::dmabuf_test` pin the fourcc family.
#[cfg(target_os = "linux")]
#[must_use]
pub fn build_p010_dmabuf_frame(
    fd: std::os::fd::OwnedFd,
    size: u64,
    modifier: u64,
    y_offset: u64,
    y_stride: u64,
    uv_offset: u64,
    uv_stride: u64,
) -> DmaBufFrame {
    DmaBufFrame {
        fourcc: u32::from_le_bytes(*b"P010"),
        objects: vec![DmaBufObject {
            fd,
            size,
            drm_format_modifier: modifier,
        }],
        layers: vec![
            DmaBufLayer {
                drm_format: u32::from_le_bytes(*b"R16 "),
                num_planes: 1,
                object_index: [0, 0, 0, 0],
                offset: [
                    u32::try_from(y_offset).expect("Y plane offset fits in u32"),
                    0,
                    0,
                    0,
                ],
                pitch: [
                    u32::try_from(y_stride).expect("Y plane stride fits in u32"),
                    0,
                    0,
                    0,
                ],
            },
            DmaBufLayer {
                drm_format: u32::from_le_bytes(*b"GR32"),
                num_planes: 1,
                object_index: [0, 0, 0, 0],
                offset: [
                    u32::try_from(uv_offset).expect("UV plane offset fits in u32"),
                    0,
                    0,
                    0,
                ],
                pitch: [
                    u32::try_from(uv_stride).expect("UV plane stride fits in u32"),
                    0,
                    0,
                    0,
                ],
            },
        ],
    }
}

/// Build a `DmaBufFrame` matching the XV30 (HEVC Main 4:4:4 10-bit)
/// descriptor shape FFmpeg's `vaapi_drm_format_map` expects: one DRM
/// object, one layer (DRM_FORMAT_XV30, packed 10:10:10:2), one plane
/// pointing at `object_index=0` with the bridge-reported offset and
/// pitch.
///
/// Same one-source-of-truth contract as [`build_p010_dmabuf_frame`] —
/// the production send loop in `tether-host` and the startup probe in
/// `tether-probe` both call this helper. Cross-table consistency tests
/// in `tether-render::dmabuf_test` pin the fourcc family.
#[cfg(target_os = "linux")]
#[must_use]
pub fn build_xv30_dmabuf_frame(
    fd: std::os::fd::OwnedFd,
    size: u64,
    modifier: u64,
    offset: u64,
    stride: u64,
) -> DmaBufFrame {
    DmaBufFrame {
        fourcc: u32::from_le_bytes(*b"XV30"),
        objects: vec![DmaBufObject {
            fd,
            size,
            drm_format_modifier: modifier,
        }],
        layers: vec![DmaBufLayer {
            drm_format: u32::from_le_bytes(*b"XV30"),
            num_planes: 1,
            object_index: [0, 0, 0, 0],
            offset: [
                u32::try_from(offset).expect("XV30 plane offset fits in u32"),
                0,
                0,
                0,
            ],
            pitch: [
                u32::try_from(stride).expect("XV30 plane stride fits in u32"),
                0,
                0,
                0,
            ],
        }],
    }
}

/// macOS IOSurface descriptor for a captured (encoder input) or
/// decoded (renderer input) frame. Mirrors what
/// `tether_capture::CapturedIOSurface` carries on the capture side;
/// owned by `tether-codec` so downstream crates that don't otherwise
/// want a `tether-capture` dep (notably `tether-render`'s wgpu Metal
/// import path) can stay decoupled — same rationale as
/// [`DmaBufFrame`] vs `CapturedDmaBuf`.
///
/// Lifetime: the encoder borrows this through
/// [`GpuEncoderFrame::IOSurface`] for the duration of one
/// `encode_gpu` call; the capture-side `release_guard` keeps the
/// IOSurface alive across that call. The decoder side owns its
/// `IOSurfaceFrame` and pairs it with a [`GpuFrameGuard`] that
/// retains the `CVPixelBuffer` until the renderer is done.
#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct IOSurfaceFrame {
    /// `IOSurfaceRef` — opaque Apple type. Non-owning: lifetime is the
    /// retaining guard alongside this frame (capture's
    /// `release_guard`, decoder's `GpuFrameGuard`).
    pub surface: *mut std::ffi::c_void,
    /// `kCVPixelFormatType_*` fourcc, e.g. `'420v'` for NV12 video
    /// range. The encoder's `AVHWFramesContext` is configured to
    /// match.
    pub pixel_format: u32,
    pub width: u32,
    pub height: u32,
}

// No `&mut` access is possible to the IOSurface through this raw
// pointer from Rust; all mutation goes through Apple's IOSurface C
// API, which is itself thread-safe (CF-style refcounted, kernel
// surface thread-shareable). The struct carries no Rust state that
// would conflict with crossing a thread boundary.
#[cfg(target_os = "macos")]
unsafe impl Send for IOSurfaceFrame {}

/// Windows D3D11 texture descriptor for encoder input. Carries the
/// raw COM pointer to an `ID3D11Texture2D` on the shared device plus
/// the device/context pointers needed for the encoder to map the
/// texture into an FFmpeg `AVFrame` via `d3d11va` hwaccel.
///
/// Lifetime: the texture stays valid for the duration of the
/// `encode_gpu` call (guarded by the capture-side `release_guard`
/// or the pool's reference semantics). The device and context are
/// `Arc`-shared and outlive any single frame.
#[cfg(target_os = "windows")]
#[derive(Debug)]
pub struct D3D11TextureFrame {
    /// `ID3D11Texture2D` — the BGRA (or NV12) texture to encode.
    /// Non-owning: lifetime is the capture pool or the guard.
    pub texture: *mut std::ffi::c_void,
    /// `ID3D11Device` — shared device between capture and encoder.
    pub device: *mut std::ffi::c_void,
    /// `ID3D11DeviceContext` — immediate context on the shared device.
    pub device_context: *mut std::ffi::c_void,
    pub width: u32,
    pub height: u32,
    /// DXGI_FORMAT value (e.g. `DXGI_FORMAT_B8G8R8A8_UNORM` = 87,
    /// `DXGI_FORMAT_NV12` = 103).
    pub format: u32,
}

#[cfg(target_os = "windows")]
unsafe impl Send for D3D11TextureFrame {}

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

    /// Drain the codec's internal state without tearing the context
    /// down. After `flush`, the decoder is ready to accept a fresh
    /// IDR + subsequent packets as if it had just been constructed —
    /// reference frames, B-frame reorder queue, and pending output
    /// are all discarded.
    ///
    /// Default impl is a no-op so backends that don't need it (and
    /// the test `FakeDecoder`) compile unchanged. Production VAAPI
    /// and VideoToolbox override.
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// First-call hook for any cross-crate ffmpeg setup we want to run
/// exactly once per process. rsmpeg handles codec registration lazily
/// on first use; what we install here is the `av_log` -> tracing
/// bridge so libavcodec messages don't slip through to raw stderr and
/// so the client can observe `av_log::warning_or_above_count` to
/// detect "decoder is stuck" states (e.g. HEVC RPS reconstruction
/// failures) that don't surface as `Err` from the codec API. The
/// `install()` callee is itself `Once`-guarded, so calling
/// `init_ffmpeg` from every codec constructor is cheap on the steady
/// path.
pub(crate) fn init_ffmpeg() {
    av_log::install();
}
