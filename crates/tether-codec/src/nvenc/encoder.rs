//! NVENC hardware video encoder. Codec-parameterized via [`CodecKind`]:
//! H.264 selects `h264_nvenc`, HEVC selects `hevc_nvenc`, AV1 selects
//! `av1_nvenc`. Structurally a twin of [`crate::vaapi::VaapiEncoder`] — same
//! hwframes-pool / swscale-upload / extradata-snapshot shape — with a CUDA
//! device + pool in place of VAAPI, and NVENC's own low-latency AVOption set.
//!
//! Two input paths, as on VAAPI:
//!   * [`Encoder::encode_bgra`] — CPU upload (swscale BGRA → NV12/P010 →
//!     `hwframe_transfer_data` host→CUDA). Used by the test pattern, the
//!     bench harness, and the 8-bit encode probe.
//!   * [`NvencEncoder::submit_dmabuf`] — the zero-copy production path
//!     (capture DMA-BUF imported into CUDA via EGLImage interop, see
//!     [`super::ffi`]). `encode_gpu` routes `DmaBuf` frames here; if the
//!     EGL/CUDA stack is unavailable it reports `NoHardwareCodec` so the
//!     host falls back to the CPU path rather than dropping frames.

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{ra, AVDictionary, AVFrame, AVHWDeviceContext};
use rsmpeg::ffi;
use rsmpeg::swscale::SwsContext;
use rsmpeg::UnsafeDerefMut;
use tracing::warn;

use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

use crate::encoder_common::{drain_encoder, snapshot_extradata};
use crate::h264::frame_plane_mut;
use crate::{init_ffmpeg, CodecError, DmaBufFrame, EncodedPacket, Encoder, Result, GOP_SECONDS};

use super::ffi as nvffi;
use super::NVENC_POOL_SIZE;

pub struct NvencEncoder {
    kind: CodecKind,
    /// Negotiated bit depth (8 or 10). Pinned at construction; gates the
    /// 8-bit-only CPU-upload path in `encode_bgra`.
    bit_depth: u8,
    encoder: AVCodecContext,
    /// BGRA → encoder-input swscale context. Output format matches the
    /// encoder's CUDA `sw_format`; pinned to a single (src, dst) pair at
    /// construction.
    bgra_to_encoder_input: SwsContext,
    sw_frame: AVFrame,
    bgra_frame: AVFrame,
    /// Codec parameter sets (Annex-B SPS/PPS for H.264, VPS+SPS+PPS for
    /// HEVC) captured once after `open()` and prepended to every keyframe
    /// so any IDR is independently decodable. See [`crate::encoder_common`].
    extradata: Vec<u8>,
    // Keep the CUDA device context alive for the encoder's lifetime. Drop
    // order is declaration order, so `encoder` (and its hw_frames_ctx) tear
    // down before `_hw_device` — surfaces freed before the device that
    // allocated them. Same contract as VaapiEncoder.
    _hw_device: AVHWDeviceContext,
    /// Negotiated chroma subsampling. Pinned at construction; gates the
    /// DMA-BUF fourcc the zero-copy path accepts (`expected_nvenc_dmabuf_fourcc`).
    chroma: ChromaSubsampling,
    width: u32,
    height: u32,
    /// FFmpeg's `CUcontext` for this encoder, read once from the CUDA
    /// `AVHWDeviceContext`. The EGL→CUDA import in `submit_dmabuf`
    /// registers against this exact context so the device pointers it
    /// hands NVENC are dereferenceable by the encoder.
    ///
    /// A raw pointer field; covered by the `unsafe impl Send` above (the
    /// CUcontext is owned by `_hw_device`, which is dropped after the
    /// encoder, and only touched under `&mut self`).
    cuda_ctx: nvffi::CuContext,
    bgra_row_bytes: usize,
    /// AVOption keys `open()` left unconsumed. Empty when every requested
    /// option was accepted. Exposed via [`Self::unused_avoptions`] so tests
    /// can confirm a latency knob actually took effect on the live driver.
    unused_avoptions: Vec<String>,
}

// SAFETY: same move-fine / share-bad contract as the other encoders. The
// CUDA device + frames context and the codec context are safe to MOVE
// between threads but not to SHARE; only `&mut self` methods are exposed,
// so the borrow checker serialises access within a single thread.
unsafe impl Send for NvencEncoder {}

impl NvencEncoder {
    /// Construct an NVENC encoder for the given profile at the given
    /// dimensions. `profile.codec` selects `h264_nvenc` / `hevc_nvenc` /
    /// `av1_nvenc`.
    ///
    /// Returns `Err(CodecError::CodecNotFound)` when the linked FFmpeg has
    /// no NVENC build for the codec (the artifact must be built with
    /// `--enable-nvenc --enable-cuda`), and any `RsmpegError` when the CUDA
    /// device can't be created (no NVIDIA GPU / driver, `libcuda.so`
    /// absent) or the codec rejects the configuration (e.g. AV1 on a
    /// pre-Ada card). The probe walks every profile and records which
    /// failed and why; `build_encoder` falls back to VAAPI.
    pub fn new(
        profile: VideoProfile,
        width: u32,
        height: u32,
        fps: u32,
        bitrate_kbps: u32,
    ) -> Result<Self> {
        init_ffmpeg();

        let kind = profile.codec;
        let chroma = profile.chroma;
        let bit_depth = profile.bit_depth;

        // 4:4:4 is deferred: NVENC's native 4:4:4 input is planar
        // (YUV444P / YUV444P16), not the packed VUYX/XV30 the VAAPI
        // DRM_PRIME path uses, so the zero-copy CUDA path needs a planar
        // gpuconvert bridge that doesn't exist yet. Until that lands,
        // map only the 4:2:0 formats; nvenc_sw_format returns
        // UnsupportedInputFormat for 4:4:4 and the probe records it
        // unsupported, so host-authoritative negotiation never picks a
        // profile we'd have to silently downsample.
        let sw_format = nvenc_sw_format(chroma, bit_depth)?;

        let codec_cname = nvenc_codec_cname(kind);
        let codec = AVCodec::find_encoder_by_name(codec_cname)
            .ok_or(CodecError::CodecNotFound(nvenc_codec_name(kind)))?;

        // Default CUDA device (GPU 0). None lets FFmpeg pick; explicit
        // device strings only matter on multi-GPU systems, where the
        // producer (wgpu) and importer (CUDA) must agree on the physical
        // GPU — handled when a multi-GPU user needs it.
        let hw_device = AVHWDeviceContext::create(ffi::AV_HWDEVICE_TYPE_CUDA, None, None, 0)?;

        // Read FFmpeg's CUcontext out of the CUDA AVHWDeviceContext so the
        // zero-copy DMA-BUF import (`submit_dmabuf`) registers EGL images
        // against the exact context NVENC allocates surfaces in. The
        // navigation mirrors the C cast in hwcontext_cuda users:
        //   AVBufferRef::data → AVHWDeviceContext::hwctx → AVCUDADeviceContext::cuda_ctx
        //
        // SAFETY: `create` returned a valid CUDA device-context buffer.
        // `data` points at an `AVHWDeviceContext`, whose `hwctx` points at
        // an `AVCUDADeviceContext` (our `AvCudaDeviceContext`, prefix-ABI
        // pinned by the const asserts in `nvffi`). We only read `cuda_ctx`.
        let cuda_ctx: nvffi::CuContext = unsafe {
            let buf_ref = hw_device.as_ptr();
            let device_ctx = (*buf_ref).data as *const ffi::AVHWDeviceContext;
            let cuda_device_ctx = (*device_ctx).hwctx as *const nvffi::AvCudaDeviceContext;
            (*cuda_device_ctx).cuda_ctx
        };

        let width_i32 = i32::try_from(width).expect("width fits in i32");
        let height_i32 = i32::try_from(height).expect("height fits in i32");
        let fps_i32 = i32::try_from(fps.max(1)).unwrap_or(60);

        let mut encoder = AVCodecContext::new(&codec);
        encoder.set_width(width_i32);
        encoder.set_height(height_i32);
        // pix_fmt is CUDA: pixels live in device memory; the storage layout
        // is `sw_format` on the hwframes context below.
        encoder.set_pix_fmt(ffi::AV_PIX_FMT_CUDA);
        encoder.set_time_base(ra(1, fps_i32));
        encoder.set_framerate(ra(fps_i32, 1));
        encoder.set_bit_rate(i64::from(bitrate_kbps) * 1000);
        // GOP cadence matches the VAAPI path so the on-wire worst-case
        // "garbled until next IDR" window is identical regardless of which
        // backend the probe picked.
        let gop_frames =
            fps_i32.saturating_mul(i32::try_from(GOP_SECONDS).expect("GOP_SECONDS fits in i32"));
        encoder.set_gop_size(gop_frames);
        encoder.set_max_b_frames(0);

        // CUDA hwframes pool. `sw_format` is the on-device pixel layout
        // (NV12 / P010LE); NVENC reads from surfaces allocated here.
        let mut hw_frames_ref = hw_device.hwframe_ctx_alloc();
        hw_frames_ref.data().format = ffi::AV_PIX_FMT_CUDA;
        hw_frames_ref.data().sw_format = sw_format;
        hw_frames_ref.data().width = width_i32;
        hw_frames_ref.data().height = height_i32;
        hw_frames_ref.data().initial_pool_size = NVENC_POOL_SIZE;
        hw_frames_ref.init()?;
        encoder.set_hw_frames_ctx(hw_frames_ref);

        // Color identity in the SPS/VPS VUI. Same rationale as VaapiEncoder:
        // default "Unspecified" lets decoders guess, which shows as a hue
        // shift on saturated content. Pin BT.709 limited-range + centered
        // chroma siting to match the gpuconvert shaders' math.
        //
        // SAFETY: these AVCodecContext fields must be written before
        // avcodec_open2.
        unsafe {
            let raw = encoder.deref_mut();
            raw.color_primaries = ffi::AVCOL_PRI_BT709;
            raw.color_trc = ffi::AVCOL_TRC_BT709;
            raw.colorspace = ffi::AVCOL_SPC_BT709;
            raw.color_range = ffi::AVCOL_RANGE_MPEG;
            raw.chroma_sample_location = ffi::AVCHROMA_LOC_CENTER;
            // Profile pin per codec, mirroring the VAAPI / D3D11 / VideoToolbox
            // siblings — never leave avctx->profile at UNKNOWN. FFmpeg's nvenc.c
            // maps avctx->profile to the NV profile GUID; pinning makes the
            // selection authoritative and agrees with the pix_fmt-derived depth.
            //
            // H.264 gets High here, NOT Main. The VAAPI path pins Main because
            // some Intel/AMD VAAPI entrypoints only advertise VAProfileH264Main
            // and an unpinned High default fails `open()` there — a driver
            // constraint NVENC doesn't share. `h264_nvenc` defaults to High and
            // every decoder we target (VAAPI / VideoToolbox / D3D11VA) handles
            // it, so pinning High is both the safest pin (it's the backend
            // default) and the better one (8×8 transform → tighter compression).
            raw.profile = match (kind, bit_depth) {
                (CodecKind::H264, _) => ffi::AV_PROFILE_H264_HIGH as i32,
                (CodecKind::Hevc, 10) => ffi::AV_PROFILE_HEVC_MAIN_10 as i32,
                (CodecKind::Hevc, _) => ffi::AV_PROFILE_HEVC_MAIN as i32,
                (CodecKind::Av1, _) => ffi::AV_PROFILE_AV1_MAIN as i32,
            };
        }

        // NVENC private options — the low-latency set verified on the
        // Windows D3D11 path (`d3d11/encoder.rs`), kept duplicated inline
        // here rather than shared (the Sunshine pattern: per-backend option
        // sets drift independently). Unlike VAAPI these are read per-frame
        // by FFmpeg's nvenc wrapper:
        //   delay=0      — nvenc defaults output delay to INT_MAX (holds
        //                  packets indefinitely to reorder); 0 emits each
        //                  frame's packet immediately. Non-negotiable for
        //                  realtime.
        //   forced-idr=1 — make AV_PICTURE_TYPE_I produce a real IDR (so
        //                  on-demand IDR requests land), not just an I-frame.
        //   zerolatency=1 + tune=ull — disable lookahead/reordering; the
        //                  ultra-low-latency tuning preset.
        //   rc=cbr       — constant bitrate. Predictable wire rate for the
        //                  congestion controller, and a clean signal for the
        //                  live-retune test (the target is tracked tightly).
        let dict = AVDictionary::new(c"delay", c"0", 0)
            .set(c"forced-idr", c"1", 0)
            .set(c"zerolatency", c"1", 0)
            .set(c"tune", c"ull", 0)
            .set(c"rc", c"cbr", 0);

        // AV_CODEC_FLAG_GLOBAL_HEADER routes parameter sets into
        // `extradata` at open() so we can prepend them to every keyframe.
        #[allow(clippy::cast_possible_wrap)]
        encoder.set_flags(encoder.flags | ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32);
        let leftover = encoder.open(Some(dict))?;
        let mut unused_avoptions: Vec<String> = Vec::new();
        if let Some(unused) = leftover {
            // Encoder didn't recognise one or more opts — not fatal, but
            // surfaced so "why is latency higher than expected?" is
            // diagnosable. Stored as bare keys for presence checks; full
            // key=value pairs go to the warn log.
            let mut unused_pairs: Vec<(String, String)> = Vec::new();
            for entry in unused.iter() {
                let key = entry.key().to_string_lossy().into_owned();
                let value = entry.value().to_string_lossy().into_owned();
                unused_avoptions.push(key.clone());
                unused_pairs.push((key, value));
            }
            if !unused_pairs.is_empty() {
                warn!(
                    codec = nvenc_codec_name(kind),
                    unused = ?unused_pairs,
                    "NVENC encoder ignored some private options; latency knobs may not be applied"
                );
            }
        }

        let scaler_label = crate::encoder_common::pix_fmt_scaler_label(sw_format);
        let mut bgra_to_encoder_input = SwsContext::get_context(
            width_i32,
            height_i32,
            ffi::AV_PIX_FMT_BGRA,
            width_i32,
            height_i32,
            sw_format,
            ffi::SWS_FAST_BILINEAR,
            None,
            None,
            None,
        )
        .ok_or(CodecError::ScalerInit(scaler_label))?;

        // Pin swscale's RGB→YUV matrix to BT.709 limited so the encoded
        // bytes match the VUI written above. Same neutral-defaults call as
        // VaapiEncoder; only the CPU-upload (test/bench/probe) path uses
        // this — the production DMA-BUF path arrives pre-converted.
        //
        // SAFETY: coefficient tables are static, owned by libswscale.
        unsafe {
            let coeffs = ffi::sws_getCoefficients(ffi::SWS_CS_ITU709 as i32);
            let rc = ffi::sws_setColorspaceDetails(
                bgra_to_encoder_input.as_mut_ptr(),
                coeffs,
                1, // src_range: BGRA is full range
                coeffs,
                0, // dst_range: YUV video range (AVCOL_RANGE_MPEG)
                0,
                65536,
                65536,
            );
            if rc != 0 {
                warn!(
                    rc,
                    "sws_setColorspaceDetails refused; BGRA→YUV may fall back to BT.601 for <576-line sources"
                );
            }
        }

        let mut bgra_frame = AVFrame::new();
        bgra_frame.set_format(ffi::AV_PIX_FMT_BGRA);
        bgra_frame.set_width(width_i32);
        bgra_frame.set_height(height_i32);
        bgra_frame.alloc_buffer()?;

        let mut sw_frame = AVFrame::new();
        sw_frame.set_format(sw_format);
        sw_frame.set_width(width_i32);
        sw_frame.set_height(height_i32);
        sw_frame.alloc_buffer()?;

        let bgra_row_bytes = (width as usize) * 4;

        let extradata = snapshot_extradata(&encoder, nvenc_codec_name(kind), kind)?;

        Ok(Self {
            kind,
            bit_depth,
            encoder,
            bgra_to_encoder_input,
            sw_frame,
            bgra_frame,
            extradata,
            _hw_device: hw_device,
            chroma,
            width,
            height,
            cuda_ctx,
            bgra_row_bytes,
            unused_avoptions,
        })
    }

    /// AVOption keys the encoder did not consume at `open()`. A non-empty
    /// list means the corresponding knobs fell back to encoder defaults.
    #[must_use]
    pub fn unused_avoptions(&self) -> &[String] {
        &self.unused_avoptions
    }

    /// The zero-copy production path: import a capture DMA-BUF into CUDA
    /// (EGLImage → `cuGraphicsEGLRegisterImage`) and feed the resulting
    /// device pointers to `*_nvenc` as an `AV_PIX_FMT_CUDA` frame. The
    /// NVENC twin of [`crate::vaapi::VaapiEncoder::submit_dmabuf`].
    ///
    /// `frame.fourcc` must match the negotiated chroma — `NV12` for 4:2:0
    /// 8-bit, `P010` for 4:2:0 10-bit (see [`expected_nvenc_dmabuf_fourcc`]).
    /// 4:4:4 is not accepted here: NVENC wants planar input the gpuconvert
    /// bridge doesn't produce yet (same deferral as `nvenc_sw_format`).
    /// Width/height are pinned to the encoder's construction values;
    /// resolution changes go through a full encoder rebuild.
    ///
    /// Returns `NoHardwareCodec` when the EGL/CUDA stack is unavailable
    /// (no NVIDIA driver / missing EGL device extensions) so the host's
    /// send loop falls back to `encode_bgra` rather than dropping frames.
    // AVFrame::linesize / packet.size are non-negative i32 ABI fields;
    // same rationale as encode_bgra and the VAAPI sibling.
    #[allow(clippy::cast_sign_loss)]
    pub fn submit_dmabuf(
        &mut self,
        frame: &DmaBufFrame,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>> {
        let expected_fourcc = expected_nvenc_dmabuf_fourcc(self.chroma, self.bit_depth)
            .ok_or(CodecError::UnsupportedInputFormat)?;
        if frame.fourcc != expected_fourcc {
            // A fourcc mismatch means a stale dma-buf bridge built for one
            // chroma was fed a frame from the other. Both numbers logged so
            // the root cause is obvious without AV_LOG_DEBUG.
            warn!(
                actual = format_args!("0x{:08x}", frame.fourcc),
                expected = format_args!("0x{:08x}", expected_fourcc),
                chroma = ?self.chroma,
                "imported dma-buf fourcc does not match the negotiated chroma (nvenc)"
            );
            return Err(CodecError::UnsupportedInputFormat);
        }

        // Lazily-initialised, process-global EGL/CUDA importer. `None`
        // means this host can't do the interop at all.
        let importer = nvffi::importer()
            .ok_or_else(|| CodecError::NoHardwareCodec("EGL/CUDA import unavailable".to_string()))?;

        // Import every plane into CUDA. `imported` owns the EGL images +
        // CUDA graphics resources; it must outlive `send_frame` + `drain`
        // so the device pointers stay mapped while NVENC reads them.
        let imported = importer
            .import(self.cuda_ctx, frame, self.width, self.height)
            .map_err(CodecError::NoHardwareCodec)?;

        // Allocate a destination frame from the encoder's own CUDA pool. It
        // is a contiguous NV12/P010 surface with `buf[0]` set — what NVENC's
        // input path requires. NVENC registers ONE pool surface per input
        // frame (a single base pointer + pitch, UV at pitch*height); it can't
        // consume the imported planes as separate external pointers, so we
        // assemble them into this pool frame with device→device copies (the
        // same hop Sunshine makes). A null `buf[0]` (the bare user-pointer
        // frame we'd otherwise build) is exactly what made send_frame EINVAL.
        let mut dst = AVFrame::new();
        self.encoder
            .hw_frames_ctx_mut()
            .expect("hw_frames_ctx set in new")
            .get_buffer(&mut dst)?;

        // Copy each imported plane into the pool frame's matching plane.
        // 4:2:0: plane 0 = full height, plane 1 (interleaved UV) = half
        // height; both rows are `width * bytes_per_sample` bytes wide
        // (NV12 UV packs w/2 CbCr pairs = w bytes; P010 = 2w bytes).
        let bytes_per_sample = if self.bit_depth == 8 { 1 } else { 2 };
        let width_bytes = self.width as usize * bytes_per_sample;
        for i in 0..imported.plane_count() {
            let plane_height = if i == 0 {
                self.height as usize
            } else {
                (self.height as usize).div_ceil(2)
            };
            imported
                .copy_plane_into(
                    i,
                    dst.data[i],
                    dst.linesize[i].max(0) as usize,
                    width_bytes,
                    plane_height,
                )
                .map_err(CodecError::NoHardwareCodec)?;
        }
        // The pixels now live in the pool frame; the EGL/CUDA import
        // resources can be released.
        drop(imported);

        dst.set_pts(pts);
        dst.set_pict_type(if force_keyframe {
            ffi::AV_PICTURE_TYPE_I
        } else {
            ffi::AV_PICTURE_TYPE_NONE
        });

        self.encoder.send_frame(Some(&dst))?;
        drain_encoder(&mut self.encoder, &self.extradata)
    }
}

impl Encoder for NvencEncoder {
    // ffmpeg's i32 ABI fields (linesize, packet.size) are non-negative on
    // allocated frames / valid packets. Same rationale as h264.rs.
    #[allow(clippy::cast_sign_loss)]
    fn encode_bgra(
        &mut self,
        bgra: &[u8],
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>> {
        // CPU-upload path is 8-bit-only by design: the BGRA→sw_format
        // swscale stage here has no 10-bit branch. The gpuconvert P010
        // bridge is the production answer for 10-bit and the probe skips
        // this method for 10-bit. Fail explicitly rather than ship
        // 8-bit-source-as-10-bit. Mirrors VaapiEncoder::encode_bgra.
        if self.bit_depth != 8 {
            return Err(CodecError::UnsupportedInputFormat);
        }
        let height = self.height as usize;
        let expected = self.bgra_row_bytes * height;
        if bgra.len() != expected {
            return Err(CodecError::BufferSizeMismatch {
                got: bgra.len(),
                expected,
            });
        }

        // 1. Copy BGRA into the staging AVFrame, stride-aware.
        {
            let stride = self.bgra_frame.linesize[0] as usize;
            let plane = frame_plane_mut(&mut self.bgra_frame, 0, height);
            if stride == self.bgra_row_bytes {
                plane[..expected].copy_from_slice(bgra);
            } else {
                for row in 0..height {
                    let src = row * self.bgra_row_bytes;
                    let dst = row * stride;
                    plane[dst..dst + self.bgra_row_bytes]
                        .copy_from_slice(&bgra[src..src + self.bgra_row_bytes]);
                }
            }
        }

        // 2. swscale BGRA → encoder input (NV12) into the CPU-side sw_frame.
        self.bgra_to_encoder_input.scale_frame(
            &self.bgra_frame,
            0,
            i32::try_from(height).expect("height fits in i32"),
            &mut self.sw_frame,
        )?;

        // 3. Allocate a CUDA surface from the encoder's pool and upload the
        // sw_frame via hwframe_transfer_data (host→device copy, natively
        // supported by hwcontext_cuda).
        let mut hw_frame = AVFrame::new();
        self.encoder
            .hw_frames_ctx_mut()
            .expect("hw_frames_ctx set in new")
            .get_buffer(&mut hw_frame)?;
        hw_frame.hwframe_transfer_data(&self.sw_frame)?;
        hw_frame.set_pts(pts);
        hw_frame.set_pict_type(if force_keyframe {
            ffi::AV_PICTURE_TYPE_I
        } else {
            ffi::AV_PICTURE_TYPE_NONE
        });

        // 4. Submit + drain.
        self.encoder.send_frame(Some(&hw_frame))?;
        drain_encoder(&mut self.encoder, &self.extradata)
    }

    fn is_hardware(&self) -> bool {
        true
    }

    // `supports_changing_bitrate` / `set_bitrate_kbps` are intentionally
    // NOT overridden yet. NVENC genuinely supports live retune (FFmpeg's
    // nvenc wrapper re-reads bit_rate and reconfigures rate control per
    // frame — the architectural reason NVENC exists alongside VAAPI), but
    // per the project rule we don't advertise a capability until a hardware
    // test proves the bytecount delta on a real NVIDIA GPU. That override +
    // its `nvenc_bitrate_retune_changes_bitstream_size` test land together
    // in a later milestone, once an NVENC-enabled FFmpeg artifact is pinned.

    fn codec_kind(&self) -> CodecKind {
        self.kind
    }

    fn name(&self) -> &'static str {
        nvenc_codec_display_name(self.kind)
    }

    fn encode_gpu(
        &mut self,
        frame: crate::GpuEncoderFrame<'_>,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>> {
        match frame {
            // Zero-copy DMA-BUF → CUDA import (EGLImage interop). Falls
            // back via NoHardwareCodec / UnsupportedInputFormat so the
            // host send loop drops to encode_bgra rather than dropping the
            // frame when interop is unavailable or the chroma is 4:4:4.
            crate::GpuEncoderFrame::DmaBuf(f) => self.submit_dmabuf(f, pts, force_keyframe),
            crate::GpuEncoderFrame::_Phantom(_) => unreachable!("phantom variant"),
        }
    }
}

/// FFmpeg codec name (`CStr` for `find_encoder_by_name`) per codec kind.
fn nvenc_codec_cname(kind: CodecKind) -> &'static std::ffi::CStr {
    match kind {
        CodecKind::H264 => c"h264_nvenc",
        CodecKind::Hevc => c"hevc_nvenc",
        CodecKind::Av1 => c"av1_nvenc",
    }
}

/// Stable encoder name for `CodecNotFound` / logs.
pub(super) fn nvenc_codec_name(kind: CodecKind) -> &'static str {
    match kind {
        CodecKind::H264 => "h264_nvenc",
        CodecKind::Hevc => "hevc_nvenc",
        CodecKind::Av1 => "av1_nvenc",
    }
}

/// Human-readable name for the `Encoder::name` trait method / ServerHello
/// descriptor. The vendor suffix is honest: this backend only constructs on
/// a confirmed-NVIDIA host (see [`super::nvidia_gpu_present`]).
fn nvenc_codec_display_name(kind: CodecKind) -> &'static str {
    match kind {
        CodecKind::H264 => "h264_nvenc (NVIDIA)",
        CodecKind::Hevc => "hevc_nvenc (NVIDIA)",
        CodecKind::Av1 => "av1_nvenc (NVIDIA)",
    }
}

/// DRM fourcc the zero-copy DMA-BUF path expects for a negotiated
/// `(chroma, bit_depth)` — the **surface-level** fourcc gpuconvert's
/// NV12/P010 bridges stamp on the frame they hand the encoder.
///
/// 4:2:0 only, matching [`nvenc_sw_format`]: `NV12` (8-bit) / `P010`
/// (10-bit). 4:4:4 returns `None` (deferred — no planar gpuconvert
/// bridge yet), which `submit_dmabuf` maps to `UnsupportedInputFormat`.
/// Same `(chroma, bit_depth) → fourcc` relation as the VAAPI sibling's
/// [`crate::vaapi::expected_dmabuf_fourcc`] for the 4:2:0 rows.
pub(super) fn expected_nvenc_dmabuf_fourcc(
    chroma: ChromaSubsampling,
    bit_depth: u8,
) -> Option<u32> {
    Some(match (chroma, bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => u32::from_le_bytes(*b"NV12"),
        (ChromaSubsampling::Yuv420, 10) => u32::from_le_bytes(*b"P010"),
        _ => return None,
    })
}

/// FFmpeg CUDA `sw_format` (and matching BGRA→YUV swscale destination /
/// staging `sw_frame`) for a negotiated `(chroma, bit_depth)`.
///
/// 4:2:0 only for now: NV12 (8-bit) / P010LE (10-bit) match what the
/// gpuconvert NV12/P010 bridges already produce, so the zero-copy DMA-BUF
/// import can reuse them. 4:4:4 is deferred — NVENC wants planar
/// `YUV444P`/`YUV444P16`, not the packed VUYX/XV30 of the VAAPI path, which
/// needs a new gpuconvert bridge. Unmapped tuples return
/// `UnsupportedInputFormat` so the profile is recorded unsupported rather
/// than silently mis-encoded.
pub(super) fn nvenc_sw_format(chroma: ChromaSubsampling, bit_depth: u8) -> Result<i32> {
    Ok(match (chroma, bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => ffi::AV_PIX_FMT_NV12,
        (ChromaSubsampling::Yuv420, 10) => ffi::AV_PIX_FMT_P010LE,
        _ => return Err(CodecError::UnsupportedInputFormat),
    })
}
