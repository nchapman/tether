//! VideoToolbox hardware video encoder. Codec-parameterized via
//! [`CodecKind`]: H.264 selects `h264_videotoolbox`, HEVC selects
//! `hevc_videotoolbox`. Shares the same `Encoder` trait shape as the
//! VAAPI sibling so the host's send loop is backend-agnostic.

use std::ptr;

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{ra, AVFrame, AVHWDeviceContext};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use rsmpeg::swscale::SwsContext;
use rsmpeg::UnsafeDerefMut;

use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

use crate::encoder_common::{drain_encoder, snapshot_extradata};
use crate::h264::frame_plane_mut;
use crate::{init_ffmpeg, CodecError, EncodedPacket, Encoder, IOSurfaceFrame, Result, GOP_SECONDS};

use super::ffi::{
    kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2,
    kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2,
    kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, CFRelease,
    CVBufferSetAttachment, CVPixelBufferCreateWithIOSurface, CVPixelBufferRef,
    K_CV_ATTACHMENT_MODE_SHOULD_PROPAGATE, K_CV_RETURN_SUCCESS,
};

/// Pool size for the VideoToolbox surface pool. With `async_depth=1`
/// (synchronous send/receive) we never have more than one VT frame in
/// flight at a time; a few extra surfaces absorb the brief windows
/// where the encoder transiently holds an extra reference. Matches the
/// VAAPI sibling's headroom.
const VT_POOL_SIZE: i32 = 8;

pub struct VideoToolboxEncoder {
    kind: CodecKind,
    /// Negotiated chroma sampling. Pinned at construction; the encoder
    /// holds an FFmpeg pix_fmt that matches.
    chroma: ChromaSubsampling,
    /// Negotiated bit depth (8 or 10 in practice). VideoToolbox doesn't
    /// expose 12-bit encode for anything we care about; the probe layer
    /// surfaces a clean error if the wrapper refuses other combos.
    bit_depth: u8,
    encoder: AVCodecContext,
    bgra_to_sw: SwsContext,
    sw_frame: AVFrame,
    bgra_frame: AVFrame,
    /// Annex-B SPS/PPS (and VPS for HEVC) captured at `open()` via
    /// `AV_CODEC_FLAG_GLOBAL_HEADER`. Prepended to every keyframe so
    /// each IDR is self-decodable; clients that join mid-session or
    /// rebuild their decoder don't have to wait for the next periodic
    /// IDR to recover. Empty only if libavcodec refused to populate
    /// extradata at open() (warned at construction).
    // pub(super): exposed for hardware tests in `tests.rs` only, not
    // part of the type's public contract.
    pub(super) extradata: Vec<u8>,
    // Keep the device context alive for the encoder's lifetime. Drop
    // order: `encoder` (and its `hw_frames_ctx`) tear down before
    // `_hw_device`, which is the correct order — surfaces must be freed
    // before the device that allocated them.
    _hw_device: AVHWDeviceContext,
    width: u32,
    height: u32,
    bgra_row_bytes: usize,
}

// SAFETY: same contract as `VaapiEncoder` — the AVCodecContext is fine
// to *move* between threads but unsafe to share. `&mut self` methods
// serialise access within a thread; the `unsafe impl Send` matches the
// move-fine / share-bad shape of every encoder we ship.
unsafe impl Send for VideoToolboxEncoder {}

impl VideoToolboxEncoder {
    /// Construct a VideoToolbox encoder for the given video profile at
    /// the given dimensions.
    ///
    /// Returns `Err(CodecError::CodecNotFound)` if the linked FFmpeg
    /// wasn't built with VideoToolbox support for the requested codec
    /// (Homebrew's `ffmpeg` formula enables `--enable-videotoolbox`
    /// by default); `Err(CodecError::UnsupportedInputFormat)` if no
    /// FFmpeg pix_fmt matches the requested `(chroma, bit_depth)`;
    /// any `RsmpegError` if the VT device fails to open or the FFmpeg
    /// wrapper rejects the configuration. The probe layer treats every
    /// failure path equivalently as `encode=false`, so an empirical
    /// "VT silently does/doesn't support Main10/Main444/etc." answer
    /// falls out of attempting `encoder.open()` rather than from any
    /// table this code maintains.
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
        let sw_format = vt_sw_format(chroma, bit_depth)?;

        let codec_cname = vt_codec_cname(kind)?;
        let codec = AVCodec::find_encoder_by_name(codec_cname)
            .ok_or(CodecError::CodecNotFound(vt_codec_name(kind)))?;

        let hw_device =
            AVHWDeviceContext::create(ffi::AV_HWDEVICE_TYPE_VIDEOTOOLBOX, None, None, 0)?;

        let width_i32 = i32::try_from(width).expect("width fits in i32");
        let height_i32 = i32::try_from(height).expect("height fits in i32");
        let fps_i32 = i32::try_from(fps.max(1)).unwrap_or(60);

        let mut encoder = AVCodecContext::new(&codec);
        encoder.set_width(width_i32);
        encoder.set_height(height_i32);
        // pix_fmt is VIDEOTOOLBOX: pixels live in a CVPixelBuffer; the
        // sw_format on the hwframes context describes the in-memory
        // layout VT and FFmpeg's `*_videotoolbox` wrapper agree on.
        // For (Yuv420, 8) that's NV12 — what SCK hands us via IOSurface
        // and what swscale targets from BGRA. Higher-bit / 4:4:4 paths
        // pick P010 / NV24 / P410 (see `vt_sw_format`); whether the
        // wrapper actually accepts those is what the probe surfaces.
        encoder.set_pix_fmt(ffi::AV_PIX_FMT_VIDEOTOOLBOX);
        encoder.set_time_base(ra(1, fps_i32));
        encoder.set_framerate(ra(fps_i32, 1));
        let bitrate_bps = i64::from(bitrate_kbps) * 1000;
        encoder.set_bit_rate(bitrate_bps);
        // Match Sunshine's baseline rate-control shape for low-latency
        // streaming: set an explicit peak equal to the average bitrate
        // and a one-frame VBV budget. Supplying only `bit_rate` leaves
        // VideoToolbox's FFmpeg wrapper to infer buffering policy, which
        // can smear detail on desktop content while still hitting the
        // nominal average. This is a static construction-time baseline;
        // live bitrate retune remains deliberately unwired here.
        unsafe {
            let raw = encoder.deref_mut();
            raw.rc_max_rate = bitrate_bps;
            raw.rc_buffer_size =
                i32::try_from((bitrate_bps / i64::from(fps_i32)).max(1)).unwrap_or(i32::MAX);
        }
        let gop_frames =
            fps_i32.saturating_mul(i32::try_from(GOP_SECONDS).expect("GOP_SECONDS fits in i32"));
        encoder.set_gop_size(gop_frames);
        encoder.set_max_b_frames(0);

        // AV_CODEC_FLAG_GLOBAL_HEADER routes parameter sets (SPS/PPS,
        // and VPS for HEVC) into `AVCodecContext::extradata` at open()
        // rather than emitting them only in band with the first IDR.
        // We then prepend extradata to every keyframe at drain time so
        // any IDR is independently decodable — required for clients
        // that join mid-session, rebuild their decoder on device loss,
        // or lose the session's first IDR on the wire. Parity with the
        // VAAPI sibling (see `vaapi/encoder.rs` AV_CODEC_FLAG_GLOBAL_HEADER).
        #[allow(clippy::cast_possible_wrap)]
        encoder.set_flags(encoder.flags | ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32);

        let mut hw_frames_ref = hw_device.hwframe_ctx_alloc();
        hw_frames_ref.data().format = ffi::AV_PIX_FMT_VIDEOTOOLBOX;
        hw_frames_ref.data().sw_format = sw_format;
        hw_frames_ref.data().width = width_i32;
        hw_frames_ref.data().height = height_i32;
        hw_frames_ref.data().initial_pool_size = VT_POOL_SIZE;
        hw_frames_ref.init()?;
        encoder.set_hw_frames_ctx(hw_frames_ref);

        // Color identity. Must be written before avcodec_open2 so the
        // SPS VUI (and HEVC equivalent) records them. Without this,
        // libavcodec leaves all four "Unspecified" and decoders guess —
        // typically BT.709 at HD, BT.601 below, which silently produces
        // a hue shift right at the 720p resolution boundary. The macOS
        // host is NV12 video-range only ('420v' from ScreenCaptureKit),
        // so pin to BT.709 limited unconditionally. Parity with the
        // VAAPI encoder.
        //
        // SAFETY: AVCodecContext color fields must be set before
        // avcodec_open2 — same invariant as the global-header flag.
        unsafe {
            let raw = encoder.deref_mut();
            raw.color_primaries = ffi::AVCOL_PRI_BT709;
            raw.color_trc = ffi::AVCOL_TRC_BT709;
            raw.colorspace = ffi::AVCOL_SPC_BT709;
            raw.color_range = ffi::AVCOL_RANGE_MPEG;
            // Chroma siting. Capture is NV12 from ScreenCaptureKit,
            // which Apple documents as centred chroma (matches what
            // gpuconvert produces on Linux for the same shader).
            // AVCHROMA_LOC_CENTER → HEVC SPS chroma_sample_loc_type=1.
            // Parity with the VAAPI sibling; without this the SPS
            // VUI defaults to type-0 and decoders apply a half-pixel
            // chroma offset on saturated edges.
            raw.chroma_sample_location = ffi::AVCHROMA_LOC_CENTER;

            // ALWAYS pin an explicit profile — never rely on VideoToolbox's
            // default. With `avctx->profile` unset, FFmpeg's videotoolboxenc
            // (`get_vt_{h264,hevc}_profile_level`) leaves the profile-level
            // to VT's auto-selection (H.264 and 8-bit HEVC) or infers Main10
            // only from `bit_depth==10`. That non-determinism is the same
            // class of bug the D3D11 path hit (an unpinned backend picking a
            // profile a downstream decoder rejects). Pin every case; the
            // mapping matches the VAAPI sibling (`vaapi/encoder.rs`):
            //   H.264          → Main    (kVTProfileLevel_H264_Main)
            //   HEVC 4:2:0  8b → Main    (…_HEVC_Main)
            //   HEVC 4:2:0 10b → Main10  (…_HEVC_Main10)
            //   HEVC 4:4:4     → REXT    (…_HEVC_Main42210)
            // videotoolboxenc maps all four (HARD-ERRORS on an unsupported
            // profile — surfaced cleanly by the probe as Construct, never a
            // silent downsample). Verified against FFmpeg 8.1 videotoolboxenc.c.
            //
            // IMPORTANT (4:4:4): VT exposes NO Main444 profile-level — REXT
            // maps to `kVTProfileLevel_HEVC_Main42210_AutoLevel`, which is
            // **4:2:2**, not 4:4:4 (see docs/CODEC_CAPABILITIES.md: forcing
            // `rext` turns the failure from 4:2:0 into `x422`). So this arm
            // does NOT make HEVC 4:4:4 encode work — the end-to-end chroma-
            // survival probe still rejects 4:4:4 on Apple Silicon, and the
            // host never advertises it. The arm exists only so the pin is
            // exhaustive and reads in parity with VAAPI; it fires only during
            // a probe attempt that is expected to fail. Only the profiles
            // `vt_codec_cname` accepts (H.264/HEVC) reach here; AV1 bails
            // earlier (the AV1 arm below is unreachable, present for
            // exhaustiveness).
            raw.profile = match (kind, chroma, bit_depth) {
                (CodecKind::H264, _, _) => ffi::AV_PROFILE_H264_MAIN as i32,
                (CodecKind::Hevc, ChromaSubsampling::Yuv444, _) => ffi::AV_PROFILE_HEVC_REXT as i32,
                (CodecKind::Hevc, _, 10) => ffi::AV_PROFILE_HEVC_MAIN_10 as i32,
                (CodecKind::Hevc, _, _) => ffi::AV_PROFILE_HEVC_MAIN as i32,
                // Unreachable: `vt_codec_cname(Av1)` rejects AV1 before
                // construction reaches here. Pinned for match exhaustiveness.
                (CodecKind::Av1, _, _) => ffi::AV_PROFILE_AV1_MAIN as i32,
            };
        }

        // VideoToolbox-specific private options. The defaults are
        // tuned for file-based transcoding; we override for realtime:
        //   realtime=1 — bias the encoder toward low-latency mode.
        //     Sunshine and OBS both set this; the cost is a slight
        //     hit to compression efficiency that's a clear win at
        //     60 fps interactive workloads.
        //   allow_sw=0 — refuse to fall back to a software encoder
        //     if the hardware path can't be opened. We hard-require
        //     hardware encode across the project; the probe layer
        //     surfaces a clean `NoHardwareCodec` error instead.
        //
        // Sunshine also sets `prio_speed=1` and HEVC `max_ref_frames=1`
        // for its game-streaming profile. We leave both unset here while
        // chasing macOS softness: `prio_speed` maps to VT's
        // speed-over-quality bias, and this Apple Silicon device reports
        // `max_ref_frames` unsupported anyway.
        //
        // Note: VideoToolbox doesn't expose an `idr_interval`-style
        // knob the way VAAPI does. We bound the IDR cadence via
        // `gop_size` above and drive on-demand IDRs through the
        // `AVPicture::AV_PICTURE_TYPE_I` pict_type on individual
        // input frames (same channel as the VAAPI path).
        let dict =
            rsmpeg::avutil::AVDictionary::new(c"realtime", c"1", 0).set(c"allow_sw", c"0", 0);
        let leftover = encoder.open(Some(dict))?;
        // The "GPU or nothing" invariant rides on `allow_sw=0` actually
        // being consumed by the encoder. If a build of FFmpeg silently
        // ignores it (older versions, or a custom build with a stale
        // option table), the encoder may fall back to SW without us
        // noticing. Treat any unconsumed `allow_sw` as a hard error —
        // the diagnostic is more actionable than discovering it via a
        // CPU-bound encode loop in production. We log other unused
        // options as warnings (defensive parity with the VAAPI path).
        if let Some(unused) = leftover {
            let mut allow_sw_ignored = false;
            let mut other_unused: Vec<String> = Vec::new();
            for entry in unused.iter() {
                let key = entry.key().to_string_lossy().into_owned();
                let val = entry.value().to_string_lossy().into_owned();
                if key == "allow_sw" {
                    allow_sw_ignored = true;
                } else {
                    other_unused.push(format!("{key}={val}"));
                }
            }
            if !other_unused.is_empty() {
                tracing::warn!(
                    codec = vt_codec_name(kind),
                    unused = ?other_unused,
                    "VideoToolbox encoder ignored some private options"
                );
            }
            if allow_sw_ignored {
                return Err(CodecError::NoHardwareCodec(format!(
                    "VideoToolbox encoder for {kind:?} ignored `allow_sw=0` — \
                     this build of FFmpeg may not enforce the GPU-only contract. \
                     Verify with `ffmpeg -h encoder={}` that `allow_sw` is listed \
                     under private options.",
                    vt_codec_name(kind)
                )));
            }
        }

        let scaler_label = crate::encoder_common::pix_fmt_scaler_label(sw_format);
        let mut bgra_to_sw = SwsContext::get_context(
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
        // bytes match the VUI we wrote above. Default behaviour picks
        // BT.601 for sources shorter than 576 lines and BT.709 above —
        // a silent hue shift right at 720p that the round-trip test
        // (320×240) would otherwise hit.
        //
        // SAFETY: sws_setColorspaceDetails takes the SwsContext pointer
        // plus two coefficient tables returned by sws_getCoefficients.
        // The tables are static lookups owned by libswscale; we never
        // free them. brightness=0, contrast/saturation=65536 is the
        // documented neutral default.
        unsafe {
            let coeffs = ffi::sws_getCoefficients(ffi::SWS_CS_ITU709 as i32);
            let rc = ffi::sws_setColorspaceDetails(
                bgra_to_sw.as_mut_ptr(),
                coeffs,
                1, // src_range: BGRA is full range
                coeffs,
                0, // dst_range: YUV video range (matches AVCOL_RANGE_MPEG)
                0,
                65536,
                65536,
            );
            if rc != 0 {
                tracing::warn!(
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

        // Snapshot the parameter-set bundle libavcodec just wrote into
        // `extradata` so we can prepend it to every keyframe at drain
        // time. See `encoder_common` for the contract — VideoToolbox does
        // not emit in-band parameter sets, so an empty extradata here is
        // fatal (would break self-decodable IDRs).
        let extradata = snapshot_extradata(&encoder, vt_codec_name(kind), kind, false)?;

        Ok(Self {
            kind,
            chroma,
            bit_depth,
            encoder,
            bgra_to_sw,
            sw_frame,
            bgra_frame,
            extradata,
            _hw_device: hw_device,
            width,
            height,
            bgra_row_bytes,
        })
    }

    /// Encode one IOSurface-backed frame without a CPU upload. We wrap
    /// the IOSurface in a fresh `CVPixelBufferRef` and feed it to
    /// `h264_videotoolbox` via the AVFrame `data[3]` slot. The
    /// `AVBufferRef` we attach owns the +1 retain of the CVPixelBuffer
    /// — FFmpeg's `av_frame_unref` releases it when the frame's last
    /// reference drops.
    ///
    /// The caller's `IOSurfaceFrame` is borrowed: the apple-cf side
    /// keeps the IOSurface alive via the `GpuCapturedGuard` on the
    /// originating `CapturedFrame`. `CVPixelBufferCreateWithIOSurface`
    /// internally retains the IOSurface, so we don't need to extend
    /// the guard's lifetime past this call.
    pub fn submit_iosurface(
        &mut self,
        frame: &IOSurfaceFrame,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>> {
        // Fourcc cross-check: the IOSurface's pixel format must match
        // the encoder's configured `sw_format` family. A
        // mismatched-format IOSurface wrapped in a CVPixelBuffer would
        // silently mis-encode (the FFmpeg VT hwaccel reads plane
        // dimensions from the encoder's `sw_format`, not from the
        // CVPixelBuffer's runtime format). Refuse and let the caller
        // route through `encode_bgra` rather than ship corrupted
        // video.
        //
        // Allowed families:
        //   - (Yuv420, 8)  → `'420v'` / `'420f'` (NV12)
        //   - (Yuv420, 10) → `'x420'` / `'xf20'` (P010-family video
        //                    range / full range)
        //   - (Yuv444, 8)  → `'444v'` / `'444f'` (NV24)  *
        //   - (Yuv444, 10) → `'x444'` / `'xf44'` (P410-family)        *
        //
        // (*) the encoder probe currently rejects 4:4:4 on macOS due
        // to a VT silent-downsample bug, so those rows aren't
        // exercised by a negotiated session today — they're listed
        // for symmetry with `vt_sw_format`.
        if !iosurface_fourcc_matches(self.chroma, self.bit_depth, frame.pixel_format) {
            tracing::debug!(
                encoder_chroma = ?self.chroma,
                encoder_bit_depth = self.bit_depth,
                iosurface_fourcc = format_args!("0x{:08x}", frame.pixel_format),
                "IOSurface fourcc does not match encoder sw_format; refusing zero-copy submit"
            );
            return Err(CodecError::UnsupportedInputFormat);
        }
        if frame.width != self.width || frame.height != self.height {
            return Err(CodecError::UnsupportedInputFormat);
        }
        if frame.surface.is_null() {
            return Err(CodecError::UnsupportedInputFormat);
        }

        // 1. Wrap the IOSurface in a fresh CVPixelBuffer. Apple's
        // allocator default + null attributes is the standard call
        // shape for the zero-copy path.
        let mut pixbuf: CVPixelBufferRef = ptr::null_mut();
        // SAFETY: surface is non-null (checked above), pixbuf is a
        // valid out-pointer, the allocator/attributes are documented
        // as nullable for "use the default".
        let rc = unsafe {
            CVPixelBufferCreateWithIOSurface(ptr::null(), frame.surface, ptr::null(), &mut pixbuf)
        };
        if rc != K_CV_RETURN_SUCCESS || pixbuf.is_null() {
            return Err(CodecError::Ffmpeg(RsmpegError::from(ffi::AVERROR_EXTERNAL)));
        }

        // 1a. Pin BT.709 limited-range color attachments on the
        // wrapped buffer. `CVPixelBufferCreateWithIOSurface` carries
        // forward whatever the IOSurface was tagged with; on
        // pre-Sequoia macOS, ScreenCaptureKit has been observed to
        // attach BT.601 for sub-720p capture regions. The VT encoder
        // does not re-matrix the IOSurface — it consumes the bytes
        // as-is and just writes the `color_*` fields we set in
        // `new()` into the SPS VUI. Without these attachments, a
        // small-region capture would ship as BT.601-encoded bytes
        // tagged as BT.709, decoding with crushed greens and shifted
        // skin tones. The encoder VUI is hardcoded ITU_R_709 in
        // `new()`; mirror it here so the input and the tag agree.
        //
        // SAFETY: pixbuf is non-null (checked above). The key/value
        // arguments are `CV_NONNULL` in the SDK header; the dynamic
        // linker resolves both from CoreVideo's data segment at load
        // time, so the static pointers we hand in are always non-null
        // by the time any code in this function runs (a missing symbol
        // would have failed dyld load, not this call).
        unsafe {
            CVBufferSetAttachment(
                pixbuf,
                kCVImageBufferYCbCrMatrixKey,
                kCVImageBufferYCbCrMatrix_ITU_R_709_2,
                K_CV_ATTACHMENT_MODE_SHOULD_PROPAGATE,
            );
            CVBufferSetAttachment(
                pixbuf,
                kCVImageBufferColorPrimariesKey,
                kCVImageBufferColorPrimaries_ITU_R_709_2,
                K_CV_ATTACHMENT_MODE_SHOULD_PROPAGATE,
            );
            CVBufferSetAttachment(
                pixbuf,
                kCVImageBufferTransferFunctionKey,
                kCVImageBufferTransferFunction_ITU_R_709_2,
                K_CV_ATTACHMENT_MODE_SHOULD_PROPAGATE,
            );
        }

        // 2. Wrap the +1 retained CVPixelBufferRef in an AVBufferRef
        // so the AVFrame owns its lifetime. Free callback CFReleases
        // exactly once when the last ref drops. This mirrors the
        // DRM_PRIME descriptor pattern in the VAAPI encoder.
        unsafe extern "C" fn pixbuf_free(_opaque: *mut std::ffi::c_void, data: *mut u8) {
            // SAFETY: `data` was passed in as the +1 retained
            // CVPixelBufferRef; this is the only path that releases it.
            unsafe { CFRelease(data.cast::<std::ffi::c_void>().cast_const()) };
        }
        // SAFETY: pixbuf is non-null (checked above). Pass it to
        // av_buffer_create as a u8* — FFmpeg treats it as opaque.
        // av_buffer_create returns null only on OOM; we CFRelease
        // and surface a clean error if that happens.
        let pixbuf_buf = unsafe {
            ffi::av_buffer_create(
                pixbuf.cast::<u8>(),
                0,
                Some(pixbuf_free),
                ptr::null_mut(),
                0,
            )
        };
        if pixbuf_buf.is_null() {
            // SAFETY: av_buffer_create didn't take ownership of the
            // CVPixelBufferRef on failure, so we still own the +1
            // retain and must release it.
            unsafe { CFRelease(pixbuf.cast_const()) };
            return Err(CodecError::Ffmpeg(RsmpegError::from(ffi::AVERROR(
                ffi::ENOMEM,
            ))));
        }

        // 3. Build the AVFrame. Format = VIDEOTOOLBOX, data[3] points
        // at the CVPixelBufferRef, buf[0] owns the AVBufferRef
        // wrapping it. hw_frames_ctx points at the encoder's pool so
        // the encoder accepts the frame on send_frame.
        let mut src = AVFrame::new();
        src.set_format(ffi::AV_PIX_FMT_VIDEOTOOLBOX);
        src.set_width(i32::try_from(self.width).expect("width fits in i32"));
        src.set_height(i32::try_from(self.height).expect("height fits in i32"));
        // SAFETY: deref_mut exposes the raw ffi::AVFrame so we can
        // poke buf[0]/data[3]/hw_frames_ctx (rsmpeg doesn't wrap those
        // slots). The AVFrame's Drop releases buf[0] via
        // av_frame_unref, which calls `pixbuf_free` once the last ref
        // drops. hw_frames_ctx is reffed below.
        let frames_ref = unsafe {
            let raw = src.deref_mut();
            raw.buf[0] = pixbuf_buf;
            raw.data[3] = pixbuf.cast::<u8>();
            let enc_frames_ref = self
                .encoder
                .hw_frames_ctx_mut()
                .expect("hw_frames_ctx set in new")
                .as_ptr();
            let frames_ref = ffi::av_buffer_ref(enc_frames_ref as *mut _);
            raw.hw_frames_ctx = frames_ref;
            frames_ref
        };
        // av_buffer_ref returns null on OOM. Without hw_frames_ctx the
        // encoder will reject the frame; bail with a clean error rather
        // than letting send_frame surface a generic EINVAL. The
        // CVPixelBuffer +1 retain is already owned by `pixbuf_buf` /
        // `src.buf[0]`, so the AVFrame's Drop releases it.
        if frames_ref.is_null() {
            return Err(CodecError::Ffmpeg(RsmpegError::from(ffi::AVERROR(
                ffi::ENOMEM,
            ))));
        }

        src.set_pts(pts);
        src.set_pict_type(if force_keyframe {
            ffi::AV_PICTURE_TYPE_I
        } else {
            ffi::AV_PICTURE_TYPE_NONE
        });

        self.encoder.send_frame(Some(&src))?;
        drop(src);

        drain_encoder(&mut self.encoder, &self.extradata)
    }
}

impl VideoToolboxEncoder {
    /// Signal EOF and drain any packets the encoder was still buffering.
    /// VideoToolbox typically holds the latest one or two submitted
    /// frames in its pipeline; without an explicit flush, the last
    /// keyframe of a short sequence can be left inside the encoder.
    /// Used by hardware tests and by the capability probe's encode →
    /// decode round-trip (which feeds one BGRA frame and needs to
    /// drain whatever VT buffered before the round-trip can examine
    /// the emitted bitstream).
    ///
    /// `pub` rather than `pub(crate)` so cross-crate hardware tests
    /// (e.g. `tether-render::iosurface_test`) can also drive the
    /// flush — the production trait shape (`Encoder`) deliberately
    /// doesn't expose flush because the live send loop never needs
    /// it, but tests that submit a short frame burst do.
    pub fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        self.encoder.send_frame(None)?;
        drain_encoder(&mut self.encoder, &self.extradata)
    }
}

impl Encoder for VideoToolboxEncoder {
    #[allow(clippy::cast_sign_loss)]
    fn encode_bgra(
        &mut self,
        bgra: &[u8],
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>> {
        let height = self.height as usize;
        let expected = self.bgra_row_bytes * height;
        if bgra.len() != expected {
            return Err(CodecError::BufferSizeMismatch {
                got: bgra.len(),
                expected,
            });
        }

        // 1. Copy BGRA into the encoder-side BGRA AVFrame, stride-aware.
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

        // 2. swscale BGRA -> hwframes sw_format (NV12 / P010 / NV24 / P410).
        self.bgra_to_sw.scale_frame(
            &self.bgra_frame,
            0,
            i32::try_from(height).expect("height fits in i32"),
            &mut self.sw_frame,
        )?;

        // 3. Allocate a VideoToolbox frame from the encoder's pool and
        // upload the NV12 bytes via av_hwframe_transfer_data.
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

        self.encoder.send_frame(Some(&hw_frame))?;
        drain_encoder(&mut self.encoder, &self.extradata)
    }

    fn is_hardware(&self) -> bool {
        true
    }

    fn codec_kind(&self) -> CodecKind {
        self.kind
    }

    fn name(&self) -> &'static str {
        vt_codec_name(self.kind)
    }

    fn encode_gpu(
        &mut self,
        frame: crate::GpuEncoderFrame<'_>,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>> {
        match frame {
            crate::GpuEncoderFrame::IOSurface(f) => {
                VideoToolboxEncoder::submit_iosurface(self, f, pts, force_keyframe)
            }
            crate::GpuEncoderFrame::_Phantom(_) => unreachable!("phantom variant"),
        }
    }
}

/// FFmpeg codec name for `find_encoder_by_name`. Errors for codecs we
/// don't ship a VideoToolbox path for yet.
///
/// AV1: FFmpeg 8.1 has no `av1_videotoolbox` encoder — verified by
/// `ffmpeg -encoders | grep videotoolbox` (only h264 / hevc / prores
/// appear) and by searching ffmpeg-devel archives for a submitted
/// wrapper (none found as of 2026-05). Whether any currently-shipped
/// Apple Silicon exposes hardware AV1 encode at all is independently
/// unconfirmed in public sources; either way, this codepath needs a
/// direct `VTCompressionSession` integration to reach it. Decode is
/// handled independently by `VideoToolboxDecoder`.
fn vt_codec_cname(kind: CodecKind) -> Result<&'static std::ffi::CStr> {
    match kind {
        CodecKind::H264 => Ok(c"h264_videotoolbox"),
        CodecKind::Hevc => Ok(c"hevc_videotoolbox"),
        CodecKind::Av1 => Err(CodecError::CodecNotFound(
            "av1_videotoolbox (no FFmpeg wrapper exists)",
        )),
    }
}

/// Human-readable backend name for logs and `Encoder::name`. Also
/// exercised by the `videotoolbox_codec_name_maps` default-on test
/// (the test calls *this* function — pinning the strings the
/// encoder actually uses, not a separate copy of them).
pub(crate) fn vt_codec_name(kind: CodecKind) -> &'static str {
    match kind {
        CodecKind::H264 => "h264_videotoolbox",
        CodecKind::Hevc => "hevc_videotoolbox",
        CodecKind::Av1 => "av1_videotoolbox",
    }
}

/// Pick the FFmpeg pix_fmt that goes in the hwframes context's
/// `sw_format` (and also in the BGRA→YUV swscale destination + the
/// staging `sw_frame`) for a given `(chroma, bit_depth)` combination.
///
/// We map every combination FFmpeg has a pix_fmt for and let
/// `encoder.open()` be the authority on whether the VideoToolbox
/// wrapper actually accepts that pix_fmt as input. The probe layer
/// treats an `encoder.open()` failure as `encode=false`, so anything
/// VT silently refuses falls out as a clean negative result rather
/// than as a panic. Combos with no matching FFmpeg pix_fmt return
/// `UnsupportedInputFormat` here — there's nothing to probe.
fn vt_sw_format(chroma: ChromaSubsampling, bit_depth: u8) -> Result<i32> {
    Ok(match (chroma, bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => ffi::AV_PIX_FMT_NV12,
        (ChromaSubsampling::Yuv420, 10) => ffi::AV_PIX_FMT_P010LE,
        (ChromaSubsampling::Yuv444, 8) => ffi::AV_PIX_FMT_NV24,
        (ChromaSubsampling::Yuv444, 10) => ffi::AV_PIX_FMT_P410LE,
        _ => return Err(CodecError::UnsupportedInputFormat),
    })
}

/// Does the IOSurface's `kCVPixelFormatType_*` fourcc match the
/// encoder's configured `(chroma, bit_depth)`? Used by
/// [`VideoToolboxEncoder::submit_iosurface`] to reject zero-copy
/// submissions whose surface format would silently mis-encode against
/// the encoder's `sw_format` — VT's hwaccel reads plane dimensions
/// from `sw_format`, not from the runtime CVPixelBuffer, so a mismatch
/// produces corrupted output rather than an error.
///
/// **Range policy:** only video-range fourccs (`'420v'`, `'x420'`,
/// `'444v'`) are accepted for the families that have both range
/// variants. The encoder VUI is hardcoded to `AVCOL_RANGE_MPEG`
/// (limited) in `new()` and the renderer is hardcoded to BT.709
/// limited; a full-range IOSurface (`'420f'`, `'xf20'`, `'444f'`)
/// would land as full-range bytes in a limited-tagged bitstream and
/// decode with crushed blacks and clipped whites on the client. The
/// macOS live host path captures BGRA and the Metal bridge asks for
/// video range where CoreVideo exposes such a label. Rejecting full
/// range here is the defence-in-depth guard if that ever changes.
/// The 4:4:4 10-bit bridge path uses the video-range `'x444'` label;
/// `'xf44'` remains accepted for direct SCK/diagnostic submissions.
///
/// Exposed at `pub` so cross-crate consistency tests can confirm the
/// encoder's accept set is a superset of what the macOS host bridge
/// can deliver — drift in either direction would crash a session at
/// first bridged frame.
#[must_use]
pub fn iosurface_fourcc_matches(chroma: ChromaSubsampling, bit_depth: u8, fourcc: u32) -> bool {
    const NV12_VIDEO: u32 = u32::from_be_bytes(*b"420v");
    const P010_VIDEO: u32 = u32::from_be_bytes(*b"x420");
    const NV24_VIDEO: u32 = u32::from_be_bytes(*b"444v");
    const P410_VIDEO: u32 = u32::from_be_bytes(*b"x444");
    const P410_FULL: u32 = u32::from_be_bytes(*b"xf44");
    matches!(
        (chroma, bit_depth, fourcc),
        (ChromaSubsampling::Yuv420, 8, NV12_VIDEO)
            | (ChromaSubsampling::Yuv420, 10, P010_VIDEO)
            | (ChromaSubsampling::Yuv444, 8, NV24_VIDEO)
            | (ChromaSubsampling::Yuv444, 10, P410_VIDEO | P410_FULL)
    )
}

#[cfg(test)]
mod fourcc_match_tests {
    use super::*;

    #[test]
    fn fourcc_matches_for_each_supported_combo() {
        // Encoder configured (chroma, bit_depth) ↔ IOSurface fourcc.
        // Only video-range fourccs are accepted for families that
        // have both range variants — the encoder VUI is hardcoded
        // AVCOL_RANGE_MPEG, so a full-range surface would mis-tag.
        // 4:4:4 10-bit accepts the bridge's video-range `'x444'`
        // and the direct-SCK / diagnostic full-range `'xf44'` label.
        for (chroma, bd, fourcc, accept) in [
            (
                ChromaSubsampling::Yuv420,
                8,
                u32::from_be_bytes(*b"420v"),
                true,
            ),
            (
                ChromaSubsampling::Yuv420,
                10,
                u32::from_be_bytes(*b"x420"),
                true,
            ),
            (
                ChromaSubsampling::Yuv444,
                8,
                u32::from_be_bytes(*b"444v"),
                true,
            ),
            (
                ChromaSubsampling::Yuv444,
                10,
                u32::from_be_bytes(*b"x444"),
                true,
            ),
            (
                ChromaSubsampling::Yuv444,
                10,
                u32::from_be_bytes(*b"xf44"),
                true,
            ),
            // Full-range siblings are rejected: encoder VUI is
            // limited, full-range bytes would mis-tag.
            (
                ChromaSubsampling::Yuv420,
                8,
                u32::from_be_bytes(*b"420f"),
                false,
            ),
            (
                ChromaSubsampling::Yuv420,
                10,
                u32::from_be_bytes(*b"xf20"),
                false,
            ),
            (
                ChromaSubsampling::Yuv444,
                8,
                u32::from_be_bytes(*b"444f"),
                false,
            ),
            // Cross-bucket mismatches: 10-bit cells for an 8-bit
            // encoder, 4:4:4 fourcc for a 4:2:0 encoder, etc.
            (
                ChromaSubsampling::Yuv420,
                8,
                u32::from_be_bytes(*b"x420"),
                false,
            ),
            (
                ChromaSubsampling::Yuv420,
                10,
                u32::from_be_bytes(*b"420v"),
                false,
            ),
            (
                ChromaSubsampling::Yuv420,
                8,
                u32::from_be_bytes(*b"444v"),
                false,
            ),
            (
                ChromaSubsampling::Yuv444,
                10,
                u32::from_be_bytes(*b"x420"),
                false,
            ),
        ] {
            assert_eq!(
                iosurface_fourcc_matches(chroma, bd, fourcc),
                accept,
                "({chroma:?}, {bd}, 0x{fourcc:08x}) accept={accept}"
            );
        }
    }
}
