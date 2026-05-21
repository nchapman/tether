//! VAAPI hardware video encoder. Codec-parameterized via
//! [`CodecKind`]: H.264 selects `h264_vaapi`, HEVC selects
//! `hevc_vaapi`. Both encoders share the NV12 surface pool, swscale
//! upload path, and DMA-BUF import path — only the FFmpeg codec name
//! and profile differ per codec.

use std::os::fd::AsRawFd;
use std::slice;

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{ra, AVDictionary, AVFrame, AVHWDeviceContext};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use rsmpeg::swscale::SwsContext;
use rsmpeg::UnsafeDerefMut;
use tracing::warn;

use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

use crate::h264::frame_plane_mut;
use crate::{
    init_ffmpeg, CodecError, DmaBufFrame, Encoder, EncodedPacket, Result, GOP_SECONDS,
};

use super::ffi::{
    AVDRMFrameDescriptor, AVDRMLayerDescriptor, AVDRMObjectDescriptor, AVDRMPlaneDescriptor,
    AV_DRM_MAX_PLANES,
};
use super::VAAPI_POOL_SIZE;

pub struct VaapiEncoder {
    kind: CodecKind,
    /// Negotiated chroma sampling. Determines:
    /// - `sw_format` on the encoder's hwframes pool (`NV12` for 4:2:0,
    ///   `YUV444P` for 4:4:4).
    /// - The pixel-conversion swscale context fed in `encode_bgra`.
    /// - The VAAPI driver profile string (`main` for 4:2:0, `rext` for
    ///   HEVC Main444).
    /// - The expected fourcc on imported DMA-BUFs in `submit_dmabuf`.
    chroma: ChromaSubsampling,
    encoder: AVCodecContext,
    /// BGRA → encoder-input swscale context. NV12 for 4:2:0, YUV444P
    /// for 4:4:4. The output format matches the encoder's `sw_format`;
    /// SwsContext is statically pinned to a single (src, dst) pair so
    /// the chroma choice is baked in at construction.
    bgra_to_encoder_input: SwsContext,
    sw_frame: AVFrame,
    bgra_frame: AVFrame,
    /// Codec parameter sets (Annex-B SPS/PPS for H.264, VPS+SPS+PPS
    /// for HEVC) captured once after `encoder.open()`. We prepend
    /// these to every keyframe packet so a decoder that joins
    /// mid-stream, rebuilds after a device loss, or loses the
    /// session's very first IDR can recover on the next IDR rather
    /// than getting stuck waiting for parameter sets that libavcodec
    /// only emitted in band on the encoder's first packet.
    ///
    /// Empty if the encoder didn't populate `extradata` (shouldn't
    /// happen for h264_vaapi/hevc_vaapi at Main profile — both write
    /// Annex-B parameter sets to extradata at open()).
    extradata: Vec<u8>,
    // Keep the device context alive for the encoder's lifetime. The
    // encoder's `hw_frames_ctx` holds an internal ref-counted handle
    // to the device, so dropping this field early wouldn't free VAAPI
    // resources prematurely — but keeping the explicit owner here
    // documents the lifetime relationship and is cheap.
    //
    // Drop order: struct fields drop in declaration order, so `encoder`
    // (and its internal hw_frames_ctx) tear down before `_hw_device`,
    // which is the correct order — surfaces must be freed before the
    // device that allocated them.
    _hw_device: AVHWDeviceContext,
    width: u32,
    height: u32,
    bgra_row_bytes: usize,
}

// SAFETY: ffmpeg HW codec context, VAAPI device, and per-encoder frames
// are safe to MOVE between threads but unsafe to SHARE. We only expose
// `&mut self` methods so the borrow checker serialises access within a
// single thread; the manual `unsafe impl Send` matches the move-fine /
// share-bad contract that all our other encoders document.
unsafe impl Send for VaapiEncoder {}

impl VaapiEncoder {
    /// Construct a VAAPI encoder for the given codec at the given
    /// dimensions. `kind` selects between `h264_vaapi` and
    /// `hevc_vaapi`; AV1 is not yet supported.
    ///
    /// Returns `Err(CodecError::CodecNotFound)` if the installed
    /// FFmpeg wasn't built with VAAPI support for the requested codec,
    /// and any `RsmpegError` if the VAAPI device can't be opened (no
    /// `/dev/dri/renderD*` accessible, driver mismatch, etc.). The
    /// probe in `crate::probe` walks the client's preferred-codec
    /// list, calling this for each kind until one succeeds.
    ///
    /// The low-power VAAPI encode entrypoint
    /// (`VAEntrypointEncSliceLP`) would shave more latency on Intel
    /// hardware that exposes it, but we don't currently probe whether
    /// LP is supported for the (codec, profile) combo. Asking for LP
    /// when the device doesn't expose it produces a partially-init'd
    /// FFmpeg encoder context that segfaults on Drop instead of
    /// returning a clean error — and Meteor Lake Arc is one of those
    /// devices (no LP for H.264). A safe LP path needs an explicit
    /// libva capability query before encoder.open(); deferring until
    /// the win is worth the FFI surface.
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
        // 8-bit is the only depth wired today. The probe layer enforces
        // this so we should never reach here with anything else; panic
        // rather than silently mis-encode if the contract slips.
        assert_eq!(
            profile.bit_depth, 8,
            "VAAPI encoder only supports 8-bit profiles; got {}-bit",
            profile.bit_depth
        );
        // VAAPI has no H.264 4:4:4 encode profile across any driver we
        // target (Sunshine confirms the same — see
        // refs/Sunshine/src/platform/linux/vaapi.cpp:202). Refuse early
        // rather than producing a confusing FFmpeg open() failure.
        if kind == CodecKind::H264 && chroma == ChromaSubsampling::Yuv444 {
            return Err(CodecError::CodecNotFound(
                "VAAPI does not expose an H.264 4:4:4 encode profile",
            ));
        }

        let codec_cname = vaapi_codec_cname(kind)?;
        let codec = AVCodec::find_encoder_by_name(codec_cname)
            .ok_or(CodecError::CodecNotFound(vaapi_codec_name(kind)))?;

        // Default VAAPI device (typically /dev/dri/renderD128). None
        // lets FFmpeg pick — explicit device strings only matter on
        // multi-GPU systems, which we'll handle when a user with that
        // setup hits us up.
        let hw_device =
            AVHWDeviceContext::create(ffi::AV_HWDEVICE_TYPE_VAAPI, None, None, 0)?;

        let width_i32 = i32::try_from(width).expect("width fits in i32");
        let height_i32 = i32::try_from(height).expect("height fits in i32");
        let fps_i32 = i32::try_from(fps.max(1)).unwrap_or(60);

        let mut encoder = AVCodecContext::new(&codec);
        encoder.set_width(width_i32);
        encoder.set_height(height_i32);
        // pix_fmt is VAAPI: pixels live on the GPU; the actual storage
        // layout is in `sw_format` on the hwframes context below.
        encoder.set_pix_fmt(ffi::AV_PIX_FMT_VAAPI);
        encoder.set_time_base(ra(1, fps_i32));
        encoder.set_framerate(ra(fps_i32, 1));
        encoder.set_bit_rate(i64::from(bitrate_kbps) * 1000);
        // GOP cadence matches the libx264 fallback so the on-wire
        // worst-case "garbled until next IDR" window is the same
        // regardless of which encoder the probe picked.
        let gop_frames = fps_i32
            .saturating_mul(i32::try_from(GOP_SECONDS).expect("GOP_SECONDS fits in i32"));
        encoder.set_gop_size(gop_frames);
        encoder.set_max_b_frames(0);

        // Build + attach the hwframes context. The `sw_format` decides
        // the on-device pixel layout. NV12 for 4:2:0 (interleaved UV
        // at half resolution). For 4:4:4 we use VUYX (packed
        // X|Y|U|V 8:8:8:8, 32 bpp) rather than planar YUV444P — the
        // packed layout is the only 4:4:4 8-bit format ffmpeg's
        // `vaapi_drm_format_map` can import via DRM_PRIME on the
        // gpuconvert→encoder hop (no entry exists for planar
        // YUV444P or its R8/YU24 DRM encodings). VAAPI's encoder
        // accepts VUYX-format surfaces as input to HEVC Main 4:4:4.
        let sw_format = match chroma {
            ChromaSubsampling::Yuv420 => ffi::AV_PIX_FMT_NV12,
            ChromaSubsampling::Yuv444 => ffi::AV_PIX_FMT_VUYX,
        };
        let mut hw_frames_ref = hw_device.hwframe_ctx_alloc();
        hw_frames_ref.data().format = ffi::AV_PIX_FMT_VAAPI;
        hw_frames_ref.data().sw_format = sw_format;
        hw_frames_ref.data().width = width_i32;
        hw_frames_ref.data().height = height_i32;
        hw_frames_ref.data().initial_pool_size = VAAPI_POOL_SIZE;
        hw_frames_ref.init()?;
        encoder.set_hw_frames_ctx(hw_frames_ref);

        // Color identity. The SPS VUI records what conforming decoders
        // need to interpret the YCbCr bytes correctly. libavcodec
        // defaults all four to "Unspecified" when these fields are
        // untouched — and conforming decoders are then free to guess.
        // 4:2:0 hardware decoders mostly guess BT.709 at HD+ and get
        // it right by luck; 4:4:4 on saturated UI text reveals the
        // mismatch as a visible color shift. Set explicitly so the
        // gpuconvert shader's BT.709 limited-range math matches what
        // the decoder applies on the other end.
        //
        // SAFETY: these AVCodecContext fields must be written before
        // avcodec_open2 — same invariant as `extra_hw_frames` below.
        unsafe {
            let raw = encoder.deref_mut();
            raw.color_primaries = ffi::AVCOL_PRI_BT709;
            raw.color_trc = ffi::AVCOL_TRC_BT709;
            raw.colorspace = ffi::AVCOL_SPC_BT709;
            raw.color_range = ffi::AVCOL_RANGE_MPEG;
            // Pin the HEVC REXT profile via the context field rather
            // than the `profile=rext` AVOption string. The string form
            // is brittle: libavcodec's REXT umbrella covers Main 4:4:4
            // 8-bit, Main 4:4:4 10-bit, Main Intra, and several other
            // sub-profiles; the actual one selected is inferred from
            // the hwframes `sw_format`. Setting the field directly is
            // Sunshine's reference pattern (refs/Sunshine/src/video.cpp:1687)
            // and future-proofs against a 10-bit code path drifting
            // the inferred sub-profile.
            if matches!(chroma, ChromaSubsampling::Yuv444) && kind == CodecKind::Hevc {
                raw.profile = ffi::AV_PROFILE_HEVC_REXT as i32;
            }
        }

        // VAAPI private options. The defaults are tuned for file-based
        // transcoding throughput, not realtime; we override the knobs
        // that cost us the most:
        //   profile=main — Main is the safest broadly-supported profile
        //     across Intel/AMD/NVIDIA VAAPI drivers (H.264 Main, HEVC
        //     Main — same option string for both codecs). The encoder
        //     defaults to High for H.264, which fails to open on hardware
        //     that exposes only Main for the chosen entrypoint. Every
        //     realistic decoder we'll talk to handles Main. Per-codec
        //     profile probing via vaQueryConfigProfiles +
        //     vaGetConfigAttributes is the principled replacement; it
        //     needs a small libva FFI extension (vaQueryConfigProfiles,
        //     vaGetConfigAttributes) and an AMD test machine to validate.
        //     Deferred until we have both.
        //   async_depth=1 — synchronous mode. Default is 4, which buys
        //     throughput at the cost of three extra frames of latency
        //     before the first packet emerges. We need the opposite trade.
        //   rc_mode=VBR — VBR matches Intel's recommended low-latency
        //     mode (CBR-style buffering also works but introduces a
        //     visible bitrate floor on static content). Apollo carries
        //     an explicit CBR/VBR probe for AMD VAAPI (commit 2aa5a396)
        //     because the default of CQP produces bad bitrate behavior
        //     on Radeon. We'll add the same probe alongside the profile
        //     probe above when we have AMD hardware to validate against;
        //     today the VBR hard-code works on Intel and NVIDIA via
        //     nvidia-vaapi-driver, which is what tether users run.
        //   idr_interval=INT_MAX — disable the encoder's internal periodic
        //     IDR. We drive IDRs on demand via IdrSignal at the host
        //     orchestration layer; the GOP_SECONDS-based gop_size above
        //     bounds the recovery window for clients that miss a forced
        //     IDR. Setting this to INT_MAX matches Sunshine.
        //   sei=0 — suppress SEI prefix NAL units (timing info, recovery
        //     point, etc.). Saves a few bytes per IDR and no decoder we
        //     ship to needs them.
        // Profile string for the 4:2:0 path. HEVC Main444 uses the
        // direct AVCodecContext.profile assignment above instead — see
        // the comment there for why. For 4:2:0 we still go through the
        // string AVOption because libavcodec's `profile=main` is the
        // safest broadly-supported value across drivers; the encoder
        // defaults to High for H.264, which fails to open on hardware
        // that only exposes Main for the chosen entrypoint.
        let dict_builder = AVDictionary::new(c"async_depth", c"1", 0)
            .set(c"rc_mode", c"VBR", 0)
            .set(c"idr_interval", c"2147483647", 0)
            .set(c"sei", c"0", 0);
        let dict = match chroma {
            ChromaSubsampling::Yuv420 => dict_builder.set(c"profile", c"main", 0),
            // 4:4:4 path: profile pinned via context field above, no
            // AVOption needed.
            ChromaSubsampling::Yuv444 => dict_builder,
        };
        // AV_CODEC_FLAG_GLOBAL_HEADER routes the codec's parameter
        // sets (SPS/PPS for H.264, VPS+SPS+PPS for HEVC) into
        // `AVCodecContext.extradata` at open() rather than emitting
        // them only in band with the first IDR. We then prepend
        // extradata to every keyframe at drain time, so any IDR is
        // independently decodable — required for clients that join
        // mid-session, rebuild their decoder on device loss, or
        // lose the session's first IDR on the wire.
        #[allow(clippy::cast_possible_wrap)]
        encoder.set_flags(encoder.flags | ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32);
        let leftover = encoder.open(Some(dict))?;
        if let Some(unused) = leftover {
            // Driver/encoder didn't recognise one or more opts. Not
            // fatal — the encoder will still work, just with the
            // unrecognised setting at its default. Surfacing the keys
            // helps diagnose "why is latency higher than expected?"
            let mut unused_keys: Vec<String> = Vec::new();
            for entry in unused.iter() {
                unused_keys.push(format!(
                    "{}={}",
                    entry.key().to_string_lossy(),
                    entry.value().to_string_lossy()
                ));
            }
            if !unused_keys.is_empty() {
                warn!(
                    codec = vaapi_codec_name(kind),
                    unused = ?unused_keys,
                    "VAAPI encoder ignored some private options; latency knobs may not be applied"
                );
            }
        }

        let scaler_label = match chroma {
            ChromaSubsampling::Yuv420 => "BGRA -> NV12",
            ChromaSubsampling::Yuv444 => "BGRA -> VUYX",
        };
        let bgra_to_encoder_input = SwsContext::get_context(
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

        // Snapshot the encoder's parameter-set extradata. For
        // h264_vaapi/hevc_vaapi at Main, this is the Annex-B SPS/PPS
        // (HEVC also includes VPS). We prepend this to keyframe
        // packets at drain time. Without it, only the encoder's very
        // first packet would carry parameter sets — a decoder that
        // rebuilds (resume, resize) or loses that first IDR has no
        // recovery path.
        //
        // SAFETY: extradata is populated by libavcodec inside
        // open() when AV_CODEC_FLAG_GLOBAL_HEADER is set (above).
        // libavcodec does not update extradata mid-stream for
        // h264_vaapi/hevc_vaapi at fixed resolution; the VaapiEncoder
        // rebuilds entirely on resolution change rather than mutating
        // in place. We copy into an owned Vec immediately so no
        // subsequent encoder operations can race with our read.
        let extradata = unsafe {
            let raw = encoder.extradata;
            let size = encoder.extradata_size;
            if raw.is_null() || size <= 0 {
                Vec::new()
            } else {
                slice::from_raw_parts(raw, size as usize).to_vec()
            }
        };
        if extradata.is_empty() {
            warn!(
                codec = vaapi_codec_name(kind),
                "encoder.extradata was empty after open(); keyframes will not carry SPS/PPS \
                 (clients that lose the first IDR will be stuck)"
            );
        }

        Ok(Self {
            kind,
            chroma,
            encoder,
            bgra_to_encoder_input,
            sw_frame,
            bgra_frame,
            extradata,
            _hw_device: hw_device,
            width,
            height,
            bgra_row_bytes,
        })
    }

    /// Encode one DMA-BUF-backed frame without a CPU upload. The fds
    /// inside `frame` are *borrowed* — ffmpeg's VAAPI import dups them
    /// via `vaCreateSurfaces` with PRIME_2 mem-type, so the caller can
    /// drop the source `DmaBufFrame` (and its `GpuFrameGuard`)
    /// immediately after this returns. `force_keyframe` mirrors
    /// `encode_bgra`.
    ///
    /// Constraints: `frame.fourcc` must match the negotiated chroma
    /// — `NV12` for 4:2:0 (the encoder's NV12 `sw_format`) or `XYUV`
    /// (DRM_FORMAT_XYUV8888 packed) for HEVC Main444. Width/height
    /// are pinned to the encoder's construction values; resolution
    /// changes go through a full encoder rebuild.
    ///
    /// Approach: reuse the encoder's existing VAAPI hwframes pool and
    /// let `vaapi_map_from_drm` add each imported surface into it
    /// on-demand. An earlier attempt to use
    /// `av_hwframe_ctx_create_derived(VAAPI ← DRM)` for a pool that
    /// "started" with DRM as its source returned ENOSYS — that helper
    /// is only set up for the export direction (VAAPI → DRM). Per-frame
    /// map into a regular pool is what works.
    #[allow(clippy::cast_sign_loss)]
    pub fn submit_dmabuf(
        &mut self,
        frame: &DmaBufFrame,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>> {
        // DRM fourccs for the two pixel formats the encoder accepts.
        // The dma-buf bridge must hand us a frame matching the
        // negotiated chroma — otherwise av_hwframe_map would fail
        // mid-pipeline with a much less actionable error.
        const NV12_FOURCC: u32 = u32::from_le_bytes(*b"NV12");
        // 4:4:4 path: DRM_FORMAT_XYUV8888 (`XYUV` in fourcc form;
        // bytes V, U, Y, X in memory little-endian). This is the only
        // 4:4:4 8-bit format ffmpeg's `vaapi_drm_format_map` recognises
        // for DRM_PRIME import — planar YUV444P / YU24 / three-R8-layer
        // shapes fail with "DRM format not supported by VAAPI".
        const XYUV_FOURCC: u32 = u32::from_le_bytes(*b"XYUV");
        let expected_fourcc = match self.chroma {
            ChromaSubsampling::Yuv420 => NV12_FOURCC,
            ChromaSubsampling::Yuv444 => XYUV_FOURCC,
        };
        if frame.fourcc != expected_fourcc {
            // Mismatch usually means a stale dma-buf bridge initialised
            // for one chroma was fed a frame from the other (or
            // gpuconvert and the encoder disagree on the YUV444P
            // wire fourcc). Both numbers in the log so the root cause
            // is obvious without running with AV_LOG_DEBUG.
            warn!(
                actual = format_args!("0x{:08x}", frame.fourcc),
                expected = format_args!("0x{:08x}", expected_fourcc),
                chroma = ?self.chroma,
                "imported dma-buf fourcc does not match the negotiated chroma"
            );
            return Err(CodecError::UnsupportedInputFormat);
        }
        if frame.objects.len() > AV_DRM_MAX_PLANES
            || frame.layers.len() > AV_DRM_MAX_PLANES
        {
            return Err(CodecError::UnsupportedInputFormat);
        }
        // Same upper bound on per-layer planes — otherwise the
        // `min(AV_DRM_MAX_PLANES)` clamp in the layer loop would
        // silently drop planes the source actually has, which is a
        // recipe for "looks fine, decodes garbage" downstream.
        for layer in &frame.layers {
            if (layer.num_planes as usize) > AV_DRM_MAX_PLANES {
                return Err(CodecError::UnsupportedInputFormat);
            }
        }

        // Build the AVDRMFrameDescriptor. We need it owned by an
        // AVBufferRef stored in src->buf[0] — not just dangling off
        // src->data[0] — because ff_hwframe_map_create internally
        // calls av_frame_ref(hwmap->source, src), and av_frame_ref
        // deep-copies when src->buf[0] is null. Deep-copy invokes
        // av_frame_get_buffer which DRM_PRIME can't satisfy (the
        // hwcontext_drm.h contract is "user-allocated only"), so it
        // returns EINVAL and the whole map fails. With buf[0] set,
        // av_frame_ref takes the simple ref-bump path.
        let mut desc: Box<AVDRMFrameDescriptor> =
            Box::new(unsafe { std::mem::zeroed() });
        desc.nb_objects = i32::try_from(frame.objects.len()).expect("<= 4");
        for (i, obj) in frame.objects.iter().enumerate() {
            desc.objects[i] = AVDRMObjectDescriptor {
                fd: obj.fd.as_raw_fd(),
                size: usize::try_from(obj.size).expect("size fits in usize"),
                format_modifier: obj.drm_format_modifier,
            };
        }
        desc.nb_layers = i32::try_from(frame.layers.len()).expect("<= 4");
        for (i, layer) in frame.layers.iter().enumerate() {
            let mut planes = [AVDRMPlaneDescriptor {
                object_index: 0,
                offset: 0,
                pitch: 0,
            }; AV_DRM_MAX_PLANES];
            for p in 0..(layer.num_planes as usize).min(AV_DRM_MAX_PLANES) {
                planes[p] = AVDRMPlaneDescriptor {
                    object_index: i32::try_from(layer.object_index[p])
                        .expect("object_index fits in i32"),
                    offset: isize::try_from(layer.offset[p])
                        .expect("offset fits in isize"),
                    pitch: isize::try_from(layer.pitch[p])
                        .expect("pitch fits in isize"),
                };
            }
            desc.layers[i] = AVDRMLayerDescriptor {
                format: layer.drm_format,
                nb_planes: i32::try_from(layer.num_planes).expect("<= 4"),
                planes,
            };
        }

        // Wrap the descriptor in an AVBufferRef so src->buf[0] is set
        // (see comment above). The free callback drops the Box that
        // owns the descriptor's heap allocation. av_buffer_create takes
        // ownership of `desc`; we Box::into_raw to surrender the Box.
        let desc_ptr = Box::into_raw(desc);
        unsafe extern "C" fn drm_desc_free(
            _opaque: *mut std::ffi::c_void,
            data: *mut u8,
        ) {
            // SAFETY: data was produced by Box::into_raw on a
            // Box<AVDRMFrameDescriptor>; this is the only path that
            // reclaims it.
            drop(unsafe { Box::from_raw(data as *mut AVDRMFrameDescriptor) });
        }
        // SAFETY: desc_ptr is a valid heap allocation of size
        // size_of::<AVDRMFrameDescriptor>(); the free callback matches
        // its allocation strategy. av_buffer_create returns null only
        // on OOM, which we convert to a clean error rather than
        // unwinding through the raw pointer.
        let desc_buf = unsafe {
            ffi::av_buffer_create(
                desc_ptr as *mut u8,
                std::mem::size_of::<AVDRMFrameDescriptor>(),
                Some(drm_desc_free),
                std::ptr::null_mut(),
                0,
            )
        };
        if desc_buf.is_null() {
            // SAFETY: av_buffer_create didn't take ownership, so we
            // still own desc_ptr and must reclaim it.
            unsafe { drop(Box::from_raw(desc_ptr)) };
            return Err(CodecError::Ffmpeg(RsmpegError::from(ffi::AVERROR(ffi::ENOMEM))));
        }

        // Build the source DRM_PRIME AVFrame. Pointing data[0] +
        // buf[0] at the descriptor is the ffmpeg convention;
        // hw_frames_ctx stays NULL per the hwcontext_drm.h contract.
        let mut src = AVFrame::new();
        src.set_format(ffi::AV_PIX_FMT_DRM_PRIME);
        src.set_width(i32::try_from(self.width).expect("width fits in i32"));
        src.set_height(i32::try_from(self.height).expect("height fits in i32"));
        // SAFETY: deref_mut() exposes the raw ffi::AVFrame so we can
        // poke buf[0]/data[0] (rsmpeg doesn't wrap them). buf[0] takes
        // ownership of the ref we got from av_buffer_create; the
        // AVFrame's Drop releases it via av_frame_unref, which calls
        // the free callback when the last ref drops.
        unsafe {
            let raw = src.deref_mut();
            raw.buf[0] = desc_buf;
            raw.data[0] = desc_ptr as *mut u8;
        }

        // Destination VAAPI frame. Pre-set format + hw_frames_ctx so
        // av_hwframe_map sees the target context and creates a VA
        // surface inside the encoder's pool that wraps the imported
        // DMA-BUF (AV_HWFRAME_MAP_DIRECT semantics).
        let mut dst = AVFrame::new();
        dst.set_format(ffi::AV_PIX_FMT_VAAPI);
        dst.set_width(i32::try_from(self.width).expect("width fits in i32"));
        dst.set_height(i32::try_from(self.height).expect("height fits in i32"));
        // SAFETY: pointing dst.hw_frames_ctx at the encoder's pool via
        // av_buffer_ref bumps the ref count; Drop releases it. Without
        // this, av_hwframe_map allocates a fresh internal pool that
        // the encoder wouldn't accept on send_frame.
        unsafe {
            let enc_frames_ref = self
                .encoder
                .hw_frames_ctx_mut()
                .expect("hw_frames_ctx set in new_bgra")
                .as_ptr();
            dst.deref_mut().hw_frames_ctx = ffi::av_buffer_ref(enc_frames_ref as *mut _);
        }

        // SAFETY: src is a fully-populated DRM_PRIME frame whose
        // descriptor lives in src->buf[0]; dst is a freshly-allocated
        // empty frame whose hw_frames_ctx points at the encoder's
        // pool. AV_HWFRAME_MAP_DIRECT requests a zero-copy mapping.
        // No AV_HWFRAME_MAP_READ/WRITE: those flags describe the
        // user's intended access pattern to the *mapped* output, and
        // the encoder reads via VAAPI ops rather than direct CPU/EGL
        // access — setting MAP_READ here would tell the driver to
        // wire up CPU readback fencing, which radeonsi has been
        // observed to reject on tiled modifiers.
        let rc = unsafe {
            ffi::av_hwframe_map(
                dst.as_mut_ptr(),
                src.as_ptr(),
                ffi::AV_HWFRAME_MAP_DIRECT as i32,
            )
        };
        if rc < 0 {
            // Map failures here are usually a modifier mismatch
            // between the source DMA-BUF and what the encoder's pool
            // accepts. Log the descriptor shape so field bugs are
            // tractable without re-running with AV_LOG_DEBUG.
            warn!(
                rc,
                fourcc = format_args!("{:08x}", frame.fourcc),
                num_objects = frame.objects.len(),
                num_layers = frame.layers.len(),
                modifier = frame.objects.first().map(|o| o.drm_format_modifier),
                "av_hwframe_map(DRM_PRIME -> VAAPI) failed"
            );
            return Err(CodecError::Ffmpeg(RsmpegError::from(rc)));
        }

        dst.set_pts(pts);
        dst.set_pict_type(if force_keyframe {
            ffi::AV_PICTURE_TYPE_I
        } else {
            ffi::AV_PICTURE_TYPE_NONE
        });

        self.encoder.send_frame(Some(&dst))?;

        // src + dst are no longer needed past this point — the
        // dst frame holds a fresh VA surface that internally references
        // the dup'd DMA-BUF fds. Explicit drop documents the lifetime;
        // src's buf[0] release triggers the descriptor's free callback.
        drop(dst);
        drop(src);

        drain_encoder(&mut self.encoder, &self.extradata)
    }
}

impl Encoder for VaapiEncoder {
    // ffmpeg's i32 ABI fields (linesize, packet.size) are non-negative
    // on allocated frames / valid packets. Same rationale as h264.rs.
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

        // 1. Copy BGRA bytes into the encoder-side BGRA AVFrame,
        // stride-aware in case the row alignment differs from
        // width*4.
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

        // 2. swscale BGRA -> encoder input format (NV12 or YUV444P
        // depending on the negotiated chroma) into the CPU-side
        // sw_frame. The DMA-BUF zero-copy path delivers the same
        // pixel format pre-converted by tether-gpuconvert.
        self.bgra_to_encoder_input.scale_frame(
            &self.bgra_frame,
            0,
            i32::try_from(height).expect("height fits in i32"),
            &mut self.sw_frame,
        )?;

        // 3. Allocate a VAAPI surface from the encoder's hwframes pool
        // and upload the NV12 bytes via av_hwframe_transfer_data
        // (CPU memcpy into GPU memory). The pool blocks if all
        // surfaces are still in flight at the encoder, which with
        // async_depth=1 + 8 surfaces basically never happens.
        let mut hw_frame = AVFrame::new();
        self.encoder
            .hw_frames_ctx_mut()
            .expect("hw_frames_ctx set in new_bgra")
            .get_buffer(&mut hw_frame)?;
        hw_frame.hwframe_transfer_data(&self.sw_frame)?;
        hw_frame.set_pts(pts);
        hw_frame.set_pict_type(if force_keyframe {
            ffi::AV_PICTURE_TYPE_I
        } else {
            ffi::AV_PICTURE_TYPE_NONE
        });

        // 4. Submit + drain. With async_depth=1 the receive_packet
        // immediately yields a packet on the first call and EAGAIN on
        // the second; the loop generalises gracefully if a future
        // async_depth bump emits multiple packets per submit.
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
        // Could be refined to "h264_vaapi (Intel)" / "(AMD)" / "(NVIDIA)"
        // by querying the VA driver string from the hw_device. Not
        // worth the extra FFI on first pass — the log already shows
        // is_hardware=true alongside the name.
        vaapi_codec_name(self.kind)
    }

    fn encode_gpu(
        &mut self,
        frame: crate::GpuEncoderFrame<'_>,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedPacket>> {
        match frame {
            crate::GpuEncoderFrame::DmaBuf(f) => {
                VaapiEncoder::submit_dmabuf(self, f, pts, force_keyframe)
            }
            crate::GpuEncoderFrame::_Phantom(_) => unreachable!("phantom variant"),
        }
    }
}

/// FFmpeg codec name (as a `CStr` for `find_encoder_by_name`) for
/// the given codec kind. Errors for codecs we don't have a VAAPI
/// build path for yet.
fn vaapi_codec_cname(kind: CodecKind) -> Result<&'static std::ffi::CStr> {
    match kind {
        CodecKind::H264 => Ok(c"h264_vaapi"),
        CodecKind::Hevc => Ok(c"hevc_vaapi"),
        CodecKind::Av1 => Err(CodecError::CodecNotFound("av1_vaapi (not yet supported)")),
    }
}

/// Human-readable VAAPI encoder name for logs and the `Encoder::name`
/// trait method. Returns a stable `&'static str` per codec.
fn vaapi_codec_name(kind: CodecKind) -> &'static str {
    match kind {
        CodecKind::H264 => "h264_vaapi",
        CodecKind::Hevc => "hevc_vaapi",
        CodecKind::Av1 => "av1_vaapi",
    }
}

/// Drain all packets currently buffered in the encoder. Returns the
/// (possibly empty) list of fresh `EncodedPacket`s — empty is normal
/// on the very first frame while libavcodec buffers SPS/PPS.
///
/// `extradata` is the Annex-B parameter set bundle captured from the
/// encoder at construction. For each keyframe packet we prepend it so
/// the decoder gets fresh SPS/PPS/VPS in band with every IDR (the
/// streaming-friendly contract that mp4 muxers don't need but our
/// raw wire format does). Cost: one allocation per keyframe of size
/// `extradata.len()` (~25 bytes H.264, ~50 bytes HEVC). P-frames
/// pass through untouched.
///
/// Without this, only the encoder's first packet carries parameter
/// sets, and any client that joins mid-session or rebuilds its
/// decoder (resume, resolution change) is stuck.
#[allow(clippy::cast_sign_loss)]
fn drain_encoder(encoder: &mut AVCodecContext, extradata: &[u8]) -> Result<Vec<EncodedPacket>> {
    let mut out = Vec::new();
    loop {
        let packet = match encoder.receive_packet() {
            Ok(p) => p,
            Err(RsmpegError::EncoderDrainError | RsmpegError::EncoderFlushedError) => break,
            Err(e) => return Err(CodecError::Ffmpeg(e)),
        };
        let size = packet.size as usize;
        // SAFETY: packet.data points to packet.size valid bytes
        // owned by the AVPacket; we copy them before drop.
        let raw = unsafe { slice::from_raw_parts(packet.data, size) };
        let keyframe = (packet.flags & ffi::AV_PKT_FLAG_KEY as i32) != 0;
        let data = if keyframe && !extradata.is_empty() {
            let mut buf = Vec::with_capacity(extradata.len() + raw.len());
            buf.extend_from_slice(extradata);
            buf.extend_from_slice(raw);
            buf
        } else {
            raw.to_vec()
        };
        let pts_out = if packet.pts == ffi::AV_NOPTS_VALUE {
            None
        } else {
            Some(packet.pts)
        };
        out.push(EncodedPacket {
            data,
            pts: pts_out,
            keyframe,
        });
    }
    Ok(out)
}
