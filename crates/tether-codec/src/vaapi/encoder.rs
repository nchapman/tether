//! VAAPI hardware video encoder. Codec-parameterized via
//! [`CodecKind`]: H.264 selects `h264_vaapi`, HEVC selects
//! `hevc_vaapi`. Both encoders share the NV12 surface pool, swscale
//! upload path, and DMA-BUF import path — only the FFmpeg codec name
//! and profile differ per codec.

use std::os::fd::AsRawFd;

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{ra, AVDictionary, AVFrame, AVHWDeviceContext};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use rsmpeg::swscale::SwsContext;
use rsmpeg::UnsafeDerefMut;
use tracing::{debug, warn};

use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

use crate::encoder_common::{drain_encoder, snapshot_extradata};
use crate::h264::frame_plane_mut;
use crate::{init_ffmpeg, CodecError, DmaBufFrame, EncodedPacket, Encoder, Result, GOP_SECONDS};

use super::ffi::{
    AVDRMFrameDescriptor, AVDRMLayerDescriptor, AVDRMObjectDescriptor, AVDRMPlaneDescriptor,
    AV_DRM_MAX_PLANES,
};
use super::VAAPI_POOL_SIZE;

pub struct VaapiEncoder {
    kind: CodecKind,
    /// Negotiated chroma sampling. Pinned at construction; resolves
    /// (alongside `bit_depth`) to a single FFmpeg pix_fmt through
    /// [`vaapi_sw_format`], which is then used for the hwframes pool's
    /// `sw_format`, the BGRA→YUV swscale destination, and the expected
    /// fourcc on imported DMA-BUFs in `submit_dmabuf`.
    chroma: ChromaSubsampling,
    /// Negotiated bit depth (8 or 10 in practice). Same role as
    /// `chroma`: feeds `vaapi_sw_format`. Stored separately so
    /// `submit_dmabuf` can validate `(chroma, bit_depth)` against the
    /// imported DMA-BUF before mapping it.
    bit_depth: u8,
    encoder: AVCodecContext,
    /// BGRA → encoder-input swscale context. Output format matches the
    /// encoder's `sw_format` (see [`vaapi_sw_format`]); SwsContext is
    /// statically pinned to a single (src, dst) pair so the
    /// (chroma, bit_depth) choice is baked in at construction.
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
    /// AVOption keys that `encoder.open()` left in the dict because the
    /// driver/encoder didn't recognise them. Empty when every requested
    /// option was accepted. Exposed via [`Self::unused_avoptions`] so
    /// tests can confirm an opt-in setting (e.g. `intra_refresh`) was
    /// actually consumed by the driver, not silently ignored.
    unused_avoptions: Vec<String>,
}

// SAFETY: ffmpeg HW codec context, VAAPI device, and per-encoder frames
// are safe to MOVE between threads but unsafe to SHARE. We only expose
// `&mut self` methods so the borrow checker serialises access within a
// single thread; the manual `unsafe impl Send` matches the move-fine /
// share-bad contract that all our other encoders document.
unsafe impl Send for VaapiEncoder {}

impl VaapiEncoder {
    /// Construct a VAAPI encoder for the given codec at the given
    /// dimensions. `profile.codec` selects between `h264_vaapi`,
    /// `hevc_vaapi`, and `av1_vaapi`.
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
        let bit_depth = profile.bit_depth;
        // VAAPI has no H.264 4:4:4 encode profile across any driver we
        // target (Sunshine confirms the same — see
        // refs/Sunshine/src/platform/linux/vaapi.cpp:202). Refuse early
        // rather than producing a confusing FFmpeg open() failure. This
        // is a libva spec-level absence (no VAProfile enum entry), not
        // a driver-version question — keep it short-circuited rather
        // than feeding it to the probe.
        if kind == CodecKind::H264 && chroma == ChromaSubsampling::Yuv444 {
            return Err(CodecError::CodecNotFound(
                "VAAPI does not expose an H.264 4:4:4 encode profile",
            ));
        }
        // Map (chroma, bit_depth) to the FFmpeg pix_fmt used for both
        // the hwframes context's `sw_format` and the BGRA→YUV swscale
        // destination. Unsupported combinations (no FFmpeg pix_fmt for
        // that input shape) return a clean `UnsupportedInputFormat`;
        // probes record `encode=false`. Whether a *driver* accepts the
        // mapped pix_fmt at `encoder.open()` time is a separate
        // question the probe answers empirically by attempting the
        // open — no documented VAAPI driver exposes 10-bit encode for
        // any of these as of writing, but the probe will discover the
        // day that changes without code changes here.
        let sw_format = vaapi_sw_format(chroma, bit_depth)?;

        let codec_cname = vaapi_codec_cname(kind)?;
        let codec = AVCodec::find_encoder_by_name(codec_cname)
            .ok_or(CodecError::CodecNotFound(vaapi_codec_name(kind)))?;

        // Default VAAPI device (typically /dev/dri/renderD128). None
        // lets FFmpeg pick — explicit device strings only matter on
        // multi-GPU systems, which we'll handle when a user with that
        // setup hits us up.
        let hw_device = AVHWDeviceContext::create(ffi::AV_HWDEVICE_TYPE_VAAPI, None, None, 0)?;

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
        let gop_frames =
            fps_i32.saturating_mul(i32::try_from(GOP_SECONDS).expect("GOP_SECONDS fits in i32"));
        encoder.set_gop_size(gop_frames);
        encoder.set_max_b_frames(0);

        // Build + attach the hwframes context. The `sw_format` decides
        // the on-device pixel layout (see `vaapi_sw_format` above).
        // 4:4:4 8-bit uses VUYX (packed X|Y|U|V 8:8:8:8, 32 bpp)
        // rather than planar YUV444P — packed is the only 4:4:4 8-bit
        // format ffmpeg's `vaapi_drm_format_map` can import via
        // DRM_PRIME on the gpuconvert→encoder hop (no entry exists
        // for planar YUV444P or its R8/YU24 DRM encodings).
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
            // Chroma siting. AVCHROMA_LOC_CENTER (= HEVC
            // chroma_sample_loc_type=1, JPEG/MPEG-1 siting) matches
            // what the gpuconvert shaders produce: a 2x2 box-average
            // places the U/V sample at the geometric centre of the
            // 2x2 luma block (`bgra_to_nv12.wgsl:104`). Without this
            // the SPS VUI omits the chroma_loc fields and decoders
            // default to type-0 (MPEG-2 left/co-sited), introducing
            // a half-pixel chroma offset on saturated edges. 4:4:4
            // has no chroma subsampling so the field is ignored
            // there; setting it unconditionally is harmless.
            raw.chroma_sample_location = ffi::AVCHROMA_LOC_CENTER;
            // HEVC profile pin. Sunshine's reference pattern (see
            // refs/Sunshine/src/video.cpp:1687) sets `profile` on the
            // context field rather than via the `profile=` AVOption
            // string. FFmpeg derives the bit depth and chroma of the
            // VAProfile match from the hwframes `sw_format` descriptor
            // (`vaapi_encode.c` walks `vaapi_encode_h265_profiles[]`
            // keyed on depth + components), and treats `avctx->profile`
            // as an extra filter that's skipped when AV_PROFILE_UNKNOWN.
            // So these pins are *defensive*, not corrective — FFmpeg
            // already reaches the right VAProfile from sw_format alone:
            //   - REXT is the only `av_profile` on the two 4:4:4 rows
            //     (`{REXT,8,...,Main444}` / `{REXT,10,...,Main444_10}`),
            //     so packed 4:4:4 selects it regardless. We pin it to
            //     keep the choice explicit across FFmpeg versions.
            //   - MAIN_10 for Yuv420 10-bit. (The MAIN row is rejected
            //     on the depth check before the profile compare, so a
            //     10-bit sw_format can't actually fall through to MAIN;
            //     pinning just documents intent.)
            // For Yuv420 8-bit we leave the field at FFmpeg's default
            // and let the `profile=main` AVOption (below) drive it.
            if kind == CodecKind::Hevc {
                if matches!(chroma, ChromaSubsampling::Yuv444) {
                    raw.profile = ffi::AV_PROFILE_HEVC_REXT as i32;
                } else if matches!(chroma, ChromaSubsampling::Yuv420) && bit_depth == 10 {
                    raw.profile = ffi::AV_PROFILE_HEVC_MAIN_10 as i32;
                }
            }
            // AV1 Main profile (Profile 0) covers 4:2:0 at both 8 and
            // 10-bit; FFmpeg's `av1_vaapi` infers the bit depth from
            // the hwframes `sw_format` (NV12 vs P010LE). Pin the
            // context field so the choice is deterministic across
            // driver versions rather than relying on the FFmpeg
            // auto-pick heuristic.
            if kind == CodecKind::Av1 {
                raw.profile = ffi::AV_PROFILE_AV1_MAIN as i32;
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
        // Encoder min-QP floor: Sunshine's defaults (H.264=19,
        // HEVC=23) bound the visual quality on static desktop frames
        // where VBR alone over-compresses. Surfaced via env var so a
        // hardware test session can validate before we bake it into
        // the default config; production stays unset until verified.
        //
        // **Observed driver gap.** On Intel iHD (Meteor Lake), `qmin`
        // appears consumed by AVCodecContext (does not appear in
        // `unused_avoptions()`) but produces no bitstream-level
        // effect. The mechanism: FFmpeg's `vaapi_encode_init_rate_control`
        // builds `VAEncMiscParameterRateControl` (incl. min_qp /
        // max_qp fields) exactly once at `init()` time; on iHD the
        // VBR rate-control pathway doesn't read those min/max fields
        // from the misc buffer back into per-picture QP selection.
        // Verified across two bitrate targets (10 Mbps and 500 kbps)
        // — qmin=1 and qmin=45 produce byte-identical bitstreams at
        // both, ruling out "bitrate headroom masked the difference."
        // The hardware test `vaapi_min_qp_floor_reduces_bitstream`
        // SKIPs on the byte-count-ratio signal and will start
        // asserting the floor the moment any driver + ffmpeg combo
        // plumbs qmin through to the rate-control choice.
        let default_min_qp_c = match kind {
            CodecKind::H264 => c"19",
            CodecKind::Hevc => c"23",
            CodecKind::Av1 => c"32",
        };
        let mut dict_builder = AVDictionary::new(c"async_depth", c"1", 0)
            .set(c"rc_mode", c"VBR", 0)
            .set(c"idr_interval", c"2147483647", 0);
        // `sei` is an H.264/HEVC-only AVOption — it controls Annex-B
        // SEI prefix NAL emission, which has no AV1 analogue (AV1
        // carries metadata as OBUs, not SEI). `av1_vaapi` does not
        // expose the option, so setting it for AV1 would land in
        // `unused_avoptions()` and trigger the noisy "encoder ignored
        // some private options" warn! log on every encoder
        // construction. Gate it on the codec rather than relying on
        // the unused-options sink.
        if kind != CodecKind::Av1 {
            dict_builder = dict_builder.set(c"sei", c"0", 0);
        }
        if let Ok(qmin_str) = std::env::var("TETHER_MIN_QP") {
            if let Ok(qmin) = qmin_str.parse::<u32>() {
                if qmin > 0 && qmin <= 51 {
                    let qmin_cstr = std::ffi::CString::new(qmin.to_string())
                        .expect("integer string has no NULs");
                    dict_builder = dict_builder.set(c"qmin", qmin_cstr.as_c_str(), 0);
                    tracing::info!(
                        qmin,
                        "min-QP floor set via TETHER_MIN_QP (untested on hardware as of this writing)"
                    );
                }
            }
        } else {
            // Apply the Sunshine default. Keep behind a separate
            // env-var gate so the bake-in lands deliberately rather
            // than as a hidden defaults change on a routine upgrade.
            if std::env::var("TETHER_MIN_QP_DEFAULTS").is_ok() {
                dict_builder = dict_builder.set(c"qmin", default_min_qp_c, 0);
                tracing::info!(
                    codec = ?kind,
                    qmin = default_min_qp_c.to_string_lossy().as_ref(),
                    "min-QP floor set to per-codec default via TETHER_MIN_QP_DEFAULTS"
                );
            }
        }

        // Intra refresh: spreads intra-coded macroblocks across a
        // refresh period instead of one oversized IDR. Reads
        // `TETHER_INTRA_REFRESH_PERIOD` (frame count) on every
        // encoder construction. Off by default.
        //
        // **Observed driver gap.** On Intel iHD (Meteor Lake), the
        // `intra_refresh` option appears in `open()`'s leftover dict
        // — the `h264_vaapi` encoder did not consume it, and the
        // bitstream stays IDR-paced at the default GOP. Whether the
        // gap is in ffmpeg's VAAPI encoder wrapper (option not
        // registered as a private AVOption for this encoder) or in
        // the underlying VAEntrypoint (driver did not advertise a
        // gradual-refresh capability) is unconfirmed; resolving it
        // means reading `libavcodec/vaapi_encode.c` /
        // `vaapi_encode_h264.c` against the FFmpeg version in use.
        // The hardware test `vaapi_intra_refresh_round_trip` SKIPs
        // when the option lands in `unused_avoptions()`, and asserts
        // the full bitstream-shape contract the moment a driver +
        // ffmpeg combination consumes the option.
        //
        // Kept env-var-gated and live so (a) the seam exists for
        // drivers that *do* land support, and (b) the diagnostic
        // path through `unused_avoptions` keeps surfacing the gap
        // loudly in logs rather than silently degrading.
        if let Ok(period_str) = std::env::var("TETHER_INTRA_REFRESH_PERIOD") {
            if let Ok(period) = period_str.parse::<u32>() {
                if period > 0 {
                    let period_cstr = std::ffi::CString::new(period.to_string())
                        .expect("integer string has no NULs");
                    dict_builder = dict_builder.set(c"intra_refresh", period_cstr.as_c_str(), 0);
                    tracing::info!(
                        period,
                        "intra refresh enabled via TETHER_INTRA_REFRESH_PERIOD (untested on hardware as of this writing)"
                    );
                }
            }
        }
        let dict = match (chroma, bit_depth, kind) {
            // H.264 / HEVC 4:2:0 8-bit: pin Main profile via AVOption —
            // broadly compatible default the encoder picks up regardless
            // of driver capability advertisements.
            (ChromaSubsampling::Yuv420, 8, CodecKind::H264 | CodecKind::Hevc) => {
                dict_builder.set(c"profile", c"main", 0)
            }
            // AV1 (any depth), HEVC 4:2:0 10-bit, any 4:4:4: profile
            // pinned via the context-field assignment above (AV1 Main /
            // HEVC Main10 / HEVC REXT). The `profile=` AVOption string
            // would override that field; skip it.
            _ => dict_builder,
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
        let mut unused_avoptions: Vec<String> = Vec::new();
        if let Some(unused) = leftover {
            // Driver/encoder didn't recognise one or more opts. Not
            // fatal — the encoder will still work, just with the
            // unrecognised setting at its default. Surfacing the keys
            // helps diagnose "why is latency higher than expected?"
            //
            // Stored as bare option keys (not `key=value` pairs) so
            // callers can do straightforward presence checks; the
            // values still appear in the warn! log below for full
            // diagnostic context.
            let mut unused_pairs: Vec<(String, String)> = Vec::new();
            for entry in unused.iter() {
                let key = entry.key().to_string_lossy().into_owned();
                let value = entry.value().to_string_lossy().into_owned();
                unused_avoptions.push(key.clone());
                unused_pairs.push((key, value));
            }
            if !unused_pairs.is_empty() {
                warn!(
                    codec = vaapi_codec_name(kind),
                    unused = ?unused_pairs,
                    "VAAPI encoder ignored some private options; latency knobs may not be applied"
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

        // Pin swscale's RGB→YUV matrix to BT.709 limited so the
        // encoded bytes match the VUI we wrote above (lines 210-213).
        // Default behaviour picks BT.601 for sources shorter than 576
        // lines and BT.709 above — a silent hue shift at 720p that the
        // bench-path encode would otherwise hit on test fixtures
        // smaller than 576 lines. Production sessions never run this
        // path (the live host pipeline uses `encode_vaapi_dma_buf`,
        // which bypasses swscale entirely), but the bench / test cells
        // do, and a BT.601-matrixed test fixture mis-tagged BT.709
        // would skew measured PSNR / SSIM.
        //
        // SAFETY: same shape as the VideoToolbox sibling
        // (`videotoolbox/encoder.rs:265-277`). Coefficient tables are
        // static, owned by libswscale; never freed by us.
        // brightness=0, contrast/saturation=65536 is the documented
        // neutral default.
        unsafe {
            let coeffs = ffi::sws_getCoefficients(ffi::SWS_CS_ITU709 as i32);
            let rc = ffi::sws_setColorspaceDetails(
                bgra_to_encoder_input.as_mut_ptr(),
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

        // Snapshot the encoder's parameter-set extradata so we can
        // prepend it to every keyframe at drain time. For
        // h264_vaapi/hevc_vaapi at Main this is the Annex-B SPS/PPS
        // (HEVC also includes VPS). See `encoder_common` for why we
        // do this.
        let extradata = snapshot_extradata(&encoder, vaapi_codec_name(kind), kind)?;

        Ok(Self {
            kind,
            chroma,
            bit_depth,
            encoder,
            bgra_to_encoder_input,
            sw_frame,
            bgra_frame,
            extradata,
            _hw_device: hw_device,
            width,
            height,
            bgra_row_bytes,
            unused_avoptions,
        })
    }

    /// AVOption keys that the driver did not consume at `open()`. A
    /// non-empty list means the corresponding latency knobs (intra
    /// refresh, qmin, etc.) silently fell back to encoder defaults —
    /// useful for tests that need to assert an opt-in setting actually
    /// took effect on the current driver.
    #[must_use]
    pub fn unused_avoptions(&self) -> &[String] {
        &self.unused_avoptions
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
        // DRM fourccs for the four pixel formats the encoder accepts,
        // keyed on `(chroma, bit_depth)`. The dma-buf bridge must hand
        // us a frame matching the negotiated profile — otherwise
        // av_hwframe_map would fail mid-pipeline with a much less
        // actionable error.
        let expected_fourcc = expected_dmabuf_fourcc(self.chroma, self.bit_depth)
            .ok_or(CodecError::UnsupportedInputFormat)?;
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
        if frame.objects.len() > AV_DRM_MAX_PLANES || frame.layers.len() > AV_DRM_MAX_PLANES {
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
        let mut desc: Box<AVDRMFrameDescriptor> = Box::new(unsafe { std::mem::zeroed() });
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
            let num_planes = (layer.num_planes as usize).min(AV_DRM_MAX_PLANES);
            for (p, plane) in planes.iter_mut().enumerate().take(num_planes) {
                *plane = AVDRMPlaneDescriptor {
                    object_index: i32::try_from(layer.object_index[p])
                        .expect("object_index fits in i32"),
                    offset: isize::try_from(layer.offset[p]).expect("offset fits in isize"),
                    pitch: isize::try_from(layer.pitch[p]).expect("pitch fits in isize"),
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
        unsafe extern "C" fn drm_desc_free(_opaque: *mut std::ffi::c_void, data: *mut u8) {
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
            return Err(CodecError::Ffmpeg(RsmpegError::from(ffi::AVERROR(
                ffi::ENOMEM,
            ))));
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
            // accepts, or a layer fourcc absent from FFmpeg's
            // `vaapi_drm_format_map`. Log the descriptor shape so field
            // bugs are tractable without re-running with AV_LOG_DEBUG.
            // Inside a probe (the host startup capability probe exercises
            // every PROFILE_PREFERENCE entry, some of which legitimately
            // aren't supported on a given driver) drop to debug so the
            // expected failure doesn't surface as a scary warning on
            // every host launch.
            if crate::av_log::probe_suppression_active() {
                debug!(
                    rc,
                    fourcc = format_args!("{:08x}", frame.fourcc),
                    num_objects = frame.objects.len(),
                    num_layers = frame.layers.len(),
                    modifier = frame.objects.first().map(|o| o.drm_format_modifier),
                    "av_hwframe_map(DRM_PRIME -> VAAPI) failed (probe; expected on some drivers)"
                );
            } else {
                warn!(
                    rc,
                    fourcc = format_args!("{:08x}", frame.fourcc),
                    num_objects = frame.objects.len(),
                    num_layers = frame.layers.len(),
                    modifier = frame.objects.first().map(|o| o.drm_format_modifier),
                    "av_hwframe_map(DRM_PRIME -> VAAPI) failed"
                );
            }
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
        // CPU-upload path: the BGRA→sw_format swscale stage here is
        // 8-bit-only by design. Adding a 10-bit swscale path would
        // need its own conversion table + verification that swscale's
        // P010LE output lands in the right MSB-align convention; the
        // gpuconvert P010 bridge is the production answer for 10-bit
        // and the CPU-upload path is only used by the test pattern
        // and the bench harness. Probe layer also skips this method
        // for 10-bit (see `vaapi::probe::probe_encode`); the
        // production send loop never reaches here for a 10-bit
        // session. If anyone does, fail explicitly rather than ship
        // 8-bit-source-as-10-bit.
        //
        // When a 10-bit CPU-upload path is eventually needed (e.g. for
        // the bench harness to report Main10 numbers), add a
        // `(Yuv420, 10)` swscale branch + matching sw_frame allocation
        // here and re-enable 10-bit probing through `encode_bgra` in
        // `probe.rs::probe_encode`. Do not just delete this guard —
        // the swscale conversion needs the bit-depth-aware
        // configuration too.
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
        // and upload the sw_frame bytes via av_hwframe_transfer_data
        // (CPU memcpy into GPU memory; sw_frame's pix_fmt was picked
        // by `vaapi_sw_format` from the negotiated (chroma, bit_depth)).
        // The pool blocks if all surfaces are still in flight at the
        // encoder, which with async_depth=1 + 8 surfaces basically
        // never happens.
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

    // `supports_changing_bitrate` and `set_bitrate_kbps` deliberately
    // not overridden. The trait defaults (`false` / Ok-no-op) are
    // load-bearing: the rate-control misc buffer in
    // `vaapi_encode_init_rate_control` is built once at `init()` and
    // not re-emitted, so post-open `AVCodecContext.bit_rate` writes
    // mutate the field but nothing downstream observes them.
    // Verified on Intel iHD Meteor Lake via
    // `vaapi_bitrate_retune_changes_bitstream_size`: retuning 1 Mbps
    // → 20 Mbps produces a 1.13× ratio (would be 5–10× if honoured).
    // Effective retunes would require either an upstream FFmpeg fix
    // that wires the `reconfigure` callback for `bit_rate` and
    // re-emits the misc buffer, or a direct libva call submitting
    // a fresh `VAEncMiscParameterRateControl` buffer that races
    // with the wrapper's own buffer management — neither path is
    // worth the cost on a backend slated for replacement by NVENC
    // (GH #16) on resilience-sensitive setups. Until then the host
    // reads this predicate and disables ABR entirely on VAAPI
    // hosts. A future driver/wrapper that plumbs live retune
    // through end-to-end can override this pair; the hw test
    // becomes the green-light gate.

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
        CodecKind::Av1 => Ok(c"av1_vaapi"),
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

/// FFmpeg pix_fmt for the hwframes `sw_format` (and the matching
/// BGRA→YUV swscale destination + staging `sw_frame`) at the given
/// `(chroma, bit_depth)`. Mirrors `videotoolbox::encoder::vt_sw_format`
/// in spirit: every combination FFmpeg has a pix_fmt for is mapped;
/// `encoder.open()` is then the authority on whether the live VAAPI
/// driver accepts that pix_fmt as encode input. Unmapped combinations
/// return `UnsupportedInputFormat` here — there's nothing to attempt.
///
/// Notes on the 4:4:4 column: VUYX is the *only* 8-bit pix_fmt
/// ffmpeg's `vaapi_drm_format_map` will import via DRM_PRIME, so
/// gpuconvert must produce it for the zero-copy path; the planar
/// alternative YUV444P fails at `av_hwframe_map` time. The 10-bit
/// equivalents (XV30 / X2V10) sit alongside it — VAAPI's
/// `vaapi_drm_format_map` table covers them, but no driver we've
/// found exposes the matching encode profile, so the probe is the
/// only honest signal.
/// The DRM fourcc the VAAPI dma-buf importer (`av_hwframe_map`)
/// expects for a given negotiated (chroma, bit_depth). Single source of
/// truth shared by [`VaapiEncoder::submit_dmabuf`] and the cross-crate
/// consistency tests in `tether-render::dmabuf_test`. Returning `None`
/// for unsupported tuples (Yuv422, Yuv444 12-bit, …) lets the caller
/// surface a clean `UnsupportedInputFormat`.
///
/// Field reference:
/// - `NV12` — 4:2:0 8-bit biplanar; AV_PIX_FMT_NV12.
/// - `P010` — 4:2:0 10-bit biplanar (R16 + RG32 layers; 10 bits
///   MSB-aligned in 16-bit cells); AV_PIX_FMT_P010LE. See
///   `build_p010_dmabuf_frame` for why the UV layer is RG32 not GR32.
/// - `XYUV` — 4:4:4 8-bit packed 32 bpp (V,U,Y,X bytes LE);
///   AV_PIX_FMT_VUYX. The only 4:4:4 8-bit format in
///   `vaapi_drm_format_map`.
/// - `XV30` — 4:4:4 10-bit packed 10:10:10:2 ("X:Cr:Y:Cb": bits[9:0]=Cb,
///   bits[19:10]=Y, bits[29:20]=Cr); AV_PIX_FMT_XV30LE. The
///   entry is preserved so adding an XV30 bridge later doesn't need
///   to touch this table.
pub fn expected_dmabuf_fourcc(chroma: ChromaSubsampling, bit_depth: u8) -> Option<u32> {
    Some(match (chroma, bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => u32::from_le_bytes(*b"NV12"),
        (ChromaSubsampling::Yuv420, 10) => u32::from_le_bytes(*b"P010"),
        (ChromaSubsampling::Yuv444, 8) => u32::from_le_bytes(*b"XYUV"),
        (ChromaSubsampling::Yuv444, 10) => u32::from_le_bytes(*b"XV30"),
        _ => return None,
    })
}

fn vaapi_sw_format(chroma: ChromaSubsampling, bit_depth: u8) -> Result<i32> {
    Ok(match (chroma, bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => ffi::AV_PIX_FMT_NV12,
        (ChromaSubsampling::Yuv420, 10) => ffi::AV_PIX_FMT_P010LE,
        (ChromaSubsampling::Yuv444, 8) => ffi::AV_PIX_FMT_VUYX,
        // FFmpeg exposes both packed (XV30) and planar (X2V10) 4:4:4
        // 10-bit; VAAPI's vaapi_drm_format_map carries XV30. P410LE
        // is biplanar and not in the DRM map, so we don't pick it
        // here for VAAPI parity.
        (ChromaSubsampling::Yuv444, 10) => ffi::AV_PIX_FMT_XV30LE,
        _ => return Err(CodecError::UnsupportedInputFormat),
    })
}
