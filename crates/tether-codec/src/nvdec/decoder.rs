//! NVDEC-accelerated video decoder. Codec-parameterized via [`CodecKind`]:
//! H.264 selects the `h264` decoder, HEVC the `hevc` decoder, AV1 the `av1`
//! decoder — each with a CUDA `hw_device_ctx` and a `get_format` callback
//! pinning `AV_PIX_FMT_CUDA`, which is what engages NVDEC. Structurally a
//! twin of [`crate::vaapi::VaapiDecoder`] with CUDA in place of VAAPI: D2
//! exports the decoded surface as an NV12 dma-buf the renderer imports
//! directly, rather than downloading to system memory.

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{AVFrame, AVHWDeviceContext};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use rsmpeg::UnsafeDerefMut;

use tether_protocol::control::CodecKind;

use crate::h264::packet_from_bytes;
use crate::nvenc::ffi as nvffi;
use crate::vaapi::DECODE_EXTRA_HW_FRAMES;
use crate::{
    build_nv12_dmabuf_frame, init_ffmpeg, CodecError, Decoder, Frame, GpuFrame, GpuFrameSource,
    Result,
};

use super::surface_pool::{NvdecSurfacePool, PooledSurface};

/// NVDEC-accelerated video decoder. Uses FFmpeg's generic `h264` / `hevc` /
/// `av1` decoder with NVDEC hwaccel selected via `get_format` returning
/// `AV_PIX_FMT_CUDA`. D2: decoded CUDA surfaces are EGL-imported into an NV12
/// dma-buf and handed to the renderer as a zero-copy [`Frame::Gpu`].
pub struct NvdecDecoder {
    kind: CodecKind,
    decoder: AVCodecContext,
    /// FFmpeg's `CUcontext` for this decoder, read once from the CUDA
    /// `AVHWDeviceContext`. The EGL→CUDA import in [`Self::export_cuda_frame`]
    /// registers the NV12 surface dma-buf against this exact context so the
    /// device→device copy from the NVDEC frame is on the same context the
    /// decoded surface lives in.
    ///
    /// A raw pointer field; covered by the `unsafe impl Send` below (the
    /// CUcontext is owned by `_hw_device`, dropped after the decoder, and only
    /// touched under `&mut self`).
    cuda_ctx: nvffi::CuContext,
    /// NV12 dma-buf surface pool, lazily created on the first decoded frame
    /// once the real dimensions are known. `None` until then. Rebuilt if a
    /// later frame's dims change (resolution switch / stream restart).
    pool: Option<NvdecSurfacePool>,
    // Decoder holds a cloned (ref-counted) handle to the device internally
    // via set_hw_device_ctx; keeping our own ref documents the lifetime
    // relationship. Drop order is declaration order, so `decoder` (and any
    // CUDA surfaces allocated against the device) tears down before
    // `_hw_device`. Same contract as VaapiDecoder / NvencEncoder.
    _hw_device: AVHWDeviceContext,
}

// SAFETY: same move-fine / share-bad rationale as VaapiDecoder. The CUDA
// device context and codec context are safe to MOVE between threads but not
// to SHARE; only `&mut self` methods are exposed, so the borrow checker
// serialises access within a single thread. `cuda_ctx` is a raw pointer into
// `_hw_device`; `pool` is itself `Send`.
unsafe impl Send for NvdecDecoder {}

impl NvdecDecoder {
    /// Construct an NVDEC decoder for the given codec.
    ///
    /// Returns `Err(CodecError::CodecNotFound)` when the linked FFmpeg has no
    /// decoder advertising the CUDA (NVDEC) hwaccel for that codec (the
    /// artifact must be built with NVDEC/cuvid support), and any
    /// `RsmpegError` when the CUDA device can't be created (no NVIDIA GPU /
    /// driver, `libcuda.so` / `libnvcuvid.so` absent) or `open()` rejects the
    /// configuration.
    pub fn new(kind: CodecKind) -> Result<Self> {
        init_ffmpeg();

        let codec_id = nvdec_av_codec_id(kind);
        let codec =
            AVCodec::find_decoder(codec_id).ok_or(CodecError::CodecNotFound(nvdec_name(kind)))?;

        // Walk the decoder's HW config table to confirm NVDEC (CUDA) is
        // actually supported in this FFmpeg build before we open the context.
        // Without this probe an unsupported config would fail later at
        // decoder.open() with a less actionable error. Mirrors the VAAPI
        // sibling's hwaccel probe.
        let mut cuda_supported = false;
        for i in 0.. {
            let Some(config) = codec.hw_config(i) else {
                break;
            };
            #[allow(clippy::cast_possible_wrap)] // single-bit constant
            let supports_device_ctx =
                config.methods & ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32 != 0;
            if supports_device_ctx && config.device_type == ffi::AV_HWDEVICE_TYPE_CUDA {
                cuda_supported = true;
                break;
            }
        }
        if !cuda_supported {
            return Err(CodecError::CodecNotFound(nvdec_name(kind)));
        }

        // Default CUDA device (GPU 0). None lets FFmpeg pick; explicit device
        // strings only matter on multi-GPU systems. Same call as NvencEncoder.
        let hw_device = AVHWDeviceContext::create(ffi::AV_HWDEVICE_TYPE_CUDA, None, None, 1)?;

        // Read FFmpeg's CUcontext out of the CUDA AVHWDeviceContext so the
        // zero-copy NV12 dma-buf export registers EGL images against the exact
        // context NVDEC decoded into. Same navigation as NvencEncoder::new:
        //   AVBufferRef::data → AVHWDeviceContext::hwctx → AVCUDADeviceContext::cuda_ctx
        //
        // SAFETY: `create` returned a valid CUDA device-context buffer.
        // `data` points at an `AVHWDeviceContext`, whose `hwctx` points at an
        // `AVCUDADeviceContext` (our `AvCudaDeviceContext`, prefix-ABI pinned
        // by the const asserts in `nvffi`). We only read `cuda_ctx`.
        let cuda_ctx: nvffi::CuContext = unsafe {
            let buf_ref = hw_device.as_ptr();
            let device_ctx = (*buf_ref).data as *const ffi::AVHWDeviceContext;
            let cuda_device_ctx = (*device_ctx).hwctx as *const nvffi::AvCudaDeviceContext;
            (*cuda_device_ctx).cuda_ctx
        };

        let mut decoder = AVCodecContext::new(&codec);
        // Cloning an AVHWDeviceContext is an av_buffer_ref under the hood; the
        // decoder gets a fresh ref-counted handle while we keep our own owner.
        decoder.set_hw_device_ctx(hw_device.clone());
        decoder.set_get_format(Some(get_cuda_format));
        // Size the internal hwframes pool with headroom for surfaces in flight,
        // matching the VAAPI decoder. rsmpeg doesn't wrap this field;
        // AVCodecContext derefs to ffi::AVCodecContext so we poke it directly.
        //
        // SAFETY: extra_hw_frames must be written before avcodec_open2 — that's
        // the libavcodec ordering invariant the unsafe here attests to.
        unsafe {
            decoder.deref_mut().extra_hw_frames = DECODE_EXTRA_HW_FRAMES;
        }
        decoder.open(None)?;

        Ok(Self {
            kind,
            decoder,
            cuda_ctx,
            pool: None,
            _hw_device: hw_device,
        })
    }

    /// Signal end-of-stream so the decoder emits any buffered frames. FFmpeg's
    /// HEVC (and AV1) decoders hold the first frame in the reorder DPB until a
    /// subsequent packet or EOF arrives; a lone IDR (e.g. the capability probe)
    /// needs this explicit drain to emit. After this, `next_frame` drains until
    /// it returns `None`. Mirrors `VaapiDecoder::signal_eof`.
    pub fn signal_eof(&mut self) -> Result<()> {
        self.decoder.send_packet(None)?;
        Ok(())
    }

    /// Export a decoded `AV_PIX_FMT_CUDA` frame as a zero-copy NV12 dma-buf.
    ///
    /// Acquires a free pool surface, EGL-imports its dma-buf into CUDA, and
    /// `cuMemcpy2D`s the NVDEC frame's Y and UV planes into the imported
    /// surface (device→device, on the decoder's CUDA context). The decoded
    /// pixels are then *in the surface*, so the NVDEC `AVFrame` and the
    /// EGL/CUDA import resources are released immediately; only the pool slot
    /// stays reserved (via the `GpuFrameGuard`) until the renderer drops the
    /// frame. The renderer imports the resulting NV12 dma-buf exactly as it
    /// does the VAAPI decoder's.
    ///
    /// 10-bit (P010) is not yet wired here — the pool only allocates NV12
    /// (8-bit) surfaces. A 10-bit decode would need a P010 surface pool +
    /// `build_p010_dmabuf_frame`; deferred until the 10-bit NVDEC profile is
    /// negotiated. For now non-NV12 sw_format reports `UnsupportedInputFormat`
    /// rather than silently mis-copying.
    // ffmpeg's i32 ABI fields (width, height, linesize) are non-negative on
    // allocated decoded frames; cast sites are at the FFI boundary and follow
    // that invariant. Same rationale as the VAAPI sibling.
    #[allow(clippy::cast_sign_loss)]
    fn export_cuda_frame(&mut self, frame: &AVFrame) -> Result<Frame> {
        let w = frame.width as u32;
        let h = frame.height as u32;
        let pts_out = if frame.pts == ffi::AV_NOPTS_VALUE {
            None
        } else {
            Some(frame.pts)
        };

        // The CUDA hwframes sw_format tells us the on-device layout. Only NV12
        // (8-bit) is wired on the pool; reject other layouts (P010 10-bit)
        // loudly rather than copying NV12-shaped from a P010 surface.
        if !self.frame_is_nv12(frame) {
            return Err(CodecError::UnsupportedInputFormat);
        }

        // (Re)build the pool when dims change or on first frame. The decoded
        // frame's width/height are authoritative.
        if self.pool.as_ref().map(NvdecSurfacePool::dims) != Some((w, h)) {
            self.pool = Some(NvdecSurfacePool::new(w, h)?);
        }
        let pool = self.pool.as_ref().expect("pool just set");

        // Acquire a free surface. None means every surface is still held by
        // the renderer — fail cleanly (the parent may add back-pressure).
        let surface = pool.acquire().ok_or_else(|| {
            CodecError::NoHardwareCodec(
                "NVDEC surface pool exhausted — every NV12 surface still held by the renderer"
                    .to_string(),
            )
        })?;

        self.fill_surface(frame, &surface)?;

        // Build the OUTPUT dma-buf frame from a *fresh* dup of the surface's
        // fd (the renderer's importer owns its dup; the surface keeps its own).
        let out_fd = surface.fd.try_clone().map_err(|e| {
            CodecError::NoHardwareCodec(format!("NVDEC: dup surface fd for output frame: {e}"))
        })?;
        let out = build_nv12_dmabuf_frame(
            out_fd,
            surface.size,
            surface.modifier,
            surface.y_offset,
            surface.y_pitch,
            surface.uv_offset,
            surface.uv_pitch,
        );

        // The guard is the pool-slot release handle: dropping the GpuFrame
        // frees the slot back to the pool. The decoded pixels are already
        // copied into the surface's dma-buf, so the NVDEC AVFrame doesn't need
        // to outlive this call — the surface memory (kept alive by the pool) is
        // what backs the export. `SlotGuard` is `Send + 'static`, exactly the
        // `GpuFrameGuard` contract; no AVFrame wrapper needed (unlike VAAPI,
        // which stashes the surface-owning AVFrame).
        Ok(Frame::Gpu(GpuFrame::new(
            w,
            h,
            pts_out,
            GpuFrameSource::DmaBuf(out),
            surface.release,
        )))
    }

    /// EGL-import the surface dma-buf into CUDA and copy the NVDEC frame's
    /// planes into it (device→device). Factored out so `export_cuda_frame`
    /// reads top-down.
    // linesize is a non-negative i32 ABI field on allocated frames.
    #[allow(clippy::cast_sign_loss)]
    fn fill_surface(&self, frame: &AVFrame, surface: &PooledSurface) -> Result<()> {
        let importer = nvffi::importer().ok_or_else(|| {
            CodecError::NoHardwareCodec("EGL/CUDA import unavailable".to_string())
        })?;

        // Build a DmaBufFrame describing the surface and import it into CUDA.
        // A dup of the surface fd is consumed by EGL; the surface keeps its own
        // copy (and the output frame gets a third dup), so the import dup is
        // independent.
        let import_fd = surface.fd.try_clone().map_err(|e| {
            CodecError::NoHardwareCodec(format!("NVDEC: dup surface fd for EGL import: {e}"))
        })?;
        let surface_dmabuf = build_nv12_dmabuf_frame(
            import_fd,
            surface.size,
            surface.modifier,
            surface.y_offset,
            surface.y_pitch,
            surface.uv_offset,
            surface.uv_pitch,
        );

        let imported = importer
            .import(self.cuda_ctx, &surface_dmabuf, surface.width, surface.height)
            .map_err(CodecError::NoHardwareCodec)?;

        // Copy each NVDEC plane INTO the imported surface plane. NV12 8-bit:
        // plane 0 = Y, full height, `w` bytes/row; plane 1 = interleaved UV,
        // half height, `w` bytes/row (w/2 CbCr pairs × 2 bytes). The source is
        // the NVDEC frame's device pointer + linesize; the destination is the
        // EGL-imported surface plane (copy_plane_from handles PITCH vs ARRAY).
        let width_bytes = surface.width as usize;
        for i in 0..imported.plane_count() {
            let plane_height = if i == 0 {
                surface.height as usize
            } else {
                (surface.height as usize).div_ceil(2)
            };
            imported
                .copy_plane_from(
                    i,
                    frame.data[i],
                    frame.linesize[i].max(0) as usize,
                    width_bytes,
                    plane_height,
                )
                .map_err(CodecError::NoHardwareCodec)?;
        }
        // Pixels now live in the surface; release the EGL/CUDA registration.
        drop(imported);
        Ok(())
    }

    /// Whether the decoded CUDA frame's on-device layout is NV12 (the only
    /// layout the surface pool supports). Reads the hwframes context's
    /// `sw_format`. A null hwframes ctx would violate the libavutil hwaccel
    /// contract for an `AV_PIX_FMT_CUDA` frame; treat that as "not NV12".
    fn frame_is_nv12(&self, frame: &AVFrame) -> bool {
        // SAFETY: for an AV_PIX_FMT_CUDA frame FFmpeg sets hw_frames_ctx to the
        // pool's buffer ref; `data` points at an AVHWFramesContext whose
        // `sw_format` is the on-device pixel layout. We only read scalar
        // fields. A null ptr (contract violation) short-circuits to false.
        unsafe {
            let hwf_ref = frame.hw_frames_ctx;
            if hwf_ref.is_null() {
                return false;
            }
            let hwf = (*hwf_ref).data as *const ffi::AVHWFramesContext;
            if hwf.is_null() {
                return false;
            }
            (*hwf).sw_format == ffi::AV_PIX_FMT_NV12
        }
    }
}

impl Decoder for NvdecDecoder {
    fn submit(&mut self, encoded: &[u8]) -> Result<()> {
        if encoded.is_empty() {
            return Ok(());
        }
        let packet = packet_from_bytes(encoded)?;
        self.decoder.send_packet(Some(&packet))?;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        let frame = match self.decoder.receive_frame() {
            Ok(f) => f,
            Err(RsmpegError::DecoderDrainError | RsmpegError::DecoderFlushedError) => {
                return Ok(None)
            }
            Err(e) => return Err(CodecError::Ffmpeg(e)),
        };

        // Reaching a non-CUDA frame means FFmpeg decoded into system memory
        // despite our get_format callback insisting on CUDA — a sign the NVDEC
        // hwaccel bailed mid-stream (HW context loss, OOM, unsupported profile
        // slipping through). Silently CPU-decoding contradicts CLAUDE.md's
        // "Hard requirement on hardware codecs" contract, so refuse.
        if frame.format != ffi::AV_PIX_FMT_CUDA {
            return Err(CodecError::UnsupportedInputFormat);
        }

        self.export_cuda_frame(&frame).map(Some)
    }

    fn codec_kind(&self) -> CodecKind {
        self.kind
    }

    fn is_hardware(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        nvdec_name(self.kind)
    }

    fn flush(&mut self) -> Result<()> {
        // avcodec_flush_buffers drops the reorder queue, releases pending
        // output, and resets internal state without tearing the context down —
        // ready to accept a fresh IDR after this. The CUDA surface pool stays
        // allocated; only references are released. Cheap (~µs) and idempotent.
        self.decoder.flush_buffers();
        Ok(())
    }
}

/// FFmpeg `AVCodecID` for the given codec kind.
fn nvdec_av_codec_id(kind: CodecKind) -> ffi::AVCodecID {
    match kind {
        CodecKind::H264 => ffi::AV_CODEC_ID_H264,
        CodecKind::Hevc => ffi::AV_CODEC_ID_HEVC,
        CodecKind::Av1 => ffi::AV_CODEC_ID_AV1,
    }
}

/// Human-readable decoder name for `CodecNotFound` / `Decoder::name`. The
/// `(NVIDIA)` suffix mirrors the NVENC encoder's `name()` style and is honest:
/// this backend only constructs on a confirmed-NVIDIA host.
fn nvdec_name(kind: CodecKind) -> &'static str {
    match kind {
        CodecKind::H264 => "h264_nvdec (NVIDIA)",
        CodecKind::Hevc => "hevc_nvdec (NVIDIA)",
        CodecKind::Av1 => "av1_nvdec (NVIDIA)",
    }
}

/// get_format callback the decoder invokes once the bitstream's SPS/PPS reveal
/// the format options. `pix_fmts` is a null-terminated array of candidates
/// including `AV_PIX_FMT_CUDA` (because we set `hw_device_ctx`). Returning
/// `AV_PIX_FMT_CUDA` tells FFmpeg to allocate CUDA surfaces and engage NVDEC;
/// otherwise `AV_PIX_FMT_NONE` makes the decoder bail rather than fall back to
/// software. Mirrors the VAAPI sibling's `get_vaapi_format`.
unsafe extern "C" fn get_cuda_format(
    _ctx: *mut ffi::AVCodecContext,
    pix_fmts: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    let fmts = unsafe { rsmpeg::build_array(pix_fmts, ffi::AV_PIX_FMT_NONE) }.unwrap_or_default();
    for &fmt in fmts {
        if fmt == ffi::AV_PIX_FMT_CUDA {
            return fmt;
        }
    }
    ffi::AV_PIX_FMT_NONE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h264::{fixture_access_units, H264_320X240_INTRA};

    /// Decode the committed 320×240 H.264 IDR fixture through a real NVDEC
    /// decoder and assert a zero-copy GPU NV12 dma-buf frame of the expected
    /// dims comes back with the right descriptor shape (1 object, 2 layers,
    /// plausible strides). Pixel correctness needs a GPU import, which this
    /// unit test can't do — the harness round-trip tests cover that.
    ///
    /// H.264 is used (not HEVC) because `tether-codec` ships an in-crate H.264
    /// fixture (`H264_320X240_INTRA`); there's no committed HEVC bitstream
    /// loadable from this crate. `#[ignore]` — needs an NVIDIA GPU + an FFmpeg
    /// with NVDEC + a Vulkan dma-buf device.
    #[test]
    #[ignore = "requires NVIDIA GPU + NVDEC FFmpeg + Vulkan dma-buf"]
    fn decodes_committed_h264_fixture_via_nvdec() {
        let w = 320u32;
        let h = 240u32;
        let mut dec = NvdecDecoder::new(CodecKind::H264).expect("nvdec decoder");

        let mut got = None;
        for au in fixture_access_units(H264_320X240_INTRA) {
            dec.submit(au).expect("submit");
            while let Some(f) = dec.next_frame().expect("next_frame") {
                got = Some(f);
            }
        }
        // A lone IDR sits in the DPB until EOF; drain it.
        dec.signal_eof().expect("signal_eof");
        while let Some(f) = dec.next_frame().expect("next_frame") {
            got = Some(f);
        }

        let frame = got.expect("decoder produced a frame from the fixture");
        let Frame::Gpu(gpu) = frame else {
            panic!("D2 NvdecDecoder must export a zero-copy Gpu frame");
        };
        assert_eq!(gpu.width, w);
        assert_eq!(gpu.height, h);

        let (_w, _h, _pts, source, _guard) = gpu.into_parts();
        let GpuFrameSource::DmaBuf(dmabuf) = source;
        // NV12 surface descriptor shape: fourcc 'NV12', one DRM object, two
        // layers (Y as R8, UV as GR88).
        assert_eq!(dmabuf.fourcc, u32::from_le_bytes(*b"NV12"));
        assert_eq!(dmabuf.objects.len(), 1, "NV12 dma-buf is one DRM object");
        assert_eq!(dmabuf.layers.len(), 2, "NV12 dma-buf has Y + UV layers");
        // Strides must at least span a row of each plane (Y: w bytes; UV: w
        // bytes — w/2 CbCr pairs × 2). Alignment may make them larger.
        assert!(
            dmabuf.layers[0].pitch[0] >= w,
            "Y stride {} < width {w}",
            dmabuf.layers[0].pitch[0]
        );
        assert!(
            dmabuf.layers[1].pitch[0] >= w,
            "UV stride {} < width {w}",
            dmabuf.layers[1].pitch[0]
        );

        // Pixel correctness: read the decoded Y plane back off the GPU (EGL
        // re-import the output surface into a fresh CUDA context on the same
        // GPU + cuMemcpy2D device→host) and compare against a software decode
        // of the same fixture. Proves the zero-copy NVDEC → CUDA → dma-buf
        // path delivered the right pixels, not just the right shape.
        let y_stride = dmabuf.layers[0].pitch[0] as usize;
        let nvdec_y = readback_surface_y(&dmabuf, w, h, y_stride);
        let sw_y = sw_decode_h264_y(w, h);
        let mae = y_plane_mae(&nvdec_y, y_stride, &sw_y, w as usize, w as usize, h as usize);
        eprintln!("nvdec vs sw-decode Y MAE = {mae:.3}");
        assert!(
            mae < 3.0,
            "NVDEC zero-copy Y plane differs from the software decode (MAE {mae:.2}) — \
             the dma-buf surface does not hold the correctly-decoded image"
        );
    }

    /// EGL-import the decoded surface into a fresh CUDA context (device 0,
    /// the same GPU the surface lives on) and copy its Y plane to host.
    /// Returns the strided Y buffer (`y_stride * height` bytes).
    fn readback_surface_y(dmabuf: &crate::DmaBufFrame, w: u32, h: u32, y_stride: usize) -> Vec<u8> {
        let cuda_dev = AVHWDeviceContext::create(ffi::AV_HWDEVICE_TYPE_CUDA, None, None, 1)
            .expect("CUDA device for readback");
        // SAFETY: same AVBufferRef::data → AVHWDeviceContext::hwctx →
        // AVCUDADeviceContext::cuda_ctx navigation the decoder uses.
        let cuda_ctx = unsafe {
            let buf = cuda_dev.as_ptr();
            let dctx = (*buf).data as *const ffi::AVHWDeviceContext;
            let cdev = (*dctx).hwctx as *const nvffi::AvCudaDeviceContext;
            (*cdev).cuda_ctx
        };
        let importer = nvffi::importer().expect("EGL/CUDA importer");
        let imported = importer
            .import(cuda_ctx, dmabuf, w, h)
            .expect("re-import output surface");
        let mut y = vec![0u8; y_stride * h as usize];
        imported
            .copy_plane_to_host(0, y.as_mut_ptr(), y_stride, w as usize, h as usize)
            .expect("readback Y plane");
        drop(imported);
        drop(cuda_dev);
        y
    }

    /// Software-decode the committed H.264 fixture and return its tight Y plane.
    fn sw_decode_h264_y(w: u32, h: u32) -> Vec<u8> {
        use crate::Decoder as _;
        let mut dec = crate::h264::H264Decoder::new().expect("sw h264 decoder");
        let mut y = None;
        for au in fixture_access_units(H264_320X240_INTRA) {
            dec.submit(au).expect("sw submit");
            while let Some(f) = dec.next_frame().expect("sw next_frame") {
                if let Frame::Cpu(c) = f {
                    y = Some(c.y);
                }
            }
        }
        let y = y.expect("software decoder produced a frame");
        assert_eq!(y.len(), (w * h) as usize, "sw Y plane size");
        y
    }

    /// Mean absolute difference between a strided Y plane and a tight one,
    /// over the visible `w × h` region.
    fn y_plane_mae(
        a: &[u8],
        a_stride: usize,
        b: &[u8],
        b_stride: usize,
        w: usize,
        h: usize,
    ) -> f64 {
        let mut sum = 0u64;
        for row in 0..h {
            for x in 0..w {
                let da = i32::from(a[row * a_stride + x]);
                let db = i32::from(b[row * b_stride + x]);
                sum += u64::from(da.abs_diff(db));
            }
        }
        sum as f64 / (w * h) as f64
    }
}


