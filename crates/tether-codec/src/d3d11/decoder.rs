//! D3D11VA hardware video decoder. Uses FFmpeg's generic `h264` or
//! `hevc` decoder with D3D11VA hwaccel for GPU-accelerated decode.
//!
//! Decoded D3D11 surfaces are copied to a shared-handle-enabled
//! staging texture, exported via DXGI shared handle, and returned as
//! `Frame::Gpu` for zero-copy import into wgpu's Vulkan backend via
//! `VK_KHR_external_memory_win32`. The one GPU-side CopyResource per
//! frame avoids the PCIe roundtrip of the previous CPU download path.

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{AVFrame, AVHWDeviceContext};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use rsmpeg::UnsafeDerefMut;
use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_DEFAULT,
};
use windows::Win32::Graphics::Dxgi::{IDXGIResource1, DXGI_SHARED_RESOURCE_READ};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_NV12, DXGI_FORMAT_P010, DXGI_SAMPLE_DESC,
};

use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

use crate::h264::packet_from_bytes;
use crate::{
    init_ffmpeg, CodecError, D3D11DecodedTexture, DecodedFrame, Decoder, Frame, GpuFrame,
    GpuFrameSource, Result,
};

const DECODE_EXTRA_HW_FRAMES: i32 = 4;
// Shareable so a separate device (wgpu's Vulkan backend) can open the
// staging texture. `SHARED_NTHANDLE` produces an NT handle via
// `IDXGIResource1::CreateSharedHandle`, which is what Vulkan's
// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT` import expects — the
// legacy `GetSharedHandle` KMT handle is the wrong type for that import.
// Matches Moonlight's cross-device share (d3d11va.cpp). `SHARED` must be
// set alongside `SHARED_NTHANDLE`.
const D3D11_RESOURCE_MISC_SHARED: u32 = 0x2;
const D3D11_RESOURCE_MISC_SHARED_NTHANDLE: u32 = 0x800;
const D3D11_BIND_SHADER_RESOURCE: u32 = 0x8;

/// The `DXGI_FORMAT` (as a raw `u32`) the D3D11VA decoder emits for a
/// given negotiated profile's GPU export, or `None` when Windows has no
/// decode path for it.
///
/// Windows decodes H.264/HEVC 4:2:0 only. The Video Processor and encode
/// path reject 4:4:4 at construction, and `d3d11_av_codec_id` rejects AV1
/// (no D3D11VA AV1 decoder in this build), so a Windows client never
/// advertises either — anything outside `{H.264, HEVC} × 4:2:0 × {8,10}`
/// returns `None`. Note `PROFILE_PREFERENCE` *does* list AV1 4:2:0; this
/// function returning `None` for it is what keeps the cross-crate test
/// from asserting renderer support for a codec the decoder can't open.
///
/// For the formats it does decode, FFmpeg's d3d11va decoder picks the
/// surface format from the bit depth: NV12 for 8-bit, P010 (MSB-aligned
/// 10-bit) for 10-bit. `export_gpu_frame` carries that surface format
/// straight through as `D3D11DecodedTexture::format`, so this function *is*
/// the set of formats the renderer's import path must accept.
///
/// This is the decode-side analog of the macOS
/// `videotoolbox::expected_iosurface_fourccs`. It speaks raw `DXGI_FORMAT`
/// `u32`s — the same neutral currency the renderer's
/// `tether_render::decode_plane_srv_formats` consumes — so the cross-crate
/// agreement test in tether-client needs no `windows` dependency. The
/// co-located `expected_decode_format_agrees_with_encoder_sw_format` test
/// pins this in lock-step with the encoder's `d3d11_sw_format`.
#[allow(clippy::cast_sign_loss)]
pub fn expected_decode_dxgi_format(profile: VideoProfile) -> Option<u32> {
    match (profile.codec, profile.chroma, profile.bit_depth) {
        (CodecKind::H264 | CodecKind::Hevc, ChromaSubsampling::Yuv420, 8) => {
            Some(DXGI_FORMAT_NV12.0 as u32)
        }
        (CodecKind::H264 | CodecKind::Hevc, ChromaSubsampling::Yuv420, 10) => {
            Some(DXGI_FORMAT_P010.0 as u32)
        }
        _ => None,
    }
}

pub struct D3D11Decoder {
    kind: CodecKind,
    decoder: AVCodecContext,
    _hw_device: AVHWDeviceContext,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    staging: Option<PlaneStaging>,
    /// When true we export decoded frames GPU-resident as a shared-handle
    /// `Frame::Gpu` (the native D3D11 renderer opens the handle on its own
    /// device — see `tether_render::d3d11`); when false we download to a
    /// CPU `Frame::Cpu`. GPU-resident is the rule for the live client; the
    /// CPU path is the `download_frame_cpu` fallback for the null-decode-
    /// texture anomaly, and is 8-bit NV12 only.
    gpu_export: bool,
}

/// Single biplanar staging texture for GPU-side plane extraction. The
/// decode pool surface (an NV12 or P010 texture array) is copied
/// slice→staging per plane — `CopySubresourceRegion` only works between
/// same-format subresources, so the staging must itself match the decode
/// format (a separate split-plane destination would make the copy a
/// silent no-op). The consumer opens the shared handle and views plane 0
/// and plane 1 with the format's per-plane SRV formats (R8/R8G8 for NV12,
/// R16/R16G16 for P010).
struct PlaneStaging {
    tex: ID3D11Texture2D,
    /// NT handle from `CreateSharedHandle` — owned by us, closed on drop.
    /// (`tex` is allocated at even-rounded dims; `width`/`height` are the
    /// codec-reported dims used for the resolution-change staleness check
    /// and propagated to the renderer.)
    handle: HANDLE,
    width: u32,
    height: u32,
    /// `DXGI_FORMAT` of the staging texture (NV12 or P010). Part of the
    /// staleness key and reported to the renderer.
    format: DXGI_FORMAT,
}

impl Drop for PlaneStaging {
    fn drop(&mut self) {
        // `CreateSharedHandle` returns an owned NT handle; without this it
        // leaks one kernel handle per resolution change / decoder teardown.
        // The renderer's `OpenSharedResource1` copy is independent and is
        // released when its `ID3D11Texture2D` drops.
        if !self.handle.is_invalid() {
            unsafe { let _ = CloseHandle(self.handle); }
        }
    }
}

unsafe impl Send for D3D11Decoder {}

impl D3D11Decoder {
    /// `gpu_export` is the renderer's D3D11→Vulkan import capability (see
    /// the field doc). The host probe builds a decoder with `false` — it
    /// only checks that decode produces a frame and never renders.
    pub fn new(kind: CodecKind, gpu_export: bool) -> Result<Self> {
        init_ffmpeg();

        let codec_id = d3d11_av_codec_id(kind)?;
        let codec = AVCodec::find_decoder(codec_id)
            .ok_or(CodecError::CodecNotFound(d3d11_decoder_name(kind)))?;

        let mut d3d11va_supported = false;
        for i in 0.. {
            let Some(config) = codec.hw_config(i) else { break };
            #[allow(clippy::cast_possible_wrap)]
            let supports_device_ctx =
                config.methods & ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32 != 0;
            if supports_device_ctx && config.device_type == ffi::AV_HWDEVICE_TYPE_D3D11VA {
                d3d11va_supported = true;
                break;
            }
        }
        if !d3d11va_supported {
            return Err(CodecError::CodecNotFound(d3d11_decoder_name(kind)));
        }

        let hw_device =
            AVHWDeviceContext::create(ffi::AV_HWDEVICE_TYPE_D3D11VA, None, None, 0)?;

        // Extract the D3D11 device FFmpeg created for us.
        let (device, context) = unsafe {
            let buf_ptr = hw_device.as_ptr();
            let data = &*((*buf_ptr).data as *const ffi::AVHWDeviceContext);
            let hwctx = data.hwctx as *const super::encoder::AvD3D11VADeviceContext;
            let dev_ptr = (*hwctx).device;
            let ctx_ptr = (*hwctx).device_context;
            let device: ID3D11Device =
                ID3D11Device::from_raw_borrowed(&dev_ptr)
                    .ok_or(CodecError::CodecNotFound("d3d11va device null"))?
                    .clone();
            let context: ID3D11DeviceContext =
                ID3D11DeviceContext::from_raw_borrowed(&ctx_ptr)
                    .ok_or(CodecError::CodecNotFound("d3d11va context null"))?
                    .clone();
            (device, context)
        };

        let mut decoder = AVCodecContext::new(&codec);
        decoder.set_hw_device_ctx(hw_device.clone());
        decoder.set_get_format(Some(get_d3d11va_format));
        unsafe {
            decoder.deref_mut().extra_hw_frames = DECODE_EXTRA_HW_FRAMES;
        }
        decoder.open(None)?;

        tracing::info!(
            codec = d3d11_decoder_name(kind),
            gpu_export,
            "D3D11VA decoder opened"
        );

        Ok(Self {
            kind,
            decoder,
            _hw_device: hw_device,
            device,
            context,
            staging: None,
            gpu_export,
        })
    }

    /// Signal end-of-stream so the decoder emits any buffered frames.
    /// D3D11VA's wrapper (like VideoToolbox) holds a single-IDR submit
    /// until either another packet arrives or EOF is signalled.
    pub fn signal_eof(&mut self) -> Result<()> {
        self.decoder.send_packet(None)?;
        Ok(())
    }

    /// Export a D3D11VA surface as `Frame::Gpu` backed by a single shared
    /// handle. Copies both planes (Y and UV) from the decode pool slice
    /// into a MISC_SHARED biplanar staging texture (NV12 or P010, matching
    /// the decode format), entirely on the GPU.
    fn export_gpu_frame(&mut self, hw_frame: &AVFrame) -> Result<Frame> {
        let width = hw_frame.width as u32;
        let height = hw_frame.height as u32;
        let pts = if hw_frame.pts == ffi::AV_NOPTS_VALUE {
            None
        } else {
            Some(hw_frame.pts)
        };

        let src_texture_ptr = hw_frame.data[0] as *mut std::ffi::c_void;
        // D3D11VA: array index stored as intptr_t in data[1].
        let array_index = hw_frame.data[1] as usize as u32;

        if src_texture_ptr.is_null() {
            return self.download_frame_cpu(hw_frame);
        }

        let src_texture: ID3D11Texture2D = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&src_texture_ptr)
                .ok_or(CodecError::CodecNotFound("null decode texture"))?
                .clone()
        };

        // Read the decode surface descriptor once: its format drives the
        // staging format (NV12 8-bit vs P010 10-bit) and its ArraySize
        // drives the plane-subresource calc below.
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { src_texture.GetDesc(&mut desc) };
        let src_format = desc.Format;
        let array_size = desc.ArraySize;

        // Rebuild staging if dimensions or the decode format changed. The
        // format only changes across a session/codec restart, but keying
        // on it keeps the staging in lock-step with the decode surface so
        // an 8↔10-bit reconfigure can't sample a stale NV12 staging.
        if self.staging.as_ref().map_or(true, |s| {
            s.width != width || s.height != height || s.format != src_format
        }) {
            self.staging = Some(Self::create_staging(&self.device, width, height, src_format)?);
        }
        let staging = self.staging.as_ref().unwrap();

        // Biplanar texture-array subresource layout is PLANE-MAJOR:
        // D3D11CalcSubresource(plane, slice) =
        // planeSlice * (MipLevels * ArraySize) + arraySlice * MipLevels.
        // With MipLevels == 1 that's Y of slice N = N, UV of slice N =
        // ArraySize + N. We copy each plane of the decode slice into the
        // matching plane subresource of the single-slice staging
        // (subresource 0 = Y, 1 = UV). Both endpoints are same-format
        // plane subresources, so the copy is honoured — copying a plane
        // into a separate split-plane texture is a silent no-op (the
        // all-zero "green screen" bug).
        let y_subresource = array_index;
        let uv_subresource = array_size + array_index;

        unsafe {
            self.context.CopySubresourceRegion(
                &staging.tex, 0, 0, 0, 0,
                &src_texture, y_subresource, None,
            );
            self.context.CopySubresourceRegion(
                &staging.tex, 1, 0, 0, 0,
                &src_texture, uv_subresource, None,
            );
            // Submit the copies so the consumer device sees them.
            gpu_sync(&self.device, &self.context);
        }

        Ok(Frame::Gpu(GpuFrame::new(
            width,
            height,
            pts,
            GpuFrameSource::D3D11Texture(D3D11DecodedTexture {
                handle: staging.handle.0 as *mut std::ffi::c_void,
                width,
                height,
                format: staging.format.0 as u32,
            }),
            hw_frame.clone(),
        )))
    }

    fn create_staging(
        device: &ID3D11Device,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<PlaneStaging> {
        // NV12/P010 require even dimensions; round up so the chroma plane
        // is whole. The decoder declares the real width/height downstream,
        // so a 1px pad on odd inputs is harmless.
        let w = (width + 1) & !1;
        let h = (height + 1) & !1;
        let tex = Self::create_shared_texture(device, w, h, format)?;
        let handle = Self::get_shared_handle(&tex)?;
        Ok(PlaneStaging { tex, handle, width, height, format })
    }

    fn create_shared_texture(
        device: &ID3D11Device,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<ID3D11Texture2D> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE,
            CPUAccessFlags: 0,
            MiscFlags: D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
        };
        let mut texture = None;
        unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture)) }
            .map_err(|_| CodecError::CodecNotFound("D3D11 staging CreateTexture2D failed (NV12/P010 shared)"))?;
        Ok(texture.unwrap())
    }

    fn get_shared_handle(texture: &ID3D11Texture2D) -> Result<HANDLE> {
        // NT handle (not the legacy `GetSharedHandle` KMT handle): Vulkan's
        // `D3D11_TEXTURE` external-memory import opens an NT handle, and
        // the texture is created with `SHARED_NTHANDLE` above. Read-only
        // because the renderer only samples it.
        let dxgi_resource: IDXGIResource1 = texture
            .cast()
            .map_err(|_| CodecError::CodecNotFound("IDXGIResource1 cast failed"))?;
        unsafe {
            dxgi_resource.CreateSharedHandle(
                None,
                DXGI_SHARED_RESOURCE_READ.0,
                windows::core::PCWSTR::null(),
            )
        }
        .map_err(|_| CodecError::CodecNotFound("CreateSharedHandle failed"))
    }

    /// Fallback: download to CPU when GPU export isn't possible. 8-bit
    /// NV12 only — the `DecodedFrame` plane buffers are `Vec<u8>`, so a
    /// 10-bit P010 surface can't be represented faithfully. The live
    /// client always exports GPU-resident (`gpu_export = true`), so this
    /// path is reached only on the null-decode-texture anomaly; a 10-bit
    /// stream there fails loudly rather than emitting downscaled garbage.
    /// 10-bit goes through `export_gpu_frame` (P010 staging) instead.
    #[allow(clippy::cast_sign_loss)]
    fn download_frame_cpu(&self, hw_frame: &AVFrame) -> Result<Frame> {
        let mut sw_frame = AVFrame::new();
        // Leave the format unset so `av_hwframe_transfer_data` picks the
        // surface's native sw format (NV12 for 8-bit, P010 for 10-bit) —
        // forcing NV12 on a P010 surface is rejected by the transfer.
        let rc = unsafe {
            ffi::av_hwframe_transfer_data(sw_frame.as_mut_ptr(), hw_frame.as_ptr(), 0)
        };
        if rc < 0 {
            return Err(CodecError::Ffmpeg(RsmpegError::AVError(rc)));
        }
        if sw_frame.format != ffi::AV_PIX_FMT_NV12 {
            // P010 (or anything else): the 8-bit CPU path can't carry it.
            return Err(CodecError::UnsupportedInputFormat);
        }

        let width = sw_frame.width as u32;
        let height = sw_frame.height as u32;
        let y_stride = unsafe { (*sw_frame.as_ptr()).linesize[0] as usize };
        let uv_stride = unsafe { (*sw_frame.as_ptr()).linesize[1] as usize };
        let y_ptr = sw_frame.data[0];
        let uv_ptr = sw_frame.data[1];

        let w = width as usize;
        let h = height as usize;
        let chroma_h = (h + 1) / 2;
        let chroma_w = (w + 1) / 2;

        let mut y = Vec::with_capacity(w * h);
        for row in 0..h {
            let src = unsafe { y_ptr.add(row * y_stride) };
            y.extend_from_slice(unsafe { std::slice::from_raw_parts(src, w) });
        }

        let uv_row_bytes = chroma_w * 2;
        let mut uv = Vec::with_capacity(uv_row_bytes * chroma_h);
        for row in 0..chroma_h {
            let src = unsafe { uv_ptr.add(row * uv_stride) };
            uv.extend_from_slice(unsafe { std::slice::from_raw_parts(src, uv_row_bytes) });
        }

        let pts = if sw_frame.pts == ffi::AV_NOPTS_VALUE {
            None
        } else {
            Some(sw_frame.pts)
        };

        Ok(Frame::Cpu(DecodedFrame {
            width,
            height,
            y,
            uv,
            pts,
        }))
    }
}

impl Decoder for D3D11Decoder {
    fn submit(&mut self, encoded: &[u8]) -> Result<()> {
        let packet = packet_from_bytes(encoded)?;
        match self.decoder.send_packet(Some(&packet)) {
            Ok(()) => Ok(()),
            Err(RsmpegError::DecoderFlushedError) => Ok(()),
            Err(e) => Err(CodecError::Ffmpeg(e)),
        }
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        match self.decoder.receive_frame() {
            Ok(frame) => {
                // GPU-resident by default: export the decoded D3D11
                // surface as a shared-handle `Frame::Gpu` for zero-copy
                // import into wgpu's Vulkan backend
                // (VK_KHR_external_memory_win32). `gpu_export` is false
                // only when the renderer's driver lacks that extension
                // (some AMD Vulkan stacks); then we download to CPU.
                // `export_gpu_frame` itself also falls back to CPU if the
                // decode surface pointer is unexpectedly null.
                let decoded = if self.gpu_export {
                    self.export_gpu_frame(&frame)?
                } else {
                    self.download_frame_cpu(&frame)?
                };
                Ok(Some(decoded))
            }
            Err(RsmpegError::DecoderDrainError) => Ok(None),
            Err(e) => Err(CodecError::Ffmpeg(e)),
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.decoder.flush_buffers();
        Ok(())
    }

    fn codec_kind(&self) -> CodecKind {
        self.kind
    }

    fn is_hardware(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        match self.kind {
            CodecKind::H264 => "h264 (d3d11va)",
            CodecKind::Hevc => "hevc (d3d11va)",
            CodecKind::Av1 => "av1 (d3d11va)",
        }
    }
}

/// Submit the prior `CopySubresourceRegion` copies to the GPU so the
/// renderer's (separate) D3D11 device can open the shared texture and
/// sample the result. `Flush` guarantees submission, not completion;
/// there is no keyed mutex, so cross-device visibility relies on the
/// decode→render handoff latency on a shared iGPU (validated by
/// `d3d11_cross_device_shared_handle_coherency`). On discrete GPUs with
/// separate hardware queues this assumption may not hold — the durable
/// fix is a single shared device for decode + present (Moonlight's model).
unsafe fn gpu_sync(_device: &ID3D11Device, context: &ID3D11DeviceContext) {
    unsafe { context.Flush() };
}

fn d3d11_av_codec_id(kind: CodecKind) -> Result<ffi::AVCodecID> {
    Ok(match kind {
        CodecKind::H264 => ffi::AV_CODEC_ID_H264,
        CodecKind::Hevc => ffi::AV_CODEC_ID_HEVC,
        CodecKind::Av1 => {
            return Err(CodecError::CodecNotFound("av1 d3d11va decoder"));
        }
    })
}

fn d3d11_decoder_name(kind: CodecKind) -> &'static str {
    match kind {
        CodecKind::H264 => "h264 (d3d11va)",
        CodecKind::Hevc => "hevc (d3d11va)",
        CodecKind::Av1 => "av1 (d3d11va)",
    }
}

/// FFmpeg get_format callback: select D3D11VA when offered.
unsafe extern "C" fn get_d3d11va_format(
    _ctx: *mut ffi::AVCodecContext,
    pix_fmts: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    let mut p = pix_fmts;
    unsafe {
        while *p != ffi::AV_PIX_FMT_NONE {
            if *p == ffi::AV_PIX_FMT_D3D11 {
                return ffi::AV_PIX_FMT_D3D11;
            }
            p = p.add(1);
        }
    }
    ffi::AV_PIX_FMT_NONE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::d3d11::encoder::d3d11_sw_format;

    /// The decoder's expected GPU-export `DXGI_FORMAT` and the encoder's
    /// `sw_format` (the FFmpeg pixel format fed to the hw_frames pool) are
    /// two parallel `profile → format` tables for the same physical chroma
    /// + bit depth. They must agree, or the renderer would import a P010
    /// surface while the host encoded NV12 (or vice versa). This is the
    /// encoder↔decoder half of the cross-table consistency check; the
    /// decoder↔renderer half lives in tether-client. No GPU.
    #[test]
    fn expected_decode_format_agrees_with_encoder_sw_format() {
        let profile = |bit_depth| VideoProfile {
            codec: CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth,
        };

        // 8-bit 4:2:0: NV12 on both sides.
        assert_eq!(expected_decode_dxgi_format(profile(8)), Some(DXGI_FORMAT_NV12.0 as u32));
        assert_eq!(d3d11_sw_format(profile(8)), ffi::AV_PIX_FMT_NV12);

        // 10-bit 4:2:0: P010 on both sides.
        assert_eq!(expected_decode_dxgi_format(profile(10)), Some(DXGI_FORMAT_P010.0 as u32));
        assert_eq!(d3d11_sw_format(profile(10)), ffi::AV_PIX_FMT_P010LE);
    }

    /// Profiles the Windows decoder can't open must return `None`, so the
    /// cross-crate test never asserts renderer support for a profile Windows
    /// can't carry. Covers all three axes of the `_` arm:
    ///   * **AV1 4:2:0** — listed in `PROFILE_PREFERENCE` but rejected at
    ///     `d3d11_av_codec_id`; the most important case, since it's a
    ///     negotiable profile, not a hypothetical one.
    ///   * **4:4:4** — rejected at encoder construction.
    ///   * **Out-of-range bit depth** (12-bit 4:2:0) — D3D11VA emits neither
    ///     NV12 nor P010 for it.
    /// The encoder's `d3d11_sw_format` falls back to NV12 for some of these,
    /// but that fallback is unreachable: 4:4:4 is rejected at construction
    /// and the negotiator never selects AV1 or an unmodeled bit depth on
    /// Windows.
    #[test]
    fn expected_decode_format_rejects_unmodeled_profiles() {
        // AV1 4:2:0 at both bit depths — in PROFILE_PREFERENCE, no D3D11VA
        // decoder, so the Windows client never advertises it.
        for bit_depth in [8, 10] {
            let av1 = VideoProfile {
                codec: CodecKind::Av1,
                chroma: ChromaSubsampling::Yuv420,
                bit_depth,
            };
            assert_eq!(
                expected_decode_dxgi_format(av1),
                None,
                "Windows has no AV1 {bit_depth}-bit decode path (d3d11_av_codec_id rejects AV1)"
            );
        }
        // Non-4:2:0 chroma at every modeled bit depth.
        for bit_depth in [8, 10] {
            let profile = VideoProfile {
                codec: CodecKind::Hevc,
                chroma: ChromaSubsampling::Yuv444,
                bit_depth,
            };
            assert_eq!(
                expected_decode_dxgi_format(profile),
                None,
                "Windows has no 4:4:4 {bit_depth}-bit decode path"
            );
        }
        // Out-of-range bit depth on the otherwise-supported 4:2:0 path.
        let odd_depth = VideoProfile {
            codec: CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 12,
        };
        assert_eq!(
            expected_decode_dxgi_format(odd_depth),
            None,
            "12-bit 4:2:0 has no Windows decode path"
        );
    }
}
