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
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_NV12, DXGI_SAMPLE_DESC};

use tether_protocol::control::CodecKind;

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

pub struct D3D11Decoder {
    kind: CodecKind,
    decoder: AVCodecContext,
    _hw_device: AVHWDeviceContext,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    staging: Option<Nv12Staging>,
    /// Whether the renderer can import D3D11 shared-handle textures into
    /// wgpu (Vulkan `VK_KHR_external_memory_win32`). When true we export
    /// decoded frames GPU-resident (`Frame::Gpu`); when false — the
    /// driver lacks the extension (some AMD Vulkan stacks) — we fall back
    /// to a CPU download. GPU-resident is the rule; CPU is the exception.
    gpu_export: bool,
}

/// Single NV12 staging texture for GPU-side plane extraction. The decode
/// pool surface (an NV12 texture array) is copied slice→staging per plane
/// — `CopySubresourceRegion` only works between same-format subresources,
/// so the staging must itself be NV12 (a separate R8 / R8G8 destination
/// would make the copy a silent no-op). The consumer opens the shared
/// handle and views plane 0 as R8 and plane 1 as R8G8.
struct Nv12Staging {
    tex: ID3D11Texture2D,
    /// NT handle from `CreateSharedHandle` — owned by us, closed on drop.
    /// (`tex` is allocated at even-rounded dims; `width`/`height` are the
    /// codec-reported dims used for the resolution-change staleness check
    /// and propagated to the renderer.)
    handle: HANDLE,
    width: u32,
    height: u32,
}

impl Drop for Nv12Staging {
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

    /// Export a D3D11VA surface as `Frame::Gpu` with per-plane shared
    /// handles. Copies each NV12 plane (Y and UV) from the decode pool
    /// into separate MISC_SHARED staging textures, entirely on the GPU.
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

        // Rebuild staging if dimensions changed.
        if self.staging.as_ref().map_or(true, |s| s.width != width || s.height != height) {
            self.staging = Some(Self::create_nv12_staging(&self.device, width, height)?);
        }
        let staging = self.staging.as_ref().unwrap();

        // NV12 texture-array subresource layout is PLANE-MAJOR:
        // D3D11CalcSubresource(plane, slice) =
        // planeSlice * (MipLevels * ArraySize) + arraySlice * MipLevels.
        // With MipLevels == 1 that's Y of slice N = N, UV of slice N =
        // ArraySize + N. We copy each plane of the decode slice into the
        // matching plane subresource of the single-slice NV12 staging
        // (subresource 0 = Y, 1 = UV). Both endpoints are NV12 plane
        // subresources, so the formats match and the copy is honoured —
        // copying an NV12 plane into a separate R8/R8G8 texture is a
        // silent no-op (the all-zero "green screen" bug).
        let array_size = {
            let mut desc = D3D11_TEXTURE2D_DESC::default();
            unsafe { src_texture.GetDesc(&mut desc) };
            desc.ArraySize
        };
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
            }),
            hw_frame.clone(),
        )))
    }

    fn create_nv12_staging(
        device: &ID3D11Device,
        width: u32,
        height: u32,
    ) -> Result<Nv12Staging> {
        // NV12 requires even dimensions; round up so the chroma plane is
        // whole. The decoder declares the real width/height downstream, so
        // a 1px pad on odd inputs is harmless.
        let w = (width + 1) & !1;
        let h = (height + 1) & !1;
        let tex = Self::create_shared_texture(device, w, h, DXGI_FORMAT_NV12)?;
        let handle = Self::get_shared_handle(&tex)?;
        Ok(Nv12Staging { tex, handle, width, height })
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
            .map_err(|_| CodecError::CodecNotFound("staging texture creation failed"))?;
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

    /// Fallback: download to CPU when GPU export isn't possible.
    #[allow(clippy::cast_sign_loss)]
    fn download_frame_cpu(&self, hw_frame: &AVFrame) -> Result<Frame> {
        let mut sw_frame = AVFrame::new();
        sw_frame.set_format(ffi::AV_PIX_FMT_NV12);

        let rc = unsafe {
            ffi::av_hwframe_transfer_data(sw_frame.as_mut_ptr(), hw_frame.as_ptr(), 0)
        };
        if rc < 0 {
            return Err(CodecError::Ffmpeg(RsmpegError::AVError(rc)));
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
