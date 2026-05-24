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
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_DEFAULT,
};
use windows::Win32::Graphics::Dxgi::IDXGIResource;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_NV12, DXGI_SAMPLE_DESC};

use tether_protocol::control::CodecKind;

use crate::h264::packet_from_bytes;
use crate::{
    init_ffmpeg, CodecError, D3D11DecodedTexture, DecodedFrame, Decoder, Frame, GpuFrame,
    GpuFrameSource, Result,
};

const DECODE_EXTRA_HW_FRAMES: i32 = 4;
const D3D11_RESOURCE_MISC_SHARED: u32 = 0x2;

pub struct D3D11Decoder {
    kind: CodecKind,
    decoder: AVCodecContext,
    _hw_device: AVHWDeviceContext,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    staging: Option<StagingTexture>,
}

struct StagingTexture {
    texture: ID3D11Texture2D,
    width: u32,
    height: u32,
}

unsafe impl Send for D3D11Decoder {}

impl D3D11Decoder {
    pub fn new(kind: CodecKind) -> Result<Self> {
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
            "D3D11VA decoder opened"
        );

        Ok(Self {
            kind,
            decoder,
            _hw_device: hw_device,
            device,
            context,
            staging: None,
        })
    }

    /// Export a D3D11VA surface as a `Frame::Gpu` via shared DXGI handle.
    /// Copies the decode pool surface to a staging texture with shared-
    /// handle flags, then exports via `CreateSharedHandle`.
    fn export_gpu_frame(&mut self, hw_frame: &AVFrame) -> Result<Frame> {
        let width = hw_frame.width as u32;
        let height = hw_frame.height as u32;
        let pts = if hw_frame.pts == ffi::AV_NOPTS_VALUE {
            None
        } else {
            Some(hw_frame.pts)
        };

        // D3D11VA frames: data[0] = ID3D11Texture2D*, data[1] = array index (as intptr_t)
        let src_texture_ptr = hw_frame.data[0] as *mut std::ffi::c_void;
        let array_index = hw_frame.data[1] as usize as u32;

        if src_texture_ptr.is_null() {
            return self.download_frame_cpu(hw_frame);
        }

        let src_texture: ID3D11Texture2D = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&src_texture_ptr)
                .ok_or(CodecError::CodecNotFound("null decode texture"))?
                .clone()
        };

        // Ensure staging texture exists at the right dimensions.
        if self.staging.as_ref().map_or(true, |s| s.width != width || s.height != height) {
            self.staging = Some(self.create_staging(width, height)?);
        }
        let staging = self.staging.as_ref().unwrap();

        // Copy from the decode pool array slice to our staging texture.
        // Flush ensures the copy is visible to the Vulkan importer
        // (legacy MISC_SHARED textures have no keyed-mutex sync).
        unsafe {
            self.context.CopySubresourceRegion(
                &staging.texture,
                0,
                0,
                0,
                0,
                &src_texture,
                array_index,
                None,
            );
            self.context.Flush();
        }

        // Export shared handle via IDXGIResource::GetSharedHandle.
        let dxgi_resource: IDXGIResource = staging
            .texture
            .cast()
            .map_err(|_| CodecError::CodecNotFound("IDXGIResource cast failed"))?;

        let handle: HANDLE = unsafe { dxgi_resource.GetSharedHandle() }
            .map_err(|_| CodecError::CodecNotFound("GetSharedHandle failed"))?;

        Ok(Frame::Gpu(GpuFrame::new(
            width,
            height,
            pts,
            GpuFrameSource::D3D11Texture(D3D11DecodedTexture {
                shared_handle: handle.0 as *mut std::ffi::c_void,
                width,
                height,
            }),
            hw_frame.clone(),
        )))
    }

    fn create_staging(&self, width: u32, height: u32) -> Result<StagingTexture> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_NV12,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: 0,
            CPUAccessFlags: 0,
            MiscFlags: D3D11_RESOURCE_MISC_SHARED,
        };
        let mut texture = None;
        unsafe { self.device.CreateTexture2D(&desc, None, Some(&mut texture)) }
            .map_err(|e| CodecError::CodecNotFound("staging texture creation failed"))?;
        Ok(StagingTexture {
            texture: texture.unwrap(),
            width,
            height,
        })
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
                let decoded = self.export_gpu_frame(&frame)?;
                Ok(Some(decoded))
            }
            Err(RsmpegError::DecoderDrainError) => Ok(None),
            Err(e) => Err(CodecError::Ffmpeg(e)),
        }
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
