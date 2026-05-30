//! D3D11 Video Processor BGRA→NV12 blit.
//!
//! The Video Processor is a fixed-function hardware unit on all modern
//! GPUs that performs color-space conversion, scaling, and deinterlacing.
//! We use it for a single purpose: converting the BGRA capture texture
//! into the NV12 surface expected by the encoder's hw_frames pool.
//!
//! The VP is initialized lazily on first use and cached for the encoder's
//! lifetime. Input/output views are created per-frame because the
//! hw_frames pool rotates across array slices.

use std::mem::ManuallyDrop;

use windows::core::Interface;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D, ID3D11VideoContext,
    ID3D11VideoDevice, ID3D11VideoProcessor, ID3D11VideoProcessorEnumerator,
    ID3D11VideoProcessorInputView, ID3D11VideoProcessorOutputView,
    D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE, D3D11_VIDEO_PROCESSOR_CONTENT_DESC,
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC, D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0,
    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC, D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0,
    D3D11_VIDEO_PROCESSOR_STREAM, D3D11_VIDEO_USAGE_PLAYBACK_NORMAL,
    D3D11_VPIV_DIMENSION_TEXTURE2D, D3D11_VPOV_DIMENSION_TEXTURE2DARRAY,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_RATIONAL;

/// Cached Video Processor state. Created once per encoder lifetime,
/// reused across frames. Supports input→output scaling when
/// `input_width != output_width` or `input_height != output_height`.
pub(crate) struct VideoProcessorState {
    video_device: ID3D11VideoDevice,
    video_context: ID3D11VideoContext,
    /// Base immediate context, kept to `Flush()` queued blits before the
    /// encoder reads the surface (QSV/MFX reads out-of-band).
    context: ID3D11DeviceContext,
    processor: ID3D11VideoProcessor,
    enumerator: ID3D11VideoProcessorEnumerator,
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
}

impl VideoProcessorState {
    pub(crate) fn new(
        device_ptr: *mut std::ffi::c_void,
        context_ptr: *mut std::ffi::c_void,
        input_width: u32,
        input_height: u32,
        output_width: u32,
        output_height: u32,
    ) -> std::result::Result<Self, String> {
        if device_ptr.is_null() || context_ptr.is_null() {
            return Err("null device or context pointer".into());
        }

        let device: ID3D11Device =
            unsafe { ID3D11Device::from_raw_borrowed(&device_ptr) }
                .ok_or("failed to borrow ID3D11Device")?
                .clone();
        let context: ID3D11DeviceContext =
            unsafe { ID3D11DeviceContext::from_raw_borrowed(&context_ptr) }
                .ok_or("failed to borrow ID3D11DeviceContext")?
                .clone();

        let video_device: ID3D11VideoDevice = device
            .cast()
            .map_err(|e| format!("QueryInterface ID3D11VideoDevice: {e}"))?;
        let video_context: ID3D11VideoContext = context
            .cast()
            .map_err(|e| format!("QueryInterface ID3D11VideoContext: {e}"))?;

        let content_desc = D3D11_VIDEO_PROCESSOR_CONTENT_DESC {
            InputFrameFormat: D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
            InputFrameRate: DXGI_RATIONAL {
                Numerator: 60,
                Denominator: 1,
            },
            InputWidth: input_width,
            InputHeight: input_height,
            OutputFrameRate: DXGI_RATIONAL {
                Numerator: 60,
                Denominator: 1,
            },
            OutputWidth: output_width,
            OutputHeight: output_height,
            Usage: D3D11_VIDEO_USAGE_PLAYBACK_NORMAL,
        };

        let enumerator: ID3D11VideoProcessorEnumerator = unsafe {
            video_device.CreateVideoProcessorEnumerator(&content_desc)
        }
        .map_err(|e| format!("CreateVideoProcessorEnumerator: {e}"))?;

        let processor: ID3D11VideoProcessor = unsafe {
            video_device.CreateVideoProcessor(&enumerator, 0)
        }
        .map_err(|e| format!("CreateVideoProcessor: {e}"))?;

        unsafe {
            video_context.VideoProcessorSetStreamAutoProcessingMode(
                &processor,
                0,
                false,
            );
            video_context.VideoProcessorSetStreamFrameFormat(
                &processor,
                0,
                D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
            );
        }

        Ok(Self {
            video_device,
            video_context,
            context,
            processor,
            enumerator,
            input_width,
            input_height,
            output_width,
            output_height,
        })
    }

    pub(crate) fn input_width(&self) -> u32 {
        self.input_width
    }

    pub(crate) fn input_height(&self) -> u32 {
        self.input_height
    }

    pub(crate) fn blit(
        &self,
        src_texture_ptr: *mut std::ffi::c_void,
        dst_texture_ptr: *mut std::ffi::c_void,
        dst_array_index: u32,
    ) -> std::result::Result<(), String> {
        if src_texture_ptr.is_null() || dst_texture_ptr.is_null() {
            return Err("null texture pointer".into());
        }

        let src_texture: ID3D11Texture2D =
            unsafe { ID3D11Texture2D::from_raw_borrowed(&src_texture_ptr) }
                .ok_or("failed to borrow src ID3D11Texture2D")?
                .clone();
        let dst_texture: ID3D11Texture2D =
            unsafe { ID3D11Texture2D::from_raw_borrowed(&dst_texture_ptr) }
                .ok_or("failed to borrow dst ID3D11Texture2D")?
                .clone();

        // Input view: BGRA source texture, single 2D texture (not array).
        let input_view_desc = D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC {
            FourCC: 0,
            ViewDimension: D3D11_VPIV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0 {
                Texture2D: Default::default(),
            },
        };
        let mut input_view: Option<ID3D11VideoProcessorInputView> = None;
        unsafe {
            self.video_device.CreateVideoProcessorInputView(
                &src_texture,
                &self.enumerator,
                &input_view_desc,
                Some(&mut input_view),
            )
        }
        .map_err(|e| format!("CreateVideoProcessorInputView: {e}"))?;
        let input_view = input_view.ok_or("CreateVideoProcessorInputView returned None")?;

        // Output view: NV12 destination texture array, targeting one slice.
        let output_view_desc = D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC {
            ViewDimension: D3D11_VPOV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0 {
                Texture2DArray: Default::default(),
            },
        };
        // Patch the array slice fields — the union default zeros
        // everything; we need FirstArraySlice = dst_array_index, ArraySize = 1.
        let mut output_view_desc = output_view_desc;
        output_view_desc.Anonymous.Texture2DArray.MipSlice = 0;
        output_view_desc.Anonymous.Texture2DArray.FirstArraySlice = dst_array_index;
        output_view_desc.Anonymous.Texture2DArray.ArraySize = 1;

        let mut output_view: Option<ID3D11VideoProcessorOutputView> = None;
        unsafe {
            self.video_device.CreateVideoProcessorOutputView(
                &dst_texture,
                &self.enumerator,
                &output_view_desc,
                Some(&mut output_view),
            )
        }
        .map_err(|e| format!("CreateVideoProcessorOutputView: {e}"))?;
        let output_view = output_view.ok_or("CreateVideoProcessorOutputView returned None")?;

        // Build the stream descriptor and blit.
        let stream = D3D11_VIDEO_PROCESSOR_STREAM {
            Enable: true.into(),
            OutputIndex: 0,
            InputFrameOrField: 0,
            PastFrames: 0,
            FutureFrames: 0,
            ppPastSurfaces: std::ptr::null_mut(),
            pInputSurface: ManuallyDrop::new(Some(input_view)),
            ppFutureSurfaces: std::ptr::null_mut(),
            ppPastSurfacesRight: std::ptr::null_mut(),
            pInputSurfaceRight: ManuallyDrop::new(None),
            ppFutureSurfacesRight: std::ptr::null_mut(),
        };

        let source_rect = RECT {
            left: 0,
            top: 0,
            right: self.input_width as i32,
            bottom: self.input_height as i32,
        };
        let target_rect = RECT {
            left: 0,
            top: 0,
            right: self.output_width as i32,
            bottom: self.output_height as i32,
        };
        unsafe {
            self.video_context.VideoProcessorSetStreamSourceRect(
                &self.processor,
                0,
                true,
                Some(&source_rect),
            );
            self.video_context.VideoProcessorSetOutputTargetRect(
                &self.processor,
                true,
                Some(&target_rect),
            );
            self.video_context
                .VideoProcessorBlt(
                    &self.processor,
                    &output_view,
                    0,
                    &[stream],
                )
                .map_err(|e| format!("VideoProcessorBlt: {e}"))?;
            // Flush so the blit is submitted to the GPU before the
            // encoder (esp. QSV/MFX, which reads the surface out-of-band)
            // samples it. Without this MFX can read a stale/incomplete
            // surface → AVERROR_INVALIDDATA on send_frame.
            self.context.Flush();
        }

        Ok(())
    }
}

unsafe impl Send for VideoProcessorState {}
