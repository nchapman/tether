//! Native D3D11 client renderer (Windows).
//!
//! Decode and present both live in D3D11 — no cross-API bridge. The
//! D3D11VA decoder hands the decoded NV12 surface across as two NT
//! shared handles (`IDXGIResource1::CreateSharedHandle`; Y as R8, UV as
//! R8G8); this renderer opens them with `OpenSharedResource1` onto its
//! own device, builds SRVs, and samples them through a YUV→RGB HLSL
//! pixel shader into a DXGI swapchain on the winit window. This mirrors
//! Moonlight's `d3d11va.cpp` cross-device path rather than importing into
//! wgpu/Vulkan, which keeps the whole hot path inside one graphics API on
//! the platform's first-class one.
//!
//! This backend is selected in place of the wgpu `GpuState` on Windows
//! via the `Backend` type alias in `lib.rs`; it exposes the same method
//! surface (`new`, `resize`, `apply_frame`, `render`, `dimensions`) so
//! the shared `App` event loop drives it unchanged.

mod cursor;

use std::ffi::c_void;
use std::sync::Arc;

use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use windows::core::{s, Interface, PCSTR};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::D3D_SRV_DIMENSION_TEXTURE2D;
use windows::Win32::Graphics::Direct3D::{
    ID3DBlob, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_DRIVER_TYPE_HARDWARE,
    D3D_FEATURE_LEVEL_11_0,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11CreateDevice, ID3D11BlendState, ID3D11Buffer, ID3D11Device, ID3D11Device1,
    ID3D11DeviceContext, ID3D11PixelShader, ID3D11RasterizerState, ID3D11RenderTargetView,
    ID3D11SamplerState, ID3D11ShaderResourceView, ID3D11Texture2D, ID3D11VertexShader,
    D3D11_BIND_CONSTANT_BUFFER, D3D11_BIND_SHADER_RESOURCE, D3D11_BUFFER_DESC,
    D3D11_CPU_ACCESS_WRITE, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_FORMAT_SUPPORT_TEXTURE2D,
    D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_WRITE_DISCARD, D3D11_RASTERIZER_DESC,
    D3D11_RENDER_TARGET_VIEW_DESC, D3D11_RENDER_TARGET_VIEW_DESC_0, D3D11_RTV_DIMENSION_TEXTURE2D,
    D3D11_SAMPLER_DESC, D3D11_SDK_VERSION, D3D11_TEX2D_RTV, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_DEFAULT, D3D11_USAGE_DYNAMIC, D3D11_VIEWPORT,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CULL_NONE, D3D11_FILL_SOLID, D3D11_SHADER_RESOURCE_VIEW_DESC,
    D3D11_SHADER_RESOURCE_VIEW_DESC_0, D3D11_TEX2D_SRV,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_CLAMP,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_ALPHA_MODE_IGNORE, DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM,
    DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_NV12, DXGI_FORMAT_P010, DXGI_FORMAT_R16G16_UNORM,
    DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8_UNORM, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    IDXGIAdapter, IDXGIDevice, IDXGIFactory2, IDXGISwapChain1, DXGI_PRESENT, DXGI_SCALING_STRETCH,
    DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG, DXGI_SWAP_EFFECT_FLIP_DISCARD,
    DXGI_USAGE_RENDER_TARGET_OUTPUT,
};
use winit::window::Window;

use tether_protocol::control::{ChromaSubsampling, ColorTransfer, VideoColorSpec};

use tether_codec::GpuFrameSource;

use crate::cursor_overlay::CursorChannel;
use crate::{letterbox_scale, Frame, RenderError, Result};

// Mirror of shader.wgsl's color_params tags (see `yuv.hlsl`).
const TRANSFER_KIND_BT709: u32 = 0;
const TRANSFER_KIND_SRGB: u32 = 1;
const RANGE_KIND_LIMITED_8: u32 = 0;
const RANGE_KIND_LIMITED_10: u32 = 1;

/// The `Params` cbuffer the HLSL shader reads at `b0`. `#[repr(C)]` with
/// the 16-byte-aligned layout HLSL cbuffers require: a `float4` then a
/// `uint4`.
#[repr(C)]
#[derive(Clone, Copy)]
struct ShaderParams {
    /// `.xy` = letterbox NDC scale; `.zw` padding.
    scale: [f32; 4],
    /// `.x` = transfer kind, `.y` = range kind; `.zw` reserved.
    color_params: [u32; 4],
}

/// Map a windows-rs `Error` to our string-based render error.
fn d3d_err(context: &str, e: windows::core::Error) -> RenderError {
    RenderError::GraphicsApi(format!("{context}: {e}"))
}

/// EOTF tag for the negotiated color space — local mirror of
/// `gpu::transfer_kind_for` (private + cfg'd out on Windows).
fn transfer_kind_for(spec: VideoColorSpec) -> u32 {
    match spec.transfer {
        ColorTransfer::Bt709 => TRANSFER_KIND_BT709,
        ColorTransfer::Srgb => TRANSFER_KIND_SRGB,
        // PQ / HLG / Linear aren't implemented in the shader yet; fall
        // back to BT.709 (slightly off, not unwatchable). HDR work
        // promotes these once the surface format / tone-map chain lands.
        ColorTransfer::Pq | ColorTransfer::Hlg | ColorTransfer::Linear => {
            tracing::warn!(side = "d3d11", ?spec.transfer, "EOTF not implemented; using BT.709");
            TRANSFER_KIND_BT709
        }
    }
}

/// Windows counterpart to the wgpu backend's `supports_10bit_render`,
/// queried by the client to gate 10-bit profiles in its decode advert.
///
/// 10-bit render means decoding P010 and sampling it through R16 / R16G16
/// plane SRVs (the shader's `RANGE_KIND_LIMITED_10` branch normalises the
/// MSB-aligned values). R16-norm formats are mandatory on any D3D11 11.0
/// device, so the only real variable is whether the GPU supports P010
/// textures at all — probe a throwaway hardware device for that. The
/// client's per-profile decode probe is the authoritative Main10 gate;
/// this just keeps the renderer from advertising a format it can't open
/// (and returns `false` if no D3D11 device can be created — negotiation
/// then settles on an 8-bit profile).
pub async fn supports_10bit_render() -> bool {
    unsafe {
        // Throwaway device: the renderer's own device isn't created until
        // `D3D11RenderState::new`, well after this startup-time probe.
        let mut device: Option<ID3D11Device> = None;
        if D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            Default::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            Some(&[D3D_FEATURE_LEVEL_11_0]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            None,
        )
        .is_err()
        {
            return false;
        }
        let Some(device) = device else { return false };
        // Check P010 TEXTURE2D support only — NOT SHADER_SAMPLE on P010
        // itself, which would be wrong: we never sample P010 directly, we
        // sample its R16 / R16G16 plane SRVs, and those norm formats are a
        // mandatory FL11.0 baseline. So "can a P010 texture exist" is the
        // only real variable.
        match device.CheckFormatSupport(DXGI_FORMAT_P010) {
            #[allow(clippy::cast_sign_loss)]
            Ok(flags) => (flags & D3D11_FORMAT_SUPPORT_TEXTURE2D.0 as u32) != 0,
            Err(_) => false,
        }
    }
}

/// Render pipeline objects built once at construction and reused every
/// frame: the YUV→RGB shaders, the bilinear sampler, and the per-frame
/// constant buffer.
struct YuvPipeline {
    vs: ID3D11VertexShader,
    ps: ID3D11PixelShader,
    /// The constant buffer, for per-frame `UpdateSubresource`.
    cbuffer: ID3D11Buffer,
    /// Pre-built single-element binding arrays so the hot `render` path
    /// hands the `Set*` calls a borrow instead of cloning (AddRef-ing) the
    /// COM objects every frame.
    cbuffers: [Option<ID3D11Buffer>; 1],
    samplers: [Option<ID3D11SamplerState>; 1],
    /// No-cull rasterizer state. D3D11's default culls back faces, and the
    /// fullscreen quad is wound counterclockwise (back-facing under the
    /// default front=CW), so without this every triangle is culled and
    /// nothing is drawn. wgpu's default is no-cull, which is why the
    /// identical WGSL needs no equivalent on the Linux path.
    rasterizer: ID3D11RasterizerState,
}

/// Identity of the currently-imported source, so `apply_frame` only
/// rebuilds SRVs when the underlying surface actually changes.
#[derive(PartialEq, Eq, Clone, Copy)]
enum SourceKey {
    /// GPU path: the decoder's NV12 shared-handle pointer plus dims.
    /// Stable across frames at a fixed resolution (the decoder reuses its
    /// staging texture), so we open it once and resample each frame. Dims
    /// are part of the key because the kernel can recycle a closed NT
    /// handle's pointer value for a new texture; pairing the pointer with
    /// dims makes a false cache hit require both an identical recycled
    /// handle value AND identical dims (the decoder closes the old handle
    /// on `Drop`, so the recycling window is a single resolution flip).
    Gpu {
        handle: *mut c_void,
        width: u32,
        height: u32,
        format: u32,
    },
    /// CPU fallback: dims of the dynamic upload textures. Pixels are
    /// re-mapped every frame; the textures are recreated only on resize.
    Cpu { width: u32, height: u32 },
}

/// The Y + UV shader resource views currently bound for sampling, plus
/// the textures backing them (kept alive for as long as the SRVs are).
/// SRVs are held as a pre-built `[Y, UV]` array so the `render` path can
/// bind them by borrow without per-frame clones.
struct Imported {
    key: SourceKey,
    /// Backing textures. GPU path: both are the SAME opened NV12 texture
    /// (the two SRVs are plane views of it) — kept as redundant keepalive
    /// alongside the SRVs. CPU path: two distinct dynamic textures (R8 +
    /// R8G8) that `upload_cpu_frame` re-maps each frame.
    y_tex: ID3D11Texture2D,
    uv_tex: ID3D11Texture2D,
    srvs: [Option<ID3D11ShaderResourceView>; 2],
}

/// Where `render` presents. Production always uses `Window` (a DXGI
/// swapchain on the winit HWND). `Offscreen` is a test-only target that
/// renders into an owned texture for CPU readback, so the real `render`
/// path can be exercised headlessly.
enum PresentTarget {
    Window {
        _window: Arc<Window>,
        swapchain: IDXGISwapChain1,
    },
    #[cfg(test)]
    Offscreen { color: ID3D11Texture2D },
}

pub(crate) struct D3D11RenderState {
    device: ID3D11Device,
    /// `ID3D11Device1` view of `device`, needed for `OpenSharedResource1`
    /// (the NT-handle open; the legacy `OpenSharedResource` is KMT-only).
    device1: ID3D11Device1,
    context: ID3D11DeviceContext,
    target: PresentTarget,
    /// `None` only between a failed `ResizeBuffers` and the next
    /// successful resize; `render` skips the draw while it's `None`.
    rtv: Option<ID3D11RenderTargetView>,
    pipeline: YuvPipeline,
    /// Backbuffer / window size in physical pixels.
    surface_size: (u32, u32),
    /// Most recent decoded frame, presented on the next `render`. Carries
    /// its own backend guard so the source surface stays alive until
    /// displaced.
    latest: Option<Frame>,
    /// SRVs derived from `latest`; rebuilt only when the source identity
    /// (`SourceKey`) changes.
    imported: Option<Imported>,
    /// Encoded video dimensions of `latest` (drives letterbox math in the
    /// shared `App`); falls back to the surface size until a frame lands.
    video_size: (u32, u32),
    /// EOTF + range tags baked into the per-frame cbuffer.
    transfer_kind: u32,
    range_kind: u32,
    /// Cursor overlay pass. Drawn over the video each frame (no-op when
    /// no sprite is active / cursor hidden / relative-locked).
    cursor: cursor::D3D11CursorOverlay,
    /// Shared cursor state, written by the client's wire-receive task
    /// and read by `cursor` each frame.
    cursor_channel: CursorChannel,
}

impl D3D11RenderState {
    /// Async to match the wgpu `GpuState::new` signature the shared
    /// `App` calls through `pollster::block_on`; no actual awaiting.
    pub(crate) async fn new(
        window: Arc<Window>,
        color_space: VideoColorSpec,
        chroma: ChromaSubsampling,
        bit_depth: u8,
        cursor_channel: CursorChannel,
    ) -> Result<Self> {
        // Windows is 4:2:0-only today; 4:4:4 has no D3D11 sample path and
        // is rejected at negotiation. Guard here so a mis-negotiation
        // surfaces as a clear error rather than wrong colors.
        if chroma != ChromaSubsampling::Yuv420 {
            return Err(RenderError::GraphicsApi(format!(
                "D3D11 renderer supports 4:2:0 only, got {chroma:?}"
            )));
        }
        let transfer_kind = transfer_kind_for(color_space);
        let range_kind = if bit_depth == 10 {
            RANGE_KIND_LIMITED_10
        } else {
            RANGE_KIND_LIMITED_8
        };

        let size = window.inner_size();
        let surface_size = (size.width.max(1), size.height.max(1));

        let hwnd = hwnd_from_window(&window)?;

        // Hardware device with BGRA support (required for swapchain RTVs
        // and D3D11 interop). Single immediate context.
        let mut device: Option<ID3D11Device> = None;
        let mut context: Option<ID3D11DeviceContext> = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                Default::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                Some(&[D3D_FEATURE_LEVEL_11_0]),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        }
        .map_err(|e| d3d_err("D3D11CreateDevice", e))?;
        let device = device.ok_or_else(|| RenderError::GraphicsApi("null D3D11 device".into()))?;
        let context =
            context.ok_or_else(|| RenderError::GraphicsApi("null D3D11 context".into()))?;
        let device1: ID3D11Device1 = device
            .cast()
            .map_err(|e| d3d_err("ID3D11Device1 cast", e))?;

        // Walk device → DXGI factory to create a swapchain for the HWND.
        let dxgi_device: IDXGIDevice = device.cast().map_err(|e| d3d_err("IDXGIDevice cast", e))?;
        let adapter: IDXGIAdapter =
            unsafe { dxgi_device.GetAdapter() }.map_err(|e| d3d_err("GetAdapter", e))?;
        let factory: IDXGIFactory2 =
            unsafe { adapter.GetParent() }.map_err(|e| d3d_err("GetParent IDXGIFactory2", e))?;

        // FLIP_DISCARD + 2 buffers is the right baseline for a live-stream
        // viewer (the previous frame is never reused after present). The
        // backbuffer is plain UNORM; the RTV reinterprets it as *_SRGB so
        // the hardware applies the sRGB OETF on store — parity with wgpu's
        // sRGB surface write (flip swapchains can't take an _SRGB
        // swapchain format directly, hence the UNORM-buffer / SRGB-view
        // split). Scaling STRETCH is fine: the shader does its own aspect
        // letterbox; revisit if a future latency pass wants ALLOW_TEARING
        // (which requires DXGI_SCALING_NONE).
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: surface_size.0,
            Height: surface_size.1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 2,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_IGNORE,
            Flags: 0,
        };
        let swapchain = unsafe { factory.CreateSwapChainForHwnd(&device, hwnd, &desc, None, None) }
            .map_err(|e| d3d_err("CreateSwapChainForHwnd", e))?;

        let rtv = create_rtv(&device, &swapchain)?;
        let pipeline = build_pipeline(&device)?;
        let cursor = cursor::D3D11CursorOverlay::new(&device)?;

        tracing::info!(
            width = surface_size.0,
            height = surface_size.1,
            ?chroma,
            bit_depth,
            transfer_kind,
            "D3D11 renderer initialised"
        );

        Ok(Self {
            device,
            device1,
            context,
            target: PresentTarget::Window {
                _window: window,
                swapchain,
            },
            rtv: Some(rtv),
            pipeline,
            surface_size,
            latest: None,
            imported: None,
            video_size: surface_size,
            transfer_kind,
            range_kind,
            cursor,
            cursor_channel,
        })
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == self.surface_size {
            return;
        }
        // Only the windowed swapchain resizes; an offscreen target is
        // fixed-size for the test that owns it.
        let swapchain = match &self.target {
            PresentTarget::Window { swapchain, .. } => swapchain.clone(),
            #[cfg(test)]
            PresentTarget::Offscreen { .. } => return,
        };
        // The RTV holds the only reference to the backbuffer and must be
        // released before ResizeBuffers — drop it by clearing the slot.
        // On failure we leave it `None` and bail; `render` then skips the
        // draw until a later resize rebuilds it.
        self.rtv = None;
        match resize_and_rebuild_rtv(&self.device, &swapchain, width, height) {
            Ok(rtv) => {
                self.rtv = Some(rtv);
                self.surface_size = (width, height);
            }
            Err(e) => tracing::error!(error = %e, "D3D11 swapchain resize failed"),
        }
    }

    pub(crate) fn apply_frame(&mut self, frame: Frame) -> Result<()> {
        self.video_size = (frame.width(), frame.height());
        // Rebuild the sampled SRVs from the new frame. The GPU path opens
        // the decoder's shared handles (cached by handle); the CPU
        // fallback re-uploads into dynamic textures.
        match &frame {
            Frame::Gpu(g) => {
                // `D3D11Texture` is the only `GpuFrameSource` variant
                // compiled on Windows (dma-buf / IOSurface are cfg-gated
                // to Linux/macOS), so this destructure is irrefutable.
                let GpuFrameSource::D3D11Texture(tex) = &g.source;
                self.import_shared_biplanar(tex.handle, tex.width, tex.height, tex.format)?;
            }
            Frame::Cpu(c) => self.upload_cpu_frame(c)?,
        }
        self.latest = Some(frame);
        Ok(())
    }

    /// Open the decoder's biplanar shared NT handle onto our device and
    /// build the two plane SRVs. The SRV format selects the plane and
    /// depends on the decode format: NV12 → R8 luma + R8G8 chroma (8-bit),
    /// P010 → R16 luma + R16G16 chroma (10-bit MSB-aligned, normalised by
    /// the shader's `RANGE_KIND_LIMITED_10` branch). No-op when the
    /// (handle, dims, format) match what's already imported — the decoder
    /// reuses its staging texture across frames at a fixed resolution, so
    /// this opens once per resolution and resamples after.
    ///
    /// Synchronization: the staging texture carries no keyed mutex, so
    /// there is no cross-device GPU fence between the decoder's
    /// `CopySubresourceRegion` (+ `Flush`) and our sample. We rely on the
    /// decode→render channel handoff latency (~ms) dwarfing the copy
    /// (~µs) on a shared iGPU. If the hardware loopback shows tearing,
    /// the correct fix is a single shared D3D11 device for decode+present
    /// (Moonlight's model) — not a keyed mutex on this single staging
    /// texture, which would serialize producer/consumer and defeat the
    /// drop-oldest `LatestFrame` handoff.
    fn import_shared_biplanar(
        &mut self,
        handle: *mut c_void,
        width: u32,
        height: u32,
        format: u32,
    ) -> Result<()> {
        if handle.is_null() {
            return Err(RenderError::GraphicsApi("null D3D11 shared handle".into()));
        }
        let key = SourceKey::Gpu {
            handle,
            width,
            height,
            format,
        };
        if self.imported.as_ref().is_some_and(|i| i.key == key) {
            return Ok(());
        }

        // Per-plane SRV formats from the staging texture's DXGI format.
        let (y_fmt, uv_fmt) = plane_srv_formats(format)?;

        let tex: ID3D11Texture2D = unsafe { self.device1.OpenSharedResource1(HANDLE(handle)) }
            .map_err(|e| d3d_err("OpenSharedResource1(biplanar)", e))?;

        let y_srv = create_plane_srv(&self.device, &tex, y_fmt)?;
        let uv_srv = create_plane_srv(&self.device, &tex, uv_fmt)?;

        tracing::debug!(
            width,
            height,
            format,
            "opened decoder biplanar shared handle"
        );
        self.imported = Some(Imported {
            key,
            y_tex: tex.clone(),
            uv_tex: tex,
            srvs: [Some(y_srv), Some(uv_srv)],
        });
        Ok(())
    }

    /// CPU fallback: upload the NV12 planes into dynamic textures. Only
    /// reached if the decoder couldn't export a GPU surface (a null decode
    /// texture) — on Windows the GPU path is the rule. Recreates the
    /// textures only on a dimension change; re-maps the pixels each frame.
    fn upload_cpu_frame(&mut self, frame: &crate::CpuFrame) -> Result<()> {
        let (chroma_w, chroma_h) = frame.chroma_dims();
        let key = SourceKey::Cpu {
            width: frame.width,
            height: frame.height,
        };
        if !self.imported.as_ref().is_some_and(|i| i.key == key) {
            let y_tex = create_dynamic_texture(
                &self.device,
                frame.width,
                frame.height,
                DXGI_FORMAT_R8_UNORM,
            )?;
            let uv_tex =
                create_dynamic_texture(&self.device, chroma_w, chroma_h, DXGI_FORMAT_R8G8_UNORM)?;
            let y_srv = create_srv(&self.device, &y_tex)?;
            let uv_srv = create_srv(&self.device, &uv_tex)?;
            self.imported = Some(Imported {
                key,
                y_tex,
                uv_tex,
                srvs: [Some(y_srv), Some(uv_srv)],
            });
        }
        let imported = self.imported.as_ref().expect("just set");
        // 1 byte/texel for Y, 2 bytes/texel for interleaved UV.
        upload_rows(
            &self.context,
            &imported.y_tex,
            &frame.y,
            frame.width as usize,
            frame.height,
        )?;
        upload_rows(
            &self.context,
            &imported.uv_tex,
            &frame.uv,
            chroma_w as usize * 2,
            chroma_h,
        )?;
        Ok(())
    }

    pub(crate) fn render(&mut self) -> std::result::Result<(), String> {
        // Skip the frame if the swapchain has no live RTV (failed resize);
        // the next successful resize rebuilds it.
        let Some(rtv) = self.rtv.as_ref() else {
            return Ok(());
        };

        unsafe {
            self.context
                .ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0]);
            // Reset to opaque blending each frame: the cursor pass below
            // sets a straight-alpha blend state that must not leak into
            // the next frame's (opaque) video draw.
            self.context
                .OMSetBlendState(None::<&ID3D11BlendState>, None, 0xffff_ffff);

            // Draw the video only once a frame has been imported; until
            // then the cleared black backbuffer is presented.
            if let Some(imported) = self.imported.as_ref() {
                let (sx, sy) = letterbox_scale(self.video_size, self.surface_size);
                let params = ShaderParams {
                    scale: [sx, sy, 0.0, 0.0],
                    color_params: [self.transfer_kind, self.range_kind, 0, 0],
                };
                self.context.UpdateSubresource(
                    &self.pipeline.cbuffer,
                    0,
                    None,
                    std::ptr::from_ref(&params).cast(),
                    0,
                    0,
                );

                let viewport = D3D11_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: self.surface_size.0 as f32,
                    Height: self.surface_size.1 as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                };
                self.context.RSSetViewports(Some(&[viewport]));
                self.context.RSSetState(&self.pipeline.rasterizer);
                self.context
                    .OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);

                self.context
                    .IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                self.context.VSSetShader(&self.pipeline.vs, None);
                self.context.PSSetShader(&self.pipeline.ps, None);
                // Bind the pre-built arrays by borrow — no per-frame clones.
                self.context
                    .VSSetConstantBuffers(0, Some(&self.pipeline.cbuffers));
                self.context
                    .PSSetConstantBuffers(0, Some(&self.pipeline.cbuffers));
                self.context.PSSetShaderResources(0, Some(&imported.srvs));
                self.context.PSSetSamplers(0, Some(&self.pipeline.samplers));

                self.context.Draw(6, 0);

                // Cursor overlay over the video. `fit_dims` is the pixel
                // rect the video covers inside the window — the same
                // `(sx, sy)` letterbox scale the YUV draw used — so the
                // sprite lands in the exact video rect. No-op when no
                // sprite is active. The RTV + viewport set above stay
                // bound for this pass.
                #[allow(
                    clippy::cast_precision_loss,
                    clippy::cast_sign_loss,
                    clippy::cast_possible_truncation
                )]
                let fit_dims = (
                    (self.surface_size.0 as f32 * sx).round() as u32,
                    (self.surface_size.1 as f32 * sy).round() as u32,
                );
                self.cursor.render(
                    &self.device,
                    &self.context,
                    &self.cursor_channel,
                    self.video_size,
                    self.surface_size,
                    fit_dims,
                );
            }

            match &self.target {
                PresentTarget::Window { swapchain, .. } => {
                    swapchain
                        .Present(1, DXGI_PRESENT(0))
                        .ok()
                        .map_err(|e| format!("Present: {e}"))?;
                }
                // Offscreen has nothing to flip; the draw lands in `color`,
                // which the test reads back. Flush so the GPU work runs.
                #[cfg(test)]
                PresentTarget::Offscreen { .. } => self.context.Flush(),
            }
        }
        Ok(())
    }

    /// `((video_w, video_h), (surface_w, surface_h))` — same shape the
    /// wgpu backend returns, consumed by the shared `App` for letterbox
    /// and cursor-normalisation math.
    pub(crate) fn dimensions(&self) -> ((u32, u32), (u32, u32)) {
        (self.video_size, self.surface_size)
    }
}

#[cfg(test)]
impl D3D11RenderState {
    /// Headless variant for hardware tests: a self-owned device rendering
    /// into an offscreen BGRA texture instead of a window swapchain.
    /// Exercises the real `render`/import/shader path with CPU readback.
    pub(crate) fn new_headless(
        width: u32,
        height: u32,
        color_space: VideoColorSpec,
        bit_depth: u8,
    ) -> Result<Self> {
        let mut device: Option<ID3D11Device> = None;
        let mut context: Option<ID3D11DeviceContext> = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                Default::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                Some(&[D3D_FEATURE_LEVEL_11_0]),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        }
        .map_err(|e| d3d_err("D3D11CreateDevice (headless)", e))?;
        let device = device.ok_or_else(|| RenderError::GraphicsApi("null device".into()))?;
        let context = context.ok_or_else(|| RenderError::GraphicsApi("null context".into()))?;
        let device1: ID3D11Device1 = device.cast().map_err(|e| d3d_err("ID3D11Device1", e))?;

        // Offscreen render target: TYPELESS BGRA so the SRGB RTV is a
        // valid cast (a plain UNORM resource can't take an SRGB view —
        // only DXGI's swapchain buffers special-case that). The SRGB RTV
        // then reinterprets it exactly as the swapchain path does.
        let color_desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_TYPELESS,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: windows::Win32::Graphics::Direct3D11::D3D11_BIND_RENDER_TARGET.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut color = None;
        unsafe { device.CreateTexture2D(&color_desc, None, Some(&mut color)) }
            .map_err(|e| d3d_err("offscreen CreateTexture2D", e))?;
        let color = color.ok_or_else(|| RenderError::GraphicsApi("null offscreen tex".into()))?;
        let rtv = create_rtv_for_texture(&device, &color)?;
        let pipeline = build_pipeline(&device)?;
        let cursor = cursor::D3D11CursorOverlay::new(&device)?;

        Ok(Self {
            device,
            device1,
            context,
            target: PresentTarget::Offscreen { color },
            rtv: Some(rtv),
            pipeline,
            surface_size: (width, height),
            latest: None,
            imported: None,
            video_size: (width, height),
            transfer_kind: transfer_kind_for(color_space),
            range_kind: if bit_depth == 10 {
                RANGE_KIND_LIMITED_10
            } else {
                RANGE_KIND_LIMITED_8
            },
            cursor,
            // Detached channel: headless tests with no wire-side producer
            // get an overlay that exists but never draws. The cursor
            // hardware test below replaces this via `cursor_channel()`.
            cursor_channel: CursorChannel::new(),
        })
    }

    /// Test-only accessor to the cursor channel, so a headless test can
    /// feed sprites the same way the client's wire-receive task does.
    #[cfg(test)]
    pub(crate) fn cursor_channel(&self) -> CursorChannel {
        self.cursor_channel.clone()
    }

    /// Build R8 (Y) + R8G8 (UV) textures from raw plane bytes, create
    /// their SRVs, and install them as the imported frame — lets a test
    /// drive the render path with synthetic planes, no decoder/`Frame`
    /// plumbing. `y` is `width*height`; `uv` is `chroma_w*chroma_h*2`.
    pub(crate) fn upload_test_planes(&mut self, y: &[u8], uv: &[u8], width: u32, height: u32) {
        let chroma_w = width.div_ceil(2);
        let chroma_h = height.div_ceil(2);
        let y_tex =
            make_immutable_texture(&self.device, width, height, DXGI_FORMAT_R8_UNORM, y, width);
        let uv_tex = make_immutable_texture(
            &self.device,
            chroma_w,
            chroma_h,
            DXGI_FORMAT_R8G8_UNORM,
            uv,
            chroma_w * 2,
        );
        let y_srv = create_srv(&self.device, &y_tex).expect("y srv");
        let uv_srv = create_srv(&self.device, &uv_tex).expect("uv srv");
        self.video_size = (width, height);
        self.imported = Some(Imported {
            key: SourceKey::Cpu { width, height },
            y_tex,
            uv_tex,
            srvs: [Some(y_srv), Some(uv_srv)],
        });
    }

    /// Copy the offscreen color texture to a staging texture and read it
    /// back as tightly-packed BGRA (`width * height * 4` bytes).
    pub(crate) fn read_back_bgra(&self) -> Vec<u8> {
        use windows::Win32::Graphics::Direct3D11::{
            D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_USAGE_STAGING,
        };
        let PresentTarget::Offscreen { color } = &self.target else {
            panic!("read_back_bgra requires an offscreen target");
        };
        let (w, h) = self.surface_size;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: w,
            Height: h,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
        };
        let mut staging = None;
        unsafe { self.device.CreateTexture2D(&desc, None, Some(&mut staging)) }
            .expect("staging CreateTexture2D");
        let staging = staging.unwrap();
        let mut out = vec![0u8; (w * h * 4) as usize];
        unsafe {
            self.context.CopyResource(&staging, color);
            self.context.Flush();
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            self.context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .expect("Map staging");
            let row_bytes = (w * 4) as usize;
            for row in 0..h as usize {
                let src = (mapped.pData as *const u8).add(row * mapped.RowPitch as usize);
                std::ptr::copy_nonoverlapping(
                    src,
                    out.as_mut_ptr().add(row * row_bytes),
                    row_bytes,
                );
            }
            self.context.Unmap(&staging, 0);
        }
        out
    }
}

fn hwnd_from_window(window: &Window) -> Result<windows::Win32::Foundation::HWND> {
    let handle = window
        .window_handle()
        .map_err(|e| RenderError::GraphicsApi(format!("window handle: {e}")))?
        .as_raw();
    match handle {
        RawWindowHandle::Win32(h) => {
            Ok(windows::Win32::Foundation::HWND(h.hwnd.get() as *mut c_void))
        }
        other => Err(RenderError::GraphicsApi(format!(
            "expected a Win32 window handle, got {other:?}"
        ))),
    }
}

/// Build a `*_SRGB` RTV over the (UNORM) backbuffer so writes are sRGB-
/// encoded by the hardware — see the swapchain comment in `new`.
fn create_rtv(
    device: &ID3D11Device,
    swapchain: &IDXGISwapChain1,
) -> Result<ID3D11RenderTargetView> {
    let backbuffer: ID3D11Texture2D =
        unsafe { swapchain.GetBuffer(0) }.map_err(|e| d3d_err("GetBuffer(0)", e))?;
    create_rtv_for_texture(device, &backbuffer)
}

/// `*_SRGB` RTV over an arbitrary BGRA-UNORM texture (the swapchain
/// backbuffer in production, an offscreen color texture in tests). The
/// SRGB view reinterprets the UNORM resource so the hardware applies the
/// sRGB OETF on store.
fn create_rtv_for_texture(
    device: &ID3D11Device,
    texture: &ID3D11Texture2D,
) -> Result<ID3D11RenderTargetView> {
    let rtv_desc = D3D11_RENDER_TARGET_VIEW_DESC {
        Format: DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        ViewDimension: D3D11_RTV_DIMENSION_TEXTURE2D,
        Anonymous: D3D11_RENDER_TARGET_VIEW_DESC_0 {
            Texture2D: D3D11_TEX2D_RTV { MipSlice: 0 },
        },
    };
    let mut rtv = None;
    unsafe { device.CreateRenderTargetView(texture, Some(&rtv_desc), Some(&mut rtv)) }
        .map_err(|e| d3d_err("CreateRenderTargetView", e))?;
    rtv.ok_or_else(|| RenderError::GraphicsApi("null render target view".into()))
}

fn resize_and_rebuild_rtv(
    device: &ID3D11Device,
    swapchain: &IDXGISwapChain1,
    width: u32,
    height: u32,
) -> Result<ID3D11RenderTargetView> {
    unsafe {
        // 0 buffer count = preserve existing; keep format + flags.
        swapchain.ResizeBuffers(
            0,
            width,
            height,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            DXGI_SWAP_CHAIN_FLAG(0),
        )
    }
    .map_err(|e| d3d_err("ResizeBuffers", e))?;
    create_rtv(device, swapchain)
}

/// Default SRV (inherits the texture's declared format) — used by the
/// CPU upload path's standalone R8 / R8G8 textures.
fn create_srv(
    device: &ID3D11Device,
    texture: &ID3D11Texture2D,
) -> Result<ID3D11ShaderResourceView> {
    let mut srv = None;
    unsafe { device.CreateShaderResourceView(texture, None, Some(&mut srv)) }
        .map_err(|e| d3d_err("CreateShaderResourceView", e))?;
    srv.ok_or_else(|| RenderError::GraphicsApi("null shader resource view".into()))
}

/// Map a biplanar staging `DXGI_FORMAT` (as a raw `u32`) to its
/// `(luma, chroma)` plane SRV formats. NV12 is 8-bit (R8 + R8G8); P010 is
/// 10-bit stored MSB-aligned in 16-bit cells (R16 + R16G16), normalised
/// back by the shader's limited-range-10 branch. Any other format is a
/// negotiation/decoder bug — fail loudly rather than sample garbage.
#[allow(clippy::cast_sign_loss)]
fn plane_srv_formats(format: u32) -> Result<(DXGI_FORMAT, DXGI_FORMAT)> {
    if format == DXGI_FORMAT_NV12.0 as u32 {
        Ok((DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM))
    } else if format == DXGI_FORMAT_P010.0 as u32 {
        Ok((DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16G16_UNORM))
    } else {
        Err(RenderError::GraphicsApi(format!(
            "unsupported decode texture format {format:#x}; expected NV12 (8-bit) or P010 (10-bit)"
        )))
    }
}

/// Cross-crate accessor for the renderer's decode-format accept table.
/// Returns the `(luma, chroma)` plane-SRV `DXGI_FORMAT`s (as raw `u32`s)
/// the native D3D11 renderer would bind to sample a decoded biplanar
/// texture of `format`, or `None` if the renderer rejects it.
///
/// Exists so a no-hardware unit test in `tether-client` can confirm this
/// accept table agrees with the decoder's
/// `tether_codec::d3d11::expected_decode_dxgi_format` — the Windows analog
/// of the macOS `accepts_iosurface_fourcc` cross-table check. A decoder
/// that emits a format the renderer silently drops is exactly bug
/// `621badc` (Main10's `'x420'` rejected by the renderer), so catching the
/// drift in default CI is cheaper than on a user's desk. The neutral `u32`
/// currency keeps `tether-client` free of a `windows`-crate dependency.
#[allow(clippy::cast_sign_loss)]
pub fn decode_plane_srv_formats(format: u32) -> Option<(u32, u32)> {
    plane_srv_formats(format)
        .ok()
        .map(|(y, uv)| (y.0 as u32, uv.0 as u32))
}

/// Format-explicit SRV used to view one plane of a biplanar texture:
/// e.g. NV12 luma as `R8_UNORM`, chroma as `R8G8_UNORM`. A default SRV
/// on a biplanar (NV12/P010) texture is invalid — the plane format must
/// be stated explicitly.
fn create_plane_srv(
    device: &ID3D11Device,
    texture: &ID3D11Texture2D,
    format: DXGI_FORMAT,
) -> Result<ID3D11ShaderResourceView> {
    let desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D_SRV_DIMENSION_TEXTURE2D,
        Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D11_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
            },
        },
    };
    let mut srv = None;
    unsafe { device.CreateShaderResourceView(texture, Some(&desc), Some(&mut srv)) }
        .map_err(|e| d3d_err("CreateShaderResourceView (plane)", e))?;
    srv.ok_or_else(|| RenderError::GraphicsApi("null plane SRV".into()))
}

/// Dynamic, CPU-writable, shader-readable texture for the CPU upload path.
fn create_dynamic_texture(
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
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        MiscFlags: 0,
    };
    let mut texture = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture)) }
        .map_err(|e| d3d_err("CreateTexture2D (dynamic)", e))?;
    texture.ok_or_else(|| RenderError::GraphicsApi("null dynamic texture".into()))
}

/// Map a dynamic texture and copy `src` into it row by row, honoring the
/// driver-chosen destination row pitch.
fn upload_rows(
    context: &ID3D11DeviceContext,
    texture: &ID3D11Texture2D,
    src: &[u8],
    src_row_bytes: usize,
    rows: u32,
) -> Result<()> {
    // Validate the source is large enough BEFORE mapping: a truncated
    // plane (e.g. a partial frame under loss) must not read past the
    // allocation inside the unsafe copy. Done here so nothing fallible
    // sits between Map and Unmap — a leaked map would wedge the texture.
    let required = (rows as usize)
        .checked_mul(src_row_bytes)
        .ok_or_else(|| RenderError::GraphicsApi("upload_rows: dimension overflow".into()))?;
    if src.len() < required {
        return Err(RenderError::GraphicsApi(format!(
            "upload_rows: src too short ({} < {required})",
            src.len()
        )));
    }

    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    unsafe { context.Map(texture, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut mapped)) }
        .map_err(|e| d3d_err("Map (dynamic texture)", e))?;
    let dst_pitch = mapped.RowPitch as usize;
    // SAFETY: bounds checked above; dst spans `rows * dst_pitch` (the
    // driver-allocated mapped region). No fallible call between here and
    // Unmap, so the map is always released.
    unsafe {
        for row in 0..rows as usize {
            let dst = (mapped.pData as *mut u8).add(row * dst_pitch);
            std::ptr::copy_nonoverlapping(
                src.as_ptr().add(row * src_row_bytes),
                dst,
                src_row_bytes,
            );
        }
        context.Unmap(texture, 0);
    }
    Ok(())
}

/// Compile the embedded HLSL and build the shaders, sampler, and the
/// per-frame constant buffer.
// The shader-params struct size is a small constant that fits u32.
#[allow(clippy::cast_possible_truncation)]
fn build_pipeline(device: &ID3D11Device) -> Result<YuvPipeline> {
    const HLSL: &[u8] = include_bytes!("yuv.hlsl");

    let vs_blob = compile_shader(HLSL, s!("vs"), s!("vs_5_0"))?;
    let ps_blob = compile_shader(HLSL, s!("ps"), s!("ps_5_0"))?;

    let mut vs = None;
    unsafe { device.CreateVertexShader(blob_bytes(&vs_blob), None, Some(&mut vs)) }
        .map_err(|e| d3d_err("CreateVertexShader", e))?;
    let mut ps = None;
    unsafe { device.CreatePixelShader(blob_bytes(&ps_blob), None, Some(&mut ps)) }
        .map_err(|e| d3d_err("CreatePixelShader", e))?;

    let sampler_desc = D3D11_SAMPLER_DESC {
        Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
        AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
        AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    let mut sampler = None;
    unsafe { device.CreateSamplerState(&sampler_desc, Some(&mut sampler)) }
        .map_err(|e| d3d_err("CreateSamplerState", e))?;

    let cbuffer_desc = D3D11_BUFFER_DESC {
        ByteWidth: std::mem::size_of::<ShaderParams>() as u32,
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
        ..Default::default()
    };
    let mut cbuffer = None;
    unsafe { device.CreateBuffer(&cbuffer_desc, None, Some(&mut cbuffer)) }
        .map_err(|e| d3d_err("CreateBuffer (cbuffer)", e))?;

    // No-cull rasterizer: the fullscreen quad is back-facing under the
    // default state, which would cull it entirely (see field doc).
    let raster_desc = D3D11_RASTERIZER_DESC {
        FillMode: D3D11_FILL_SOLID,
        CullMode: D3D11_CULL_NONE,
        ..Default::default()
    };
    let mut rasterizer = None;
    unsafe { device.CreateRasterizerState(&raster_desc, Some(&mut rasterizer)) }
        .map_err(|e| d3d_err("CreateRasterizerState", e))?;

    let cbuffer = cbuffer.ok_or_else(|| RenderError::GraphicsApi("null cbuffer".into()))?;
    let sampler = sampler.ok_or_else(|| RenderError::GraphicsApi("null sampler".into()))?;
    Ok(YuvPipeline {
        vs: vs.ok_or_else(|| RenderError::GraphicsApi("null vertex shader".into()))?,
        ps: ps.ok_or_else(|| RenderError::GraphicsApi("null pixel shader".into()))?,
        cbuffers: [Some(cbuffer.clone())],
        samplers: [Some(sampler)],
        rasterizer: rasterizer.ok_or_else(|| RenderError::GraphicsApi("null rasterizer".into()))?,
        cbuffer,
    })
}

fn compile_shader(src: &[u8], entry: PCSTR, target: PCSTR) -> Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    let result = unsafe {
        D3DCompile(
            src.as_ptr().cast(),
            src.len(),
            PCSTR::null(),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = result {
        let detail = errors
            .as_ref()
            .map(|b| unsafe {
                let bytes = std::slice::from_raw_parts(
                    b.GetBufferPointer().cast::<u8>(),
                    b.GetBufferSize(),
                );
                String::from_utf8_lossy(bytes).into_owned()
            })
            .unwrap_or_default();
        return Err(RenderError::GraphicsApi(format!(
            "D3DCompile: {e}: {detail}"
        )));
    }
    code.ok_or_else(|| RenderError::GraphicsApi("D3DCompile produced no bytecode".into()))
}

/// Bytecode view over a compiled shader blob.
fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    }
}

/// IMMUTABLE, shader-readable texture initialised from `data` — the
/// synthetic-plane source for headless render tests.
#[cfg(test)]
fn make_immutable_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    data: &[u8],
    row_pitch: u32,
) -> ID3D11Texture2D {
    use windows::Win32::Graphics::Direct3D11::{D3D11_SUBRESOURCE_DATA, D3D11_USAGE_IMMUTABLE};
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_IMMUTABLE,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let init = D3D11_SUBRESOURCE_DATA {
        pSysMem: data.as_ptr().cast(),
        SysMemPitch: row_pitch,
        SysMemSlicePitch: 0,
    };
    let mut tex = None;
    unsafe { device.CreateTexture2D(&desc, Some(&init), Some(&mut tex)) }
        .expect("make_immutable_texture");
    tex.unwrap()
}

#[cfg(test)]
// DXGI_FORMAT newtypes wrap i32; comparing/forwarding them as raw u32 (the
// cross-crate currency) is the same cast the production helpers above
// silence. Scope the allow to the test module rather than dotting it on
// each assertion.
#[allow(clippy::cast_sign_loss)]
mod tests {
    use super::*;
    use tether_protocol::control::{CodecKind, VideoColorSpec};

    /// Pins the renderer's decode-format → plane-SRV table against the
    /// real `DXGI_FORMAT` constants: NV12 (8-bit) samples through R8 luma
    /// plus R8G8 chroma; P010 (10-bit MSB-aligned) through R16 plus R16G16
    /// chroma. Anything else is a negotiation/decoder bug and must be
    /// rejected rather than sampled as garbage. No GPU — pure table check,
    /// the renderer side of the cross-crate agreement asserted in
    /// tether-client's
    /// `windows_format_tables::decoder_output_is_subset_of_renderer_accept`.
    #[test]
    fn plane_srv_formats_maps_nv12_and_p010_rejects_others() {
        assert_eq!(
            plane_srv_formats(DXGI_FORMAT_NV12.0 as u32).unwrap(),
            (DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM),
            "NV12 must sample as R8 luma + R8G8 chroma"
        );
        assert_eq!(
            plane_srv_formats(DXGI_FORMAT_P010.0 as u32).unwrap(),
            (DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16G16_UNORM),
            "P010 must sample as R16 luma + R16G16 chroma"
        );
        // A 4:2:0 8-bit *render-target* format and an unrelated format both
        // stand in for "anything the decoder should never emit": rejected.
        assert!(plane_srv_formats(DXGI_FORMAT_B8G8R8A8_UNORM.0 as u32).is_err());
        assert!(plane_srv_formats(0).is_err());
    }

    /// The public `u32`-currency accessor must agree with the internal
    /// `plane_srv_formats`: `Some` exactly when the internal table accepts,
    /// carrying the same plane formats. Guards against the wrapper drifting
    /// from the table it exposes to the cross-crate test.
    #[test]
    fn decode_plane_srv_formats_mirrors_internal_table() {
        for fmt in [DXGI_FORMAT_NV12.0 as u32, DXGI_FORMAT_P010.0 as u32] {
            let (y, uv) = plane_srv_formats(fmt).unwrap();
            assert_eq!(
                decode_plane_srv_formats(fmt),
                Some((y.0 as u32, uv.0 as u32))
            );
        }
        assert_eq!(decode_plane_srv_formats(0), None);
    }

    /// Drives the real `render` path headlessly with synthetic NV12 planes
    /// (mid-luma + neutral chroma) and reads back the result. Isolates the
    /// import → SRV → YUV shader → RTV chain from the decoder and the
    /// swapchain: if this is black, the bug is in the render/shader path;
    /// if it's the expected gray, the black-screen loopback bug is
    /// swapchain/present-side (or decoder-export-side).
    #[test]
    #[ignore = "requires D3D11 GPU (Windows)"]
    fn synthetic_nv12_renders_non_black_gray() {
        let (w, h) = (64u32, 64u32);
        let mut state = D3D11RenderState::new_headless(w, h, VideoColorSpec::sdr_bt709(), 8)
            .expect("headless renderer");

        // Y = 180 (limited-range luma); neutral chroma (128,128) → gray.
        let y = vec![180u8; (w * h) as usize];
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let uv = vec![128u8; (cw * ch * 2) as usize];
        state.upload_test_planes(&y, &uv, w, h);

        state.render().expect("render");
        let bgra = state.read_back_bgra();

        let idx = (((h / 2) * w + (w / 2)) * 4) as usize;
        let (b, g, r) = (bgra[idx], bgra[idx + 1], bgra[idx + 2]);
        eprintln!("center BGRA = ({b}, {g}, {r})");
        assert!(
            r > 40 && g > 40 && b > 40,
            "render produced ~black for mid-luma input: ({b}, {g}, {r})"
        );
        let (max, min) = (i32::from(r.max(g).max(b)), i32::from(r.min(g).min(b)));
        assert!(
            max - min < 40,
            "neutral chroma should be near-gray: ({b}, {g}, {r})"
        );
    }

    /// Drives the cursor overlay through the real `render` path: a gray
    /// video frame underneath, then an opaque red cursor sprite fed via
    /// the cursor channel exactly as the client's wire task does. Reads
    /// back and asserts the sprite's center is red (overlay composited)
    /// while a corner outside the sprite stays gray (video untouched).
    /// Exercises sprite upload → SRV → alpha-blend draw on top of the
    /// video, and that the blend state doesn't bleed onto the video.
    #[test]
    #[ignore = "requires D3D11 GPU (Windows)"]
    fn cursor_overlay_composites_over_video() {
        let (w, h) = (64u32, 64u32);
        let mut state = D3D11RenderState::new_headless(w, h, VideoColorSpec::sdr_bt709(), 8)
            .expect("headless renderer");

        // Gray video underneath (mid-luma, neutral chroma).
        let y = vec![180u8; (w * h) as usize];
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let uv = vec![128u8; (cw * ch * 2) as usize];
        state.upload_test_planes(&y, &uv, w, h);

        // 16×16 fully-opaque red sprite (straight RGBA8). Place it
        // centered with a centered hotspot so its 16×16 footprint
        // straddles (24..40, 24..40) in window pixels (video==surface).
        let (cur_w, cur_h) = (16u32, 16u32);
        let mut pixels = Vec::with_capacity((cur_w * cur_h * 4) as usize);
        for _ in 0..(cur_w * cur_h) {
            pixels.extend_from_slice(&[255, 0, 0, 255]); // R,G,B,A
        }
        let channel = state.cursor_channel();
        channel.with(|s| {
            s.enqueue_shape(1, cur_w, cur_h, cur_w / 2, cur_h / 2, pixels);
            s.activate(1);
            s.set_host_visible(true);
            s.set_local_pointer(Some(((w / 2) as f32, (h / 2) as f32)));
        });

        state.render().expect("render");
        let bgra = state.read_back_bgra();

        // Center is inside the sprite → red.
        let c = (((h / 2) * w + (w / 2)) * 4) as usize;
        let (cb, cg, cr) = (bgra[c], bgra[c + 1], bgra[c + 2]);
        eprintln!("cursor center BGRA = ({cb}, {cg}, {cr})");
        assert!(
            cr > 150 && cg < 80 && cb < 80,
            "sprite center should be red (overlay), got ({cb}, {cg}, {cr})"
        );

        // A corner well outside the 16×16 footprint → untouched gray.
        let k = ((2 * w + 2) * 4) as usize;
        let (kb, kg, kr) = (bgra[k], bgra[k + 1], bgra[k + 2]);
        eprintln!("corner BGRA = ({kb}, {kg}, {kr})");
        let (max, min) = (i32::from(kr.max(kg).max(kb)), i32::from(kr.min(kg).min(kb)));
        assert!(
            kr > 40 && kg > 40 && kb > 40 && max - min < 40,
            "corner outside sprite should stay near-gray video, got ({kb}, {kg}, {kr})"
        );
    }

    /// Full cross-device chain on the coordinate fixture: QSV encodes the
    /// gradient fixture, the D3D11VA decoder exports it GPU-resident
    /// (shared NT handles), the renderer opens those handles on its own
    /// device and renders offscreen. The geometric residual catches both
    /// decoder-export corruption (wrong array slice / plane / format) and
    /// renderer sampling bugs — a uniform-green or scrambled result blows
    /// the residual far past the quantisation floor. Windows analog of the
    /// Linux `dmabuf_test` roundtrip. Intel-only (QSV); SKIPs elsewhere.
    ///
    /// Run at both bit depths via the two `#[test]` wrappers below: 8-bit
    /// exercises the NV12 → R8/R8G8 path, 10-bit the P010 → R16/R16G16
    /// path (Main10 decode + the shader's limited-range-10 branch).
    #[test]
    #[ignore = "requires Intel QSV (Windows) + working oneVPL-over-D3D11"]
    fn d3d11_coord_fixture_decode_render_roundtrip_8bit() {
        d3d11_roundtrip(CodecKind::Hevc, 8, false, None);
    }

    #[test]
    #[ignore = "requires Intel QSV (Windows) + working oneVPL-over-D3D11 (Main10)"]
    fn d3d11_coord_fixture_decode_render_roundtrip_10bit() {
        d3d11_roundtrip(CodecKind::Hevc, 10, false, None);
    }

    /// Client upscale: decode at video dims (1280×720) and render into a
    /// larger surface (1920×1080), so the renderer's Mitchell upscale +
    /// letterbox stage runs — the one render-path stage the identity cells
    /// above never exercise. Geometric residual (coord fixture) confirms the
    /// scale/letterbox math, not just that pixels survive. Mirrors the Linux
    /// `*_client_upscale` dma-buf cells.
    #[test]
    #[ignore = "requires Intel QSV (Windows) + working oneVPL-over-D3D11"]
    fn d3d11_hevc_client_upscale_render_roundtrip_8bit() {
        d3d11_roundtrip(CodecKind::Hevc, 8, false, Some((1920, 1080)));
    }

    /// Surface below video: decode at 1280×720 and render into a smaller
    /// surface (960×540), so the renderer letterbox-fits by *downscaling* —
    /// the blit pass's non-identity-but-shrinking branch, separate arithmetic
    /// from the upscale path above. Mirrors the Linux
    /// `h264_8bit_surface_below_video` cell.
    #[test]
    #[ignore = "requires Intel QSV (Windows) + working oneVPL-over-D3D11"]
    fn d3d11_hevc_surface_below_video_render_roundtrip_8bit() {
        d3d11_roundtrip(CodecKind::Hevc, 8, false, Some((960, 540)));
    }

    /// Colour-decode validation: red/green/blue/white bars through the
    /// production QSV encode → D3D11VA decode → native D3D11 render path.
    /// CoordEncoded's near-constant chroma can't catch a hue cast / Cb/Cr
    /// swap; the white bar here can. Shared pattern with macOS/Linux via
    /// `crate::color_fixture`. Windows is 4:2:0 only (NV12 / P010).
    #[test]
    #[ignore = "requires Intel QSV (Windows) + working oneVPL-over-D3D11"]
    fn d3d11_colorbars_decode_render_roundtrip_8bit() {
        d3d11_roundtrip(CodecKind::Hevc, 8, true, None);
    }

    #[test]
    #[ignore = "requires Intel QSV (Windows) + working oneVPL-over-D3D11 (Main10)"]
    fn d3d11_colorbars_decode_render_roundtrip_10bit() {
        d3d11_roundtrip(CodecKind::Hevc, 10, true, None);
    }

    /// H.264 (the floor codec) through the full QSV encode → D3D11VA decode →
    /// native D3D11 render path. HEVC has 8 + 10-bit cells above; H.264 ships
    /// as Main 8-bit 4:2:0 only (NV12 → R8/R8G8), so it gets the same
    /// coord-geometry + colour-bar pair at 8-bit. Guards the H.264 decode SRV
    /// + shader branch end-to-end, not just the codec round-trip.
    #[test]
    #[ignore = "requires Intel QSV (Windows) + working oneVPL-over-D3D11"]
    fn d3d11_h264_coord_fixture_decode_render_roundtrip_8bit() {
        d3d11_roundtrip(CodecKind::H264, 8, false, None);
    }

    #[test]
    #[ignore = "requires Intel QSV (Windows) + working oneVPL-over-D3D11"]
    fn d3d11_h264_colorbars_decode_render_roundtrip_8bit() {
        d3d11_roundtrip(CodecKind::H264, 8, true, None);
    }

    /// `surface`: the render-target dims. `None` = identity (surface == video,
    /// the encode/decode/render measurement is pure). `Some(dims)` larger than
    /// the video drives the renderer's upscale + letterbox path.
    fn d3d11_roundtrip(
        codec: CodecKind,
        bit_depth: u8,
        colorbars: bool,
        surface: Option<(u32, u32)>,
    ) {
        use tether_codec::d3d11::{D3D11Decoder, D3D11Encoder};
        use tether_codec::{D3D11TextureFrame, Decoder, Encoder, Frame as CodecFrame};
        use tether_protocol::control::{ChromaSubsampling, VideoProfile};
        use tether_scaler::test_util::{
            coord_fixture_fill, coord_fixture_residual_px_rms, LetterboxMap,
        };
        use windows::core::Interface;
        use windows::Win32::Foundation::HMODULE;
        use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
        use windows::Win32::Graphics::Direct3D11::{
            D3D11CreateDevice, ID3D11Multithread, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT, D3D11_SDK_VERSION, D3D11_SUBRESOURCE_DATA,
            D3D11_USAGE_DEFAULT,
        };
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        use windows::Win32::Graphics::Dxgi::{IDXGIAdapter, IDXGIDevice};

        const VENDOR_INTEL: u32 = 0x8086;
        let (w, h) = (1280u32, 720u32);

        // Shared video device for the QSV encoder (VIDEO_SUPPORT + the
        // multithread protection QSV requires on a derived device).
        let mut device = None;
        let mut context = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        }
        .expect("D3D11CreateDevice");
        let device: ID3D11Device = device.unwrap();
        let context = context.unwrap();
        if let Ok(mt) = device.cast::<ID3D11Multithread>() {
            let _ = unsafe { mt.SetMultithreadProtected(true) };
        }

        let vendor = unsafe {
            let dxgi: IDXGIDevice = device.cast().unwrap();
            let adapter: IDXGIAdapter = dxgi.GetAdapter().unwrap();
            adapter.GetDesc().map(|d| d.VendorId).unwrap_or(0)
        };
        if vendor != VENDOR_INTEL {
            eprintln!("SKIP: GPU vendor 0x{vendor:04x} != Intel; run on a QSV-capable GPU");
            return;
        }

        // BGRA source at capture==encode dims (identity VP scale keeps
        // the metric a pure encode/decode/render measurement).
        let fixture = if colorbars {
            crate::color_fixture::colorbars_bgra((w, h))
        } else {
            coord_fixture_fill((w, h))
        };
        let tex_desc = D3D11_TEXTURE2D_DESC {
            Width: w,
            Height: h,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            // No bind flags: the encoder's VP-blit input validation
            // rejects a SHADER_RESOURCE-bound source (UnsupportedInputFormat).
            BindFlags: 0,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let init = D3D11_SUBRESOURCE_DATA {
            pSysMem: fixture.as_ptr().cast(),
            SysMemPitch: w * 4,
            SysMemSlicePitch: 0,
        };
        let mut src = None;
        unsafe { device.CreateTexture2D(&tex_desc, Some(&init), Some(&mut src)) }
            .expect("fixture CreateTexture2D");
        let src = src.unwrap();

        let profile = VideoProfile {
            codec,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth,
        };
        let expected_backend = match codec {
            CodecKind::H264 => "h264_qsv",
            CodecKind::Hevc => "hevc_qsv",
            CodecKind::Av1 => "av1_qsv",
        };
        let mut enc = D3D11Encoder::new(
            profile,
            w,
            h,
            60,
            8000,
            device.as_raw() as *mut _,
            context.as_raw() as *mut _,
            VENDOR_INTEL,
        )
        .expect("QSV encoder");
        assert_eq!(
            enc.name(),
            expected_backend,
            "QSV unavailable; got {}",
            enc.name()
        );

        let mut dec = D3D11Decoder::new(codec, true).expect("decoder");
        let frame_desc = D3D11TextureFrame {
            texture: src.as_raw() as *mut _,
            device: device.as_raw() as *mut _,
            device_context: context.as_raw() as *mut _,
            width: w,
            height: h,
            format: DXGI_FORMAT_B8G8R8A8_UNORM.0 as u32,
        };

        // Encode the static fixture and decode until a GPU frame settles;
        // keep the latest (post-IDR, fully-formed).
        let mut decoded: Option<CodecFrame> = None;
        for pts in 0..90 {
            let pkts = enc
                .submit_d3d11_texture(&frame_desc, pts, pts == 0)
                .expect("encode");
            for pkt in &pkts {
                dec.submit(&pkt.data).expect("submit");
            }
            if let Some(f) = dec.next_frame().expect("next_frame") {
                decoded = Some(f);
            }
        }
        let CodecFrame::Gpu(g) = decoded.expect("no GPU frame decoded") else {
            panic!("gpu_export must yield Frame::Gpu");
        };
        let (gw, gh, _pts, source, guard) = g.into_parts();
        let render_frame = Frame::Gpu(crate::GpuFrame {
            width: gw,
            height: gh,
            t_capture_client_clock: None,
            source,
            guard,
        });

        // Render through the production import/SRV/shader path on a fresh
        // device (cross-device, like the real client), read back, measure.
        // `surface` defaults to the video dims (identity); a larger surface
        // engages the renderer's upscale + letterbox stage.
        let (sw, sh) = surface.unwrap_or((w, h));
        let mut state =
            D3D11RenderState::new_headless(sw, sh, VideoColorSpec::sdr_desktop(), bit_depth)
                .expect("headless");
        state.apply_frame(render_frame).expect("apply_frame");
        state.render().expect("render");
        let bgra = state.read_back_bgra();

        if colorbars {
            crate::color_fixture::assert_colorbars(
                "d3d11 colorbars",
                &bgra,
                sw,
                sh,
                crate::color_fixture::ChannelOrder::Bgra,
            );
        } else {
            let map = LetterboxMap::new((w, h), (sw, sh));
            let residual = coord_fixture_residual_px_rms(&bgra, (sw, sh), &map);
            let mid = (((sh / 2) * sw + (sw / 2)) * 4) as usize;
            eprintln!(
                "coord-fixture residual = {residual:.1}px  center BGRA = ({}, {}, {})",
                bgra[mid],
                bgra[mid + 1],
                bgra[mid + 2]
            );
            // Upscale/letterbox adds interpolation spread on top of encoder
            // quantisation, so allow more headroom when the surface differs
            // from the video; identity stays tight.
            let threshold = if surface.is_some() { 150.0 } else { 80.0 };
            assert!(
                residual < threshold,
                "geometric residual {residual:.1}px exceeds {threshold:.0}px — decode-export or \
                 render corruption (uniform-green / scrambled chroma blows this metric past the \
                 quantisation floor)"
            );
        }
    }
}
