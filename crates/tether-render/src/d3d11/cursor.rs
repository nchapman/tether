//! D3D11 cursor overlay pass (Windows).
//!
//! Windows counterpart to the wgpu [`crate::cursor_overlay::CursorOverlay`].
//! Both consume the same backend-agnostic [`CursorState`] (fed by the
//! client's wire-receive task) and the shared [`cursor_uniform_for`]
//! placement math; this module owns only the D3D11 GPU side — compiling
//! `cursor.hlsl`, uploading the active sprite into an RGBA8 texture, and
//! drawing an alpha-blended quad over the already-rendered video.
//!
//! It runs inside [`super::D3D11RenderState::render`] after the YUV draw
//! and before `Present`: the swapchain RTV is still bound and holds the
//! video pixels, so the pass sets a straight-alpha blend state and draws
//! on top without clearing (the D3D11 analog of wgpu's `LoadOp::Load`).

use windows::core::s;
use windows::Win32::Graphics::Direct3D::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11BlendState, ID3D11Buffer, ID3D11Device, ID3D11DeviceContext, ID3D11PixelShader,
    ID3D11RasterizerState, ID3D11SamplerState, ID3D11ShaderResourceView, ID3D11Texture2D,
    ID3D11VertexShader, D3D11_BIND_CONSTANT_BUFFER, D3D11_BIND_SHADER_RESOURCE, D3D11_BLEND_DESC,
    D3D11_BLEND_INV_SRC_ALPHA, D3D11_BLEND_ONE, D3D11_BLEND_OP_ADD, D3D11_BLEND_SRC_ALPHA,
    D3D11_BUFFER_DESC, D3D11_COLOR_WRITE_ENABLE_ALL, D3D11_CULL_NONE, D3D11_FILL_SOLID,
    D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_RASTERIZER_DESC, D3D11_RENDER_TARGET_BLEND_DESC,
    D3D11_SAMPLER_DESC, D3D11_SUBRESOURCE_DATA, D3D11_TEXTURE2D_DESC, D3D11_TEXTURE_ADDRESS_CLAMP,
    D3D11_USAGE_DEFAULT, D3D11_USAGE_IMMUTABLE,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

use crate::cursor_overlay::{
    cursor_uniform_for, CursorChannel, CursorLogDims, CursorSnapshot, CURSOR_UNIFORM_BYTES,
};
use crate::{RenderError, Result};

use super::{blob_bytes, compile_shader, d3d_err};

/// The active sprite uploaded to a D3D11 texture, keyed by id so an
/// unchanged cursor isn't re-uploaded each frame. Owning the texture +
/// SRV here (rather than in [`CursorState`]) means a cache eviction
/// can't drop the texture out from under an in-flight draw.
struct ActiveSprite {
    id: u64,
    // Held for RAII: keeps the texture alive while the SRV views it.
    _texture: ID3D11Texture2D,
    /// Pre-built single-element binding array so `render` hands the
    /// `PSSetShaderResources` call a borrow rather than cloning the COM
    /// object each frame.
    srvs: [Option<ID3D11ShaderResourceView>; 1],
}

/// GPU-side D3D11 cursor overlay: shaders, sampler, blend state, the
/// per-frame constant buffer, and the currently-uploaded sprite.
pub(crate) struct D3D11CursorOverlay {
    vs: ID3D11VertexShader,
    ps: ID3D11PixelShader,
    cbuffer: ID3D11Buffer,
    /// Pre-built binding arrays so the hot path binds by borrow.
    cbuffers: [Option<ID3D11Buffer>; 1],
    samplers: [Option<ID3D11SamplerState>; 1],
    /// Straight-alpha (SrcAlpha, 1-SrcAlpha) blend so the sprite
    /// composites over the opaque video pixels.
    blend: ID3D11BlendState,
    /// No-cull rasterizer. The quad is wound the same as the video pass's
    /// (back-facing under D3D11's default front=CW + cull-back), so the
    /// pass sets its own `CULL_NONE` state rather than depending on the
    /// video pass leaving a compatible one bound.
    rasterizer: ID3D11RasterizerState,
    active: Option<ActiveSprite>,
    /// Dedup key for the cursor-render-params diagnostic — see
    /// [`cursor_uniform_for`].
    last_logged_dims: Option<CursorLogDims>,
}

impl D3D11CursorOverlay {
    pub(crate) fn new(device: &ID3D11Device) -> Result<Self> {
        const HLSL: &[u8] = include_bytes!("cursor.hlsl");
        let vs_blob = compile_shader(HLSL, s!("vs"), s!("vs_5_0"))?;
        let ps_blob = compile_shader(HLSL, s!("ps"), s!("ps_5_0"))?;

        let mut vs = None;
        unsafe { device.CreateVertexShader(blob_bytes(&vs_blob), None, Some(&mut vs)) }
            .map_err(|e| d3d_err("CreateVertexShader (cursor)", e))?;
        let mut ps = None;
        unsafe { device.CreatePixelShader(blob_bytes(&ps_blob), None, Some(&mut ps)) }
            .map_err(|e| d3d_err("CreatePixelShader (cursor)", e))?;

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
            .map_err(|e| d3d_err("CreateSamplerState (cursor)", e))?;

        let cbuffer_desc = D3D11_BUFFER_DESC {
            ByteWidth: CURSOR_UNIFORM_BYTES as u32,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
            ..Default::default()
        };
        let mut cbuffer = None;
        unsafe { device.CreateBuffer(&cbuffer_desc, None, Some(&mut cbuffer)) }
            .map_err(|e| d3d_err("CreateBuffer (cursor cbuffer)", e))?;

        // Straight-alpha blend: out = src.rgb*src.a + dst.rgb*(1-src.a),
        // out.a = src.a + dst.a*(1-src.a). Mirrors the wgpu BlendState.
        let mut blend_desc = D3D11_BLEND_DESC::default();
        blend_desc.RenderTarget[0] = D3D11_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            SrcBlend: D3D11_BLEND_SRC_ALPHA,
            DestBlend: D3D11_BLEND_INV_SRC_ALPHA,
            BlendOp: D3D11_BLEND_OP_ADD,
            SrcBlendAlpha: D3D11_BLEND_ONE,
            DestBlendAlpha: D3D11_BLEND_INV_SRC_ALPHA,
            BlendOpAlpha: D3D11_BLEND_OP_ADD,
            RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8,
        };
        let mut blend = None;
        unsafe { device.CreateBlendState(&blend_desc, Some(&mut blend)) }
            .map_err(|e| d3d_err("CreateBlendState (cursor)", e))?;

        let raster_desc = D3D11_RASTERIZER_DESC {
            FillMode: D3D11_FILL_SOLID,
            CullMode: D3D11_CULL_NONE,
            ..Default::default()
        };
        let mut rasterizer = None;
        unsafe { device.CreateRasterizerState(&raster_desc, Some(&mut rasterizer)) }
            .map_err(|e| d3d_err("CreateRasterizerState (cursor)", e))?;

        let cbuffer =
            cbuffer.ok_or_else(|| RenderError::GraphicsApi("null cursor cbuffer".into()))?;
        let sampler =
            sampler.ok_or_else(|| RenderError::GraphicsApi("null cursor sampler".into()))?;
        Ok(Self {
            vs: vs.ok_or_else(|| RenderError::GraphicsApi("null cursor vertex shader".into()))?,
            ps: ps.ok_or_else(|| RenderError::GraphicsApi("null cursor pixel shader".into()))?,
            cbuffers: [Some(cbuffer.clone())],
            cbuffer,
            samplers: [Some(sampler)],
            blend: blend
                .ok_or_else(|| RenderError::GraphicsApi("null cursor blend state".into()))?,
            rasterizer: rasterizer
                .ok_or_else(|| RenderError::GraphicsApi("null cursor rasterizer".into()))?,
            active: None,
            last_logged_dims: None,
        })
    }

    /// Upload `snap`'s straight-alpha RGBA8 pixels into a fresh immutable
    /// texture + SRV and make it the active sprite. Called only when the
    /// active id changes, so a static cursor pays nothing here. The
    /// texture is sampled non-sRGB (`R8G8B8A8_UNORM`) to match the wgpu
    /// path's `Rgba8Unorm` cursor texture — identical composite over the
    /// sRGB render target.
    fn upload_sprite(&mut self, device: &ID3D11Device, snap: &CursorSnapshot<'_>) -> Result<()> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: snap.width,
            Height: snap.height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_IMMUTABLE,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        // Host sends tight RGBA8 (4 bytes/texel, no row padding).
        let init = D3D11_SUBRESOURCE_DATA {
            pSysMem: snap.pixels.as_ptr().cast(),
            SysMemPitch: snap.width * 4,
            SysMemSlicePitch: 0,
        };
        let mut texture = None;
        unsafe { device.CreateTexture2D(&desc, Some(&init), Some(&mut texture)) }
            .map_err(|e| d3d_err("CreateTexture2D (cursor sprite)", e))?;
        let texture =
            texture.ok_or_else(|| RenderError::GraphicsApi("null cursor sprite texture".into()))?;

        let mut srv = None;
        unsafe { device.CreateShaderResourceView(&texture, None, Some(&mut srv)) }
            .map_err(|e| d3d_err("CreateShaderResourceView (cursor sprite)", e))?;
        let srv = srv.ok_or_else(|| RenderError::GraphicsApi("null cursor sprite SRV".into()))?;

        self.active = Some(ActiveSprite {
            id: snap.id,
            _texture: texture,
            srvs: [Some(srv)],
        });
        Ok(())
    }

    /// Draw the cursor over the bound render target. Best-effort: the
    /// video frame has already been drawn into `rtv`, so a sprite-upload
    /// failure logs and skips the overlay rather than failing the frame.
    /// No-ops (without touching device state) when the cursor is hidden,
    /// in relative mode, or no sprite is active.
    pub(crate) fn render(
        &mut self,
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        channel: &CursorChannel,
        video_dims: (u32, u32),
        surface_dims: (u32, u32),
        fit_dims: (u32, u32),
    ) {
        let (vw, vh) = video_dims;
        let (surf_w, surf_h) = surface_dims;
        if surf_w == 0 || surf_h == 0 || vw == 0 || vh == 0 {
            return;
        }
        // Snapshot + upload under the cursor-state lock; return the packed
        // uniform bytes to write after releasing it. `None` => don't draw.
        let bytes = channel.with(|state| {
            let snap = state.snapshot()?;
            if self.active.as_ref().map(|a| a.id) != Some(snap.id) {
                if let Err(e) = self.upload_sprite(device, &snap) {
                    tracing::warn!(error = %e, id = snap.id, "cursor sprite upload failed; skipping overlay");
                    // Clear any stale sprite so the upload is retried next
                    // frame — otherwise a transient failure (recovered OOM)
                    // would leave the unchanged active id matching forever
                    // and silently draw the previous sprite.
                    self.active = None;
                    return None;
                }
            }
            Some(cursor_uniform_for(
                &snap,
                video_dims,
                surface_dims,
                fit_dims,
                &mut self.last_logged_dims,
            ))
        });
        let Some(bytes) = bytes else {
            return;
        };
        // The overlay owns the active texture, so it outlives the lock
        // released above — the SRV bound below can't dangle.
        let Some(active) = self.active.as_ref() else {
            return;
        };

        unsafe {
            context.UpdateSubresource(&self.cbuffer, 0, None, bytes.as_ptr().cast(), 0, 0);
            // Straight-alpha blend over the loaded (video) RTV contents.
            // The video pass resets to opaque each frame, so this only
            // affects the cursor draw.
            context.OMSetBlendState(&self.blend, None, 0xffff_ffff);
            context.RSSetState(&self.rasterizer);
            context.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context.VSSetShader(&self.vs, None);
            context.PSSetShader(&self.ps, None);
            context.VSSetConstantBuffers(0, Some(&self.cbuffers));
            context.PSSetShaderResources(0, Some(&active.srvs));
            context.PSSetSamplers(0, Some(&self.samplers));
            context.Draw(6, 0);
        }
    }
}
