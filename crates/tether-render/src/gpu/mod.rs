use std::sync::{Arc, OnceLock};

use tether_codec::{GpuFrameGuard, GpuFrameSource};
use tether_protocol::control::{ChromaSubsampling, ColorTransfer, VideoColorSpec, VideoProfile};
use winit::window::Window;

use crate::{CpuFrame, Frame, GpuFrame, RenderError, Result};

#[cfg(target_os = "linux")]
mod import;
#[cfg(target_os = "linux")]
pub(crate) use import::import_dmabuf_textures;

#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
pub(crate) use metal::import_iosurface_textures;
// Re-export from tether-codec's shared interop module. Lived in
// `metal.rs` historically; moved to `tether-codec::macos_interop` so
// `tether-gpuconvert::nv12_iosurface` (host scaler bridge) can share
// the same table without depending on the renderer.
#[cfg(target_os = "macos")]
pub use tether_codec::macos_interop::accepts_iosurface_fourcc;

/// One frame's worth of YUV plane textures plus the bind group that
/// points at them. The variant matches the negotiated chroma — NV12
/// shape for 4:2:0, three R8 planes for 4:4:4. Bundled with the bind
/// group so resize-on-frame can swap them atomically.
pub(crate) struct YuvTextures {
    /// Per-chroma plane storage. Different variants in different
    /// modes; bind group already references the correct plane set
    /// via the (chroma-specific) bind-group layout.
    ///
    /// FIELD ORDER IS LOAD-BEARING. Rust drops struct fields in
    /// declaration order, and on macOS the MTLTexture inside each plane
    /// holds the IOSurface alive (via Metal's internal retain), while
    /// `_guard` below holds it alive via the AVFrame's CVPixelBuffer
    /// retain. The texture drop must run first so Metal releases its
    /// retain before the AVFrame drops the CVPixelBuffer; otherwise a
    /// future reorder could leave a dangling MTLTexture pointing at a
    /// released IOSurface. Same ordering serves the Linux side (wgpu
    /// texture releases the imported DMA-BUF before the VAAPI surface
    /// returns to the pool).
    planes: YuvPlanes,
    pub(crate) bind_group: wgpu::BindGroup,
    /// Luma-plane dimensions (chroma is derived from chroma kind).
    size: (u32, u32),
    /// Backend-side lifetime extender from the decoder (DMA-BUF path
    /// on Linux, AVFrame holding the CVPixelBuffer on macOS). `None`
    /// for CPU-uploaded textures. Declared *after* `planes` so the
    /// textures' Drop runs first — see field-order note above.
    _guard: Option<GpuFrameGuard>,
}

/// Chroma-specific plane storage. The texture handles are kept alive
/// solely for the bind group's views; we never reference them by name
/// after the bind group is built.
#[allow(dead_code)]
pub(crate) enum YuvPlanes {
    /// 8-bit biplanar: full-res R8 Y plus Rg8 UV. NV12 (4:2:0
    /// half-res UV) and NV24 (4:4:4 full-res UV) take the same
    /// variant; the dimensions differ but the format and shader
    /// don't.
    Biplanar8 { y: wgpu::Texture, uv: wgpu::Texture },
    /// 10-bit (or 16-bit) biplanar: R16Unorm Y plus Rg16Unorm UV.
    /// Used by macOS `'P410'`/`'xf44'` IOSurfaces and Linux
    /// `DRM_FORMAT_P010`/`P410` dma-bufs. The 10-bit data is
    /// MSB-aligned in the 16-bit storage cell (Apple + DRM
    /// convention); the shader's `luma_scale` uniform compensates
    /// the resulting `≈0.999` max-value sampler reading back to
    /// `1.0` so the limited-range expansion lands on the right
    /// breakpoints.
    Biplanar16 { y: wgpu::Texture, uv: wgpu::Texture },
    /// Packed 4:4:4 in a single texture: 8-bit XYUV8888 (Rgba8, byte
    /// order V/U/Y/X — see `shader_yuv444.wgsl`) or 10-bit Y410 (Rgb10a2,
    /// 10:10:10:2, channels R=U/G=Y/B=V — DRM "X:Cr:Y:Cb" spec order, see
    /// `shader_yuv444_10bit.wgsl`).
    /// The variant holds one texture either way; the `RenderLayout`
    /// (`PackedXYUV` vs `PackedY410`) picks the matching shader.
    Yuv444Packed { packed: wgpu::Texture },
    /// Planar 8-bit 4:4:4: full-resolution R8 Y, U, and V planes.
    /// NVIDIA NVDEC exports HEVC Main 4:4:4 this way (`DRM_FORMAT_YUV444` /
    /// `YU24`), unlike VAAPI's packed XYUV output.
    Yuv444Planar {
        y: wgpu::Texture,
        u: wgpu::Texture,
        v: wgpu::Texture,
    },
}

/// Where rendered frames go. The production path owns a swapchain
/// (`Swapchain`); the round-trip test harness binds an offscreen target
/// (`Offscreen`) so the same multi-pass `render()` body drives both,
/// closing the gap that previously kept production-renderer bugs out
/// of hardware-test coverage. Test renderer is created via
/// [`GpuState::new_headless`].
pub(crate) enum RenderTarget {
    /// Live present path. Owns the wgpu surface + configuration; the
    /// only path that calls `output.present()`.
    Swapchain {
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    },
    /// Headless target for round-trip tests. The render output is read
    /// back from this texture, never presented. Same multi-pass body,
    /// same blit color attachment format, no synthetic differences in
    /// the GPU command stream. Consumed by the test harness in
    /// `tether-render/src/test_harness.rs`.
    #[allow(dead_code)]
    Offscreen {
        target: wgpu::Texture,
        dims: (u32, u32),
        format: wgpu::TextureFormat,
    },
}

impl RenderTarget {
    fn dimensions(&self) -> (u32, u32) {
        match self {
            RenderTarget::Swapchain { config, .. } => (config.width, config.height),
            RenderTarget::Offscreen { dims, .. } => *dims,
        }
    }

    #[allow(dead_code)] // used by the offscreen test harness
    fn format(&self) -> wgpu::TextureFormat {
        match self {
            RenderTarget::Swapchain { config, .. } => config.format,
            RenderTarget::Offscreen { format, .. } => *format,
        }
    }

    /// Resize routes through here so the swapchain variant reconfigures
    /// while the offscreen variant reallocates its target texture.
    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if self.dimensions() == (width, height) {
            return;
        }
        match self {
            RenderTarget::Swapchain { surface, config } => {
                config.width = width;
                config.height = height;
                surface.configure(device, config);
            }
            RenderTarget::Offscreen {
                target,
                dims,
                format,
            } => {
                *dims = (width, height);
                *target = make_offscreen_target(device, *format, (width, height));
            }
        }
    }
}

pub(crate) struct GpuState {
    target: RenderTarget,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    yuv_bgl: wgpu::BindGroupLayout,
    /// Negotiated chroma for this session. Fixed at construction —
    /// a chroma switch needs a full GpuState rebuild, the same way
    /// a resolution change works.
    chroma: ChromaSubsampling,
    /// Negotiated bit depth (8 or 10 in practice). Pairs with
    /// `chroma` to pick the `RenderLayout` and the texture formats
    /// in [`make_yuv_textures`]. A bit-depth switch needs a full
    /// rebuild for the same reason chroma does (different texture
    /// formats, different shader luma_scale).
    bit_depth: u8,
    sampler: wgpu::Sampler,
    textures: YuvTextures,
    /// Previous frame's textures + guard, retired for one extra render
    /// cycle so any in-flight GPU sampling completes before the dma-buf
    /// memory is freed. Without this, the AVFrame guard would drop the
    /// VAAPI surface back to the pool the instant `textures` is
    /// replaced; the decoder could then reuse it for the next frame
    /// while the GPU is still mid-sample, producing torn output on
    /// drivers that don't attach Vulkan read fences to the dma-buf
    /// reservation object. Mesa+Intel does attach them; this slot is
    /// cheap insurance against drivers that don't.
    retired: Option<YuvTextures>,
    scale_buffer: wgpu::Buffer,
    scale_bind_group: wgpu::BindGroup,
    /// Color-params uniform: holds the EOTF kind the fragment shader
    /// dispatches on (`bt709_eotf` vs `srgb_eotf`). Written once at
    /// pipeline-build time from the negotiated `VideoColorSpec`.
    /// Kept on the struct (rather than dropped after the write) so
    /// a future mid-session resign — when the protocol grows one —
    /// can `queue.write_buffer` into it without rebuilding the pipeline.
    #[allow(dead_code)]
    color_params_buffer: wgpu::Buffer,
    color_params_bind_group: wgpu::BindGroup,
    /// True if `VULKAN_EXTERNAL_MEMORY_DMA_BUF` was negotiated on the
    /// adapter. Gates the DMA-BUF import path; without it, a `GpuFrame`
    /// from the decoder gets dropped with a warn — the decoder probe
    /// shouldn't have picked the HW backend in that case, so reaching
    /// this branch indicates a misconfiguration.
    #[cfg(target_os = "linux")]
    dmabuf_import_supported: bool,
    /// True if the wgpu device exposes a Metal HAL. Required by the
    /// IOSurface→MTLTexture→wgpu import path. Probed at construction so
    /// any backend mismatch (e.g. a forced Vulkan adapter on macOS via
    /// MoltenVK) surfaces here instead of as a first-frame error.
    #[cfg(target_os = "macos")]
    metal_import_supported: bool,
    /// Mitchell upscale stage. The renderer is a multi-pass pipeline:
    ///   pass 1: YUV → linear-light RGB into `rgb_intermediate`.
    ///   pass 2 (optional): Mitchell upscale `rgb_intermediate` → `scaler.output()`.
    ///   pass 3: blit (`rgb_intermediate` or `scaler.output()`) to the
    ///           swapchain with letterbox.
    ///   pass 4 (optional): cursor overlay sprite on top of the blit
    ///           output. Skipped when no sprite is active or the
    ///           cursor is hidden / in relative-locked mode.
    /// Building the intermediate + blit pipeline up-front avoids
    /// per-frame allocation; the Mitchell scaler is built lazily on
    /// the first frame that calls for upscale and rebuilt on window
    /// resize (or video-dims change).
    upscale: UpscaleStage,
    /// Cursor overlay pass + shared cursor state. The state is held
    /// behind an `Arc<Mutex<_>>` so the client's wire-receive task can
    /// write to it from another thread while the renderer reads each
    /// frame.
    cursor_overlay: crate::cursor_overlay::CursorOverlay,
    cursor_channel: crate::cursor_overlay::CursorChannel,
}

/// Holds the resources for the multi-pass present pipeline.
pub(crate) struct UpscaleStage {
    /// Rgba16Float texture at video dims. Render target of pass 1,
    /// sample source of pass 2 (Mitchell) or pass 3 (direct blit).
    /// `None` until the first frame's video dims are known.
    rgb_intermediate: Option<wgpu::Texture>,
    /// Cached video dims of the intermediate; lets us detect when
    /// the decoded frame changes shape and rebuild.
    intermediate_dims: (u32, u32),
    /// Shared Mitchell shader compilation. Built once on the device,
    /// reused across scaler rebuilds. Lazily constructed on first
    /// need so a session that never upscales never pays for shader
    /// compilation.
    scaler_pipelines: Option<Arc<tether_scaler::Pipelines>>,
    /// Mitchell upscale stage. `Some` only when `window > video`
    /// in at least one axis. Rebuilt on window resize or video
    /// dims change.
    scaler: Option<tether_scaler::Scaler>,
    /// Sticky failure marker: when scaler construction fails for a
    /// given target `upscale_dims`, we record those dims so the
    /// per-frame retry-loop skips that specific construction without
    /// silencing future attempts at a *different* set of dims (e.g.
    /// the user resizes the window again and the new ratio happens
    /// to compile fine). `None` means "no known failure" — try
    /// construction normally. Without the per-dims keying, a
    /// failure at one ratio would mask later valid ratios for as
    /// long as the video dims stayed the same.
    scaler_failed_for: Option<(u32, u32)>,
    /// Blit pipeline + bind-group layout for the swapchain pass.
    /// Samples an Rgba16Float texture with the project-standard
    /// bilinear letterbox. Bind groups are rebuilt per render
    /// because the source texture (intermediate vs scaler output)
    /// can change.
    blit_pipeline: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,
    /// Scratch (1,1,0,0) and letterbox (sx,sy,0,0) uniform buffers
    /// share the layout but need two separate writes per frame —
    /// pass 1 wants identity coverage, pass 3 wants the letterbox
    /// scale. Two buffers + two bind groups avoids the ordering
    /// hazard of writing the same buffer twice in one encoder.
    /// Retained to keep the bind group's referenced buffer alive
    /// (the bind group only borrows the binding handle, not the
    /// underlying allocation).
    #[allow(dead_code)]
    identity_scale_buffer: wgpu::Buffer,
    identity_scale_bind_group: wgpu::BindGroup,
}

/// Renderer plane layout — decoupled from `ChromaSubsampling` because
/// the same chroma can arrive in different GPU shapes depending on
/// backend + bit depth:
///
/// - **Biplanar8** = NV12 / NV24-style: Y in an R8 texture plus
///   interleaved UV in an Rg8 texture. Plane resolutions differ
///   between 4:2:0 (half-res UV) and 4:4:4 (full-res UV), but the
///   bind-group layout and fragment shader are identical — bilinear
///   sampling reads the right value at either resolution. Used by
///   the macOS IOSurface import path for both Yuv420 (`'420v'`) and
///   Yuv444 (`'444v'`), and by the Linux dma-buf path for Yuv420.
/// - **Biplanar16** = P010 / P410 / xf44-style: Y in R16Unorm plus
///   interleaved UV in Rg16Unorm, with 10-bit data MSB-aligned in
///   the 16-bit storage cells. Apple's `'P410'` IOSurface (10-bit
///   4:4:4 from VT HEVC Main 4:4:4 10-bit decode), `'xf44'` (same
///   shape via SCK capture), and Linux's `DRM_FORMAT_P010` dma-bufs land
///   here. Same bind-group layout + same shader as Biplanar8 — the
///   `range_kind` uniform in `color_params`
///   selects the 10-bit limited-range branch, which compensates the
///   MSB-align so max-value samples normalise to 1.0 instead of
///   `≈0.999`.
/// - **PackedXYUV** = one Rgba8 texture with byte order V/U/Y/X.
///   Only the Linux dma-buf path for Yuv444 8-bit — VAAPI's 4:4:4
///   8-bit surfaces are exposed this way and the shader pulls Y
///   from `.z`, U from `.y`, V from `.x`. There is no 10-bit
///   PackedXYUV path; Linux 10-bit 4:4:4 uses the separate
///   `PackedY410` / `Rgb10a2Unorm` shader path.
///
/// Adding a new `(chroma, bit_depth, backend)` combo means returning
/// the right variant from [`render_layout_for`] and (if it's a third
/// shape) wiring a new fragment shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RenderLayout {
    Biplanar8,
    Biplanar16,
    Planar444,
    PackedXYUV,
    /// Packed 4:4:4 10-bit (Y410 / `Rgb10a2Unorm`): one 10:10:10:2
    /// texture, native 10-bit (no MSB-align). Linux VAAPI decode-output
    /// shape for HEVC Main 4:4:4 10-bit; see `shader_yuv444_10bit.wgsl`.
    PackedY410,
}

#[cfg(target_os = "linux")]
fn linux_nvidia_planar444_active() -> bool {
    tether_codec::nvenc::nvidia_gpu_present()
}

#[cfg(not(target_os = "linux"))]
fn linux_nvidia_planar444_active() -> bool {
    false
}

pub(crate) fn render_layout_for(chroma: ChromaSubsampling, bit_depth: u8) -> RenderLayout {
    match (chroma, bit_depth) {
        // 8-bit: existing macOS-IOSurface / Linux-dma-buf split. macOS
        // gives biplanar NV12/NV24; Linux gives packed XYUV for 4:4:4
        // and biplanar NV12 for 4:2:0.
        (ChromaSubsampling::Yuv420, 8) => RenderLayout::Biplanar8,
        (ChromaSubsampling::Yuv444, 8) => {
            if cfg!(target_os = "macos") {
                RenderLayout::Biplanar8
            } else if linux_nvidia_planar444_active() {
                RenderLayout::Planar444
            } else {
                RenderLayout::PackedXYUV
            }
        }
        // 10-bit 4:2:0 (Main10): biplanar 16-bit on both platforms.
        // macOS gets `'P010'`, Linux gets `DRM_FORMAT_P010`; the
        // 10-in-16 MSB-align applies uniformly.
        (ChromaSubsampling::Yuv420, 10) => RenderLayout::Biplanar16,
        // 10-bit 4:4:4: same platform split as the 8-bit 4:4:4 case.
        // macOS decodes to biplanar `'P410'`/`'xf44'` (16-bit cells →
        // Biplanar16). Linux's VAAPI decoder exports packed Y410 (single
        // 10:10:10:2 plane → PackedY410). Mirrors the `Yuv444, 8` XYUV
        // split above. (Encoder-input XV30 and decode-output Y410 share the
        // same "X:Cr:Y:Cb" lane order — AV_PIX_FMT_XV30LE / Y410LE — so both
        // are conformant; they carry different fourccs but identical packing.)
        (ChromaSubsampling::Yuv444, 10) => {
            if cfg!(target_os = "macos") {
                RenderLayout::Biplanar16
            } else {
                RenderLayout::PackedY410
            }
        }
        // Anything we haven't enumerated (12-bit, future chroma like
        // Yuv422) reaches the renderer by construction — the
        // negotiator filters on PROFILE_PREFERENCE, and the encoder
        // probes return false for any combo no layer here can
        // deliver. If we *do* reach this arm, the upstream filter
        // is broken: defaulting to Biplanar16 silently would
        // mis-render an 8-bit Yuv422 stream as if it were 16-bit
        // (wrong texture formats, wrong sampler scale). Panic loud
        // — it's a programmer error, not a runtime condition.
        (chroma, bit_depth) => panic!(
            "render_layout_for: unhandled (chroma={chroma:?}, bit_depth={bit_depth}). \
             This combination must not reach the renderer — either the negotiator \
             filter is broken or this enum needs a new arm wired to the right \
             RenderLayout variant.",
        ),
    }
}

/// Numeric tag for the WGSL fragment shader's EOTF dispatch. Keep in
/// sync with the `bt709_eotf` / `srgb_eotf` switch in `shader.wgsl`.
/// Encoding it as a single integer (rather than separate uniforms or
/// pipeline variants) means the renderer never recompiles when a
/// future stream renegotiates the transfer; the shader picks the
/// right path per draw.
const TRANSFER_KIND_BT709: u32 = 0;
const TRANSFER_KIND_SRGB: u32 = 1;

fn transfer_kind_for(spec: VideoColorSpec) -> u32 {
    match spec.transfer {
        ColorTransfer::Bt709 => TRANSFER_KIND_BT709,
        ColorTransfer::Srgb => TRANSFER_KIND_SRGB,
        // PQ / HLG / Linear are reserved on the wire; the shader
        // doesn't implement them yet. Fall back to BT.709 with a
        // warn — the picture will look slightly off but won't be
        // unwatchable. Promoting these to a hard error is what the
        // HDR work should do once the surface format / tone-map
        // chain is in place.
        ColorTransfer::Pq | ColorTransfer::Hlg | ColorTransfer::Linear => {
            tracing::warn!(
                side = "gpu",
                ?spec.transfer,
                "EOTF not yet implemented; falling back to BT.709"
            );
            TRANSFER_KIND_BT709
        }
    }
}

/// Whether the renderer can allocate the `R16Unorm` Y + `Rg16Unorm` UV
/// textures used by the 10-bit biplanar (`Biplanar16`) layout.
///
/// Probes the same adapter the renderer would pick on this host
/// (`HighPerformance`, no surface tied) and reports whether
/// `wgpu::Features::TEXTURE_FORMAT_16BIT_NORM` is advertised. Run this
/// at handshake time so the client's decode-capability advert can
/// filter out 10-bit profiles on adapters that wouldn't be able to
/// render the frames anyway (lavapipe, SwiftShader, very old mobile
/// GPUs).
///
/// On adapter-init failure this returns `false` — better to under-
/// advertise and have negotiation pick a working 8-bit profile than to
/// claim 10-bit support and panic on the first frame allocation. The
/// renderer's own `GpuState::new` will surface a real error at session
/// start if no adapter can be found at all.
///
/// Note that this probe and `GpuState::new`'s feature check run
/// independent `request_adapter` calls. On multi-GPU hosts (laptop
/// igpu + dgpu) those two calls *could* in principle pick different
/// adapters with different feature sets — the probe says yes, the
/// real renderer picks a different adapter and lacks the feature.
/// `GpuState::new` remains the authoritative gate (it opts the
/// feature in iff the renderer's own adapter advertises it), so the
/// worst case is a 10-bit session that builds the decoder, fails at
/// `make_yuv_textures`, and panics — same failure mode as before
/// the probe existed, just narrower. Closing the gap fully would
/// mean sharing one adapter handle between this probe and
/// `GpuState::new`, which is more plumbing than the residual risk
/// justifies today.
fn probe_10bit_render() -> bool {
    let instance = wgpu::Instance::default();
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    };
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&options)) else {
        return false;
    };
    adapter
        .features()
        .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
}

pub async fn supports_10bit_render() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(probe_10bit_render)
}

/// Whether this backend can render a negotiated video profile.
///
/// The check is layout-specific: 10-bit biplanar layouts need
/// `TEXTURE_FORMAT_16BIT_NORM`, while Linux's packed 10-bit 4:4:4 path imports
/// one `Rgb10a2Unorm` texture and must not be rejected by the biplanar gate.
pub async fn supports_video_profile_render(profile: VideoProfile) -> bool {
    match render_layout_for(profile.chroma, profile.bit_depth) {
        RenderLayout::Biplanar8 | RenderLayout::Planar444 | RenderLayout::PackedXYUV => true,
        RenderLayout::Biplanar16 => supports_10bit_render().await,
        RenderLayout::PackedY410 => true,
    }
}

impl GpuState {
    pub(crate) async fn new(
        window: Arc<Window>,
        color_space: VideoColorSpec,
        chroma: ChromaSubsampling,
        bit_depth: u8,
        cursor_channel: crate::cursor_overlay::CursorChannel,
    ) -> Result<Self> {
        let size = window.inner_size();
        let (width, height) = (size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window)?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                // Added on wgpu trunk past 29.x; opting out keeps the
                // pre-trunk per-backend limit behaviour. Revisit when we
                // bump to wgpu 30.
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| RenderError::NoAdapter)?;

        // Hard-require DMA-BUF import on Linux. The decoder side
        // (tether-codec's probe) hard-requires VAAPI and the decoded
        // frames arrive as `Frame::Gpu` carrying dma-buf fds; the
        // renderer needs the matching Vulkan extension to consume them.
        // If the adapter doesn't advertise it (lavapipe, very old
        // Mesa, missing VK_EXT_image_drm_format_modifier), fail loudly
        // here rather than silently dropping every frame later.
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        let info = adapter.get_info();
        let adapter_features = adapter.features();
        #[cfg(target_os = "linux")]
        if !adapter_features.contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF) {
            return Err(RenderError::DmaBufImport(format!(
                "wgpu adapter '{}' (driver: '{}', backend: {:?}) does not advertise \
                 VULKAN_EXTERNAL_MEMORY_DMA_BUF. Check that the system Vulkan ICD \
                 is a real GPU driver (Mesa 24+ for Intel/AMD) and exposes \
                 VK_EXT_external_memory_dma_buf + VK_EXT_image_drm_format_modifier; \
                 lavapipe and similar software stacks do not. Tether requires \
                 zero-copy decode — there is no CPU-upload fallback.",
                info.name, info.driver, info.backend
            )));
        }
        let mut required = wgpu::Features::empty();
        if adapter_features.contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF) {
            required |= wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF;
        }
        // 10-bit biplanar (`Biplanar16`) layouts allocate `R16Unorm` Y +
        // `Rg16Unorm` UV textures (see `make_yuv_textures`). Both are
        // gated behind the `TEXTURE_FORMAT_16BIT_NORM` feature in wgpu,
        // which is widely supported on real hardware (every Apple GPU
        // since A11, every desktop Vulkan ICD on Mesa 24+ / NVIDIA /
        // AMD) but is *not* enabled by default — creating these
        // textures without the opt-in raises a Validation Error and
        // the wgpu default error handler then panics. Opting in is
        // safe on any adapter that advertises it; on adapters that
        // don't (lavapipe / SwiftShader / very old mobile), the
        // negotiator would still pick a 10-bit profile and we'd
        // panic at `make_yuv_textures` on first frame allocation.
        // The client filters 10-bit profiles out of its decode-capability
        // advert when this feature is absent (`supports_10bit_render`),
        // so by the time we reach here, negotiation has already picked
        // an 8-bit profile if we're on such an adapter — the opt-in
        // below is harmless either way.
        if adapter_features.contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM) {
            required |= wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tether-render device"),
                required_features: required,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await?;
        #[cfg(target_os = "linux")]
        let dmabuf_import_supported = device
            .features()
            .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF);
        // Probe the Metal HAL once at startup. If the wgpu device wasn't
        // built on the Metal backend (could happen if a future config
        // forces Vulkan via MoltenVK on macOS) we want a clean failure
        // here, not a panic deep in `import_iosurface_textures` on the
        // first decoded frame.
        #[cfg(target_os = "macos")]
        let metal_import_supported = unsafe {
            // SAFETY: as_hal is only unsafe because the returned guard
            // must not outlive the device; we drop the result immediately.
            device.as_hal::<wgpu::hal::api::Metal>().is_some()
        };
        #[cfg(target_os = "macos")]
        if !metal_import_supported {
            return Err(RenderError::DmaBufImport(format!(
                "wgpu adapter '{}' (driver: '{}', backend: {:?}) is not Metal-backed; \
                 macOS IOSurface import requires the Metal HAL. Tether requires \
                 zero-copy decode — there is no CPU-upload fallback.",
                info.name, info.driver, info.backend
            )));
        }
        #[cfg(target_os = "linux")]
        tracing::info!(
            adapter = info.name,
            driver = info.driver,
            backend = ?info.backend,
            dmabuf_import_supported,
            "wgpu device initialised"
        );
        #[cfg(target_os = "macos")]
        tracing::info!(
            adapter = info.name,
            driver = info.driver,
            backend = ?info.backend,
            metal_import_supported,
            "wgpu device initialised"
        );

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        // Present-mode preference is driven by the
        // `crate::present_policy::PresentPolicy` enum. Default is
        // `LowLatency` (Immediate > Mailbox > Fifo), matching the
        // pre-policy behavior. The `Smooth` profile flips to
        // Fifo-first for users who want vsync-locked presentation.
        // The enum's `preference()` returns the ordered list; we
        // pick the first one the surface advertises and fall back
        // to Fifo (universal) if none match (impossible given
        // wgpu's contract that Fifo is always supported, but
        // tracked defensively).
        let policy = crate::present_policy::PresentPolicy::default();
        let present_mode = policy
            .preference()
            .into_iter()
            .find(|m| surface_caps.present_modes.contains(m))
            .unwrap_or(wgpu::PresentMode::Fifo);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &surface_config);

        Self::build_state(
            device,
            queue,
            RenderTarget::Swapchain {
                surface,
                config: surface_config,
            },
            format,
            color_space,
            chroma,
            bit_depth,
            cursor_channel,
            #[cfg(target_os = "linux")]
            dmabuf_import_supported,
            #[cfg(target_os = "macos")]
            metal_import_supported,
        )
    }

    /// Headless constructor for the round-trip test harness. Skips
    /// instance/adapter/surface creation — caller passes a pre-built
    /// device + queue (typically shared with the gpuconvert bridge, so
    /// the test runs on the same device the production hot path
    /// would). Allocates a `Bgra8Unorm` offscreen target with
    /// `RENDER_ATTACHMENT | COPY_SRC` at `target_dims`; the harness
    /// reads it back after `render()`. The blit color-attachment
    /// format matches the typical production swapchain
    /// (`Bgra8UnormSrgb`) so the render-pipeline cache key and
    /// fragment-write code path are bit-identical to the live path.
    #[allow(dead_code)] // consumed by tether-render's test_harness module
    pub(crate) fn new_headless(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_dims: (u32, u32),
        color_space: VideoColorSpec,
        chroma: ChromaSubsampling,
        bit_depth: u8,
    ) -> Result<Self> {
        // Headless test harness has no wire-side producer of cursor
        // events; instantiate a detached `CursorChannel` so the pass
        // exists but never draws.
        let cursor_channel = crate::cursor_overlay::CursorChannel::new();
        // Match the production swapchain format: production picks an
        // sRGB-encoded BGRA format from `surface_caps.formats` via
        // `is_srgb()`. The blit pipeline writes linear values to this
        // target and wgpu applies the sRGB OETF on store, so the
        // readback bytes are sRGB-encoded the same way the swapchain
        // bytes would be. Diverging from this (e.g. `Bgra8Unorm`
        // non-sRGB) would test a sibling of the production chain, not
        // production itself.
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        // Mirror production: the device must advertise the same
        // platform-specific zero-copy import capability the live `new`
        // path enforces — otherwise the first `apply_gpu` call hits a
        // `debug_assert!` and the test panics with a confusing
        // message. Caller passes a device they got from
        // `try_init_wgpu_for_dmabuf` (or the macOS equivalent), so a
        // hard error here means the test's adapter setup is broken,
        // not that the test environment is degraded.
        #[cfg(target_os = "linux")]
        let dmabuf_import_supported = {
            let features = device.features();
            if !features.contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF) {
                return Err(RenderError::DmaBufImport(
                    "new_headless: device does not advertise \
                     VULKAN_EXTERNAL_MEMORY_DMA_BUF. The harness must pass a device \
                     obtained from try_init_wgpu_for_dmabuf with the dma-buf feature \
                     enabled."
                        .into(),
                ));
            }
            true
        };
        #[cfg(target_os = "macos")]
        let metal_import_supported = {
            // SAFETY: as_hal's returned guard is dropped immediately.
            let supported = unsafe { device.as_hal::<wgpu::hal::api::Metal>().is_some() };
            if !supported {
                return Err(RenderError::DmaBufImport(
                    "new_headless: device is not Metal-backed; macOS IOSurface \
                     import requires the Metal HAL."
                        .into(),
                ));
            }
            true
        };
        let target_tex = make_offscreen_target(&device, format, target_dims);
        Self::build_state(
            device,
            queue,
            RenderTarget::Offscreen {
                target: target_tex,
                dims: target_dims,
                format,
            },
            format,
            color_space,
            chroma,
            bit_depth,
            cursor_channel,
            #[cfg(target_os = "linux")]
            dmabuf_import_supported,
            #[cfg(target_os = "macos")]
            metal_import_supported,
        )
    }

    /// Pipeline + upscale-stage setup shared between [`new`] and
    /// [`new_headless`]. Takes a pre-built `target` and the resolved
    /// final color-attachment `format`; the swapchain vs offscreen
    /// distinction is invisible from here down.
    #[allow(clippy::too_many_arguments)]
    fn build_state(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target: RenderTarget,
        format: wgpu::TextureFormat,
        color_space: VideoColorSpec,
        chroma: ChromaSubsampling,
        bit_depth: u8,
        cursor_channel: crate::cursor_overlay::CursorChannel,
        #[cfg(target_os = "linux")] dmabuf_import_supported: bool,
        #[cfg(target_os = "macos")] metal_import_supported: bool,
    ) -> Result<Self> {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tether-render sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Bind-group layout depends on the renderer's plane shape, not
        // directly on chroma — see `RenderLayout` for why. Biplanar8
        // and Biplanar16 share a single bgl (R8/R16 + Rg8/Rg16 are all
        // filterable-float-sampleable, so the bgl can stay format-blind);
        // PackedXYUV has one Rgba8 texture + sampler at binding 1.
        let layout = render_layout_for(chroma, bit_depth);
        let yuv_bgl =
            match layout {
                RenderLayout::Biplanar8 | RenderLayout::Biplanar16 => device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("tether-render yuv bgl (biplanar)"),
                        entries: &[
                            bgl_texture_entry(0),
                            bgl_texture_entry(1),
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                                count: None,
                            },
                        ],
                    }),
                RenderLayout::Planar444 => {
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("tether-render yuv bgl (444 planar)"),
                        entries: &[
                            bgl_texture_entry(0),
                            bgl_texture_entry(1),
                            bgl_texture_entry(2),
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                                count: None,
                            },
                        ],
                    })
                }
                RenderLayout::PackedXYUV | RenderLayout::PackedY410 => device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("tether-render yuv bgl (444 packed)"),
                        entries: &[
                            bgl_texture_entry(0),
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                                count: None,
                            },
                        ],
                    }),
            };

        let scale_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tether-render scale bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let scale_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tether-render scale uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Seed identity so the first draw before any frame upload
        // doesn't read undefined memory off the GPU.
        let identity_scale: [f32; 4] = [1.0, 1.0, 0.0, 0.0];
        queue.write_buffer(&scale_buffer, 0, &bytes_of_f32x4(&identity_scale));
        let scale_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tether-render scale bind group"),
            layout: &scale_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scale_buffer.as_entire_binding(),
            }],
        });

        // Color-params bind group: a `vec4<u32>` whose .x is the EOTF
        // dispatch tag the WGSL fragment shader switches on. The
        // other three slots are reserved for future axes (matrix
        // kind, range kind) when those gain shader variants — today
        // only the transfer is variable. (WGSL's std140 layout pads
        // a bare `u32` in a uniform block to 16 bytes anyway;
        // `vec4<u32>` makes that explicit and the reserved slots
        // self-documenting.)
        let color_params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tether-render color-params bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let color_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tether-render color-params uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transfer_kind = transfer_kind_for(color_space);
        let range_kind = range_kind_for(bit_depth, layout);
        let color_params: [u32; 4] = [transfer_kind, range_kind, 0, 0];
        queue.write_buffer(&color_params_buffer, 0, &bytes_of_u32x4(&color_params));
        tracing::info!(
            matrix = ?color_space.matrix,
            range = ?color_space.range,
            transfer = ?color_space.transfer,
            primaries = ?color_space.primaries,
            transfer_kind,
            bit_depth,
            range_kind,
            "renderer color spec applied"
        );
        let color_params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tether-render color-params bind group"),
            layout: &color_params_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: color_params_buffer.as_entire_binding(),
            }],
        });

        // Shader matches the bind-group layout. Biplanar (8 or 16-bit)
        // uses the chroma-resolution-agnostic Y+UV biplanar shader —
        // works for NV12, NV24, P010, P410 alike since bilinear
        // sampling is the same math at any UV resolution and the
        // 10-in-16 MSB-align is normalised by `color_params.y`.
        // PackedXYUV uses a separate shader that pulls Y/U/V from the
        // channels of one Rgba8 sample.
        let shader_src = match layout {
            RenderLayout::Biplanar8 | RenderLayout::Biplanar16 => {
                include_str!("../shader.wgsl")
            }
            RenderLayout::Planar444 => include_str!("../shader_yuv444p.wgsl"),
            RenderLayout::PackedXYUV => include_str!("../shader_yuv444.wgsl"),
            RenderLayout::PackedY410 => include_str!("../shader_yuv444_10bit.wgsl"),
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(match layout {
                RenderLayout::Biplanar8 => "tether-render shader (biplanar 8)",
                RenderLayout::Biplanar16 => "tether-render shader (biplanar 16)",
                RenderLayout::Planar444 => "tether-render shader (444 planar)",
                RenderLayout::PackedXYUV => "tether-render shader (444 packed)",
                RenderLayout::PackedY410 => "tether-render shader (444 packed 10-bit)",
            }),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tether-render pipeline layout"),
            bind_group_layouts: &[Some(&yuv_bgl), Some(&scale_bgl), Some(&color_params_bgl)],
            immediate_size: 0,
        });

        // Both shader files name their fragment entry point `fs`; the
        // module selected above is what determines which fragment body
        // gets compiled into this pipeline.
        let fragment_entry = "fs";
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tether-render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fragment_entry),
                // The YUV→RGB pass now writes to an Rgba16Float
                // intermediate, not directly to the swapchain. The
                // shader emits linear-light values; Rgba16Float
                // stores them verbatim (no OETF), so the Mitchell
                // upscale stage (or the direct blit when no upscale)
                // consumes them in linear light. The swapchain blit
                // pipeline (built below) targets the surface format
                // and lets wgpu apply the sRGB OETF on write, same
                // as the pre-refactor single-pass behaviour.
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let textures = make_yuv_textures(&device, &yuv_bgl, &sampler, chroma, bit_depth, 1, 1);

        // === UpscaleStage construction ===
        //
        // The blit pipeline samples an Rgba16Float texture (either the
        // YUV→RGB intermediate at video dims, or the Mitchell scaler's
        // output at letterbox-fit window dims) and writes to the
        // swapchain. The sampler is `Linear` magnification — when
        // Mitchell didn't run (window ≤ video, or scaler-disabled
        // path) this gives the previous-behaviour bilinear blit; when
        // Mitchell did run, the sampler operates on already-Mitchell-
        // filtered values and the bilinear here is a marginal smooth.
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tether-render blit sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tether-render blit bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tether-render blit shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../blit.wgsl").into()),
        });
        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tether-render blit pipeline layout"),
            bind_group_layouts: &[Some(&blit_bgl), Some(&scale_bgl)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tether-render blit pipeline"),
            layout: Some(&blit_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Identity-scale uniform for pass 1 (YUV → intermediate).
        // The YUV shader's vertex applies `pos * scale.xy` for
        // letterbox; in the multi-pass model we always want full
        // coverage in pass 1 and letterbox only in pass 3 (blit),
        // so we use a dedicated `(1,1,0,0)` buffer that never
        // changes.
        let identity_scale_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tether-render identity-scale uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &identity_scale_buffer,
            0,
            &bytes_of_f32x4(&[1.0, 1.0, 0.0, 0.0]),
        );
        let identity_scale_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tether-render identity-scale bind group"),
            layout: &scale_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: identity_scale_buffer.as_entire_binding(),
            }],
        });

        let upscale = UpscaleStage {
            rgb_intermediate: None,
            intermediate_dims: (0, 0),
            scaler_pipelines: None,
            scaler: None,
            scaler_failed_for: None,
            blit_pipeline,
            blit_bgl,
            blit_sampler,
            identity_scale_buffer,
            identity_scale_bind_group,
        };

        let cursor_overlay = crate::cursor_overlay::CursorOverlay::new(&device, format);

        Ok(Self {
            target,
            device,
            queue,
            pipeline,
            yuv_bgl,
            chroma,
            bit_depth,
            sampler,
            textures,
            retired: None,
            scale_buffer,
            scale_bind_group,
            color_params_buffer,
            color_params_bind_group,
            #[cfg(target_os = "linux")]
            dmabuf_import_supported,
            #[cfg(target_os = "macos")]
            metal_import_supported,
            upscale,
            cursor_overlay,
            cursor_channel,
        })
    }

    /// Current video texture size and surface size. Returned together
    /// because the cursor-normalisation math in `lib.rs` needs both.
    pub(crate) fn dimensions(&self) -> ((u32, u32), (u32, u32)) {
        (self.textures.size, self.target.dimensions())
    }

    /// Borrow the offscreen target so the round-trip harness can copy
    /// it into a readback buffer. Panics on the swapchain variant —
    /// production never reads back through this accessor.
    #[allow(dead_code)] // consumed by tether-render's test_harness module
    pub(crate) fn offscreen_target_for_readback(&self) -> &wgpu::Texture {
        match &self.target {
            RenderTarget::Offscreen { target, .. } => target,
            RenderTarget::Swapchain { .. } => {
                panic!("offscreen_target_for_readback called on swapchain target")
            }
        }
    }

    /// Borrow the queue so the harness can submit its readback copy on
    /// the same queue the production renderer just submitted the blit
    /// pass on (so the readback orders-after the render).
    #[allow(dead_code)] // consumed by tether-render's test_harness module
    pub(crate) fn queue_for_readback(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.target.resize(&self.device, width, height);
        // The video textures are at decoded resolution and don't change
        // on window resize. `render_to_view` recomputes the letterbox
        // scale uniform every frame, so the next redraw automatically
        // fits the new window without re-uploading or re-importing.
    }

    /// Take ownership of a new frame and either upload it (Cpu) or
    /// import it via DMA-BUF (Gpu). The previous frame's textures
    /// (and, for Gpu, its decoder-side guard) drop here, releasing
    /// any VAAPI surface they kept alive.
    pub(crate) fn apply_frame(&mut self, frame: Frame) -> Result<()> {
        match frame {
            Frame::Cpu(cpu) => self.apply_cpu(cpu),
            Frame::Gpu(gpu) => self.apply_gpu(gpu),
        }
    }

    fn apply_cpu(&mut self, frame: CpuFrame) -> Result<()> {
        if frame.width == 0 || frame.height == 0 {
            return Ok(());
        }
        // CPU upload path is NV12 8-bit only today — the only CPU
        // producer (sw-decoded fallback in dmabuf_test) emits NV12. A
        // YUV444 or 10-bit CPU producer would need a CpuFrame variant
        // change, not a patch here.
        if self.chroma != ChromaSubsampling::Yuv420 || self.bit_depth != 8 {
            return Err(crate::RenderError::DmaBufImport(format!(
                "CPU upload path is NV12 8-bit only; session negotiated {:?} {}-bit",
                self.chroma, self.bit_depth,
            )));
        }
        // Two reasons to rebuild: the previous frame was DMA-BUF-imported
        // (those textures aren't COPY_DST so write_texture would fail —
        // checking `_guard.is_some()` is load-bearing; the size match
        // would silently skip the rebuild otherwise), or the resolution
        // changed.
        if self.textures._guard.is_some() || (frame.width, frame.height) != self.textures.size {
            let fresh = make_yuv_textures(
                &self.device,
                &self.yuv_bgl,
                &self.sampler,
                self.chroma,
                self.bit_depth,
                frame.width,
                frame.height,
            );
            self.retire_textures(fresh);
        }
        let (chroma_w, chroma_h) = frame.chroma_dims();
        let YuvPlanes::Biplanar8 { y, uv } = &self.textures.planes else {
            unreachable!("guarded by chroma + bit_depth check above");
        };
        // Y is R8 — one byte per texel, so bytes_per_row == width.
        // UV is Rg8 — two bytes per texel, so bytes_per_row == chroma_w * 2.
        write_plane_r8(&self.queue, y, &frame.y, frame.width, frame.height);
        write_plane_rg8(&self.queue, uv, &frame.uv, chroma_w, chroma_h);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn apply_gpu(&mut self, frame: GpuFrame) -> Result<()> {
        // `GpuState::new` hard-errors when the dma-buf feature isn't
        // available on Linux, so reaching this branch with the flag
        // false would mean the constructor's check is broken — not
        // a runtime condition the caller can recover from.
        debug_assert!(
            self.dmabuf_import_supported,
            "apply_gpu reached without dma-buf feature; GpuState::new should have failed"
        );
        // Use an explicit match (not `let else`) so adding a future
        // GpuFrameSource variant (NVDEC over CUDA-Vulkan interop, a
        // VideoToolbox CVPixelBuffer, etc.) is a compile error here
        // rather than a silent fallthrough.
        let GpuFrameSource::DmaBuf(dmabuf) = frame.source;
        let fresh = import_dmabuf_textures(
            &self.device,
            &self.yuv_bgl,
            &self.sampler,
            self.chroma,
            self.bit_depth,
            &dmabuf,
            frame.width,
            frame.height,
            frame.guard,
        )?;
        self.retire_textures(fresh);
        Ok(())
    }

    /// Swap in `fresh` as the active textures and move the previous
    /// set into the retired slot. Anything already in the retired
    /// slot is dropped — that frame's GPU submission has had at least
    /// one `render()` cycle's worth of opportunity to complete on the
    /// GPU, so its dma-buf is safe to unmap and its VAAPI surface is
    /// safe to return to the pool.
    fn retire_textures(&mut self, fresh: YuvTextures) {
        let previous = std::mem::replace(&mut self.textures, fresh);
        self.retired = Some(previous);
    }

    #[cfg(target_os = "macos")]
    fn apply_gpu(&mut self, frame: GpuFrame) -> Result<()> {
        // `GpuState::new` hard-errors when the Metal HAL isn't
        // available on macOS, so reaching this branch with the flag
        // false would mean the constructor's check is broken — not
        // a runtime condition the caller can recover from. Mirrors
        // the dma-buf invariant on the Linux side.
        debug_assert!(
            self.metal_import_supported,
            "apply_gpu reached without Metal HAL; GpuState::new should have failed"
        );
        // Match shape mirrors the Linux side (where `GpuFrameSource` has a
        // second variant, so the match is fallible): an explicit pattern, not
        // `let else`, so adding a future variant surfaces here at compile time.
        // Infallible on macOS today, hence the scoped allow.
        #[allow(clippy::infallible_destructuring_match)]
        let iosurface = match &frame.source {
            GpuFrameSource::IOSurface(s) => s,
        };
        let fresh = import_iosurface_textures(
            &self.device,
            &self.yuv_bgl,
            &self.sampler,
            self.chroma,
            self.bit_depth,
            iosurface,
            frame.guard,
        )?;
        self.retire_textures(fresh);
        Ok(())
    }

    // Windows renders natively in D3D11 (`crate::d3d11`), not through this
    // wgpu backend, so there is no Windows `apply_gpu` here.
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn apply_gpu(&mut self, _frame: GpuFrame) -> Result<()> {
        unreachable!("GpuFrame cannot be constructed on this platform")
    }

    pub(crate) fn render(&mut self) -> std::result::Result<(), String> {
        match &mut self.target {
            RenderTarget::Swapchain { .. } => self.render_swapchain(),
            RenderTarget::Offscreen { .. } => self.render_offscreen(),
        }
    }

    /// Swapchain present path. Acquires a backbuffer, calls
    /// [`render_to_view`], presents.
    fn render_swapchain(&mut self) -> std::result::Result<(), String> {
        use wgpu::CurrentSurfaceTexture::*;
        // Handle all seven variants from wgpu 29 deliberately. Defaults
        // matter: Outdated/Lost without reconfigure leaves the window
        // permanently black; Occluded without silencing spams logs while
        // the window is minimised; Suboptimal still has a valid texture
        // that should be presented for this frame.
        let RenderTarget::Swapchain { surface, config } = &self.target else {
            unreachable!("render_swapchain dispatched on non-swapchain target");
        };
        let (output, reconfigure_after) = match surface.get_current_texture() {
            Success(f) => (f, false),
            Suboptimal(f) => {
                tracing::debug!("wgpu surface suboptimal; reconfigure after present");
                (f, true)
            }
            Outdated => {
                tracing::debug!("wgpu surface outdated; reconfiguring");
                surface.configure(&self.device, config);
                return Ok(());
            }
            Lost => {
                tracing::warn!("wgpu surface lost; attempting best-effort reconfigure");
                surface.configure(&self.device, config);
                return Ok(());
            }
            Timeout | Occluded => {
                return Ok(());
            }
            Validation => {
                return Err("wgpu surface validation error".into());
            }
        };
        let target_dims = (config.width, config.height);
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.render_to_view(&view, target_dims)?;
        output.present();

        if reconfigure_after {
            let RenderTarget::Swapchain { surface, config } = &self.target else {
                unreachable!();
            };
            surface.configure(&self.device, config);
        }
        Ok(())
    }

    /// Headless render path. Builds a view of the offscreen target and
    /// calls [`render_to_view`] — no surface acquire, no `present`. The
    /// round-trip harness reads the target back after this returns.
    fn render_offscreen(&mut self) -> std::result::Result<(), String> {
        // Materialise `view` + `target_dims` in their own scope so the
        // borrow of `self.target` ends before `render_to_view` takes
        // `&mut self`. The view holds an internal Arc on the texture,
        // so dropping the destructure binding is fine — the GPU
        // resource stays alive for the render. Mirrors the swapchain
        // pattern (acquire output, drop the &self.target borrow, then
        // call render_to_view).
        let (view, target_dims) = {
            let RenderTarget::Offscreen { target, dims, .. } = &self.target else {
                unreachable!("render_offscreen dispatched on non-offscreen target");
            };
            (
                target.create_view(&wgpu::TextureViewDescriptor::default()),
                *dims,
            )
        };
        self.render_to_view(&view, target_dims)
    }

    /// Multi-pass renderer body — production code path. Same function
    /// drives both [`render_swapchain`] (present) and [`render_offscreen`]
    /// (test readback) so any bug in the multi-pass chain shows up in
    /// hardware tests.
    fn render_to_view(
        &mut self,
        target_view: &wgpu::TextureView,
        target_dims: (u32, u32),
    ) -> std::result::Result<(), String> {
        let video_dims = self.textures.size;
        let surface_dims = target_dims;

        // Make sure the intermediate exists at the right dims. If
        // the decoded video changed size between frames, also rebuild
        // the Mitchell scaler and clear the sticky failure flag (a
        // device reset accompanying the resize can become valid).
        let video_changed = self.upscale.intermediate_dims != video_dims;
        if self.upscale.rgb_intermediate.is_none() || video_changed {
            self.upscale.rgb_intermediate = Some(make_rgb_intermediate(&self.device, video_dims));
            self.upscale.intermediate_dims = video_dims;
            self.upscale.scaler = None;
            self.upscale.scaler_failed_for = None;
        }
        // Decide if we need to (re)build the Mitchell scaler:
        // upscale only, never for window ≤ video. Skipping the
        // scaler then is the previous-behaviour single-pass bilinear
        // (now spread across two passes — YUV→intermediate then
        // sampled blit).
        let need_upscale = surface_dims.0 > video_dims.0 || surface_dims.1 > video_dims.1;
        let upscale_dims = if need_upscale {
            letterbox_fit_dims(video_dims, surface_dims)
        } else {
            video_dims
        };
        let scaler_needs_rebuild = need_upscale
            && self.upscale.scaler_failed_for != Some(upscale_dims)
            && self
                .upscale
                .scaler
                .as_ref()
                .is_none_or(|s| s.dst_dims() != upscale_dims || s.src_dims() != video_dims);
        if scaler_needs_rebuild {
            match build_upscale_scaler(
                &self.device,
                &self.queue,
                &mut self.upscale.scaler_pipelines,
                video_dims,
                upscale_dims,
            ) {
                Ok(Some(s)) => self.upscale.scaler = Some(s),
                Ok(None) => self.upscale.scaler = None,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        src = ?video_dims, dst = ?upscale_dims,
                        "Mitchell upscale scaler construction failed; falling back to bilinear blit for this (video, window) pair"
                    );
                    self.upscale.scaler = None;
                    self.upscale.scaler_failed_for = Some(upscale_dims);
                }
            }
        } else if !need_upscale {
            self.upscale.scaler = None;
        }

        // Letterbox scale for the blit pass — computed from the
        // *blit source's* dims, not the original video dims. When
        // Mitchell ran, the source is its output at `letterbox_fit_dims`
        // (already aspect-matched); the blit scale is then (1, 1) or
        // very close. When Mitchell didn't run, the source is the
        // intermediate at video dims and the blit scale does the
        // aspect correction. Using the source's actual dims here
        // avoids sub-pixel aspect drift on unusual ratios.
        let blit_source_dims = match self.upscale.scaler.as_ref() {
            Some(s) => s.dst_dims(),
            None => video_dims,
        };
        let (sx, sy) = letterbox_scale(blit_source_dims, surface_dims);
        self.queue
            .write_buffer(&self.scale_buffer, 0, &bytes_of_f32x4(&[sx, sy, 0.0, 0.0]));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tether-render encoder"),
            });

        // Pass 1: YUV → linear-light RGB into the intermediate. The
        // YUV shader is unchanged; only the render target and scale
        // uniform differ from the pre-refactor single-pass form.
        let intermediate = self
            .upscale
            .rgb_intermediate
            .as_ref()
            .expect("intermediate built above");
        let inter_view = intermediate.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tether-render YUV->RGB pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &inter_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.textures.bind_group, &[]);
            pass.set_bind_group(1, &self.upscale.identity_scale_bind_group, &[]);
            pass.set_bind_group(2, &self.color_params_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }

        // Pass 2 (optional): Mitchell upscale → scaler.output().
        // Skipped when window ≤ video; the blit pass below then
        // samples the intermediate directly.
        if let Some(scaler) = self.upscale.scaler.as_ref() {
            // Submit pass 1 first so the scaler's read of the
            // intermediate sees the YUV pass's write. The Mitchell
            // scaler's scale() submits internally too.
            self.queue.submit(std::iter::once(encoder.finish()));
            if let Err(e) = scaler.scale(intermediate) {
                tracing::warn!(error = %e, "Mitchell upscale failed; falling back to direct blit");
                self.upscale.scaler = None;
            }
            encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tether-render blit encoder"),
                });
        }

        // Pass 3: blit (scaler output or intermediate) → swapchain
        // with letterbox.
        let blit_source = match self.upscale.scaler.as_ref() {
            Some(s) => s.output(),
            None => intermediate,
        };
        let blit_source_view = blit_source.create_view(&wgpu::TextureViewDescriptor::default());
        let blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tether-render blit bind group"),
            layout: &self.upscale.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blit_source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.upscale.blit_sampler),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tether-render blit pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.upscale.blit_pipeline);
            pass.set_bind_group(0, &blit_bg, &[]);
            pass.set_bind_group(1, &self.scale_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }

        // Pass 4: cursor overlay on top of the blit output. The pass
        // is a no-op when no sprite is active, the cursor is hidden,
        // or relative-locked mode is on — `CursorOverlay::render`
        // bails before touching the encoder in those cases. The
        // letterbox-fit rect inside the window is computed from the
        // same `(sx, sy)` the blit pass used so the sprite lands in
        // the exact pixel rect covered by the video.
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation
        )]
        let fit_dims = (
            ((surface_dims.0 as f32) * sx).round() as u32,
            ((surface_dims.1 as f32) * sy).round() as u32,
        );
        self.cursor_overlay.render(
            &mut encoder,
            &self.device,
            &self.queue,
            target_view,
            &self.cursor_channel,
            video_dims,
            surface_dims,
            fit_dims,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Drop the previously-retired textures after the *next* submit
        // completes the previous frame's read. Conservative: this drops
        // one cycle after the texture stopped being bound, not one
        // cycle after the GPU finished reading from it. Tightening to
        // a real signal (queue submission index, fence) is the kind of
        // optimisation worth doing only if profiling shows the extra
        // dma-buf memory held across one frame matters.
        self.retired = None;

        Ok(())
    }
}

/// Build the offscreen render target used by [`GpuState::new_headless`].
/// `RENDER_ATTACHMENT` for the blit pass to write to; `COPY_SRC` so the
/// harness can read it back into a CPU buffer for metric comparison.
fn make_offscreen_target(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    dims: (u32, u32),
) -> wgpu::Texture {
    let (w, h) = (dims.0.max(1), dims.1.max(1));
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tether-render offscreen target"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Build the Rgba16Float intermediate that the YUV→RGB pass renders
/// into. `RENDER_ATTACHMENT` for pass 1 writes, `TEXTURE_BINDING` so
/// pass 2 (Mitchell) or pass 3 (direct blit) can sample it.
fn make_rgb_intermediate(device: &wgpu::Device, dims: (u32, u32)) -> wgpu::Texture {
    let (w, h) = (dims.0.max(1), dims.1.max(1));
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tether-render rgb intermediate"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

/// Compute the *target* dimensions for the Mitchell upscale: the
/// letterbox-fit rect inside the window. The Mitchell pass produces
/// at this size and the blit pass stretches a centered quad at the
/// same aspect ratio — the letterbox bars become the swapchain's
/// cleared-black margin.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
fn letterbox_fit_dims(video: (u32, u32), window: (u32, u32)) -> (u32, u32) {
    if video.0 == 0 || video.1 == 0 || window.0 == 0 || window.1 == 0 {
        return video;
    }
    let video_aspect = video.0 as f32 / video.1 as f32;
    let window_aspect = window.0 as f32 / window.1 as f32;
    if video_aspect > window_aspect {
        // Width-bound: fit to window width, derive height.
        let w = window.0;
        let h = (w as f32 / video_aspect).round().max(1.0) as u32;
        (w, h)
    } else {
        let h = window.1;
        let w = (h as f32 * video_aspect).round().max(1.0) as u32;
        (w, h)
    }
}

/// Build (or lazily compile) the Mitchell scaler pipelines and a
/// scaler instance for the given dims.
///
/// Returns:
/// - `Ok(Some(scaler))` — normal upscale case.
/// - `Ok(None)` — `src_dims == dst_dims`; no scaler needed, blit
///   samples the intermediate directly.
/// - `Err(_)` — construction genuinely failed. Caller logs once and
///   sets a sticky failure flag to avoid per-frame retries.
fn build_upscale_scaler(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cached_pipelines: &mut Option<Arc<tether_scaler::Pipelines>>,
    src_dims: (u32, u32),
    dst_dims: (u32, u32),
) -> std::result::Result<Option<tether_scaler::Scaler>, tether_scaler::ScalerError> {
    if src_dims == dst_dims {
        return Ok(None);
    }
    let pipelines = cached_pipelines
        .get_or_insert_with(|| Arc::new(tether_scaler::Pipelines::build(device)))
        .clone();
    tether_scaler::Scaler::new_with_color_space(
        pipelines,
        device.clone(),
        queue.clone(),
        src_dims,
        dst_dims,
        tether_scaler::ColorSpace::LinearF16,
    )
    .map(Some)
}

/// Compute the (x, y) NDC scale factors that letterbox / pillarbox the
/// source texture inside the surface while preserving its aspect ratio.
/// Returns (1.0, 1.0) for matching aspect ratios. The unused axis gets
/// the proportional shrink; the dominant axis stays at 1.0.
#[allow(clippy::cast_precision_loss)]
fn letterbox_scale(src: (u32, u32), dst: (u32, u32)) -> (f32, f32) {
    if src.0 == 0 || src.1 == 0 || dst.0 == 0 || dst.1 == 0 {
        return (1.0, 1.0);
    }
    let src_aspect = src.0 as f32 / src.1 as f32;
    let dst_aspect = dst.0 as f32 / dst.1 as f32;
    if (src_aspect - dst_aspect).abs() < f32::EPSILON {
        (1.0, 1.0)
    } else if src_aspect > dst_aspect {
        (1.0, dst_aspect / src_aspect)
    } else {
        (src_aspect / dst_aspect, 1.0)
    }
}

fn bytes_of_f32x4(v: &[f32; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (i, x) in v.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_of_u32x4(v: &[u32; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (i, x) in v.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&x.to_le_bytes());
    }
    out
}

fn bgl_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn make_yuv_textures(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    chroma: ChromaSubsampling,
    bit_depth: u8,
    width: u32,
    height: u32,
) -> YuvTextures {
    let render_layout = render_layout_for(chroma, bit_depth);
    match render_layout {
        RenderLayout::Biplanar8 | RenderLayout::Biplanar16 => {
            let (y_format, uv_format) = match render_layout {
                RenderLayout::Biplanar8 => {
                    (wgpu::TextureFormat::R8Unorm, wgpu::TextureFormat::Rg8Unorm)
                }
                RenderLayout::Biplanar16 => (
                    wgpu::TextureFormat::R16Unorm,
                    wgpu::TextureFormat::Rg16Unorm,
                ),
                RenderLayout::Planar444 | RenderLayout::PackedXYUV | RenderLayout::PackedY410 => {
                    unreachable!("guarded by outer match")
                }
            };
            let y = make_plane_texture(device, "tether-render y plane", width, height, y_format);
            let y_view = y.create_view(&wgpu::TextureViewDescriptor::default());
            // UV plane dims: half-res for 4:2:0 (NV12/P010), full-res
            // for 4:4:4 (NV24/P410/xf44). The Rg{8,16}Unorm plane works
            // for either size — bilinear sampling reads the right value
            // regardless.
            let (chroma_w, chroma_h) = match chroma {
                ChromaSubsampling::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
                ChromaSubsampling::Yuv444 => (width, height),
            };
            let uv = make_plane_texture(
                device,
                "tether-render uv plane",
                chroma_w,
                chroma_h,
                uv_format,
            );
            let uv_view = uv.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tether-render yuv bind group (biplanar)"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&y_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&uv_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            let planes = match render_layout {
                RenderLayout::Biplanar8 => YuvPlanes::Biplanar8 { y, uv },
                RenderLayout::Biplanar16 => YuvPlanes::Biplanar16 { y, uv },
                RenderLayout::Planar444 | RenderLayout::PackedXYUV | RenderLayout::PackedY410 => {
                    unreachable!()
                }
            };
            YuvTextures {
                planes,
                bind_group,
                size: (width, height),
                _guard: None,
            }
        }
        RenderLayout::Planar444 => {
            let y = make_plane_texture(
                device,
                "tether-render y plane (444 planar)",
                width,
                height,
                wgpu::TextureFormat::R8Unorm,
            );
            let u = make_plane_texture(
                device,
                "tether-render u plane (444 planar)",
                width,
                height,
                wgpu::TextureFormat::R8Unorm,
            );
            let v = make_plane_texture(
                device,
                "tether-render v plane (444 planar)",
                width,
                height,
                wgpu::TextureFormat::R8Unorm,
            );
            let y_view = y.create_view(&wgpu::TextureViewDescriptor::default());
            let u_view = u.create_view(&wgpu::TextureViewDescriptor::default());
            let v_view = v.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tether-render yuv bind group (444 planar)"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&y_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&u_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&v_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            YuvTextures {
                planes: YuvPlanes::Yuv444Planar { y, u, v },
                bind_group,
                size: (width, height),
                _guard: None,
            }
        }
        RenderLayout::PackedXYUV | RenderLayout::PackedY410 => {
            // 8-bit XYUV → Rgba8Unorm; 10-bit Y410 → Rgb10a2Unorm. Both
            // bind as a single packed texture + sampler.
            let (label, format) = match render_layout {
                RenderLayout::PackedY410 => (
                    "tether-render y410 packed (444 10-bit)",
                    wgpu::TextureFormat::Rgb10a2Unorm,
                ),
                _ => (
                    "tether-render xyuv packed (444)",
                    wgpu::TextureFormat::Rgba8Unorm,
                ),
            };
            let packed = make_plane_texture(device, label, width, height, format);
            let packed_view = packed.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tether-render yuv bind group (444 packed)"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&packed_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            YuvTextures {
                planes: YuvPlanes::Yuv444Packed { packed },
                bind_group,
                size: (width, height),
                _guard: None,
            }
        }
    }
}

/// Range-kind dispatch tag the WGSL fragment shader uses to pick the
/// right limited-range breakpoints for the sampled `(y_lim, c_lim)`.
/// Keep in sync with the `RANGE_KIND_LIMITED_*` constants at the top
/// of `shader.wgsl`.
///
/// **Why bit-depth-parameterised breakpoints:** 8-bit limited-range
/// Y' lives in `[16, 235]` (`16/255 .. 235/255` normalised); 10-bit
/// limited-range Y' lives in `[64, 940]` (10-bit raw), which when
/// stored MSB-aligned in an `R16Unorm` cell lands at `[4096/65535,
/// 60160/65535]`. The 8-bit math `(y_lim - 16/255) * (255/219)` and
/// the 10-bit math `(y_lim - 4096/65535) * (65535/56064)` differ by
/// about 1% in mid-tone normalised luma when the sample is 10-bit
/// data — a small but systematic offset that's invisible in SDR but
/// compounds under PQ's steep mid-tone slope. The earlier renderer
/// version compensated for the storage MSB-align with a single
/// `luma_scale` multiplier and reused the 8-bit math, which left
/// the offset in place; this version branches in the shader and
/// gets it right at all luma levels for both bit depths.
const RANGE_KIND_LIMITED_8: u32 = 0;
const RANGE_KIND_LIMITED_10: u32 = 1;

fn range_kind_for(bit_depth: u8, layout: RenderLayout) -> u32 {
    match (bit_depth, layout) {
        (10, RenderLayout::Biplanar16) => RANGE_KIND_LIMITED_10,
        _ => RANGE_KIND_LIMITED_8,
    }
}

fn make_plane_texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Non-sRGB variants on purpose: YUV values are in a linear
        // encoding space already (the BT.709 limited-range expansion
        // in the fragment shader handles gamma); applying sRGB on
        // sample would double-decode and crush highlights.
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn write_plane_r8(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    bytes: &[u8],
    width: u32,
    height: u32,
) {
    write_plane(queue, texture, bytes, width, height, 1);
}

fn write_plane_rg8(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    bytes: &[u8],
    width: u32,
    height: u32,
) {
    write_plane(queue, texture, bytes, width, height, 2);
}

fn write_plane(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    bytes: &[u8],
    width: u32,
    height: u32,
    bytes_per_texel: u32,
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * bytes_per_texel),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::{
        letterbox_fit_dims, range_kind_for, render_layout_for, transfer_kind_for, RenderLayout,
        RANGE_KIND_LIMITED_10, RANGE_KIND_LIMITED_8, TRANSFER_KIND_BT709, TRANSFER_KIND_SRGB,
    };
    use tether_protocol::control::{ChromaSubsampling, ColorTransfer, VideoColorSpec};

    /// 16:9 video in a 4:3 window: width-bound. Mitchell upscale
    /// produces 800×450 with letterbox bars top and bottom.
    #[test]
    fn letterbox_fit_landscape_in_landscape_window() {
        assert_eq!(letterbox_fit_dims((1920, 1080), (800, 600)), (800, 450));
    }

    /// 9:16 video in a 16:9 window: height-bound. Pillarbox bars
    /// left and right, content fills the height.
    #[test]
    fn letterbox_fit_portrait_in_landscape_window() {
        let (w, h) = letterbox_fit_dims((1080, 1920), (1920, 1080));
        assert_eq!(h, 1080);
        // Aspect preserved to within rounding.
        assert!((w as f32 / h as f32 - 1080.0 / 1920.0).abs() < 0.01);
    }

    #[test]
    fn letterbox_fit_matching_aspect_is_full_window() {
        // 16:9 video in 16:9 window: scaler output exactly fills.
        assert_eq!(letterbox_fit_dims((1920, 1080), (1280, 720)), (1280, 720));
    }

    #[test]
    fn letterbox_fit_zero_dim_returns_video() {
        // Defensive: window zero in either axis (transient on resize)
        // shouldn't divide-by-zero. Returning the source dims is a
        // safe pass-through.
        assert_eq!(letterbox_fit_dims((1920, 1080), (0, 600)), (1920, 1080));
        assert_eq!(letterbox_fit_dims((1920, 1080), (800, 0)), (1920, 1080));
        assert_eq!(letterbox_fit_dims((0, 1080), (800, 600)), (0, 1080));
    }

    /// Pin the Rust → shader range-kind constants so a renumber in
    /// either side surfaces here, not as a silent miscoloured 10-bit
    /// session. Same shape as `transfer_kind_for_pins_the_mapping`.
    #[test]
    fn range_kind_dispatch_matches_bit_depth() {
        assert_eq!(
            range_kind_for(10, RenderLayout::Biplanar16),
            RANGE_KIND_LIMITED_10,
            "10-bit biplanar input must use the 10-bit range breakpoints"
        );
        assert_eq!(
            range_kind_for(8, RenderLayout::Biplanar8),
            RANGE_KIND_LIMITED_8
        );
        assert_eq!(
            range_kind_for(8, RenderLayout::Biplanar16),
            RANGE_KIND_LIMITED_8
        );
        assert_eq!(
            range_kind_for(8, RenderLayout::PackedXYUV),
            RANGE_KIND_LIMITED_8
        );
        assert_eq!(
            range_kind_for(8, RenderLayout::Planar444),
            RANGE_KIND_LIMITED_8
        );
    }

    /// Algebraic check on the 10-bit range constants the shader uses.
    /// 10-bit limited-range Y' = 940 (white) in MSB-aligned 16-bit
    /// storage = `940 * 64 = 60160`. As `R16Unorm` that samples as
    /// `60160 / 65535`. The shader's 10-bit range expansion is
    /// `(y_lim - 4096/65535) * (65535/56064)` — feeding the sample
    /// value back through that math should land at exactly 1.0
    /// (within f32 round-off). Same check for black at the footroom
    /// floor (64 * 64 = 4096).
    #[test]
    fn ten_bit_breakpoints_map_white_and_black_correctly() {
        let footroom = 4096.0_f32 / 65535.0;
        let headroom = 60160.0_f32 / 65535.0;
        let range = 65535.0_f32 / 56064.0;
        let white = (headroom - 4096.0 / 65535.0) * range;
        let black = (footroom - 4096.0 / 65535.0) * range;
        assert!(
            (white - 1.0).abs() < 1e-6,
            "10-bit white expected 1.0; got {white}"
        );
        assert!(black.abs() < 1e-6, "10-bit black expected 0.0; got {black}");
    }

    /// `render_layout_for` dispatch table — pinned so the (chroma,
    /// bit_depth) → RenderLayout mapping doesn't drift without an
    /// explicit test update. The macOS branch for Yuv444 8-bit is
    /// covered indirectly by the cfg switch in the function; this
    /// test runs whichever side the build is on.
    #[test]
    fn render_layout_dispatch_pins_the_mapping() {
        // 8-bit baseline.
        assert_eq!(
            render_layout_for(ChromaSubsampling::Yuv420, 8),
            RenderLayout::Biplanar8
        );
        let yuv444_8bit = render_layout_for(ChromaSubsampling::Yuv444, 8);
        if cfg!(target_os = "macos") {
            assert_eq!(yuv444_8bit, RenderLayout::Biplanar8);
        } else if super::linux_nvidia_planar444_active() {
            assert_eq!(yuv444_8bit, RenderLayout::Planar444);
        } else {
            assert_eq!(yuv444_8bit, RenderLayout::PackedXYUV);
        }
        // 10-bit 4:2:0 (Main10) is biplanar P010 on both platforms.
        assert_eq!(
            render_layout_for(ChromaSubsampling::Yuv420, 10),
            RenderLayout::Biplanar16,
            "Yuv420 10-bit must dispatch to Biplanar16; the renderer's import \
             path relies on this for UV-plane dimensioning"
        );
        // 10-bit 4:4:4: same platform split as 8-bit 4:4:4 — macOS
        // decodes to biplanar P410 (Biplanar16), Linux to packed Y410.
        let yuv444_10bit = render_layout_for(ChromaSubsampling::Yuv444, 10);
        if cfg!(target_os = "macos") {
            assert_eq!(yuv444_10bit, RenderLayout::Biplanar16);
        } else {
            assert_eq!(yuv444_10bit, RenderLayout::PackedY410);
        }
    }

    /// Pins the Rust→shader integer mapping so a refactor that
    /// renumbers the WGSL constants (or reorders the Rust match
    /// arms) fails LOUDLY rather than silently rendering with the
    /// wrong EOTF. Cross-language constant agreement is otherwise
    /// only documented in comments; this test makes the Rust side
    /// machine-checked. The WGSL side stays unchecked at default
    /// `cargo test` time (a GPU-readback integration test would
    /// close that loop; deferred).
    ///
    /// Note on exhaustiveness: `transfer_kind_for`'s `match`
    /// enforces exhaustiveness at compile time, so a new
    /// `ColorTransfer` variant fails to build until it's wired into
    /// the function. *This test* doesn't add coverage for that
    /// case — the array below is hand-written and would silently
    /// not exercise a future variant. The test pins the *intended*
    /// mapping (sdr_desktop → Srgb, sdr_bt709 → BT.709, all current
    /// unimplemented variants → BT.709 fallback) so an inadvertent
    /// renumbering fails here.
    #[test]
    fn transfer_kind_for_pins_the_mapping() {
        assert_eq!(
            transfer_kind_for(VideoColorSpec::sdr_desktop()),
            TRANSFER_KIND_SRGB
        );
        assert_eq!(
            transfer_kind_for(VideoColorSpec::sdr_bt709()),
            TRANSFER_KIND_BT709
        );

        // Each currently-unimplemented variant should hit the
        // BT.709 fallback. Keep this array in sync with the
        // `Pq | Hlg | Linear` arm in `transfer_kind_for`; a future
        // variant added there should also be appended here.
        for transfer in [ColorTransfer::Pq, ColorTransfer::Hlg, ColorTransfer::Linear] {
            let spec = VideoColorSpec {
                transfer,
                ..VideoColorSpec::sdr_desktop()
            };
            assert_eq!(transfer_kind_for(spec), TRANSFER_KIND_BT709);
        }
    }
}
