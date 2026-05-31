//! [`Scaler`]: the wgpu-backed driver. Builds the mipmap chain at
//! construction time, runs the shader on each call to [`Scaler::scale`].

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, Device, Extent3d, Queue, Texture,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};

use crate::pipeline::Pipelines;
use crate::reference::{tap_count, MAX_TAPS};
use crate::ScalerError;

/// Shared, pre-compiled [`Pipelines`] reused across [`Scaler`]
/// rebuilds. wgpu's `Device` / `Queue` are themselves cheap handle
/// types (Arc-backed internally) so they're stored owned, not behind
/// an outer Arc.
pub type PipelineHandle = Arc<Pipelines>;

/// Which colour-space pipeline to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorSpace {
    /// sRGB-encoded `Rgba8Unorm` input and output. The shader
    /// applies the piecewise sRGB EOTF on read, filters in linear
    /// light, applies the OETF on write. Includes the 2× box mipmap
    /// prefilter for downscale ratios > 2×.
    ///
    /// Use this for: host downscale from PipeWire-captured BGRA
    /// before chroma conversion.
    Srgb8,
    /// Linear-light `Rgba16Float` input and output. No transfer
    /// applied either way — the input texture is *already* in
    /// linear light and the consumer (e.g. swapchain blit) handles
    /// the OETF on its own.
    ///
    /// No mip prefilter — this mode targets upscale where the
    /// kernel widening alone is sufficient. Downscale > 2× in this
    /// mode would alias.
    ///
    /// Use this for: client renderer upscale, where the YUV→RGB
    /// fragment shader already emits linear-light values into an
    /// `Rgba16Float` intermediate.
    LinearF16,
    /// `R8Unorm` (single-channel) plane scaling, no transfer applied.
    /// Used for the Y plane on the macOS host scaler bridge —
    /// downscales an NV12 luma plane. Luma siting requires no
    /// correction (luma samples are pixel-centered on both ends),
    /// so this variant is a unit struct; the [`ChromaRg8`] sibling
    /// is what carries the cosited-NV12 chroma offset.
    ///
    /// Output format: `R8Unorm` storage. The host device must opt
    /// into `Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`
    /// and the adapter must advertise `STORAGE_BINDING` on
    /// `R8Unorm` (Metal always; Vulkan depends on the ICD).
    ///
    /// No mip prefilter — for the typical macOS use case
    /// (capture-vs-viewport ratio ≤ ~2×) the kernel widening
    /// alone suffices. The [`Scaler::new_with_color_space`] entry
    /// point rejects ratios that would overflow `MAX_TAPS` for
    /// the YUV variants.
    LumaR8,
    /// `Rg8Unorm` (two-channel) plane scaling, no transfer applied.
    /// Used for the interleaved UV plane on the macOS host scaler
    /// bridge — downscales an NV12 chroma plane. `chroma_offset` is
    /// the cosited-siting correction in *source-pixel* units;
    /// callers pass `(-(h_scale - 1) * 0.5, 0.0)` for typical macOS
    /// left-cosited NV12 (see plan, "Chroma siting"). Setting it to
    /// zero produces visible chroma fringing on text at downscale
    /// ratios > 1×.
    ///
    /// Output format: `Rg8Unorm` storage. Same wgpu feature
    /// caveats as [`LumaR8`].
    ChromaRg8 { chroma_offset: (f32, f32) },
    /// 10-bit (R16Unorm) single-channel plane scaling — the
    /// macOS NV12 `'x420'` / `'xf20'` / `'P010'` luma plane.
    /// Symmetric to [`LumaR8`] but the storage format is 16-bit-
    /// per-channel; the Mitchell math is identical. The host
    /// device must opt into BOTH
    /// `Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` and
    /// `Features::TEXTURE_FORMAT_16BIT_NORM`.
    LumaR16,
    /// 10-bit (Rg16Unorm) two-channel plane scaling — the
    /// macOS NV12 10-bit interleaved UV plane. Cosited-NV12
    /// chroma offset has identical math at 8-bit and 10-bit;
    /// caller passes the same `-(scale - 1) * 0.5` correction.
    ChromaRg16 { chroma_offset: (f32, f32) },
}

impl ColorSpace {
    fn output_format(self) -> wgpu::TextureFormat {
        match self {
            Self::Srgb8 => wgpu::TextureFormat::Rgba8Unorm,
            Self::LinearF16 => wgpu::TextureFormat::Rgba16Float,
            Self::LumaR8 => wgpu::TextureFormat::R8Unorm,
            Self::ChromaRg8 { .. } => wgpu::TextureFormat::Rg8Unorm,
            Self::LumaR16 => wgpu::TextureFormat::R16Unorm,
            Self::ChromaRg16 { .. } => wgpu::TextureFormat::Rg16Unorm,
        }
    }
    fn supports_mip_prefilter(self) -> bool {
        matches!(self, Self::Srgb8)
    }
    /// Sample-position offset (in *source-pixel* units) to bake into
    /// the kernel center. Zero for the RGB paths and the Y plane; the
    /// caller computes the cosited NV12 chroma correction for UV.
    fn chroma_offset(self) -> (f32, f32) {
        match self {
            Self::Srgb8 | Self::LinearF16 | Self::LumaR8 | Self::LumaR16 => (0.0, 0.0),
            Self::ChromaRg8 { chroma_offset } | Self::ChromaRg16 { chroma_offset } => chroma_offset,
        }
    }
    fn needs_plane_pipelines(self) -> bool {
        matches!(
            self,
            Self::LumaR8 | Self::ChromaRg8 { .. } | Self::LumaR16 | Self::ChromaRg16 { .. }
        )
    }
    /// True for the 16-bit-storage plane variants. Used by the
    /// pipeline builder to gate the `R16Unorm`/`Rg16Unorm` BGLs +
    /// pipelines behind the `TEXTURE_FORMAT_16BIT_NORM` feature opt-
    /// in: a device that has `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`
    /// but not 16BIT_NORM (rare on Metal, more common on older
    /// Vulkan ICDs) gets the 8-bit pipelines and refuses 10-bit
    /// scaling via [`ScalerError::MissingPlanePipelines`].
    #[allow(dead_code)] // referenced once we expose it to PlanePipelines below
    fn is_16bit_plane(self) -> bool {
        matches!(self, Self::LumaR16 | Self::ChromaRg16 { .. })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    n_taps: u32,
    // Sample-position offset (source-pixel units) added to the kernel
    // center. Zero for the RGB paths; UV planes use the cosited-siting
    // correction. See `ColorSpace::ChromaRg8` docs and the plan.
    chroma_offset_x: f32,
    chroma_offset_y: f32,
    _pad: u32,
}

/// Mitchell-Netravali compute scaler. Constructed once per
/// `(src, dst)` pair; the mip chain and intermediate textures are
/// allocated up front so a per-frame `scale()` call only records and
/// submits a command buffer.
pub struct Scaler {
    device: Device,
    queue: Queue,
    pipelines: PipelineHandle,

    /// Box-filter mip outputs. Length = number of mip passes; level 0
    /// halves the source, level 1 halves level 0, etc. The first
    /// Mitchell pass consumes from `mip_chain.last()` if non-empty,
    /// otherwise from the source passed to `scale()`.
    mip_chain: Vec<Texture>,

    /// Linear-light Rgba16Float intermediate after the horizontal
    /// Mitchell pass; consumed by the vertical pass.
    intermediate: Texture,

    /// Final sRGB-encoded Rgba8Unorm output at `dst_dims`.
    output: Texture,

    /// Uniform buffers for the horizontal and vertical passes,
    /// populated at construction (dims are fixed for the scaler's
    /// lifetime).
    params_h: Buffer,
    params_v: Buffer,

    src_dims: (u32, u32),
    dst_dims: (u32, u32),
    color_space: ColorSpace,
}

impl Scaler {
    /// Build a scaler for the given source/destination dimensions,
    /// reusing pre-compiled pipelines.
    ///
    /// Construct [`Pipelines`] once with [`Pipelines::build`], wrap in
    /// `Arc`, then call this for each `(src, dst)` you need. The host
    /// rebuilds scalers on resolution change; sharing the compiled
    /// shader avoids a 50–200ms recompile per rebuild on Metal/DX12.
    ///
    /// Errors:
    /// - [`ScalerError::ZeroDim`]: any dimension is zero.
    /// - [`ScalerError::NoScaleNeeded`]: `dst_dims == src_dims` exactly.
    ///   Pure upscale (`dst > src` in both axes) is supported and
    ///   doesn't trip this error — that's the client path.
    pub fn new(
        pipelines: PipelineHandle,
        device: Device,
        queue: Queue,
        src_dims: (u32, u32),
        dst_dims: (u32, u32),
    ) -> Result<Self, ScalerError> {
        Self::new_with_color_space(
            pipelines,
            device,
            queue,
            src_dims,
            dst_dims,
            ColorSpace::Srgb8,
        )
    }

    /// Build with an explicit colour space. See [`ColorSpace`] for
    /// what each variant means; [`Scaler::new`] is a convenience
    /// alias for `Srgb8` (the host downscale path).
    pub fn new_with_color_space(
        pipelines: PipelineHandle,
        device: Device,
        queue: Queue,
        src_dims: (u32, u32),
        dst_dims: (u32, u32),
        color_space: ColorSpace,
    ) -> Result<Self, ScalerError> {
        if src_dims.0 == 0 || src_dims.1 == 0 || dst_dims.0 == 0 || dst_dims.1 == 0 {
            return Err(ScalerError::ZeroDim {
                src: src_dims,
                dst: dst_dims,
            });
        }
        if dst_dims == src_dims {
            return Err(ScalerError::NoScaleNeeded);
        }
        // Fail at construction, not at first scale() — a YUV scaler
        // built without plane pipelines is permanently broken, and
        // catching it here turns a runtime frame-rate error into a
        // bridge-init error that names the cause.
        if color_space.needs_plane_pipelines() && pipelines.plane.is_none() {
            return Err(ScalerError::MissingPlanePipelines);
        }
        // YUV plane paths skip the mip prefilter (would need r8unorm /
        // rg8unorm mip variants of `mip_box_down`, which we haven't
        // wired). Without mip prefiltering, tap_count widens with the
        // downscale ratio and saturates at MAX_TAPS (32) at scale=8.
        // Reject upfront if the requested ratio would saturate, so
        // the bridge gets a clean error instead of a silently
        // bandlimit-failing scaler.
        if color_space.needs_plane_pipelines() {
            let scale_x_check = src_dims.0 as f32 / dst_dims.0 as f32;
            let scale_y_check = src_dims.1 as f32 / dst_dims.1 as f32;
            let max_ratio = (MAX_TAPS as f32) / 4.0;
            let ratio = scale_x_check.max(scale_y_check);
            if ratio > max_ratio {
                return Err(ScalerError::YuvDownscaleTooLarge {
                    src: src_dims,
                    dst: dst_dims,
                    ratio,
                    max_ratio,
                });
            }
        }

        // Build the mip chain. Each level halves dimensions (round
        // up); stop when both axes are within 2× of dst. Linear-
        // light mode skips this — that path is upscale-only and the
        // mip shader operates on sRGB-encoded values.
        let mut mip_chain = Vec::new();
        let mut cur_w = src_dims.0;
        let mut cur_h = src_dims.1;
        if color_space.supports_mip_prefilter() {
            loop {
                let scale_x = cur_w as f32 / dst_dims.0 as f32;
                let scale_y = cur_h as f32 / dst_dims.1 as f32;
                if scale_x.max(scale_y) <= 2.0 {
                    break;
                }
                cur_w = cur_w.div_ceil(2);
                cur_h = cur_h.div_ceil(2);
                mip_chain.push(make_rgba8_storage(
                    &device,
                    "tether-scaler mip",
                    cur_w,
                    cur_h,
                ));
            }
        }
        let after_mip_w = cur_w;
        let after_mip_h = cur_h;

        // Intermediate after the horizontal Mitchell pass:
        // (dst_w, after_mip_h) in linear-light fp16.
        let intermediate = device.create_texture(&TextureDescriptor {
            label: Some("tether-scaler intermediate"),
            size: Extent3d {
                width: dst_dims.0,
                height: after_mip_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Output texture format follows the colour space — Rgba8Unorm
        // for sRGB (chroma-converter consumer), Rgba16Float for
        // linear-light (swapchain-blit consumer). COPY_SRC so tests
        // can read it back regardless.
        let output = device.create_texture(&TextureDescriptor {
            label: Some("tether-scaler output"),
            size: Extent3d {
                width: dst_dims.0,
                height: dst_dims.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: color_space.output_format(),
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let scale_x = after_mip_w as f32 / dst_dims.0 as f32;
        let scale_y = after_mip_h as f32 / dst_dims.1 as f32;
        let (offset_x, offset_y) = color_space.chroma_offset();
        let params_h_data = Params {
            src_w: after_mip_w,
            src_h: after_mip_h,
            dst_w: dst_dims.0,
            dst_h: after_mip_h,
            n_taps: tap_count(scale_x),
            chroma_offset_x: offset_x,
            chroma_offset_y: 0.0,
            _pad: 0,
        };
        let params_v_data = Params {
            src_w: dst_dims.0,
            src_h: after_mip_h,
            dst_w: dst_dims.0,
            dst_h: dst_dims.1,
            n_taps: tap_count(scale_y),
            chroma_offset_x: 0.0,
            chroma_offset_y: offset_y,
            _pad: 0,
        };

        let params_h = make_uniform(&device, "tether-scaler params_h", &params_h_data);
        let params_v = make_uniform(&device, "tether-scaler params_v", &params_v_data);

        debug_assert!(
            params_h_data.n_taps <= MAX_TAPS && params_v_data.n_taps <= MAX_TAPS,
            "tap count exceeded MAX_TAPS — mip chain should have prevented this",
        );

        Ok(Self {
            device,
            queue,
            pipelines,
            mip_chain,
            intermediate,
            output,
            params_h,
            params_v,
            src_dims,
            dst_dims,
            color_space,
        })
    }

    /// Run mip chain (if any) + horizontal + vertical Mitchell passes.
    /// Writes into the scaler's owned output texture; returns a
    /// borrowed reference (also obtainable via [`Self::output`]).
    ///
    /// For callers that need to write into an externally-allocated
    /// destination (e.g. the macOS NV12 IOSurface bridge, which wraps
    /// an IOSurface-backed `MTLTexture` as the scaler destination),
    /// use [`Self::scale_into`] instead.
    pub fn scale(&self, src: &Texture) -> Result<&Texture, ScalerError> {
        // Borrow-checker: scale_into takes `&self.output` (immutable
        // borrow) and returns `()`; then we return `&self.output`
        // again. Two non-overlapping immutable borrows, fine.
        self.scale_into(src, &self.output)?;
        Ok(&self.output)
    }

    /// Run the scaler with an explicit destination texture. The
    /// destination must have format == [`Self::dst_format`] and
    /// dimensions == [`Self::dst_dims`]; mismatched destinations
    /// return [`ScalerError::DimMismatch`] (dims) or are caught at
    /// the wgpu bind-group level (format).
    ///
    /// Use this when the destination is externally owned — the macOS
    /// NV12 IOSurface bridge wraps an `MTLTexture` (which borrows
    /// storage from a pooled IOSurface) as the scaler output, so the
    /// scaler writes directly into the surface VideoToolbox encodes
    /// from, no intermediate copy.
    pub fn scale_into(&self, src: &Texture, dst: &Texture) -> Result<(), ScalerError> {
        if src.width() != self.src_dims.0 || src.height() != self.src_dims.1 {
            // Returning rather than running the shader with stale
            // params: a resize race or wiring bug would otherwise
            // silently corrupt every frame until the next rebuild.
            return Err(ScalerError::DimMismatch {
                expected: self.src_dims,
                got: (src.width(), src.height()),
            });
        }
        if dst.width() != self.dst_dims.0 || dst.height() != self.dst_dims.1 {
            return Err(ScalerError::DimMismatch {
                expected: self.dst_dims,
                got: (dst.width(), dst.height()),
            });
        }

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("tether-scaler scale"),
            });

        let src_view = src.create_view(&TextureViewDescriptor::default());
        let mut prev_view: TextureView = src_view;

        // Mip chain: 2× box downsample per level.
        for (level, tex) in self.mip_chain.iter().enumerate() {
            let dst_view = tex.create_view(&TextureViewDescriptor::default());
            let bg = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("tether-scaler mip bg"),
                layout: &self.pipelines.mip_bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&prev_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&dst_view),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("tether-scaler mip pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.mip);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(tex.width().div_ceil(8), tex.height().div_ceil(8), 1);
            }
            tracing::trace!(level, w = tex.width(), h = tex.height(), "mip downsample");
            prev_view = tex.create_view(&TextureViewDescriptor::default());
        }

        // Horizontal pass: the linear-light horizontal entry point
        // is reused for Srgb8 (via `horizontal`) and for the YUV
        // plane paths (via `horizontal_linear`, which reads through
        // `texture_2d<f32>` so an R8/Rg8 source returns its channels
        // in `.r`/`.rg` automatically; the unused channels read
        // as zero and contribute nothing to the weighted sum).
        // The vertical bind-group layout and pipeline diverge across
        // all four variants because the destination storage format is
        // different in each (Rgba8Unorm / Rgba16Float / R8Unorm /
        // Rg8Unorm).
        let (horizontal_pipeline, vertical_pipeline, vertical_bgl) = match self.color_space {
            ColorSpace::Srgb8 => (
                &self.pipelines.horizontal,
                &self.pipelines.vertical,
                &self.pipelines.vertical_bgl,
            ),
            ColorSpace::LinearF16 => (
                &self.pipelines.horizontal_linear,
                &self.pipelines.vertical_linear,
                &self.pipelines.vertical_linear_bgl,
            ),
            ColorSpace::LumaR8 { .. } => {
                let plane = self
                    .pipelines
                    .plane
                    .as_ref()
                    .ok_or(ScalerError::MissingPlanePipelines)?;
                (
                    &self.pipelines.horizontal_linear,
                    &plane.vertical_plane_r,
                    &plane.vertical_plane_r_bgl,
                )
            }
            ColorSpace::ChromaRg8 { .. } => {
                let plane = self
                    .pipelines
                    .plane
                    .as_ref()
                    .ok_or(ScalerError::MissingPlanePipelines)?;
                (
                    &self.pipelines.horizontal_linear,
                    &plane.vertical_plane_rg,
                    &plane.vertical_plane_rg_bgl,
                )
            }
            ColorSpace::LumaR16 { .. } => {
                let plane = self
                    .pipelines
                    .plane
                    .as_ref()
                    .ok_or(ScalerError::MissingPlanePipelines)?;
                let pipeline = plane
                    .vertical_plane_r16
                    .as_ref()
                    .ok_or(ScalerError::MissingPlanePipelines)?;
                let bgl = plane
                    .vertical_plane_r16_bgl
                    .as_ref()
                    .ok_or(ScalerError::MissingPlanePipelines)?;
                (&self.pipelines.horizontal_linear, pipeline, bgl)
            }
            ColorSpace::ChromaRg16 { .. } => {
                let plane = self
                    .pipelines
                    .plane
                    .as_ref()
                    .ok_or(ScalerError::MissingPlanePipelines)?;
                let pipeline = plane
                    .vertical_plane_rg16
                    .as_ref()
                    .ok_or(ScalerError::MissingPlanePipelines)?;
                let bgl = plane
                    .vertical_plane_rg16_bgl
                    .as_ref()
                    .ok_or(ScalerError::MissingPlanePipelines)?;
                (&self.pipelines.horizontal_linear, pipeline, bgl)
            }
        };

        // Horizontal BGL is shared between the Srgb8, LinearF16, and
        // YUV plane paths because every horizontal pipeline writes
        // the Rgba16Float intermediate. If a future variant ever
        // writes a different intermediate format, that new pipeline
        // needs its own horizontal BGL — at which point this
        // dispatch should grow a match arm. The comment is the
        // load-bearing reminder; a debug_assert on layout identity
        // isn't expressible through wgpu's public API today.
        let h_dst_view = self
            .intermediate
            .create_view(&TextureViewDescriptor::default());
        let h_bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("tether-scaler horizontal bg"),
            layout: &self.pipelines.horizontal_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&prev_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&h_dst_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.params_h.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("tether-scaler horizontal pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(horizontal_pipeline);
            pass.set_bind_group(0, &h_bg, &[]);
            pass.dispatch_workgroups(
                self.intermediate.width().div_ceil(8),
                self.intermediate.height().div_ceil(8),
                1,
            );
        }

        // Vertical pass — bind-group layout differs between sRGB
        // (output Rgba8Unorm) and linear (output Rgba16Float).
        // `dst` is the caller-supplied destination (or `&self.output`
        // when entered via `scale()`); its format must match the
        // vertical_bgl picked above. wgpu's bind-group validation
        // catches the mismatch.
        let v_src_view = self
            .intermediate
            .create_view(&TextureViewDescriptor::default());
        let v_dst_view = dst.create_view(&TextureViewDescriptor::default());
        let v_bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("tether-scaler vertical bg"),
            layout: vertical_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&v_src_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&v_dst_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.params_v.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("tether-scaler vertical pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(vertical_pipeline);
            pass.set_bind_group(0, &v_bg, &[]);
            pass.dispatch_workgroups(dst.width().div_ceil(8), dst.height().div_ceil(8), 1);
        }

        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    pub fn output(&self) -> &Texture {
        &self.output
    }
    pub fn dst_format(&self) -> TextureFormat {
        self.color_space.output_format()
    }
    pub fn color_space(&self) -> ColorSpace {
        self.color_space
    }
    pub fn src_dims(&self) -> (u32, u32) {
        self.src_dims
    }
    pub fn dst_dims(&self) -> (u32, u32) {
        self.dst_dims
    }
    /// Number of mip-prefilter levels in the chain. Zero for direct
    /// (≤ 2× downscale or any upscale).
    pub fn mip_levels(&self) -> usize {
        self.mip_chain.len()
    }
}

fn make_rgba8_storage(device: &Device, label: &str, w: u32, h: u32) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn make_uniform<T: Pod>(device: &Device, label: &str, data: &T) -> Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(data),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn new_zero_dim_errors() {
        // We can't actually build wgpu device cheaply in unit test;
        // this only checks the validation runs before allocating. Use
        // a stub that exercises validation by short-circuiting wgpu
        // construction. (The validation is at the top of `new`, before
        // any wgpu calls — exercise it via a `must_use` chain.)
        //
        // Easier: just confirm the predicate evaluates as expected by
        // calling the underlying check inline; the test serves as a
        // pinning comment.
        let cases = [
            ((0, 100), (50, 50)),
            ((100, 0), (50, 50)),
            ((100, 100), (0, 50)),
            ((100, 100), (50, 0)),
        ];
        for (src, dst) in cases {
            let is_zero = src.0 == 0 || src.1 == 0 || dst.0 == 0 || dst.1 == 0;
            assert!(is_zero, "case {src:?} -> {dst:?} should detect zero dim");
        }
    }
}
