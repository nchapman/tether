//! wgpu pipeline construction for the three shader entry points.
//!
//! Split out from [`crate::scaler`] so the pipeline-construction
//! boilerplate doesn't clutter the `Scaler` impl.

use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayout, PipelineLayoutDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, TextureFormat,
    TextureSampleType, TextureViewDimension,
};

/// All five pipelines + the bind-group layouts they reference.
///
/// The sRGB pipelines (`horizontal`, `vertical`) and the mip
/// prefilter (`mip`) implement the host downscale path: read
/// sRGB-encoded `Rgba8Unorm`, filter in linear-light, write
/// sRGB-encoded `Rgba8Unorm`. The linear-light pipelines
/// (`horizontal_linear`, `vertical_linear`) implement the client
/// upscale path: read linear-light `Rgba16Float`, filter, write
/// linear-light `Rgba16Float` — no transfer functions either way,
/// since the upstream YUV→RGB fragment shader already emits linear
/// light.
///
/// Shader compilation is non-trivial on Metal and DX12 (50–200ms for
/// the scaler shader); the host rebuilds its [`crate::Scaler`] on every
/// resolution change, so the pipelines need to be reusable across
/// rebuilds. Construct once via [`Pipelines::build`], wrap in
/// [`std::sync::Arc`], and hand to each `Scaler::new` call.
pub struct Pipelines {
    // sRGB path: Rgba8Unorm in/out (with Rgba16Float intermediate).
    pub(crate) horizontal: ComputePipeline,
    pub(crate) horizontal_bgl: BindGroupLayout,
    pub(crate) vertical: ComputePipeline,
    pub(crate) vertical_bgl: BindGroupLayout,
    pub(crate) mip: ComputePipeline,
    pub(crate) mip_bgl: BindGroupLayout,
    // Linear-light path: Rgba16Float throughout. Vertical output
    // format differs from the sRGB vertical (Rgba16Float vs
    // Rgba8Unorm) so it needs its own BGL + pipeline; the horizontal
    // *intermediate* format is the same (Rgba16Float) but the shader
    // entry point differs, so it also gets its own pipeline.
    pub(crate) horizontal_linear: ComputePipeline,
    pub(crate) vertical_linear: ComputePipeline,
    pub(crate) vertical_linear_bgl: BindGroupLayout,
    // YUV plane vertical passes (host NV12 scaler). Opt-in via
    // [`Pipelines::build_with_plane_storage`] because R8Unorm and
    // Rg8Unorm `STORAGE_BINDING` are not in WebGPU core — they
    // require the device to have opted into
    // `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` and the adapter to
    // actually report `STORAGE_BINDING` on those formats. Creating
    // the bind-group layouts without those preconditions raises a
    // validation error, so we gate construction behind the
    // [`PlanePipelines`] sub-struct and leave it `None` for callers
    // that don't need them (client renderer, default scaler tests).
    pub(crate) plane: Option<PlanePipelines>,
}

/// YUV-plane vertical pipelines + bind-group layouts. Constructed
/// only when [`Pipelines::build_with_plane_storage`] is used. The
/// horizontal pass for YUV planes reuses `Pipelines::horizontal_linear`
/// — texture sampling treats R8/Rg8 sources transparently, only the
/// destination storage format forces a new vertical entry point.
pub(crate) struct PlanePipelines {
    pub(crate) vertical_plane_r: ComputePipeline,
    pub(crate) vertical_plane_r_bgl: BindGroupLayout,
    pub(crate) vertical_plane_rg: ComputePipeline,
    pub(crate) vertical_plane_rg_bgl: BindGroupLayout,
}

impl Pipelines {
    /// Compile the shader and build the core pipelines + their
    /// bind-group layouts (sRGB and linear-light RGB paths). Caller
    /// wraps in `Arc` and reuses across `Scaler::new` calls — see
    /// crate-level docs.
    ///
    /// For the macOS host's NV12 YUV-plane scaler, use
    /// [`Pipelines::build_with_plane_storage`] instead — that adds the
    /// R8Unorm / Rg8Unorm vertical pipelines this constructor omits.
    pub fn build(device: &Device) -> Self {
        Self::build_internal(device, false)
    }

    /// Like [`Pipelines::build`] but also constructs the YUV-plane
    /// vertical pipelines used by [`crate::ColorSpace::LumaR8`] /
    /// [`crate::ColorSpace::ChromaRg8`]. The device must have opted
    /// into `Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` and
    /// the adapter must advertise `STORAGE_BINDING` support on
    /// R8Unorm/Rg8Unorm (Metal always; Vulkan depends on the ICD) —
    /// otherwise bind-group-layout creation will validation-error.
    pub fn build_with_plane_storage(device: &Device) -> Self {
        Self::build_internal(device, true)
    }

    fn build_internal(device: &Device, with_plane_storage: bool) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("tether-scaler shader"),
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let (horizontal_bgl, horizontal_layout) = build_mitchell_layout(
            device,
            "tether-scaler horizontal bgl",
            TextureFormat::Rgba16Float,
        );
        let horizontal = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("tether-scaler horizontal"),
            layout: Some(&horizontal_layout),
            module: &module,
            entry_point: Some("horizontal"),
            compilation_options: Default::default(),
            cache: None,
        });

        let (vertical_bgl, vertical_layout) = build_mitchell_layout(
            device,
            "tether-scaler vertical bgl",
            TextureFormat::Rgba8Unorm,
        );
        let vertical = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("tether-scaler vertical"),
            layout: Some(&vertical_layout),
            module: &module,
            entry_point: Some("vertical"),
            compilation_options: Default::default(),
            cache: None,
        });

        let (mip_bgl, mip_layout) = build_mip_layout(device);
        let mip = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("tether-scaler mip"),
            layout: Some(&mip_layout),
            module: &module,
            entry_point: Some("mip_box_down"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Linear-light path. The horizontal_linear shader writes to
        // Rgba16Float (same as the sRGB horizontal); the
        // vertical_linear shader also writes to Rgba16Float
        // (different from the sRGB vertical which writes
        // Rgba8Unorm) so it needs its own bind-group layout.
        let horizontal_linear = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("tether-scaler horizontal_linear"),
            layout: Some(&horizontal_layout),
            module: &module,
            entry_point: Some("horizontal_linear"),
            compilation_options: Default::default(),
            cache: None,
        });
        let (vertical_linear_bgl, vertical_linear_layout) = build_mitchell_layout(
            device,
            "tether-scaler vertical_linear bgl",
            TextureFormat::Rgba16Float,
        );
        let vertical_linear = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("tether-scaler vertical_linear"),
            layout: Some(&vertical_linear_layout),
            module: &module,
            entry_point: Some("vertical_linear"),
            compilation_options: Default::default(),
            cache: None,
        });

        // YUV plane vertical entry points. Built only when opted in
        // via `build_with_plane_storage` — see [`Pipelines::plane`]
        // for the rationale (R8Unorm / Rg8Unorm storage isn't in
        // WebGPU core and would validation-error on a default device).
        let plane = if with_plane_storage {
            let (vertical_plane_r_bgl, vertical_plane_r_layout) = build_mitchell_layout(
                device,
                "tether-scaler vertical_plane_r bgl",
                TextureFormat::R8Unorm,
            );
            let vertical_plane_r = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("tether-scaler vertical_plane_r"),
                layout: Some(&vertical_plane_r_layout),
                module: &module,
                entry_point: Some("vertical_plane_r"),
                compilation_options: Default::default(),
                cache: None,
            });
            let (vertical_plane_rg_bgl, vertical_plane_rg_layout) = build_mitchell_layout(
                device,
                "tether-scaler vertical_plane_rg bgl",
                TextureFormat::Rg8Unorm,
            );
            let vertical_plane_rg = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("tether-scaler vertical_plane_rg"),
                layout: Some(&vertical_plane_rg_layout),
                module: &module,
                entry_point: Some("vertical_plane_rg"),
                compilation_options: Default::default(),
                cache: None,
            });
            Some(PlanePipelines {
                vertical_plane_r,
                vertical_plane_r_bgl,
                vertical_plane_rg,
                vertical_plane_rg_bgl,
            })
        } else {
            None
        };

        Self {
            horizontal,
            horizontal_bgl,
            vertical,
            vertical_bgl,
            mip,
            mip_bgl,
            horizontal_linear,
            vertical_linear,
            vertical_linear_bgl,
            plane,
        }
    }
}

/// Layout shared by the horizontal and vertical Mitchell passes:
/// `binding 0` = source `texture_2d<f32>`, `binding 1` = destination
/// storage texture, `binding 2` = `Params` uniform. The destination
/// format differs between the two passes (Rgba16Float for horizontal,
/// Rgba8Unorm for vertical) so we build the layout twice with the
/// right storage format.
fn build_mitchell_layout(
    device: &Device,
    label: &str,
    dst_format: TextureFormat,
) -> (BindGroupLayout, PipelineLayout) {
    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: dst_format,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        // wgpu trunk renamed push_constant_ranges → immediate_size.
        immediate_size: 0,
    });
    (bgl, layout)
}

/// Mip-pass layout: source texture + destination storage texture. No
/// uniforms — the shader reads dimensions via `textureDimensions`.
fn build_mip_layout(device: &Device) -> (BindGroupLayout, PipelineLayout) {
    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("tether-scaler mip bgl"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba8Unorm,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("tether-scaler mip"),
        bind_group_layouts: &[Some(&bgl)],
        // wgpu trunk renamed push_constant_ranges → immediate_size.
        immediate_size: 0,
    });
    (bgl, layout)
}
