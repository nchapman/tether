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

/// All three pipelines + the bind-group layouts they reference.
pub(crate) struct Pipelines {
    pub horizontal: ComputePipeline,
    pub horizontal_bgl: BindGroupLayout,
    pub vertical: ComputePipeline,
    pub vertical_bgl: BindGroupLayout,
    pub mip: ComputePipeline,
    pub mip_bgl: BindGroupLayout,
}

impl Pipelines {
    pub(crate) fn build(device: &Device) -> Self {
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

        Self {
            horizontal,
            horizontal_bgl,
            vertical,
            vertical_bgl,
            mip,
            mip_bgl,
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
