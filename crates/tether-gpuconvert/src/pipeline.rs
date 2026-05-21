//! Shared compute-pipeline construction for the BGRA→NV12 shader.
//!
//! Two consumers — [`crate::Bgra2Nv12`] (CPU↔CPU API, test/fallback) and
//! [`crate::nv12_dmabuf::Nv12DmaBuf`] (production DMA-BUF path). They
//! differ in what they put on either side of the pipeline but use the
//! same shader, the same bind-group layout, and the same dispatch
//! shape, so the construction logic lives here. Keeping it shared also
//! means a binding-order change in the WGSL only needs one corresponding
//! Rust edit.

pub(crate) const SHADER_SRC: &str = include_str!("bgra_to_nv12.wgsl");
pub(crate) const YUV444_SHADER_SRC: &str = include_str!("bgra_to_yuv444.wgsl");

/// Build the BGRA→NV12 compute pipeline and its bind-group layout.
/// Caller owns both; bind groups are constructed per-call-site because
/// the source texture varies (resident texture for [`crate::Bgra2Nv12`],
/// imported DMA-BUF for the bridge).
pub(crate) fn build_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bgra_to_nv12"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgra_to_nv12 bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rg8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bgra_to_nv12 pl"),
        bind_group_layouts: &[Some(&bgl)],
        // wgpu trunk renamed push_constant_ranges → immediate_size; we use neither.
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bgra_to_nv12"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (pipeline, bgl)
}

/// Build the BGRA → packed YUV 4:4:4 (DRM_FORMAT_XYUV8888 layout)
/// compute pipeline. The output is a single Rgba8Unorm storage texture
/// at full resolution; per-byte layout matches what ffmpeg's
/// `vaapi_map_from_drm` recognises as VA_FOURCC_XYUV. See the comment
/// at the top of `bgra_to_yuv444.wgsl` for why packed instead of
/// planar.
pub(crate) fn build_yuv444_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bgra_to_yuv444"),
        source: wgpu::ShaderSource::Wgsl(YUV444_SHADER_SRC.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgra_to_yuv444 bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bgra_to_yuv444 pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bgra_to_yuv444"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (pipeline, bgl)
}
