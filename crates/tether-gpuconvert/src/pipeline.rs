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

// YUV444 (8-bit) is scaffolding kept alive for an as-yet-unwired Linux
// 4:4:4 8-bit path through `Yuv444DmaBuf`. P010 (10-bit 4:2:0) is live
// via [`crate::Bgra2P010DmaBuf`] — `build_p010_pipeline` is no longer
// dead. P410 (10-bit 4:4:4 biplanar) is intentionally still dead on
// Linux: VAAPI's `vaapi_drm_format_map` has no P410 entry, so the
// encoder takes `AV_PIX_FMT_XV30LE` (packed) for 4:4:4 10-bit input —
// not biplanar. The biplanar shader stays in-tree because it covers
// the macOS render-side IOSurface fourcc `'P410'`; an eventual macOS
// gpuconvert-equivalent or a future driver that *does* accept P410
// would revive it. `include_str!` validates the WGSL syntax at build
// time regardless.
#[allow(dead_code)]
pub(crate) const YUV444_SHADER_SRC: &str = include_str!("bgra_to_yuv444.wgsl");
pub(crate) const P010_SHADER_SRC: &str = include_str!("bgra_to_p010.wgsl");
#[allow(dead_code)]
pub(crate) const P410_SHADER_SRC: &str = include_str!("bgra_to_p410.wgsl");

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
#[allow(dead_code)] // scaffolding; see module-level YUV444_SHADER_SRC comment
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

/// Build the BGRA → P010 (10-bit biplanar 4:2:0) compute pipeline and
/// its bind-group layout. Y plane is `R16Unorm`, UV plane is `Rg16Unorm`,
/// half-resolution UV (4:2:0). 10-bit data is MSB-aligned in the
/// 16-bit cells — see `bgra_to_p010.wgsl` for the encoding.
///
/// Live via [`crate::Bgra2P010DmaBuf`]. Callers MUST first gate on
/// [`crate::storable_dmabuf_modifiers`] for `R16` + `GR32` — see the
/// rationale on [`build_biplanar_16_pipeline`].
pub(crate) fn build_p010_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    build_biplanar_16_pipeline(
        device,
        "bgra_to_p010",
        P010_SHADER_SRC,
    )
}

/// Build the BGRA → P410 (10-bit biplanar 4:4:4) compute pipeline.
/// Same plane formats as P010 but full-resolution UV; see
/// `bgra_to_p410.wgsl`.
///
/// Not wired into any Linux production path — VAAPI's
/// `vaapi_drm_format_map` has no P410 entry, so 4:4:4 10-bit encode
/// input takes packed XV30 instead. This builder is kept for an
/// eventual macOS gpuconvert-equivalent path that does want biplanar
/// 10-bit 4:4:4 (IOSurface fourcc `'P410'`).
#[allow(dead_code)] // see module-level comment on P410_SHADER_SRC
pub(crate) fn build_p410_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    build_biplanar_16_pipeline(
        device,
        "bgra_to_p410",
        P410_SHADER_SRC,
    )
}

/// Shared 10-bit biplanar pipeline construction — the bind-group
/// layout and dispatch shape are identical between P010 and P410 (both
/// have an R16 Y plane + Rg16 UV plane); only the shader's per-pixel
/// math + the runtime dispatch dims differ.
///
/// Storage-feature gating: `R16Unorm` / `Rg16Unorm` as compute storage
/// outputs require the Vulkan format feature flag `STORAGE_IMAGE_BIT`
/// on the chosen `VkFormat`. Some drivers expose 16-bit unorm as
/// sampleable but not as storage-writable; on those, this pipeline
/// would fail at `create_compute_pipeline`. The
/// [`crate::storable_dmabuf_modifiers`] probe is the gate — callers
/// invoke it before constructing the bridge (see
/// [`crate::Bgra2P010DmaBuf::new`] for the live wiring).
fn build_biplanar_16_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    shader_src: &str,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let bgl_label = format!("{label} bgl");
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&bgl_label),
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
                    format: wgpu::TextureFormat::R16Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rg16Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let pl_label = format!("{label} pl");
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&pl_label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (pipeline, bgl)
}
