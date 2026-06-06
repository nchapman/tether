//! Compute-pipeline construction for the BGRA→YUV conversion shaders.
//!
//! One builder per encoder-input format the host advertises:
//!
//! - [`build_pipeline`] — BGRA→NV12 (4:2:0 8-bit). Two consumers:
//!   [`crate::Bgra2Nv12`] (CPU↔CPU API, test/fallback) and
//!   [`crate::nv12_dmabuf::Nv12DmaBuf`] (production DMA-BUF path).
//! - `build_yuv444_pipeline` — BGRA→XYUV packed (4:4:4 8-bit), via
//!   `crate::Yuv444DmaBuf`. **Linux only.**
//! - `build_p010_pipeline` — BGRA→P010 biplanar (4:2:0 10-bit), via
//!   `crate::Bgra2P010DmaBuf` on Linux and the macOS BGRA IOSurface
//!   bridge.
//! - `build_xv30_pipeline` — BGRA→XV30 packed (4:4:4 10-bit), via
//!   `crate::Bgra2Xv30DmaBuf`. **Linux only.**
//!
//! macOS reaches 4:4:4 encoder inputs without these builders, but the
//! host-side BGRA capture path uses the 4:2:0 builders to preserve the
//! desktop in BGRA until the final VideoToolbox input conversion.
//!
//! Per-pipeline bind-group layouts live alongside their builder so a
//! binding-order change in a WGSL file only needs one matching Rust
//! edit.

pub(crate) const SHADER_SRC: &str = include_str!("bgra_to_nv12.wgsl");
// The YUV444 / P010 / XV30 builders only have consumers in the Linux
// DMA-BUF bridges (`yuv444_dmabuf`, `bgra_to_p010_dmabuf`, `xv30_dmabuf`).
// macOS reaches the same encoder inputs via SCK-delivered IOSurfaces
// (`444v`, `x420`, `xf44`) handed straight to VideoToolbox — no
// host-side BGRA→YUV conversion exists or is wanted. Cfg-gating to
// linux keeps the macOS build warning-free without `#[allow(dead_code)]`.
#[cfg(target_os = "linux")]
pub(crate) const YUV444_SHADER_SRC: &str = include_str!("bgra_to_yuv444.wgsl");
#[cfg(target_os = "linux")]
pub(crate) const YUV444P_SHADER_SRC: &str = include_str!("bgra_to_yuv444p.wgsl");
#[cfg(target_os = "linux")]
pub(crate) const P010_SHADER_SRC: &str = include_str!("bgra_to_p010.wgsl");
#[cfg(target_os = "linux")]
pub(crate) const XV30_SHADER_SRC: &str = include_str!("bgra_to_xv30.wgsl");

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
#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
pub(crate) fn build_yuv444p_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bgra_to_yuv444p"),
        source: wgpu::ShaderSource::Wgsl(YUV444P_SHADER_SRC.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgra_to_yuv444p bgl"),
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
                    format: wgpu::TextureFormat::R8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bgra_to_yuv444p pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bgra_to_yuv444p"),
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
/// Storage-feature gating: `R16Unorm` / `Rg16Unorm` as compute storage
/// outputs require the Vulkan format feature flag `STORAGE_IMAGE_BIT`
/// on the chosen `VkFormat`. Some drivers expose 16-bit unorm as
/// sampleable but not as storage-writable; on those, this pipeline
/// would fail at `create_compute_pipeline`. The
/// Linux DMA-BUF callers gate this via [`crate::storable_dmabuf_modifiers`]
/// before constructing the bridge.
#[cfg(target_os = "linux")]
pub(crate) fn build_p010_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bgra_to_p010"),
        source: wgpu::ShaderSource::Wgsl(P010_SHADER_SRC.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgra_to_p010 bgl"),
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

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bgra_to_p010 pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bgra_to_p010"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (pipeline, bgl)
}

/// Build the BGRA → XV30 (packed 10-bit 4:4:4, DRM_FORMAT_XV30 layout)
/// compute pipeline. Single Rgb10a2Unorm storage texture at full
/// resolution; per-pixel byte layout matches what ffmpeg's
/// `vaapi_drm_format_map` recognises as `VA_FOURCC_XV30` /
/// `AV_PIX_FMT_XV30LE` for HEVC Main 4:4:4 10-bit encode input.
///
/// Storage-feature gating: `Rgb10a2Unorm` as a compute storage output
/// requires `STORAGE_IMAGE_BIT` on `VK_FORMAT_A2B10G10R10_UNORM_PACK32`,
/// which isn't in WebGPU's portable storage set. Callers must gate on
/// the adapter's format features before invoking — see
/// [`crate::Bgra2Xv30DmaBuf::new`].
#[cfg(target_os = "linux")]
pub(crate) fn build_xv30_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bgra_to_xv30"),
        source: wgpu::ShaderSource::Wgsl(XV30_SHADER_SRC.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgra_to_xv30 bgl"),
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
                    format: wgpu::TextureFormat::Rgb10a2Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bgra_to_xv30 pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bgra_to_xv30"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (pipeline, bgl)
}
