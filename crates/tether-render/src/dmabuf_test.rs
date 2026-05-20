//! Headless zero-copy round-trip test for the VAAPI->DMA-BUF->wgpu
//! decode path. Encodes a known input, decodes via VAAPI, imports the
//! resulting DMA-BUF as wgpu textures via the production import path,
//! renders through the production fragment shader to an offscreen RGBA
//! target, reads the pixels back, and asserts the regions reconstruct
//! to roughly the input colours.
//!
//! Catches three classes of bug the existing decoder smoke test misses:
//!   * Vulkan refusing the export (modifier mismatch, missing extension)
//!   * Sync races (would manifest as torn/garbled pixels — pixel check
//!     would fail the colour assertions)
//!   * Shader correctness (wrong matrix, wrong limited-range expansion)
//!
//! Marked `#[ignore]` because it needs both VAAPI hardware *and* a
//! Vulkan adapter advertising `VULKAN_EXTERNAL_MEMORY_DMA_BUF` —
//! lavapipe and most CI environments lack the latter. Run on real
//! hardware with:
//!   `cargo test -p tether-render --release -- --ignored dmabuf`

#![cfg(target_os = "linux")]
#![allow(clippy::cast_possible_truncation, clippy::cast_lossless)]

use std::sync::mpsc;

use tether_codec::vaapi::{VaapiDecoder, VaapiEncoder};
use tether_codec::{Decoder, Encoder, Frame as CodecFrame, GpuFrameSource};

use crate::gpu;

/// Two solid colour regions — left half red, right half blue. Chroma
/// 4:2:0 blurs the boundary but the region averages reconstruct to
/// the source colours within the H.264 quantisation noise floor, so
/// the assertion has comfortable headroom.
fn make_test_bgra(w: u32, h: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for _y in 0..h {
        for x in 0..w {
            let (r, g, b) = if x < w / 2 {
                (210u8, 30u8, 30u8)
            } else {
                (30u8, 30u8, 210u8)
            };
            data.extend_from_slice(&[b, g, r, 255]);
        }
    }
    data
}

/// Average RGB over a sub-rectangle of the readback buffer.
fn region_average_rgb(
    rgba: &[u8],
    w: u32,
    x0: u32,
    y0: u32,
    rw: u32,
    rh: u32,
) -> (u8, u8, u8) {
    let mut sum = [0u64; 3];
    let mut count = 0u64;
    for y in y0..y0 + rh {
        for x in x0..x0 + rw {
            let idx = ((y * w + x) * 4) as usize;
            sum[0] += u64::from(rgba[idx]);
            sum[1] += u64::from(rgba[idx + 1]);
            sum[2] += u64::from(rgba[idx + 2]);
            count += 1;
        }
    }
    (
        (sum[0] / count) as u8,
        (sum[1] / count) as u8,
        (sum[2] / count) as u8,
    )
}

/// Initialise wgpu headless with the dma-buf import feature. Returns
/// `None` if no adapter advertises it (no Vulkan, lavapipe, etc.) so
/// the test can SKIP rather than fail in environments that can't
/// exercise zero-copy.
async fn try_init_wgpu_for_dmabuf() -> Option<(wgpu::Device, wgpu::Queue, wgpu::Adapter)> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .ok()?;
    if !adapter
        .features()
        .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
    {
        return None;
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("tether-render dmabuf-roundtrip test"),
            required_features: wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
        .ok()?;
    Some((device, queue, adapter))
}

/// Mirror of the production pipeline (`gpu.rs`) — same shader, same
/// bind group layouts, same vertex layout. Renders to whatever
/// `target_format` the caller passes. We rebuild instead of reusing
/// production code because `GpuState` is tightly coupled to a winit
/// `Window`/`Surface`; the test wants the pipeline without surface
/// management. Drift between this and `gpu.rs` would cause the test
/// to false-pass, but since both use the exact same shader file the
/// realistic drift is in bind group layout — covered by the fact
/// that the production `import_dmabuf_textures` builds bind groups
/// against `yuv_bgl` and the test reuses that same layout below.
struct TestPipeline {
    yuv_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pipeline: wgpu::RenderPipeline,
    scale_bind_group: wgpu::BindGroup,
}

fn build_test_pipeline(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_format: wgpu::TextureFormat,
) -> TestPipeline {
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("test sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let yuv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("test yuv bgl"),
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
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let scale_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("test scale bgl"),
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
        label: Some("test scale uniform"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Identity scale: full NDC quad, no letterboxing.
    let identity: [f32; 4] = [1.0, 1.0, 0.0, 0.0];
    let mut bytes = [0u8; 16];
    for (i, f) in identity.iter().enumerate() {
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&f.to_le_bytes());
    }
    queue.write_buffer(&scale_buffer, 0, &bytes);
    let scale_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test scale bg"),
        layout: &scale_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: scale_buffer.as_entire_binding(),
        }],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("test pipeline layout"),
        bind_group_layouts: &[Some(&yuv_bgl), Some(&scale_bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("test pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
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

    TestPipeline {
        yuv_bgl,
        sampler,
        pipeline,
        scale_bind_group,
    }
}

#[test]
#[ignore = "requires VAAPI HW + Vulkan dma-buf import; run with: cargo test -p tether-render --release -- --ignored dmabuf"]
fn dmabuf_zero_copy_roundtrip_yields_recognisable_pixels() {
    let _ = tracing_subscriber::fmt::try_init();

    let (device, queue, adapter) = match pollster::block_on(try_init_wgpu_for_dmabuf()) {
        Some(t) => t,
        None => {
            eprintln!(
                "SKIPPED: no Vulkan adapter advertising VULKAN_EXTERNAL_MEMORY_DMA_BUF \
                 — this test exercises a zero-copy path that requires real GPU integration"
            );
            return;
        }
    };
    let info = adapter.get_info();
    eprintln!(
        "wgpu adapter: {} (driver: {}, backend: {:?})",
        info.name, info.driver, info.backend
    );

    let w: u32 = 320;
    let h: u32 = 240;
    let input_bgra = make_test_bgra(w, h);

    // Encode the test pattern. First frame is forced IDR; we feed a
    // few frames so the decoder has enough to actually emit one.
    let mut enc = VaapiEncoder::new_bgra(w, h, 30, 4_000).expect("VAAPI encoder");
    let mut packets = Vec::new();
    for t in 0..6i64 {
        packets.extend(
            enc.encode_bgra(&input_bgra, t, t == 0).expect("encode"),
        );
    }

    // Decode and grab the first GPU-resident frame.
    let mut dec = VaapiDecoder::new().expect("VAAPI decoder");
    let mut codec_gpu: Option<tether_codec::GpuFrame> = None;
    for pkt in &packets {
        dec.submit(&pkt.data).expect("submit");
        while let Some(f) = dec.next_frame().expect("next_frame") {
            if let CodecFrame::Gpu(g) = f {
                codec_gpu = Some(g);
                break;
            }
        }
        if codec_gpu.is_some() {
            break;
        }
    }
    let codec_gpu = codec_gpu.expect("decoder produced a Frame::Gpu");
    let (gw, gh, _pts, source, guard) = codec_gpu.into_parts();
    assert_eq!((gw, gh), (w, h));
    let dmabuf = match source {
        GpuFrameSource::DmaBuf(d) => d,
    };

    // Render to a *non-sRGB* RGBA target so the readback gives us
    // linear values that we can compare directly to the input. The
    // shader writes linear RGB; an sRGB target would gamma-encode it.
    let target_format = wgpu::TextureFormat::Rgba8Unorm;
    let pipeline = build_test_pipeline(&device, &queue, target_format);

    // Production import path — the actual code under test.
    let textures = gpu::import_dmabuf_textures(
        &device,
        &pipeline.yuv_bgl,
        &pipeline.sampler,
        &dmabuf,
        gw,
        gh,
        guard,
    )
    .expect("dma-buf import");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen target"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: target_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    // Single-row alignment for copy_texture_to_buffer must be a
    // multiple of COPY_BYTES_PER_ROW_ALIGNMENT (256).
    let unpadded_bpr = u64::from(w * 4);
    let padded_bpr = unpadded_bpr.next_multiple_of(u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT));
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: padded_bpr * u64::from(h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
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
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &textures.bind_group, &[]);
        pass.set_bind_group(1, &pipeline.scale_bind_group, &[]);
        pass.draw(0..6, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr as u32),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let (tx, rx) = mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).expect("send map result");
        });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().expect("map callback").expect("map ok");

    // Strip the per-row padding so the asserts work in (w * 4) stride.
    let mapped = readback.slice(..).get_mapped_range().expect("get mapped range");
    let mut rgba: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h as usize {
        let start = row * padded_bpr as usize;
        let end = start + unpadded_bpr as usize;
        rgba.extend_from_slice(&mapped[start..end]);
    }
    drop(mapped);
    readback.unmap();

    // Sample the inner quarter of each colour region, away from the
    // boundary where chroma blur would skew the average.
    let left = region_average_rgb(&rgba, w, w / 8, h / 4, w / 4, h / 2);
    let right = region_average_rgb(&rgba, w, 5 * w / 8, h / 4, w / 4, h / 2);
    eprintln!("left avg RGB  = {left:?}");
    eprintln!("right avg RGB = {right:?}");

    // Generous bounds: H.264 at 4 Mbps on a 320×240 source preserves
    // these flat colour regions easily, but BT.709 limited-range
    // round-tripping and quantisation can shift channels by ~20.
    assert!(
        left.0 > 130 && left.1 < 80 && left.2 < 80,
        "left region should be reddish; got {left:?}"
    );
    assert!(
        right.2 > 130 && right.0 < 80 && right.1 < 80,
        "right region should be blueish; got {right:?}"
    );
}
