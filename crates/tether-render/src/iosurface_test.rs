//! Headless zero-copy round-trip test for the VideoToolbox →
//! IOSurface → wgpu/Metal decode path. Encodes a known BGRA pattern
//! via the production `VideoToolboxEncoder`, decodes via the
//! production `VideoToolboxDecoder`, imports the resulting IOSurface
//! through the production `gpu::metal::import_iosurface_textures`,
//! renders through the production fragment shader to an offscreen
//! RGBA target, reads pixels back, and asserts that the source
//! regions reconstruct to roughly the input colours.
//!
//! Symmetric to `dmabuf_test.rs` on Linux. Catches the family of bug
//! the existing macOS hardware tests *don't*:
//!
//!   * Missing wgpu feature opt-in for the texture formats the
//!     renderer allocates per bit-depth (commit `942ba53`:
//!     `TEXTURE_FORMAT_16BIT_NORM` was the gap).
//!   * Renderer fourcc accept-set drift against what VT actually
//!     emits (commit `621badc`: VT delivered `'x420'`, renderer
//!     only accepted `'P010'`).
//!   * Sync races / wrong matrix / wrong limited-range expansion in
//!     the 10-bit shader path.
//!
//! Marked `#[ignore]` because it needs real VideoToolbox + a Metal
//! adapter advertising `TEXTURE_FORMAT_16BIT_NORM`. Run on a real
//! Mac with:
//!   `cargo test -p tether-render --release -- --ignored iosurface`
//!
//! Coverage today: HEVC 4:2:0 8-bit and 10-bit. The 4:4:4 paths are
//! gated upstream by VT's silent-downsample limitation (the encoder
//! probe rejects them), so a renderer round-trip for 4:4:4 would
//! never run in a real session — not worth the fixture wiring yet.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_possible_truncation, clippy::cast_lossless)]

use std::sync::mpsc;

use tether_codec::videotoolbox::{VideoToolboxDecoder, VideoToolboxEncoder};
use tether_codec::{Decoder, Encoder, Frame as CodecFrame, GpuFrameSource};
use tether_protocol::control::{ChromaSubsampling, VideoProfile};

use crate::gpu;

/// Two solid colour regions — left half red, right half blue. Same
/// pattern as the Linux dmabuf round-trip test. Chroma sub-sampling
/// blurs the boundary; region averages reconstruct to the source
/// colours within HEVC quantisation noise.
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

/// Initialise wgpu headless for the IOSurface import path. Returns
/// `None` if no Metal adapter is available or it lacks the wgpu
/// features the renderer needs for the requested profile. Both
/// 8-bit and 10-bit allocate textures via `make_yuv_textures`; the
/// 10-bit path additionally needs `TEXTURE_FORMAT_16BIT_NORM` for
/// `R16Unorm` / `Rg16Unorm`. Missing-feature returns `None` so the
/// test SKIPs cleanly on an oddly-configured Metal adapter rather
/// than failing.
async fn try_init_wgpu_for_iosurface(
    bit_depth: u8,
) -> Option<(wgpu::Device, wgpu::Queue, wgpu::Adapter)> {
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
    let mut required = wgpu::Features::empty();
    if bit_depth == 10 {
        if !adapter
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        {
            return None;
        }
        required |= wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("tether-render iosurface-roundtrip test"),
            required_features: required,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
        .ok()?;
    Some((device, queue, adapter))
}

/// Mirror of the production pipeline — same shader, same bind group
/// layouts, same vertex layout. Identical shape to the Linux
/// `dmabuf_test::build_test_pipeline` (intentional: the production
/// fragment shader is platform-independent). Re-built here rather
/// than reusing `GpuState` because the latter is tightly coupled to
/// a winit `Window` / `Surface` and this test wants only the
/// pipeline.
struct TestPipeline {
    yuv_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pipeline: wgpu::RenderPipeline,
    scale_bind_group: wgpu::BindGroup,
    color_params_bind_group: wgpu::BindGroup,
}

fn build_test_pipeline(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_format: wgpu::TextureFormat,
    bit_depth: u8,
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
    let color_params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("test color params bgl"),
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

    // Identity scale: full NDC quad, no letterboxing.
    let scale_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test scale uniform"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
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

    // Color params: [transfer_kind=Srgb(1), range_kind, 0, 0]. The
    // range_kind dispatch is exactly what catches the 8-bit-on-10-bit
    // luma drift the renderer used to have; an 8-bit profile picks
    // RANGE_KIND_LIMITED_8 (0), a 10-bit profile picks
    // RANGE_KIND_LIMITED_10 (1). Pinned in `gpu::range_kind_for`.
    let color_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test color params uniform"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let range_kind: u32 = if bit_depth == 10 { 1 } else { 0 };
    let mut cp_bytes = [0u8; 16];
    cp_bytes[0..4].copy_from_slice(&1u32.to_le_bytes()); // transfer_kind = Srgb
    cp_bytes[4..8].copy_from_slice(&range_kind.to_le_bytes());
    queue.write_buffer(&color_params_buffer, 0, &cp_bytes);
    let color_params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test color params bg"),
        layout: &color_params_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: color_params_buffer.as_entire_binding(),
        }],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("test pipeline layout"),
        bind_group_layouts: &[Some(&yuv_bgl), Some(&scale_bgl), Some(&color_params_bgl)],
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
        color_params_bind_group,
    }
}

/// Drive a single round-trip for the given profile. Encode →
/// decode → import → render → readback. Returns the inner-quarter
/// averages of the left (should-be-red) and right (should-be-blue)
/// halves of the rendered output so the per-profile test can
/// apply its assertions.
///
/// `RUST_BACKTRACE=1` highly recommended on failure — the
/// IOSurface fourcc the decoder produced is in the import path's
/// error message and the test panics on the first mismatch.
fn run_roundtrip(profile: VideoProfile) -> Option<((u8, u8, u8), (u8, u8, u8))> {
    let _ = tracing_subscriber::fmt::try_init();

    let (device, queue, adapter) =
        match pollster::block_on(try_init_wgpu_for_iosurface(profile.bit_depth)) {
            Some(t) => t,
            None => {
                eprintln!(
                    "SKIPPED: no Metal adapter with required features for {profile:?} \
                     (10-bit needs TEXTURE_FORMAT_16BIT_NORM)"
                );
                return None;
            }
        };
    let info = adapter.get_info();
    eprintln!(
        "[{profile:?}] wgpu adapter: {} (driver: {}, backend: {:?})",
        info.name, info.driver, info.backend
    );

    let w: u32 = 320;
    let h: u32 = 240;
    let input_bgra = make_test_bgra(w, h);

    // Encode several frames so the decoder has enough to flush at
    // least one decoded surface. First frame is an IDR.
    let mut enc = VideoToolboxEncoder::new(profile, w, h, 30, 4_000)
        .expect("VT encoder construction must succeed for a probed profile");
    let mut packets = Vec::new();
    for t in 0..6i64 {
        packets.extend(
            enc.encode_bgra(&input_bgra, t, t == 0)
                .expect("encode_bgra"),
        );
    }
    // VT typically buffers the latest one or two frames internally;
    // an explicit flush drains them before we hand the packet stream
    // to the decoder. Without this the test would race on whatever
    // VT happened to emit during the encode loop.
    packets.extend(enc.flush().expect("encoder flush"));

    let mut dec = VideoToolboxDecoder::new(profile.codec).expect("VT decoder construction");
    let mut codec_gpu: Option<tether_codec::GpuFrame> = None;
    for pkt in &packets {
        dec.submit(&pkt.data).expect("decoder submit");
        while let Some(f) = dec.next_frame().expect("decoder next_frame") {
            if let CodecFrame::Gpu(g) = f {
                codec_gpu = Some(g);
                break;
            }
        }
        if codec_gpu.is_some() {
            break;
        }
    }
    if codec_gpu.is_none() {
        // VT decoder buffers — signal EOF so it flushes whatever it's
        // holding. Same hook the probe layer uses.
        dec.signal_eof().expect("decoder signal_eof");
        while let Some(f) = dec.next_frame().expect("decoder next_frame after EOF") {
            if let CodecFrame::Gpu(g) = f {
                codec_gpu = Some(g);
                break;
            }
        }
    }
    let codec_gpu = codec_gpu.expect("decoder must produce at least one Frame::Gpu");
    let (gw, gh, _pts, source, guard) = codec_gpu.into_parts();
    assert_eq!((gw, gh), (w, h), "decoded dims must match encoded dims");
    let iosurface = match source {
        GpuFrameSource::IOSurface(io) => io,
    };
    eprintln!(
        "[{profile:?}] decoded IOSurface fourcc: 0x{:08x}",
        iosurface.pixel_format
    );

    // Render to a non-sRGB RGBA target so the readback gives us
    // linear values comparable to the input. Same convention as the
    // Linux test.
    let target_format = wgpu::TextureFormat::Rgba8Unorm;
    let pipeline = build_test_pipeline(&device, &queue, target_format, profile.bit_depth);

    // Production import path — the actual code under test.
    let textures = gpu::import_iosurface_textures(
        &device,
        &pipeline.yuv_bgl,
        &pipeline.sampler,
        profile.chroma,
        profile.bit_depth,
        &iosurface,
        guard,
    )
    .expect("IOSurface import — this is the path the live bugs lived in");

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
        pass.set_bind_group(2, &pipeline.color_params_bind_group, &[]);
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
        .expect("device poll");
    rx.recv().expect("map callback").expect("map ok");

    let mapped = readback
        .slice(..)
        .get_mapped_range()
        .expect("get mapped range");
    let mut rgba: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h as usize {
        let start = row * padded_bpr as usize;
        let end = start + unpadded_bpr as usize;
        rgba.extend_from_slice(&mapped[start..end]);
    }
    drop(mapped);
    readback.unmap();

    let left = region_average_rgb(&rgba, w, w / 8, h / 4, w / 4, h / 2);
    let right = region_average_rgb(&rgba, w, 5 * w / 8, h / 4, w / 4, h / 2);
    eprintln!("[{profile:?}] left avg RGB  = {left:?}");
    eprintln!("[{profile:?}] right avg RGB = {right:?}");
    Some((left, right))
}

/// HEVC 4:2:0 8-bit (Main). The cheapest cell — every M-series Mac
/// can produce this, the renderer's 8-bit path is the most-exercised
/// configuration in practice, and this lets us assert the test
/// harness itself is correct before trusting the 10-bit cell.
#[test]
#[ignore = "requires macOS + VideoToolbox + Metal; run with: cargo test -p tether-render --release -- --ignored iosurface"]
fn iosurface_zero_copy_roundtrip_hevc_main_8bit() {
    let Some((left, right)) = run_roundtrip(VideoProfile {
        codec: tether_protocol::control::CodecKind::Hevc,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 8,
    }) else {
        return;
    };
    // Generous bounds: HEVC at 4 Mbps on a 320×240 source preserves
    // these flat colour regions easily. BT.709 limited-range
    // round-trip + quantisation can shift channels by ~20.
    assert!(
        left.0 > 130 && left.1 < 80 && left.2 < 80,
        "left region should be reddish; got {left:?}"
    );
    assert!(
        right.2 > 130 && right.0 < 80 && right.1 < 80,
        "right region should be blueish; got {right:?}"
    );
}

/// HEVC 4:2:0 10-bit (Main10). The cell that bit us across three
/// commits this session: the renderer needed
/// `TEXTURE_FORMAT_16BIT_NORM` opt-in (commit `942ba53`), and the
/// IOSurface fourcc accept set needed `'x420'` / `'xf20'`
/// (commit `621badc`). This test would have caught both without
/// needing a live host/client session.
#[test]
#[ignore = "requires macOS + VideoToolbox + Metal Main10; run with: cargo test -p tether-render --release -- --ignored iosurface"]
fn iosurface_zero_copy_roundtrip_hevc_main10() {
    let Some((left, right)) = run_roundtrip(VideoProfile {
        codec: tether_protocol::control::CodecKind::Hevc,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 10,
    }) else {
        return;
    };
    // Same bounds as 8-bit — 10-bit's added precision doesn't change
    // the average colour of a flat region, only how cleanly we hit
    // it. If anything, the 10-bit cell should reconstruct *closer*
    // to the source.
    assert!(
        left.0 > 130 && left.1 < 80 && left.2 < 80,
        "left region should be reddish; got {left:?}"
    );
    assert!(
        right.2 > 130 && right.0 < 80 && right.1 < 80,
        "right region should be blueish; got {right:?}"
    );
}
