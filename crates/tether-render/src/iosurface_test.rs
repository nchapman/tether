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
//! Coverage today: HEVC 4:2:0 8-bit, HEVC 4:2:0 10-bit (encode →
//! decode → render), and HEVC 4:4:4 8-bit + 10-bit (fixture-decode →
//! render). VideoToolbox has no Main444 encode path so the 4:4:4
//! cells can't go through the local encoder, but a Linux→Mac session
//! *does* negotiate 4:4:4 (M-series VT decodes Main444 to a `'444v'`
//! / `'xf44'` NV24 IOSurface) — the import side is real production
//! code that the 4:2:0 cells don't reach because NV24 has full-res UV
//! stride. We drive the import path from a bundled HEVC 4:4:4 IDR
//! fixture (128×128 grey from `tether-probe`).

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

/// Decode-only render path. Used by the 4:4:4 cells: VideoToolbox has
/// no Main444 encode, but it decodes Main444 fine to an NV24 IOSurface,
/// so we feed it a Linux-encoded HEVC 4:4:4 IDR fixture and exercise
/// the import + render half. The fixture is grey, so the assertion is
/// "rendered output is approximately neutral grey" — R≈G≈B with both
/// near a sane luminance midpoint — rather than the red/blue check the
/// encode-roundtrip cells use.
fn run_fixture_render(
    profile: VideoProfile,
    bitstream: &[u8],
) -> Option<((u8, u8, u8), (u8, u8, u8))> {
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

    // Fixture geometry is fixed at 128×128 by the probe regenerator.
    let w: u32 = 128;
    let h: u32 = 128;

    let mut dec = VideoToolboxDecoder::new(profile.codec).expect("VT decoder construction");
    dec.submit(bitstream).expect("decoder submit fixture IDR");
    dec.signal_eof().expect("decoder signal_eof");
    let mut codec_gpu: Option<tether_codec::GpuFrame> = None;
    while let Some(f) = dec.next_frame().expect("decoder next_frame after EOF") {
        if let CodecFrame::Gpu(g) = f {
            codec_gpu = Some(g);
            break;
        }
    }
    let codec_gpu = codec_gpu.expect("decoder must produce at least one Frame::Gpu");
    let (gw, gh, _pts, source, guard) = codec_gpu.into_parts();
    assert_eq!((gw, gh), (w, h), "decoded dims must match fixture dims");
    let iosurface = match source {
        GpuFrameSource::IOSurface(io) => io,
    };
    eprintln!(
        "[{profile:?}] decoded IOSurface fourcc: 0x{:08x}",
        iosurface.pixel_format
    );

    let target_format = wgpu::TextureFormat::Rgba8Unorm;
    let pipeline = build_test_pipeline(&device, &queue, target_format, profile.bit_depth);

    let textures = gpu::import_iosurface_textures(
        &device,
        &pipeline.yuv_bgl,
        &pipeline.sampler,
        profile.chroma,
        profile.bit_depth,
        &iosurface,
        guard,
    )
    .expect("IOSurface import for 4:4:4 (NV24 / full-res UV)");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen target (fixture)"),
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
        label: Some("readback (fixture)"),
        size: padded_bpr * u64::from(h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fixture test encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fixture test pass"),
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

/// Assert that a region average looks like neutral grey near the
/// fixture's input level. The probe fixtures are generated from
/// ffmpeg's `color=c=gray` filter, which produces sRGB (128, 128, 128)
/// — Y'=126 in BT.709 limited-range, Cb=Cr=128. After our shader's
/// limited-range expansion this should reconstruct to roughly
/// RGB(128, 128, 128) per channel.
///
/// Tolerances:
///   * `spread <= 24` rejects a wrong-matrix bug that lands a colour
///     cast (high G, low R/B; or chroma swap producing pink/cyan).
///   * `avg ∈ 90..=170` rejects "stuck black" / "stuck white" import
///     failures and gross wrong-range expansion. Loose enough to
///     absorb HEVC quantisation noise on a 128×128 source.
fn assert_grey(label: &str, rgb: (u8, u8, u8)) {
    let (r, g, b) = rgb;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let spread = max - min;
    assert!(
        spread <= 24,
        "{label}: R/G/B spread too wide for grey input — got {rgb:?} (spread {spread})"
    );
    let avg = (u32::from(r) + u32::from(g) + u32::from(b)) / 3;
    assert!(
        (90..=170).contains(&avg),
        "{label}: average channel value {avg} not near grey midpoint (expected ~128) — got {rgb:?}"
    );
}

/// HEVC 4:4:4 8-bit (Main 4:4:4). Renderer-side coverage for the
/// Linux-host → Mac-client path: NV24 IOSurface fourcc `'444v'`,
/// full-res UV stride, biplanar path with the same R8 Y + Rg8 UV
/// shader as NV12. No local encode — VT can't produce Main444 — so
/// the fixture is a Linux-encoded HEVC 4:4:4 IDR (128×128 grey).
#[test]
#[ignore = "requires macOS + VideoToolbox + Metal Main444; run with: cargo test -p tether-render --release -- --ignored iosurface"]
fn iosurface_zero_copy_roundtrip_hevc_main_444_8bit() {
    const FIXTURE: &[u8] =
        include_bytes!("../../tether-probe/fixtures/probe/hevc_yuv444_8bit.idr");
    let Some((left, right)) = run_fixture_render(
        VideoProfile {
            codec: tether_protocol::control::CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv444,
            bit_depth: 8,
        },
        FIXTURE,
    ) else {
        return;
    };
    assert_grey("4:4:4 8-bit left", left);
    assert_grey("4:4:4 8-bit right", right);
}

/// HEVC 4:4:4 10-bit (Main 4:4:4 10). The matrix cell that depends on
/// both `TEXTURE_FORMAT_16BIT_NORM` (R16/Rg16) and the full-res UV
/// stride of NV24. Same fixture-decode path as the 8-bit cell.
#[test]
#[ignore = "requires macOS + VideoToolbox + Metal Main444 10-bit; run with: cargo test -p tether-render --release -- --ignored iosurface"]
fn iosurface_zero_copy_roundtrip_hevc_main_444_10bit() {
    const FIXTURE: &[u8] =
        include_bytes!("../../tether-probe/fixtures/probe/hevc_yuv444_10bit.idr");
    let Some((left, right)) = run_fixture_render(
        VideoProfile {
            codec: tether_protocol::control::CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv444,
            bit_depth: 10,
        },
        FIXTURE,
    ) else {
        return;
    };
    assert_grey("4:4:4 10-bit left", left);
    assert_grey("4:4:4 10-bit right", right);
}

/// Host-scaler round-trip: encode BGRA at capture dims → decode →
/// route through the production `Nv12IOSurfaceBridge` → render the
/// downscaled IOSurface. This exercises every layer Stage 3 added:
/// IOSurface plane import (read-only), YUV-plane scaler (Y + UV with
/// cosited chroma siting), destination IOSurface allocation, and
/// colorimetry-attachment copy from source to destination.
///
/// Returns the left/right region averages from the rendered output at
/// `dst_dims`, sampled from the same 1/8–3/8 and 5/8–7/8 columns as
/// the no-scaling cells.
#[cfg(target_os = "macos")]
fn run_host_scaler_roundtrip(
    profile: VideoProfile,
    src_dims: (u32, u32),
    dst_dims: (u32, u32),
) -> Option<((u8, u8, u8), (u8, u8, u8))> {
    use tether_gpuconvert::nv12_iosurface::Nv12IOSurfaceBridge;

    let _ = tracing_subscriber::fmt::try_init();

    // The renderer device opt-ins (16BIT_NORM for 10-bit) cover
    // *renderer* texture allocation. The bridge additionally needs
    // TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES for its R8/Rg8 storage.
    // We build one device that satisfies *both* sets so the imported
    // source-IOSurface textures (renderer-side), the bridge's scaler
    // pipelines, and the rendered destination all live on one Metal
    // device — same constraint the production host pipeline lives
    // under.
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let have = adapter.features();
    if !have.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
        eprintln!(
            "SKIPPED: adapter does not advertise \
             TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES (host bridge requirement) — \
             features = {have:?}"
        );
        return None;
    }
    let mut required = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    if profile.bit_depth == 10 {
        if !have.contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM) {
            eprintln!(
                "SKIPPED: 10-bit profile needs TEXTURE_FORMAT_16BIT_NORM and adapter \
                 does not advertise it"
            );
            return None;
        }
        required |= wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("tether-render iosurface host-scaler roundtrip test"),
        required_features: required,
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    }))
    .ok()?;
    eprintln!(
        "[{profile:?}] host-scaler roundtrip {}x{} -> {}x{}",
        src_dims.0, src_dims.1, dst_dims.0, dst_dims.1
    );

    // === 1) encode + decode at src_dims to produce a representative
    // NV12 IOSurface (what SCK would deliver on the host) ===
    let (sw, sh) = src_dims;
    let input_bgra = make_test_bgra(sw, sh);
    let mut enc = VideoToolboxEncoder::new(profile, sw, sh, 30, 4_000)
        .expect("VT encoder construction");
    let mut packets = Vec::new();
    for t in 0..6i64 {
        packets.extend(enc.encode_bgra(&input_bgra, t, t == 0).expect("encode_bgra"));
    }
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
    // Hard-assert the decoder reproduced the encoder's dims. If VT
    // ever pads or crops, the bridge constructor below would build
    // with wrong src_dims and panic with an opaque error — better to
    // fail here with a clear message.
    assert_eq!((gw, gh), src_dims, "decoded src dims must match encoded src dims");
    let src_iosurface = match source {
        GpuFrameSource::IOSurface(io) => io,
    };
    eprintln!(
        "[{profile:?}] decoded src IOSurface fourcc: 0x{:08x}",
        src_iosurface.pixel_format
    );

    // === 2) drive the production bridge ===
    let bridge = Nv12IOSurfaceBridge::new(
        device.clone(),
        queue.clone(),
        src_dims,
        dst_dims,
        src_iosurface.pixel_format,
    )
    .expect("Nv12IOSurfaceBridge::new");
    let pooled = bridge
        .scale_to_iosurface(&src_iosurface)
        .expect("scale_to_iosurface");
    // Once we have the pooled destination, the source IOSurface's
    // lifetime guard can drop. The bridge's `scale_to_iosurface`
    // already called `queue.submit`, so the GPU passes against the
    // source's imported planes are queued — on M-series unified
    // memory that's the load-bearing fence (no PCIe DMA to wait
    // on). A discrete-GPU path with a separate IOSurface backing
    // store would need a `device.poll(wait)` here instead.
    drop(guard);

    // === 3) render the downscaled IOSurface and read back ===
    let (dw, dh) = dst_dims;
    let target_format = wgpu::TextureFormat::Rgba8Unorm;
    let pipeline = build_test_pipeline(&device, &queue, target_format, profile.bit_depth);

    // The pooled frame's `surface` aliases the bridge's owning slot;
    // the textures cloned in the renderer's import path retain the
    // IOSurface internally so the pooled handle can drop after import.
    let textures = gpu::import_iosurface_textures(
        &device,
        &pipeline.yuv_bgl,
        &pipeline.sampler,
        profile.chroma,
        profile.bit_depth,
        &pooled.frame,
        // No decoder-guard for the bridge-output IOSurface; the
        // pooled-slot handle below keeps it alive. Pass a no-op
        // guard (a unit payload) so the production import signature
        // stays the same.
        tether_codec::GpuFrameGuard::new(()),
    )
    .expect("import_iosurface_textures (host-scaler dst)");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen target"),
        size: wgpu::Extent3d {
            width: dw,
            height: dh,
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
    let unpadded_bpr = u64::from(dw * 4);
    let padded_bpr = unpadded_bpr.next_multiple_of(u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT));
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: padded_bpr * u64::from(dh),
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
                rows_per_image: Some(dh),
            },
        },
        wgpu::Extent3d {
            width: dw,
            height: dh,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    let (tx, rx) = mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).expect("send"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    rx.recv().expect("map callback").expect("map ok");
    let mapped = readback.slice(..).get_mapped_range().expect("range");
    let mut rgba: Vec<u8> = Vec::with_capacity((dw * dh * 4) as usize);
    for row in 0..dh as usize {
        let start = row * padded_bpr as usize;
        let end = start + unpadded_bpr as usize;
        rgba.extend_from_slice(&mapped[start..end]);
    }
    drop(mapped);
    readback.unmap();

    // The pooled handle stays alive throughout the render; dropping
    // it here returns the slot to the bridge's pool.
    drop(pooled);
    drop(textures);
    drop(bridge);

    let left = region_average_rgb(&rgba, dw, dw / 8, dh / 4, dw / 4, dh / 2);
    let right = region_average_rgb(&rgba, dw, 5 * dw / 8, dh / 4, dw / 4, dh / 2);
    eprintln!("[{profile:?}] dst left avg RGB  = {left:?}");
    eprintln!("[{profile:?}] dst right avg RGB = {right:?}");
    Some((left, right))
}

/// HEVC 4:2:0 8-bit, host-scaler round-trip 640×480 → 320×240. The
/// macOS analog of Linux's `h264_8bit_host_scaler` dmabuf cell.
/// End-to-end smoke check: encode → decode → bridge → render must
/// produce reddish-on-left + blueish-on-right region averages,
/// matching the input pattern. A wrong color matrix, swapped planes,
/// black output, or a bridge that drops every frame would all fail
/// here.
///
/// Does NOT independently verify the cosited chroma-siting math —
/// at integer 2× downscale the scale-aware `-(scale - 1) * 0.5`
/// formula and the constant `-0.5` collapse to the same value, so a
/// siting-only regression isn't observable through region averages.
/// That math is guarded by
/// `tether-scaler::tests::hardware::uv_chroma_siting_no_half_pixel_shift`
/// which measures UV centroid drift directly at the scaler level.
#[test]
#[ignore = "requires macOS + VideoToolbox + Metal + TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES; run with: cargo test -p tether-render --release -- --ignored iosurface_host_scaler"]
#[cfg(target_os = "macos")]
fn iosurface_host_scaler_hevc_8bit_downscale() {
    let Some((left, right)) = run_host_scaler_roundtrip(
        VideoProfile {
            codec: tether_protocol::control::CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 8,
        },
        (640, 480),
        (320, 240),
    ) else {
        return;
    };
    assert!(
        left.0 > 130 && left.1 < 80 && left.2 < 80,
        "left region should be reddish; got {left:?}"
    );
    assert!(
        right.2 > 130 && right.0 < 80 && right.1 < 80,
        "right region should be blueish; got {right:?}"
    );
}

/// HEVC 4:2:0 8-bit, host-scaler round-trip 1920×1080 → 1280×720.
/// 1.5× downscale — a non-integer ratio where the scale-aware
/// `-(scale - 1) * 0.5` correction (-0.25 src-pixels here) diverges
/// from the simpler-but-wrong constant `-0.5`. The seam-region
/// assertions below sample columns adjacent to the red/blue
/// boundary; a constant-offset regression shifts the UV plane by
/// ~0.33 dst luma pixels, enough to bleed the opposite colour into
/// the seam averages even though the wide-region averages stay
/// dominantly red and blue.
///
/// (The scaler-level guard is still
/// `uv_chroma_siting_no_half_pixel_shift`, which measures centroid
/// drift directly. This cell is an end-to-end cross-check that the
/// full pipeline preserves siting through encode/decode/render too.)
#[test]
#[ignore = "requires macOS + VideoToolbox + Metal + TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES; run with: cargo test -p tether-render --release -- --ignored iosurface_host_scaler"]
#[cfg(target_os = "macos")]
fn iosurface_host_scaler_hevc_8bit_nonintegral_downscale() {
    let Some((left, right)) = run_host_scaler_roundtrip(
        VideoProfile {
            codec: tether_protocol::control::CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 8,
        },
        (1920, 1080),
        (1280, 720),
    ) else {
        return;
    };
    assert!(
        left.0 > 130 && left.1 < 80 && left.2 < 80,
        "left region should be reddish; got {left:?}"
    );
    assert!(
        right.2 > 130 && right.0 < 80 && right.1 < 80,
        "right region should be blueish; got {right:?}"
    );
}
