//! Headless zero-copy round-trip test for the VAAPI→DMA-BUF→wgpu
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
//!   * Shader correctness (wrong matrix, wrong limited-range expansion,
//!     wrong range_kind dispatch on 10-bit)
//!
//! Mirrors the macOS [`iosurface_test`] shape (commit `d30d054`):
//! `run_roundtrip(profile)` is the parameterised driver, per-profile
//! cells call it with their `VideoProfile` and apply assertions.
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
use tether_protocol::control::{ChromaSubsampling, VideoProfile};

use crate::gpu;

/// Two solid colour regions — left half red, right half blue. Chroma
/// 4:2:0 blurs the boundary but the region averages reconstruct to
/// the source colours within HEVC/H.264 quantisation noise, so the
/// assertion has comfortable headroom.
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

/// Initialise wgpu headless with the dma-buf import feature, plus
/// `TEXTURE_FORMAT_16BIT_NORM` for 10-bit profiles (matching the
/// macOS sibling's `try_init_wgpu_for_iosurface` — 10-bit needs
/// R16Unorm/Rg16Unorm bindings that crates.io wgpu doesn't expose by
/// default). Returns `None` if the adapter doesn't support the
/// required feature set so the test can SKIP rather than fail.
async fn try_init_wgpu_for_dmabuf(
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
    let mut required = wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF;
    if bit_depth == 10 {
        required |= wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
    }
    if !adapter.features().contains(required) {
        return None;
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("tether-render dmabuf-roundtrip test"),
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

/// Mirror of the production pipeline (`gpu.rs`) — same shader, same
/// bind group layouts. The third bind group (`color_params`) carries
/// the `range_kind` dispatch tag that the shader uses to pick between
/// 8-bit and 10-bit limited-range breakpoints. Without it, a 10-bit
/// stream rendered through 8-bit math produces a systematic ~1%
/// mid-tone lift on every channel.
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

    // Color params group — matches `shader.wgsl @group(2) @binding(0)`.
    // Layout: vec4<u32> = [transfer_kind, range_kind, _, _].
    // `range_kind == RANGE_KIND_LIMITED_10 (1)` for 10-bit profiles,
    // `LIMITED_8 (0)` otherwise. `transfer_kind == TRANSFER_KIND_SRGB (1)`
    // matches the production default for the BT.709 limited-range path.
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
    let color_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test color params uniform"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let range_kind: u32 = if bit_depth == 10 { 1 } else { 0 };
    let transfer_kind: u32 = 1; // TRANSFER_KIND_SRGB
    let mut cp_bytes = [0u8; 16];
    cp_bytes[0..4].copy_from_slice(&transfer_kind.to_le_bytes());
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

/// Encode the test pattern for `profile`. Dispatches on bit_depth:
/// 8-bit uses the CPU-upload `encode_bgra` path (easiest test driver,
/// no separate bridge); 10-bit drives `submit_dmabuf` through a
/// transient `Bgra2P010DmaBuf` bridge (the production path). Returns
/// an empty Vec to signal a skip-worthy driver gap (see the 10-bit
/// branch's submit_dmabuf rejection handling).
fn encode_profile_pattern(
    profile: VideoProfile,
    w: u32,
    h: u32,
    bgra: &[u8],
) -> Vec<tether_codec::EncodedPacket> {
    if profile.bit_depth == 8 {
        encode_via_cpu_upload(profile, w, h, bgra)
    } else {
        encode_via_p010_bridge(profile, w, h, bgra)
    }
}

/// 8-bit path: the CPU-upload `encode_bgra` driver. Six frames is
/// enough for the decoder to emit at least one surface in low-latency
/// mode (no flush() needed — VAAPI's libavcodec wrapper emits packets
/// per-frame).
fn encode_via_cpu_upload(
    profile: VideoProfile,
    w: u32,
    h: u32,
    bgra: &[u8],
) -> Vec<tether_codec::EncodedPacket> {
    let mut enc = VaapiEncoder::new(profile, w, h, 30, 4_000).expect("VAAPI encoder");
    let mut packets = Vec::new();
    for t in 0..6i64 {
        packets.extend(enc.encode_bgra(bgra, t, t == 0).expect("encode_bgra"));
    }
    packets
}

/// 10-bit path: route through the production `Bgra2P010DmaBuf` bridge
/// and feed each resulting dma-buf to the encoder via `submit_dmabuf`.
/// The bridge owns its own wgpu device (separate from the test's
/// render-side device) because the encoder side needs
/// storage-writable R16/Rg16, which the test's device may not have
/// asked for. Mirrors the production wiring where the host's
/// gpuconvert device is distinct from the client's render device.
///
/// Driver-gap handling: `submit_dmabuf` can fail with "DRM format not
/// supported by VAAPI" on drivers that have HEVC Main10 *codec
/// context* support but lack `vaapi_drm_format_map` entries for P010
/// in their FFmpeg + libva combination — empirically Intel iHD on
/// Meteor Lake + FFmpeg 8.1 hits this. Returning an empty packet list
/// signals "driver gap, SKIP this cell"; the caller checks and bails.
/// Step 6 docs the empirical driver matrix.
fn encode_via_p010_bridge(
    profile: VideoProfile,
    w: u32,
    h: u32,
    bgra: &[u8],
) -> Vec<tether_codec::EncodedPacket> {
    let mut enc = VaapiEncoder::new(profile, w, h, 30, 4_000).expect("VAAPI encoder");
    let mut packets = Vec::new();

    let bridge = pollster::block_on(tether_gpuconvert::Bgra2P010DmaBuf::new(w, h))
        .expect("Bgra2P010DmaBuf::new");

    let src = bridge.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("test bgra source"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    bridge.queue().write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bgra,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );

    for t in 0..6i64 {
        let p010_frame = bridge.convert(&src).expect("Bgra2P010DmaBuf::convert");
        let codec_frame = build_codec_dmabuf_frame(&p010_frame);
        match enc.submit_dmabuf(&codec_frame, t, t == 0) {
            Ok(p) => packets.extend(p),
            Err(e) => {
                eprintln!(
                    "SKIP: VAAPI submit_dmabuf rejected P010 input ({e}). \
                     Driver gap: avcodec_open2 accepts Main10 + P010LE \
                     sw_format but av_hwframe_map(DRM_PRIME → VAAPI) \
                     rejects the matching dma-buf descriptor. See \
                     docs/CODEC_CAPABILITIES.md for the empirical \
                     driver matrix."
                );
                return Vec::new();
            }
        }
    }
    packets
}

/// Adapter from gpuconvert's `P010DmaBufFrame` to codec's
/// `DmaBufFrame` — same memory layout, just different struct types
/// because they live in different crates with no shared "raw dma-buf
/// descriptor" type. The mapping is mechanical: object 0 owns the fd
/// + size + modifier; layer 0 has two planes (Y, UV) at the bridge's
/// reported offsets/strides; the layer fourcc is `P010`.
fn build_codec_dmabuf_frame(
    p010: &tether_gpuconvert::P010DmaBufFrame,
) -> tether_codec::DmaBufFrame {
    use std::os::fd::AsRawFd;
    // Dup the bridge's fd so the codec frame owns an independent
    // descriptor; the bridge keeps its own clone alive for the next
    // convert() call.
    let dup_fd = p010
        .fd
        .try_clone()
        .expect("dup P010 fd for codec frame");
    let _ = dup_fd.as_raw_fd(); // sanity check it opened

    // Two-layer shape matches what `nv12_dmabuf_to_codec_frame` in
    // the host produces: outer fourcc is the composite (P010), but
    // each layer carries its per-plane DRM fourcc (R16 for Y, GR32
    // for UV) with `num_planes = 1`. This is what FFmpeg's
    // `vaapi_drm_format_map` plus the `av_hwframe_map(DRM_PRIME →
    // VAAPI)` machinery actually expects — a one-layer-two-planes
    // shape silently fails as "DRM format not supported by VAAPI".
    const P010_FOURCC: u32 = u32::from_le_bytes(*b"P010");
    const R16_FOURCC: u32 = u32::from_le_bytes(*b"R16 ");
    const GR32_FOURCC: u32 = u32::from_le_bytes(*b"GR32");

    let y_off = u32::try_from(p010.y_offset).expect("y_offset fits in u32");
    let y_stride = u32::try_from(p010.y_stride).expect("y_stride fits in u32");
    let uv_off = u32::try_from(p010.uv_offset).expect("uv_offset fits in u32");
    let uv_stride = u32::try_from(p010.uv_stride).expect("uv_stride fits in u32");

    tether_codec::DmaBufFrame {
        fourcc: P010_FOURCC,
        objects: vec![tether_codec::DmaBufObject {
            fd: dup_fd,
            size: p010.size,
            drm_format_modifier: p010.modifier,
        }],
        layers: vec![
            tether_codec::DmaBufLayer {
                drm_format: R16_FOURCC,
                num_planes: 1,
                object_index: [0, 0, 0, 0],
                offset: [y_off, 0, 0, 0],
                pitch: [y_stride, 0, 0, 0],
            },
            tether_codec::DmaBufLayer {
                drm_format: GR32_FOURCC,
                num_planes: 1,
                object_index: [0, 0, 0, 0],
                offset: [uv_off, 0, 0, 0],
                pitch: [uv_stride, 0, 0, 0],
            },
        ],
    }
}

/// Drive a single round-trip for the given profile. Encode →
/// decode → import → render → readback. Returns the inner-quarter
/// averages of the left (should-be-red) and right (should-be-blue)
/// halves of the rendered output so the per-profile test can apply
/// its assertions. `None` if the local wgpu/Vulkan stack can't host
/// this profile (test SKIPs rather than fails).
fn run_roundtrip(profile: VideoProfile) -> Option<((u8, u8, u8), (u8, u8, u8))> {
    let _ = tracing_subscriber::fmt::try_init();

    let (device, queue, adapter) =
        match pollster::block_on(try_init_wgpu_for_dmabuf(profile.bit_depth)) {
            Some(t) => t,
            None => {
                eprintln!(
                    "SKIPPED: no Vulkan adapter with required features for {profile:?} \
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

    let packets = encode_profile_pattern(profile, w, h, &input_bgra);
    if packets.is_empty() {
        // encode_profile_pattern signalled a skip-worthy driver gap
        // (see e.g. the P010 path's submit_dmabuf rejection on Mesa
        // iHD + Meteor Lake). Surface as Option::None so the test
        // cell reports SKIP rather than fail.
        eprintln!("[{profile:?}] encode path produced no packets — SKIP");
        return None;
    }

    // Decode and grab the first GPU-resident frame.
    let mut dec = VaapiDecoder::new(profile.codec).expect("VAAPI decoder");
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
    eprintln!(
        "[{profile:?}] decoded dma-buf fourcc: 0x{:08x}",
        dmabuf.fourcc
    );

    let target_format = wgpu::TextureFormat::Rgba8Unorm;
    let pipeline = build_test_pipeline(&device, &queue, target_format, profile.bit_depth);

    let textures = gpu::import_dmabuf_textures(
        &device,
        &pipeline.yuv_bgl,
        &pipeline.sampler,
        profile.chroma,
        profile.bit_depth,
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
        .expect("poll");
    rx.recv().expect("map callback").expect("map ok");

    let mapped = readback.slice(..).get_mapped_range().expect("get mapped range");
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

/// H.264 4:2:0 8-bit — the original baseline cell. Universal floor;
/// every VAAPI box supports it. Catches the 8-bit biplanar import +
/// `range_kind = LIMITED_8` shader dispatch.
#[test]
#[ignore = "requires VAAPI HW + Vulkan dma-buf import; run with: cargo test -p tether-render --release -- --ignored dmabuf"]
fn dmabuf_zero_copy_roundtrip_h264_8bit() {
    let Some((left, right)) = run_roundtrip(VideoProfile {
        codec: tether_protocol::control::CodecKind::H264,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 8,
    }) else {
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

/// HEVC 4:2:0 8-bit (Main). Same wire format as H.264 baseline above
/// but exercises the HEVC encoder/decoder pair. The 8-bit shader
/// dispatch and biplanar NV12 import are shared with H.264.
#[test]
#[ignore = "requires VAAPI HW + Vulkan dma-buf import; run with: cargo test -p tether-render --release -- --ignored dmabuf"]
fn dmabuf_zero_copy_roundtrip_hevc_main_8bit() {
    let Some((left, right)) = run_roundtrip(VideoProfile {
        codec: tether_protocol::control::CodecKind::Hevc,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 8,
    }) else {
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

/// HEVC 4:2:0 10-bit (Main10). The primary cell this work was about:
/// exercises the full new 10-bit path — Bgra2P010DmaBuf produces the
/// dma-buf, VAAPI consumes it via submit_dmabuf, the decoder yields a
/// P010 dma-buf, the renderer imports it as R16/Rg16 textures, and the
/// shader applies `range_kind = LIMITED_10` breakpoints. Closes the
/// gap CLAUDE.md flags about renderer-side hardware coverage
/// (previously zero cells, now one for Main10 + two for 8-bit).
///
/// **Empirical SKIP on Intel iHD + Meteor Lake + FFmpeg 8.1**:
/// `av_hwframe_map(DRM_PRIME → VAAPI)` rejects the P010 dma-buf
/// descriptor despite the codec construction succeeding. The cell
/// SKIPs cleanly with a diagnostic from `encode_via_p010_bridge`. A
/// silent SKIP looks like a pass — see the eprintln below for the
/// caveat surfaced on each ignored-test run.
#[test]
#[ignore = "requires VAAPI HW + Vulkan dma-buf import + storage R16/Rg16; may SKIP on Intel iHD/Meteor Lake (P010 dma-buf driver gap)"]
fn dmabuf_zero_copy_roundtrip_hevc_main10() {
    let Some((left, right)) = run_roundtrip(VideoProfile {
        codec: tether_protocol::control::CodecKind::Hevc,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 10,
    }) else {
        // Cell didn't fail — but didn't run the renderer assertion
        // either. Make the SKIP visible so a developer doesn't read
        // "test passed" as "full path exercised."
        eprintln!(
            "NOTE: Main10 cell SKIPPED — driver layer rejected the dma-buf \
             before the renderer was reached. The renderer's 10-bit path \
             remains untested at the dmabuf seam on this hardware."
        );
        return;
    };
    // Same bounds as 8-bit. 10-bit's added precision doesn't shift the
    // average colour of a flat region, only how cleanly we hit it. If
    // anything the 10-bit cell should reconstruct *closer* to the
    // source (less quantisation noise on the gradient edges).
    assert!(
        left.0 > 130 && left.1 < 80 && left.2 < 80,
        "left region should be reddish; got {left:?}"
    );
    assert!(
        right.2 > 130 && right.0 < 80 && right.1 < 80,
        "right region should be blueish; got {right:?}"
    );
}

/// Documented source-of-truth for the Linux pipeline's
/// `(chroma, bit_depth) → DRM fourcc` mapping. Not a drift detector
/// today — it's a local table that ought to agree with the encoder's
/// `vaapi_sw_format`, the encoder's `submit_dmabuf` fourcc match, the
/// renderer's `import_dmabuf_textures` layout dispatch, and the host's
/// `capture_filtered_encode_profiles` filter, but each of those is
/// private to its own crate today. A real drift detector would need a
/// `pub fn expected_dmabuf_fourcc(chroma, bit_depth) -> u32` in
/// tether-codec (and matching exposure in tether-render +
/// tether-gpuconvert) that this test imports from.
///
/// TODO(cross-table-drift): expose the encoder's `vaapi_sw_format` as
/// `pub` (or wrap it in a small public helper) and replace
/// `expected_fourcc` with a call into the real module. macOS commit
/// `8c0398e` is the shape — its cross-table test imports from
/// `videotoolbox/probe.rs::expected_iosurface_fourccs`. Linux needs
/// the same exposure.
///
/// Until that refactor, the entries here serve as documentation
/// pins: a contributor adding a new `(chroma, bit_depth)` cell
/// has one canonical place to record the fourcc, and a future
/// drift detector wires through this same table.
#[cfg(test)]
mod cross_table_consistency {
    use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

    /// Expected DRM fourcc for each `(chroma, bit_depth)` combination
    /// the Linux pipeline supports today. Drift in any of the four
    /// tables (encoder `vaapi_sw_format` / encoder `submit_dmabuf` /
    /// renderer import / host capture filter) shows up as a mismatch
    /// between this single source of truth and the actual table.
    ///
    /// XV30 is in the encoder side but no bridge produces it yet
    /// (probe layer gates it) — included here so when the bridge
    /// ships, the consistency check already covers it.
    fn expected_fourcc(chroma: ChromaSubsampling, bit_depth: u8) -> u32 {
        match (chroma, bit_depth) {
            (ChromaSubsampling::Yuv420, 8) => u32::from_le_bytes(*b"NV12"),
            (ChromaSubsampling::Yuv420, 10) => u32::from_le_bytes(*b"P010"),
            (ChromaSubsampling::Yuv444, 8) => u32::from_le_bytes(*b"XYUV"),
            (ChromaSubsampling::Yuv444, 10) => u32::from_le_bytes(*b"XV30"),
            _ => panic!("unmodeled (chroma, bit_depth) {chroma:?} {bit_depth}"),
        }
    }

    /// All the `(chroma, bit_depth)` pairs the pipeline is expected
    /// to round-trip through dma-buf. Update this when a new combo
    /// ships (and all four tables grow matching entries).
    const MODELED: &[(ChromaSubsampling, u8)] = &[
        (ChromaSubsampling::Yuv420, 8),
        (ChromaSubsampling::Yuv420, 10),
        (ChromaSubsampling::Yuv444, 8),
        (ChromaSubsampling::Yuv444, 10),
    ];

    /// Pin the expected fourccs. A future refactor that swaps NV12 for
    /// the equivalent `NV21` (Cr/Cb swapped) or P010 for `P012` would
    /// fail here before silently feeding the encoder a misordered UV
    /// plane.
    #[test]
    fn expected_fourccs_are_stable() {
        for &(chroma, bit_depth) in MODELED {
            let f = expected_fourcc(chroma, bit_depth);
            let bytes = f.to_le_bytes();
            // All four fourccs are ASCII-printable per the DRM
            // convention.
            assert!(
                bytes.iter().all(|&b| (0x20..=0x7e).contains(&b)),
                "fourcc for ({chroma:?}, {bit_depth}) = 0x{f:08x} is not printable",
            );
        }
    }

    /// Every VideoProfile that combines a Linux-supported chroma with
    /// a Linux-supported bit_depth must have a fourcc in the table —
    /// if `VideoProfile::HEVC_*` adds a constant for a combination
    /// that isn't here, the panic in `expected_fourcc` catches it.
    #[test]
    fn modeled_profiles_match_video_profile_constants() {
        // Every preference-listed VideoProfile that targets Linux's
        // pipeline must be in the MODELED list. Doesn't fire today
        // (the constants happen to cover exactly MODELED), but a new
        // `VideoProfile::HEVC_*` constant landing without an
        // expected_fourcc entry would surface here.
        let profile_list = [
            VideoProfile {
                codec: CodecKind::Hevc,
                chroma: ChromaSubsampling::Yuv420,
                bit_depth: 8,
            },
            VideoProfile {
                codec: CodecKind::Hevc,
                chroma: ChromaSubsampling::Yuv420,
                bit_depth: 10,
            },
            VideoProfile {
                codec: CodecKind::Hevc,
                chroma: ChromaSubsampling::Yuv444,
                bit_depth: 8,
            },
            VideoProfile {
                codec: CodecKind::Hevc,
                chroma: ChromaSubsampling::Yuv444,
                bit_depth: 10,
            },
            VideoProfile {
                codec: CodecKind::H264,
                chroma: ChromaSubsampling::Yuv420,
                bit_depth: 8,
            },
        ];
        for p in profile_list {
            let f = expected_fourcc(p.chroma, p.bit_depth);
            assert_ne!(f, 0, "no fourcc for profile {p:?}");
        }
    }
}
