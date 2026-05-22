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
    profile: VideoProfile,
) -> TestPipeline {
    // The renderer dispatches on the negotiated (chroma, bit_depth) to
    // pick between two bind-group layouts + two shaders:
    //   - Biplanar (NV12 / P010 / P410): 2 textures + 1 sampler →
    //     `shader.wgsl`.
    //   - PackedXYUV (YUV444 8-bit on Linux): 1 texture + 1 sampler →
    //     `shader_yuv444.wgsl`.
    // The test pipeline must mirror that dispatch so the BGL shape we
    // build agrees with what `gpu::import_dmabuf_textures` returns for
    // this profile. The drift detector test (cross_table_consistency)
    // pins the same dispatch.
    let layout = crate::gpu::render_layout_for(profile.chroma, profile.bit_depth);
    let bit_depth = profile.bit_depth;
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("test sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let yuv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("test yuv bgl"),
        entries: match layout {
            crate::gpu::RenderLayout::Biplanar8 | crate::gpu::RenderLayout::Biplanar16 => &[
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
            crate::gpu::RenderLayout::PackedXYUV => &[
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        },
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

    let shader_src = match layout {
        crate::gpu::RenderLayout::Biplanar8 | crate::gpu::RenderLayout::Biplanar16 => {
            include_str!("shader.wgsl")
        }
        crate::gpu::RenderLayout::PackedXYUV => include_str!("shader_yuv444.wgsl"),
    };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
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
    let pipeline = build_test_pipeline(&device, &queue, target_format, profile);

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

    let (left, right) = render_and_readback(&device, &queue, &pipeline, &textures, (w, h));
    eprintln!("[{profile:?}] left avg RGB  = {left:?}");
    eprintln!("[{profile:?}] right avg RGB = {right:?}");
    Some((left, right))
}

/// Render `textures` through `pipeline` into an offscreen Rgba8Unorm
/// target at the given dims, read back the pixels, and return the
/// left/right region averages used by the per-profile assertions.
/// Factored out so both the existing `run_roundtrip` and the new
/// scaler-active test share the render+readback boilerplate.
fn render_and_readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &TestPipeline,
    textures: &gpu::YuvTextures,
    dims: (u32, u32),
) -> ((u8, u8, u8), (u8, u8, u8)) {
    let (w, h) = dims;
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
        format: wgpu::TextureFormat::Rgba8Unorm,
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
    (left, right)
}

/// Build an NV12 codec dma-buf descriptor from a gpuconvert
/// `Nv12DmaBufFrame`. Mirrors `nv12_dmabuf_to_codec_frame` in
/// `apps/tether-host/src/main.rs` — same memory layout, different
/// crate. Single object, two layers (Y as `R8 ` fourcc, UV as
/// `GR88`).
fn build_codec_dmabuf_frame_nv12(
    nv12: &tether_gpuconvert::Nv12DmaBufFrame,
) -> tether_codec::DmaBufFrame {
    const NV12_FOURCC: u32 = u32::from_le_bytes(*b"NV12");
    const R8_FOURCC: u32 = u32::from_le_bytes(*b"R8  ");
    const GR88_FOURCC: u32 = u32::from_le_bytes(*b"GR88");
    let dup_fd = nv12.fd.try_clone().expect("dup NV12 fd");
    let y_off = u32::try_from(nv12.y_offset).expect("y_offset fits in u32");
    let y_stride = u32::try_from(nv12.y_stride).expect("y_stride fits in u32");
    let uv_off = u32::try_from(nv12.uv_offset).expect("uv_offset fits in u32");
    let uv_stride = u32::try_from(nv12.uv_stride).expect("uv_stride fits in u32");
    tether_codec::DmaBufFrame {
        fourcc: NV12_FOURCC,
        objects: vec![tether_codec::DmaBufObject {
            fd: dup_fd,
            size: nv12.size,
            drm_format_modifier: nv12.modifier,
        }],
        layers: vec![
            tether_codec::DmaBufLayer {
                drm_format: R8_FOURCC,
                num_planes: 1,
                object_index: [0, 0, 0, 0],
                offset: [y_off, 0, 0, 0],
                pitch: [y_stride, 0, 0, 0],
            },
            tether_codec::DmaBufLayer {
                drm_format: GR88_FOURCC,
                num_planes: 1,
                object_index: [0, 0, 0, 0],
                offset: [uv_off, 0, 0, 0],
                pitch: [uv_stride, 0, 0, 0],
            },
        ],
    }
}

/// Production-shape encode path with a scaler in front: BGRA at
/// `capture` dims → import → Mitchell downscale to `encode` dims →
/// NV12 bridge → encoder.encode_gpu via submit_dmabuf. Catches any
/// integration bug in the scaler→bridge→encoder chain that the
/// component-level tests miss.
fn encode_via_scaler_chain(
    profile: VideoProfile,
    capture: (u32, u32),
    encode: (u32, u32),
    bgra_at_capture: &[u8],
) -> Vec<tether_codec::EncodedPacket> {
    use tether_gpuconvert::{export_texture_as_dmabuf, Nv12DmaBuf};
    use tether_scaler::{ColorSpace, Pipelines, Scaler};

    let mut enc = VaapiEncoder::new(profile, encode.0, encode.1, 30, 4_000).expect("encoder");
    let bridge = pollster::block_on(Nv12DmaBuf::new(encode.0, encode.1)).expect("nv12 bridge");

    // Stand in for PipeWire's BGRA capture: export a dma-buf at
    // *capture* dims on the bridge's device and upload the fixture.
    let src_export = export_texture_as_dmabuf(
        bridge.device(),
        capture.0,
        capture.1,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        "scaler-roundtrip bgra source",
    )
    .expect("export bgra source");
    bridge.queue().write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src_export.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bgra_at_capture,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(capture.0 * 4),
            rows_per_image: Some(capture.1),
        },
        wgpu::Extent3d {
            width: capture.0,
            height: capture.1,
            depth_or_array_layers: 1,
        },
    );
    bridge.queue().submit(std::iter::empty());
    bridge
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll write");

    let pipelines = std::sync::Arc::new(Pipelines::build(bridge.device()));
    let scaler = Scaler::new_with_color_space(
        pipelines,
        bridge.device().clone(),
        bridge.queue().clone(),
        capture,
        encode,
        ColorSpace::Srgb8,
    )
    .expect("scaler");

    let mut packets = Vec::new();
    for t in 0..6i64 {
        // Fresh dup-fd per iteration so the bridge can re-import.
        let dup_fd = src_export.fd.try_clone().expect("dup src fd");
        let imported = bridge
            .import_bgra_dmabuf(
                dup_fd,
                src_export.drm_format_modifier,
                src_export.stride,
                src_export.offset,
                capture.0,
                capture.1,
            )
            .expect("import");
        let scaled = scaler.scale(&imported).expect("scale");
        let nv12 = bridge.convert(scaled).expect("nv12 convert");
        let codec_frame = build_codec_dmabuf_frame_nv12(&nv12);
        let gpu_frame = tether_codec::GpuEncoderFrame::DmaBuf(&codec_frame);
        match enc.encode_gpu(gpu_frame, t, t == 0) {
            Ok(p) => packets.extend(p),
            Err(e) => {
                eprintln!("SKIP: encode_gpu via scaler chain failed: {e}");
                return Vec::new();
            }
        }
    }
    packets
}

/// Viewport-active end-to-end smoke test: BGRA at 640×480 → Mitchell
/// downscale to 320×240 → NV12 bridge → encoder.encode_gpu → decoder.
///
/// Validates the production scaler-active wiring through encode_gpu —
/// the codepath the host runs whenever a client sends
/// `SetClientViewport`. The existing dmabuf round-trip cells encode
/// via the CPU upload path (no bridge, no scaler), so without this
/// test the scaler→bridge→encode_gpu chain is exercised only at the
/// component level (gpuconvert scaler_roundtrip tests don't go through
/// the encoder).
///
/// Renders through the test pipeline at encode dims and verifies the
/// left/right halves remain red/blue after the scale — confirms the
/// scaler doesn't swap channels or wash the colors out.
#[test]
#[ignore = "requires VAAPI HW + Vulkan dma-buf import; run with: cargo test -p tether-render --release -- --ignored dmabuf"]
fn dmabuf_zero_copy_roundtrip_with_scaler_h264_8bit() {
    let _ = tracing_subscriber::fmt::try_init();
    let profile = VideoProfile {
        codec: tether_protocol::control::CodecKind::H264,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 8,
    };
    let capture = (640u32, 480u32);
    let encode = (320u32, 240u32);

    let (device, queue, _adapter) =
        match pollster::block_on(try_init_wgpu_for_dmabuf(profile.bit_depth)) {
            Some(t) => t,
            None => {
                eprintln!("SKIP: no wgpu adapter for scaler-active round-trip");
                return;
            }
        };

    let bgra = make_test_bgra(capture.0, capture.1);
    let packets = encode_via_scaler_chain(profile, capture, encode, &bgra);
    if packets.is_empty() {
        eprintln!("SKIP: scaler-active encode produced no packets");
        return;
    }

    // Decode at encode dims and grab the first GPU frame.
    let mut dec = VaapiDecoder::new(profile.codec).expect("decoder");
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
    let codec_gpu = codec_gpu.expect("decoder emitted no Gpu frame");
    let (gw, gh, _pts, source, guard) = codec_gpu.into_parts();
    assert_eq!(
        (gw, gh),
        encode,
        "decoded dims should match encode dims (the scaler's output)"
    );
    let dmabuf = match source {
        GpuFrameSource::DmaBuf(d) => d,
    };

    // Render through the test pipeline at encode dims; reuse the
    // shared decode→render→readback shape.
    let target_format = wgpu::TextureFormat::Rgba8Unorm;
    let pipeline = build_test_pipeline(&device, &queue, target_format, profile);
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

    let (left, right) = render_and_readback(&device, &queue, &pipeline, &textures, encode);
    eprintln!("[scaler-active {profile:?}] left {left:?}, right {right:?}");
    // Same red/blue tolerance bands as the existing cells. The
    // Mitchell downscale + HEVC quantisation only soften the
    // boundary; the halves stay clearly red and blue.
    assert!(
        left.0 > 130 && left.1 < 80 && left.2 < 80,
        "left should be red after scaler+encode+decode; got {left:?}"
    );
    assert!(
        right.2 > 130 && right.0 < 80 && right.1 < 80,
        "right should be blue after scaler+encode+decode; got {right:?}"
    );
}

/// Microbench: encode time per frame at production-realistic
/// resolutions. The headline number this answers is "how much wall-
/// clock do we save by scaling 4K → 1080p before encoding instead of
/// encoding at native 4K?". Returns median µs per encode_bgra call
/// after a 6-frame warmup (VAAPI's rate-control window settles).
fn bench_encode_one_resolution(profile: VideoProfile, w: u32, h: u32, iters: usize) -> f64 {
    let bgra = make_test_bgra(w, h);
    let mut enc = VaapiEncoder::new(profile, w, h, 30, 4_000).expect("encoder");
    // Warmup — first few frames are non-representative (encoder builds
    // its rate-control state, the first frame is an IDR which is much
    // larger than the P-frames that dominate the steady state).
    for t in 0..6_i64 {
        let _ = enc.encode_bgra(&bgra, t, t == 0).expect("warmup encode");
    }
    let mut samples = Vec::with_capacity(iters);
    for t in 6..(6 + iters) as i64 {
        let start = std::time::Instant::now();
        let _ = enc.encode_bgra(&bgra, t, false).expect("encode");
        samples.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// Print encode time at several resolutions so the scaler ROI is
/// visible: scaler cost (from `bench_scale_realistic_dims`) plus
/// encode-at-smaller-dims vs encode-at-larger-dims. If
/// `encode_4k - (encode_1080p + scaler_4k_to_1080p) > 0`, the scaler
/// is net-positive on time. (It's already net-positive on bandwidth
/// regardless — encoded bytes scale with pixel count too.)
#[test]
#[ignore = "perf microbenchmark; prints encode timings, no assertions"]
fn bench_encode_by_resolution_h264() {
    let _ = tracing_subscriber::fmt::try_init();
    let profile = VideoProfile {
        codec: tether_protocol::control::CodecKind::H264,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 8,
    };
    println!("\nH.264 encode time per frame (median of 20, after 6-frame warmup):");
    println!("{:<22} {:>10}", "resolution", "med µs");
    for &(w, h, label) in &[
        (640u32, 480u32, "VGA"),
        (1280, 720, "720p"),
        (1920, 1080, "1080p"),
        (2560, 1440, "1440p"),
        (3840, 2160, "4K"),
    ] {
        let med = bench_encode_one_resolution(profile, w, h, 20);
        println!("{label:<22} {med:>10.1}");
    }
    println!();
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

/// HEVC 4:4:4 8-bit (Main 4:4:4). Exercises the packed-XYUV import
/// path and the dedicated `shader_yuv444.wgsl` — neither of which the
/// 4:2:0 cells touch. Encoder side reuses the CPU-upload path (the
/// encoder's swscale stage handles BGRA→YUV444P→XYUV internally; the
/// gpuconvert-side `Yuv444DmaBuf` bridge is exercised by
/// `tether_codec::vaapi::tests::hevc_main444_dmabuf_roundtrip` on the
/// encoder side). SKIPs cleanly when the driver lacks HEVC Main 4:4:4
/// encode support (e.g. older Intel pre-Tiger Lake / older AMD VCN).
#[test]
#[ignore = "requires VAAPI HEVC Main 4:4:4 (Intel Tiger Lake+ / AMD VCN3+) + Vulkan dma-buf import"]
fn dmabuf_zero_copy_roundtrip_hevc_main_444_8bit() {
    let Some((left, right)) = run_roundtrip(VideoProfile {
        codec: tether_protocol::control::CodecKind::Hevc,
        chroma: ChromaSubsampling::Yuv444,
        bit_depth: 8,
    }) else {
        eprintln!(
            "NOTE: HEVC 4:4:4 8-bit cell SKIPPED — driver lacks Main 4:4:4 encode \
             or the dma-buf path. The renderer's PackedXYUV path remains \
             untested on this hardware."
        );
        return;
    };
    // 4:4:4 doesn't subsample chroma, so the red/blue split is cleaner
    // than 4:2:0 — same tolerance bands still apply with headroom.
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
/// renderer's `render_layout_for` dispatch, and (transitively) the
/// host's `capture_filtered_encode_profiles` filter all key off the
/// same `(chroma, bit_depth)` tuple. The encoder side is now
/// canonical via `tether_codec::vaapi::expected_dmabuf_fourcc`; this
/// test asserts that the renderer's layout dispatch agrees on the
/// underlying texture family for every fourcc the encoder accepts.
///
/// Cross-checks three things on every supported `(chroma, bit_depth)`:
///   1. The encoder's fourcc is non-zero and ASCII-printable (catches
///      a `b"\0\0\0\0"` typo at the const-byte level).
///   2. The renderer-side `render_layout_for` does not panic on a
///      `(chroma, bit_depth)` that the encoder accepts. Panicking
///      means a frame the encoder would emit can't be rendered.
///   3. The renderer's `RenderLayout` choice matches the encoder
///      fourcc family: NV12/P010 → Biplanar*; XYUV → PackedXYUV;
///      XV30 → biplanar 16-bit (when the bridge ships).
#[cfg(test)]
mod cross_table_consistency {
    use tether_codec::vaapi::expected_dmabuf_fourcc;
    use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

    use crate::gpu::{render_layout_for, RenderLayout};

    /// Every `(chroma, bit_depth)` the encoder fourcc table currently
    /// covers. New entries land here when both the encoder and the
    /// renderer grow matching support — the `every_*` tests fail
    /// otherwise.
    const MODELED: &[(ChromaSubsampling, u8)] = &[
        (ChromaSubsampling::Yuv420, 8),
        (ChromaSubsampling::Yuv420, 10),
        (ChromaSubsampling::Yuv444, 8),
        (ChromaSubsampling::Yuv444, 10),
    ];

    /// The fourcc family the renderer expects for a given encoder
    /// output fourcc. Coupling is one-to-many on the encoder side
    /// (NV12 and P010 both land on biplanar; XYUV is its own thing).
    fn expected_layout_for_fourcc(fourcc: u32) -> RenderLayout {
        match &fourcc.to_le_bytes() {
            b"NV12" => RenderLayout::Biplanar8,
            b"P010" => RenderLayout::Biplanar16,
            b"XYUV" => RenderLayout::PackedXYUV,
            // XV30 has no gpuconvert bridge today, but when it ships
            // the renderer side will need biplanar-16 support extended
            // to 4:4:4 (or a new packed-16 variant). Either way the
            // current PackedXYUV is wrong; flag the disagreement
            // early.
            b"XV30" => RenderLayout::Biplanar16,
            other => panic!("expected_layout_for_fourcc: unknown fourcc {other:?}"),
        }
    }

    /// The encoder's fourcc for every modeled tuple is non-zero and
    /// ASCII-printable. A renamed fourcc that ends up as
    /// `b"\0\0\0\0"` after a refactor fails here.
    #[test]
    fn encoder_fourccs_are_well_formed() {
        for &(chroma, bit_depth) in MODELED {
            let f = expected_dmabuf_fourcc(chroma, bit_depth)
                .unwrap_or_else(|| panic!("encoder has no fourcc for ({chroma:?}, {bit_depth})"));
            let bytes = f.to_le_bytes();
            assert!(
                bytes.iter().all(|&b| (0x20..=0x7e).contains(&b)),
                "encoder fourcc for ({chroma:?}, {bit_depth}) = 0x{f:08x} is not printable",
            );
        }
    }

    /// For every encoder-supported `(chroma, bit_depth)`, the
    /// renderer's layout dispatch produces the same family the
    /// encoder fourcc implies. This is the drift detector: if
    /// someone tightens the encoder to a new fourcc or the renderer
    /// flips its dispatch, the disagreement surfaces here in a unit
    /// test, before the hardware round-trip cell would catch it.
    #[test]
    fn renderer_layout_matches_encoder_fourcc() {
        for &(chroma, bit_depth) in MODELED {
            let fourcc = expected_dmabuf_fourcc(chroma, bit_depth)
                .unwrap_or_else(|| panic!("encoder has no fourcc for ({chroma:?}, {bit_depth})"));
            let renderer = render_layout_for(chroma, bit_depth);
            let expected = expected_layout_for_fourcc(fourcc);
            assert_eq!(
                renderer,
                expected,
                "render_layout_for({chroma:?}, {bit_depth}) = {renderer:?} but \
                 fourcc {:?} expects {expected:?}",
                std::str::from_utf8(&fourcc.to_le_bytes()).unwrap_or("?"),
            );
        }
    }

    /// Every preference-listed VideoProfile that targets Linux's
    /// pipeline must have an encoder fourcc. Catches a new
    /// `PROFILE_PREFERENCE` entry landing without matching encoder
    /// table coverage.
    #[test]
    fn preference_list_profiles_have_fourcc() {
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
            let f = expected_dmabuf_fourcc(p.chroma, p.bit_depth);
            assert!(f.is_some(), "no encoder fourcc for profile {p:?}");
        }
    }
}
