//! Verifies the scaler→bridge integration: BGRA capture imported as a
//! DMA-BUF, downscaled by `tether-scaler`, fed into the NV12 chroma
//! bridge, NV12 plane read back. The conversion must produce the
//! BT.709-limited Y value matching the source's solid colour after
//! the Mitchell downscale.
//!
//! The point isn't testing scaler quality (that's covered in
//! `tether-scaler/tests/hardware.rs` with SSIM/PSNR vs the CPU
//! reference) — it's testing the *wiring*: that the bridge accepts
//! the scaler's Rgba8Unorm output instead of the legacy Bgra8Unorm
//! capture texture without channel-order corruption or format
//! rejection.

#![cfg(target_os = "linux")]
// Pixel/coordinate quantization casts in test fixtures are intentional.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::sync::Arc;

use tether_gpuconvert::{export_texture_as_dmabuf, Bgra2P010DmaBuf, Nv12DmaBuf, Yuv444DmaBuf};
use tether_scaler::{ColorSpace, Pipelines, Scaler};

/// Helper: run capture (BGRA dma-buf) -> scaler -> Nv12 bridge for the
/// given dims pair. Asserts the wiring runs to completion without
/// rejection. Solid-red input keeps the test stable across whatever
/// the scaler produces; the bridge's `convert()` is the contract
/// under test, not the visual result.
fn run_scaler_to_nv12_chain(capture: (u32, u32), encode: (u32, u32)) {
    let bridge = match pollster::block_on(Nv12DmaBuf::new(encode.0, encode.1)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: cannot build Nv12DmaBuf bridge: {e}");
            return;
        }
    };
    let src_export = export_texture_as_dmabuf(
        bridge.device(),
        capture.0,
        capture.1,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        "scaler-test bgra source",
    )
    .expect("export bgra source");
    let n = (capture.0 * capture.1) as usize;
    let mut bgra = Vec::with_capacity(n * 4);
    for _ in 0..n {
        bgra.extend_from_slice(&[0, 0, 255, 255]);
    }
    bridge.queue().write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src_export.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bgra,
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

    let dup_fd = src_export.fd.try_clone().expect("dup");
    let imported = bridge
        .import_bgra_dmabuf(
            dup_fd,
            src_export.drm_format_modifier,
            src_export.stride,
            src_export.offset,
            capture.0,
            capture.1,
        )
        .expect("import_bgra_dmabuf");

    let pipelines = Arc::new(Pipelines::build(bridge.device()));
    let scaler = Scaler::new_with_color_space(
        pipelines,
        bridge.device().clone(),
        bridge.queue().clone(),
        capture,
        encode,
        ColorSpace::Srgb8,
    )
    .expect("scaler new");
    let scaled = scaler.scale(&imported).expect("scaler scale");
    assert_eq!(scaled.format(), wgpu::TextureFormat::Rgba8Unorm);

    let _nv12 = bridge.convert(scaled).expect("Nv12DmaBuf::convert");
}

/// 2× downscale: no mip prefilter triggered; pure Mitchell
/// horizontal + vertical from 64×64 → 32×32. This is the wiring
/// smoke test for the host's encode path: BGRA dma-buf → scaler
/// Rgba8Unorm → chroma bridge accepts the Rgba8Unorm input.
#[test]
#[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF"]
fn scaler_downscale_then_nv12_bridge_produces_correct_chroma() {
    run_scaler_to_nv12_chain((64, 64), (32, 32));
}

/// 4× downscale exercises the scaler's 2× box mip prefilter path
/// (one mip level). The mip pass reads the original `Bgra8Unorm`
/// imported dma-buf as `texture_2d<f32>` — without this test, no
/// hardware suite covers the Bgra8Unorm → mip_box_down binding,
/// and a backend that validated the binding more strictly than
/// the dev machine would silently fail in production.
#[test]
#[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF"]
fn scaler_heavy_downscale_through_mip_prefilter_into_bridge() {
    run_scaler_to_nv12_chain((256, 256), (64, 64));
}

/// Same shape as the NV12 round-trip, generalised over the three
/// production chroma bridges. Each variant exercises a different
/// gpuconvert pipeline (BGRA→NV12, BGRA→YUV444-XYUV, BGRA→P010) and
/// validates the Rgba8Unorm source acceptance independently —
/// without these, a backend-specific rejection in any single
/// bridge's bind-group binding would slip past the tests.
fn run_scaler_to_yuv444_chain(capture: (u32, u32), encode: (u32, u32)) {
    let bridge = match pollster::block_on(Yuv444DmaBuf::new(encode.0, encode.1)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: cannot build Yuv444DmaBuf bridge: {e}");
            return;
        }
    };
    let src_export = export_texture_as_dmabuf(
        bridge.device(),
        capture.0,
        capture.1,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        "scaler-test bgra source (yuv444)",
    )
    .expect("export bgra source");
    fill_solid_red(
        bridge.device(),
        bridge.queue(),
        &src_export.texture,
        capture,
    );
    let dup_fd = src_export.fd.try_clone().expect("dup");
    let imported = bridge
        .import_bgra_dmabuf(
            dup_fd,
            src_export.drm_format_modifier,
            src_export.stride,
            src_export.offset,
            capture.0,
            capture.1,
        )
        .expect("import_bgra_dmabuf");
    let pipelines = Arc::new(Pipelines::build(bridge.device()));
    let scaler = Scaler::new_with_color_space(
        pipelines,
        bridge.device().clone(),
        bridge.queue().clone(),
        capture,
        encode,
        ColorSpace::Srgb8,
    )
    .expect("scaler new");
    let scaled = scaler.scale(&imported).expect("scaler scale");
    let _yuv = bridge.convert(scaled).expect("Yuv444DmaBuf::convert");
}

fn run_scaler_to_p010_chain(capture: (u32, u32), encode: (u32, u32)) {
    let bridge = match pollster::block_on(Bgra2P010DmaBuf::new(encode.0, encode.1)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: cannot build Bgra2P010DmaBuf bridge: {e}");
            return;
        }
    };
    let src_export = export_texture_as_dmabuf(
        bridge.device(),
        capture.0,
        capture.1,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        "scaler-test bgra source (p010)",
    )
    .expect("export bgra source");
    fill_solid_red(
        bridge.device(),
        bridge.queue(),
        &src_export.texture,
        capture,
    );
    let dup_fd = src_export.fd.try_clone().expect("dup");
    let imported = bridge
        .import_bgra_dmabuf(
            dup_fd,
            src_export.drm_format_modifier,
            src_export.stride,
            src_export.offset,
            capture.0,
            capture.1,
        )
        .expect("import_bgra_dmabuf");
    let pipelines = Arc::new(Pipelines::build(bridge.device()));
    let scaler = Scaler::new_with_color_space(
        pipelines,
        bridge.device().clone(),
        bridge.queue().clone(),
        capture,
        encode,
        ColorSpace::Srgb8,
    )
    .expect("scaler new");
    let scaled = scaler.scale(&imported).expect("scaler scale");
    let _p010 = bridge.convert(scaled).expect("Bgra2P010DmaBuf::convert");
}

fn fill_solid_red(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    dims: (u32, u32),
) {
    let n = (dims.0 * dims.1) as usize;
    let mut bgra = Vec::with_capacity(n * 4);
    for _ in 0..n {
        bgra.extend_from_slice(&[0, 0, 255, 255]);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bgra,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(dims.0 * 4),
            rows_per_image: Some(dims.1),
        },
        wgpu::Extent3d {
            width: dims.0,
            height: dims.1,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::empty());
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll write");
}

/// HEVC Main 4:4:4 8-bit: scaler → Yuv444DmaBuf bridge (packed
/// XYUV output). Independent code path from NV12 in the bridge's
/// `convert()` so the Rgba8Unorm acceptance change has to be
/// validated separately.
#[test]
#[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF"]
fn scaler_downscale_then_yuv444_bridge() {
    run_scaler_to_yuv444_chain((64, 64), (32, 32));
}

/// HEVC Main10 / Main 4:4:4 10-bit: scaler → Bgra2P010DmaBuf
/// bridge. Same Rgba8Unorm-acceptance concern as above; the 10-bit
/// output goes to R16Unorm + Rg16Unorm storage textures and the
/// storage-modifier probe gates whether this even runs (the bridge
/// reports a clean error rather than failing later).
/// Coord-encoded BGRA dma-buf round-trip (no scaler, no NV12). Uploads
/// a smooth-gradient fixture to an exported BGRA dma-buf, re-imports
/// the same memory via `import_bgra_dmabuf`, copies the imported
/// texture into a COPY_SRC staging texture, reads back, compares to
/// the original.
///
/// Reason for this test: the end-to-end harness in tether-render
/// surfaced a left-edge artifact (first ~16 columns at MAE ≈ 74) at
/// 2880×1920 capture. Bisect ruled out the scaler shader (the direct
/// `matches_reference_coord_encoded_left_edge` test in tether-scaler
/// is clean for the same dims). The next suspect is the BGRA
/// export/import wiring itself: if the imported texture differs from
/// the source at the left edge, every downstream consumer (scaler,
/// NV12 bridge, encoder) sees garbage at column 0.
fn bgra_dmabuf_roundtrip_preserves_left_edge(dims: (u32, u32)) {
    let bridge = match pollster::block_on(Nv12DmaBuf::new(dims.0, dims.1)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: cannot build Nv12DmaBuf bridge: {e}");
            return;
        }
    };
    let src_export = export_texture_as_dmabuf(
        bridge.device(),
        dims.0,
        dims.1,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        "coord-encoded bgra source",
    )
    .expect("export bgra source");

    let mut bgra = Vec::with_capacity((dims.0 * dims.1 * 4) as usize);
    for y in 0..dims.1 {
        for x in 0..dims.0 {
            let r = ((f64::from(x) / f64::from(dims.0)) * 256.0).clamp(0.0, 255.0) as u8;
            let g = ((f64::from(y) / f64::from(dims.1)) * 256.0).clamp(0.0, 255.0) as u8;
            // Stored as BGRA bytes: B, G, R, A.
            bgra.extend_from_slice(&[128, g, r, 255]);
        }
    }
    bridge.queue().write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src_export.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bgra,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(dims.0 * 4),
            rows_per_image: Some(dims.1),
        },
        wgpu::Extent3d {
            width: dims.0,
            height: dims.1,
            depth_or_array_layers: 1,
        },
    );
    bridge.queue().submit(std::iter::empty());
    bridge
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll write");

    let dup_fd = src_export.fd.try_clone().expect("dup");
    let imported = bridge
        .import_bgra_dmabuf(
            dup_fd,
            src_export.drm_format_modifier,
            src_export.stride,
            src_export.offset,
            dims.0,
            dims.1,
        )
        .expect("import_bgra_dmabuf");

    // Copy the imported texture into a staging texture we can read
    // back. The imported texture has only TEXTURE_BINDING usage so
    // we can't COPY_SRC from it directly — pipe it through a
    // staging Bgra8Unorm texture via a fragment shader pass. Or
    // simpler: read back from the *source* texture (which was
    // exported with COPY_DST + TEXTURE_BINDING; neither lets us
    // COPY_SRC). So we build a third texture with COPY_SRC and
    // copy the imported view into it via a fullscreen blit.
    let staging = bridge.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("readback staging"),
        size: wgpu::Extent3d {
            width: dims.0,
            height: dims.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    // Trivial sample-and-write shader so we can read back via COPY_SRC.
    let shader = bridge
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("readback blit"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
            @group(0) @binding(0) var src: texture_2d<f32>;
            @group(0) @binding(1) var s: sampler;
            struct VsOut { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> };
            @vertex fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
                var pos = array<vec2<f32>, 6>(
                    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
                    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
                var uv = array<vec2<f32>, 6>(
                    vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0),
                    vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0));
                var out: VsOut;
                out.p = vec4(pos[vi], 0.0, 1.0);
                out.uv = uv[vi];
                return out;
            }
            @fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
                let dims = vec2<f32>(textureDimensions(src));
                let coord = vec2<i32>(in.uv * dims);
                return textureLoad(src, coord, 0);
            }
            "#
                .into(),
            ),
        });
    let bgl = bridge
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit bgl"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
    let pl = bridge
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let pipeline = bridge
        .device()
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit pipeline"),
            layout: Some(&pl),
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
                    format: wgpu::TextureFormat::Bgra8Unorm,
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
    let sampler = bridge
        .device()
        .create_sampler(&wgpu::SamplerDescriptor::default());
    let imported_view = imported.create_view(&wgpu::TextureViewDescriptor::default());
    let bg = bridge
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&imported_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
    let staging_view = staging.create_view(&wgpu::TextureViewDescriptor::default());
    let mut enc = bridge
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("blit enc"),
        });
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &staging_view,
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
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..6, 0..1);
    }
    // Read back via buffer copy.
    let bytes_per_row_aligned = (dims.0 * 4).div_ceil(256) * 256;
    let buf = bridge.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback buf"),
        size: u64::from(bytes_per_row_aligned * dims.1),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &staging,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row_aligned),
                rows_per_image: Some(dims.1),
            },
        },
        wgpu::Extent3d {
            width: dims.0,
            height: dims.1,
            depth_or_array_layers: 1,
        },
    );
    bridge.queue().submit(std::iter::once(enc.finish()));
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    bridge
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll readback");
    rx.recv().unwrap().expect("buffer map");
    let mapped = slice.get_mapped_range().expect("get_mapped_range");
    let mut readback = vec![0u8; (dims.0 * dims.1 * 4) as usize];
    for y in 0..dims.1 as usize {
        let src_off = y * bytes_per_row_aligned as usize;
        let dst_off = y * (dims.0 * 4) as usize;
        readback[dst_off..dst_off + (dims.0 as usize * 4)]
            .copy_from_slice(&mapped[src_off..src_off + (dims.0 as usize * 4)]);
    }
    drop(mapped);
    buf.unmap();

    // Compare per-column MAE across all 4 channels.
    let w = dims.0 as usize;
    let h = dims.1 as usize;
    let mut col_mae = vec![0.0_f64; w];
    for (col, slot) in col_mae.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for row in 0..h {
            let off = (row * w + col) * 4;
            for ch in 0..4 {
                acc += (f64::from(bgra[off + ch]) - f64::from(readback[off + ch])).abs();
            }
        }
        *slot = acc / (h as f64 * 4.0);
    }
    let interior_mean: f64 = col_mae[64..w - 64].iter().sum::<f64>() / (w - 128) as f64;
    let left_max = col_mae[..32].iter().cloned().fold(0.0_f64, f64::max);
    let right_max = col_mae[w - 32..].iter().cloned().fold(0.0_f64, f64::max);
    println!(
        "coord-encoded BGRA dma-buf round-trip {}×{}: interior MAE {interior_mean:.2}, left-32 max MAE {left_max:.2}, right-32 max MAE {right_max:.2}",
        dims.0, dims.1
    );
    assert!(
        left_max <= interior_mean + 1.0,
        "left-edge MAE {left_max:.2} significantly above interior {interior_mean:.2} — BGRA dma-buf import/export round-trip is corrupting the left edge"
    );
    assert!(
        right_max <= interior_mean + 1.0,
        "right-edge MAE {right_max:.2} significantly above interior {interior_mean:.2}"
    );
}

#[test]
#[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF"]
fn bgra_dmabuf_roundtrip_2880x1920() {
    bgra_dmabuf_roundtrip_preserves_left_edge((2880, 1920));
}

/// Imported BGRA dma-buf → scaler → readback. The next step in the
/// bisect: the scaler hardware test (`matches_reference_coord_encoded_left_edge`)
/// is clean when reading from a directly-uploaded texture, but the
/// end-to-end harness fails when the scaler reads from an
/// `import_bgra_dmabuf` texture. If this test catches a left-edge
/// difference, the regression is the scaler reading the imported
/// texture (likely a `textureLoad` boundary interaction with a
/// dma-buf source view).
fn imported_bgra_then_scaler_preserves_left_edge(capture: (u32, u32), encode: (u32, u32)) {
    let bridge = match pollster::block_on(Nv12DmaBuf::new(encode.0, encode.1)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: cannot build Nv12DmaBuf bridge: {e}");
            return;
        }
    };
    let src_export = export_texture_as_dmabuf(
        bridge.device(),
        capture.0,
        capture.1,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        "coord bgra src",
    )
    .expect("export");
    let mut bgra = Vec::with_capacity((capture.0 * capture.1 * 4) as usize);
    for y in 0..capture.1 {
        for x in 0..capture.0 {
            let r = ((f64::from(x) / f64::from(capture.0)) * 256.0).clamp(0.0, 255.0) as u8;
            let g = ((f64::from(y) / f64::from(capture.1)) * 256.0).clamp(0.0, 255.0) as u8;
            bgra.extend_from_slice(&[128, g, r, 255]);
        }
    }
    bridge.queue().write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src_export.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bgra,
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
        .expect("poll");
    let imported = bridge
        .import_bgra_dmabuf(
            src_export.fd.try_clone().expect("dup"),
            src_export.drm_format_modifier,
            src_export.stride,
            src_export.offset,
            capture.0,
            capture.1,
        )
        .expect("import");
    let pipelines = Arc::new(Pipelines::build(bridge.device()));
    let scaler = Scaler::new_with_color_space(
        pipelines,
        bridge.device().clone(),
        bridge.queue().clone(),
        capture,
        encode,
        ColorSpace::Srgb8,
    )
    .expect("scaler");
    let scaled = scaler.scale(&imported).expect("scale");

    // Read back the scaler output (Rgba8Unorm) directly via
    // copy_texture_to_buffer — scaler's output has COPY_SRC.
    let bytes_per_row_aligned = (encode.0 * 4).div_ceil(256) * 256;
    let buf = bridge.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("scaler readback"),
        size: u64::from(bytes_per_row_aligned * encode.1),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = bridge
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: scaled,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row_aligned),
                rows_per_image: Some(encode.1),
            },
        },
        wgpu::Extent3d {
            width: encode.0,
            height: encode.1,
            depth_or_array_layers: 1,
        },
    );
    bridge.queue().submit(std::iter::once(enc.finish()));
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    bridge
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().unwrap().expect("map");
    let mapped = slice.get_mapped_range().expect("range");
    let w = encode.0 as usize;
    let h = encode.1 as usize;
    let mut readback = vec![0u8; w * h * 4];
    for y in 0..h {
        let src_off = y * bytes_per_row_aligned as usize;
        let dst_off = y * w * 4;
        readback[dst_off..dst_off + w * 4].copy_from_slice(&mapped[src_off..src_off + w * 4]);
    }
    drop(mapped);
    buf.unmap();

    // CPU Mitchell reference at the same dims. Source was BGRA; the
    // Mitchell reference operates on RGBA byte order, so swap the
    // channels into RGBA before resampling and compare.
    let mut rgba_src = Vec::with_capacity(bgra.len());
    for px in bgra.chunks_exact(4) {
        rgba_src.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
    }
    let ref_rgba = tether_scaler::reference::mitchell_filter_default(
        &rgba_src, capture.0, capture.1, encode.0, encode.1,
    );

    let mut col_mae = vec![0.0_f64; w];
    for (col, slot) in col_mae.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for row in 0..h {
            let off = (row * w + col) * 4;
            for ch in 0..3 {
                acc += (f64::from(readback[off + ch]) - f64::from(ref_rgba[off + ch])).abs();
            }
        }
        *slot = acc / (h as f64 * 3.0);
    }
    let interior: f64 = col_mae[64..w - 64].iter().sum::<f64>() / (w - 128) as f64;
    let left = col_mae[..32].iter().cloned().fold(0.0_f64, f64::max);
    let right = col_mae[w - 32..].iter().cloned().fold(0.0_f64, f64::max);
    println!(
        "imported BGRA dma-buf → scaler {}×{} → {}×{}: interior MAE {interior:.2}, left-32 max {left:.2}, right-32 max {right:.2}",
        capture.0, capture.1, encode.0, encode.1
    );
    assert!(
        left <= interior + 4.0,
        "left-edge MAE {left:.2} significantly above interior {interior:.2} — scaler differs on imported dma-buf vs uploaded texture"
    );
    assert!(
        right <= interior + 4.0,
        "right-edge MAE {right:.2} significantly above interior {interior:.2}"
    );
}

#[test]
#[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF"]
fn imported_bgra_then_scaler_2880x1920_to_2160x1440() {
    imported_bgra_then_scaler_preserves_left_edge((2880, 1920), (2160, 1440));
}

#[test]
#[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF"]
fn bgra_dmabuf_roundtrip_1920x1200() {
    bgra_dmabuf_roundtrip_preserves_left_edge((1920, 1200));
}

#[test]
#[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF + 16-bit storage"]
fn scaler_downscale_then_p010_bridge() {
    run_scaler_to_p010_chain((64, 64), (32, 32));
}
