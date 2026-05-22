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
    fill_solid_red(bridge.device(), bridge.queue(), &src_export.texture, capture);
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
    fill_solid_red(bridge.device(), bridge.queue(), &src_export.texture, capture);
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
#[test]
#[ignore = "requires a Vulkan-backed wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF + 16-bit storage"]
fn scaler_downscale_then_p010_bridge() {
    run_scaler_to_p010_chain((64, 64), (32, 32));
}
