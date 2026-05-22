//! Hardware tests for the wgpu scaler.
//!
//! These tests require a working wgpu adapter (Vulkan on Linux, Metal
//! on macOS). They're `#[ignore]`'d by default — run with
//!     cargo test -p tether-scaler --test hardware -- --ignored --nocapture
//!
//! Validations:
//!  1. **Channel order**: a red source pixel must read back as red.
//!     The output texture is Rgba8Unorm and the host downstream is
//!     written for BGRA capture — this catches the BGRA↔RGBA landmine
//!     in seconds.
//!  2. **Match CPU reference**: GPU shader output vs the CPU reference
//!     in [`tether_scaler::reference`] must agree within an fp16-
//!     calibrated PSNR bound, on the full plan test matrix (typical
//!     downscale, 4K→720p with mip, upscale, heavy down with multi-
//!     level mip).

use std::sync::Arc;

use tether_scaler::reference;
use tether_scaler::{Pipelines, Scaler, ScalerError};
use wgpu::util::DeviceExt;

/// Build a wgpu device for tests. No special features required —
/// the scaler uses only core wgpu (Rgba16Float storage is core).
async fn build_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
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
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("tether-scaler test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
        .ok()?;
    Some((Arc::new(device), Arc::new(queue)))
}

/// Upload an RGBA8 buffer as a `Rgba8Unorm` texture suitable for
/// the scaler's source binding (TEXTURE_BINDING + COPY_DST).
fn upload_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bytes: &[u8],
    width: u32,
    height: u32,
) -> wgpu::Texture {
    let tex = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("scaler-test source"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytes,
    );
    tex
}

/// Map a texture's contents back to a Vec<u8>. Adds a row-padding
/// awareness because wgpu requires 256-byte row alignment for buffer
/// copies.
fn read_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
) -> Vec<u8> {
    let (width, height) = (tex.width(), tex.height());
    let bytes_per_pixel = 4u32;
    let unpadded_row = width * bytes_per_pixel;
    // 256-byte alignment is the wgpu spec for buffer-image copy.
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_row = unpadded_row.div_ceil(align) * align;
    let buf_size = (padded_row * height) as u64;

    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scaler-test readback"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("scaler-test readback"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);

    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    rx.recv().expect("map_async result").expect("map ok");
    let data = slice.get_mapped_range().expect("range");

    // Strip row padding.
    let mut out = Vec::with_capacity((unpadded_row * height) as usize);
    for y in 0..height {
        let start = (y * padded_row) as usize;
        let end = start + unpadded_row as usize;
        out.extend_from_slice(&data[start..end]);
    }
    drop(data);
    buf.unmap();
    out
}

/// Build a simple checkerboard test pattern at the given dimensions.
fn checkerboard(width: u32, height: u32, cell: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let on = ((x / cell) + (y / cell)) % 2 == 0;
            let c = if on { 255 } else { 0 };
            v.extend_from_slice(&[c, c, c, 255]);
        }
    }
    v
}

/// 256×256 RGB-quadrant test pattern. Each quadrant solid colour:
/// top-left red, top-right green, bottom-left blue, bottom-right white.
fn quadrant_image(width: u32, height: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((width * height * 4) as usize);
    let half_w = width / 2;
    let half_h = height / 2;
    for y in 0..height {
        for x in 0..width {
            let px = match (x < half_w, y < half_h) {
                (true, true) => [255, 0, 0, 255],
                (false, true) => [0, 255, 0, 255],
                (true, false) => [0, 0, 255, 255],
                (false, false) => [255, 255, 255, 255],
            };
            v.extend_from_slice(&px);
        }
    }
    v
}

/// Mean squared error per channel across a flat RGBA buffer
/// (excluding alpha). Lower is better.
fn mse_rgb(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0_f64;
    let mut n = 0u64;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for c in 0..3 {
            let d = f64::from(pa[c]) - f64::from(pb[c]);
            sum += d * d;
            n += 1;
        }
    }
    sum / (n as f64)
}

/// PSNR in dB. Returns `f64::INFINITY` if `mse == 0`.
fn psnr_db(mse: f64) -> f64 {
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        20.0 * (255.0_f64 / mse.sqrt()).log10()
    }
}

/// Max absolute per-channel difference (excluding alpha).
fn max_abs_diff_rgb(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    let mut m = 0u8;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for c in 0..3 {
            m = m.max(pa[c].abs_diff(pb[c]));
        }
    }
    m
}

#[test]
#[ignore = "requires wgpu adapter"]
fn fiducial_channel_order_red_stays_red() {
    // A 4×4 image of solid red pixels (255, 0, 0, 255) downscaled to
    // 2×2 must come out red, not blue (which would mean the source
    // texture was being read as BGRA when the bytes were RGBA, or vice
    // versa). This is the single most likely class of "looks scaled
    // but colours are wrong" bug; catch it first.
    let (device, queue) = pollster::block_on(build_device()).expect("wgpu device");
    let src_bytes: Vec<u8> = (0..4 * 4).flat_map(|_| [255u8, 0, 0, 255]).collect();
    let src_tex = upload_rgba8(&device, &queue, &src_bytes, 4, 4);
    let pipelines = Arc::new(Pipelines::build(&device));
    let scaler = Scaler::new(pipelines, device.clone(), queue.clone(), (4, 4), (2, 2))
        .expect("scaler");
    let out_tex = scaler.scale(&src_tex).expect("scale");
    let out = read_texture(&device, &queue, out_tex);
    for (i, chunk) in out.chunks_exact(4).enumerate() {
        assert!(
            chunk[0] > 200 && chunk[1] < 50 && chunk[2] < 50,
            "pixel {i} expected red, got ({}, {}, {})",
            chunk[0],
            chunk[1],
            chunk[2]
        );
    }
}

/// Round-trip a scaler at various (src, dst) ratios and assert the
/// shader output matches the CPU reference within PSNR ≥ 38 dB and
/// max per-channel diff ≤ 4. Bounds are tuned for fp16 vs fp32
/// precision in the intermediate texture; tightening below this
/// chases noise.
fn assert_matches_reference(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    pipelines: &Arc<Pipelines>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    src_bytes: &[u8],
    case_label: &str,
) {
    let src_tex = upload_rgba8(device, queue, src_bytes, src_w, src_h);
    let scaler = Scaler::new(
        pipelines.clone(),
        device.clone(),
        queue.clone(),
        (src_w, src_h),
        (dst_w, dst_h),
    )
    .expect("scaler new");
    let out_tex = scaler.scale(&src_tex).expect("scale");
    let gpu_out = read_texture(device, queue, out_tex);

    let ref_out = reference::mitchell_filter_default(src_bytes, src_w, src_h, dst_w, dst_h);
    assert_eq!(
        gpu_out.len(),
        ref_out.len(),
        "size mismatch {case_label}: gpu {} vs ref {}",
        gpu_out.len(),
        ref_out.len(),
    );

    let mse = mse_rgb(&gpu_out, &ref_out);
    let psnr = psnr_db(mse);
    let max_diff = max_abs_diff_rgb(&gpu_out, &ref_out);
    println!(
        "{case_label}: {}x{} -> {}x{} (mip levels {}) — PSNR {:.2} dB, max diff {max_diff}",
        src_w,
        src_h,
        dst_w,
        dst_h,
        scaler.mip_levels(),
        psnr,
    );
    // Measured baseline (Vulkan/Linux trunk wgpu): 57–inf dB across
    // the test matrix, max diff 0–1. Floor is set 7 dB below the worst
    // observed result to absorb backend variation (Metal vs Vulkan
    // fp16 arithmetic can differ by 1–2 ULP in the last mantissa bit)
    // while still catching real regressions. 50 dB is a ~3000× tighter
    // MSE bound than the original 38; below this means the shader is
    // visibly diverging from the spec.
    assert!(
        psnr >= 50.0,
        "{case_label}: PSNR {psnr:.2} dB below 50 (mse {mse:.2})"
    );
    assert!(
        max_diff <= 2,
        "{case_label}: max per-channel diff {max_diff} > 2"
    );
}

#[test]
#[ignore = "requires wgpu adapter"]
fn matches_reference_typical_downscale() {
    let (device, queue) = pollster::block_on(build_device()).expect("wgpu device");
    let pipelines = Arc::new(Pipelines::build(&device));
    let src = quadrant_image(256, 256);
    assert_matches_reference(&device, &queue, &pipelines, 256, 256, 128, 128, &src, "256→128 quadrants");
    let cb = checkerboard(256, 256, 8);
    assert_matches_reference(&device, &queue, &pipelines, 256, 256, 200, 150, &cb, "256→200×150 checker");
}

#[test]
#[ignore = "requires wgpu adapter"]
fn matches_reference_upscale() {
    let (device, queue) = pollster::block_on(build_device()).expect("wgpu device");
    let pipelines = Arc::new(Pipelines::build(&device));
    let src = quadrant_image(64, 64);
    assert_matches_reference(&device, &queue, &pipelines, 64, 64, 256, 256, &src, "64→256 upscale");
}

#[test]
#[ignore = "requires wgpu adapter"]
fn matches_reference_heavy_downscale_with_mip() {
    // 8× downscale exercises the mip prefilter chain (3 levels).
    let (device, queue) = pollster::block_on(build_device()).expect("wgpu device");
    let pipelines = Arc::new(Pipelines::build(&device));
    let cb = checkerboard(256, 256, 4);
    assert_matches_reference(&device, &queue, &pipelines, 256, 256, 32, 32, &cb, "256→32 heavy down");
}

#[test]
#[ignore = "requires wgpu adapter"]
fn matches_reference_realistic_screen_dim() {
    // The motivating production case: 4K capture → 720p encode. We
    // use a smaller-but-proportional 1920×1080 → 1280×720 to keep the
    // test fast.
    let (device, queue) = pollster::block_on(build_device()).expect("wgpu device");
    let pipelines = Arc::new(Pipelines::build(&device));
    let src = quadrant_image(1920, 1080);
    assert_matches_reference(&device, &queue, &pipelines, 1920, 1080, 1280, 720, &src, "1080p→720p");
}

#[test]
#[ignore = "requires wgpu adapter"]
fn matches_reference_asymmetric_scale() {
    // Aspect-changing scale: 256×256 → 192×128 (0.75× horizontal,
    // 0.5× vertical). Exercises that x and y scale independently
    // (separate scale_x/scale_y, separate tap counts) through the
    // shader on a real device.
    let (device, queue) = pollster::block_on(build_device()).expect("wgpu device");
    let pipelines = Arc::new(Pipelines::build(&device));
    let src = quadrant_image(256, 256);
    assert_matches_reference(
        &device,
        &queue,
        &pipelines,
        256,
        256,
        192,
        128,
        &src,
        "256→192×128 asymmetric",
    );
}

#[test]
#[ignore = "requires wgpu adapter"]
fn no_scale_needed_errors_at_exact_match() {
    // Validation path: 1:1 dims return NoScaleNeeded so callers can
    // skip the scaler entirely. Pure upscale (dst > src) must NOT
    // trip this — the client upscale path depends on it succeeding.
    let (device, queue) = pollster::block_on(build_device()).expect("wgpu device");
    let pipelines = Arc::new(Pipelines::build(&device));
    let res = Scaler::new(pipelines.clone(), device.clone(), queue.clone(), (64, 64), (64, 64));
    assert!(matches!(res, Err(ScalerError::NoScaleNeeded)));
    // Upscale must succeed.
    let ok = Scaler::new(pipelines.clone(), device.clone(), queue.clone(), (64, 64), (128, 128));
    assert!(ok.is_ok());
    // Zero dim must error with ZeroDim, not NoScaleNeeded.
    let z = Scaler::new(pipelines, device, queue, (0, 64), (32, 32));
    assert!(matches!(z, Err(ScalerError::ZeroDim { .. })));
}

#[test]
#[ignore = "requires wgpu adapter"]
fn dim_mismatch_errors_typed() {
    // Build a scaler for 64×64 → 32×32 but hand it a 128×128 source.
    // The scaler must return DimMismatch rather than silently scaling
    // with stale params (which would produce visibly corrupt output).
    let (device, queue) = pollster::block_on(build_device()).expect("wgpu device");
    let pipelines = Arc::new(Pipelines::build(&device));
    let scaler =
        Scaler::new(pipelines, device.clone(), queue.clone(), (64, 64), (32, 32)).expect("scaler");
    let wrong = upload_rgba8(&device, &queue, &checkerboard(128, 128, 8), 128, 128);
    match scaler.scale(&wrong) {
        Err(ScalerError::DimMismatch { expected, got }) => {
            assert_eq!(expected, (64, 64));
            assert_eq!(got, (128, 128));
        }
        other => panic!("expected DimMismatch, got {other:?}"),
    }
}
