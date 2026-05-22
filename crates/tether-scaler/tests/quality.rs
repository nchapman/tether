//! Quality tests for the CPU reference Mitchell implementation.
//!
//! Runs without a GPU — `make test` default suite.
//!
//! The point is to bound the reference's behaviour against an
//! independent implementation (the `image` crate's CatmullRom filter)
//! and against analytic properties (symmetry, smoothness, energy
//! preservation). These tests don't directly cover the shader — the
//! hardware test in `tests/hardware.rs` is the GPU↔CPU comparison —
//! but they keep the CPU reference honest. If the reference is wrong,
//! the hardware test is comparing two wrongs.

use image::{ImageBuffer, Rgba};
use tether_scaler::reference::{
    mitchell_filter_default, mitchell_weight, srgb_to_linear,
};

/// Mean squared error per channel across two RGBA buffers (ignoring
/// alpha).
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

/// Render via the `image` crate as a baseline. CatmullRom is the
/// closest filter the crate exposes — not identical to Mitchell-1/3
/// but close enough to bound by a few percent.
fn image_crate_catmull_rom(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    let buf: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::from_raw(src_w, src_h, src.to_vec()).expect("buffer");
    let dyn_img = image::DynamicImage::ImageRgba8(buf);
    let resized = dyn_img.resize_exact(dst_w, dst_h, image::imageops::FilterType::CatmullRom);
    resized.to_rgba8().into_raw()
}

/// Build a 256×256 smooth gradient fixture — photographic-like
/// content where linear-light and naive sRGB averaging produce
/// numerically close results. Strict text fixtures pull the
/// linear-light result several percent away from naive bicubic (and
/// that's the whole point of this work) so they're a poor baseline
/// for the "close to an independent bicubic" check.
fn smooth_gradient(width: u32, height: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = (((x + y) as f32 / (width + height) as f32) * 255.0) as u8;
            v.extend_from_slice(&[r, g, b, 255]);
        }
    }
    v
}

#[test]
fn reference_close_to_catmull_rom_baseline() {
    // CatmullRom (B=0, C=0.5) vs Mitchell (B=C=1/3) — both bicubic
    // 4-tap, differ in B/C. Expected MSE per channel should be small;
    // generous bound because the kernels really are different.
    let src = smooth_gradient(256, 256);
    let ours = mitchell_filter_default(&src, 256, 256, 128, 128);
    let theirs = image_crate_catmull_rom(&src, 256, 256, 128, 128);
    let mse = mse_rgb(&ours, &theirs);
    // ~75 MSE / channel = ~9 RMS per channel — typical drift between
    // Mitchell-1/3 (linear-light) and CatmullRom (naïve sRGB). The
    // gap on smooth content is dominated by the linear-light step;
    // on text fixtures it would be much larger (which is what we
    // want — separate test below).
    assert!(
        mse < 150.0,
        "MSE {mse:.2} between Mitchell-linear and CatmullRom-naïve smooth-content larger than expected"
    );
}

#[test]
fn upscale_then_downscale_preserves_solid_color() {
    // Round-trip a solid block through up + down and confirm we're
    // back close to the original (no gamma-mangling drift).
    let src: Vec<u8> = (0..128 * 128).flat_map(|_| [200u8, 150, 80, 255]).collect();
    let up = mitchell_filter_default(&src, 128, 128, 256, 256);
    let back = mitchell_filter_default(&up, 256, 256, 128, 128);
    for chunk in back.chunks_exact(4) {
        assert!((chunk[0] as i32 - 200).abs() <= 2, "R drift: {}", chunk[0]);
        assert!((chunk[1] as i32 - 150).abs() <= 2, "G drift: {}", chunk[1]);
        assert!((chunk[2] as i32 - 80).abs() <= 2, "B drift: {}", chunk[2]);
    }
}

#[test]
fn mitchell_kernel_weight_sum_scales_with_support() {
    // When the kernel widens by `support = max(scale, 1)`, the
    // un-normalized weight sum at integer offsets is approximately
    // `scale` (because we're sampling a kernel whose total integrated
    // area is 1 at unit support — widening to `support` widens that
    // area proportionally). This is *why* the runtime normalizes by
    // weight_sum: without normalization, downscale output would be
    // brightened by the support factor.
    //
    // The CPU reference + the shader both do that runtime
    // normalization. This test pins the un-normalized behaviour so a
    // future "optimize away the division" rewrite has to confront
    // the math.
    for scale in [1.0_f32, 1.5, 2.0, 3.0, 4.0] {
        let support = scale.max(1.0);
        let center = 0.5_f32;
        let n_taps = tether_scaler::reference::tap_count(scale) as i32;
        let i0 = center.floor() as i32 - n_taps / 2 + 1;
        let sum: f32 = (0..n_taps)
            .map(|k| {
                let x = (i0 + k) as f32;
                mitchell_weight((x - center) / support, 1.0 / 3.0, 1.0 / 3.0)
            })
            .sum();
        // Allow ±10% — the discrete sample of a continuous kernel.
        let expected = scale;
        let tol = expected * 0.10;
        assert!(
            (sum - expected).abs() < tol,
            "scale {scale} weight sum {sum} expected ≈ {expected} (n_taps {n_taps})",
        );
    }
}

#[test]
fn linear_light_visibly_differs_from_naive_on_dark_to_light() {
    // Downscale a 4×4 checkerboard of pure black + pure white pixels
    // to 1×1. The "true" linear-light result is mid-grey ≈ 188 in
    // sRGB; a naive sRGB-average gives 127. Our function must produce
    // the linear-light answer.
    let src: Vec<u8> = (0..4 * 4)
        .flat_map(|i: u32| {
            let on = ((i % 2) == 0) ^ ((i / 4) % 2 == 0);
            let v = if on { 0u8 } else { 255 };
            [v, v, v, 255]
        })
        .collect();
    let dst = mitchell_filter_default(&src, 4, 4, 1, 1);
    let g = dst[0];
    // Linear-light midpoint of black+white is sRGB 188; allow a band
    // around it. Definitely > 160 (well above naive 127).
    assert!(
        (170..=205).contains(&g),
        "expected linear-light mid-grey ~188, got {g}"
    );
    // Sanity: this would fail under naive sRGB averaging.
    assert!(
        g > 150,
        "naive sRGB average would put this at 127; our result {g} confirms linear-light"
    );
}

#[test]
fn asymmetric_scale_handles_independent_axes() {
    // Aspect-changing downscale: 256×128 → 64×96 (4× horizontal,
    // 0.75× vertical → actually a vertical *upscale*). Tests that
    // scale_x and scale_y are computed independently and the right
    // tap counts are used per axis. Output must be correct dims, no
    // NaN-ish (Mitchell's negative lobes can produce small overshoot
    // but encode_u8 clamps), and solid-colour preserved.
    let src: Vec<u8> = (0..256 * 128).flat_map(|_| [80u8, 160, 40, 255]).collect();
    let dst = mitchell_filter_default(&src, 256, 128, 64, 96);
    assert_eq!(dst.len(), 64 * 96 * 4);
    for (i, chunk) in dst.chunks_exact(4).enumerate() {
        assert!((chunk[0] as i32 - 80).abs() <= 2, "px {i} R drift: {}", chunk[0]);
        assert!((chunk[1] as i32 - 160).abs() <= 2, "px {i} G drift: {}", chunk[1]);
        assert!((chunk[2] as i32 - 40).abs() <= 2, "px {i} B drift: {}", chunk[2]);
        assert_eq!(chunk[3], 0xFF);
    }
}

#[test]
fn srgb_transfer_is_accurate_at_known_points() {
    // Spot-check the piecewise transfer at the breakpoint (0.04045
    // / 12.92 = ~0.00313) and at a few intermediate values. Use
    // generous tolerances because powf() is float arithmetic.
    let cases = [
        (0.0_f32, 0.0_f32),
        (0.04045, 0.04045 / 12.92), // exactly at the linear-segment top
        (0.5, 0.21404114),          // canonical sRGB → linear value
        (1.0, 1.0),
    ];
    for (srgb, expected_linear) in cases {
        let got = srgb_to_linear(srgb);
        assert!(
            (got - expected_linear).abs() < 1e-4,
            "srgb_to_linear({srgb}) = {got}, expected {expected_linear}"
        );
    }
}
