//! CPU reference implementation of the Mitchell-Netravali scaler.
//!
//! This is the *spec* the WGSL shader must match. The shader and this
//! module implement the same algorithm; the SSIM/PSNR hardware test
//! drives both with identical inputs and asserts the outputs agree
//! within an fp16-calibrated tolerance.
//!
//! Keeping the reference in fp32 Rust serves two purposes beyond the
//! comparison:
//!
//! - **Executable spec.** A reviewer can read the algorithm here
//!   without parsing WGSL. Comments call out the conventions (texel-
//!   center mapping, tap-count scaling, mip prefilter threshold) that
//!   the shader silently relies on.
//! - **CPU-side tests.** Quality tests in `tests/quality.rs` run
//!   entirely on the CPU reference — no GPU required for `make test`.
//!
//! All sample arrays in this module are RGBA8 (sRGB-encoded), row-
//! major, 4 bytes per pixel, `stride = width * 4`. RGBA and not BGRA:
//! the shader writes RGBA storage (BGRA storage isn't a portable wgpu
//! feature) and the host's downstream chroma shaders are taught to
//! accept either. Stick to one convention here.

/// Mitchell-Netravali bicubic kernel.
///
/// `b = c = 1/3` is the canonical "balanced" form from
/// Mitchell & Netravali, "Reconstruction Filters in Computer Graphics"
/// (SIGGRAPH '88). Variants: Catmull-Rom (b=0, c=0.5) is sharper but
/// has mild ringing; Robidoux (b≈0.379, c≈0.310) is tuned for
/// downscale specifically.
#[must_use]
pub fn mitchell_weight(x: f32, b: f32, c: f32) -> f32 {
    let ax = x.abs();
    if ax < 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * ax * ax * ax
            + (-18.0 + 12.0 * b + 6.0 * c) * ax * ax
            + (6.0 - 2.0 * b))
            / 6.0
    } else if ax < 2.0 {
        ((-b - 6.0 * c) * ax * ax * ax
            + (6.0 * b + 30.0 * c) * ax * ax
            + (-12.0 * b - 48.0 * c) * ax
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

/// IEC 61966-2-1 sRGB to linear transfer. Piecewise — *not* the
/// `pow(c, 2.2)` approximation. The two differ enough in the dark
/// pixel regime (where text antialiasing lives) that the approximation
/// would visibly undo the linear-light quality win.
#[must_use]
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse of [`srgb_to_linear`]. Same piecewise definition.
#[must_use]
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Hard ceiling on the per-axis tap count. Enforced in
/// [`crate::scaler`] before pipeline dispatch — at downscale ratios
/// where `4 * scale > MAX_TAPS`, the scaler routes through additional
/// mipmap prefilter levels instead of stretching the kernel further.
/// The shader receives `n_taps` as a uniform and has no compile-time
/// bound of its own.
pub const MAX_TAPS: u32 = 32;

/// Tap count for a given downscale ratio (`scale >= 1` means
/// downscale, `< 1` means upscale). For upscale we stay at 4 taps
/// (Mitchell's natural support); for downscale we widen so the
/// effective filter still bandlimits — a fixed 4-tap kernel at, say,
/// 4× downscale would alias visibly even with the support-widening
/// trick alone.
#[must_use]
pub fn tap_count(scale: f32) -> u32 {
    let raw = (4.0_f32 * scale.max(1.0)).ceil() as u32;
    raw.max(4).min(MAX_TAPS)
}

/// Box-filter downsample by 2× in linear-light. Each output pixel
/// averages the four corresponding source pixels in linear space, then
/// re-encodes sRGB so the next pass reads sRGB-encoded values (same
/// shape as the original input).
///
/// `(out_w, out_h) = ((src_w + 1) / 2, (src_h + 1) / 2)` — odd source
/// dimensions round up; the last column/row clamps via edge-replicate.
#[must_use]
pub fn mip_box_down(src: &[u8], src_w: u32, src_h: u32) -> (Vec<u8>, u32, u32) {
    let dst_w = src_w.div_ceil(2);
    let dst_h = src_h.div_ceil(2);
    let mut dst = vec![0u8; (dst_w * dst_h * 4) as usize];
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;
    for y in 0..dh {
        for x in 0..dw {
            let sx = x * 2;
            let sy = y * 2;
            let sx1 = (sx + 1).min(sw - 1);
            let sy1 = (sy + 1).min(sh - 1);
            let mut accum = [0.0_f32; 3];
            for &(px, py) in &[(sx, sy), (sx1, sy), (sx, sy1), (sx1, sy1)] {
                let off = (py * sw + px) * 4;
                accum[0] += srgb_to_linear(f32::from(src[off]) / 255.0);
                accum[1] += srgb_to_linear(f32::from(src[off + 1]) / 255.0);
                accum[2] += srgb_to_linear(f32::from(src[off + 2]) / 255.0);
            }
            let lin = [accum[0] * 0.25, accum[1] * 0.25, accum[2] * 0.25];
            let off = (y * dw + x) * 4;
            dst[off] = encode_u8(linear_to_srgb(lin[0]));
            dst[off + 1] = encode_u8(linear_to_srgb(lin[1]));
            dst[off + 2] = encode_u8(linear_to_srgb(lin[2]));
            dst[off + 3] = 0xFF;
        }
    }
    (dst, dst_w, dst_h)
}

fn encode_u8(c: f32) -> u8 {
    // Saturating cast — Mitchell's negative lobes can push values
    // slightly out of [0, 1] near sharp edges. Clamp before
    // quantizing, otherwise the wrap produces visible speckles.
    (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Mitchell-Netravali resample from `(src_w, src_h)` to
/// `(dst_w, dst_h)` in linear-light. Includes the mip prefilter for
/// downscale ratios > 2× — same algorithm the shader runs.
///
/// `b, c` default to `(1/3, 1/3)` via [`mitchell_filter_default`];
/// expose them here so the manual side-by-side test can compare
/// Catmull-Rom / Robidoux variants without code churn.
///
/// # Panics
///
/// Panics if any dimension is zero or if `src.len() != src_w * src_h *
/// 4`. Callers (Rust API + tests) validate at the boundary; this is
/// the inner kernel.
#[must_use]
pub fn mitchell_filter(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    b: f32,
    c: f32,
) -> Vec<u8> {
    assert!(src_w > 0 && src_h > 0 && dst_w > 0 && dst_h > 0);
    assert_eq!(
        src.len(),
        (src_w as usize) * (src_h as usize) * 4,
        "src buffer size mismatch"
    );

    // Mip prefilter: while either axis is more than 2× downscale, box-
    // halve it. The Mitchell pass then runs from at most 2× downscale,
    // where a 4-tap (or 8-tap at scale=2) kernel is sufficient.
    let mut cur = src.to_vec();
    let mut cur_w = src_w;
    let mut cur_h = src_h;
    loop {
        let scale_x = cur_w as f32 / dst_w as f32;
        let scale_y = cur_h as f32 / dst_h as f32;
        if scale_x.max(scale_y) <= 2.0 {
            break;
        }
        let (next, nw, nh) = mip_box_down(&cur, cur_w, cur_h);
        cur = next;
        cur_w = nw;
        cur_h = nh;
    }

    // Horizontal pass → intermediate at (dst_w × cur_h) in linear fp32.
    let scale_x = cur_w as f32 / dst_w as f32;
    let n_taps_x = tap_count(scale_x);
    let support_x = scale_x.max(1.0);
    let mut intermediate = vec![0.0_f32; (dst_w as usize) * (cur_h as usize) * 3];
    for y in 0..cur_h as usize {
        for ox in 0..dst_w as usize {
            let center = (ox as f32 + 0.5) * scale_x - 0.5;
            let half = (n_taps_x / 2) as i32;
            let i0 = center.floor() as i32 - half + 1;
            let mut sum = [0.0_f32; 3];
            let mut w_sum = 0.0_f32;
            for k in 0..n_taps_x as i32 {
                let x = i0 + k;
                let xc = x.clamp(0, cur_w as i32 - 1) as usize;
                let w = mitchell_weight((x as f32 - center) / support_x, b, c);
                let off = (y * cur_w as usize + xc) * 4;
                sum[0] += srgb_to_linear(f32::from(cur[off]) / 255.0) * w;
                sum[1] += srgb_to_linear(f32::from(cur[off + 1]) / 255.0) * w;
                sum[2] += srgb_to_linear(f32::from(cur[off + 2]) / 255.0) * w;
                w_sum += w;
            }
            let ws = if w_sum.abs() < 1e-6 { 1.0 } else { w_sum };
            let off = (y * dst_w as usize + ox) * 3;
            intermediate[off] = sum[0] / ws;
            intermediate[off + 1] = sum[1] / ws;
            intermediate[off + 2] = sum[2] / ws;
        }
    }

    // Vertical pass → output at (dst_w × dst_h) in sRGB u8.
    let scale_y = cur_h as f32 / dst_h as f32;
    let n_taps_y = tap_count(scale_y);
    let support_y = scale_y.max(1.0);
    let mut dst = vec![0u8; (dst_w as usize) * (dst_h as usize) * 4];
    for oy in 0..dst_h as usize {
        let center = (oy as f32 + 0.5) * scale_y - 0.5;
        let half = (n_taps_y / 2) as i32;
        let i0 = center.floor() as i32 - half + 1;
        for ox in 0..dst_w as usize {
            let mut sum = [0.0_f32; 3];
            let mut w_sum = 0.0_f32;
            for k in 0..n_taps_y as i32 {
                let y = i0 + k;
                let yc = y.clamp(0, cur_h as i32 - 1) as usize;
                let w = mitchell_weight((y as f32 - center) / support_y, b, c);
                let off = (yc * dst_w as usize + ox) * 3;
                sum[0] += intermediate[off] * w;
                sum[1] += intermediate[off + 1] * w;
                sum[2] += intermediate[off + 2] * w;
                w_sum += w;
            }
            let ws = if w_sum.abs() < 1e-6 { 1.0 } else { w_sum };
            let lin = [sum[0] / ws, sum[1] / ws, sum[2] / ws];
            let off = (oy * dst_w as usize + ox) * 4;
            dst[off] = encode_u8(linear_to_srgb(lin[0]));
            dst[off + 1] = encode_u8(linear_to_srgb(lin[1]));
            dst[off + 2] = encode_u8(linear_to_srgb(lin[2]));
            dst[off + 3] = 0xFF;
        }
    }
    dst
}

/// Convenience: [`mitchell_filter`] with the canonical `B = C = 1/3`.
#[must_use]
pub fn mitchell_filter_default(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    mitchell_filter(src, src_w, src_h, dst_w, dst_h, 1.0 / 3.0, 1.0 / 3.0)
}

/// Linear-light Mitchell-Netravali scaler over a packed N-channel
/// plane. Used as the CPU reference for the [`crate::Scaler`] YUV
/// plane paths ([`crate::ColorSpace::LumaR8`] /
/// [`crate::ColorSpace::ChromaRg8`]).
///
/// `channels` is 1 (Y plane) or 2 (interleaved UV plane). No transfer
/// applied either way — NV12 planes are already in the encoded domain
/// VT expects, and Mitchell-on-gamma is the project trade-off for
/// 8-bit YUV planes (see plan).
///
/// `chroma_offset` is in *source-pixel* units, added to the kernel
/// center before clamp/sum — the same uniform the shader receives.
/// Y planes pass `(0.0, 0.0)`; UV planes pass the cosited-siting
/// correction.
///
/// No mip prefilter — the YUV plane scaler currently relies on the
/// tap-count widening for downscale > 2× (see `Scaler::scale`).
///
/// Panics if `src.len() != src_w * src_h * channels` or any dimension
/// is zero. Caller validates at the boundary.
#[must_use]
pub fn mitchell_filter_plane(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: usize,
    chroma_offset: (f32, f32),
) -> Vec<u8> {
    assert!(src_w > 0 && src_h > 0 && dst_w > 0 && dst_h > 0);
    assert!(channels == 1 || channels == 2, "channels must be 1 or 2");
    assert_eq!(
        src.len(),
        (src_w as usize) * (src_h as usize) * channels,
        "src buffer size mismatch"
    );

    let (offset_x, offset_y) = chroma_offset;
    // Horizontal pass → intermediate fp32 at (dst_w × src_h × channels).
    let scale_x = src_w as f32 / dst_w as f32;
    let n_taps_x = tap_count(scale_x);
    let support_x = scale_x.max(1.0);
    let mut intermediate = vec![0.0_f32; (dst_w as usize) * (src_h as usize) * channels];
    for y in 0..src_h as usize {
        for ox in 0..dst_w as usize {
            let center = (ox as f32 + 0.5) * scale_x - 0.5 + offset_x;
            let half = (n_taps_x / 2) as i32;
            let i0 = center.floor() as i32 - half + 1;
            let mut sum = [0.0_f32; 2];
            let mut w_sum = 0.0_f32;
            for k in 0..n_taps_x as i32 {
                let x = i0 + k;
                let xc = x.clamp(0, src_w as i32 - 1) as usize;
                let w = mitchell_weight((x as f32 - center) / support_x, 1.0 / 3.0, 1.0 / 3.0);
                let off = (y * src_w as usize + xc) * channels;
                for ch in 0..channels {
                    sum[ch] += f32::from(src[off + ch]) / 255.0 * w;
                }
                w_sum += w;
            }
            let ws = if w_sum.abs() < 1e-6 { 1.0 } else { w_sum };
            let off = (y * dst_w as usize + ox) * channels;
            for ch in 0..channels {
                intermediate[off + ch] = sum[ch] / ws;
            }
        }
    }

    // Vertical pass → output u8 at (dst_w × dst_h × channels).
    let scale_y = src_h as f32 / dst_h as f32;
    let n_taps_y = tap_count(scale_y);
    let support_y = scale_y.max(1.0);
    let mut dst = vec![0u8; (dst_w as usize) * (dst_h as usize) * channels];
    for oy in 0..dst_h as usize {
        let center = (oy as f32 + 0.5) * scale_y - 0.5 + offset_y;
        let half = (n_taps_y / 2) as i32;
        let i0 = center.floor() as i32 - half + 1;
        for ox in 0..dst_w as usize {
            let mut sum = [0.0_f32; 2];
            let mut w_sum = 0.0_f32;
            for k in 0..n_taps_y as i32 {
                let y = i0 + k;
                let yc = y.clamp(0, src_h as i32 - 1) as usize;
                let w = mitchell_weight((y as f32 - center) / support_y, 1.0 / 3.0, 1.0 / 3.0);
                let off = (yc * dst_w as usize + ox) * channels;
                for ch in 0..channels {
                    sum[ch] += intermediate[off + ch] * w;
                }
                w_sum += w;
            }
            let ws = if w_sum.abs() < 1e-6 { 1.0 } else { w_sum };
            let off = (oy * dst_w as usize + ox) * channels;
            for ch in 0..channels {
                let v = (sum[ch] / ws).clamp(0.0, 1.0);
                dst[off + ch] = (v * 255.0 + 0.5) as u8;
            }
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mitchell_weight_partition_of_unity_at_integer_shifts() {
        // The four Mitchell weights at integer-shifted offsets
        // (t, t-1, t-2, t-3) for t ∈ [1, 2) must sum to exactly 1 for
        // any t. This is the "reproduces constants" property — without
        // it, scaling a flat-colour image would bias the brightness.
        for t_int in 0..100 {
            let t = 1.0 + (t_int as f32) / 100.0;
            let sum: f32 = [t, t - 1.0, t - 2.0, t - 3.0]
                .iter()
                .map(|&x| mitchell_weight(x, 1.0 / 3.0, 1.0 / 3.0))
                .sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Mitchell weights at t={t} sum to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn srgb_roundtrip_is_identity_to_quantization() {
        // Decode then encode each 8-bit value; expect the result to
        // round back to the same byte (allowing for one-bit drift at a
        // handful of boundary values that float arithmetic can shift).
        let mut drift = 0;
        for b in 0..=255u8 {
            let c = f32::from(b) / 255.0;
            let round = linear_to_srgb(srgb_to_linear(c));
            let back = (round.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            if back.abs_diff(b) > 1 {
                panic!("sRGB round-trip drift at {b}: got {back}");
            }
            if back != b {
                drift += 1;
            }
        }
        assert!(drift < 8, "too many drifting values: {drift}");
    }

    #[test]
    fn linear_light_avg_is_brighter_than_naive() {
        // The whole point of linear-light filtering: averaging black
        // (0) and white (255) in linear space and re-encoding gives
        // sRGB ~188 (≈ 0.5^(1/2.4) re-encoded). Naively averaging in
        // sRGB space gives 127.5 — visibly darker. This test pins the
        // behaviour so a future "perf optimization" that skips the
        // transfer can't quietly regress to naive averaging.
        let avg_linear = (srgb_to_linear(0.0) + srgb_to_linear(1.0)) * 0.5;
        let result_u8 = (linear_to_srgb(avg_linear).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        assert!(
            (180..=195).contains(&result_u8),
            "expected ~188 (linear-light mid-grey), got {result_u8}"
        );
    }

    #[test]
    fn identity_scale_round_trips_solid_color() {
        // 1:1 scale of a solid-colour image must produce the same bytes
        // back (modulo sRGB round-trip drift of at most 1).
        let src: Vec<u8> = (0..16 * 16)
            .flat_map(|_| [128, 64, 200, 255])
            .collect();
        let dst = mitchell_filter_default(&src, 16, 16, 16, 16);
        for (i, (s, d)) in src.iter().zip(dst.iter()).enumerate() {
            assert!(
                s.abs_diff(*d) <= 1,
                "channel {i} drifted {s} -> {d}",
            );
        }
    }

    #[test]
    fn downscale_solid_color_preserves_color() {
        // Solid-colour 64×64 → 16×16 must stay the same colour (no
        // edge effects to bias toward 0 or 255).
        let src: Vec<u8> = (0..64 * 64).flat_map(|_| [50, 100, 200, 255]).collect();
        let dst = mitchell_filter_default(&src, 64, 64, 16, 16);
        for chunk in dst.chunks_exact(4) {
            assert!((chunk[0] as i32 - 50).abs() <= 2);
            assert!((chunk[1] as i32 - 100).abs() <= 2);
            assert!((chunk[2] as i32 - 200).abs() <= 2);
        }
    }

    #[test]
    fn heavy_downscale_triggers_mip_prefilter() {
        // 256×256 → 32×32 is 8× downscale — mip path must engage.
        // Test indirectly: the function must complete and produce a
        // plausible output (alpha stays 255, no NaNs).
        let src: Vec<u8> = (0..256 * 256)
            .flat_map(|i: u32| {
                let v = (i % 256) as u8;
                [v, v, v, 0xFF]
            })
            .collect();
        let dst = mitchell_filter_default(&src, 256, 256, 32, 32);
        assert_eq!(dst.len(), 32 * 32 * 4);
        // The source is a horizontal ramp repeated every row, so every
        // output row is identical. The leftmost output pixel averages
        // the dark end of the ramp (small values), the rightmost
        // averages the bright end. Check: alpha is intact, no NaNs,
        // and the per-row average is roughly mid-grey in linear-light
        // (sRGB ≈ 188, not the naive-average 127).
        let mut row_sum_linear = 0.0_f32;
        for chunk in dst.chunks_exact(4) {
            assert_eq!(chunk[3], 0xFF);
            row_sum_linear += srgb_to_linear(f32::from(chunk[0]) / 255.0);
        }
        let row_avg_srgb = linear_to_srgb(row_sum_linear / (32 * 32) as f32) * 255.0;
        // Naive sRGB average would be ~127; linear-light average of a
        // uniform sRGB distribution is around 150 (most values cluster
        // in the dark linear range). Allow a generous band — this is
        // a smoke test that the pipeline didn't gamma-mangle the
        // result, not a precision check.
        assert!(
            (130.0..=200.0).contains(&row_avg_srgb),
            "linear-light avg of downscaled ramp out of range: {row_avg_srgb}"
        );
    }

    #[test]
    fn tap_count_grows_with_downscale() {
        assert_eq!(tap_count(1.0), 4); // upscale or 1:1
        assert_eq!(tap_count(1.5), 6); // 1.5× down
        assert_eq!(tap_count(2.0), 8);
        assert_eq!(tap_count(3.0), 12);
        assert_eq!(tap_count(8.0), 32); // clamped
        assert_eq!(tap_count(100.0), 32);
    }
}
