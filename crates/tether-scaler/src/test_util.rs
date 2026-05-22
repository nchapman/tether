//! Test utilities shared across the workspace.
//!
//! Gated behind the `test-util` cargo feature. Consumers (currently
//! `tether-render`'s round-trip harness, plus `tether-scaler`'s own
//! hardware tests) add `tether-scaler = { features = ["test-util"] }`
//! as a `[dev-dependencies]` entry.
//!
//! ## Why everything lives here
//!
//! The same SSIM / PSNR / Mitchell-reference / coordinate-residual code
//! is needed by every hardware round-trip cell in the repo. Putting it
//! in `tether-render`'s test tree would force a circular dep when the
//! macOS sibling round-trip wants the same helpers. Keeping it on
//! `tether-scaler` matches `tether-transport::test_support` — a
//! workspace-wide feature-gated test-only module.
//!
//! ## What's here
//!
//! - Photometric metrics: [`ssim_rgb`], [`psnr_db_rgb`], [`psnr_db_y`]
//! - Geometric metric for coordinate-encoded fixtures: [`coord_fixture_fill`],
//!   [`coord_fixture_residual_px_rms`]
//! - CPU references: [`cpu_mitchell_resize_bgra`], [`cpu_chroma_roundtrip_bgra`]
//! - Letterbox geometry: [`letterbox_fit_dims`], [`letterbox_video_rect`]
//!
//! Everything operates on 4-channel byte buffers. Channel order is
//! explicit in the function name (`_bgra` vs `_rgba`); the SSIM/PSNR
//! helpers are order-agnostic on RGB channels and ignore alpha.

use crate::reference::mitchell_filter_default;

/// 2D SSIM (Structural Similarity Index) per RGB channel, averaged.
///
/// Catches structured error patterns (ringing, edge smearing,
/// stride-induced ghosting) that PSNR is blind to. Uses an 8×8 sliding
/// window — smaller than the canonical 11×11 for test speed; the
/// difference is < 0.001 on natural images. Alpha channel is ignored.
///
/// Returns a value in `[-1, 1]`. 1.0 is identical; ≥ 0.99 is
/// "perceptually indistinguishable" on natural content; the threshold
/// for "lossy-encoded screen capture survives intact" is empirical and
/// derived per-cell on green main rather than hardcoded.
///
/// Channel order is irrelevant — the per-channel computation is
/// symmetric in R/G/B as long as both inputs use the same order.
#[must_use]
pub fn ssim_rgb(a: &[u8], b: &[u8], width: u32, height: u32) -> f64 {
    assert_eq!(a.len(), b.len(), "ssim inputs must be the same length");
    assert_eq!(a.len(), (width * height * 4) as usize, "ssim length must match width*height*4");
    const WIN: i32 = 8;
    let c1: f64 = (0.01_f64 * 255.0).powi(2);
    let c2: f64 = (0.03_f64 * 255.0).powi(2);
    let w = width as i32;
    let h = height as i32;
    let mut total = 0.0_f64;
    let mut n_windows: u64 = 0;
    for wy in 0..(h - WIN + 1).max(0) {
        for wx in 0..(w - WIN + 1).max(0) {
            for ch in 0..3 {
                let mut sum_a = 0.0_f64;
                let mut sum_b = 0.0_f64;
                let mut sum_aa = 0.0_f64;
                let mut sum_bb = 0.0_f64;
                let mut sum_ab = 0.0_f64;
                let n_pix = (WIN * WIN) as f64;
                for dy in 0..WIN {
                    for dx in 0..WIN {
                        let off = (((wy + dy) * w + (wx + dx)) * 4 + ch) as usize;
                        let pa = f64::from(a[off]);
                        let pb = f64::from(b[off]);
                        sum_a += pa;
                        sum_b += pb;
                        sum_aa += pa * pa;
                        sum_bb += pb * pb;
                        sum_ab += pa * pb;
                    }
                }
                let mu_a = sum_a / n_pix;
                let mu_b = sum_b / n_pix;
                let var_a = (sum_aa / n_pix) - mu_a * mu_a;
                let var_b = (sum_bb / n_pix) - mu_b * mu_b;
                let cov_ab = (sum_ab / n_pix) - mu_a * mu_b;
                let num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2);
                let den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2);
                total += num / den;
                n_windows += 1;
            }
        }
    }
    if n_windows == 0 {
        return 1.0;
    }
    total / (n_windows as f64)
}

/// Mean-squared error over RGB channels (alpha ignored). Operates on
/// 4-byte-stride buffers; channel order is irrelevant.
#[must_use]
pub fn mse_rgb(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "mse inputs must be the same length");
    let mut sum = 0.0_f64;
    let mut n = 0_u64;
    for px in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let (pa, pb) = px;
        for ch in 0..3 {
            let d = f64::from(pa[ch]) - f64::from(pb[ch]);
            sum += d * d;
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / (n as f64) }
}

/// PSNR over RGB channels in dB. Returns `f64::INFINITY` if identical.
#[must_use]
pub fn psnr_db_rgb(a: &[u8], b: &[u8]) -> f64 {
    psnr_from_mse(mse_rgb(a, b))
}

/// Convert MSE to PSNR (peak = 255 for 8-bit channels).
fn psnr_from_mse(mse: f64) -> f64 {
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        20.0 * (255.0_f64 / mse.sqrt()).log10()
    }
}

/// PSNR on the BT.709 luma channel only. Isolates the luminance
/// pipeline (renderer/scaler/encoder Y plane) from the chroma noise
/// floor that 4:2:0 subsampling dominates. The SME review specifically
/// called this out as worth adding alongside RGB SSIM.
///
/// Inputs are BGRA. The Y' coefficients are Rec.709 (full range,
/// 0..255).
#[must_use]
pub fn psnr_db_y_bgra(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0_f64;
    let mut n = 0_u64;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let ya = bgra_to_y_709(pa);
        let yb = bgra_to_y_709(pb);
        let d = ya - yb;
        sum += d * d;
        n += 1;
    }
    psnr_from_mse(if n == 0 { 0.0 } else { sum / (n as f64) })
}

fn bgra_to_y_709(px: &[u8]) -> f64 {
    // BT.709 full-range Y'. px is [B, G, R, A].
    let b = f64::from(px[0]);
    let g = f64::from(px[1]);
    let r = f64::from(px[2]);
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

// ---------------------------------------------------------------------
// Coordinate-encoded fixture
// ---------------------------------------------------------------------

/// Encode (x, y) into a BGRA pixel so every pixel literally tells you
/// where it came from. R = x%256, G = y%256, B = ((x/256) << 4) | (y/256).
///
/// Covers source dimensions up to 4096×4096 — a 12-bit budget split
/// evenly between x and y. Larger dims would alias and break the
/// residual computation; the harness asserts `capture_dims <= 4096` for
/// coord-encoded cases.
#[inline]
#[must_use]
pub fn coord_encode_bgra(x: u32, y: u32) -> [u8; 4] {
    debug_assert!(x < 4096 && y < 4096);
    let r = (x % 256) as u8;
    let g = (y % 256) as u8;
    let b = (((x / 256) << 4) | (y / 256)) as u8;
    [b, g, r, 0xFF]
}

/// Fill a BGRA buffer with the coordinate fixture at `dims`. Used by
/// the harness when a case requests `Fixture::CoordEncoded`.
#[must_use]
pub fn coord_fixture_fill(dims: (u32, u32)) -> Vec<u8> {
    let (w, h) = dims;
    let mut buf = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let px = coord_encode_bgra(x, y);
            let off = ((y * w + x) * 4) as usize;
            buf[off..off + 4].copy_from_slice(&px);
        }
    }
    buf
}

/// Decode the (x, y) that a pixel's RGB claims to encode. Inverse of
/// [`coord_encode_bgra`]; lossy encoding may have shifted channel
/// values, so the recovered coordinate is approximate.
#[inline]
#[must_use]
pub fn coord_decode_bgra(px: &[u8]) -> (u32, u32) {
    let b = u32::from(px[0]);
    let g = u32::from(px[1]);
    let r = u32::from(px[2]);
    let x_hi = (b >> 4) & 0xF;
    let y_hi = b & 0xF;
    let x = (x_hi << 8) | r;
    let y = (y_hi << 8) | g;
    (x, y)
}

/// The letterbox-fit mapping from a readback pixel (in surface
/// coordinates) back to the original capture coordinate. Pure function
/// shared between the renderer and the CPU reference path so the
/// metric compare doesn't drift on aspect.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxMap {
    pub capture_dims: (u32, u32),
    pub surface_dims: (u32, u32),
    /// Inclusive-exclusive pixel rect inside `surface_dims` that holds
    /// the (non-letterbox-bar) video region.
    pub video_x0: u32,
    pub video_y0: u32,
    pub video_x1: u32,
    pub video_y1: u32,
}

impl LetterboxMap {
    /// Build the mapping for `capture_dims` letterbox-fit into
    /// `surface_dims`.
    #[must_use]
    pub fn new(capture_dims: (u32, u32), surface_dims: (u32, u32)) -> Self {
        let video_dims = letterbox_fit_dims(capture_dims, surface_dims);
        let video_x0 = (surface_dims.0 - video_dims.0) / 2;
        let video_y0 = (surface_dims.1 - video_dims.1) / 2;
        Self {
            capture_dims,
            surface_dims,
            video_x0,
            video_y0,
            video_x1: video_x0 + video_dims.0,
            video_y1: video_y0 + video_dims.1,
        }
    }

    #[inline]
    #[must_use]
    pub fn video_dims(&self) -> (u32, u32) {
        (self.video_x1 - self.video_x0, self.video_y1 - self.video_y0)
    }

    /// `(sx, sy)` in surface coordinates → `(cx, cy)` in capture
    /// coordinates if the pixel is inside the video rect; `None`
    /// otherwise (letterbox bar — skipped by the residual metric).
    #[inline]
    #[must_use]
    pub fn surface_to_capture(&self, sx: u32, sy: u32) -> Option<(f64, f64)> {
        if sx < self.video_x0 || sx >= self.video_x1 {
            return None;
        }
        if sy < self.video_y0 || sy >= self.video_y1 {
            return None;
        }
        let (vw, vh) = self.video_dims();
        let (cw, ch) = self.capture_dims;
        let u = (f64::from(sx - self.video_x0) + 0.5) / f64::from(vw);
        let v = (f64::from(sy - self.video_y0) + 0.5) / f64::from(vh);
        let cx = u * f64::from(cw) - 0.5;
        let cy = v * f64::from(ch) - 0.5;
        Some((cx, cy))
    }
}

/// RMS geometric residual in capture pixels between the recovered
/// (claimed) coordinate at each readback pixel and the expected
/// coordinate from the letterbox-fit mapping.
///
/// A correct round-trip on `Fixture::CoordEncoded` produces residuals
/// dominated by sub-pixel scatter from lossy encoding (typically
/// < 0.5 px at 4 Mbps H.264 on a flat-region fixture). A ghosted-copy
/// stride bug produces residuals of hundreds of pixels — the primary
/// catch this metric is designed for.
///
/// Skips:
/// - Pixels in the letterbox bars (no coordinate to compare against).
/// - Pixels where the recovered coordinate is wildly out-of-range
///   (channel-corruption from encode noise on solid color blocks at the
///   edges of the encoded source); above 4096 we treat the recovered
///   value as garbage rather than letting one outlier dominate the RMS.
///
/// `readback_bgra` is the rendered surface in BGRA at `surface_dims`.
#[must_use]
pub fn coord_fixture_residual_px_rms(
    readback_bgra: &[u8],
    surface_dims: (u32, u32),
    map: &LetterboxMap,
) -> f64 {
    assert_eq!(map.surface_dims, surface_dims);
    assert_eq!(readback_bgra.len(), (surface_dims.0 * surface_dims.1 * 4) as usize);
    assert!(
        map.video_x1 > map.video_x0 && map.video_y1 > map.video_y0,
        "degenerate letterbox map has no video rect — would silently return 0.0",
    );
    let mut sum_sq = 0.0_f64;
    let mut n = 0_u64;
    for sy in 0..surface_dims.1 {
        for sx in 0..surface_dims.0 {
            let Some((cx_expected, cy_expected)) = map.surface_to_capture(sx, sy) else {
                continue;
            };
            let off = ((sy * surface_dims.0 + sx) * 4) as usize;
            let (cx_claim, cy_claim) = coord_decode_bgra(&readback_bgra[off..off + 4]);
            if cx_claim >= 4096 || cy_claim >= 4096 {
                continue;
            }
            let dx = f64::from(cx_claim) - cx_expected;
            let dy = f64::from(cy_claim) - cy_expected;
            sum_sq += dx * dx + dy * dy;
            n += 1;
        }
    }
    if n == 0 {
        return 0.0;
    }
    (sum_sq / (n as f64)).sqrt()
}

// ---------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------

/// Mitchell-Netravali (B = C = 1/3) resize on a BGRA buffer.
/// Delegates to [`mitchell_filter_default`] — the same reference the
/// scaler's quality tests already validate against. The SME review
/// flagged using `image::imageops::resize(CatmullRom)` as a reference
/// because the B=0/C=0.5 drift relative to our actual scaler is real
/// (0.5–1 dB of PSNR floor noise).
#[must_use]
pub fn cpu_mitchell_resize_bgra(
    src_bgra: &[u8],
    src_dims: (u32, u32),
    dst_dims: (u32, u32),
) -> Vec<u8> {
    if src_dims == dst_dims {
        return src_bgra.to_vec();
    }
    // mitchell_filter_default expects 4-byte stride and does per-
    // channel work — channel order is irrelevant. The result has the
    // same channel order as the input.
    mitchell_filter_default(src_bgra, src_dims.0, src_dims.1, dst_dims.0, dst_dims.1)
}

/// Codec chroma siting. H.264 uses MPEG-2-style left-cosited horizontal
/// + interstitial vertical 4:2:0 chroma by default. HEVC defaults to
/// centered ("MPEG-1") 4:2:0 unless the bitstream's
/// `chroma_sample_loc_type` says otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSiting {
    /// H.264 default — chroma samples cosited with luma horizontally,
    /// interstitial vertically (centered between two luma rows).
    H264LeftInterstitial,
    /// HEVC default — chroma samples centered between four luma
    /// samples.
    HevcCentered,
    /// 4:4:4 — no subsampling, sitings are irrelevant.
    None,
}

/// Models the lossy YUV step in the reference: BGRA → BT.709 limited
/// Y'CbCr → chroma-subsample with the codec's siting → chroma-upsample
/// (bilinear) → back to BGRA.
///
/// Without this, a CPU-RGB-only reference would skip the chroma-shift
/// step entirely and the metric floor would be dominated by the
/// resulting drift rather than by real regressions. The SME review
/// named this as the most-overlooked source of reference unfairness.
///
/// For `ChromaSiting::None` (4:4:4) this returns a clone — chroma is
/// not subsampled so the roundtrip is lossless modulo float rounding.
/// For 10-bit, chroma is quantised to 10-bit precision before upsample;
/// the inputs/outputs are still 8-bit BGRA so the quantisation is a
/// no-op above the 8-bit floor (this is what a real Main10 chain looks
/// like to a downstream 8-bit consumer).
#[must_use]
pub fn cpu_chroma_roundtrip_bgra(
    src_bgra: &[u8],
    dims: (u32, u32),
    siting: ChromaSiting,
) -> Vec<u8> {
    if siting == ChromaSiting::None {
        return src_bgra.to_vec();
    }
    let (w, h) = (dims.0 as usize, dims.1 as usize);
    assert_eq!(src_bgra.len(), w * h * 4);

    // Step 1: BGRA → Y'CbCr using the BT.709 *full-range* matrix.
    //
    // The renderer's YUV→RGB shader applies the inverse expansion
    // (limited → full range) before producing RGB, so by the time the
    // harness reads back the surface, both readback and reference live
    // in full-range RGB. A limited-range reference would inject a
    // 16/235 level shift that doesn't exist in the renderer's output
    // and would mask real regressions. Keep the matrix full-range.
    let n = w * h;
    let mut y = vec![0.0_f32; n];
    let mut cb_full = vec![0.0_f32; n];
    let mut cr_full = vec![0.0_f32; n];
    for i in 0..n {
        let b = f32::from(src_bgra[i * 4]);
        let g = f32::from(src_bgra[i * 4 + 1]);
        let r = f32::from(src_bgra[i * 4 + 2]);
        y[i]       =  0.2126   * r + 0.7152   * g + 0.0722   * b;
        cb_full[i] = -0.114572 * r - 0.385428 * g + 0.5      * b + 128.0;
        cr_full[i] =  0.5      * r - 0.454153 * g - 0.045847 * b + 128.0;
    }

    // Step 2: subsample chroma to half resolution with siting offsets.
    let cw = w.div_ceil(2);
    let ch_h = h.div_ceil(2);
    let mut cb_sub = vec![0.0_f32; cw * ch_h];
    let mut cr_sub = vec![0.0_f32; cw * ch_h];
    for sy in 0..ch_h {
        for sx in 0..cw {
            // Center of the chroma sample in full-res luma coordinates.
            let (sample_x, sample_y) = match siting {
                ChromaSiting::H264LeftInterstitial => {
                    // Cosited horizontal (sample_x = 2*sx), interstitial
                    // vertical (sample_y = 2*sy + 0.5).
                    (2.0 * sx as f32, 2.0 * sy as f32 + 0.5)
                }
                ChromaSiting::HevcCentered => {
                    // Centered between four luma samples.
                    (2.0 * sx as f32 + 0.5, 2.0 * sy as f32 + 0.5)
                }
                ChromaSiting::None => unreachable!(),
            };
            cb_sub[sy * cw + sx] = bilinear_sample(&cb_full, w, h, sample_x, sample_y);
            cr_sub[sy * cw + sx] = bilinear_sample(&cr_full, w, h, sample_x, sample_y);
        }
    }

    // Step 3: upsample chroma back to full resolution with matching
    // siting (bilinear, which is what every real decoder + renderer
    // does on the output side).
    let mut cb_up = vec![0.0_f32; n];
    let mut cr_up = vec![0.0_f32; n];
    for ly in 0..h {
        for lx in 0..w {
            // Inverse of the subsample step — find the chroma-grid
            // coordinate this luma sample interpolates from.
            let (cx, cy) = match siting {
                ChromaSiting::H264LeftInterstitial => {
                    (lx as f32 / 2.0, (ly as f32 - 0.5) / 2.0)
                }
                ChromaSiting::HevcCentered => {
                    ((lx as f32 - 0.5) / 2.0, (ly as f32 - 0.5) / 2.0)
                }
                ChromaSiting::None => unreachable!(),
            };
            cb_up[ly * w + lx] = bilinear_sample(&cb_sub, cw, ch_h, cx, cy);
            cr_up[ly * w + lx] = bilinear_sample(&cr_sub, cw, ch_h, cx, cy);
        }
    }

    // Step 4: Y'CbCr → BGRA. Inverse of the matrix in step 1.
    let mut out = vec![0u8; w * h * 4];
    for i in 0..n {
        let yp = y[i];
        let cb = cb_up[i] - 128.0;
        let cr = cr_up[i] - 128.0;
        let r = yp                 + 1.5748   * cr;
        let g = yp - 0.18733 * cb - 0.46813  * cr;
        let b = yp + 1.8556  * cb;
        out[i * 4]     = clamp_u8(b);
        out[i * 4 + 1] = clamp_u8(g);
        out[i * 4 + 2] = clamp_u8(r);
        out[i * 4 + 3] = src_bgra[i * 4 + 3];
    }
    out
}

fn bilinear_sample(buf: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let xc = x.clamp(0.0, (w - 1) as f32);
    let yc = y.clamp(0.0, (h - 1) as f32);
    let x0 = xc.floor() as usize;
    let y0 = yc.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = xc - x0 as f32;
    let fy = yc - y0 as f32;
    let p00 = buf[y0 * w + x0];
    let p10 = buf[y0 * w + x1];
    let p01 = buf[y1 * w + x0];
    let p11 = buf[y1 * w + x1];
    let top = p00 * (1.0 - fx) + p10 * fx;
    let bot = p01 * (1.0 - fx) + p11 * fx;
    top * (1.0 - fy) + bot * fy
}

fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------
// Letterbox geometry
// ---------------------------------------------------------------------

/// Largest rectangle of `src`'s aspect ratio that fits inside `dst`.
/// Mirrors the renderer's [`letterbox_fit_dims`] (gpu/mod.rs) — pure
/// function, used from both the GPU side and the CPU reference path so
/// the metric compare doesn't drift on aspect.
#[must_use]
pub fn letterbox_fit_dims(src: (u32, u32), dst: (u32, u32)) -> (u32, u32) {
    let (sw, sh) = (src.0 as f64, src.1 as f64);
    let (dw, dh) = (dst.0 as f64, dst.1 as f64);
    let scale = (dw / sw).min(dh / sh);
    let w = (sw * scale).round() as u32;
    let h = (sh * scale).round() as u32;
    (w.max(1).min(dst.0), h.max(1).min(dst.1))
}

/// Resize a BGRA buffer at `src_dims` to `surface_dims` via Mitchell at
/// the letterbox-fit dims, then pad to `surface_dims` with black bars.
/// The renderer-side path goes capture → Mitchell-fit → bilinear-blit
/// to surface with the same letterbox math; producing the reference
/// the same way removes aspect drift from the metric comparison.
#[must_use]
pub fn cpu_letterbox_to_surface_bgra(
    src_bgra: &[u8],
    src_dims: (u32, u32),
    surface_dims: (u32, u32),
) -> Vec<u8> {
    let video_dims = letterbox_fit_dims(src_dims, surface_dims);
    let resized = if video_dims == src_dims {
        src_bgra.to_vec()
    } else {
        cpu_mitchell_resize_bgra(src_bgra, src_dims, video_dims)
    };
    let (sw, sh) = surface_dims;
    let (vw, vh) = video_dims;
    let x0 = (sw - vw) / 2;
    let y0 = (sh - vh) / 2;
    let mut out = vec![0u8; (sw * sh * 4) as usize];
    for row in 0..vh {
        let dst_off = (((y0 + row) * sw + x0) * 4) as usize;
        let src_off = (row * vw * 4) as usize;
        out[dst_off..dst_off + (vw * 4) as usize]
            .copy_from_slice(&resized[src_off..src_off + (vw * 4) as usize]);
    }
    out
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_roundtrip_recovers_position() {
        for x in [0u32, 1, 127, 128, 255, 256, 1023, 2049, 3839, 4095] {
            for y in [0u32, 1, 127, 128, 255, 256, 1023, 2049, 1439, 4095] {
                let px = coord_encode_bgra(x, y);
                let (xr, yr) = coord_decode_bgra(&px);
                assert_eq!((xr, yr), (x, y), "round-trip failed at ({x}, {y})");
            }
        }
    }

    #[test]
    fn coord_residual_zero_on_exact_letterbox_fit() {
        // 100×100 capture letterbox-fit into 100×100 surface = identity
        // mapping. Filling the surface from coord_fixture_fill at the
        // same dims must give zero RMS residual.
        let dims = (100u32, 100u32);
        let fixture = coord_fixture_fill(dims);
        let map = LetterboxMap::new(dims, dims);
        let r = coord_fixture_residual_px_rms(&fixture, dims, &map);
        // The map uses center-of-pixel sampling (`+0.5` / `-0.5`),
        // which is the correct convention; in the identity case
        // sx==cx_expected to within rounding, so the recovered
        // coordinate (which is integer because the fixture writes
        // integer indices) differs from the float expected by ≤ 0.5 px
        // per axis. RMS over the buffer is bounded by √(0.25+0.25) ≈ 0.71.
        assert!(r < 0.75, "identity-letterbox residual {} too high", r);
    }

    #[test]
    fn coord_residual_huge_on_horizontal_shift() {
        // Shift the fixture horizontally by 50 px before passing it
        // through the residual computation. Expected residual = 50.
        let dims = (100u32, 100u32);
        let mut shifted = coord_fixture_fill(dims);
        // Copy each row from x+50 (wrapping is fine; we just want a
        // structured discrepancy that the metric must catch).
        let row_bytes = (dims.0 * 4) as usize;
        let mut tmp = vec![0u8; row_bytes];
        for y in 0..dims.1 as usize {
            let off = y * row_bytes;
            tmp.copy_from_slice(&shifted[off..off + row_bytes]);
            for x in 0..dims.0 as usize {
                let src_x = (x + 50) % dims.0 as usize;
                shifted[off + x * 4..off + x * 4 + 4]
                    .copy_from_slice(&tmp[src_x * 4..src_x * 4 + 4]);
            }
        }
        let map = LetterboxMap::new(dims, dims);
        let r = coord_fixture_residual_px_rms(&shifted, dims, &map);
        assert!(r > 30.0, "shifted-fixture residual {} too low — metric is blind", r);
    }

    #[test]
    fn ssim_self_is_one() {
        let buf = vec![42u8; 64 * 64 * 4];
        let s = ssim_rgb(&buf, &buf, 64, 64);
        assert!((s - 1.0).abs() < 1e-9, "ssim self = {}", s);
    }

    #[test]
    fn psnr_self_is_infinite() {
        let buf = vec![42u8; 16 * 16 * 4];
        assert_eq!(psnr_db_rgb(&buf, &buf), f64::INFINITY);
        assert_eq!(psnr_db_y_bgra(&buf, &buf), f64::INFINITY);
    }

    #[test]
    fn chroma_roundtrip_444_is_identity() {
        let dims = (32u32, 32u32);
        let src = coord_fixture_fill(dims);
        let out = cpu_chroma_roundtrip_bgra(&src, dims, ChromaSiting::None);
        assert_eq!(out, src);
    }

    #[test]
    fn chroma_roundtrip_420_blurs_chroma_but_preserves_luma_approximately() {
        // The fixture isn't pure-luma so Y' will shift slightly on the
        // RGB→YUV→RGB round-trip due to matrix rounding. PSNR on Y
        // should still be very high (chroma-roundtrip doesn't touch
        // luma at all).
        let dims = (64u32, 64u32);
        let src = coord_fixture_fill(dims);
        let out = cpu_chroma_roundtrip_bgra(&src, dims, ChromaSiting::H264LeftInterstitial);
        assert_eq!(out.len(), src.len());
        let psnr_y = psnr_db_y_bgra(&src, &out);
        assert!(psnr_y > 40.0, "luma should survive chroma roundtrip; got psnr_y = {} dB", psnr_y);
    }

    #[test]
    fn mitchell_resize_identity_is_copy() {
        let buf = vec![17u8; 32 * 32 * 4];
        let out = cpu_mitchell_resize_bgra(&buf, (32, 32), (32, 32));
        assert_eq!(out, buf);
    }

    #[test]
    fn chroma_roundtrip_h264_and_hevc_siting_differ() {
        // The whole purpose of the ChromaSiting enum is that H.264
        // (left-cosited horizontal) and HEVC (centered horizontal)
        // produce different chroma reconstructions. Verify the two
        // variants don't collapse to identical output. A horizontal
        // red/blue split is the most direct fixture for catching a
        // half-pixel horizontal shift.
        let dims = (32u32, 32u32);
        let mut src = vec![0u8; (dims.0 * dims.1 * 4) as usize];
        for y in 0..dims.1 {
            for x in 0..dims.0 {
                let off = ((y * dims.0 + x) * 4) as usize;
                if x < dims.0 / 2 {
                    src[off..off + 4].copy_from_slice(&[0, 0, 255, 255]); // BGRA = red
                } else {
                    src[off..off + 4].copy_from_slice(&[255, 0, 0, 255]); // BGRA = blue
                }
            }
        }
        let h264 = cpu_chroma_roundtrip_bgra(&src, dims, ChromaSiting::H264LeftInterstitial);
        let hevc = cpu_chroma_roundtrip_bgra(&src, dims, ChromaSiting::HevcCentered);
        assert_ne!(h264, hevc, "H.264 and HEVC siting must produce different chroma reconstructions");
        // The horizontal boundary at x=dims.0/2 should sit at slightly
        // different sub-pixel positions between the two — pick a row
        // and confirm the boundary pixel's chroma differs.
        let mid_y = (dims.1 / 2) as usize;
        let boundary_x = (dims.0 / 2) as usize;
        let off = ((mid_y * dims.0 as usize + boundary_x) * 4) as usize;
        assert_ne!(
            (h264[off], h264[off + 1], h264[off + 2]),
            (hevc[off], hevc[off + 1], hevc[off + 2]),
            "boundary pixel chroma must reflect siting difference",
        );
    }

    #[test]
    fn letterbox_fit_landscape_in_landscape_matches_renderer() {
        // 16:9 fit in 4:3 surface — width-limited.
        assert_eq!(letterbox_fit_dims((1920, 1080), (800, 600)), (800, 450));
        // 4:3 fit in 16:9 surface — height-limited.
        assert_eq!(letterbox_fit_dims((800, 600), (1920, 1080)), (1440, 1080));
        // Same aspect → fills.
        assert_eq!(letterbox_fit_dims((1280, 720), (1920, 1080)), (1920, 1080));
    }
}
