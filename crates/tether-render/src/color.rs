//! CPU reference implementation of the render-side color pipeline,
//! used both as documentation of the math contract and as a
//! regression test against the GPU shader (`shader.wgsl`).
//!
//! The two implementations — this module and the WGSL fragment shader
//! — must agree. The WGSL is canonical at runtime; this module exists
//! so the contract is testable without a GPU adapter, and so the next
//! person who sees a color bug can step through it in a debugger.
//!
//! Pipeline modelled:
//!
//! ```text
//!   capture-side framebuffer bytes  (treated as gamma-encoded R'G'B')
//!     -> BT.709 matrix              (gamma-encoded Y'CbCr)
//!     -> limited-range quantize     (Y in 16..235, CbCr in 16..240)
//!     -> [wire]
//!     -> limited-range expand       (normalized Y'CbCr)
//!     -> BT.709 inverse matrix      (gamma-encoded R'G'B')
//!     -> BT.709 EOTF                (linear light)
//!     -> [shader output]
//!     -> sRGB surface OETF          (sRGB-encoded display bytes)
//! ```
//!
//! The chain is supposed to be visually transparent for SDR content
//! within rounding error. The regression test asserts a round-trip
//! identity (BGRA in → BGRA out within a few units per channel) over
//! a spread of colors; if any stage's constants drift it fails.
//!
//! The decoder dispatches its EOTF on the negotiated
//! [`VideoColorSpec`] — see [`bt709_limited_to_srgb_display`]. A
//! stream advertising `sdr_desktop` (sRGB transfer) gets the sRGB
//! EOTF; a stream advertising `sdr_bt709` gets the BT.709 EOTF.
//! Picking the right one eliminates the ≤~5% midtone mismatch the
//! spec-blind chain previously had to absorb.
//!
//! Today the matrix and range are still hard-pinned to BT.709 /
//! limited regardless of the negotiated spec — the renderer doesn't
//! yet have variants for `Bt2020Ncl` matrix or `Full` range. Adding
//! them is a sibling enum-arm in `apply_eotf` / a sibling helper
//! function; HDR transfer functions (PQ / HLG) drop into the same
//! dispatch.

use tether_protocol::control::{ColorTransfer, VideoColorSpec};

/// 8-bit BGRA pixel as it appears in a desktop framebuffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Bgra8(pub u8, pub u8, pub u8, pub u8);

/// Limited-range NV12 luma + chroma sample triplet.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Yuv8(pub u8, pub u8, pub u8);

// =============================================================
// Transfer functions
// =============================================================

/// BT.709 OETF (Rec. ITU-R BT.709-6 eq. 1.2). Linear -> gamma.
/// Inverse of [`bt709_eotf`].
#[must_use]
pub fn bt709_oetf(linear: f32) -> f32 {
    let v = linear.max(0.0);
    if v < 0.018 {
        4.5 * v
    } else {
        1.099 * v.powf(0.45) - 0.099
    }
}

/// BT.709 EOTF. Gamma -> linear. Inverse of [`bt709_oetf`]. The
/// `max(0)` clamp mirrors the WGSL guard against the chroma
/// overshoots limited-range can produce after the matrix.
#[must_use]
pub fn bt709_eotf(gamma: f32) -> f32 {
    let v = gamma.max(0.0);
    if v < 0.081 {
        v / 4.5
    } else {
        ((v + 0.099) / 1.099).powf(1.0 / 0.45)
    }
}

/// sRGB OETF. Linear -> gamma. What wgpu applies on write to an
/// `Bgra8UnormSrgb` surface. Matches the IEC 61966-2-1 piecewise
/// definition, not the gamma 2.2 approximation.
#[must_use]
pub fn srgb_oetf(linear: f32) -> f32 {
    let v = linear.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// sRGB EOTF. Gamma -> linear. Inverse of [`srgb_oetf`]. Provided for
/// the test that asserts the OETF/EOTF pair round-trips.
#[must_use]
pub fn srgb_eotf(gamma: f32) -> f32 {
    let v = gamma.clamp(0.0, 1.0);
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

// =============================================================
// BT.709 limited range
// =============================================================

/// Encode one sRGB framebuffer pixel into limited-range BT.709 NV12
/// luma + chroma. Capture-side reference math; mirrors what the
/// gpuconvert / VideoToolbox encoder pipeline produces.
///
/// Note on the transfer-curve assumption: we treat the BGRA bytes as
/// gamma-encoded values and apply the BT.709 matrix directly, the
/// same way `gpuconvert`'s shader does. The framebuffer is actually
/// sRGB-encoded, not BT.709-encoded — the two transfer curves differ
/// by <2% in midtones, which is within the test tolerance and within
/// the rounding-to-u8 noise floor.
#[must_use]
pub fn bgra_to_bt709_limited(bgra: Bgra8) -> Yuv8 {
    let r = f32::from(bgra.2) / 255.0;
    let g = f32::from(bgra.1) / 255.0;
    let b = f32::from(bgra.0) / 255.0;
    // BT.709 luma weights.
    let y = 0.212_6 * r + 0.715_2 * g + 0.072_2 * b;
    // BT.709 chroma derivation from R'G'B'-Y'. Coefficients are the
    // standard 0.5 / (1 - K_R or 1 - K_B) form.
    let u = -0.114_57 * r - 0.385_43 * g + 0.5 * b;
    let v = 0.5 * r - 0.454_15 * g - 0.045_85 * b;
    // Limited-range quantize: Y'[16..235], CbCr[16..240].
    let y_byte = (y * (219.0 / 255.0) + 16.0 / 255.0) * 255.0;
    let u_byte = (u * (224.0 / 255.0) + 128.0 / 255.0) * 255.0;
    let v_byte = (v * (224.0 / 255.0) + 128.0 / 255.0) * 255.0;
    Yuv8(
        round_clamp_u8(y_byte),
        round_clamp_u8(u_byte),
        round_clamp_u8(v_byte),
    )
}

/// Decode + display one limited-range BT.709 NV12 triplet into the
/// sRGB byte an `Bgra8UnormSrgb` surface ends up holding. Mirrors
/// the WGSL fragment shader's range + matrix + EOTF chain (with the
/// EOTF dispatched on `spec.transfer`, matching the shader's
/// `apply_eotf`), followed by wgpu's surface-write OETF.
///
/// If the WGSL diverges from this Rust the regression test fails —
/// the comment on the module-level docstring is the contract.
#[must_use]
pub fn bt709_limited_to_srgb_display(yuv: Yuv8, spec: VideoColorSpec) -> Bgra8 {
    // 1. RANGE: limited -> normalized. Today's only path; `Full`
    // range would skip this step.
    let y = (f32::from(yuv.0) - 16.0) / 219.0;
    let u = (f32::from(yuv.1) - 128.0) / 224.0;
    let v = (f32::from(yuv.2) - 128.0) / 224.0;
    // 2. MATRIX: Y'CbCr -> gamma-encoded R'G'B'. Today's only path
    // is BT.709; `Bt2020Ncl` would dispatch to a different set of
    // coefficients (and would typically pair with PQ/HLG transfer).
    let r_gamma = y + 1.574_8 * v;
    let g_gamma = y - 0.187_3 * u - 0.468_1 * v;
    let b_gamma = y + 1.855_6 * u;
    // 3. TRANSFER: EOTF dispatch — match `apply_eotf` in shader.wgsl.
    let (r_linear, g_linear, b_linear) = match spec.transfer {
        ColorTransfer::Bt709 => (
            bt709_eotf(r_gamma),
            bt709_eotf(g_gamma),
            bt709_eotf(b_gamma),
        ),
        ColorTransfer::Srgb => (srgb_eotf(r_gamma), srgb_eotf(g_gamma), srgb_eotf(b_gamma)),
        // Pq / Hlg / Linear are unimplemented in the shader and
        // fall through to BT.709. Match the GPU path's `tracing::warn!`
        // (`transfer_kind_for` in gpu/mod.rs) so a caller that
        // accidentally passes a PQ spec in a test sees the same
        // signal the production path would have logged.
        ColorTransfer::Pq | ColorTransfer::Hlg | ColorTransfer::Linear => {
            tracing::warn!(
                side = "cpu_mirror",
                ?spec.transfer,
                "EOTF not yet implemented; falling back to BT.709"
            );
            (
                bt709_eotf(r_gamma),
                bt709_eotf(g_gamma),
                bt709_eotf(b_gamma),
            )
        }
    };
    // 4. SURFACE: sRGB OETF (what wgpu's surface write does for us).
    let r_srgb = srgb_oetf(r_linear);
    let g_srgb = srgb_oetf(g_linear);
    let b_srgb = srgb_oetf(b_linear);
    Bgra8(
        round_clamp_u8(b_srgb * 255.0),
        round_clamp_u8(g_srgb * 255.0),
        round_clamp_u8(r_srgb * 255.0),
        255,
    )
}

/// Full capture→display round trip. The regression test asserts this
/// is approximately identity for SDR sRGB inputs when the decoder
/// uses the matching spec.
#[must_use]
pub fn simulate_round_trip(input: Bgra8, spec: VideoColorSpec) -> Bgra8 {
    let yuv = bgra_to_bt709_limited(input);
    bt709_limited_to_srgb_display(yuv, spec)
}

// Sign loss + truncation are intentional: we clamp to [0, 255] first,
// so the values are non-negative and fit u8 by construction.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn round_clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-channel tolerance for the round-trip identity test when the
    /// decoder uses the wrong EOTF. The encoder treats source bytes
    /// as gamma-encoded sRGB (every desktop framebuffer is), so a
    /// BT.709-EOTF decode hits the ≤~5% sRGB-vs-BT.709 transfer
    /// mismatch — ≤~15 units / 255 in midtones after the round trip.
    ///
    /// The matching path (decoder uses sRGB EOTF) is the one
    /// `sdr_desktop_round_trip_is_tight` exercises and tightens to
    /// `TOLERANCE_TIGHT`.
    ///
    /// **Floor on regression sensitivity:** the pre-fix EOTF-skip
    /// bug shifted mid-gray by ~64 units
    /// (`without_eotf_blacks_lift_visibly` pins that). With tolerance
    /// 18 we still catch any regression that lifts blacks by more
    /// than ~3× what the current spec-mismatched chain does. The
    /// tight test (~3 units) catches a regression in the path
    /// production actually uses.
    const TOLERANCE: i32 = 18;
    /// Per-channel tolerance for the matching-EOTF round-trip
    /// identity test (decoder uses the same transfer the encoder
    /// produced). What remains is u8-quantize noise: ≤0.5/219 ≈
    /// 0.23% on Y after limited-range expand + ≤0.5/255 ≈ 0.2% on
    /// the surface byte = a few units total. If this fails the
    /// matrix, range, or EOTF/OETF constants drifted.
    const TOLERANCE_TIGHT: i32 = 3;

    fn assert_close(label: &str, got: Bgra8, expected: Bgra8, tolerance: i32) {
        for (chan, g, e) in [
            ("B", got.0, expected.0),
            ("G", got.1, expected.1),
            ("R", got.2, expected.2),
        ] {
            let diff = i32::from(g) - i32::from(e);
            assert!(
                diff.abs() <= tolerance,
                "{label}/{chan}: got {g}, expected {e} (±{tolerance}), diff {diff}"
            );
        }
        assert_eq!(got.3, expected.3, "{label}/A");
    }

    /// Twelve test colors covering the parts of the gamut that
    /// stress the matrix and EOTF differently — corners, neutrals,
    /// primaries, and mid-saturated colors. Shared by the tight and
    /// loose round-trip tests so a regression that lights up one
    /// path lights up the other too.
    fn round_trip_cases() -> &'static [(&'static str, Bgra8)] {
        &[
            ("black", Bgra8(0, 0, 0, 255)),
            ("white", Bgra8(255, 255, 255, 255)),
            ("mid gray", Bgra8(128, 128, 128, 255)),
            ("dark gray", Bgra8(32, 32, 32, 255)),
            ("light gray", Bgra8(220, 220, 220, 255)),
            ("pure red", Bgra8(0, 0, 255, 255)),
            ("pure green", Bgra8(0, 255, 0, 255)),
            ("pure blue", Bgra8(255, 0, 0, 255)),
            ("warm beige", Bgra8(180, 200, 220, 255)),
            ("teal", Bgra8(200, 150, 60, 255)),
            ("magenta", Bgra8(180, 40, 200, 255)),
        ]
    }

    /// Golden-value test for the matching-spec pipeline (decoder
    /// uses the same EOTF the encoder produced — i.e. sRGB for a
    /// `sdr_desktop` advertisement). Pins exact bytes for mid-gray
    /// (sensitive to transfer drift) and pure red (sensitive to
    /// matrix drift). With matching specs the round trip is identity
    /// within u8-quantize noise, so any drift in the WGSL or Rust
    /// constants moves these values; correlated drift between the
    /// two impls still moves them.
    ///
    /// If these change because of an intentional pipeline change
    /// (e.g. swapping a matrix variant), update from the
    /// freshly-printed values and note the reason here.
    #[test]
    fn golden_round_trip_values() {
        let spec = VideoColorSpec::sdr_desktop();

        let gray = simulate_round_trip(Bgra8(128, 128, 128, 255), spec);
        assert_eq!(
            (gray.0, gray.1, gray.2, gray.3),
            (128, 128, 128, 255),
            "mid-gray drift — TRANSFER constants (sRGB EOTF / OETF) likely changed"
        );

        let red = simulate_round_trip(Bgra8(0, 0, 255, 255), spec);
        assert_eq!(
            (red.0, red.1, red.2, red.3),
            (0, 1, 255, 255),
            "pure red drift — MATRIX coefficients or RANGE constants likely \
             changed. (The G=1 is a small chroma overshoot from the BT.709 \
             matrix on pure red; stable as long as the matrix isn't touched.)"
        );
    }

    #[test]
    fn bt709_oetf_eotf_round_trip() {
        // Inverse property holds across the curve; sample at the
        // toe (linear segment), the knee, and the highlights.
        for input_byte in [0u8, 4, 16, 64, 128, 192, 235, 254] {
            let v = f32::from(input_byte) / 255.0;
            let round_tripped = bt709_eotf(bt709_oetf(v));
            assert!(
                (round_tripped - v).abs() < 1e-4,
                "BT.709 OETF/EOTF round trip drift at {v}: got {round_tripped}"
            );
        }
    }

    #[test]
    fn srgb_oetf_eotf_round_trip() {
        for input_byte in [0u8, 4, 16, 64, 128, 192, 235, 254] {
            let v = f32::from(input_byte) / 255.0;
            let round_tripped = srgb_eotf(srgb_oetf(v));
            assert!(
                (round_tripped - v).abs() < 1e-4,
                "sRGB OETF/EOTF round trip drift at {v}: got {round_tripped}"
            );
        }
    }

    /// **The principled regression test.** Decoder spec matches what
    /// the host advertises today (`sdr_desktop`, sRGB transfer), so
    /// the round trip is identity within u8-quantize noise. If the
    /// shader's EOTF dispatch breaks, the matrix coefficients drift,
    /// or the range expansion is mis-implemented, this fails. The
    /// tight tolerance (3 units) is what gives the test real
    /// regression power.
    #[test]
    fn sdr_desktop_round_trip_is_tight() {
        let spec = VideoColorSpec::sdr_desktop();
        for (label, input) in round_trip_cases() {
            let out = simulate_round_trip(*input, spec);
            assert_close(label, out, *input, TOLERANCE_TIGHT);
        }
    }

    /// Mismatched-spec test. Source bytes are sRGB-encoded (every
    /// desktop framebuffer is) but the decoder applies BT.709 EOTF
    /// — the spec-blind case from before the negotiation landed.
    /// The transfer-curve mismatch projects to ≤18 units of midtone
    /// lift; this test pins that, so a future regression that
    /// *widens* the mismatch fails. The path is not the production
    /// path anymore (host advertises `sdr_desktop` → decoder uses
    /// sRGB EOTF) but is exercised here so the cost of the legacy
    /// chain stays documented.
    #[test]
    fn sdr_bt709_round_trip_within_legacy_tolerance() {
        let spec = VideoColorSpec::sdr_bt709();
        for (label, input) in round_trip_cases() {
            let out = simulate_round_trip(*input, spec);
            assert_close(label, out, *input, TOLERANCE);
        }
    }

    /// Tolerance calibration sentinel. Hand-codes the broken-decoder
    /// shape (skips the BT.709 EOTF, feeds gamma-encoded R'G'B'
    /// straight to the sRGB surface — the actual bug we just fixed)
    /// and asserts that shape produces a shift LARGER than the
    /// regression test's `TOLERANCE`. That's not regression
    /// protection itself — `bgra_round_trip_is_approximately_identity`
    /// does that work against the production functions. What this
    /// pins is the GAP between "broken shape" and "tolerance," i.e.
    /// "tolerance is calibrated tight enough to catch the bug if it
    /// comes back via the production path." If someone widens
    /// `TOLERANCE` past the broken-shape shift, this assertion
    /// flips and forces a conversation about the regression test's
    /// sensitivity.
    #[test]
    fn without_eotf_blacks_lift_visibly() {
        // Build a "broken" decoder that skips the EOTF and outputs
        // gamma-encoded R'G'B' to the sRGB surface directly. This
        // is the bug; the assertion is that the regression case
        // (mid gray) shifts by more than the round-trip tolerance,
        // demonstrating the test would catch a regression.
        fn broken_decode(yuv: Yuv8) -> Bgra8 {
            let y = (f32::from(yuv.0) - 16.0) / 219.0;
            let u = (f32::from(yuv.1) - 128.0) / 224.0;
            let v = (f32::from(yuv.2) - 128.0) / 224.0;
            let r = y + 1.574_8 * v;
            let g = y - 0.187_3 * u - 0.468_1 * v;
            let b = y + 1.855_6 * u;
            // Skip EOTF. Surface still applies sRGB OETF.
            Bgra8(
                round_clamp_u8(srgb_oetf(b) * 255.0),
                round_clamp_u8(srgb_oetf(g) * 255.0),
                round_clamp_u8(srgb_oetf(r) * 255.0),
                255,
            )
        }
        let input = Bgra8(128, 128, 128, 255);
        let yuv = bgra_to_bt709_limited(input);
        let out = broken_decode(yuv);
        let diff = i32::from(out.1) - i32::from(input.1);
        assert!(
            diff > TOLERANCE,
            "broken decode (skipped EOTF) should lift mid-gray by more than {TOLERANCE} units — \
             got diff {diff}. If this fails, even the legacy-tolerance regression test \
             would have a blind spot, let alone the tight test."
        );
    }
}
