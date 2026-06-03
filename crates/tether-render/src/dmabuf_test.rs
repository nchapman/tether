//! Linux DMA-BUF zero-copy round-trip test cells.
//!
//! Each `#[test]` here is a [`RoundtripCase`] struct literal plus an
//! assertion block — the harness in `test_harness.rs` owns everything
//! else (fixture loading, encode dispatch, decoder loop, production-
//! renderer dispatch via `Gpu::new_headless`, CPU reference build, and
//! metric computation). Adding a new cell means writing one struct
//! literal; this file is the matrix.
//!
//! ## Matrix
//!
//! - **Identity rows** (`*_identity`): capture == encode == surface.
//!   Smallest possible chain — encoder + decoder + renderer at native
//!   dims. Isolates regressions in any of those three from the
//!   scaler-active rows.
//! - **Host scaler** (`h264_8bit_host_scaler`): capture > encode ==
//!   surface. The host's Mitchell scaler is the only scaling stage.
//! - **Surface below video** (`h264_8bit_surface_below_video`):
//!   capture == encode > surface. Engages the blit pass's non-identity
//!   `letterbox_scale` branch (`gpu/mod.rs:1141`) without engaging
//!   `need_upscale`. Separate arithmetic from the upscale branch.
//! - **Client upscale** (`*_client_upscale`): capture == encode <
//!   surface. The renderer's Mitchell upscale stage is the only
//!   scaling stage. This is the cell whose existence would have
//!   caught the steady-state 4-overlapping-copies bug.
//! - **Full chain** (`*_full_chain`): capture > encode < surface, with
//!   surface aspect ≠ encode aspect so letterbox bars appear. Both
//!   scaler stages plus letterboxing in one cell.
//!
//! Bit-depth coverage: 4:2:0 8-bit (H.264 + HEVC), 4:4:4 8-bit (HEVC),
//! 10-bit (HEVC). 10-bit `_full_chain` is intentionally omitted — the
//! P010 scaler input path doesn't exist in production. 10-bit gets an
//! identity row (encoder + decoder + Biplanar16 import) and a
//! `client_upscale` row (renderer upscale of decoded 10-bit content).
//! `surface_below` is feasible at 10-bit (capture == encode, no
//! scaler needed) but out of scope for this step; add when a P010-
//! capable box is in CI.
//!
//! ## Metrics
//!
//! Three assertions per cell, in priority order:
//!   1. **Geometric residual** (`CoordEncoded` fixtures only): how
//!      many pixels did the recovered (x, y) coordinate drift? Lossy
//!      encode moves them by < 1 px; stride bugs move them by
//!      hundreds.
//!   2. **SSIM**: per-window structural similarity. Catches the
//!      next-bug-we-haven't-thought-of.
//!   3. **BT.709 Y-channel PSNR**: isolates the luminance pipeline
//!      from the 4:2:0 chroma noise floor.
//!
//! Per-cell floors live in the `FLOOR_*` constants in this file. They
//! were derived from a green-main run on real VAAPI hardware
//! (`measured − 0.01` for SSIM, `measured − 1.5 dB` for PSNR-Y,
//! roughly `observed × 1.25` for geometric residual). The numbers
//! catch 1%-class regressions while absorbing hardware-variant drift;
//! a real pipeline bug overshoots them by an order of magnitude.
//!
//! ## Marked `#[ignore]`
//!
//! Every roundtrip cell needs VAAPI hardware + a Vulkan adapter
//! advertising `VULKAN_EXTERNAL_MEMORY_DMA_BUF`. Lavapipe + most CI
//! environments lack the latter. Run on real hardware with:
//!   `cargo test -p tether-render -- --ignored roundtrip`

#![cfg(target_os = "linux")]
#![allow(clippy::cast_possible_truncation, clippy::cast_lossless)]

use tether_codec::vaapi::VaapiEncoder;
use tether_codec::Encoder;
use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoColorSpec, VideoProfile};

use crate::test_harness::{
    dump_failure_diagnostics, run_roundtrip, Capability, Fixture, Floors, RoundtripCase,
    RoundtripOutcome, RoundtripResult,
};

// =====================================================================
// Per-cell assertion helper
// =====================================================================

/// Apply the three (or four, with steady-state) assertions a cell
/// shares. Keeps each `#[test]` body short: a struct literal + one
/// call. Logs SKIP loudly with the missing capability so a CI runner
/// can distinguish hardware gaps from real passes.
fn assert_outcome(case: &RoundtripCase, result: RoundtripResult) {
    let outcome: RoundtripOutcome = match result {
        RoundtripResult::Skip { capability, detail } => {
            eprintln!("SKIPPED {} (missing {capability:?}): {detail}", case.name);
            return;
        }
        RoundtripResult::Ok(o) => o,
    };
    eprintln!(
        "[{}] ssim={:.4} psnr_y={:.2}dB geom_residual={:.3}px steady={:?}",
        case.name,
        outcome.ssim,
        outcome.psnr_y_db,
        outcome.geometric_residual_px_rms,
        outcome.steady_state_delta,
    );

    // Collect every failed assertion before panicking. Without this,
    // the first failing assert hides the others, and (more
    // importantly) it would skip the diagnostics dump so a CI failure
    // would land without the readback/reference/diff/heatmap
    // artifacts that make remote triage tractable.
    let mut failures: Vec<String> = Vec::new();
    if matches!(case.fixture, Fixture::CoordEncoded)
        && outcome.geometric_residual_px_rms > case.floors.geom_px
    {
        failures.push(format!(
            "geometric residual {} px > floor {} px — stride or UV-addressing bug?",
            outcome.geometric_residual_px_rms, case.floors.geom_px,
        ));
    }
    if outcome.ssim < case.floors.ssim {
        failures.push(format!(
            "ssim {} < floor {}",
            outcome.ssim, case.floors.ssim,
        ));
    }
    if outcome.psnr_y_db < case.floors.psnr_y_db {
        failures.push(format!(
            "psnr_y {} dB < floor {} dB",
            outcome.psnr_y_db, case.floors.psnr_y_db,
        ));
    }
    if let (Some(eps), Some(delta)) = (case.assert_steady_state_eps, outcome.steady_state_delta) {
        if delta > eps {
            failures.push(format!(
                "steady-state MAE {} > {} between last two rendered frames",
                delta, eps,
            ));
        }
    }

    if !failures.is_empty() {
        dump_failure_diagnostics(case, &outcome);
        panic!(
            "[{}] {} metric(s) failed:\n  - {}",
            case.name,
            failures.len(),
            failures.join("\n  - ")
        );
    }
}

// =====================================================================
// Colour-decode cells (Fixture::ColorBars)
//
// SSIM / geometric residual run on the near-constant-chroma
// CoordEncoded fixture, so they can't catch a hue cast or Cb/Cr swap.
// These drive the shared red/green/blue/white colour-bar pattern
// through the SAME production host pipeline (BGRA → gpuconvert
// NV12/P010/XYUV/XV30 bridge → VAAPI encode → decode → render) and
// assert each bar reconstructs to its colour on the readback. The
// white bar is the regression guard for the host-side "colors are
// wrong" hue-cast bug. Uniform with macOS `iosurface_test` and the
// Windows `d3d11` colour cells via `crate::color_fixture`.
// =====================================================================

/// SKIP-aware colour-bar assertion on the rendered readback (BGRA,
/// matching the production `Bgra8UnormSrgb` swapchain).
fn assert_colorbars_outcome(case: &RoundtripCase, result: RoundtripResult) {
    match result {
        RoundtripResult::Skip { capability, detail } => {
            eprintln!("SKIPPED {} (missing {capability:?}): {detail}", case.name);
        }
        RoundtripResult::Ok(outcome) => {
            crate::color_fixture::assert_colorbars(
                case.name,
                &outcome.readback_bgra,
                case.surface_dims.0,
                case.surface_dims.1,
                crate::color_fixture::ChannelOrder::Bgra,
            );
        }
    }
}

/// Build an identity (no-scaler) colour-bar case at 1920×1200 for
/// `profile` with the given capability prereqs. `floors` is unused by
/// the colour assertion but required by the struct; `FLOOR_IDENTITY`
/// keeps it honest if a future edit routes this through `assert_outcome`.
fn colorbars_case(
    name: &'static str,
    profile: VideoProfile,
    requires: &'static [Capability],
) -> RoundtripCase {
    RoundtripCase {
        name,
        profile,
        fixture: Fixture::ColorBars,
        capture_dims: (1920, 1200),
        encode_dims: (1920, 1200),
        surface_dims: (1920, 1200),
        frames_encoded: 6,
        assert_steady_state_eps: None,
        color_space: VideoColorSpec::sdr_desktop(),
        requires,
        floors: FLOOR_IDENTITY,
    }
}

#[test]
#[ignore = "requires VAAPI H.264 + Vulkan dma-buf import"]
fn roundtrip_colorbars_h264_8bit() {
    let case = colorbars_case(
        "colorbars_h264_8bit",
        H264_8BIT_420,
        &[Capability::VaapiH264, Capability::VulkanDmaBufImport],
    );
    assert_colorbars_outcome(&case, run_roundtrip(&case));
}

/// AV1 4:2:0 8-bit colour bars. AV1 is a production-negotiated profile
/// (`PROFILE_PREFERENCE` ranks AV1 above HEVC Main / H.264) and decodes
/// to `RenderLayout::Biplanar8` — the same NV12 import as H.264/HEVC, so
/// the render side is codec-agnostic. The codec-level
/// `av1_main_dmabuf_roundtrip` only pushes a uniform grey frame, which
/// can't catch a hue cast or Cb/Cr swap; this is the AV1 colour guard.
/// SKIPs (via `Capability::VaapiAv1`) on drivers without AV1 encode.
#[test]
#[ignore = "requires VAAPI AV1 encode (Intel Arc / AMD RDNA3+) + Vulkan dma-buf import"]
fn roundtrip_colorbars_av1_8bit() {
    let case = colorbars_case(
        "colorbars_av1_8bit",
        AV1_8BIT_420,
        &[Capability::VaapiAv1, Capability::VulkanDmaBufImport],
    );
    assert_colorbars_outcome(&case, run_roundtrip(&case));
}

/// AV1 4:2:0 10-bit colour bars. In `PROFILE_PREFERENCE` it ranks below
/// the HEVC 4:4:4 entries but above HEVC Main10 and AV1 8-bit, and
/// decodes to `RenderLayout::Biplanar16` — same 10-in-16 import as HEVC
/// Main10, but from the AV1 decoder, with colour asserted. No codec-level
/// AV1 10-bit test exists, so this is the only AV1 10-bit coverage
/// anywhere.
/// Shares the P010 dma-buf submit path with HEVC Main10 (and the same
/// `GR32`→`RG32` fourcc fix that unblocked it); verified green on Lunar
/// Lake. SKIPs only where the driver lacks AV1 10-bit encode or R16/Rg16
/// storage.
#[test]
#[ignore = "requires VAAPI AV1 10-bit encode + storage R16/Rg16 + Vulkan dma-buf import"]
fn roundtrip_colorbars_av1_10bit() {
    let case = colorbars_case(
        "colorbars_av1_10bit",
        AV1_10BIT_420,
        &[
            Capability::VaapiAv1,
            Capability::VulkanDmaBufImport,
            Capability::BitDepth10RendererSupport,
        ],
    );
    assert_colorbars_outcome(&case, run_roundtrip(&case));
}

#[test]
#[ignore = "requires VAAPI HEVC Main + Vulkan dma-buf import"]
fn roundtrip_colorbars_hevc_main_8bit() {
    let case = colorbars_case(
        "colorbars_hevc_main_8bit",
        HEVC_MAIN_8BIT,
        &[Capability::VaapiHevcMain, Capability::VulkanDmaBufImport],
    );
    assert_colorbars_outcome(&case, run_roundtrip(&case));
}

#[test]
#[ignore = "requires VAAPI HEVC Main 10 (P010) + storage R16/Rg16"]
fn roundtrip_colorbars_hevc_main10() {
    let case = colorbars_case(
        "colorbars_hevc_main10",
        HEVC_MAIN10,
        &[
            Capability::VaapiHevcMain10DmaBuf,
            Capability::VulkanDmaBufImport,
            Capability::BitDepth10RendererSupport,
        ],
    );
    assert_colorbars_outcome(&case, run_roundtrip(&case));
}

#[test]
#[ignore = "requires VAAPI HEVC Main 4:4:4 8-bit (XYUV) + Vulkan dma-buf import"]
fn roundtrip_colorbars_hevc_main444_8bit() {
    let case = colorbars_case(
        "colorbars_hevc_main444_8bit",
        HEVC_MAIN444_8BIT,
        &[Capability::VaapiHevcMain444, Capability::VulkanDmaBufImport],
    );
    assert_colorbars_outcome(&case, run_roundtrip(&case));
}

/// The host path that produces the live purple cast: HEVC 4:4:4 10-bit
/// via `Bgra2Xv30DmaBuf`. If the host miscolors 4:4:4 10-bit, the white
/// bar comes back tinted here even though SSIM/geom stay green.
#[test]
#[ignore = "requires VAAPI HEVC Main 4:4:4 10-bit (XV30) + storage R16/Rg16 + Vulkan dma-buf import"]
fn roundtrip_colorbars_hevc_main444_10bit() {
    let case = colorbars_case(
        "colorbars_hevc_main444_10bit",
        HEVC_MAIN444_10BIT,
        &[
            Capability::VaapiHevcMain444_10DmaBuf,
            Capability::VulkanDmaBufImport,
            Capability::BitDepth10RendererSupport,
        ],
    );
    assert_colorbars_outcome(&case, run_roundtrip(&case));
}

/// Conformance anchor for the 4:4:4 10-bit decode path. The round-trip
/// cell above encodes with our own gpuconvert pass and decodes with our
/// own shader, so it cannot catch a wrong-but-symmetric packing (a Y↔Cb
/// lane swap round-trips at SSIM≈1.0 yet ships a non-conformant bitstream
/// that macOS VideoToolbox renders as the bright→purple / dark→olive
/// cast). This cell instead decodes the committed **reference** bitstream
/// (libx265-encoded, standards-correct — the exact fixture the macOS
/// `iosurface_test` cell decodes) through the real VAAPI Y410 → import →
/// shader path, pinning the decode side to the standard. Decode-conformant
/// here + a passing round-trip above ⟹ the encode side is conformant too.
#[test]
#[ignore = "requires VAAPI HEVC Main 4:4:4 10-bit (Y410) + storage R16/Rg16 + Vulkan dma-buf import"]
fn decode_reference_colorbars_hevc_main444_10bit() {
    const FIXTURE: &[u8] = include_bytes!("../fixtures/colorbars_hevc_yuv444_10bit_1920x1200.idr");
    let dims = (1920, 1200);
    let Some(bgra) = crate::test_harness::render_reference_bitstream(
        HEVC_MAIN444_10BIT,
        FIXTURE,
        dims,
        VideoColorSpec::sdr_desktop(),
    ) else {
        eprintln!("SKIPPED decode_reference_colorbars_hevc_main444_10bit: VAAPI/Vulkan 4:4:4 10-bit decode unavailable");
        return;
    };
    crate::color_fixture::assert_colorbars(
        "ref 4:4:4 10-bit",
        &bgra,
        dims.0,
        dims.1,
        crate::color_fixture::ChannelOrder::Bgra,
    );
}

/// Conformance anchor for the 4:4:4 **8-bit** (packed XYUV) decode path —
/// the 8-bit sibling of `decode_reference_colorbars_hevc_main444_10bit`.
/// The XYUV pack convention is hand-authored WGSL (`bgra_to_yuv444.wgsl`
/// / `shader_yuv444.wgsl`), the same risk class as the XV30/Y410 lane
/// swap fixed in 28c2d5b: a wrong-but-symmetric Y↔U swap round-trips at
/// SSIM≈1 and even passes the absolute-colour bars, because the in-loop
/// VAAPI encode/decode is itself lane-symmetric. Only a foreign decoder
/// breaks the symmetry — so decode the committed libx265 reference
/// (already the shared cross-platform fixture macOS `iosurface_test`
/// uses) through the real VAAPI XYUV → import → shader path. macOS had
/// this 8-bit anchor; Linux only had the 10-bit one. See the
/// fixtures/README "Linux/Windows render tests" note.
#[test]
#[ignore = "requires VAAPI HEVC Main 4:4:4 8-bit (XYUV) + Vulkan dma-buf import"]
fn decode_reference_colorbars_hevc_main444_8bit() {
    const FIXTURE: &[u8] = include_bytes!("../fixtures/colorbars_hevc_yuv444_8bit.idr");
    let dims = (128, 128);
    let Some(bgra) = crate::test_harness::render_reference_bitstream(
        HEVC_MAIN444_8BIT,
        FIXTURE,
        dims,
        VideoColorSpec::sdr_desktop(),
    ) else {
        eprintln!("SKIPPED decode_reference_colorbars_hevc_main444_8bit: VAAPI/Vulkan 4:4:4 8-bit decode unavailable");
        return;
    };
    crate::color_fixture::assert_colorbars(
        "ref 4:4:4 8-bit",
        &bgra,
        dims.0,
        dims.1,
        crate::color_fixture::ChannelOrder::Bgra,
    );
}

// =====================================================================
// Per-cell floors — derived from a green-main run on a real VAAPI box
// (Intel iHD, Mesa, dev workstation, 2026-05-22). Thresholds sit at
// `observed - 0.01` (SSIM) / `observed - 1.5 dB` (PSNR-Y) / `observed +
// ~25%` (geometric residual). A real regression (4-overlapping-copies
// stride bug; sRGB encoding bug; range-kind dispatch flip) overshoots
// these by an order of magnitude — these floors trip on a 1% regression
// while staying well above the encoder's natural quantisation noise.
//
// Geometric-residual floors differ sharply between scaler-active rows
// (where Mitchell pre-smooths the fixture's per-pixel gradient steps)
// and identity rows (where 8-bit-per-channel encoder quantisation on
// the raw gradient dominates). Both far below the 500+ px residuals a
// real stride bug produces.
// =====================================================================

// Identity / client-upscale / surface-below rows (no host scaler) —
// encoder quantisation noise on the raw fixture dominates. Each
// 8-bit channel level recovers to capture_width/255 ≈ 7.5 px at
// 1920 width; observed residual ≈ 85 px (≈ 11 levels of channel
// drift). The 130 px ceiling gives ~50% headroom for hardware-
// variant noise without masking a real stride bug, which moves
// geometry by at least one row-stride pitch (≥ 1200 px at our dims).
const FLOOR_IDENTITY: Floors = Floors {
    ssim: 0.95,
    psnr_y_db: 25.5,
    geom_px: 130.0,
};
const FLOOR_CLIENT_UPSCALE: Floors = Floors {
    ssim: 0.95,
    psnr_y_db: 26.0,
    geom_px: 130.0,
};
const FLOOR_SURFACE_BELOW: Floors = Floors {
    ssim: 0.95,
    psnr_y_db: 25.5,
    geom_px: 130.0,
};
// Host-scaler cell (PNG fixture, downscale, surface == encode). The
// geometric metric is skipped for PNG fixtures (assert_outcome gates
// on Fixture::CoordEncoded). NaN here documents that the value is
// unused; a future CoordEncoded cell that reuses this constant would
// fail loudly instead of inheriting an uncalibrated floor.
const FLOOR_HOST_SCALER: Floors = Floors {
    ssim: 0.985,
    psnr_y_db: 50.0,
    geom_px: f64::NAN,
};
// Full-chain cell — host Mitchell smooths the fixture before encode,
// so encoder noise drops by ~5×; client upscale adds back a little.
const FLOOR_FULL_CHAIN: Floors = Floors {
    ssim: 0.97,
    psnr_y_db: 50.0,
    geom_px: 30.0,
};

const H264_8BIT_420: VideoProfile = VideoProfile {
    codec: CodecKind::H264,
    chroma: ChromaSubsampling::Yuv420,
    bit_depth: 8,
};
const HEVC_MAIN_8BIT: VideoProfile = VideoProfile {
    codec: CodecKind::Hevc,
    chroma: ChromaSubsampling::Yuv420,
    bit_depth: 8,
};
const HEVC_MAIN444_8BIT: VideoProfile = VideoProfile {
    codec: CodecKind::Hevc,
    chroma: ChromaSubsampling::Yuv444,
    bit_depth: 8,
};
const HEVC_MAIN10: VideoProfile = VideoProfile {
    codec: CodecKind::Hevc,
    chroma: ChromaSubsampling::Yuv420,
    bit_depth: 10,
};
const HEVC_MAIN444_10BIT: VideoProfile = VideoProfile {
    codec: CodecKind::Hevc,
    chroma: ChromaSubsampling::Yuv444,
    bit_depth: 10,
};
const AV1_8BIT_420: VideoProfile = VideoProfile {
    codec: CodecKind::Av1,
    chroma: ChromaSubsampling::Yuv420,
    bit_depth: 8,
};
const AV1_10BIT_420: VideoProfile = VideoProfile {
    codec: CodecKind::Av1,
    chroma: ChromaSubsampling::Yuv420,
    bit_depth: 10,
};

const FIXTURE_PNG: &str = "test-pattern-3360x2100.png";

// =====================================================================
// Matrix
// =====================================================================

/// H.264 4:2:0 8-bit identity. The universal floor — every VAAPI box
/// supports it. Catches encoder/decoder/Biplanar8-import regressions
/// without engaging either scaler stage.
#[test]
#[ignore = "requires VAAPI HW + Vulkan dma-buf import"]
fn roundtrip_h264_8bit_identity() {
    let case = RoundtripCase {
        name: "h264_8bit_identity",
        profile: H264_8BIT_420,
        fixture: Fixture::CoordEncoded,
        capture_dims: (1920, 1200),
        encode_dims: (1920, 1200),
        surface_dims: (1920, 1200),
        frames_encoded: 6,
        assert_steady_state_eps: None,
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[Capability::VaapiH264, Capability::VulkanDmaBufImport],
        floors: FLOOR_IDENTITY,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// AV1 4:2:0 8-bit identity. AV1 encoder/decoder pair → Biplanar8
/// import — the geometric companion to `roundtrip_colorbars_av1_8bit`.
/// Catches AV1-specific stride / decode-output regressions the
/// uniform-grey codec-level `av1_main_dmabuf_roundtrip` can't see.
/// SKIPs on drivers without AV1 encode.
#[test]
#[ignore = "requires VAAPI AV1 encode (Intel Arc / AMD RDNA3+) + Vulkan dma-buf import"]
fn roundtrip_av1_8bit_identity() {
    let case = RoundtripCase {
        name: "av1_8bit_identity",
        profile: AV1_8BIT_420,
        fixture: Fixture::CoordEncoded,
        capture_dims: (1920, 1200),
        encode_dims: (1920, 1200),
        surface_dims: (1920, 1200),
        frames_encoded: 6,
        assert_steady_state_eps: None,
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[Capability::VaapiAv1, Capability::VulkanDmaBufImport],
        floors: FLOOR_IDENTITY,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// AV1 4:2:0 10-bit identity. AV1 encoder/decoder pair → Biplanar16
/// import — the geometric companion to `roundtrip_colorbars_av1_10bit`,
/// and the only AV1 10-bit geometry coverage anywhere. Verified green on
/// Lunar Lake after the P010 UV-plane fourcc fix.
#[test]
#[ignore = "requires VAAPI AV1 10-bit encode + storage R16/Rg16 + Vulkan dma-buf import"]
fn roundtrip_av1_10bit_identity() {
    let case = RoundtripCase {
        name: "av1_10bit_identity",
        profile: AV1_10BIT_420,
        fixture: Fixture::CoordEncoded,
        capture_dims: (1920, 1200),
        encode_dims: (1920, 1200),
        surface_dims: (1920, 1200),
        frames_encoded: 6,
        assert_steady_state_eps: None,
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[
            Capability::VaapiAv1,
            Capability::VulkanDmaBufImport,
            Capability::BitDepth10RendererSupport,
        ],
        floors: FLOOR_IDENTITY,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// HEVC Main 4:2:0 8-bit identity. Same shape as H.264 above but
/// through the HEVC encoder/decoder pair.
#[test]
#[ignore = "requires VAAPI HW + Vulkan dma-buf import"]
fn roundtrip_hevc_main_8bit_identity() {
    let case = RoundtripCase {
        name: "hevc_main_8bit_identity",
        profile: HEVC_MAIN_8BIT,
        fixture: Fixture::CoordEncoded,
        capture_dims: (1920, 1200),
        encode_dims: (1920, 1200),
        surface_dims: (1920, 1200),
        frames_encoded: 6,
        assert_steady_state_eps: None,
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[Capability::VaapiHevcMain, Capability::VulkanDmaBufImport],
        floors: FLOOR_IDENTITY,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// HEVC Main 4:4:4 8-bit identity. Exercises the PackedXYUV import +
/// `shader_yuv444.wgsl`. SKIPs cleanly on drivers without HEVC 4:4:4
/// encode support (pre-Tiger Lake Intel / pre-VCN3 AMD).
#[test]
#[ignore = "requires VAAPI HEVC Main 4:4:4 (Intel Tiger Lake+ / AMD VCN3+)"]
fn roundtrip_hevc_main444_8bit_identity() {
    let case = RoundtripCase {
        name: "hevc_main444_8bit_identity",
        profile: HEVC_MAIN444_8BIT,
        fixture: Fixture::CoordEncoded,
        capture_dims: (1920, 1200),
        encode_dims: (1920, 1200),
        surface_dims: (1920, 1200),
        frames_encoded: 6,
        assert_steady_state_eps: None,
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[Capability::VaapiHevcMain444, Capability::VulkanDmaBufImport],
        floors: FLOOR_IDENTITY,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// HEVC Main 10 identity. P010 dma-buf bridge → encoder.submit_dmabuf
/// → decoder → Biplanar16 import → `range_kind = LIMITED_10` shader
/// dispatch. Previously SKIP'd across Intel (iHD/Meteor Lake/Lunar
/// Lake) — misdiagnosed as a driver P010 gap; the real cause was the
/// UV-plane fourcc (`GR32` vs FFmpeg's `RG1616`), fixed in
/// `build_p010_dmabuf_frame`. Now runs; verified green on Lunar Lake.
#[test]
#[ignore = "requires VAAPI HEVC Main 10 + storage R16/Rg16"]
fn roundtrip_hevc_main10_identity() {
    let case = RoundtripCase {
        name: "hevc_main10_identity",
        profile: HEVC_MAIN10,
        fixture: Fixture::CoordEncoded,
        capture_dims: (1920, 1200),
        encode_dims: (1920, 1200),
        surface_dims: (1920, 1200),
        frames_encoded: 6,
        assert_steady_state_eps: None,
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[
            Capability::VaapiHevcMain10DmaBuf,
            Capability::VulkanDmaBufImport,
            Capability::BitDepth10RendererSupport,
        ],
        floors: FLOOR_IDENTITY,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// HEVC Main 4:4:4 10-bit identity. XV30 dma-buf bridge
/// (Bgra2Xv30DmaBuf, packed 10:10:10:2) → encoder.submit_dmabuf →
/// decoder → render-side import. The encoder *consumes* XV30, but Intel's
/// VAAPI decoder *outputs* packed Y410 (single 10:10:10:2 layer — same
/// size, different component order), which `render_layout_for` maps to
/// `RenderLayout::PackedY410` and `import_yuv444_packed_y410` imports as
/// one `Rgb10a2Unorm` texture (`shader_yuv444_10bit.wgsl`). This cell is
/// the gate that exercises that packed path end-to-end.
///
/// Unlike the 4:2:0 Main10 cell, this never hit the P010 fourcc gap —
/// XV30 is a single packed layer, not the R16+UV biplanar P010 layout —
/// and runs green on Lunar Lake. SKIPs only where the driver lacks HEVC
/// 4:4:4 10-bit encode.
#[test]
#[ignore = "requires VAAPI HEVC Main 4:4:4 10-bit + Rgb10a2Unorm storage"]
fn roundtrip_hevc_main444_10bit_identity() {
    let case = RoundtripCase {
        name: "hevc_main444_10bit_identity",
        profile: HEVC_MAIN444_10BIT,
        fixture: Fixture::CoordEncoded,
        capture_dims: (1920, 1200),
        encode_dims: (1920, 1200),
        surface_dims: (1920, 1200),
        frames_encoded: 6,
        assert_steady_state_eps: None,
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[
            Capability::VaapiHevcMain444_10DmaBuf,
            Capability::VulkanDmaBufImport,
            Capability::BitDepth10RendererSupport,
        ],
        floors: FLOOR_IDENTITY,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// H.264 host-scaler-only. Capture > encode == surface. The Mitchell
/// downscale stage is exercised in isolation, photometric fidelity is
/// the only thing under test (no per-pixel geometric encoding fits a
/// post-downscale check). Uses the PNG fixture so the photometric
/// metrics have meaningful content to measure.
#[test]
#[ignore = "requires VAAPI HW + Vulkan dma-buf import"]
fn roundtrip_h264_8bit_host_scaler() {
    let case = RoundtripCase {
        name: "h264_8bit_host_scaler",
        profile: H264_8BIT_420,
        fixture: Fixture::Png(FIXTURE_PNG),
        capture_dims: (1920, 1200),
        encode_dims: (1280, 800),
        surface_dims: (1280, 800),
        frames_encoded: 6,
        assert_steady_state_eps: None,
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[Capability::VaapiH264, Capability::VulkanDmaBufImport],
        floors: FLOOR_HOST_SCALER,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// H.264 surface < video. Engages the blit pass's non-identity
/// `letterbox_scale` branch (gpu/mod.rs:1141) without engaging
/// `need_upscale`. Separate arithmetic from the upscale branch.
#[test]
#[ignore = "requires VAAPI HW + Vulkan dma-buf import"]
fn roundtrip_h264_8bit_surface_below_video() {
    let case = RoundtripCase {
        name: "h264_8bit_surface_below_video",
        profile: H264_8BIT_420,
        // CoordEncoded — capture_dims == encode_dims (no scaler) so
        // the recovered coordinates survive the encode step; the
        // residual then catches bugs in the blit-pass letterbox-scale
        // arithmetic that the photometric metrics would only see as
        // mild blur. This is the metric this cell's reason-for-being
        // is most sensitive to.
        fixture: Fixture::CoordEncoded,
        capture_dims: (1920, 1200),
        encode_dims: (1920, 1200),
        surface_dims: (1280, 800),
        frames_encoded: 6,
        assert_steady_state_eps: None,
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[Capability::VaapiH264, Capability::VulkanDmaBufImport],
        floors: FLOOR_SURFACE_BELOW,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// **H.264 client-upscale**. Capture == encode < surface. The
/// renderer's Mitchell upscale stage is the only scaling stage. This
/// is the cell whose existence would have deterministically caught
/// the 4-overlapping-copies / vertical-scan-lines bug observed in
/// manual sessions — geometric residual on the broken code goes to
/// hundreds of pixels.
#[test]
#[ignore = "requires VAAPI HW + Vulkan dma-buf import (Intel Mesa is a target — load-bearing queue.submit fix lives in render_to_view)"]
fn roundtrip_h264_8bit_client_upscale() {
    let case = RoundtripCase {
        name: "h264_8bit_client_upscale",
        profile: H264_8BIT_420,
        fixture: Fixture::CoordEncoded,
        capture_dims: (1920, 1200),
        encode_dims: (1920, 1200),
        surface_dims: (2560, 1440),
        frames_encoded: 6,
        assert_steady_state_eps: Some(2.0),
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[Capability::VaapiH264, Capability::VulkanDmaBufImport],
        floors: FLOOR_CLIENT_UPSCALE,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// **H.264 client-upscale matching the manual-session repro shape.**
/// surface (2560×1440) > video (2160×1440), but `letterbox_fit_dims`
/// of the video into the surface returns exactly video_dims back —
/// because the surface is taller-aspect than the video, the fit is
/// width-unlimited and height-limited at 1.0× scale. That makes
/// `Scaler::new(2160×1440 → 2160×1440)` return `NoScaleNeeded`, so
/// `self.upscale.scaler` stays None even though `need_upscale=true`.
///
/// The blit pass then samples the intermediate at video_dims and
/// writes to the swapchain with a non-identity `letterbox_scale =
/// (0.84375, 1.0)`. None of the other cells exercise this exact
/// branch — `h264_8bit_client_upscale` above has a video aspect that
/// *does* differ from surface so Mitchell runs; `h264_8bit_full_chain`
/// has surface > video > letterbox_fit so Mitchell also runs.
///
/// If the steady-state 4-overlapping-copies bug observed in manual
/// sessions lives in this branch (need_upscale=true + scaler=None),
/// this cell catches it; if the bug is swapchain-specific
/// (back-buffer rotation, present-time sync), this cell still passes
/// and that tells us where to look next.
#[test]
#[ignore = "requires VAAPI HW + Vulkan dma-buf import"]
fn roundtrip_h264_8bit_upscale_no_scaler_branch() {
    let case = RoundtripCase {
        name: "h264_8bit_upscale_no_scaler_branch",
        profile: H264_8BIT_420,
        fixture: Fixture::CoordEncoded,
        capture_dims: (2160, 1440),
        encode_dims: (2160, 1440),
        surface_dims: (2560, 1440),
        frames_encoded: 6,
        assert_steady_state_eps: Some(2.0),
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[Capability::VaapiH264, Capability::VulkanDmaBufImport],
        floors: FLOOR_CLIENT_UPSCALE,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// **H.264 host-scaler + client-no-scaler + letterbox blit**. This is
/// the exact shape the manual-session 4-overlapping-copies repro hit
/// (capture 2880×1920 → encode 2160×1440 → surface 2560×1440), with
/// all three of:
///   1. Host Mitchell downscale engaged (capture > encode)
///   2. Client `need_upscale = true` (surface > encode)
///   3. Client `scaler = None` because letterbox_fit_dims(encode,
///      surface) returns encode_dims when surface aspect > encode
///      aspect (here 1.778 > 1.5, fit is height-limited at 1.0× scale)
///   4. Blit pass with non-identity letterbox_scale = (0.84375, 1.0)
///
/// None of the other cells combine these — full_chain has Mitchell
/// upscale running because its encode aspect differs from surface
/// just enough; upscale_no_scaler_branch above has no host scaler.
/// This cell is the intersection the bug needs.
///
/// The pre-scaler-integration session was clean (host encoded at
/// native capture dims, client surface < video, no upscale branch).
/// Post-integration sessions hit this exact path on the user's 2×
/// HiDPI display.
#[test]
#[ignore = "requires VAAPI HW + Vulkan dma-buf import"]
fn roundtrip_h264_8bit_repro_shape() {
    let case = RoundtripCase {
        name: "h264_8bit_repro_shape",
        profile: H264_8BIT_420,
        fixture: Fixture::CoordEncoded,
        capture_dims: (2880, 1920),
        encode_dims: (2160, 1440),
        surface_dims: (2560, 1440),
        frames_encoded: 6,
        assert_steady_state_eps: Some(2.0),
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[Capability::VaapiH264, Capability::VulkanDmaBufImport],
        // Use the full_chain floor: host scaler smooths the gradient
        // fixture before encode, so encoder quantisation drops by 5×
        // — same envelope as the full_chain cell.
        floors: FLOOR_FULL_CHAIN,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// **H.264 full chain**: capture > encode < surface, surface aspect ≠
/// encode aspect so letterbox bars appear in the surface. Exercises
/// both Mitchell stages plus letterbox padding. 3360×2100 (aspect
/// 1.6) → 2240×1400 (aspect 1.6, host scaler preserves aspect) →
/// 2560×1440 (aspect 16:9, letterbox bars top+bottom).
#[test]
#[ignore = "requires VAAPI HW + Vulkan dma-buf import (Intel Mesa is a target)"]
fn roundtrip_h264_8bit_full_chain() {
    let case = RoundtripCase {
        name: "h264_8bit_full_chain",
        profile: H264_8BIT_420,
        fixture: Fixture::CoordEncoded,
        capture_dims: (3360, 2100),
        encode_dims: (2240, 1400),
        surface_dims: (2560, 1440),
        frames_encoded: 6,
        assert_steady_state_eps: Some(2.0),
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[Capability::VaapiH264, Capability::VulkanDmaBufImport],
        floors: FLOOR_FULL_CHAIN,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// HEVC Main 4:2:0 8-bit full chain — same dims as H.264 full chain,
/// HEVC encoder/decoder pair. Catches HEVC-specific regressions in the
/// full-chain path that the identity row wouldn't surface.
#[test]
#[ignore = "requires VAAPI HW + Vulkan dma-buf import"]
fn roundtrip_hevc_main_8bit_full_chain() {
    let case = RoundtripCase {
        name: "hevc_main_8bit_full_chain",
        profile: HEVC_MAIN_8BIT,
        fixture: Fixture::CoordEncoded,
        capture_dims: (3360, 2100),
        encode_dims: (2240, 1400),
        surface_dims: (2560, 1440),
        frames_encoded: 6,
        assert_steady_state_eps: Some(2.0),
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[Capability::VaapiHevcMain, Capability::VulkanDmaBufImport],
        floors: FLOOR_FULL_CHAIN,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// HEVC Main 4:4:4 full chain. Exercises PackedXYUV import +
/// `shader_yuv444.wgsl` through the full scaler-and-upscaler chain.
#[test]
#[ignore = "requires VAAPI HEVC Main 4:4:4"]
fn roundtrip_hevc_main444_8bit_full_chain() {
    let case = RoundtripCase {
        name: "hevc_main444_8bit_full_chain",
        profile: HEVC_MAIN444_8BIT,
        fixture: Fixture::CoordEncoded,
        capture_dims: (3360, 2100),
        encode_dims: (2240, 1400),
        surface_dims: (2560, 1440),
        frames_encoded: 6,
        assert_steady_state_eps: Some(2.0),
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[Capability::VaapiHevcMain444, Capability::VulkanDmaBufImport],
        floors: FLOOR_FULL_CHAIN,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// HEVC Main 10 client upscale. capture == encode (no scaler — the
/// 10-bit scaler input path doesn't exist in production), surface
/// > encode so the renderer's Mitchell upscale of the decoded 10-bit
/// > content gets exercised. Targets the Biplanar16 import + 10-bit
/// > shader dispatch + Mitchell upscale all at once. Runs on Lunar Lake
/// > since the P010 UV-plane fourcc fix; SKIPs only where the driver
/// > lacks HEVC Main 10 encode or R16/Rg16 storage.
#[test]
#[ignore = "requires VAAPI HEVC Main 10 + storage R16/Rg16"]
fn roundtrip_hevc_main10_client_upscale() {
    let case = RoundtripCase {
        name: "hevc_main10_client_upscale",
        profile: HEVC_MAIN10,
        fixture: Fixture::CoordEncoded,
        capture_dims: (1920, 1200),
        encode_dims: (1920, 1200),
        surface_dims: (2560, 1440),
        frames_encoded: 6,
        assert_steady_state_eps: Some(2.0),
        color_space: VideoColorSpec::sdr_desktop(),
        requires: &[
            Capability::VaapiHevcMain10DmaBuf,
            Capability::VulkanDmaBufImport,
            Capability::BitDepth10RendererSupport,
        ],
        floors: FLOOR_CLIENT_UPSCALE,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

// =====================================================================
// Perf microbench (kept from the pre-rewrite file)
// =====================================================================

/// Two solid color regions — left half red, right half blue. Used by
/// the encode microbench below.
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

/// Median µs per encode_bgra call after a 6-frame warmup.
fn bench_encode_one_resolution(profile: VideoProfile, w: u32, h: u32, iters: usize) -> f64 {
    let bgra = make_test_bgra(w, h);
    let mut enc = VaapiEncoder::new(profile, w, h, 30, 4_000).expect("encoder");
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

#[test]
#[ignore = "perf microbenchmark; prints encode timings, no assertions"]
fn bench_encode_by_resolution_h264() {
    let _ = tracing_subscriber::fmt::try_init();
    let profile = H264_8BIT_420;
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

// =====================================================================
// Cross-table consistency (no hardware required) — kept verbatim
// =====================================================================

#[cfg(test)]
mod cross_table_consistency {
    use tether_codec::vaapi::expected_dmabuf_fourcc;
    use tether_protocol::control::ChromaSubsampling;

    use crate::gpu::{render_layout_for, RenderLayout};

    const MODELED: &[(ChromaSubsampling, u8)] = &[
        (ChromaSubsampling::Yuv420, 8),
        (ChromaSubsampling::Yuv420, 10),
        (ChromaSubsampling::Yuv444, 8),
        (ChromaSubsampling::Yuv444, 10),
    ];

    /// Maps each encoder-input fourcc to the renderer layout we expect
    /// to use for the matching `(chroma, bit_depth)`. The keys are the
    /// *encoder-input* fourccs (`expected_dmabuf_fourcc`); the values are
    /// the renderer's *decode-output* layout. For 4:2:0 and 8-bit 4:4:4
    /// the decoder emits the same fourcc family it consumes. For 4:4:4
    /// 10-bit they differ: the encoder consumes XV30 but Intel's decoder
    /// outputs Y410 (→ `PackedY410`) — same 10:10:10:2 packing, different
    /// component order. Verified end-to-end by the
    /// `roundtrip_hevc_main444_10bit_identity` cell on Intel media-driver.
    fn expected_layout_for_fourcc(fourcc: u32) -> RenderLayout {
        match &fourcc.to_le_bytes() {
            b"NV12" => RenderLayout::Biplanar8,
            b"P010" => RenderLayout::Biplanar16,
            b"XYUV" => RenderLayout::PackedXYUV,
            b"XV30" => RenderLayout::PackedY410,
            other => panic!("expected_layout_for_fourcc: unknown fourcc {other:?}"),
        }
    }

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

    #[test]
    fn preference_list_profiles_have_fourcc() {
        // Iterate `PROFILE_PREFERENCE` directly so any future profile
        // addition (AV1, future chroma) is automatically covered. A
        // hand-rolled list silently misses new entries.
        for p in tether_probe::PROFILE_PREFERENCE.iter().copied() {
            let f = expected_dmabuf_fourcc(p.chroma, p.bit_depth);
            assert!(f.is_some(), "no encoder fourcc for profile {p:?}");
        }
    }
}
