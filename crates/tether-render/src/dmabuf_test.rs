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
//! Floors below are **placeholders**, marked as such with a comment.
//! Step #26 of the plan derives the real per-cell floors from a green-
//! main run (`measured − 0.02` for SSIM, `measured − 1.5 dB` for
//! PSNR-Y, structural 1.0 px for geometric residual).
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
    Capability, Fixture, RoundtripCase, RoundtripOutcome, RoundtripResult, run_roundtrip,
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
    if matches!(case.fixture, Fixture::CoordEncoded) {
        assert!(
            outcome.geometric_residual_px_rms <= case.geometric_residual_px_max,
            "[{}] geometric residual {} px > floor {} px — stride or UV-addressing bug?",
            case.name,
            outcome.geometric_residual_px_rms,
            case.geometric_residual_px_max,
        );
    }
    assert!(
        outcome.ssim >= case.ssim_floor,
        "[{}] ssim {} < floor {}",
        case.name,
        outcome.ssim,
        case.ssim_floor,
    );
    assert!(
        outcome.psnr_y_db >= case.psnr_y_floor_db,
        "[{}] psnr_y {} dB < floor {} dB",
        case.name,
        outcome.psnr_y_db,
        case.psnr_y_floor_db,
    );
    if let (Some(eps), Some(delta)) = (case.assert_steady_state_eps, outcome.steady_state_delta) {
        assert!(
            delta <= eps,
            "[{}] steady-state MAE {} > {} between last two rendered frames",
            case.name,
            delta,
            eps,
        );
    }
}

// =====================================================================
// Placeholder floors — derived in step #26 from a green-main run.
// SSIM floor here is conservative; PSNR-Y is conservative too.
// =====================================================================

const PLACEHOLDER_SSIM: f64 = 0.85;
const PLACEHOLDER_PSNR_Y_DB: f64 = 28.0;
const STRUCTURAL_GEOMETRIC_PX: f64 = 1.0;

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
        geometric_residual_px_max: STRUCTURAL_GEOMETRIC_PX,
        ssim_floor: PLACEHOLDER_SSIM,
        psnr_y_floor_db: PLACEHOLDER_PSNR_Y_DB,
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
        geometric_residual_px_max: STRUCTURAL_GEOMETRIC_PX,
        ssim_floor: PLACEHOLDER_SSIM,
        psnr_y_floor_db: PLACEHOLDER_PSNR_Y_DB,
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
        requires: &[
            Capability::VaapiHevcMain444,
            Capability::VulkanDmaBufImport,
        ],
        geometric_residual_px_max: STRUCTURAL_GEOMETRIC_PX,
        ssim_floor: PLACEHOLDER_SSIM,
        psnr_y_floor_db: PLACEHOLDER_PSNR_Y_DB,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// HEVC Main 10 identity. P010 dma-buf bridge → encoder.submit_dmabuf
/// → decoder → Biplanar16 import → `range_kind = LIMITED_10` shader
/// dispatch. Empirically SKIPs on Intel iHD + Meteor Lake
/// (FFmpeg vaapi_drm_format_map lacks P010 entries on that combo).
#[test]
#[ignore = "requires VAAPI HEVC Main 10 + storage R16/Rg16; may SKIP on Intel iHD/Meteor Lake"]
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
        geometric_residual_px_max: STRUCTURAL_GEOMETRIC_PX,
        ssim_floor: PLACEHOLDER_SSIM,
        psnr_y_floor_db: PLACEHOLDER_PSNR_Y_DB,
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
        geometric_residual_px_max: STRUCTURAL_GEOMETRIC_PX,
        ssim_floor: PLACEHOLDER_SSIM,
        psnr_y_floor_db: PLACEHOLDER_PSNR_Y_DB,
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
        geometric_residual_px_max: STRUCTURAL_GEOMETRIC_PX,
        ssim_floor: PLACEHOLDER_SSIM,
        psnr_y_floor_db: PLACEHOLDER_PSNR_Y_DB,
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
        geometric_residual_px_max: STRUCTURAL_GEOMETRIC_PX,
        ssim_floor: PLACEHOLDER_SSIM,
        psnr_y_floor_db: PLACEHOLDER_PSNR_Y_DB,
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
        geometric_residual_px_max: STRUCTURAL_GEOMETRIC_PX,
        ssim_floor: PLACEHOLDER_SSIM,
        psnr_y_floor_db: PLACEHOLDER_PSNR_Y_DB,
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
        geometric_residual_px_max: STRUCTURAL_GEOMETRIC_PX,
        ssim_floor: PLACEHOLDER_SSIM,
        psnr_y_floor_db: PLACEHOLDER_PSNR_Y_DB,
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
        requires: &[
            Capability::VaapiHevcMain444,
            Capability::VulkanDmaBufImport,
        ],
        geometric_residual_px_max: STRUCTURAL_GEOMETRIC_PX,
        ssim_floor: PLACEHOLDER_SSIM,
        psnr_y_floor_db: PLACEHOLDER_PSNR_Y_DB,
    };
    assert_outcome(&case, run_roundtrip(&case));
}

/// HEVC Main 10 client upscale. capture == encode (no scaler — the
/// 10-bit scaler input path doesn't exist in production), surface
/// > encode so the renderer's Mitchell upscale of the decoded 10-bit
/// content gets exercised. Targets the Biplanar16 import + 10-bit
/// shader dispatch + Mitchell upscale all at once. Will SKIP on
/// Intel iHD + Meteor Lake (P010 dma-buf driver gap).
#[test]
#[ignore = "requires VAAPI HEVC Main 10 + storage R16/Rg16; may SKIP on Intel iHD/Meteor Lake"]
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
        geometric_residual_px_max: STRUCTURAL_GEOMETRIC_PX,
        ssim_floor: PLACEHOLDER_SSIM,
        psnr_y_floor_db: PLACEHOLDER_PSNR_Y_DB,
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
    use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

    use crate::gpu::{RenderLayout, render_layout_for};

    const MODELED: &[(ChromaSubsampling, u8)] = &[
        (ChromaSubsampling::Yuv420, 8),
        (ChromaSubsampling::Yuv420, 10),
        (ChromaSubsampling::Yuv444, 8),
        (ChromaSubsampling::Yuv444, 10),
    ];

    fn expected_layout_for_fourcc(fourcc: u32) -> RenderLayout {
        match &fourcc.to_le_bytes() {
            b"NV12" => RenderLayout::Biplanar8,
            b"P010" => RenderLayout::Biplanar16,
            b"XYUV" => RenderLayout::PackedXYUV,
            b"XV30" => RenderLayout::Biplanar16,
            other => panic!("expected_layout_for_fourcc: unknown fourcc {other:?}"),
        }
    }

    #[test]
    fn encoder_fourccs_are_well_formed() {
        for &(chroma, bit_depth) in MODELED {
            let f = expected_dmabuf_fourcc(chroma, bit_depth).unwrap_or_else(|| {
                panic!("encoder has no fourcc for ({chroma:?}, {bit_depth})")
            });
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
            let fourcc = expected_dmabuf_fourcc(chroma, bit_depth).unwrap_or_else(|| {
                panic!("encoder has no fourcc for ({chroma:?}, {bit_depth})")
            });
            let renderer = render_layout_for(chroma, bit_depth);
            let expected = expected_layout_for_fourcc(fourcc);
            assert_eq!(
                renderer, expected,
                "render_layout_for({chroma:?}, {bit_depth}) = {renderer:?} but \
                 fourcc {:?} expects {expected:?}",
                std::str::from_utf8(&fourcc.to_le_bytes()).unwrap_or("?"),
            );
        }
    }

    #[test]
    fn preference_list_profiles_have_fourcc() {
        let profile_list = [
            VideoProfile { codec: CodecKind::Hevc, chroma: ChromaSubsampling::Yuv420, bit_depth: 8 },
            VideoProfile { codec: CodecKind::Hevc, chroma: ChromaSubsampling::Yuv420, bit_depth: 10 },
            VideoProfile { codec: CodecKind::Hevc, chroma: ChromaSubsampling::Yuv444, bit_depth: 8 },
            VideoProfile { codec: CodecKind::Hevc, chroma: ChromaSubsampling::Yuv444, bit_depth: 10 },
            VideoProfile { codec: CodecKind::H264, chroma: ChromaSubsampling::Yuv420, bit_depth: 8 },
        ];
        for p in profile_list {
            let f = expected_dmabuf_fourcc(p.chroma, p.bit_depth);
            assert!(f.is_some(), "no encoder fourcc for profile {p:?}");
        }
    }
}
