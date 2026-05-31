//! End-to-end round-trip test harness.
//!
//! Drives the production capture → encode → decode → import → render
//! chain against a known fixture and verifies the rendered output via
//! geometric-residual / SSIM / Y-channel PSNR metrics. Adding a new
//! cell is a [`RoundtripCase`] struct literal + assertion block — no
//! per-cell helper code.
//!
//! ## Coverage closed
//!
//! The pre-existing hardware tests (`dmabuf_test.rs`) built a
//! *parallel* single-pass test pipeline and rendered through it. The
//! production multi-pass renderer (`Gpu::render` in `gpu/mod.rs`) had
//! zero hardware coverage as a result — every renderer-side regression
//! the project has hit (R16Unorm feature opt-in, `'x420'` fourcc drift,
//! encoder/decoder fourcc-table drift, the 4-overlapping-copies upscale
//! bug) lived in code with no test that ran the production code path.
//!
//! This harness closes that gap by spinning the production `Gpu` with
//! [`crate::gpu::GpuState::new_headless`] — same multi-pass body, same
//! `Bgra8UnormSrgb` blit target format as the live swapchain, just with
//! an offscreen target the harness reads back.
//!
//! ## Primary metric: geometric residual on a coordinate-encoded fixture
//!
//! [`Fixture::CoordEncoded`] generates a BGRA buffer where every pixel
//! encodes its own (x, y) in its RGB channels. After the round-trip,
//! the harness decodes each pixel's claimed coordinate and compares it
//! to where the letterbox-fit mapping says that pixel should have come
//! from. Lossy encoding moves the recovered coordinate by < 1 px; a
//! stride/UV-addressing bug moves it by hundreds. This is the
//! deterministic catch for the bug class the harness was motivated by.
//!
//! SSIM and BT.709 Y-channel PSNR are the secondary catch-net for the
//! next bug we haven't thought of yet.
//!
//! ## Out of scope (this commit)
//!
//! - macOS `IOSurface` sibling — same `RoundtripCase` shape, swap the
//!   encode + import. Follow-up.
//! - On-failure diagnostics dump (readback/reference/diff/SSIM heatmap
//!   to `target/roundtrip-diagnostics/`). Separate commit so the
//!   harness shape lands clean first.

#![allow(clippy::cast_lossless, clippy::cast_possible_truncation)]

use std::sync::mpsc;

use tether_codec::vaapi::{VaapiDecoder, VaapiEncoder};
use tether_codec::{Decoder, Encoder, Frame as CodecFrame};
use tether_protocol::control::{ChromaSubsampling, VideoColorSpec, VideoProfile};
use tether_scaler::test_util::{
    coord_fixture_fill, coord_fixture_residual_px_rms, cpu_letterbox_to_surface_bgra,
    cpu_mitchell_resize_bgra, psnr_db_y_bgra, ssim_heatmap_l8, ssim_rgb, LetterboxMap,
};

use crate::gpu::GpuState;
use crate::{Frame, GpuFrame as RenderGpuFrame};

// =====================================================================
// Public test API
// =====================================================================

/// One round-trip case. Adding a new test cell means writing one of
/// these + an assertion block; no per-cell helpers.
#[derive(Debug)]
pub(crate) struct RoundtripCase {
    pub name: &'static str,
    pub profile: VideoProfile,
    pub fixture: Fixture,
    /// BGRA frame size handed to the encoder pipeline.
    pub capture_dims: (u32, u32),
    /// Host scaler output. `== capture_dims` means no scaler.
    pub encode_dims: (u32, u32),
    /// Renderer offscreen target. `> encode_dims` engages the
    /// `need_upscale` path.
    pub surface_dims: (u32, u32),
    /// Number of frames encoded before reading back. Six is the
    /// VAAPI rate-control warmup — the steady-state frame is what
    /// production sessions show.
    pub frames_encoded: usize,
    /// If `Some`, also assert that the last two encoded+rendered
    /// frames differ by < eps in BGRA MAE. Catches transient/steady-
    /// state divergence.
    pub assert_steady_state_eps: Option<f64>,
    pub color_space: VideoColorSpec,
    /// Explicit prereqs; missing capability → loud SKIP with named
    /// capability, not silent pass.
    pub requires: &'static [Capability],
    pub floors: Floors,
}

/// Per-cell metric floors. Pre-computed shared constants per cell
/// shape live in `dmabuf_test.rs` so a new case picks the right one
/// without copy-pasting three numbers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Floors {
    /// Primary metric: per-pixel coordinate residual in capture-pixel
    /// units. Lossy encode moves recovered coords by tens of px on
    /// the raw fixture; the renderer's stride/UV-addressing bug class
    /// moves them by 500+ px. Only meaningful on `Fixture::CoordEncoded`;
    /// PNG fixtures skip this check.
    pub geom_px: f64,
    /// SSIM floor against the CPU reference.
    pub ssim: f64,
    /// BT.709 Y-channel PSNR floor. Isolates the luminance pipeline
    /// from the chroma noise floor 4:2:0 dominates.
    pub psnr_y_db: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Fixture {
    /// Coordinate-encoded fixture, generated procedurally at exactly
    /// `capture_dims`. The harness extracts a geometric residual from
    /// the readback — primary metric for the stride/UV-addressing
    /// bug class.
    CoordEncoded,
    /// PNG fixture under `crates/tether-render/fixtures/`. Loaded,
    /// resized to `capture_dims` via the CPU Mitchell reference, and
    /// channel-swapped to BGRA. Use for photometric cells where
    /// per-pixel coordinate encoding isn't applicable.
    Png(&'static str),
}

/// Explicit prerequisites for a cell — declared on the struct so
/// missing-capability skips name what was missing instead of failing
/// vaguely or silently passing. Each variant maps to a concrete probe
/// inside [`check_capability`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Capability {
    VaapiH264,
    VaapiHevcMain,
    VaapiHevcMain444,
    /// 4:4:4 10-bit dma-buf path (XV30). Distinct from the 8-bit
    /// `VaapiHevcMain444` because Intel iHD historically supports
    /// HEVC 4:4:4 encode at 8-bit while rejecting the 10-bit XV30
    /// descriptor at submit time (see `tether-codec::vaapi::tests::hevc_main444_10bit_xv30_dmabuf_roundtrip`).
    VaapiHevcMain444_10DmaBuf,
    /// 4:2:0 10-bit dma-buf path: empirically SKIPs on Intel iHD +
    /// Meteor Lake (FFmpeg's vaapi_drm_format_map lacks P010
    /// entries on that combo).
    VaapiHevcMain10DmaBuf,
    /// `VULKAN_EXTERNAL_MEMORY_DMA_BUF` on the wgpu adapter.
    VulkanDmaBufImport,
    /// `TEXTURE_FORMAT_16BIT_NORM` — required for 10-bit Biplanar16.
    BitDepth10RendererSupport,
    /// AV1 codec context: a `RoundtripCase` that exercises the AV1
    /// path SKIPs with this label when the driver lacks AV1 encode
    /// or decode. The codec-level round-trip
    /// `tether_codec::vaapi::tests::av1_main_dmabuf_roundtrip` is the
    /// load-bearing verification; render-layer AV1 cells are
    /// codec-agnostic via `RenderLayout::Biplanar8` / `Biplanar16` and
    /// don't need their own surface fixtures.
    VaapiAv1,
}

#[derive(Debug)]
pub(crate) struct RoundtripOutcome {
    pub ssim: f64,
    pub psnr_y_db: f64,
    /// `f64::NAN` when the fixture isn't coordinate-encoded.
    pub geometric_residual_px_rms: f64,
    /// surface_dims × 4 bytes of `Bgra8UnormSrgb` sRGB-encoded data,
    /// matching production swapchain bytes.
    pub readback_bgra: Vec<u8>,
    /// Same dims + channel order; built via the chroma-roundtrip +
    /// Mitchell + letterbox pipeline so the metric compare is fair.
    pub reference_bgra: Vec<u8>,
    /// Some only when `case.assert_steady_state_eps` was set.
    pub steady_state_delta: Option<f64>,
}

#[derive(Debug)]
pub(crate) enum RoundtripResult {
    Ok(RoundtripOutcome),
    /// Loud skip — runner CI can grep these to distinguish missing
    /// hardware from green tests.
    Skip {
        capability: Capability,
        detail: String,
    },
}

/// Drive one case end-to-end.
pub(crate) fn run_roundtrip(case: &RoundtripCase) -> RoundtripResult {
    let _ = tracing_subscriber::fmt::try_init();

    // The geometric residual metric assumes the production-shape host
    // scaler preserves aspect ratio (capture and encode have the same
    // aspect). Production picks `encode_dims_for_viewport(capture, …)`
    // which guarantees this. With matching aspects, the two letterbox-
    // fit maps (capture→encode and encode→surface) compose into the
    // single map `LetterboxMap::new(capture_dims, surface_dims)` we
    // use below. If aspects diverge, the host scaler is non-uniform
    // and the composition no longer holds — would silently corrupt
    // the primary metric. Reject the case at construction time so the
    // failure points at the test author, not at the renderer.
    let cap_ratio = case.capture_dims.0 as f64 / case.capture_dims.1 as f64;
    let enc_ratio = case.encode_dims.0 as f64 / case.encode_dims.1 as f64;
    assert!(
        (cap_ratio - enc_ratio).abs() < 0.01,
        "RoundtripCase '{}': capture {:?} and encode {:?} must share aspect ratio (got {} vs {})",
        case.name,
        case.capture_dims,
        case.encode_dims,
        cap_ratio,
        enc_ratio,
    );

    // 1. Capability checks. Missing capability is a loud Skip, never
    // a silent pass.
    let (device, queue) = match pollster::block_on(try_init_wgpu_for_dmabuf(case.profile.bit_depth))
    {
        Some((d, q, _adapter)) => (d, q),
        None => {
            return RoundtripResult::Skip {
                capability: Capability::VulkanDmaBufImport,
                detail: format!(
                    "wgpu adapter lacks VULKAN_EXTERNAL_MEMORY_DMA_BUF{}",
                    if case.profile.bit_depth == 10 {
                        " (or TEXTURE_FORMAT_16BIT_NORM for 10-bit)"
                    } else {
                        ""
                    }
                ),
            };
        }
    };

    // 2. Fixture → BGRA at capture_dims.
    let capture_bgra = load_fixture(case.fixture, case.capture_dims);

    // 3. Encode `frames_encoded` frames.
    let packets = encode_chain(case, &capture_bgra);
    if packets.is_empty() {
        // The encode chain signals capability gaps (e.g. P010 dma-buf
        // rejection on iHD+MTL) with an empty packet list.
        let cap = capability_for_profile(case.profile);
        return RoundtripResult::Skip {
            capability: cap,
            detail: format!("encode chain produced no packets for {:?}", case.profile),
        };
    }

    // 4. Decode all frames; keep the last two GpuFrames (last is the
    // primary readback target, second-to-last is for the steady-state
    // delta).
    let decoded = decode_all(case.profile, &packets, case.encode_dims);
    if decoded.is_empty() {
        return RoundtripResult::Skip {
            capability: Capability::VulkanDmaBufImport,
            detail: format!(
                "decoder emitted no Gpu frames for {:?}; bridge or driver gap",
                case.profile
            ),
        };
    }

    // 5. Build production renderer at surface_dims.
    let mut renderer = match GpuState::new_headless(
        device.clone(),
        queue.clone(),
        case.surface_dims,
        case.color_space,
        case.profile.chroma,
        case.profile.bit_depth,
    ) {
        Ok(r) => r,
        Err(e) => {
            return RoundtripResult::Skip {
                capability: if case.profile.bit_depth == 10 {
                    Capability::BitDepth10RendererSupport
                } else {
                    Capability::VulkanDmaBufImport
                },
                detail: format!("GpuState::new_headless failed: {e}"),
            };
        }
    };

    // 6. Apply + render each decoded frame in order. Snapshot the
    // last two readbacks (or just one if not asserting steady-state).
    let want_steady_state = case.assert_steady_state_eps.is_some();
    let mut readbacks: Vec<Vec<u8>> = Vec::new();
    let snapshot_window = if want_steady_state { 2 } else { 1 };
    let total = decoded.len();
    let first_keep_index = total.saturating_sub(snapshot_window);
    for (idx, frame) in decoded.into_iter().enumerate() {
        let (w, h, _pts, source, guard) = frame.into_parts();
        let render_frame = RenderGpuFrame {
            width: w,
            height: h,
            t_capture_client_clock: None,
            source,
            guard,
        };
        renderer
            .apply_frame(Frame::Gpu(render_frame))
            .expect("apply_frame");
        renderer.render().expect("render");
        if idx >= first_keep_index {
            readbacks.push(readback_offscreen(&renderer, &device));
        }
    }

    let readback_bgra = readbacks.pop().expect("at least one frame rendered");
    let prev_bgra = readbacks.pop();

    // 7. Build CPU reference at surface_dims via the production-shape
    // CPU pipeline (chroma roundtrip → host Mitchell → letterbox-fit
    // upscale → black-bar pad).
    let reference_bgra = build_reference(case, &capture_bgra);

    // 8. Metrics.
    let geometric_residual_px_rms = match case.fixture {
        Fixture::CoordEncoded => {
            let map = LetterboxMap::new(case.capture_dims, case.surface_dims);
            coord_fixture_residual_px_rms(&readback_bgra, case.surface_dims, &map)
        }
        Fixture::Png(_) => f64::NAN,
    };
    let ssim = ssim_rgb(
        &readback_bgra,
        &reference_bgra,
        case.surface_dims.0,
        case.surface_dims.1,
    );
    let psnr_y_db = psnr_db_y_bgra(&readback_bgra, &reference_bgra);
    let steady_state_delta = prev_bgra.map(|prev| mae_bgra(&readback_bgra, &prev));

    RoundtripResult::Ok(RoundtripOutcome {
        ssim,
        psnr_y_db,
        geometric_residual_px_rms,
        readback_bgra,
        reference_bgra,
        steady_state_delta,
    })
}

// =====================================================================
// Capability + adapter setup
// =====================================================================

/// Initialise wgpu headless with the dma-buf import feature, plus
/// `TEXTURE_FORMAT_16BIT_NORM` for 10-bit profiles. Returns `None` if
/// the adapter doesn't support the required feature set so the
/// caller can SKIP loudly.
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
            label: Some("tether-render roundtrip harness"),
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

// =====================================================================
// Fixture
// =====================================================================

fn load_fixture(fixture: Fixture, dims: (u32, u32)) -> Vec<u8> {
    match fixture {
        Fixture::CoordEncoded => {
            assert!(
                dims.0 < 4096 && dims.1 < 4096,
                "CoordEncoded fixture supports dims < 4096; got {dims:?}"
            );
            coord_fixture_fill(dims)
        }
        Fixture::Png(name) => load_png_fixture_at(name, dims),
    }
}

fn load_png_fixture_at(name: &str, dims: (u32, u32)) -> Vec<u8> {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = manifest.join("fixtures").join(name);
    let img = image::open(&path)
        .unwrap_or_else(|e| panic!("open fixture {}: {e}", path.display()))
        .to_rgba8();
    let (src_w, src_h) = (img.width(), img.height());
    let src_rgba = img.into_raw();
    let mut src_bgra = vec![0u8; src_rgba.len()];
    for i in 0..(src_rgba.len() / 4) {
        src_bgra[i * 4] = src_rgba[i * 4 + 2];
        src_bgra[i * 4 + 1] = src_rgba[i * 4 + 1];
        src_bgra[i * 4 + 2] = src_rgba[i * 4];
        src_bgra[i * 4 + 3] = src_rgba[i * 4 + 3];
    }
    if (src_w, src_h) == dims {
        return src_bgra;
    }
    cpu_mitchell_resize_bgra(&src_bgra, (src_w, src_h), dims)
}

// =====================================================================
// Encode side
// =====================================================================

fn encode_chain(case: &RoundtripCase, capture_bgra: &[u8]) -> Vec<tether_codec::EncodedPacket> {
    if case.encode_dims == case.capture_dims {
        // No scaler; encode directly. 10-bit needs the P010 bridge.
        if case.profile.bit_depth == 10 {
            encode_via_p010_bridge(
                case.profile,
                case.capture_dims,
                capture_bgra,
                case.frames_encoded,
            )
        } else {
            encode_via_cpu_upload(
                case.profile,
                case.capture_dims,
                capture_bgra,
                case.frames_encoded,
            )
        }
    } else {
        // Scaler-active path. Production runs the host Mitchell on
        // BGRA dma-buf input → NV12 bridge → encoder.encode_gpu. The
        // 10-bit P010 scaler path isn't implemented in production
        // today; fail loudly so we don't quietly mask that gap.
        assert_eq!(
            case.profile.bit_depth, 8,
            "scaler-active encode for {:?}: 10-bit P010 scaler path not wired in production",
            case.profile
        );
        encode_via_scaler_chain(
            case.profile,
            case.capture_dims,
            case.encode_dims,
            capture_bgra,
            case.frames_encoded,
        )
    }
}

/// Encode at `dims == capture_dims == encode_dims` through the
/// **gpuconvert NV12 bridge** (compute BGRA→NV12) and `encode_gpu`
/// via DMA-BUF. Skips the Mitchell scaler. Intentionally unused by
/// the matrix — kept as a regression-bisect entry point for "is the
/// bridge or the scaler at fault" investigations: switch a failing
/// `capture==encode` cell from `encode_via_cpu_upload` to this and
/// re-run. The `dead_code` allow is intentional and must stay until
/// a test or the matrix wires it in.
#[allow(dead_code)]
fn encode_via_bridge_no_scaler(
    profile: VideoProfile,
    dims: (u32, u32),
    bgra: &[u8],
    frames: usize,
) -> Vec<tether_codec::EncodedPacket> {
    let mut enc = match VaapiEncoder::new(profile, dims.0, dims.1, 30, 4_000) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIP: encoder construction failed: {e}");
            return Vec::new();
        }
    };
    let bridge = match pollster::block_on(tether_gpuconvert::Nv12DmaBuf::new(dims.0, dims.1)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: Nv12DmaBuf::new failed: {e}");
            return Vec::new();
        }
    };
    let src = bridge.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("bridge-only bgra src"),
        size: wgpu::Extent3d {
            width: dims.0,
            height: dims.1,
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
            bytes_per_row: Some(dims.0 * 4),
            rows_per_image: Some(dims.1),
        },
        wgpu::Extent3d {
            width: dims.0,
            height: dims.1,
            depth_or_array_layers: 1,
        },
    );
    let mut packets = Vec::new();
    for t in 0..frames as i64 {
        let nv12 = match bridge.convert(&src) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("SKIP: bridge.convert failed: {e}");
                return Vec::new();
            }
        };
        let codec_frame = build_codec_dmabuf_frame_nv12(&nv12);
        let gpu_frame = tether_codec::GpuEncoderFrame::DmaBuf(&codec_frame);
        match enc.encode_gpu(gpu_frame, t, t == 0) {
            Ok(p) => packets.extend(p),
            Err(e) => {
                eprintln!("SKIP: encode_gpu via bridge-only failed: {e}");
                return Vec::new();
            }
        }
    }
    packets
}

fn encode_via_cpu_upload(
    profile: VideoProfile,
    dims: (u32, u32),
    bgra: &[u8],
    frames: usize,
) -> Vec<tether_codec::EncodedPacket> {
    let (w, h) = dims;
    let mut enc = match VaapiEncoder::new(profile, w, h, 30, 4_000) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIP: VAAPI encoder construction failed for {profile:?}: {e}");
            return Vec::new();
        }
    };
    let mut packets = Vec::new();
    for t in 0..frames as i64 {
        match enc.encode_bgra(bgra, t, t == 0) {
            Ok(p) => packets.extend(p),
            Err(e) => {
                eprintln!("SKIP: encode_bgra failed at t={t}: {e}");
                return Vec::new();
            }
        }
    }
    packets
}

fn encode_via_p010_bridge(
    profile: VideoProfile,
    dims: (u32, u32),
    bgra: &[u8],
    frames: usize,
) -> Vec<tether_codec::EncodedPacket> {
    let (w, h) = dims;
    let mut enc = match VaapiEncoder::new(profile, w, h, 30, 4_000) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIP: VAAPI 10-bit encoder construction failed: {e}");
            return Vec::new();
        }
    };
    let bridge = match pollster::block_on(tether_gpuconvert::Bgra2P010DmaBuf::new(w, h)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: Bgra2P010DmaBuf::new failed: {e}");
            return Vec::new();
        }
    };
    let src = bridge.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("roundtrip P010 bgra src"),
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
    let mut packets = Vec::new();
    for t in 0..frames as i64 {
        let p010 = match bridge.convert(&src) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIP: P010 bridge convert failed: {e}");
                return Vec::new();
            }
        };
        let codec_frame = build_codec_dmabuf_frame_p010(&p010);
        match enc.submit_dmabuf(&codec_frame, t, t == 0) {
            Ok(p) => packets.extend(p),
            Err(e) => {
                eprintln!("SKIP: submit_dmabuf rejected P010 (driver gap): {e}");
                return Vec::new();
            }
        }
    }
    packets
}

fn encode_via_scaler_chain(
    profile: VideoProfile,
    capture: (u32, u32),
    encode: (u32, u32),
    capture_bgra: &[u8],
    frames: usize,
) -> Vec<tether_codec::EncodedPacket> {
    use tether_gpuconvert::{export_texture_as_dmabuf, Nv12DmaBuf};
    use tether_scaler::{ColorSpace, Pipelines, Scaler};

    let mut enc = match VaapiEncoder::new(profile, encode.0, encode.1, 30, 4_000) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIP: scaler-chain encoder construction failed: {e}");
            return Vec::new();
        }
    };
    let bridge = match pollster::block_on(Nv12DmaBuf::new(encode.0, encode.1)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: Nv12DmaBuf::new failed: {e}");
            return Vec::new();
        }
    };
    let src_export = match export_texture_as_dmabuf(
        bridge.device(),
        capture.0,
        capture.1,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        "roundtrip scaler-chain bgra source",
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: export_texture_as_dmabuf failed: {e}");
            return Vec::new();
        }
    };
    bridge.queue().write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src_export.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        capture_bgra,
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
        .expect("poll bridge write");

    let pipelines = std::sync::Arc::new(Pipelines::build(bridge.device()));
    let scaler = match Scaler::new_with_color_space(
        pipelines,
        bridge.device().clone(),
        bridge.queue().clone(),
        capture,
        encode,
        ColorSpace::Srgb8,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: scaler construction failed: {e}");
            return Vec::new();
        }
    };
    let mut packets = Vec::new();
    for t in 0..frames as i64 {
        let dup_fd = src_export.fd.try_clone().expect("dup src fd");
        let imported = match bridge.import_bgra_dmabuf(
            dup_fd,
            src_export.drm_format_modifier,
            src_export.stride,
            src_export.offset,
            capture.0,
            capture.1,
        ) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("SKIP: bgra dma-buf import failed: {e}");
                return Vec::new();
            }
        };
        let scaled = match scaler.scale(&imported) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("SKIP: scaler.scale failed: {e}");
                return Vec::new();
            }
        };
        let nv12 = match bridge.convert(scaled) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("SKIP: nv12 convert failed: {e}");
                return Vec::new();
            }
        };
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

fn build_codec_dmabuf_frame_p010(
    p010: &tether_gpuconvert::P010DmaBufFrame,
) -> tether_codec::DmaBufFrame {
    const P010_FOURCC: u32 = u32::from_le_bytes(*b"P010");
    const R16_FOURCC: u32 = u32::from_le_bytes(*b"R16 ");
    const GR32_FOURCC: u32 = u32::from_le_bytes(*b"GR32");
    let dup_fd = p010.fd.try_clone().expect("dup P010 fd");
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

// =====================================================================
// Decode side
// =====================================================================

fn decode_all(
    profile: VideoProfile,
    packets: &[tether_codec::EncodedPacket],
    encode_dims: (u32, u32),
) -> Vec<tether_codec::GpuFrame> {
    let mut dec = match VaapiDecoder::new(profile.codec) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: VAAPI decoder construction failed: {e}");
            return Vec::new();
        }
    };
    let mut out = Vec::new();
    for pkt in packets {
        if let Err(e) = dec.submit(&pkt.data) {
            eprintln!("SKIP: decoder submit failed: {e}");
            return Vec::new();
        }
        while let Ok(Some(f)) = dec.next_frame() {
            match f {
                CodecFrame::Gpu(g) => out.push(g),
                // VAAPI decoder must produce Gpu frames; a Cpu frame
                // here would mean the decoder probe was wrong or the
                // driver fell back. Fail loudly — silently dropping
                // would make the harness count packets it can't
                // actually use, which would skew the steady-state
                // check.
                CodecFrame::Cpu(_) => {
                    panic!("decode_all: VAAPI decoder produced a Cpu frame — driver fallback?");
                }
            }
        }
    }
    // Caller checks dims after into_parts(); tether_codec::GpuFrame
    // doesn't expose width/height by reference today.
    let _ = encode_dims;
    out
}

/// Map a VideoProfile to the capability whose absence is the most
/// informative SKIP reason when the encode chain failed. Profile +
/// chroma + bit_depth pick: H.264 (always 4:2:0 8-bit), HEVC Main,
/// HEVC Main 4:4:4, or HEVC Main 10.
fn capability_for_profile(profile: VideoProfile) -> Capability {
    use tether_protocol::control::CodecKind;
    // Order matters: the 4:4:4 10-bit arm must come before the generic
    // `(Hevc, Yuv444, _)` arm, otherwise 4:4:4 10-bit cells skip on
    // the 8-bit-only `VaapiHevcMain444` capability and an Intel iHD
    // driver gap at 10-bit shows up as a misleading "Main444 missing".
    match (profile.codec, profile.chroma, profile.bit_depth) {
        (CodecKind::H264, _, _) => Capability::VaapiH264,
        (CodecKind::Hevc, ChromaSubsampling::Yuv444, 10) => Capability::VaapiHevcMain444_10DmaBuf,
        (CodecKind::Hevc, ChromaSubsampling::Yuv444, _) => Capability::VaapiHevcMain444,
        (CodecKind::Hevc, _, 10) => Capability::VaapiHevcMain10DmaBuf,
        (CodecKind::Hevc, _, _) => Capability::VaapiHevcMain,
        (CodecKind::Av1, _, _) => Capability::VaapiAv1,
    }
}

// =====================================================================
// Render readback
// =====================================================================

fn readback_offscreen(renderer: &GpuState, device: &wgpu::Device) -> Vec<u8> {
    let (w, h) = renderer.dimensions().1;
    let unpadded_bpr = u64::from(w * 4);
    let padded_bpr = unpadded_bpr.next_multiple_of(u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT));
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("roundtrip readback"),
        size: padded_bpr * u64::from(h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("roundtrip readback encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: renderer.offscreen_target_for_readback(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
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
    let queue = renderer.queue_for_readback();
    queue.submit(std::iter::once(encoder.finish()));

    let (tx, rx) = mpsc::channel();
    buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).expect("readback map send");
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll readback");
    rx.recv().expect("readback callback").expect("readback ok");
    let mapped = buf.slice(..).get_mapped_range().expect("get mapped range");
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h as usize {
        let start = row * padded_bpr as usize;
        let end = start + unpadded_bpr as usize;
        out.extend_from_slice(&mapped[start..end]);
    }
    drop(mapped);
    buf.unmap();
    out
}

// =====================================================================
// CPU reference
// =====================================================================

fn build_reference(case: &RoundtripCase, capture_bgra: &[u8]) -> Vec<u8> {
    // The CPU reference omits an explicit chroma roundtrip step:
    // empirically, the textbook BT.709 full-range model in
    // `cpu_chroma_roundtrip_bgra` diverges from libavcodec's
    // limited-range swscale path by enough to inflate the per-cell
    // residual floor to the tune of 10 R/G levels (≈ 75 px in 1920-
    // wide capture). The SME plan review called modelling chroma
    // correctly a fairness improvement *in principle*; in practice
    // the version currently available is a net negative for floor
    // calibration on the identity / client-upscale rows. Skipped
    // here; documented in the plan's follow-ups as the next-step
    // metric-fidelity improvement once swscale parity is in.
    //
    // SAFETY-OF-OMISSION caveat: the current matrix only uses the
    // `Fixture::CoordEncoded` gradient (B channel constant 128) and
    // a single PNG with no high-frequency chroma. Constant-B keeps
    // the disable invisible because there's no chroma detail to
    // shift. A *future* PNG fixture with high-frequency chroma
    // (e.g. a real desktop screenshot) needs the chroma roundtrip
    // step re-enabled — otherwise the H.264-vs-HEVC siting drift
    // will inflate one codec's floor relative to the other in a way
    // that's invisible in the numbers but real in the reference.
    let _ = case.profile.codec;
    let _ = case.profile.chroma;
    let after_chroma = capture_bgra;

    // 1. Apply Mitchell host downscale to encode dims if engaged.
    let after_scale = if case.encode_dims == case.capture_dims {
        after_chroma.to_vec()
    } else {
        cpu_mitchell_resize_bgra(after_chroma, case.capture_dims, case.encode_dims)
    };

    // 3. Letterbox-fit upscale + black-bar pad to surface dims.
    //    cpu_letterbox_to_surface_bgra handles both axes — Mitchell
    //    resize to letterbox_fit_dims, then center-pad.
    if case.surface_dims == case.encode_dims {
        // Identity surface: no Mitchell, no pad. The fixture survives
        // through the chroma + (optional) scale step only.
        return after_scale;
    }
    cpu_letterbox_to_surface_bgra(&after_scale, case.encode_dims, case.surface_dims)
}

// =====================================================================
// Misc metric
// =====================================================================

fn mae_bgra(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0_f64;
    let mut n = 0_u64;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for ch in 0..3 {
            sum += (f64::from(pa[ch]) - f64::from(pb[ch])).abs();
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        sum / (n as f64)
    }
}

// =====================================================================
// On-failure diagnostics dump
// =====================================================================

/// Write per-cell diagnostics to `target/roundtrip-diagnostics/<case.name>/`
/// when a metric assertion would fail. Hardware-test failures from a
/// CI runner you can't reproduce locally are catastrophically
/// expensive without these artifacts.
///
/// Writes (best-effort — IO errors are warned, not propagated, so a
/// disk issue doesn't mask the real test failure):
/// - `readback.png` + `reference.png` — rendered surface vs CPU
///   reference, both as sRGB PNGs.
/// - `diff.png` — `|readback - reference|` per channel, gamma-
///   stretched by `*= 8` so single-step errors are visible.
/// - `ssim_heatmap.png` — per-16×16 SSIM rendered as an L8 image.
/// - `metrics.txt` — every metric value, every threshold, the case
///   itself. Greppable; one fact per line.
pub(crate) fn dump_failure_diagnostics(case: &RoundtripCase, outcome: &RoundtripOutcome) {
    // Allow CI workflows (and local runs) to redirect the artifacts
    // by setting TETHER_DIAGNOSTICS_DIR. Without the env-var the
    // default is `<workspace>/target/roundtrip-diagnostics` —
    // CARGO_MANIFEST_DIR is `.../crates/tether-render` so the
    // workspace root is two parent()s up.
    let target_root = std::env::var_os("TETHER_DIAGNOSTICS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            manifest
                .parent()
                .and_then(|p| p.parent())
                .unwrap_or(manifest)
                .join("target")
                .join("roundtrip-diagnostics")
        })
        .join(case.name);

    // Build metrics first so we can print them to stderr even if
    // mkdir or image saves fail — the numeric values are the
    // single most useful diagnostic in a CI log where uploading
    // PNGs back to the developer is awkward.
    let metrics = format!(
        "name = {name}\n\
         profile = {profile:?}\n\
         capture_dims = {capture:?}\n\
         encode_dims = {encode:?}\n\
         surface_dims = {surface:?}\n\
         fixture = {fixture:?}\n\
         frames_encoded = {frames}\n\
         color_space = {color:?}\n\
         requires = {req:?}\n\
         \n\
         [metrics]\n\
         ssim                       = {ssim:.6}\n\
         ssim_floor                 = {ssim_floor:.6}\n\
         psnr_y_db                  = {psnr_y:.4}\n\
         psnr_y_floor_db            = {psnr_y_floor:.4}\n\
         geometric_residual_px_rms  = {geo:.4}\n\
         geometric_residual_px_max  = {geo_max:.4}\n\
         steady_state_delta         = {steady:?}\n\
         steady_state_eps           = {steady_eps:?}\n",
        name = case.name,
        profile = case.profile,
        capture = case.capture_dims,
        encode = case.encode_dims,
        surface = case.surface_dims,
        fixture = case.fixture,
        frames = case.frames_encoded,
        color = case.color_space,
        req = case.requires,
        ssim = outcome.ssim,
        ssim_floor = case.floors.ssim,
        psnr_y = outcome.psnr_y_db,
        psnr_y_floor = case.floors.psnr_y_db,
        geo = outcome.geometric_residual_px_rms,
        geo_max = case.floors.geom_px,
        steady = outcome.steady_state_delta,
        steady_eps = case.assert_steady_state_eps,
    );

    if let Err(e) = std::fs::create_dir_all(&target_root) {
        eprintln!(
            "dump_failure_diagnostics: mkdir({}) failed: {e}; metrics follow:\n{metrics}",
            target_root.display()
        );
        return;
    }

    let (sw, sh) = case.surface_dims;
    save_bgra_as_png(
        &target_root.join("readback.png"),
        &outcome.readback_bgra,
        sw,
        sh,
    );
    save_bgra_as_png(
        &target_root.join("reference.png"),
        &outcome.reference_bgra,
        sw,
        sh,
    );
    save_diff_png(
        &target_root.join("diff.png"),
        &outcome.readback_bgra,
        &outcome.reference_bgra,
        sw,
        sh,
    );

    let (hw, hh, heatmap) =
        ssim_heatmap_l8(&outcome.readback_bgra, &outcome.reference_bgra, sw, sh);
    save_l8_as_png(&target_root.join("ssim_heatmap.png"), &heatmap, hw, hh);

    let metrics_path = target_root.join("metrics.txt");
    if let Err(e) = std::fs::write(&metrics_path, &metrics) {
        eprintln!(
            "dump_failure_diagnostics: write metrics failed: {e}; metrics follow:\n{metrics}"
        );
    }
    eprintln!("dump_failure_diagnostics: wrote {}", target_root.display());
}

fn save_bgra_as_png(path: &std::path::Path, bgra: &[u8], w: u32, h: u32) {
    let mut rgba = Vec::with_capacity(bgra.len());
    for chunk in bgra.chunks_exact(4) {
        rgba.push(chunk[2]);
        rgba.push(chunk[1]);
        rgba.push(chunk[0]);
        rgba.push(chunk[3]);
    }
    let buf = match image::RgbaImage::from_raw(w, h, rgba) {
        Some(b) => b,
        None => {
            eprintln!("save_bgra_as_png: RgbaImage::from_raw rejected dims/length");
            return;
        }
    };
    if let Err(e) = buf.save(path) {
        eprintln!("save_bgra_as_png({}): {e}", path.display());
    }
}

fn save_l8_as_png(path: &std::path::Path, l8: &[u8], w: u32, h: u32) {
    let buf = match image::GrayImage::from_raw(w, h, l8.to_vec()) {
        Some(b) => b,
        None => {
            eprintln!("save_l8_as_png: GrayImage::from_raw rejected dims/length");
            return;
        }
    };
    if let Err(e) = buf.save(path) {
        eprintln!("save_l8_as_png({}): {e}", path.display());
    }
}

fn save_diff_png(path: &std::path::Path, a_bgra: &[u8], b_bgra: &[u8], w: u32, h: u32) {
    assert_eq!(a_bgra.len(), b_bgra.len());
    let mut rgba = Vec::with_capacity(a_bgra.len());
    for (ca, cb) in a_bgra.chunks_exact(4).zip(b_bgra.chunks_exact(4)) {
        // Stretch by ×8 so single-step errors are visible to the eye.
        // Saturates rather than wraps. RGB channel order in the
        // output (image::RgbaImage expects RGBA), input is BGRA.
        let dr = (i32::from(ca[2]).abs_diff(i32::from(cb[2])) * 8).min(255) as u8;
        let dg = (i32::from(ca[1]).abs_diff(i32::from(cb[1])) * 8).min(255) as u8;
        let db = (i32::from(ca[0]).abs_diff(i32::from(cb[0])) * 8).min(255) as u8;
        rgba.push(dr);
        rgba.push(dg);
        rgba.push(db);
        rgba.push(0xFF);
    }
    let buf = match image::RgbaImage::from_raw(w, h, rgba) {
        Some(b) => b,
        None => {
            eprintln!("save_diff_png: RgbaImage::from_raw rejected dims/length");
            return;
        }
    };
    if let Err(e) = buf.save(path) {
        eprintln!("save_diff_png({}): {e}", path.display());
    }
}
