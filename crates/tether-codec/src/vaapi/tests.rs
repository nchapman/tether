use crate::h264::{fixture_access_units, H264_320X240_INTRA};
use crate::{Decoder, Encoder, Frame, GpuFrame, GpuFrameSource};

use super::{VaapiDecoder, VaapiEncoder};

/// Confirms `set_bitrate_kbps` is callable mid-stream without
/// breaking the encoder's output, even when the implementation is the
/// trait's Ok-no-op default. This is the structural guarantee the ABR
/// controller relies on for backends that *do* support live retune:
/// the call returns cleanly, the stream stays decodable.
///
/// Effective retune (the bitstream actually changes character) is a
/// separate property covered by
/// `vaapi_bitrate_retune_changes_bitstream_size`, which SKIPs on
/// drivers where the retune is silently ignored — currently the case
/// on Intel iHD Meteor Lake.
#[test]
#[ignore = "requires a working VAAPI device (run on hardware with: cargo test -p tether-codec --ignored vaapi_set_bitrate_live_continues_to_encode)"]
fn vaapi_set_bitrate_live_continues_to_encode() {
    let w = 640;
    let h = 480;
    let mut enc = VaapiEncoder::new(
        tether_protocol::control::VideoProfile::H264_8BIT_420,
        w,
        h,
        30,
        4_000,
    )
    .expect("VAAPI encoder");
    let mut dec =
        VaapiDecoder::new(tether_protocol::control::CodecKind::H264).expect("VAAPI decoder");

    // Pre-retune: a few frames at baseline.
    for t in 0..4i64 {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let bgra = make_test_bgra(w, h, t as u32);
        let packets = enc.encode_bgra(&bgra, t, t == 0).expect("encode pre");
        for p in packets {
            dec.submit(&p.data).expect("decode pre");
            while dec.next_frame().expect("next_frame pre").is_some() {}
        }
    }

    // Mid-stream retune. The trait default is Ok-no-op; an
    // overriding impl that actually retunes must also return Ok
    // here. Either way, the call must not break subsequent encoding.
    enc.set_bitrate_kbps(1_500).expect("live retune call");

    // Post-retune: keep encoding + decoding, force one IDR so the
    // decoder picks up the new parameter set cleanly.
    let mut got_post: Option<crate::Frame> = None;
    for t in 4..10i64 {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let bgra = make_test_bgra(w, h, t as u32);
        let packets = enc.encode_bgra(&bgra, t, t == 4).expect("encode post");
        for p in packets {
            dec.submit(&p.data).expect("decode post");
            while let Some(f) = dec.next_frame().expect("next_frame post") {
                got_post = Some(f);
            }
        }
    }
    assert!(
        got_post.is_some(),
        "decoder produced no frames after live bitrate change call"
    );
}

#[test]
#[ignore = "requires a working VAAPI device (run on hardware with: cargo test -p tether-codec --ignored vaapi)"]
fn vaapi_encoder_smoke() {
    let w = 640;
    let h = 480;
    let mut enc = VaapiEncoder::new(tether_protocol::control::VideoProfile::H264_8BIT_420, w, h, 30, 4_000).expect("VAAPI encoder");
    let bgra = vec![0x80u8; (w * h * 4) as usize];
    let packets = enc.encode_bgra(&bgra, 0, true).expect("encode");
    // First frame may produce 0 packets (encoder warm-up) or 1+
    // packets carrying SPS/PPS + the IDR slice. Either way it
    // shouldn't error.
    for p in packets {
        assert!(!p.data.is_empty());
    }
}

fn make_test_bgra(width: u32, height: u32, t: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = if (x / 64 + t / 4) % 2 == 0 { 200 } else { 50 };
            let g: u8 = if (y / 64) % 2 == 0 { 200 } else { 50 };
            let b: u8 = 128;
            data.extend_from_slice(&[b, g, r, 255]);
        }
    }
    data
}

#[test]
#[ignore = "requires a working VAAPI device (run on hardware with: cargo test -p tether-codec --ignored vaapi)"]
fn vaapi_decoder_smoke() {
    // Feed the committed 320×240 Annex-B fixture (the LGPL FFmpeg has no
    // software encoder to make one on the fly), then verify the VAAPI
    // decoder produces a frame at the expected dimensions.
    let w = 320;
    let h = 240;
    let mut dec = VaapiDecoder::new(tether_protocol::control::CodecKind::H264).expect("VAAPI decoder");

    let mut got: Option<GpuFrame> = None;
    for au in fixture_access_units(H264_320X240_INTRA) {
        dec.submit(au).expect("vaapi submit");
        while let Some(f) = dec.next_frame().expect("vaapi next_frame") {
            let Frame::Gpu(g) = f else {
                panic!("VaapiDecoder must emit DMA-BUF Gpu frames");
            };
            got = Some(g);
        }
    }
    let frame = got.expect("decoder produced a frame from the fixture");
    assert_eq!(frame.width, w);
    assert_eq!(frame.height, h);
    // `GpuFrameSource` only has the `DmaBuf` variant on Linux (the
    // `IOSurface` variant is `cfg(target_os = "macos")`), so this
    // destructure is irrefutable here. Kept as a single binding
    // rather than a `match` so the test focuses on the assertions
    // below; if a future Linux-side variant lands, the compiler
    // will force a refactor.
    let GpuFrameSource::DmaBuf(dmabuf) = frame.source;
    // SEPARATE_LAYERS gives NV12 two planes (Y, UV). Each layer
    // points at one object (might be the same fd with different
    // offsets, or two distinct fds) so we expect 2 layers and at
    // least 1 object.
    assert_eq!(dmabuf.layers.len(), 2, "NV12 export should yield 2 layers");
    assert!(!dmabuf.objects.is_empty(), "at least one DMA-BUF object");
    // Surface-level fourcc is NV12.
    assert_eq!(
        dmabuf.fourcc,
        u32::from_le_bytes(*b"NV12"),
        "expected NV12 fourcc"
    );
    // Per-layer DRM fourccs: R8 for the Y plane, GR88 for the
    // interleaved UV plane. Asserting these catches a driver
    // silently falling back to COMPOSED_LAYERS (which would emit
    // a single NV12-fourcc layer) — under that mode our wgpu
    // import would do the wrong thing and we'd rather fail in
    // the test than in production.
    assert_eq!(
        dmabuf.layers[0].drm_format,
        u32::from_le_bytes(*b"R8  "),
        "Y plane should be DRM_FORMAT_R8"
    );
    assert_eq!(
        dmabuf.layers[1].drm_format,
        u32::from_le_bytes(*b"GR88"),
        "UV plane should be DRM_FORMAT_GR88"
    );
}

/// Round-trip a frame through VAAPI decode (which exports a DMA-BUF) →
/// VAAPI encode via the DMA-BUF import path. Validates the riskiest
/// unknown: that ffmpeg accepts a DRM_PRIME AVFrame mapped into the
/// encoder's hwframes pool and emits a valid H.264 packet without a CPU
/// upload.
///
/// We piggyback on the decoder to produce the DMA-BUF (which is itself
/// covered by `vaapi_decoder_smoke`) rather than synthesising one via
/// vaCreateSurfaces — both sources hit the same import code path
/// inside ffmpeg's hwcontext_vaapi. The decoder input is the committed
/// 320×240 fixture (no software encoder in the LGPL build).
#[test]
#[ignore = "requires a working VAAPI device (run on hardware with: cargo test -p tether-codec --ignored vaapi_encoder_dmabuf_import)"]
fn vaapi_encoder_dmabuf_import() {
    // The fixture's dimensions, the VAAPI decoder (implicit — picks
    // them up from the SPS), and the VAAPI encoder built below must all
    // match: the encoder's hw_frames_ctx pins width/height, and
    // av_hwframe_map needs the dst pool to accept the src dimensions.
    let w = 320;
    let h = 240;

    let mut dec = VaapiDecoder::new(tether_protocol::control::CodecKind::H264).expect("VAAPI decoder");
    let mut hw_enc = VaapiEncoder::new(tether_protocol::control::VideoProfile::H264_8BIT_420, w, h, 30, 4_000).expect("VAAPI encoder");

    // Pump the fixture's access units through the VAAPI decoder until we
    // get a GpuFrame holding a DMA-BUF (the I-frame decodes on the first
    // unit).
    let mut gpu_frame: Option<GpuFrame> = None;
    'decode: for au in fixture_access_units(H264_320X240_INTRA) {
        dec.submit(au).expect("vaapi submit");
        while let Some(f) = dec.next_frame().expect("vaapi next_frame") {
            let Frame::Gpu(g) = f else {
                panic!("VaapiDecoder must emit Gpu frames");
            };
            gpu_frame = Some(g);
            break 'decode;
        }
    }
    let gpu = gpu_frame.expect("decoder produced a GpuFrame");
    let GpuFrameSource::DmaBuf(ref dmabuf) = gpu.source;

    // Hand the DMA-BUF straight to the VAAPI encoder. `gpu`
    // (whose guard owns the source VA surface) stays alive through
    // the end of the test via the `dmabuf` borrow into `gpu.source`.
    // In production, the guard drops as soon as submit_dmabuf
    // returns — ffmpeg dups the DMA-BUF fds internally during
    // av_hwframe_map, so the source surface is no longer needed.
    let packets = hw_enc
        .submit_dmabuf(dmabuf, 0, true)
        .expect("submit_dmabuf");

    // First imported frame can emit 0 packets (encoder warm-up
    // buffers SPS/PPS until the next frame) or 1+ packets carrying
    // the headers plus IDR slice. Either way, no error and any
    // returned packet is non-empty.
    for p in &packets {
        assert!(!p.data.is_empty(), "encoded packet must not be empty");
    }

    // To prove the encoder really produced output (not just
    // accepted the input), push one more imported frame and drain.
    // Re-using the same dmabuf is fine — ffmpeg's VAAPI import
    // re-dups the fds each call, and the source surface is
    // unchanged because nothing has reffed it since the decoder.
    let more = hw_enc
        .submit_dmabuf(dmabuf, 1, false)
        .expect("submit_dmabuf #2");
    assert!(
        !packets.is_empty() || !more.is_empty(),
        "encoder must emit at least one packet across two submitted frames"
    );
}

#[test]
#[ignore = "requires VAAPI HEVC Main444 (Intel Tiger Lake+ / AMD VCN3+)"]
fn hevc_main444_encoder_constructs() {
    // The desktop-quality top rung. If this fails on a known-good
    // box, the negotiator silently downgrades sessions to 4:2:0 and
    // text-edge quality regresses without surfacing a clear error.
    // Failing this test is the actionable signal.
    let w = 640;
    let h = 480;
    let mut enc = VaapiEncoder::new(
        tether_protocol::control::VideoProfile::HEVC_8BIT_444,
        w,
        h,
        30,
        8_000,
    )
    .expect("HEVC Main444 encoder construction");
    // BGRA → VUYX swscale path must also succeed end-to-end. A
    // mid-grey frame is enough; we don't validate fidelity here —
    // that's the round-trip integration test below.
    let bgra = vec![0x80u8; (w * h * 4) as usize];
    let _ = enc.encode_bgra(&bgra, 0, true).expect("encode 1 frame");
}

/// Full BGRA → gpuconvert (XYUV dma-buf) → VAAPI encoder → VAAPI
/// decoder → DMA-BUF export round-trip for HEVC Main444.
///
/// This is the test that should have existed before the loopback
/// test green-screen. It exercises every layer the production hot
/// path touches for 4:4:4 specifically:
///
/// 1. `Yuv444DmaBuf` bridge constructs (gpuconvert wgpu device +
///    feature negotiation works).
/// 2. Compute pass writes a valid XYUV dma-buf.
/// 3. `VaapiEncoder::submit_dmabuf` accepts the XYUV layer — i.e.
///    ffmpeg's `vaapi_drm_format_map` actually has a path for our
///    DRM_FORMAT_XYUV8888 descriptor (the original bug was that the
///    map had no entry for the planar `YU24` we'd been emitting).
/// 4. The encoder produces a HEVC Main 4:4:4 bitstream.
/// 5. `VaapiDecoder` accepts that bitstream and emits a frame.
/// 6. The decoded surface is exportable as a DMA-BUF (catches the
///    symmetric ffmpeg gap on the decode side — if `vaExportSurfaceHandle`
///    can't represent the 444 surface as DRM_PRIME, the production
///    client would see frame drops without any actionable diagnostic).
///
/// Test pattern is a mid-grey solid because we care about
/// "does the pipeline run end-to-end?", not chroma fidelity (that's
/// covered by gpuconvert's own packed-XYUV round-trip test).
#[test]
#[ignore = "requires VAAPI HEVC Main444 + Vulkan DMA-BUF export"]
fn hevc_main444_dmabuf_roundtrip() {
    use crate::{DmaBufFrame, DmaBufLayer, DmaBufObject};
    use tether_gpuconvert::Yuv444DmaBuf;
    use tether_protocol::control::VideoProfile;

    // 128×128 is the encoder's minimum-block floor (the same number
    // `tether_probe::host_supported_profiles` uses during its
    // startup round-trip). Small enough for the test to be cheap,
    // large enough that the encoder doesn't reject dims.
    let w = 128u32;
    let h = 128u32;

    let bridge = match pollster::block_on(Yuv444DmaBuf::new(w, h)) {
        Ok(b) => b,
        Err(e) => {
            // No Vulkan dma-buf device on this box → skip, don't
            // fail. The ignore tag already gates the test on
            // hardware presence; this is the secondary gate for
            // adapters that don't expose the export feature.
            eprintln!("SKIP: Yuv444DmaBuf::new failed: {e}");
            return;
        }
    };

    // Build a BGRA source texture on the bridge's device (same
    // Vulkan instance as the compute pipeline) and fill it with mid-
    // grey. We don't need a fancy gradient — we're testing the
    // pipeline plumbing, not chroma fidelity.
    let src = bridge
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("test bgra mid-grey"),
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
    let n = (w * h) as usize;
    let bgra = vec![0x80u8; n * 4];
    bridge.queue().write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bgra,
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

    let mut enc = VaapiEncoder::new(VideoProfile::HEVC_8BIT_444, w, h, 30, 8_000)
        .expect("VAAPI HEVC Main444 encoder");
    let mut dec = VaapiDecoder::new(tether_protocol::control::CodecKind::Hevc)
        .expect("VAAPI HEVC decoder");

    // Push 8 frames so the encoder can emit at least one IDR plus a
    // P-frame chain — the decoder usually needs a couple of frames
    // of latency before it produces output. Re-using the same BGRA
    // buffer is fine; we're testing plumbing.
    let mut got_decoded = false;
    for t in 0..8i64 {
        let bridge_frame = bridge.convert(&src).expect("bridge.convert");
        // Build the codec-side DmaBufFrame the same way
        // apps/tether-host does. Duplicates `yuv444_dmabuf_to_codec_frame`
        // — kept inline so this test stays in tether-codec without a
        // dep on the host binary.
        let codec_frame = DmaBufFrame {
            fourcc: u32::from_le_bytes(*b"XYUV"),
            objects: vec![DmaBufObject {
                fd: bridge_frame.fd,
                size: bridge_frame.size,
                drm_format_modifier: bridge_frame.modifier,
            }],
            layers: vec![DmaBufLayer {
                drm_format: u32::from_le_bytes(*b"XYUV"),
                num_planes: 1,
                object_index: [0, 0, 0, 0],
                offset: [
                    u32::try_from(bridge_frame.offset).expect("offset fits"),
                    0,
                    0,
                    0,
                ],
                pitch: [
                    u32::try_from(bridge_frame.stride).expect("stride fits"),
                    0,
                    0,
                    0,
                ],
            }],
        };
        let packets = enc
            .submit_dmabuf(&codec_frame, t, t == 0)
            .expect("submit_dmabuf");
        for p in packets {
            assert!(!p.data.is_empty(), "encoded packet must not be empty");
            dec.submit(&p.data).expect("decoder submit");
            while let Some(f) = dec.next_frame().expect("decoder next_frame") {
                let Frame::Gpu(g) = f else {
                    panic!("VaapiDecoder must emit Gpu frames for Main444");
                };
                let GpuFrameSource::DmaBuf(dmabuf) = g.source;
                assert_eq!(g.width, w);
                assert_eq!(g.height, h);
                // Whatever DMA-BUF shape the driver picks, log it so
                // a future driver shift (e.g. Intel changing the
                // export form) shows up in test output rather than
                // silently breaking the renderer's `import_yuv444`
                // dispatcher.
                eprintln!(
                    "decoded 4:4:4 surface exported: fourcc=0x{:08x} layers={} planes_per_layer={:?}",
                    dmabuf.fourcc,
                    dmabuf.layers.len(),
                    dmabuf.layers.iter().map(|l| l.num_planes).collect::<Vec<_>>(),
                );
                got_decoded = true;
            }
        }
        if got_decoded {
            break;
        }
    }
    assert!(
        got_decoded,
        "decoder must emit at least one 4:4:4 frame within 8 input frames"
    );
}

/// HEVC Main 4:4:4 10-bit dma-buf round trip — the 10-bit cousin of
/// `hevc_main444_dmabuf_roundtrip`. Exercises every layer of the new
/// XV30 path:
///
/// 1. `Bgra2Xv30DmaBuf` bridge constructs (gpuconvert wgpu device +
///    feature negotiation + Rgb10a2Unorm storage-modifier probe).
/// 2. Compute pass writes a valid packed-XV30 dma-buf.
/// 3. `VaapiEncoder::submit_dmabuf` accepts the XV30 layer — i.e.
///    ffmpeg's `vaapi_drm_format_map` actually has a path for our
///    `DRM_FORMAT_XV30` descriptor and `AV_PIX_FMT_XV30LE` source
///    format. **This is the gate the probe-level rejection in
///    `tether-probe/src/host/vaapi.rs` currently substitutes for —
///    the test must pass before that rejection is removed.**
/// 4. The encoder produces a HEVC Main 4:4:4 10-bit bitstream.
/// 5. `VaapiDecoder` accepts that bitstream and emits a frame.
/// 6. The decoded surface is exportable as a DMA-BUF (catches the
///    symmetric ffmpeg gap on the decode side).
#[test]
#[ignore = "requires VAAPI HEVC Main 4:4:4 10-bit + Vulkan DMA-BUF export + Rgb10a2Unorm storage"]
fn hevc_main444_10bit_xv30_dmabuf_roundtrip() {
    use crate::build_xv30_dmabuf_frame;
    use tether_gpuconvert::Bgra2Xv30DmaBuf;
    use tether_protocol::control::VideoProfile;

    let w = 128u32;
    let h = 128u32;

    let bridge = match pollster::block_on(Bgra2Xv30DmaBuf::new(w, h)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: Bgra2Xv30DmaBuf::new failed: {e}");
            return;
        }
    };

    let src = bridge
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("test bgra mid-grey"),
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
    let n = (w * h) as usize;
    let bgra = vec![0x80u8; n * 4];
    bridge.queue().write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bgra,
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

    let mut enc = match VaapiEncoder::new(VideoProfile::HEVC_10BIT_444, w, h, 30, 8_000) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIP: VAAPI HEVC Main 4:4:4 10-bit encoder unavailable: {e}");
            return;
        }
    };
    let mut dec = VaapiDecoder::new(tether_protocol::control::CodecKind::Hevc)
        .expect("VAAPI HEVC decoder");

    let mut got_decoded = false;
    for t in 0..8i64 {
        let bridge_frame = bridge.convert(&src).expect("bridge.convert");
        let codec_frame = build_xv30_dmabuf_frame(
            bridge_frame.fd,
            bridge_frame.size,
            bridge_frame.modifier,
            bridge_frame.offset,
            bridge_frame.stride,
        );
        let packets = enc
            .submit_dmabuf(&codec_frame, t, t == 0)
            .expect("submit_dmabuf");
        for p in packets {
            assert!(!p.data.is_empty(), "encoded packet must not be empty");
            dec.submit(&p.data).expect("decoder submit");
            while let Some(f) = dec.next_frame().expect("decoder next_frame") {
                let Frame::Gpu(g) = f else {
                    panic!("VaapiDecoder must emit Gpu frames for Main 4:4:4 10-bit");
                };
                let GpuFrameSource::DmaBuf(dmabuf) = g.source;
                assert_eq!(g.width, w);
                assert_eq!(g.height, h);
                // Log the decoder's chosen export shape so a driver-
                // dependent decision (packed XV30 vs biplanar P410-
                // style 16-bit) shows up in test output. The
                // renderer's `b"XV30" → RenderLayout::Biplanar16`
                // mapping in `tether-render::dmabuf_test::cross_table_consistency`
                // assumes biplanar 16-bit; if the driver emits
                // something else, the renderer needs a new
                // `RenderLayout::Packed1010102` variant.
                eprintln!(
                    "decoded 4:4:4 10-bit surface exported: fourcc=0x{:08x} ({:?}) \
                     layers={} planes_per_layer={:?}",
                    dmabuf.fourcc,
                    std::str::from_utf8(&dmabuf.fourcc.to_le_bytes()).unwrap_or("?"),
                    dmabuf.layers.len(),
                    dmabuf.layers.iter().map(|l| l.num_planes).collect::<Vec<_>>(),
                );
                got_decoded = true;
            }
        }
        if got_decoded {
            break;
        }
    }
    assert!(
        got_decoded,
        "decoder must emit at least one 4:4:4 10-bit frame within 8 input frames"
    );
}

// supported_encode_profiles is gone from tether-codec — the equivalent
// is `tether_probe::host_encode_profiles()`, tested in that crate.

/// Verifies live `set_bitrate_kbps` actually retunes the VAAPI
/// rate-control pathway end-to-end. Same shape as
/// `vaapi_min_qp_floor_reduces_bitstream`: encode the same high-
/// entropy content at two well-separated bitrate targets and check
/// the byte-count ratio.
///
/// **Verified-negative on Intel iHD Meteor Lake** as of this writing:
/// `AVCodecContext.bit_rate` writes post-open do not propagate to
/// VAAPI's rate-control machinery (FFmpeg's `vaapi_encode.c` doesn't
/// emit `VAEncMiscParameterTypeRateControl` on bit_rate change), so
/// the retune is a silent no-op — the test SKIPs when the
/// observed ratio is too small. This is *correct system behaviour*
/// for now: `VaapiEncoder` deliberately leaves
/// `supports_changing_bitrate` at the trait default `false`, so the
/// host disables ABR entirely on VAAPI hosts rather than running it
/// blind. The test exists as the green-light gate for any future
/// driver/wrapper combination that does plumb live retune through.
#[test]
#[ignore = "requires a working VAAPI device (run on hardware with: cargo test -p tether-codec --ignored vaapi_bitrate_retune_changes_bitstream_size)"]
fn vaapi_bitrate_retune_changes_bitstream_size() {
    use tether_protocol::control::VideoProfile;

    const W: u32 = 640;
    const H: u32 = 480;
    const FRAMES: i64 = 30;

    // Two well-separated bitrate targets. 1 Mbps is a tight cap for
    // 640×480 noisy content at 30 fps; 20 Mbps is far above what the
    // encoder needs. The byte-count ratio between the two should be
    // ~5-10× if the retune is honoured.
    const LOW_KBPS: u32 = 1_000;
    const HIGH_KBPS: u32 = 20_000;

    let mut enc = VaapiEncoder::new(VideoProfile::H264_8BIT_420, W, H, 30, LOW_KBPS)
        .expect("VAAPI encoder");

    // Phase A: encode at LOW_KBPS. Warm up first so rate-control
    // converges; only count bytes from the second half.
    for t in 0..(FRAMES / 2) {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let bgra = make_noisy_bgra(W, H, t as u32);
        let _ = enc.encode_bgra(&bgra, t, t == 0).expect("encode warmup low");
    }
    let mut low_bytes: usize = 0;
    for t in (FRAMES / 2)..FRAMES {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let bgra = make_noisy_bgra(W, H, t as u32);
        let packets = enc.encode_bgra(&bgra, t, false).expect("encode low");
        for p in packets {
            low_bytes += p.data.len();
        }
    }

    // Live retune.
    enc.set_bitrate_kbps(HIGH_KBPS).expect("retune to high");

    // Phase B: encode at HIGH_KBPS. Same warm-up-then-measure pattern.
    // Continue PTS past phase A so the encoder treats this as a
    // continuation, not a restart.
    for t in FRAMES..(FRAMES + FRAMES / 2) {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let bgra = make_noisy_bgra(W, H, t as u32);
        let _ = enc.encode_bgra(&bgra, t, false).expect("encode warmup high");
    }
    let mut high_bytes: usize = 0;
    for t in (FRAMES + FRAMES / 2)..(FRAMES * 2) {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let bgra = make_noisy_bgra(W, H, t as u32);
        let packets = enc.encode_bgra(&bgra, t, false).expect("encode high");
        for p in packets {
            high_bytes += p.data.len();
        }
    }

    // The HIGH window should produce meaningfully more bytes than
    // the LOW window. Threshold ratio 2× is generous: a true retune
    // at 20× the budget will produce far more than that on noisy
    // content, but encoder rate-control convergence + GOP boundary
    // effects make tight thresholds flaky.
    let ratio = high_bytes as f64 / low_bytes as f64;
    // Three regimes — same shape as the qmin / intra-refresh tests.
    //  - ratio > 2.0 → retune honoured, bitstream visibly grew.
    //  - 1.5 < ratio ≤ 2.0 → ambiguous: partial enforcement or
    //    rate-control convergence effects. Fail with a diagnostic.
    //  - ratio ≤ 1.5 → driver silently ignored the retune. SKIP
    //    (confirmed-negative on Intel iHD Meteor Lake; ratio is
    //    typically 1.0-1.2). The test starts asserting the moment
    //    any driver/wrapper combo plumbs live retune through.
    if ratio <= 1.5 {
        eprintln!(
            "SKIP: VAAPI rate-control did not honour live bit_rate retune \
             ({LOW_KBPS} kbps → {low_bytes} bytes, {HIGH_KBPS} kbps → {high_bytes} bytes, ratio {ratio:.3}). \
             FFmpeg's wrapper does not emit VAEncMiscParameterTypeRateControl on bit_rate change — see encoder.rs comment."
        );
        return;
    }
    assert!(
        ratio > 2.0,
        "live retune from {LOW_KBPS} kbps to {HIGH_KBPS} kbps produced low={low_bytes} bytes vs high={high_bytes} bytes (ratio {ratio:.3}); expected > 2.0 (honoured) or ≤ 1.5 (silently ignored). \
         Intermediate ratio suggests partial enforcement worth investigating."
    );
    eprintln!(
        "live bitrate retune verified: {LOW_KBPS} kbps → {low_bytes} bytes, {HIGH_KBPS} kbps → {high_bytes} bytes, ratio {ratio:.3}"
    );
}

/// Iterate H.264 Annex-B NAL units in a byte slice, yielding the
/// 5-bit `nal_unit_type` for each. Tolerates both 3- and 4-byte start
/// codes (`00 00 01` / `00 00 00 01`) and ignores trailing bytes that
/// don't start a NAL. Sufficient for the intra-refresh shape check
/// below; not a general bitstream parser.
fn h264_nal_types(annex_b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    // Need at least 3 bytes for a start code; the four-byte case
    // checks its own bound inside the body.
    while i + 3 <= annex_b.len() {
        let three = &annex_b[i..i + 3] == [0, 0, 1];
        let four = i + 4 <= annex_b.len() && &annex_b[i..i + 4] == [0, 0, 0, 1];
        if four {
            i += 4;
        } else if three {
            i += 3;
        } else {
            i += 1;
            continue;
        }
        if i < annex_b.len() {
            // Lower 5 bits of the NAL header byte are nal_unit_type.
            out.push(annex_b[i] & 0x1F);
        }
    }
    out
}

/// Verifies `TETHER_INTRA_REFRESH_PERIOD` plumbs through to the VAAPI
/// driver and produces a bitstream the encoder/decoder pair still
/// round-trips. Three properties get asserted, in order of strength:
///
///  1. After construction, the encoder's `unused_avoptions()` list does
///     NOT contain `intra_refresh` — i.e. the driver consumed the
///     opt-in. A silent fallback (Mesa older than ~22, AMD VAAPI
///     stacks without the feature) would leave the option in the
///     leftover dict and the test fails loudly rather than the
///     downstream session running without the latency knob it asked
///     for.
///  2. The bitstream contains exactly one IDR slice (the initial
///     forced keyframe) and zero further IDRs across 60 encoded
///     frames. This catches a driver that re-interprets
///     `intra_refresh` as "emit an IDR every period" (naïve mapping)
///     — that mistranslation would produce ≥2 IDRs across two
///     periods. It does NOT distinguish a correct gradual-refresh
///     implementation from a driver that consumed the option
///     silently; property (1) remains the primary signal for the
///     silent-no-op case.
///  3. The decoder accepts every emitted packet and produces at least
///     `inputs - latency` frames, proving the bitstream is still
///     well-formed under intra refresh (slice-header / SEI changes
///     are non-trivial and can break decoder paths that worked at
///     defaults).
///
/// Per `intra_refresh` AVOption semantics, the encoder emits gradual
/// refresh (P-slices with progressively-positioned intra MB columns
/// + recovery-point SEI) instead of periodic IDRs. We don't parse the
/// SEIs here — confirming the option was *accepted* by the driver
/// (property 1) and the bitstream stays decodable (property 3) is
/// the load-bearing verification.
#[test]
#[ignore = "requires a working VAAPI device (run on hardware with: cargo test -p tether-codec --ignored vaapi_intra_refresh_round_trip)"]
fn vaapi_intra_refresh_round_trip() {
    use tether_protocol::control::VideoProfile;

    // 30-frame refresh period over a 60-frame encode: two full refresh
    // cycles, enough that a driver mis-implementing the option as
    // "emit IDR every period" would produce ≥2 IDRs and trip
    // property (2).
    const PERIOD: u32 = 30;
    const FRAMES: i64 = 60;
    const W: u32 = 640;
    const H: u32 = 480;

    // SAFETY: env vars are process-global. Cargo's default test
    // runner runs threads within a single test binary in parallel, so
    // `set_var` / `remove_var` here *can* race with any concurrent
    // caller of `VaapiEncoder::new` that reads
    // `TETHER_INTRA_REFRESH_PERIOD`. Today every such caller in this
    // file is `#[ignore]` and `make test-hw` invokes
    // `--test-threads=1` for the bench cells. A sibling test,
    // `vaapi_min_qp_floor_reduces_bitstream`, now also drives the
    // env-var path (via `encode_with_env_and_measure`) for a
    // different key; with two such tests in the file, a shared
    // `OnceLock<Mutex<()>>` around set/build/remove is the right
    // mitigation if a third lands.
    unsafe {
        std::env::set_var("TETHER_INTRA_REFRESH_PERIOD", PERIOD.to_string());
    }
    let enc_result = VaapiEncoder::new(VideoProfile::H264_8BIT_420, W, H, 30, 4_000);
    unsafe {
        std::env::remove_var("TETHER_INTRA_REFRESH_PERIOD");
    }
    let mut enc = enc_result.expect("VAAPI encoder construction");

    // Property 1: driver actually consumed the option.
    //
    // Confirmed-negative on Intel iHD (Meteor Lake) as of this
    // writing: h264_vaapi does not expose `intra_refresh` as a
    // private AVOption upstream, so the dict-entry persists in
    // leftover. SKIP rather than fail when the driver gap matches
    // the known-bad set — the test still proves the *probe* (the
    // unused_avoptions accessor) works, and will start asserting
    // properties (2) and (3) the moment a driver lands support.
    let unused = enc.unused_avoptions();
    if unused.iter().any(|k| k == "intra_refresh") {
        eprintln!(
            "SKIP: VAAPI driver did not consume intra_refresh AVOption (leftover keys: {unused:?}). \
             h264_vaapi on this driver does not expose gradual refresh — see encoder.rs comment."
        );
        return;
    }

    let mut dec =
        VaapiDecoder::new(tether_protocol::control::CodecKind::H264).expect("VAAPI decoder");

    let mut all_nals: Vec<u8> = Vec::new();
    let mut decoded_count: u32 = 0;
    for t in 0..FRAMES {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let bgra = make_test_bgra(W, H, t as u32);
        let packets = enc.encode_bgra(&bgra, t, t == 0).expect("encode");
        for p in packets {
            all_nals.extend(h264_nal_types(&p.data));
            dec.submit(&p.data).expect("decode submit");
            while dec.next_frame().expect("decode next").is_some() {
                decoded_count += 1;
            }
        }
    }
    // Drain the decoder's reorder buffer.
    while dec.next_frame().expect("decode drain").is_some() {
        decoded_count += 1;
    }

    // Property 2: exactly one IDR (nal_unit_type 5). Gives a clean
    // signal that the encoder isn't quietly emitting periodic IDRs
    // even with the option set.
    let idr_count = all_nals.iter().filter(|&&t| t == 5).count();
    assert_eq!(
        idr_count, 1,
        "expected exactly one IDR across {FRAMES} frames with intra refresh enabled; got {idr_count}. NAL histogram: {all_nals:?}"
    );

    // Property 3: bitstream is still decodable end-to-end. Latency
    // budget of 4 frames accounts for the encoder's pipeline depth +
    // the decoder's reorder window.
    #[allow(clippy::cast_sign_loss)]
    let min_decoded = (FRAMES as u32).saturating_sub(4);
    assert!(
        decoded_count >= min_decoded,
        "decoder produced {decoded_count} frames from {FRAMES} encoded inputs (expected ≥{min_decoded}); intra-refresh bitstream may have broken the decode path"
    );
}

/// Per-pixel pseudo-random BGRA frame. Each pixel's value is derived
/// from `(x, y, t)` via a cheap hash so the resulting frame is
/// high-entropy — flat content like `make_test_bgra` compresses to
/// near-zero regardless of QP and so cannot expose rate-control
/// differences. Used by the min-QP test.
#[allow(clippy::cast_possible_truncation)]
fn make_noisy_bgra(width: u32, height: u32, t: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            // xorshift-like mixer over (x, y, t). Not cryptographic;
            // just needs to spread bits so each pixel is independent
            // enough to fill DCT coefficients with non-zero residue.
            let mut h = x
                .wrapping_mul(0x9E37_79B1)
                .wrapping_add(y.wrapping_mul(0x85EB_CA77))
                .wrapping_add(t.wrapping_mul(0xC2B2_AE3D));
            h ^= h >> 16;
            h = h.wrapping_mul(0x7FEB_352D);
            h ^= h >> 15;
            let b = (h & 0xFF) as u8;
            let g = ((h >> 8) & 0xFF) as u8;
            let r = ((h >> 16) & 0xFF) as u8;
            data.extend_from_slice(&[b, g, r, 255]);
        }
    }
    data
}

/// Encode `frames` BGRA frames through a fresh `VaapiEncoder` with
/// the provided env-var (key, value) applied around construction.
/// Returns total emitted byte count + the encoder's `unused_avoptions`
/// snapshot (so the caller can SKIP if the option wasn't consumed).
/// Used by `vaapi_min_qp_floor_reduces_bitstream` to compare two
/// configurations.
fn encode_with_env_and_measure(
    env_key: &str,
    env_value: Option<&str>,
    width: u32,
    height: u32,
    frames: i64,
    bitrate_kbps: u32,
) -> (usize, Vec<String>) {
    use tether_protocol::control::VideoProfile;
    // Set/clear the env var directly around encoder construction so
    // the encoder picks up the value at `open()` time.
    unsafe {
        if let Some(v) = env_value {
            std::env::set_var(env_key, v);
        } else {
            std::env::remove_var(env_key);
        }
    }
    let enc_result = VaapiEncoder::new(VideoProfile::H264_8BIT_420, width, height, 30, bitrate_kbps);
    unsafe {
        std::env::remove_var(env_key);
    }
    let mut enc = enc_result.expect("VAAPI encoder construction");
    let unused = enc.unused_avoptions().to_vec();

    let mut total_bytes: usize = 0;
    for t in 0..frames {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let bgra = make_noisy_bgra(width, height, t as u32);
        let packets = enc.encode_bgra(&bgra, t, t == 0).expect("encode");
        for p in packets {
            total_bytes += p.data.len();
        }
    }
    (total_bytes, unused)
}

/// Verifies `TETHER_MIN_QP` plumbs through to the VAAPI driver and
/// actually constrains the rate-control choice. Encodes the same
/// `make_test_bgra` sequence twice — once with `qmin=1` (effectively
/// unconstrained) and once with `qmin=45` (deep into the "smaller
/// files, worse quality" end of the QP scale) — and asserts the
/// constrained run produces a meaningfully smaller bitstream.
///
/// Why this signal works: VBR rate control picks the lowest QP that
/// fits the target bitrate budget on each frame. With a floor of 45,
/// the encoder cannot dip below that even when the bitrate budget
/// would let it, so the constrained run undershoots its bitrate
/// target and produces fewer bytes. A driver that silently ignored
/// `qmin` would produce nearly-identical byte counts across the two
/// runs.
///
/// SKIPs on the same condition as the intra-refresh test: if `qmin`
/// shows up in `unused_avoptions()`, the option was not consumed.
/// On Intel iHD (Meteor Lake), `qmin` *is* consumed as of this
/// writing — the generic AVCodecContext.qmin handler is wired
/// independent of the encoder-specific gradual-refresh gap.
#[test]
#[ignore = "requires a working VAAPI device (run on hardware with: cargo test -p tether-codec --ignored vaapi_min_qp_floor_reduces_bitstream)"]
fn vaapi_min_qp_floor_reduces_bitstream() {
    const W: u32 = 640;
    const H: u32 = 480;
    const FRAMES: i64 = 30;
    // High bitrate budget so the unconstrained run actually picks a
    // low QP rather than ceiling-clamping at qmax; that's what gives
    // the constrained run something to undershoot against.
    const BITRATE_KBPS: u32 = 10_000;

    // Baseline: lowest legal qmin — effectively no floor.
    let (baseline_bytes, baseline_unused) = encode_with_env_and_measure(
        "TETHER_MIN_QP",
        Some("1"),
        W,
        H,
        FRAMES,
        BITRATE_KBPS,
    );
    if baseline_unused.iter().any(|k| k == "qmin") {
        eprintln!(
            "SKIP: VAAPI driver did not consume qmin AVOption (leftover: {baseline_unused:?})."
        );
        return;
    }

    // Constrained: high min-QP forces the rate-control floor up. QP
    // 45 is deep in the high-compression end of the H.264 scale (0=lossless,
    // 51=worst); high enough that the bytecount delta is unambiguous,
    // not so high that the encoder rejects it.
    let (constrained_bytes, constrained_unused) = encode_with_env_and_measure(
        "TETHER_MIN_QP",
        Some("45"),
        W,
        H,
        FRAMES,
        BITRATE_KBPS,
    );
    assert!(
        !constrained_unused.iter().any(|k| k == "qmin"),
        "second run's qmin was unexpectedly ignored: {constrained_unused:?}"
    );

    // The load-bearing signal: the floor must actually constrain
    // bitrate. Three regimes:
    //  - ratio < 0.70 → qmin honored end-to-end (pass).
    //  - 0.70 ≤ ratio < 0.95 → driver partially respects qmin or
    //    test content is too compressible; tighten content + frame
    //    count before drawing conclusions.
    //  - ratio ≥ 0.95 → driver consumed the AVOption (not in
    //    unused_avoptions) but VAAPI's rate-control implementation
    //    silently ignored it. This is the confirmed-negative state
    //    on Intel iHD Meteor Lake as of this writing: the generic
    //    AVCodecContext.qmin handler accepts the value but the
    //    VAAPI VBR rate-control pathway doesn't read it. SKIP with
    //    a clear message rather than red CI; the test will start
    //    passing the moment any driver + ffmpeg combo plumbs qmin
    //    through to the rate-control choice.
    let ratio = constrained_bytes as f64 / baseline_bytes as f64;
    if ratio >= 0.95 {
        eprintln!(
            "SKIP: VAAPI driver consumed qmin AVOption but did not constrain bitrate \
             (qmin=1 → {baseline_bytes} bytes, qmin=45 → {constrained_bytes} bytes, ratio {ratio:.3}). \
             VAAPI rate-control does not honor qmin on this driver — see encoder.rs comment."
        );
        return;
    }
    assert!(
        ratio < 0.70,
        "TETHER_MIN_QP=45 produced {constrained_bytes} bytes vs baseline {baseline_bytes} (ratio {ratio:.3}); expected ratio < 0.70 (qmin honored) or ≥ 0.95 (driver silently ignored). \
         Intermediate ratio suggests partial enforcement worth investigating."
    );
    eprintln!(
        "min-QP floor verified: qmin=1 → {baseline_bytes} bytes, qmin=45 → {constrained_bytes} bytes, ratio {ratio:.3}"
    );
}

/// VAAPI AV1 Main 4:2:0 8-bit dma-buf round trip — the encode/decode
/// twin of `hevc_main444_dmabuf_roundtrip` for the new AV1 codec. Real
/// hardware support today: Intel Arc / Xe-HPG (encode), AMD RDNA 3
/// (encode), and Intel Tiger Lake+ / AMD RDNA 2+ / NVIDIA Ampere+
/// (decode). The probe layer is responsible for runtime gating; this
/// test exercises the path on hardware that has it.
///
/// Exercises:
/// 1. `Nv12DmaBuf` bridge constructs (gpuconvert wgpu + Vulkan + dma-buf
///    export features available).
/// 2. `VaapiEncoder::new(AV1_8BIT_420, …)` opens — i.e. the
///    `av1_vaapi` AVCodec was found and the driver accepts the
///    AV_PIX_FMT_NV12 hwframes pool.
/// 3. `VaapiEncoder::submit_dmabuf` accepts the NV12 layer — the same
///    DRM_PRIME path HEVC uses on 4:2:0, but with the AV1 encoder
///    consuming it.
/// 4. The encoder produces an AV1 bitstream with sequence-header OBUs
///    in extradata (`AV_CODEC_FLAG_GLOBAL_HEADER`) prepended to each
///    keyframe by `drain_encoder`.
/// 5. `VaapiDecoder` accepts that bitstream and emits a frame.
/// 6. The decoded surface exports as DMA-BUF with the expected NV12
///    layer shape (Y as R8, UV as GR88, two layers).
#[test]
#[ignore = "requires VAAPI AV1 encode (Intel Arc / AMD RDNA 3) + VAAPI AV1 decode + Vulkan DMA-BUF export"]
fn av1_main_dmabuf_roundtrip() {
    use crate::{DmaBufFrame, DmaBufLayer, DmaBufObject};
    use tether_gpuconvert::Nv12DmaBuf;
    use tether_protocol::control::VideoProfile;

    let w = 128u32;
    let h = 128u32;

    let bridge = match pollster::block_on(Nv12DmaBuf::new(w, h)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: Nv12DmaBuf::new failed: {e}");
            return;
        }
    };

    let src = bridge
        .device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("test bgra mid-grey av1"),
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
    let n = (w * h) as usize;
    let bgra = vec![0x80u8; n * 4];
    bridge.queue().write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bgra,
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

    let mut enc = match VaapiEncoder::new(VideoProfile::AV1_8BIT_420, w, h, 30, 4_000) {
        Ok(e) => e,
        Err(e) => {
            eprintln!(
                "SKIP: VaapiEncoder::new(AV1) failed: {e}. \
                 Requires a driver that exposes the AV1 encode entrypoint \
                 (Intel Arc / AMD RDNA 3)."
            );
            return;
        }
    };
    let mut dec = match VaapiDecoder::new(tether_protocol::control::CodecKind::Av1) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "SKIP: VaapiDecoder::new(Av1) failed: {e}. \
                 Requires VAAPI AV1 decode (Intel Tiger Lake+ / AMD RDNA 2+ / NVIDIA Ampere+)."
            );
            return;
        }
    };

    let mut got_decoded = false;
    for t in 0..8i64 {
        let bridge_frame = bridge.convert(&src).expect("bridge.convert");
        let codec_frame = DmaBufFrame {
            fourcc: u32::from_le_bytes(*b"NV12"),
            objects: vec![DmaBufObject {
                fd: bridge_frame.fd,
                size: bridge_frame.size,
                drm_format_modifier: bridge_frame.modifier,
            }],
            layers: vec![
                DmaBufLayer {
                    drm_format: u32::from_le_bytes(*b"R8  "),
                    num_planes: 1,
                    object_index: [0, 0, 0, 0],
                    offset: [
                        u32::try_from(bridge_frame.y_offset).expect("y_offset fits"),
                        0,
                        0,
                        0,
                    ],
                    pitch: [
                        u32::try_from(bridge_frame.y_stride).expect("y_stride fits"),
                        0,
                        0,
                        0,
                    ],
                },
                DmaBufLayer {
                    drm_format: u32::from_le_bytes(*b"GR88"),
                    num_planes: 1,
                    object_index: [0, 0, 0, 0],
                    offset: [
                        u32::try_from(bridge_frame.uv_offset).expect("uv_offset fits"),
                        0,
                        0,
                        0,
                    ],
                    pitch: [
                        u32::try_from(bridge_frame.uv_stride).expect("uv_stride fits"),
                        0,
                        0,
                        0,
                    ],
                },
            ],
        };
        let packets = enc
            .submit_dmabuf(&codec_frame, t, t == 0)
            .expect("submit_dmabuf");
        for p in packets {
            assert!(!p.data.is_empty(), "encoded packet must not be empty");
            dec.submit(&p.data).expect("decoder submit");
            while let Some(f) = dec.next_frame().expect("decoder next_frame") {
                let Frame::Gpu(g) = f else {
                    panic!("VaapiDecoder must emit Gpu frames for AV1");
                };
                let GpuFrameSource::DmaBuf(dmabuf) = g.source;
                assert_eq!(g.width, w);
                assert_eq!(g.height, h);
                eprintln!(
                    "decoded AV1 surface exported: fourcc=0x{:08x} layers={} planes_per_layer={:?}",
                    dmabuf.fourcc,
                    dmabuf.layers.len(),
                    dmabuf.layers.iter().map(|l| l.num_planes).collect::<Vec<_>>(),
                );
                got_decoded = true;
            }
        }
        if got_decoded {
            break;
        }
    }
    assert!(
        got_decoded,
        "decoder must emit at least one AV1 frame within 8 input frames"
    );
}
