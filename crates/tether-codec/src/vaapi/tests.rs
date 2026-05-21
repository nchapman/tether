use crate::h264::H264Encoder;
use crate::{Decoder, Encoder, Frame, GpuFrame, GpuFrameSource};

use super::{VaapiDecoder, VaapiEncoder};

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
    // Encode a few frames with the software encoder so we have a
    // valid Annex-B bitstream to decode, then verify the VAAPI
    // decoder produces a frame at the expected dimensions.
    let w = 320;
    let h = 240;
    let mut enc = H264Encoder::new_bgra(w, h, 30, 2_000).expect("sw encoder");
    let mut dec = VaapiDecoder::new(tether_protocol::control::CodecKind::H264).expect("VAAPI decoder");

    let mut got: Option<GpuFrame> = None;
    for t in 0..6i64 {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let bgra = make_test_bgra(w, h, t as u32);
        let packets = enc.encode_bgra(&bgra, t, t == 0).expect("encode");
        for p in packets {
            dec.submit(&p.data).expect("vaapi submit");
            while let Some(f) = dec.next_frame().expect("vaapi next_frame") {
                let Frame::Gpu(g) = f else {
                    panic!("VaapiDecoder must emit DMA-BUF Gpu frames");
                };
                got = Some(g);
            }
        }
    }
    let frame = got.expect("decoder produced a frame within six input frames");
    assert_eq!(frame.width, w);
    assert_eq!(frame.height, h);
    let GpuFrameSource::DmaBuf(dmabuf) = frame.source else {
        panic!("expected DmaBuf source on Linux VAAPI");
    };
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

/// Round-trip a frame through SW encode → VAAPI decode (which
/// exports a DMA-BUF) → VAAPI encode via the DMA-BUF import path.
/// Validates the riskiest unknown: that ffmpeg accepts a DRM_PRIME
/// AVFrame mapped into the encoder's hwframes pool and emits a valid
/// H.264 packet without a CPU upload.
///
/// We piggyback on the decoder to produce the DMA-BUF (which is itself
/// covered by `vaapi_decoder_smoke`) rather than synthesising one via
/// vaCreateSurfaces — both sources hit the same import code path
/// inside ffmpeg's hwcontext_vaapi.
#[test]
#[ignore = "requires a working VAAPI device (run on hardware with: cargo test -p tether-codec --ignored vaapi_encoder_dmabuf_import)"]
fn vaapi_encoder_dmabuf_import() {
    // Same dimensions on the SW encoder, the VAAPI decoder (implicit
    // — picks them up from the SPS), and the VAAPI encoder built
    // below. They must match because the encoder's hw_frames_ctx
    // pins width/height, and av_hwframe_map needs the dst pool to
    // accept the src dimensions.
    let w = 320;
    let h = 240;

    let mut sw_enc = H264Encoder::new_bgra(w, h, 30, 2_000).expect("sw encoder");
    let mut dec = VaapiDecoder::new(tether_protocol::control::CodecKind::H264).expect("VAAPI decoder");
    let mut hw_enc = VaapiEncoder::new(tether_protocol::control::VideoProfile::H264_8BIT_420, w, h, 30, 4_000).expect("VAAPI encoder");

    // Pump frames through SW encode → VAAPI decode until we get a
    // GpuFrame holding a DMA-BUF. Six frames is the same budget
    // vaapi_decoder_smoke uses (decoder usually emits on frame 0
    // for the I-frame, but allow latency for swscale warmup).
    let mut gpu_frame: Option<GpuFrame> = None;
    for t in 0..6i64 {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let bgra = make_test_bgra(w, h, t as u32);
        let packets = sw_enc.encode_bgra(&bgra, t, t == 0).expect("sw encode");
        for p in packets {
            dec.submit(&p.data).expect("vaapi submit");
            while let Some(f) = dec.next_frame().expect("vaapi next_frame") {
                let Frame::Gpu(g) = f else {
                    panic!("VaapiDecoder must emit Gpu frames");
                };
                gpu_frame = Some(g);
            }
        }
        if gpu_frame.is_some() {
            break;
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
#[ignore = "requires VAAPI; one-shot probe check"]
fn supported_profiles_smoke() {
    // Diagnostic dump of the per-profile capability matrix on this
    // box. Useful when investigating "why isn't HEVC 4:4:4 lighting
    // up?" — the probe runs a real encode + decode round trip per
    // profile and the output here shows which half failed.
    for cap in crate::supported_profiles() {
        println!(
            "{:?} encode={} decode={}",
            cap.profile, cap.encode, cap.decode
        );
    }
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
    // probe_encoder_kind uses). Small enough for the test to be
    // cheap, large enough that the encoder doesn't reject dims.
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

#[test]
#[ignore = "requires VAAPI device; exercises supported_encode_profiles"]
fn supported_encode_profiles_includes_baseline() {
    // Floor contract: on any working VAAPI box, the H.264 4:2:0
    // profile must be in the cache. Without it, no client could
    // negotiate at all. The probe result is OnceLock-cached, so
    // running this after other VAAPI tests in the same process
    // returns the same list — the determinism is the point.
    let profiles = crate::supported_encode_profiles();
    println!("supported encode profiles: {profiles:?}");
    assert!(
        profiles.contains(&tether_protocol::control::VideoProfile::H264_8BIT_420),
        "H.264 4:2:0 8-bit must be available on any VAAPI driver this build supports"
    );
}
