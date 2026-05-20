use crate::h264::H264Encoder;
use crate::{Decoder, Encoder, Frame, GpuFrame, GpuFrameSource};

use super::{VaapiDecoder, VaapiEncoder};

#[test]
#[ignore = "requires a working VAAPI device (run on hardware with: cargo test -p tether-codec --ignored vaapi)"]
fn vaapi_encoder_smoke() {
    let w = 640;
    let h = 480;
    let mut enc = VaapiEncoder::new(tether_protocol::control::CodecKind::H264, w, h, 30, 4_000).expect("VAAPI encoder");
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
    let mut hw_enc = VaapiEncoder::new(tether_protocol::control::CodecKind::H264, w, h, 30, 4_000).expect("VAAPI encoder");

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
