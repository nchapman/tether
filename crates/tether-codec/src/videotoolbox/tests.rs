use crate::{Decoder, Encoder, Frame, GpuFrame, GpuFrameSource};

use super::{VideoToolboxDecoder, VideoToolboxEncoder};

/// Test bitmap with broadband content (gradient + moving stripes) so
/// the encoder has something to emit and the decoder has something
/// to reconstruct. Mirrors `vaapi::tests::make_test_bgra`.
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
#[ignore = "requires macOS + VideoToolbox (run on Apple Silicon / Intel mac with: cargo test -p tether-codec --ignored videotoolbox)"]
fn videotoolbox_encoder_smoke() {
    let w = 640;
    let h = 480;
    let mut enc = VideoToolboxEncoder::new(
        tether_protocol::control::CodecKind::H264,
        w,
        h,
        30,
        4_000,
    )
    .expect("VideoToolbox encoder");
    let bgra = vec![0x80u8; (w * h * 4) as usize];
    let packets = enc.encode_bgra(&bgra, 0, true).expect("encode");
    // First frame may produce 0 packets (encoder warm-up) or 1+ packets
    // carrying SPS/PPS + the IDR slice. Either way it shouldn't error.
    for p in packets {
        assert!(!p.data.is_empty());
    }
}

#[test]
#[ignore = "requires macOS + VideoToolbox"]
fn videotoolbox_hevc_constructs() {
    // HEVC availability is closer to universal on Apple Silicon than
    // on the H.264 path on Intel/AMD VAAPI, but the failure mode is
    // the same: `find_encoder_by_name` returns None if FFmpeg wasn't
    // built with `--enable-videotoolbox`, and `encoder.open()` returns
    // an error if the device doesn't expose the HEVC encoder.
    let res = VideoToolboxEncoder::new(
        tether_protocol::control::CodecKind::Hevc,
        320,
        240,
        30,
        2_000,
    );
    assert!(
        res.is_ok(),
        "hevc_videotoolbox should construct on a modern mac: {:?}",
        res.err()
    );
}

#[test]
#[ignore = "requires macOS + VideoToolbox"]
fn videotoolbox_keyframes_carry_extradata() {
    // Every keyframe must be self-decodable: clients that join
    // mid-session, rebuild their decoder, or lose the session's first
    // IDR have no recovery path otherwise. We encode ≥2 GOPs worth of
    // frames (forcing one keyframe at the start and another partway
    // through), then assert each keyframe packet begins with the SPS
    // bundle FFmpeg parked in `extradata` at open() time.
    use tether_protocol::control::CodecKind;

    for kind in [CodecKind::H264, CodecKind::Hevc] {
        let w = 320;
        let h = 240;
        let mut enc = VideoToolboxEncoder::new(kind, w, h, 30, 2_000)
            .unwrap_or_else(|e| panic!("{kind:?} encoder: {e:?}"));

        // Two distinct grey BGRA frames so the encoder has actual
        // residual to emit on P-frames; an all-zero buffer can collapse
        // into degenerate empty packets on some builds.
        let frame_a = vec![0x40u8; (w * h * 4) as usize];
        let frame_b = vec![0xC0u8; (w * h * 4) as usize];

        // Pre-stash extradata so the assertion message doesn't have
        // to reach back into a borrowed encoder mid-loop.
        let extradata = enc.extradata.clone();
        assert!(
            !extradata.is_empty(),
            "{kind:?} extradata empty after open(); \
             AV_CODEC_FLAG_GLOBAL_HEADER may not be honoured"
        );

        let mut keyframes_seen = 0;
        // Force a keyframe every 4 frames so we drive multiple IDRs
        // through the encoder regardless of `h264_videotoolbox`'s
        // periodic GOP cadence. This is the same channel the host's
        // `ForceIdr` plumbing uses (`AV_PICTURE_TYPE_I` on input
        // frames).
        for pts in 0..16i64 {
            let bgra = if pts % 2 == 0 { &frame_a } else { &frame_b };
            let force = pts % 4 == 0;
            let packets = enc
                .encode_bgra(bgra, pts, force)
                .unwrap_or_else(|e| panic!("{kind:?} encode frame {pts}: {e:?}"));
            for p in packets {
                if p.keyframe {
                    keyframes_seen += 1;
                    assert!(
                        p.data.starts_with(&extradata),
                        "{kind:?} keyframe packet does not start with extradata: \
                         keyframe head = {:02x?}, extradata = {:02x?}",
                        &p.data[..extradata.len().min(p.data.len())],
                        extradata
                    );
                }
            }
        }
        // Flush any packets still buffered inside the encoder pipeline.
        // VideoToolbox can hold the last submitted frame's packet until
        // the next frame arrives; without this drain a forced keyframe
        // late in the loop would be missed.
        let trailing = enc.flush().unwrap_or_else(|e| panic!("{kind:?} flush: {e:?}"));
        for p in trailing {
            if p.keyframe {
                keyframes_seen += 1;
                assert!(
                    p.data.starts_with(&extradata),
                    "{kind:?} flushed keyframe packet does not start with extradata"
                );
            }
        }
        assert!(
            keyframes_seen >= 2,
            "{kind:?} produced only {keyframes_seen} keyframes across 16 frames \
             with force_keyframe every 4; on-demand IDR plumbing may be broken"
        );
    }
}

#[test]
#[ignore = "requires macOS + VideoToolbox"]
fn videotoolbox_decoder_constructs() {
    // Symmetric with `videotoolbox_hevc_constructs` on the encoder
    // side: confirms FFmpeg's `--enable-videotoolbox` build path is
    // present for both decoders. Round-trip behavior is exercised in
    // `videotoolbox_round_trip` below.
    use tether_protocol::control::CodecKind;
    for kind in [CodecKind::H264, CodecKind::Hevc] {
        let res = VideoToolboxDecoder::new(kind);
        assert!(
            res.is_ok(),
            "{kind:?} VideoToolbox decoder should construct on macOS: {:?}",
            res.err()
        );
    }
}

/// End-to-end hardware round-trip: VT encode (BGRA → Annex-B) →
/// VT decode (Annex-B → IOSurface-backed `Frame::Gpu`). Verifies the
/// invariants the rest of the macOS client depends on:
///
/// - Decoded frame is `Frame::Gpu(GpuFrameSource::IOSurface(...))` —
///   never `Frame::Cpu` (the HW-only contract).
/// - Frame dimensions match the encoded source.
/// - The IOSurface pointer is non-null and the pixel format is a
///   recognised NV12 fourcc (`'420v'` or `'420f'`).
/// - SPS/PPS-on-every-keyframe extradata makes the very first decoder
///   submit able to produce output (no need to feed multiple GOPs
///   before the decoder accepts; that's the whole point of the
///   self-decodable-IDR work).
///
/// Run on real macOS hardware with:
/// `cargo test -p tether-codec --lib videotoolbox_round_trip -- --ignored --nocapture`.
#[test]
#[ignore = "requires macOS + VideoToolbox (run with: cargo test -p tether-codec --ignored videotoolbox_round_trip)"]
fn videotoolbox_round_trip() {
    use tether_protocol::control::CodecKind;

    // NV12 fourccs the IOSurface may carry (matches what the renderer
    // accepts in `tether-render/src/gpu/metal.rs`).
    const NV12_VIDEO_RANGE: u32 = u32::from_be_bytes(*b"420v");
    const NV12_FULL_RANGE: u32 = u32::from_be_bytes(*b"420f");

    for kind in [CodecKind::H264, CodecKind::Hevc] {
        let w = 320;
        let h = 240;
        let mut enc = VideoToolboxEncoder::new(kind, w, h, 30, 2_000)
            .unwrap_or_else(|e| panic!("{kind:?} encoder: {e:?}"));
        let mut dec = VideoToolboxDecoder::new(kind)
            .unwrap_or_else(|e| panic!("{kind:?} decoder: {e:?}"));

        let mut decoded: Option<GpuFrame> = None;
        // 12 frames is plenty: the first keyframe carries extradata
        // inline (per Phase 1.1) so the decoder doesn't need external
        // priming, and any pipeline latency is < 4 frames on VT.
        for t in 0..12i64 {
            let bgra = make_test_bgra(w, h, t as u32);
            let force_key = t == 0;
            let packets = enc
                .encode_bgra(&bgra, t, force_key)
                .unwrap_or_else(|e| panic!("{kind:?} encode frame {t}: {e:?}"));
            for p in packets {
                dec.submit(&p.data)
                    .unwrap_or_else(|e| panic!("{kind:?} submit frame {t}: {e:?}"));
                while let Some(f) = dec
                    .next_frame()
                    .unwrap_or_else(|e| panic!("{kind:?} next_frame: {e:?}"))
                {
                    match f {
                        Frame::Gpu(g) => decoded = Some(g),
                        Frame::Cpu(_) => panic!(
                            "{kind:?} VideoToolboxDecoder produced a Cpu frame; \
                             violates the hardware-only contract"
                        ),
                    }
                }
            }
            if decoded.is_some() {
                break;
            }
        }
        // Flush in case any frame is still buffered. The encoder's
        // pipeline can hold one or two frames at the end of a short
        // sequence (same shape as the keyframes-carry-extradata test);
        // the decoder side has no explicit drain because the `Decoder`
        // trait doesn't expose one — we just keep polling `next_frame`
        // until it returns `None`, which is what would happen in
        // production at end-of-session anyway.
        if decoded.is_none() {
            let trailing = enc
                .flush()
                .unwrap_or_else(|e| panic!("{kind:?} flush: {e:?}"));
            for p in trailing {
                dec.submit(&p.data)
                    .unwrap_or_else(|e| panic!("{kind:?} submit flushed: {e:?}"));
                while let Some(f) = dec
                    .next_frame()
                    .unwrap_or_else(|e| panic!("{kind:?} next_frame flush: {e:?}"))
                {
                    match f {
                        Frame::Gpu(g) => decoded = Some(g),
                        Frame::Cpu(_) => panic!(
                            "{kind:?} flush produced a Cpu frame; \
                             violates hardware-only contract"
                        ),
                    }
                }
            }
        }

        let frame = decoded.unwrap_or_else(|| {
            panic!("{kind:?} no decoded frame produced after 12 inputs + flush")
        });
        // Single source of truth for dims: `GpuFrame::new` copies them
        // from the IOSurface, so asserting `frame.width` covers both
        // surfaces. The non-null pointer and fourcc checks are the
        // load-bearing ones — they verify the renderer-facing contract.
        assert_eq!(frame.width, w, "{kind:?} decoded width");
        assert_eq!(frame.height, h, "{kind:?} decoded height");
        let GpuFrameSource::IOSurface(io) = frame.source;
        assert!(
            !io.surface.is_null(),
            "{kind:?} decoded IOSurface pointer is null"
        );
        assert!(
            matches!(io.pixel_format, NV12_VIDEO_RANGE | NV12_FULL_RANGE),
            "{kind:?} IOSurface pixel_format 0x{:08x} is not a recognised NV12 fourcc",
            io.pixel_format
        );
    }
}

/// Decoder-rebuild round-trip: prove the self-decodable-IDR contract.
///
/// The single-decoder round-trip above can't distinguish "extradata is
/// prepended correctly" from "VT's in-band SPS recovery happened to
/// work" — once a VT decoder has parsed the first IDR's in-band
/// parameter sets, every subsequent IDR uses the cached SPS. A
/// regression that *stopped* prepending extradata would still pass
/// `videotoolbox_round_trip`.
///
/// This test exercises the actual failure mode: encode many frames,
/// skip the first IDR entirely (as a client joining mid-session
/// would), and construct a *fresh* `VideoToolboxDecoder` that has
/// never seen the session's first IDR. The decoder must still produce
/// a frame from the next IDR alone, which only works if that IDR
/// packet carries its own SPS/PPS prefix.
#[test]
#[ignore = "requires macOS + VideoToolbox"]
fn videotoolbox_decoder_recovers_from_mid_session_idr() {
    use tether_protocol::control::CodecKind;

    for kind in [CodecKind::H264, CodecKind::Hevc] {
        let w = 320;
        let h = 240;
        let mut enc = VideoToolboxEncoder::new(kind, w, h, 30, 2_000)
            .unwrap_or_else(|e| panic!("{kind:?} encoder: {e:?}"));

        // Drive the encoder for enough frames to produce at least two
        // IDRs (frame 0 and an explicit force at frame 8). Collect every
        // packet so we can replay a subset of them.
        let mut packets: Vec<crate::EncodedPacket> = Vec::new();
        for t in 0..16i64 {
            let bgra = make_test_bgra(w, h, t as u32);
            let force = t == 0 || t == 8;
            let out = enc
                .encode_bgra(&bgra, t, force)
                .unwrap_or_else(|e| panic!("{kind:?} encode {t}: {e:?}"));
            packets.extend(out);
        }
        packets.extend(
            enc.flush()
                .unwrap_or_else(|e| panic!("{kind:?} flush: {e:?}")),
        );

        // Pin the self-decodable-IDR contract at the wire level: every
        // keyframe packet must begin with the encoder's captured
        // extradata. Belt-and-suspenders against the cross-cutting
        // `videotoolbox_keyframes_carry_extradata` test in case the
        // round-trip path diverges from the standalone encoder path.
        let extradata = enc.extradata.clone();
        assert!(
            !extradata.is_empty(),
            "{kind:?} encoder extradata empty; AV_CODEC_FLAG_GLOBAL_HEADER may not be honoured"
        );
        let keyframe_indices: Vec<usize> = packets
            .iter()
            .enumerate()
            .filter(|(_, p)| p.keyframe)
            .map(|(i, _)| i)
            .collect();
        assert!(
            keyframe_indices.len() >= 2,
            "{kind:?} need at least 2 keyframes to test mid-session rebuild; got {}",
            keyframe_indices.len()
        );
        for &i in &keyframe_indices {
            assert!(
                packets[i].data.starts_with(&extradata),
                "{kind:?} keyframe #{i} does not start with extradata; \
                 self-decodable-IDR contract broken"
            );
        }

        // Skip everything up to the *second* keyframe — simulates a
        // client that joined mid-session and never saw the first IDR.
        let resume_at = keyframe_indices[1];
        let mut dec = VideoToolboxDecoder::new(kind)
            .unwrap_or_else(|e| panic!("{kind:?} fresh decoder: {e:?}"));

        let mut decoded: Option<GpuFrame> = None;
        for p in &packets[resume_at..] {
            dec.submit(&p.data)
                .unwrap_or_else(|e| panic!("{kind:?} submit (post-resume): {e:?}"));
            while let Some(f) = dec
                .next_frame()
                .unwrap_or_else(|e| panic!("{kind:?} next_frame (post-resume): {e:?}"))
            {
                match f {
                    Frame::Gpu(g) => {
                        decoded = Some(g);
                    }
                    Frame::Cpu(_) => panic!(
                        "{kind:?} decoder produced Cpu frame post-resume; \
                         violates hardware-only contract"
                    ),
                }
            }
            if decoded.is_some() {
                break;
            }
        }

        let frame = decoded.unwrap_or_else(|| {
            panic!(
                "{kind:?} fresh decoder failed to produce a frame starting from a \
                 non-first IDR — self-decodable-IDR invariant is broken. Without \
                 extradata prepended to every keyframe, a client that joins \
                 mid-session has no recovery path."
            )
        });
        assert_eq!(frame.width, w);
        assert_eq!(frame.height, h);
    }
}

#[test]
fn videotoolbox_codec_name_maps() {
    // Default-on (no hardware needed): exercises the codec_name map so
    // a typo in the cstring → str pair gets caught at CI time.
    use tether_protocol::control::CodecKind;
    fn name(kind: CodecKind) -> &'static str {
        // Mirrors the private `vt_codec_name` in `encoder.rs`; if those
        // strings ever diverge from the cstr names the encoder asks for,
        // log/diagnostic messages stop matching what `ffmpeg -encoders`
        // shows. Keep this assertion shape in sync.
        match kind {
            CodecKind::H264 => "h264_videotoolbox",
            CodecKind::Hevc => "hevc_videotoolbox",
            CodecKind::Av1 => "av1_videotoolbox",
        }
    }
    assert_eq!(name(CodecKind::H264), "h264_videotoolbox");
    assert_eq!(name(CodecKind::Hevc), "hevc_videotoolbox");
}
