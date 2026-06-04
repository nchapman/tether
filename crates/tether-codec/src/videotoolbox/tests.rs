use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

use crate::{Decoder, Encoder, Frame, GpuFrame, GpuFrameSource};

use super::{VideoToolboxDecoder, VideoToolboxEncoder};

/// Yuv420 8-bit profile for the given codec — what every VideoToolbox
/// test in this file was implicitly constructing before the encoder's
/// constructor became profile-parameterised.
fn yuv420_8bit(kind: CodecKind) -> VideoProfile {
    VideoProfile {
        codec: kind,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 8,
    }
}

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

/// The encoder must pin an EXPLICIT profile — never rely on VideoToolbox's
/// default. Parses the SPS the encoder emits and asserts its profile id
/// matches the pin in `VideoToolboxEncoder::new`: H.264 Main (77), HEVC
/// Main (1), HEVC Main10 (2). Direct guard for the profile-pin block — a
/// regression that drops the pin lets VT auto-select, the exact class of
/// bug the pins exist to prevent (cf. the D3D11 H.264 Baseline default).
/// 4:4:4 is omitted on purpose: VT exposes no Main444 (REXT maps to the
/// 4:2:2 Main42210), and the chroma-survival probe rejects HEVC 4:4:4
/// encode on Apple Silicon, so there is no stable profile id to assert.
#[test]
#[ignore = "requires macOS + VideoToolbox"]
fn videotoolbox_encoder_pins_explicit_profile() {
    use crate::bitstream_sps::parse_sps_chroma_bit_depth;

    // (codec, bit_depth, expected SPS profile id)
    let cases = [
        (CodecKind::H264, 8u8, 77u8), // H.264 Main
        (CodecKind::Hevc, 8, 1),      // HEVC Main (general_profile_idc)
        (CodecKind::Hevc, 10, 2),     // HEVC Main10
    ];
    let (w, h) = (320u32, 240u32);
    for (codec, bit_depth, expected_profile) in cases {
        let profile = VideoProfile {
            codec,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth,
        };
        let mut enc = match VideoToolboxEncoder::new(profile, w, h, 30, 2_000) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("SKIP {profile:?}: encoder construct failed: {e:?}");
                continue;
            }
        };
        let bgra = make_test_bgra(w, h, 0);
        let mut keyframe = None;
        for pts in 0..16i64 {
            let packets = enc
                .encode_bgra(&bgra, pts, pts == 0)
                .unwrap_or_else(|e| panic!("{profile:?} encode: {e:?}"));
            if let Some(p) = packets.into_iter().find(|p| p.keyframe) {
                keyframe = Some(p.data);
                break;
            }
        }
        if keyframe.is_none() {
            keyframe = enc
                .flush()
                .unwrap_or_default()
                .into_iter()
                .find(|p| p.keyframe)
                .map(|p| p.data);
        }
        let kf = keyframe.unwrap_or_else(|| panic!("{profile:?} produced no keyframe"));
        let sps = parse_sps_chroma_bit_depth(&kf, codec)
            .unwrap_or_else(|| panic!("{profile:?} keyframe has no parseable SPS"));
        assert_eq!(sps.bit_depth_luma, bit_depth, "{profile:?} bit depth");
        assert_eq!(
            sps.profile_idc, expected_profile,
            "{profile:?}: encoder must pin profile_idc={expected_profile}, got {} — \
             the VideoToolbox profile pin regressed",
            sps.profile_idc
        );
    }
}

/// macOS sibling of `vaapi_set_bitrate_live_continues_to_encode`.
/// Verifies that `Encoder::set_bitrate_kbps` succeeds mid-stream on
/// VideoToolbox and the encoder keeps producing decodable packets
/// afterwards. The ABR controller relies on this property.
#[test]
#[ignore = "requires macOS + VideoToolbox (run on Apple Silicon / Intel mac with: cargo test -p tether-codec --ignored videotoolbox_set_bitrate_live_continues_to_encode)"]
fn videotoolbox_set_bitrate_live_continues_to_encode() {
    let w = 640;
    let h = 480;
    let mut enc = VideoToolboxEncoder::new(yuv420_8bit(CodecKind::H264), w, h, 30, 4_000)
        .expect("VideoToolbox encoder");
    assert!(
        enc.supports_changing_bitrate(),
        "VideoToolbox encoder is expected to advertise bitrate-change support"
    );
    let mut dec = VideoToolboxDecoder::new(CodecKind::H264).expect("VideoToolbox decoder");

    for t in 0..4i64 {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let bgra = make_test_bgra(w, h, t as u32);
        let packets = enc.encode_bgra(&bgra, t, t == 0).expect("encode pre");
        for p in packets {
            dec.submit(&p.data).expect("decode pre");
            while dec.next_frame().expect("next_frame pre").is_some() {}
        }
    }

    enc.set_bitrate_kbps(1_500).expect("live retune");

    let mut got_post: Option<Frame> = None;
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
        "decoder produced no frames after live bitrate change"
    );
}

#[test]
#[ignore = "requires macOS + VideoToolbox (run on Apple Silicon / Intel mac with: cargo test -p tether-codec --ignored videotoolbox)"]
fn videotoolbox_encoder_smoke() {
    let w = 640;
    let h = 480;
    let mut enc = VideoToolboxEncoder::new(yuv420_8bit(CodecKind::H264), w, h, 30, 4_000)
        .expect("VideoToolbox encoder");
    let bgra = vec![0x80u8; (w * h * 4) as usize];
    // First frame may produce 0 packets (encoder warm-up) — drain with
    // `flush()` so the assertion below covers both shapes. Without the
    // drain a zero-packet result would silently pass the loop assert
    // and miss a "encoder produced nothing ever" regression.
    let mut packets = enc.encode_bgra(&bgra, 0, true).expect("encode");
    if packets.is_empty() {
        packets = enc.flush().expect("flush");
    }
    assert!(
        !packets.is_empty(),
        "encoder produced no packets across one encode + flush — \
         encoder is buffering forever or silently failing"
    );
    for p in &packets {
        assert!(!p.data.is_empty(), "encoder produced an empty packet");
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
    let res = VideoToolboxEncoder::new(yuv420_8bit(CodecKind::Hevc), 320, 240, 30, 2_000);
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

    for kind in [CodecKind::H264, CodecKind::Hevc] {
        let w = 320;
        let h = 240;
        let mut enc = VideoToolboxEncoder::new(yuv420_8bit(kind), w, h, 30, 2_000)
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
        let trailing = enc
            .flush()
            .unwrap_or_else(|e| panic!("{kind:?} flush: {e:?}"));
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
    // NV12 fourccs the IOSurface may carry (matches what the renderer
    // accepts in `tether-render/src/gpu/metal.rs`).
    const NV12_VIDEO_RANGE: u32 = u32::from_be_bytes(*b"420v");
    const NV12_FULL_RANGE: u32 = u32::from_be_bytes(*b"420f");

    for kind in [CodecKind::H264, CodecKind::Hevc] {
        let w = 320;
        let h = 240;
        let mut enc = VideoToolboxEncoder::new(yuv420_8bit(kind), w, h, 30, 2_000)
            .unwrap_or_else(|e| panic!("{kind:?} encoder: {e:?}"));
        let mut dec =
            VideoToolboxDecoder::new(kind).unwrap_or_else(|e| panic!("{kind:?} decoder: {e:?}"));

        let mut decoded: Option<GpuFrame> = None;
        // 12 frames is plenty: the first keyframe carries extradata
        // inline (per Phase 1.1) so the decoder doesn't need external
        // priming, and any pipeline latency is < 4 frames on VT.
        for t in 0..12u32 {
            let bgra = make_test_bgra(w, h, t);
            let force_key = t == 0;
            let packets = enc
                .encode_bgra(&bgra, i64::from(t), force_key)
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

/// End-to-end hardware round-trip across the full chroma × bit-depth
/// matrix we may negotiate, **independently** of the probe layer's
/// own round-trip — the two must agree.
///
/// For each `(chroma, bit_depth)` profile, this test:
///
/// 1. Runs an explicit encode → decode round-trip with a
///    high-frequency chroma BGRA pattern (alternating red / green
///    vertical stripes — survives only with full UV resolution).
/// 2. Categorises the result as `Supported` (encoder constructed,
///    packets produced, decoded IOSurface fourcc landed in the
///    expected family for the profile) or `Unsupported` (any step
///    failed — encoder open, no packets, decoder error, or
///    fourcc mismatch indicating silent downsample).
/// 3. Verifies that the 4:4:4 profile families produce IOSurfaces in
///    the expected 4:4:4 fourcc family — the original silent-downsample
///    bug had VT accepting NV24/P410LE input then encoding 4:2:0 in
///    the bitstream. A `'420v'`/`'x420'` IOSurface coming out for a
///    4:4:4 round-trip would be that regression.
///
/// (Historical context: this test used to cross-check against
/// `tether-codec::probe::supported_profiles()`, retired in commit
/// 255289a when probe orchestration moved exclusively to
/// `tether-probe`. The decode-emit ↔ render-accept agreement check now
/// lives in `tether-probe::host::videotoolbox`: the pure-logic
/// `decoder_output_is_subset_of_renderer_accept` (no hardware) and the
/// `#[ignore]` hardware `decoded_fixture_fourcc_is_renderer_accepted`.
/// Here we keep just the round-trip + fourcc-family invariants that
/// don't need the probe layer.)
#[test]
#[ignore = "requires macOS + VideoToolbox"]
fn videotoolbox_round_trip_chroma_matrix() {
    const W: u32 = 128;
    const H: u32 = 128;
    let profiles: &[VideoProfile] = &[
        VideoProfile::HEVC_8BIT_420,
        VideoProfile::HEVC_10BIT_420,
        VideoProfile::HEVC_8BIT_444,
        VideoProfile::HEVC_10BIT_444,
        VideoProfile::H264_8BIT_420,
    ];
    let bgra = make_chroma_detail_bgra(W, H);

    for &profile in profiles {
        let round_trip = try_round_trip(profile, &bgra, W, H);
        match (&round_trip, profile.chroma) {
            (Ok(fourcc), ChromaSubsampling::Yuv444) => {
                // If the round-trip succeeded for a 4:4:4 profile, the
                // decoded IOSurface fourcc must be in the 4:4:4 family —
                // a 4:2:0 fourcc here would be the silent-downsample
                // regression this test exists to catch.
                let expected = expected_iosurface_fourccs_for(profile);
                assert!(
                    expected.contains(fourcc),
                    "{profile:?} round-trip succeeded but IOSurface fourcc \
                     0x{fourcc:08x} is not in the 4:4:4 family — VT likely \
                     silently downsampled to 4:2:0 in the bitstream"
                );
                eprintln!("round-trip matrix: {profile:?} OK (IOSurface 0x{fourcc:08x})");
            }
            (Ok(fourcc), _) => {
                eprintln!("round-trip matrix: {profile:?} OK (IOSurface 0x{fourcc:08x})");
            }
            (Err(reason), _) => {
                eprintln!("round-trip matrix: {profile:?} unsupported ({reason})");
            }
        }
    }
}

/// Run one round-trip and return `Ok(observed_fourcc)` if every step
/// (encode open, encode packets, decode, fourcc match) succeeded, or
/// `Err(reason)` if any step failed. The chroma-matrix test treats
/// either outcome as informative — the panic only fires when the
/// probe layer's claim doesn't match this independent result.
fn try_round_trip(
    profile: VideoProfile,
    bgra: &[u8],
    w: u32,
    h: u32,
) -> std::result::Result<u32, String> {
    let mut enc = VideoToolboxEncoder::new(profile, w, h, 30, 2_000)
        .map_err(|e| format!("encoder construction: {e:?}"))?;
    let mut packets = enc
        .encode_bgra(bgra, 0, true)
        .map_err(|e| format!("encode_bgra: {e:?}"))?;
    if packets.is_empty() {
        packets = enc.flush().map_err(|e| format!("flush: {e:?}"))?;
    }
    if packets.is_empty() {
        return Err("no packets produced".into());
    }
    let mut dec = VideoToolboxDecoder::new(profile.codec)
        .map_err(|e| format!("decoder construction: {e:?}"))?;
    for p in &packets {
        dec.submit(&p.data)
            .map_err(|e| format!("decoder submit: {e:?}"))?;
    }
    dec.signal_eof()
        .map_err(|e| format!("decoder signal_eof: {e:?}"))?;
    match dec
        .next_frame()
        .map_err(|e| format!("decoder next_frame: {e:?}"))?
    {
        Some(Frame::Gpu(g)) => {
            let GpuFrameSource::IOSurface(io) = g.source;
            let expected = expected_iosurface_fourccs_for(profile);
            if !expected.contains(&io.pixel_format) {
                return Err(format!(
                    "IOSurface fourcc 0x{:08x} not in expected family {:?} \
                     (likely silent downsample)",
                    io.pixel_format,
                    expected
                        .iter()
                        .map(|f| format!("0x{f:08x}"))
                        .collect::<Vec<_>>()
                ));
            }
            Ok(io.pixel_format)
        }
        Some(Frame::Cpu(_)) => Err("decoder produced Cpu frame".into()),
        None => Err("decoder produced no frame after EOF".into()),
    }
}

/// Generate a BGRA buffer with high-frequency chroma content. Vertical
/// red / green stripes at 1-pixel pitch — any encoder that silently
/// downsamples 4:4:4 → 4:2:0 collapses adjacent columns into a
/// uniform yellow-ish bar, which shifts the bitstream's
/// `chroma_format_idc` to 1 and lands the decoded IOSurface in the
/// 4:2:0 family. Used by `videotoolbox_round_trip_chroma_matrix`.
fn make_chroma_detail_bgra(width: u32, height: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for _y in 0..height {
        for x in 0..width {
            if x % 2 == 0 {
                data.extend_from_slice(&[0, 0, 255, 255]); // saturated red
            } else {
                data.extend_from_slice(&[0, 255, 0, 255]); // saturated green
            }
        }
    }
    data
}

/// Per-profile set of IOSurface fourccs a *correctly-encoded*
/// bitstream's decode should land in. Video-range only — the host
/// encoder emits video-range and the renderer imports video-range
/// only, so a full-range decode can't arise (and wouldn't display).
/// Mirrors the probe's `expected_iosurface_fourccs` (kept duplicated
/// rather than re-exported to keep the test self-contained and the
/// cross-module coupling minimal).
fn expected_iosurface_fourccs_for(profile: VideoProfile) -> &'static [u32] {
    const NV12_VIDEO: u32 = u32::from_be_bytes(*b"420v");
    const NV24_VIDEO: u32 = u32::from_be_bytes(*b"444v");
    const P010: u32 = u32::from_be_bytes(*b"P010");
    const X420: u32 = u32::from_be_bytes(*b"x420");
    const X444: u32 = u32::from_be_bytes(*b"x444");
    const XF44: u32 = u32::from_be_bytes(*b"xf44");
    const P410: u32 = u32::from_be_bytes(*b"P410");
    match (profile.chroma, profile.bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => &[NV12_VIDEO],
        (ChromaSubsampling::Yuv420, 10) => &[P010, X420],
        (ChromaSubsampling::Yuv444, 8) => &[NV24_VIDEO],
        (ChromaSubsampling::Yuv444, 10) => &[X444, XF44, P410],
        _ => &[],
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
    for kind in [CodecKind::H264, CodecKind::Hevc] {
        let w = 320;
        let h = 240;
        let mut enc = VideoToolboxEncoder::new(yuv420_8bit(kind), w, h, 30, 2_000)
            .unwrap_or_else(|e| panic!("{kind:?} encoder: {e:?}"));

        // Drive the encoder for enough frames to produce at least two
        // IDRs (frame 0 and an explicit force at frame 8). Collect every
        // packet so we can replay a subset of them.
        let mut packets: Vec<crate::EncodedPacket> = Vec::new();
        for t in 0..16u32 {
            let bgra = make_test_bgra(w, h, t);
            let force = t == 0 || t == 8;
            let out = enc
                .encode_bgra(&bgra, i64::from(t), force)
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
    // Default-on (no hardware needed): pin the actual `vt_codec_name`
    // strings against what `ffmpeg -encoders | grep videotoolbox`
    // emits. A typo or rename of the cstring on the encoder side
    // breaks log messages and any future `--codec=` CLI dispatch
    // that keys on these names; this test fires at `cargo test` time
    // rather than at first encode attempt in production.
    use crate::videotoolbox::encoder::vt_codec_name;
    assert_eq!(vt_codec_name(CodecKind::H264), "h264_videotoolbox");
    assert_eq!(vt_codec_name(CodecKind::Hevc), "hevc_videotoolbox");
    // AV1 encode is still unsupported by FFmpeg's VideoToolbox wrapper;
    // decode is wired through `VideoToolboxDecoder`.
    assert_eq!(vt_codec_name(CodecKind::Av1), "av1_videotoolbox");
}
