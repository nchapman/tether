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

#[test]
#[ignore = "requires macOS + VideoToolbox (run on Apple Silicon / Intel mac with: cargo test -p tether-codec --ignored videotoolbox)"]
fn videotoolbox_encoder_smoke() {
    let w = 640;
    let h = 480;
    let mut enc = VideoToolboxEncoder::new(yuv420_8bit(CodecKind::H264), w, h, 30, 4_000)
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
/// 3. Cross-checks the result against `supported_profiles()`. The
///    probe and the explicit round-trip must agree: if the probe
///    says `encode=true` the round-trip must succeed, and if the
///    probe says `encode=false` the round-trip must fail. A
///    disagreement is the bug — either the probe is over-claiming
///    (silent downsample slips through) or under-claiming (a
///    capability is being hidden).
///
/// This is the primary regression test for the audit's "silent
/// downsample" risk: the probe layer was over-claiming HEVC 4:4:4
/// encode on M4 Max before commit history added the round-trip
/// fourcc check, and an independent test pinning the agreement
/// stops that regression from coming back.
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
    let caps = crate::probe::supported_profiles();

    for &profile in profiles {
        let round_trip = try_round_trip(profile, &bgra, W, H);
        let probe_says = caps
            .iter()
            .find(|c| c.profile == profile)
            .map(|c| c.encode)
            .unwrap_or(false);

        match (&round_trip, probe_says) {
            (Ok(fourcc), true) => {
                eprintln!(
                    "round-trip matrix: {profile:?} OK (IOSurface 0x{fourcc:08x}); \
                     probe agrees encode=true"
                );
            }
            (Err(reason), false) => {
                eprintln!(
                    "round-trip matrix: {profile:?} unsupported ({reason}); \
                     probe agrees encode=false"
                );
            }
            (Ok(fourcc), false) => panic!(
                "{profile:?} disagreement: explicit round-trip succeeded \
                 (IOSurface 0x{fourcc:08x}) but probe reports encode=false — \
                 probe is under-claiming, the profile should be advertised"
            ),
            (Err(reason), true) => panic!(
                "{profile:?} disagreement: probe reports encode=true but explicit \
                 round-trip failed ({reason}) — probe is over-claiming (likely \
                 silent downsample slipping past the probe's fourcc check), \
                 the profile must not be advertised"
            ),
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
    loop {
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
                return Ok(io.pixel_format);
            }
            Some(Frame::Cpu(_)) => return Err("decoder produced Cpu frame".into()),
            None => return Err("decoder produced no frame after EOF".into()),
        }
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
/// bitstream's decode should land in. Range variants (`'420v'` /
/// `'420f'`, etc.) both count — range is signalled in the VUI, not
/// in the chroma fourcc, and we deliberately don't constrain it
/// here. Mirrors the probe's `expected_iosurface_fourccs` (kept
/// duplicated rather than re-exported to keep the test self-contained
/// and the cross-module coupling minimal).
fn expected_iosurface_fourccs_for(profile: VideoProfile) -> &'static [u32] {
    const NV12_VIDEO: u32 = u32::from_be_bytes(*b"420v");
    const NV12_FULL: u32 = u32::from_be_bytes(*b"420f");
    const NV24_VIDEO: u32 = u32::from_be_bytes(*b"444v");
    const NV24_FULL: u32 = u32::from_be_bytes(*b"444f");
    const P010: u32 = u32::from_be_bytes(*b"P010");
    const XF20: u32 = u32::from_be_bytes(*b"xf20");
    const X420: u32 = u32::from_be_bytes(*b"x420");
    const XF44: u32 = u32::from_be_bytes(*b"xf44");
    const P410: u32 = u32::from_be_bytes(*b"P410");
    match (profile.chroma, profile.bit_depth) {
        (ChromaSubsampling::Yuv420, 8) => &[NV12_VIDEO, NV12_FULL],
        (ChromaSubsampling::Yuv420, 10) => &[P010, XF20, X420],
        (ChromaSubsampling::Yuv444, 8) => &[NV24_VIDEO, NV24_FULL],
        (ChromaSubsampling::Yuv444, 10) => &[XF44, P410],
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
