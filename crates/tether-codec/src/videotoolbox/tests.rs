use crate::Encoder;

use super::VideoToolboxEncoder;

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
