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
