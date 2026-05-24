//! Hardware tests for the D3D11 encoder and decoder. These require a
//! Windows system with a GPU that supports D3D11VA encode/decode (any
//! modern Intel/AMD/NVIDIA discrete or integrated GPU). Run with
//! `cargo test -p tether-codec --lib d3d11::tests -- --ignored`.

#[cfg(test)]
mod tests {
    use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

    use crate::d3d11::decoder::D3D11Decoder;
    use crate::d3d11::encoder::D3D11Encoder;
    use crate::{Decoder, Encoder, Frame};

    const TEST_WIDTH: u32 = 1280;
    const TEST_HEIGHT: u32 = 720;
    const TEST_FPS: u32 = 30;
    const TEST_BITRATE_KBPS: u32 = 4000;

    fn h264_profile() -> VideoProfile {
        VideoProfile {
            codec: CodecKind::H264,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 8,
        }
    }

    fn hevc_profile() -> VideoProfile {
        VideoProfile {
            codec: CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 8,
        }
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn d3d11_encoder_constructs_h264() {
        let enc = D3D11Encoder::new(
            h264_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        assert!(enc.is_ok(), "H.264 encoder construction failed: {:?}", enc.err());
        assert!(enc.unwrap().is_hardware());
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn d3d11_encoder_constructs_hevc() {
        let enc = D3D11Encoder::new(
            hevc_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        assert!(enc.is_ok(), "HEVC encoder construction failed: {:?}", enc.err());
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn d3d11_decoder_constructs_h264() {
        let dec = D3D11Decoder::new(CodecKind::H264);
        assert!(dec.is_ok(), "H.264 decoder construction failed: {:?}", dec.err());
        assert!(dec.unwrap().is_hardware());
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn d3d11_decoder_constructs_hevc() {
        let dec = D3D11Decoder::new(CodecKind::Hevc);
        assert!(dec.is_ok(), "HEVC decoder construction failed: {:?}", dec.err());
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn d3d11_h264_encode_decode_roundtrip() {
        let mut enc = D3D11Encoder::new(
            h264_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .expect("encoder construction");

        let mut dec = D3D11Decoder::new(CodecKind::H264).expect("decoder construction");

        let bgra = vec![128u8; (TEST_WIDTH * TEST_HEIGHT * 4) as usize];

        // Media Foundation encoders have an async pipeline — they may
        // need several input frames before emitting the first packet.
        // Feed frames until we get encoded output.
        let mut all_packets = Vec::new();
        for pts in 0..30 {
            let force_kf = pts == 0;
            let pkts = enc
                .encode_bgra(&bgra, pts, force_kf)
                .expect("encode_bgra failed");
            all_packets.extend(pkts);
            if !all_packets.is_empty() {
                break;
            }
        }
        assert!(!all_packets.is_empty(), "encoder produced no packets after 30 frames");

        // Submit all encoded packets to decoder.
        for pkt in &all_packets {
            dec.submit(&pkt.data).expect("submit failed");
        }

        // Pull decoded frames — may need more input for decoder warm-up.
        let mut decoded_frame = None;
        for pts in 30..60 {
            if let Some(frame) = dec.next_frame().expect("next_frame") {
                decoded_frame = Some(frame);
                break;
            }
            let pkts = enc.encode_bgra(&bgra, pts, false).expect("encode_bgra");
            for pkt in &pkts {
                dec.submit(&pkt.data).expect("submit");
            }
        }

        let frame = decoded_frame.expect("decoder never produced a frame");
        match frame {
            Frame::Cpu(f) => {
                assert_eq!(f.width, TEST_WIDTH);
                assert_eq!(f.height, TEST_HEIGHT);
                assert!(!f.y.is_empty());
                assert!(!f.uv.is_empty());
            }
            Frame::Gpu(_) => {
                panic!("expected CPU frame from D3D11 Phase 1 decoder");
            }
        }
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn d3d11_hevc_encode_decode_roundtrip() {
        let mut enc = D3D11Encoder::new(
            hevc_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .expect("encoder construction");

        let mut dec = D3D11Decoder::new(CodecKind::Hevc).expect("decoder construction");

        let bgra = vec![64u8; (TEST_WIDTH * TEST_HEIGHT * 4) as usize];

        let mut all_packets = Vec::new();
        for pts in 0..30 {
            let force_kf = pts == 0;
            let pkts = enc
                .encode_bgra(&bgra, pts, force_kf)
                .expect("encode_bgra failed");
            all_packets.extend(pkts);
            if !all_packets.is_empty() {
                break;
            }
        }
        assert!(!all_packets.is_empty(), "encoder produced no packets after 30 frames");

        for pkt in &all_packets {
            dec.submit(&pkt.data).expect("submit failed");
        }

        let mut decoded_frame = None;
        for pts in 30..60 {
            if let Some(frame) = dec.next_frame().expect("next_frame") {
                decoded_frame = Some(frame);
                break;
            }
            let pkts = enc.encode_bgra(&bgra, pts, false).expect("encode_bgra");
            for pkt in &pkts {
                dec.submit(&pkt.data).expect("submit");
            }
        }

        let frame = decoded_frame.expect("decoder never produced a frame");
        match frame {
            Frame::Cpu(f) => {
                assert_eq!(f.width, TEST_WIDTH);
                assert_eq!(f.height, TEST_HEIGHT);
                assert!(!f.y.is_empty());
                assert!(!f.uv.is_empty());
            }
            Frame::Gpu(_) => panic!("expected CPU frame"),
        }
    }
}
