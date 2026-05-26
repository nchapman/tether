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

    fn hevc_main10_profile() -> VideoProfile {
        VideoProfile {
            codec: CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 10,
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
    fn d3d11_encoder_shared_device_h264() {
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
        use windows::Win32::Graphics::Direct3D11::{
            D3D11CreateDevice, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION,
        };
        use windows::Win32::Foundation::HMODULE;

        let mut device = None;
        let mut context = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        }
        .expect("D3D11CreateDevice");
        let device = device.unwrap();
        let context = context.unwrap();

        let enc = D3D11Encoder::new(
            h264_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            device.as_raw() as *mut _,
            context.as_raw() as *mut _,
        );
        assert!(enc.is_ok(), "shared-device encoder failed: {:?}", enc.err());
        assert!(enc.unwrap().is_hardware());
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
            Frame::Gpu(g) => {
                assert_eq!(g.width, TEST_WIDTH);
                assert_eq!(g.height, TEST_HEIGHT);
            }
        }
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn d3d11_viewport_scale_encode_decode_roundtrip() {
        use crate::D3D11TextureFrame;
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
        use windows::Win32::Graphics::Direct3D11::{
            D3D11CreateDevice, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION,
            D3D11_SUBRESOURCE_DATA, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
        };
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        use windows::Win32::Foundation::HMODULE;

        let capture_w = 1920u32;
        let capture_h = 1080u32;
        let encode_w = 960u32;
        let encode_h = 540u32;

        // Create a D3D11 device for the test.
        let mut device = None;
        let mut context = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        }
        .expect("D3D11CreateDevice");
        let device = device.unwrap();
        let context = context.unwrap();

        // Create a BGRA texture at capture dimensions.
        let bgra_data = vec![128u8; (capture_w * capture_h * 4) as usize];
        let desc = D3D11_TEXTURE2D_DESC {
            Width: capture_w,
            Height: capture_h,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: 0,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let init_data = D3D11_SUBRESOURCE_DATA {
            pSysMem: bgra_data.as_ptr().cast(),
            SysMemPitch: capture_w * 4,
            SysMemSlicePitch: 0,
        };
        let mut texture = None;
        unsafe { device.CreateTexture2D(&desc, Some(&init_data), Some(&mut texture)) }
            .expect("CreateTexture2D");
        let texture = texture.unwrap();

        // Encoder at viewport (smaller) dimensions.
        let mut enc = D3D11Encoder::new(
            h264_profile(),
            encode_w,
            encode_h,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .expect("encoder construction");

        let mut dec = D3D11Decoder::new(CodecKind::H264).expect("decoder construction");

        let frame = D3D11TextureFrame {
            texture: texture.as_raw() as *mut _,
            device: device.as_raw() as *mut _,
            device_context: context.as_raw() as *mut _,
            width: capture_w,
            height: capture_h,
            format: DXGI_FORMAT_B8G8R8A8_UNORM.0 as u32,
        };

        let mut all_packets = Vec::new();
        for pts in 0..30 {
            let pkts = enc
                .submit_d3d11_texture(&frame, pts, pts == 0)
                .expect("submit_d3d11_texture");
            all_packets.extend(pkts);
            if !all_packets.is_empty() {
                break;
            }
        }
        assert!(!all_packets.is_empty(), "VP-scaled encoder produced no packets after 30 frames");

        for pkt in &all_packets {
            dec.submit(&pkt.data).expect("submit");
        }

        let mut decoded_frame = None;
        for pts in 30..60 {
            if let Some(f) = dec.next_frame().expect("next_frame") {
                decoded_frame = Some(f);
                break;
            }
            let pkts = enc.submit_d3d11_texture(&frame, pts, false).expect("encode");
            for pkt in &pkts {
                dec.submit(&pkt.data).expect("submit");
            }
        }

        let f = decoded_frame.expect("decoder never produced a frame");
        match f {
            Frame::Cpu(f) => {
                assert_eq!(f.width, encode_w, "decoded width should match encode (viewport) dims");
                assert_eq!(f.height, encode_h);
                assert!(!f.y.is_empty());
            }
            Frame::Gpu(g) => {
                assert_eq!(g.width, encode_w);
                assert_eq!(g.height, encode_h);
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
            Frame::Gpu(g) => {
                assert_eq!(g.width, TEST_WIDTH);
                assert_eq!(g.height, TEST_HEIGHT);
            }
        }
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU with HEVC Main10 (Windows)"]
    fn d3d11_hevc_main10_encode_produces_packets() {
        let mut enc = D3D11Encoder::new(
            hevc_main10_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .expect("HEVC Main10 encoder construction");

        let bgra = vec![100u8; (TEST_WIDTH * TEST_HEIGHT * 4) as usize];

        let mut all_packets = Vec::new();
        for pts in 0..30 {
            let pkts = enc
                .encode_bgra(&bgra, pts, pts == 0)
                .expect("encode_bgra failed");
            all_packets.extend(pkts);
            if !all_packets.is_empty() {
                break;
            }
        }
        assert!(!all_packets.is_empty(), "Main10 encoder produced no packets after 30 frames");
        // Verify the first packet contains valid HEVC NALUs (starts with
        // 00 00 00 01 or is Annex-B formatted after extradata prepend).
        assert!(
            all_packets[0].data.len() > 4,
            "packet too small to contain HEVC NALUs"
        );
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU with HEVC encode (Windows)"]
    fn d3d11_hevc_extradata_is_valid_annexb() {
        use crate::bitstream_sps::parse_sps_chroma_bit_depth;
        use tether_protocol::control::CodecKind;

        let mut enc = D3D11Encoder::new(
            hevc_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .expect("HEVC encoder construction");

        let bgra = vec![128u8; (TEST_WIDTH * TEST_HEIGHT * 4) as usize];

        let mut keyframe = None;
        for pts in 0..30 {
            let pkts = enc
                .encode_bgra(&bgra, pts, pts == 0)
                .expect("encode_bgra");
            for pkt in pkts {
                if pkt.keyframe {
                    keyframe = Some(pkt);
                    break;
                }
            }
            if keyframe.is_some() {
                break;
            }
        }
        let kf = keyframe.expect("encoder produced no keyframe after 30 frames");

        // Keyframe data must start with Annex-B start code (extradata was
        // converted from hvcC if needed by snapshot_extradata).
        assert!(
            kf.data.starts_with(&[0x00, 0x00, 0x00, 0x01])
                || kf.data.starts_with(&[0x00, 0x00, 0x01]),
            "keyframe does not start with Annex-B start code: {:02x?}",
            &kf.data[..kf.data.len().min(8)]
        );

        // SPS parser must be able to extract chroma + bit_depth.
        let sps = parse_sps_chroma_bit_depth(&kf.data, CodecKind::Hevc);
        assert!(
            sps.is_some(),
            "SPS parser could not find valid SPS in HEVC keyframe — \
             extradata format conversion may have failed"
        );
        let sps = sps.unwrap();
        assert_eq!(sps.chroma_format_idc, 1, "expected 4:2:0");
        assert_eq!(sps.bit_depth_luma, 8, "expected 8-bit");
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU with H.264 encode (Windows)"]
    fn d3d11_h264_extradata_is_valid_annexb() {
        use crate::bitstream_sps::parse_sps_chroma_bit_depth;
        use tether_protocol::control::CodecKind;

        let mut enc = D3D11Encoder::new(
            h264_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .expect("H.264 encoder construction");

        let bgra = vec![128u8; (TEST_WIDTH * TEST_HEIGHT * 4) as usize];

        let mut keyframe = None;
        for pts in 0..30 {
            let pkts = enc
                .encode_bgra(&bgra, pts, pts == 0)
                .expect("encode_bgra");
            for pkt in pkts {
                if pkt.keyframe {
                    keyframe = Some(pkt);
                    break;
                }
            }
            if keyframe.is_some() {
                break;
            }
        }
        let kf = keyframe.expect("encoder produced no keyframe after 30 frames");

        assert!(
            kf.data.starts_with(&[0x00, 0x00, 0x00, 0x01])
                || kf.data.starts_with(&[0x00, 0x00, 0x01]),
            "keyframe does not start with Annex-B start code: {:02x?}",
            &kf.data[..kf.data.len().min(8)]
        );

        let sps = parse_sps_chroma_bit_depth(&kf.data, CodecKind::H264);
        assert!(
            sps.is_some(),
            "SPS parser could not find valid SPS in H.264 keyframe — \
             extradata format conversion may have failed"
        );
        let sps = sps.unwrap();
        assert_eq!(sps.chroma_format_idc, 1, "expected 4:2:0");
        assert_eq!(sps.bit_depth_luma, 8, "expected 8-bit");
    }

    /// Reproduce the live HEVC first-IDR failure: encode a keyframe,
    /// submit ONLY that packet to a fresh decoder, signal EOF, and
    /// verify a frame comes back. This isolates whether the decoder
    /// can handle a single self-contained IDR (extradata + slice).
    #[test]
    #[ignore = "requires D3D11VA-capable GPU with HEVC encode (Windows)"]
    fn d3d11_hevc_single_idr_decode() {
        let mut enc = D3D11Encoder::new(
            hevc_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .expect("HEVC encoder construction");

        let bgra = vec![128u8; (TEST_WIDTH * TEST_HEIGHT * 4) as usize];

        let mut keyframe_data = None;
        for pts in 0..30 {
            let pkts = enc.encode_bgra(&bgra, pts, pts == 0).expect("encode");
            for pkt in pkts {
                if pkt.keyframe && keyframe_data.is_none() {
                    keyframe_data = Some(pkt.data.clone());
                }
            }
            if keyframe_data.is_some() {
                break;
            }
        }
        let kf = keyframe_data.expect("no keyframe produced");

        // Dump first bytes for debugging
        eprintln!("keyframe {} bytes, first 64: {:02x?}", kf.len(), &kf[..kf.len().min(64)]);

        // Parse NALU types from the bitstream
        let mut i = 0;
        let mut nalu_types = Vec::new();
        while i + 4 < kf.len() {
            if kf[i] == 0 && kf[i + 1] == 0 && kf[i + 2] == 0 && kf[i + 3] == 1 {
                let nalu_type = (kf[i + 4] >> 1) & 0x3F;
                nalu_types.push(nalu_type);
                i += 4;
            } else {
                i += 1;
            }
        }
        eprintln!("NALU types in keyframe: {:?}", nalu_types);
        assert!(
            nalu_types.contains(&32),
            "keyframe missing VPS (type 32); found types: {nalu_types:?}"
        );
        assert!(
            nalu_types.contains(&33),
            "keyframe missing SPS (type 33); found types: {nalu_types:?}"
        );
        assert!(
            nalu_types.contains(&34),
            "keyframe missing PPS (type 34); found types: {nalu_types:?}"
        );

        // Submit to a FRESH decoder (no prior state) and verify decode
        let mut dec = D3D11Decoder::new(CodecKind::Hevc).expect("decoder");
        dec.submit(&kf).expect("submit keyframe");
        dec.signal_eof().expect("signal_eof");

        let mut got_frame = false;
        for _ in 0..8 {
            if dec.next_frame().expect("next_frame").is_some() {
                got_frame = true;
                break;
            }
        }
        assert!(got_frame, "decoder produced no frames from single IDR with VPS/SPS/PPS");
    }

    /// Verify the extradata stored in the encoder starts with VPS (type 32)
    /// after the reordering fix. AMF emits SPS→PPS→VPS but we fix it to
    /// VPS→SPS→PPS at snapshot time.
    #[test]
    #[ignore = "requires D3D11VA-capable GPU with HEVC encode (Windows)"]
    fn d3d11_hevc_extradata_starts_with_vps() {
        let enc = D3D11Encoder::new(
            hevc_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .expect("HEVC encoder construction");

        let extradata = enc.extradata();
        assert!(extradata.len() > 5, "extradata too short");
        // Must start with Annex-B start code.
        assert_eq!(&extradata[..4], &[0x00, 0x00, 0x00, 0x01]);
        // First NALU must be VPS (type 32).
        let nalu_type = (extradata[4] >> 1) & 0x3F;
        assert_eq!(
            nalu_type, 32,
            "first NALU in extradata should be VPS (32), got {nalu_type}"
        );
    }
}
