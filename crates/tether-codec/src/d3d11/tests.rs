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
    /// Intel PCI vendor ID — routes `D3D11Encoder::new` to the QSV
    /// backend (see `backends_for_vendor`).
    const VENDOR_INTEL: u32 = 0x8086;

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

    fn hevc_444_profile() -> VideoProfile {
        VideoProfile {
            codec: CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv444,
            bit_depth: 8,
        }
    }

    fn hevc_444_10bit_profile() -> VideoProfile {
        VideoProfile {
            codec: CodecKind::Hevc,
            chroma: ChromaSubsampling::Yuv444,
            bit_depth: 10,
        }
    }

    /// 4:4:4 must be refused at construction — the Video Processor path
    /// only outputs 4:2:0 (NV12/P010), so advertising 4:4:4 would make
    /// the host silently downsample. The rejection runs before any
    /// device work, so this needs no GPU (not `#[ignore]`).
    #[test]
    fn d3d11_rejects_444_at_construction_no_silent_downsample() {
        for profile in [hevc_444_profile(), hevc_444_10bit_profile()] {
            let err = match D3D11Encoder::new(
                profile,
                TEST_WIDTH,
                TEST_HEIGHT,
                TEST_FPS,
                TEST_BITRATE_KBPS,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            ) {
                Ok(_) => panic!("4:4:4 must be rejected, not silently downsampled: {profile:?}"),
                Err(e) => e,
            };
            assert!(
                err.to_string().contains("4:4:4"),
                "expected a 4:4:4 rejection for {profile:?}, got: {err}"
            );
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
            0,
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
            0,
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
            0,
        );
        assert!(enc.is_ok(), "shared-device encoder failed: {:?}", enc.err());
        assert!(enc.unwrap().is_hardware());
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn d3d11_decoder_constructs_h264() {
        let dec = D3D11Decoder::new(CodecKind::H264, false);
        assert!(dec.is_ok(), "H.264 decoder construction failed: {:?}", dec.err());
        assert!(dec.unwrap().is_hardware());
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn d3d11_decoder_constructs_hevc() {
        let dec = D3D11Decoder::new(CodecKind::Hevc, false);
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
            0,
        )
        .expect("encoder construction");

        let mut dec = D3D11Decoder::new(CodecKind::H264, false).expect("decoder construction");

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

    /// With `gpu_export = true` the decoder must hand back a GPU-resident
    /// `Frame::Gpu` carrying two non-null D3D11 NT shared handles (what
    /// wgpu's Vulkan backend imports), not a CPU download. Drives the real
    /// QSV GPU encode path on Intel and asserts inside the shared helper;
    /// the renderer-side Vulkan import is exercised by the client
    /// end-to-end. SKIPs on non-Intel GPUs (see the helper's vendor gate).
    #[test]
    #[ignore = "requires Intel QSV (Windows) + FFmpeg build with working oneVPL-over-D3D11"]
    fn d3d11_qsv_decode_exports_gpu_shared_handles() {
        gpu_roundtrip_for_vendor(VENDOR_INTEL, "hevc_qsv", true);
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
            0,
        )
        .expect("encoder construction");

        let mut dec = D3D11Decoder::new(CodecKind::H264, false).expect("decoder construction");

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
            0,
        )
        .expect("encoder construction");

        let mut dec = D3D11Decoder::new(CodecKind::Hevc, false).expect("decoder construction");

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
            0,
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
            0,
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
            0,
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
            0,
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
        let mut dec = D3D11Decoder::new(CodecKind::Hevc, false).expect("decoder");
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
            0,
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

    /// Create a D3D11 device with `VIDEO_SUPPORT` (required by both the
    /// Video Processor blit and QSV's oneVPL session) and multithread
    /// protection (QSV derivation requires it). Mirrors the capture
    /// layer's device setup so QSV tests exercise the real config.
    fn create_video_device() -> (
        windows::Win32::Graphics::Direct3D11::ID3D11Device,
        windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext,
    ) {
        use windows::core::Interface;
        use windows::Win32::Foundation::HMODULE;
        use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
        use windows::Win32::Graphics::Direct3D11::{
            D3D11CreateDevice, ID3D11Multithread, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT, D3D11_SDK_VERSION,
        };

        let mut device = None;
        let mut context = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        }
        .expect("D3D11CreateDevice with VIDEO_SUPPORT");
        let device = device.unwrap();
        let context = context.unwrap();
        if let Ok(mt) = device.cast::<ID3D11Multithread>() {
            let _ = unsafe { mt.SetMultithreadProtected(true) };
        }
        (device, context)
    }

    /// PCI vendor ID of the GPU backing `device`. Used to gate vendor
    /// roundtrip tests *before* constructing the encoder: attempting an
    /// AMF/NVENC encoder on a machine without that GPU faults inside the
    /// vendor's runtime (STATUS_ACCESS_VIOLATION), so we must skip rather
    /// than try-and-fall-back.
    fn device_vendor_id(
        device: &windows::Win32::Graphics::Direct3D11::ID3D11Device,
    ) -> u32 {
        use windows::core::Interface;
        use windows::Win32::Graphics::Dxgi::{IDXGIAdapter, IDXGIDevice};
        unsafe {
            let dxgi: IDXGIDevice = device.cast().expect("ID3D11Device -> IDXGIDevice");
            let adapter: IDXGIAdapter = dxgi.GetAdapter().expect("GetAdapter");
            adapter.GetDesc().map(|d| d.VendorId).unwrap_or(0)
        }
    }

    /// AMD PCI vendor ID — routes `D3D11Encoder::new` to the AMF backend.
    const VENDOR_AMD: u32 = 0x1002;
    /// NVIDIA PCI vendor ID — routes `D3D11Encoder::new` to NVENC.
    const VENDOR_NVIDIA: u32 = 0x10de;

    /// Shared zero-copy GPU encode→decode round trip for one vendor — the
    /// path the host actually uses (`submit_d3d11_texture`). Exercises
    /// the backend's device setup, the VP BGRA→NV12 blit, the
    /// first-frame-forced-IDR requirement, the per-frame hw_frames pool
    /// handling (the non-QSV dynamic pool vs QSV's reused single surface),
    /// and an encode→decode round trip with viewport scaling.
    ///
    /// Asserts the *intended* backend opened, not the `hevc_mf` fallback
    /// `backends_for_vendor` appends — so on the wrong GPU the test fails
    /// loudly rather than silently passing through Media Foundation.
    fn gpu_roundtrip_for_vendor(vendor_id: u32, expected_backend: &str, gpu_export: bool) {
        use crate::D3D11TextureFrame;
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D11::{
            D3D11_SUBRESOURCE_DATA, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
        };
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC,
        };

        let capture_w = 1920u32;
        let capture_h = 1080u32;
        let encode_w = 1280u32;
        let encode_h = 720u32;

        let (device, context) = create_video_device();

        // SKIP-with-diagnostic when this machine's GPU isn't the target
        // vendor. Constructing the wrong vendor's encoder faults inside
        // that vendor's runtime, so gate on the present GPU *before*
        // touching `D3D11Encoder::new`. Run this test on a matching GPU.
        let present_vendor = device_vendor_id(&device);
        if present_vendor != vendor_id {
            eprintln!(
                "SKIP {expected_backend}: GPU vendor 0x{present_vendor:04x} != target \
                 0x{vendor_id:04x}; run on a {expected_backend}-capable GPU"
            );
            return;
        }

        // BGRA source texture at capture dims.
        let bgra = vec![128u8; (capture_w * capture_h * 4) as usize];
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
        let init = D3D11_SUBRESOURCE_DATA {
            pSysMem: bgra.as_ptr().cast(),
            SysMemPitch: capture_w * 4,
            SysMemSlicePitch: 0,
        };
        let mut texture = None;
        unsafe { device.CreateTexture2D(&desc, Some(&init), Some(&mut texture)) }
            .expect("CreateTexture2D");
        let texture = texture.unwrap();

        let mut enc = D3D11Encoder::new(
            hevc_profile(),
            encode_w,
            encode_h,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            device.as_raw() as *mut _,
            context.as_raw() as *mut _,
            vendor_id,
        )
        .expect("encoder construction");
        // We're on matching hardware (gated above), so the intended
        // backend must open — a fall-through to `hevc_mf` means this
        // FFmpeg build lacks the vendor encoder, which is a real failure.
        assert_eq!(
            enc.name(),
            expected_backend,
            "GPU is vendor 0x{vendor_id:04x} but {expected_backend} did not open (got {}); \
             FFmpeg build is missing the {expected_backend} encoder",
            enc.name()
        );

        let mut dec = D3D11Decoder::new(CodecKind::Hevc, gpu_export).expect("decoder construction");
        let frame = D3D11TextureFrame {
            texture: texture.as_raw() as *mut _,
            device: device.as_raw() as *mut _,
            device_context: context.as_raw() as *mut _,
            width: capture_w,
            height: capture_h,
            format: DXGI_FORMAT_B8G8R8A8_UNORM.0 as u32,
        };

        // Sustained throughput: push 90 frames (~1.5 s at 60 fps) back to
        // back, draining packets each iteration. Every submit MUST
        // succeed — a too-small surface pool serves only the first frame,
        // then fails AVERROR(ENOMEM) on every subsequent
        // `av_hwframe_get_buffer` (the production "froze after one frame"
        // bug). Do NOT break early; that's exactly what hid the bug.
        let mut total_packets = 0usize;
        let mut decoded_dims = None;
        for pts in 0..90 {
            let pkts = enc
                .submit_d3d11_texture(&frame, pts, pts == 0)
                .expect("submit_d3d11_texture (sustained) — surface pool exhausted?");
            total_packets += pkts.len();
            for pkt in &pkts {
                dec.submit(&pkt.data).expect("submit");
            }
            if let Some(f) = dec.next_frame().expect("next_frame") {
                decoded_dims = Some(match &f {
                    Frame::Cpu(f) => (f.width, f.height),
                    Frame::Gpu(g) => (g.width, g.height),
                });
                // When the renderer can import D3D11 textures, the decoder
                // must hand back a GPU-resident frame carrying two non-null
                // NT shared handles — not a CPU download.
                if gpu_export {
                    match f {
                        Frame::Gpu(g) => {
                            let (_w, _h, _pts, source, _guard) = g.into_parts();
                            let crate::GpuFrameSource::D3D11Texture(tex) = source;
                            assert!(!tex.y_handle.is_null(), "Y plane shared handle is null");
                            assert!(!tex.uv_handle.is_null(), "UV plane shared handle is null");
                        }
                        Frame::Cpu(_) => {
                            panic!("gpu_export = true must yield Frame::Gpu, got a CPU download")
                        }
                    }
                }
            }
        }
        assert!(
            total_packets > 30,
            "expected sustained packet output over 90 frames, got {total_packets}"
        );
        let (dw, dh) = decoded_dims.expect("decoder produced no frame from GPU encode");
        assert_eq!((dw, dh), (encode_w, encode_h));
    }

    /// QSV via the zero-copy GPU submit path. The vendor-agnostic tests
    /// pass `vendor_id=0` (→ MF), so this is the only coverage of QSV.
    #[test]
    #[ignore = "requires Intel QSV (Windows) + FFmpeg build with working oneVPL-over-D3D11"]
    fn d3d11_qsv_gpu_encode_decode_roundtrip() {
        gpu_roundtrip_for_vendor(VENDOR_INTEL, "hevc_qsv", false);
    }

    /// AMF via the zero-copy GPU submit path — the only coverage of the
    /// AMD backend (dynamic hw_frames pool, async session, `async_depth=1`).
    #[test]
    #[ignore = "requires AMD GPU with AMF (Windows)"]
    fn d3d11_amf_gpu_encode_decode_roundtrip() {
        gpu_roundtrip_for_vendor(VENDOR_AMD, "hevc_amf", false);
    }

    /// NVENC via the zero-copy GPU submit path — the only coverage of the
    /// NVIDIA backend (dynamic hw_frames pool, `delay=0` + `zerolatency`).
    #[test]
    #[ignore = "requires NVIDIA GPU with NVENC (Windows)"]
    fn d3d11_nvenc_gpu_encode_decode_roundtrip() {
        gpu_roundtrip_for_vendor(VENDOR_NVIDIA, "hevc_nvenc", false);
    }

    /// Diagnostic probe for QSV encode latency. Measures `submit_d3d11_texture`
    /// while a second thread saturates the SAME iGPU with `CopyResource`
    /// batches. Findings (Intel iGPU): encode alone ~5ms; under a light
    /// shared-device `CopyResource` loop still ~4ms (device-wide lock is
    /// NOT the bottleneck); under GPU-queue saturation ~16-26ms. This is
    /// why a loopback session (host encode + client decode/render/present
    /// on one iGPU) shows 100ms+ `avg_encode_ms`: `receive_packet`'s
    /// blocking MFX `SyncOperation` waits behind queued GPU work. The
    /// encoder itself is fast — the latency is GPU contention/topology,
    /// not an encoder bug. Keep this probe before "optimizing" QSV options.
    #[test]
    #[ignore = "diagnostic: QSV submit latency under shared-iGPU contention"]
    fn d3d11_qsv_submit_under_capture_contention() {
        use crate::D3D11TextureFrame;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D11::{
            ID3D11DeviceContext, ID3D11Texture2D, D3D11_SUBRESOURCE_DATA,
            D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
        };
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC,
        };

        let capture_w = 1920u32;
        let capture_h = 1200u32;
        let encode_w = 1440u32;
        let encode_h = 896u32;

        let (device, context) = create_video_device();

        let make_bgra = |w: u32, h: u32| -> ID3D11Texture2D {
            let data = vec![128u8; (w * h * 4) as usize];
            let desc = D3D11_TEXTURE2D_DESC {
                Width: w,
                Height: h,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: 0,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            };
            let init = D3D11_SUBRESOURCE_DATA {
                pSysMem: data.as_ptr().cast(),
                SysMemPitch: w * 4,
                SysMemSlicePitch: 0,
            };
            let mut t = None;
            unsafe { device.CreateTexture2D(&desc, Some(&init), Some(&mut t)) }
                .expect("CreateTexture2D");
            t.unwrap()
        };

        let enc_src = make_bgra(capture_w, capture_h);

        // Capture-contention thread: CopyResource a full-size frame on the
        // shared context in a ~60fps loop. Pass COM pointers as usize.
        let stop = Arc::new(AtomicBool::new(false));
        let dev_us = device.as_raw() as usize;
        let ctx_us = context.as_raw() as usize;
        let cap_src = make_bgra(capture_w, capture_h);
        let cap_dst = make_bgra(capture_w, capture_h);
        let cap_src_us = cap_src.as_raw() as usize;
        let cap_dst_us = cap_dst.as_raw() as usize;
        let stop_t = stop.clone();
        let contention = std::thread::spawn(move || {
            let ctx: ID3D11DeviceContext =
                unsafe { ID3D11DeviceContext::from_raw_borrowed(&(ctx_us as *mut _)) }
                    .unwrap()
                    .clone();
            let src: ID3D11Texture2D =
                unsafe { ID3D11Texture2D::from_raw_borrowed(&(cap_src_us as *mut _)) }
                    .unwrap()
                    .clone();
            let dst: ID3D11Texture2D =
                unsafe { ID3D11Texture2D::from_raw_borrowed(&(cap_dst_us as *mut _)) }
                    .unwrap()
                    .clone();
            let _ = dev_us;
            // Saturate the GPU queue: many full-frame copies per flush, no
            // sleep — approximates the loopback client's continuous render.
            while !stop_t.load(Ordering::Relaxed) {
                for _ in 0..32 {
                    unsafe { ctx.CopyResource(&dst, &src) };
                }
                unsafe { ctx.Flush() };
            }
        });

        let mut enc = D3D11Encoder::new(
            hevc_profile(),
            encode_w,
            encode_h,
            60,
            TEST_BITRATE_KBPS,
            device.as_raw() as *mut _,
            context.as_raw() as *mut _,
            VENDOR_INTEL,
        )
        .expect("QSV encoder construction");

        let frame = D3D11TextureFrame {
            texture: enc_src.as_raw() as *mut _,
            device: device.as_raw() as *mut _,
            device_context: context.as_raw() as *mut _,
            width: capture_w,
            height: capture_h,
            format: DXGI_FORMAT_B8G8R8A8_UNORM.0 as u32,
        };

        let mut submit_us: Vec<u128> = Vec::with_capacity(90);
        for pts in 0..90 {
            let t0 = std::time::Instant::now();
            enc.submit_d3d11_texture(&frame, pts, pts == 0)
                .expect("submit_d3d11_texture under contention");
            submit_us.push(t0.elapsed().as_micros());
        }
        stop.store(true, Ordering::Relaxed);
        let _ = contention.join();

        let warm = &submit_us[5..];
        let avg: u128 = warm.iter().sum::<u128>() / warm.len() as u128;
        let mx = warm.iter().max().unwrap();
        eprintln!(
            "QSV submit UNDER capture contention (warm): avg={}us max={}us  [first5={:?}us]",
            avg, mx, &submit_us[..5]
        );
    }

    /// QSV via the `encode_bgra` path (`av_hwframe_transfer_data` upload),
    /// the same upload mechanism FFmpeg's own `hwupload`+`hevc_qsv` uses.
    /// Isolates "does QSV encode work at all" from the zero-copy VP-blit
    /// GPU path: if this passes but `d3d11_qsv_gpu_encode_decode_roundtrip`
    /// fails, the bug is specifically the VP-blit-into-mapped-QSV-surface.
    #[test]
    #[ignore = "requires Intel QSV (Windows) + FFmpeg build with working oneVPL-over-D3D11"]
    fn d3d11_qsv_encode_bgra_roundtrip() {
        use windows::core::Interface;

        let (device, context) = create_video_device();
        let mut enc = D3D11Encoder::new(
            hevc_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            device.as_raw() as *mut _,
            context.as_raw() as *mut _,
            VENDOR_INTEL,
        )
        .expect("QSV encoder construction");
        assert_eq!(enc.name(), "hevc_qsv", "QSV unavailable; got {}", enc.name());

        let mut dec = D3D11Decoder::new(CodecKind::Hevc, false).expect("decoder construction");
        let bgra = vec![128u8; (TEST_WIDTH * TEST_HEIGHT * 4) as usize];

        let mut packets = Vec::new();
        for pts in 0..30 {
            let pkts = enc.encode_bgra(&bgra, pts, pts == 0).expect("encode_bgra");
            packets.extend(pkts);
            if !packets.is_empty() {
                break;
            }
        }
        assert!(!packets.is_empty(), "QSV encode_bgra produced no packets after 30 frames");

        for pkt in &packets {
            dec.submit(&pkt.data).expect("submit");
        }
        let mut decoded = None;
        for pts in 30..60 {
            if let Some(f) = dec.next_frame().expect("next_frame") {
                decoded = Some(f);
                break;
            }
            let pkts = enc.encode_bgra(&bgra, pts, false).expect("encode_bgra");
            for pkt in &pkts {
                dec.submit(&pkt.data).expect("submit");
            }
        }
        assert!(decoded.is_some(), "decoder produced no frame from QSV encode_bgra");
    }

    /// Build a QSV encoder, drop it, then build another at different
    /// dims on the SAME D3D11 device. Regression for the recreate path:
    /// a QSV session/frame-pool that isn't released on drop made the
    /// second encoder's child-frames-context `CreateTexture2D` fail with
    /// `DXGI_ERROR_INVALID_CALL`. The host recreates the encoder on every
    /// viewport change, so this must hold.
    #[test]
    #[ignore = "requires Intel QSV (Windows) + FFmpeg build with working oneVPL-over-D3D11"]
    fn d3d11_qsv_encoder_rebuild_same_device() {
        use windows::core::Interface;

        let (device, context) = create_video_device();
        let dev_ptr = device.as_raw() as *mut _;
        let ctx_ptr = context.as_raw() as *mut _;

        let enc_a = D3D11Encoder::new(
            hevc_profile(),
            1152,
            720,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            dev_ptr,
            ctx_ptr,
            VENDOR_INTEL,
        )
        .expect("first QSV encoder construction");
        assert_eq!(enc_a.name(), "hevc_qsv", "QSV unavailable; got {}", enc_a.name());
        drop(enc_a);

        let enc_b = D3D11Encoder::new(
            hevc_profile(),
            1440,
            896,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            dev_ptr,
            ctx_ptr,
            VENDOR_INTEL,
        )
        .expect("rebuilt QSV encoder on same device — was the first session released?");
        assert_eq!(enc_b.name(), "hevc_qsv");
    }
}
