//! Hardware tests for the D3D11 encoder and decoder. These require a
//! Windows system with a GPU that supports D3D11VA encode/decode (any
//! modern Intel/AMD/NVIDIA discrete or integrated GPU). Run with
//! `cargo test -p tether-codec --lib d3d11::tests -- --ignored`.

// Included from `d3d11/mod.rs` as `mod tests;`; the inner `mod tests` keeps
// these hardware tests grouped under one named block.
#[cfg(test)]
#[allow(clippy::module_inception)]
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

    /// Media Foundation encoders (`*_mf`, the unknown-vendor fallback)
    /// never populate `AVCodecContext::extradata`, so `snapshot_extradata`
    /// must NOT hard-fail construction for them (the bug that turned all 12
    /// MF-path tests red on the static FFmpeg build). They keep the
    /// self-decodable-IDR invariant a different way — parameter sets ride
    /// in-band on *every* keyframe (including mid-stream forced IDRs), so
    /// each IDR decodes standalone with no prepend:
    ///   - HEVC: VPS (32) + SPS (33) + PPS (34) NAL units
    ///   - H.264: SPS (7) + PPS (8) NAL units
    ///   - AV1: a sequence-header OBU (type 1)
    ///
    /// This pins both facts for all three MF codecs: empty extradata at
    /// open, and the in-band headers on each keyframe.
    ///
    /// Routes through vendor 0 → MF-only (`backends_for_vendor`), so it runs
    /// on any D3D11 GPU regardless of which vendor it is.
    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows); exercises the Media Foundation fallback"]
    fn d3d11_mf_keyframes_carry_inband_parameter_sets() {
        // (label, profile, codec, required in-band header types). For HEVC
        // and H.264 these are NAL unit types; for AV1, OBU types.
        let cases: &[(&str, VideoProfile, CodecKind, &[u8])] = &[
            ("hevc_mf", hevc_profile(), CodecKind::Hevc, &[32, 33, 34]),
            ("h264_mf", h264_profile(), CodecKind::H264, &[7, 8]),
            ("av1_mf", av1_profile(), CodecKind::Av1, &[1]),
        ];

        for &(label, profile, kind, want) in cases {
            let mut enc = D3D11Encoder::new(
                profile,
                TEST_WIDTH,
                TEST_HEIGHT,
                TEST_FPS,
                TEST_BITRATE_KBPS,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
            .unwrap_or_else(|e| panic!("{label}: construction must succeed, got {e:?}"));

            // MF leaves extradata empty — that's expected, not a failure.
            assert!(
                enc.extradata().is_empty(),
                "{label}: MF is expected to leave extradata empty (parameter \
                 sets ride in-band); got {} bytes",
                enc.extradata().len()
            );

            let bgra = vec![64u8; (TEST_WIDTH * TEST_HEIGHT * 4) as usize];
            let mut keyframes = Vec::new();
            for pts in 0..40i64 {
                // Force an IDR at 0 and again mid-stream at 15 — both must be
                // self-decodable, not just the first.
                let force = pts == 0 || pts == 15;
                let pkts = enc.encode_bgra(&bgra, pts, force).expect("encode_bgra");
                keyframes.extend(pkts.into_iter().filter(|p| p.keyframe));
                if keyframes.len() >= 2 {
                    break;
                }
            }
            assert!(
                keyframes.len() >= 2,
                "{label}: expected at least 2 keyframes (initial + forced), got {}",
                keyframes.len()
            );

            for (i, kf) in keyframes.iter().enumerate() {
                let found = match kind {
                    CodecKind::Av1 => av1_obu_types(&kf.data),
                    CodecKind::Hevc | CodecKind::H264 => {
                        assert!(
                            kf.data.starts_with(&[0, 0, 0, 1]) || kf.data.starts_with(&[0, 0, 1]),
                            "{label}: keyframe #{i} must be Annex-B; starts {:02x?}",
                            &kf.data[..kf.data.len().min(8)]
                        );
                        nalu_types(&kf.data, kind)
                    }
                };
                for &ps in want {
                    assert!(
                        found.contains(&ps),
                        "{label}: keyframe #{i} missing in-band header {ps} — \
                         IDR is not self-decodable. Got {found:?}"
                    );
                }
            }
        }
    }

    /// Scan an Annex-B bitstream and return the NAL unit type of each NALU.
    /// Walks one byte at a time looking for 3- or 4-byte start codes; the
    /// header byte immediately follows the start code.
    fn nalu_types(data: &[u8], kind: CodecKind) -> Vec<u8> {
        let mut types = Vec::new();
        let mut i = 0;
        while i + 3 < data.len() {
            let sc4 = data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1;
            let sc3 = data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1;
            // A 4-byte start code needs a header byte at i+4; a 3-byte one at
            // i+3 (guaranteed present by the loop condition).
            if sc4 && i + 4 < data.len() {
                let hdr = data[i + 4];
                types.push(nal_unit_type(hdr, kind));
                i += 4;
            } else if sc3 {
                let hdr = data[i + 3];
                types.push(nal_unit_type(hdr, kind));
                i += 3;
            } else {
                i += 1;
            }
        }
        types
    }

    fn nal_unit_type(hdr: u8, kind: CodecKind) -> u8 {
        match kind {
            CodecKind::H264 => hdr & 0x1F,
            CodecKind::Hevc => (hdr >> 1) & 0x3F,
            CodecKind::Av1 => hdr,
        }
    }

    /// Walk an AV1 low-overhead-format bitstream and return each OBU type.
    /// OBU header: bit7 forbidden, bits6-3 type, bit2 extension, bit1
    /// has_size, bit0 reserved; optional 1-byte extension; leb128 size.
    fn av1_obu_types(data: &[u8]) -> Vec<u8> {
        let mut types = Vec::new();
        let mut i = 0;
        while i < data.len() {
            let hdr = data[i];
            let obu_type = (hdr >> 3) & 0x0F;
            let ext = (hdr >> 2) & 1 == 1;
            let has_size = (hdr >> 1) & 1 == 1;
            types.push(obu_type);
            let mut p = i + 1;
            if ext {
                p += 1;
            }
            if !has_size {
                break; // can't advance without sizes; stop scanning
            }
            // leb128 size
            let mut size: usize = 0;
            let mut shift = 0;
            loop {
                if p >= data.len() {
                    return types;
                }
                let b = data[p];
                p += 1;
                size |= ((b & 0x7F) as usize) << shift;
                if b & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            i = p + size;
        }
        types
    }

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

    fn av1_profile() -> VideoProfile {
        VideoProfile {
            codec: CodecKind::Av1,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 8,
        }
    }

    fn av1_10bit_profile() -> VideoProfile {
        VideoProfile {
            codec: CodecKind::Av1,
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
        assert!(
            enc.is_ok(),
            "H.264 encoder construction failed: {:?}",
            enc.err()
        );
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
        assert!(
            enc.is_ok(),
            "HEVC encoder construction failed: {:?}",
            enc.err()
        );
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn d3d11_encoder_shared_device_h264() {
        use windows::core::Interface;
        use windows::Win32::Foundation::HMODULE;
        use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
        use windows::Win32::Graphics::Direct3D11::{
            D3D11CreateDevice, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION,
        };

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
        assert!(
            dec.is_ok(),
            "H.264 decoder construction failed: {:?}",
            dec.err()
        );
        assert!(dec.unwrap().is_hardware());
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn d3d11_decoder_constructs_hevc() {
        let dec = D3D11Decoder::new(CodecKind::Hevc, false);
        assert!(
            dec.is_ok(),
            "HEVC decoder construction failed: {:?}",
            dec.err()
        );
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
        assert!(
            !all_packets.is_empty(),
            "encoder produced no packets after 30 frames"
        );

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
        gpu_roundtrip_for_vendor(VENDOR_INTEL, "hevc_qsv", true, hevc_profile());
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn d3d11_viewport_scale_encode_decode_roundtrip() {
        use crate::D3D11TextureFrame;
        use windows::core::Interface;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM;

        let capture_w = 1920u32;
        let capture_h = 1080u32;
        let encode_w = 960u32;
        let encode_h = 540u32;

        // The Video Processor blit needs a VIDEO_SUPPORT device, and the
        // encoder must share that same device or the VP can't write into
        // its hw_frames pool (cross-device blit fails). `create_video_device`
        // provides both; pass its pointers to the encoder.
        let (device, context) = create_video_device();
        let texture = make_bgra_texture(&device, capture_w, capture_h);

        // Encoder at viewport (smaller) dimensions, on the shared device.
        // vendor 0 → Media Foundation, the path an unknown-vendor host uses;
        // this is also the only coverage of the MF zero-copy GPU submit.
        let mut enc = D3D11Encoder::new(
            h264_profile(),
            encode_w,
            encode_h,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            device.as_raw() as *mut _,
            context.as_raw() as *mut _,
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
        assert!(
            !all_packets.is_empty(),
            "VP-scaled encoder produced no packets after 30 frames"
        );

        for pkt in &all_packets {
            dec.submit(&pkt.data).expect("submit");
        }

        let mut decoded_frame = None;
        for pts in 30..60 {
            if let Some(f) = dec.next_frame().expect("next_frame") {
                decoded_frame = Some(f);
                break;
            }
            let pkts = enc
                .submit_d3d11_texture(&frame, pts, false)
                .expect("encode");
            for pkt in &pkts {
                dec.submit(&pkt.data).expect("submit");
            }
        }

        let f = decoded_frame.expect("decoder never produced a frame");
        match f {
            Frame::Cpu(f) => {
                assert_eq!(
                    f.width, encode_w,
                    "decoded width should match encode (viewport) dims"
                );
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
        assert!(
            !all_packets.is_empty(),
            "encoder produced no packets after 30 frames"
        );

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

    /// HEVC Main10 encode must declare 10-bit in its SPS. Routes through the
    /// *present* GPU's hardware encoder via the CPU-upload `encode_bgra`
    /// path, complementing the GPU-submit
    /// `d3d11_amf_hevc_main10_gpu_encode_decode_roundtrip`. "Produces
    /// packets" alone is half-credit — an encoder missing the Main10 profile
    /// pin still emits a (malformed, 8-bit-SPS) bitstream that passes a
    /// length check — so we parse the SPS and assert 10-bit, which catches a
    /// dropped `AV_PROFILE_HEVC_MAIN_10` regardless of backend.
    ///
    /// NOT routed through vendor 0 → Media Foundation: MF's HEVC Main10 MFT
    /// hangs `encode_bgra` on some drivers (observed on AMD RDNA 4). Skips on
    /// unknown-vendor GPUs (which would fall back to that MF path).
    #[test]
    #[ignore = "requires D3D11VA-capable GPU with HEVC Main10 (Windows)"]
    fn d3d11_hevc_main10_encode_produces_packets() {
        use windows::core::Interface;

        let (device, context) = create_video_device();
        let vendor_id = device_vendor_id(&device);
        if !matches!(vendor_id, VENDOR_INTEL | VENDOR_AMD | VENDOR_NVIDIA) {
            eprintln!(
                "SKIP d3d11_hevc_main10_encode_produces_packets: unknown vendor \
                 0x{vendor_id:04x} routes to MF, whose Main10 MFT can hang"
            );
            return;
        }

        let mut enc = D3D11Encoder::new(
            hevc_main10_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            device.as_raw() as *mut _,
            context.as_raw() as *mut _,
            vendor_id,
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
        assert!(
            !all_packets.is_empty(),
            "Main10 encoder produced no packets after 30 frames"
        );
        // Verify the first packet contains valid HEVC NALUs (starts with
        // 00 00 00 01 or is Annex-B formatted after extradata prepend).
        assert!(
            all_packets[0].data.len() > 4,
            "packet too small to contain HEVC NALUs"
        );
        // The SPS must actually declare 10-bit 4:2:0. "Produces packets" is
        // half-credit: an encoder missing the Main10 profile pin still emits
        // a (malformed) bitstream that passes the length check but reports
        // 8-bit, then fails to decode. Parsing the bit depth here catches a
        // dropped `AV_PROFILE_HEVC_MAIN_10` on any backend (incl. MF/QSV),
        // not just on the AMF decode round trip.
        use crate::bitstream_sps::parse_sps_chroma_bit_depth;
        let sps = parse_sps_chroma_bit_depth(&all_packets[0].data, CodecKind::Hevc)
            .expect("SPS parse from Main10 keyframe");
        assert_eq!(sps.chroma_format_idc, 1, "expected 4:2:0");
        assert_eq!(
            sps.bit_depth_luma, 10,
            "Main10 SPS must declare 10-bit luma — the encoder's \
             AV_PROFILE_HEVC_MAIN_10 pin likely regressed"
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
            let pkts = enc.encode_bgra(&bgra, pts, pts == 0).expect("encode_bgra");
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

        // Keyframe data must start with an Annex-B start code. This routes
        // through vendor 0 → Media Foundation, which carries VPS/SPS/PPS
        // in-band on the keyframe (no extradata prepend — see
        // `d3d11_mf_keyframes_carry_inband_parameter_sets`).
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
        // Guard the `AV_PROFILE_HEVC_MAIN` arm of the exhaustive profile
        // match: 8-bit 4:2:0 must declare general_profile_idc=1 (Main),
        // not fall through to some backend default.
        assert_eq!(
            sps.profile_idc, 1,
            "HEVC 8-bit 4:2:0 SPS must be Main (general_profile_idc 1)"
        );
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
            let pkts = enc.encode_bgra(&bgra, pts, pts == 0).expect("encode_bgra");
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
        // The SPS must declare profile_idc=77 (Main). Direct guard for the
        // `AV_PROFILE_H264_MAIN` pin: a regression that drops the pin lets
        // AMD's MF MFT default to Baseline (profile_idc=66), which the
        // D3D11VA hardware decoder rejects (INVALIDDATA) even though the
        // stream is otherwise valid — so a decode-only check wouldn't catch
        // it on the software path.
        assert_eq!(
            sps.profile_idc, 77,
            "H.264 SPS must be Main (77); 66 = Baseline means the \
             AV_PROFILE_H264_MAIN pin regressed"
        );
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
        eprintln!(
            "keyframe {} bytes, first 64: {:02x?}",
            kf.len(),
            &kf[..kf.len().min(64)]
        );

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
        assert!(
            got_frame,
            "decoder produced no frames from single IDR with VPS/SPS/PPS"
        );
    }

    /// Verify the extradata stored in the encoder starts with VPS (type 32)
    /// after the reordering fix (`snapshot_extradata` rewrites AMF's
    /// SPS→PPS→VPS to VPS→SPS→PPS). Routes through the *present* GPU vendor
    /// so it exercises a real vendor encoder's extradata (QSV/AMF/NVENC).
    ///
    /// SKIPs on unknown-vendor GPUs: those fall back to Media Foundation,
    /// which never populates `extradata` (parameter sets ride in-band — see
    /// `d3d11_mf_keyframes_carry_inband_parameter_sets`), so there is no
    /// extradata ordering to check.
    #[test]
    #[ignore = "requires D3D11VA-capable GPU with HEVC encode (Windows)"]
    fn d3d11_hevc_extradata_starts_with_vps() {
        use windows::core::Interface;

        let (device, context) = create_video_device();
        let vendor_id = device_vendor_id(&device);
        if !matches!(vendor_id, VENDOR_INTEL | VENDOR_AMD | VENDOR_NVIDIA) {
            eprintln!(
                "SKIP d3d11_hevc_extradata_starts_with_vps: unknown vendor \
                 0x{vendor_id:04x} routes to MF, which carries parameter sets \
                 in-band (no extradata to inspect)"
            );
            return;
        }

        let enc = D3D11Encoder::new(
            hevc_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            device.as_raw() as *mut _,
            context.as_raw() as *mut _,
            vendor_id,
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

    /// PCI vendor ID of the GPU backing `device`. Used by tests that run on
    /// the default adapter to gate on known-vendor hardware (skip the unknown
    /// vendors that route to Media Foundation). The per-vendor roundtrip tests
    /// instead bind to a chosen adapter up front via `video_device_for_vendor`.
    fn device_vendor_id(device: &windows::Win32::Graphics::Direct3D11::ID3D11Device) -> u32 {
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

    /// Outcome of trying to bind a video device to a vendor's GPU. Distinguishes
    /// "no such vendor in the box" from "vendor present but its device wouldn't
    /// create" so callers can print an accurate SKIP diagnostic — the two read
    /// very differently during triage.
    enum VendorVideoDevice {
        Ready(
            (
                windows::Win32::Graphics::Direct3D11::ID3D11Device,
                windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext,
            ),
        ),
        /// No adapter of this vendor is physically present.
        Absent,
        /// At least one adapter of this vendor is present, but every
        /// `D3D11CreateDevice` against it failed (carries the last error).
        Unusable(windows::core::Error),
    }

    /// Create a video device on a DXGI adapter of `vendor_id`.
    ///
    /// Vendor-roundtrip tests must run on the matching GPU's *own* device:
    /// `D3D11Encoder::new(.., vendor_id)` on a foreign vendor's device faults
    /// inside that vendor's runtime (STATUS_ACCESS_VIOLATION), not a catchable
    /// error. The production rule is "use the GPU driving the display" (the
    /// default adapter), but a hardware test should exercise *every* vendor
    /// GPU in the box, not just the default one — so here we enumerate adapters
    /// and bind to the requested vendor explicitly. A laptop with an Intel
    /// iGPU driving the panel and an NVIDIA eGPU thus runs both the QSV and the
    /// NVENC round trips; a single-vendor box SKIPs the absent vendors.
    ///
    /// Enumeration continues past a device-create failure: on a box with two
    /// adapters of the same vendor, a later one may succeed where the first
    /// didn't. `Unusable` is returned only when *every* matching adapter failed.
    ///
    /// An explicit adapter requires `D3D_DRIVER_TYPE_UNKNOWN` (passing
    /// `HARDWARE` with a non-null adapter fails) — same as the production
    /// capture fallback in `tether-capture`.
    fn video_device_for_vendor(vendor_id: u32) -> VendorVideoDevice {
        use windows::core::Interface;
        use windows::Win32::Foundation::HMODULE;
        use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_UNKNOWN;
        use windows::Win32::Graphics::Direct3D11::{
            D3D11CreateDevice, ID3D11Multithread, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT, D3D11_SDK_VERSION,
        };
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};

        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.expect("CreateDXGIFactory1");
        let mut adapter_idx = 0u32;
        // Track whether we saw the vendor at all, and the last create error, so
        // a box with several same-vendor adapters reports "unusable" only after
        // every one has failed.
        let mut last_create_err = None;
        while let Ok(adapter) = unsafe { factory.EnumAdapters1(adapter_idx) } {
            adapter_idx += 1;
            // GetDesc1 on a just-enumerated adapter shouldn't fail; a failure is
            // a driver/environment error, not "vendor absent" — surface it loudly
            // rather than silently skipping (which would read as "GPU not present").
            let desc = unsafe { adapter.GetDesc1() }.expect("IDXGIAdapter1::GetDesc1");
            if desc.VendorId != vendor_id {
                continue;
            }

            // Try this adapter. On a box with two same-vendor GPUs, index 0 is
            // the display adapter — the one most likely to support encode and
            // matching production's default pick — but if it won't create a
            // device we fall through to the next same-vendor adapter rather than
            // giving up, since a later one may still work.
            let mut device = None;
            let mut context = None;
            // The vendor GPU is present but device creation can still fail (a
            // corrupted driver, VIDEO_SUPPORT unavailable on this config). That's
            // "present but not usable", which the harness treats as SKIP, not a
            // hard failure — mirror the rest of the test infrastructure.
            if let Err(e) = unsafe {
                D3D11CreateDevice(
                    &adapter,
                    D3D_DRIVER_TYPE_UNKNOWN,
                    HMODULE::default(),
                    D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
                    None,
                    D3D11_SDK_VERSION,
                    Some(&mut device),
                    None,
                    Some(&mut context),
                )
            } {
                last_create_err = Some(e);
                continue;
            }
            let device = device.unwrap();
            let context = context.unwrap();
            if let Ok(mt) = device.cast::<ID3D11Multithread>() {
                let _ = unsafe { mt.SetMultithreadProtected(true) };
            }
            return VendorVideoDevice::Ready((device, context));
        }
        match last_create_err {
            Some(e) => VendorVideoDevice::Unusable(e),
            None => VendorVideoDevice::Absent,
        }
    }

    /// Create a video device, returning it only if an AMD GPU is present;
    /// otherwise print a SKIP diagnostic and return `None`. AMF-specific
    /// tests must gate on the real vendor *before* constructing the encoder:
    /// `D3D11Encoder::new(.., VENDOR_AMD)` on non-AMD hardware faults inside
    /// a foreign vendor runtime (STATUS_ACCESS_VIOLATION), not a catchable
    /// error. Binds to the AMD adapter itself (see `video_device_for_vendor`)
    /// so a box where AMD isn't the default adapter still runs these.
    fn amd_video_device_or_skip(
        test: &str,
    ) -> Option<(
        windows::Win32::Graphics::Direct3D11::ID3D11Device,
        windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext,
    )> {
        match video_device_for_vendor(VENDOR_AMD) {
            VendorVideoDevice::Ready(dev) => Some(dev),
            VendorVideoDevice::Absent => {
                eprintln!(
                    "SKIP {test}: no AMD GPU (0x{VENDOR_AMD:04x}) present; \
                     run on an AMF-capable AMD GPU"
                );
                None
            }
            VendorVideoDevice::Unusable(e) => {
                eprintln!(
                    "SKIP {test}: AMD GPU (0x{VENDOR_AMD:04x}) present but \
                     D3D11CreateDevice (UNKNOWN driver, VIDEO_SUPPORT) failed: {e}"
                );
                None
            }
        }
    }

    /// Allocate a mid-grey BGRA `ID3D11Texture2D` to feed the zero-copy
    /// `submit_d3d11_texture` path. Shared by the AMF latency / forced-IDR
    /// tests so each isn't re-declaring the same texture descriptor.
    fn make_bgra_texture(
        device: &windows::Win32::Graphics::Direct3D11::ID3D11Device,
        w: u32,
        h: u32,
    ) -> windows::Win32::Graphics::Direct3D11::ID3D11Texture2D {
        use windows::Win32::Graphics::Direct3D11::{
            D3D11_SUBRESOURCE_DATA, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
        };
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        let data = vec![128u8; (w * h * 4) as usize];
        let desc = D3D11_TEXTURE2D_DESC {
            Width: w,
            Height: h,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
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
    }

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
    fn gpu_roundtrip_for_vendor(
        vendor_id: u32,
        expected_backend: &str,
        gpu_export: bool,
        profile: VideoProfile,
    ) {
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

        // SKIP-with-diagnostic when this machine has no GPU of the target
        // vendor. Constructing the wrong vendor's encoder faults inside that
        // vendor's runtime, so we bind to the target vendor's *own* adapter
        // (not the default one) *before* touching `D3D11Encoder::new`. On a
        // dual-GPU box this exercises each vendor in turn; a single-vendor box
        // SKIPs the absent vendors. Run this test on a matching GPU.
        let (device, context) = match video_device_for_vendor(vendor_id) {
            VendorVideoDevice::Ready(dev) => dev,
            VendorVideoDevice::Absent => {
                eprintln!(
                    "SKIP {expected_backend}: no GPU of vendor 0x{vendor_id:04x} present; \
                     run on a {expected_backend}-capable GPU"
                );
                return;
            }
            VendorVideoDevice::Unusable(e) => {
                eprintln!(
                    "SKIP {expected_backend}: GPU of vendor 0x{vendor_id:04x} present but \
                     D3D11CreateDevice (UNKNOWN driver, VIDEO_SUPPORT) failed: {e}"
                );
                return;
            }
        };

        // BGRA source texture at capture dims.
        let bgra = vec![128u8; (capture_w * capture_h * 4) as usize];
        let desc = D3D11_TEXTURE2D_DESC {
            Width: capture_w,
            Height: capture_h,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
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
            profile,
            encode_w,
            encode_h,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            device.as_raw() as *mut _,
            context.as_raw() as *mut _,
            vendor_id,
        )
        .expect("encoder construction");
        // We're on matching hardware (gated above), so the intended backend
        // must open — a fall-through to `*_mf` normally means this FFmpeg
        // build lacks the vendor encoder, a real failure. EXCEPTION: AV1
        // hardware encode needs a recent GPU (Intel Arc / AMD RDNA 3+ /
        // NVIDIA Ada). On an older same-vendor GPU the vendor's AV1 encoder
        // is genuinely absent and construction falls back to `av1_mf`, so
        // SKIP rather than fail. HEVC/H.264 hardware encode is universal on
        // these vendors, so a fallback there stays a hard failure.
        if enc.name() != expected_backend {
            if expected_backend.starts_with("av1") && enc.name().ends_with("_mf") {
                eprintln!(
                    "SKIP {expected_backend}: vendor 0x{vendor_id:04x} GPU lacks AV1 \
                     hardware encode (opened {}); needs Intel Arc / AMD RDNA 3+ / NVIDIA Ada",
                    enc.name()
                );
                return;
            }
            panic!(
                "GPU is vendor 0x{vendor_id:04x} but {expected_backend} did not open (got {}); \
                 FFmpeg build is missing the {expected_backend} encoder",
                enc.name()
            );
        }

        let mut dec = D3D11Decoder::new(profile.codec, gpu_export).expect("decoder construction");
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
                            assert!(!tex.handle.is_null(), "NV12 shared handle is null");
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
        gpu_roundtrip_for_vendor(VENDOR_INTEL, "hevc_qsv", false, hevc_profile());
    }

    /// AV1 via Intel QSV (`av1_qsv`) — the AV1 analogue of the HEVC QSV
    /// round trip, exercising the wired AV1 encode + D3D11VA AV1 decode on
    /// Arc. Asserts `av1_qsv` opened (not the `av1_mf` fallback). SKIPs on
    /// non-Intel GPUs via the helper's vendor gate.
    #[test]
    #[ignore = "requires Intel Arc AV1 QSV (Windows) + FFmpeg oneVPL-over-D3D11"]
    fn d3d11_qsv_av1_gpu_encode_decode_roundtrip() {
        gpu_roundtrip_for_vendor(VENDOR_INTEL, "av1_qsv", false, av1_profile());
    }

    /// AMF via the zero-copy GPU submit path — the only coverage of the
    /// AMD backend (dynamic hw_frames pool, async session, `async_depth=1`).
    #[test]
    #[ignore = "requires AMD GPU with AMF (Windows)"]
    fn d3d11_amf_gpu_encode_decode_roundtrip() {
        gpu_roundtrip_for_vendor(VENDOR_AMD, "hevc_amf", false, hevc_profile());
    }

    /// AMF H.264 via the zero-copy GPU submit path. The AMF backend was
    /// first developed on this AMD GPU but every refinement since landed on
    /// Intel QSV, so the AMD H.264 encode path (`h264_amf`, distinct from
    /// the HEVC one) had no coverage at all.
    #[test]
    #[ignore = "requires AMD GPU with AMF (Windows)"]
    fn d3d11_amf_h264_gpu_encode_decode_roundtrip() {
        gpu_roundtrip_for_vendor(VENDOR_AMD, "h264_amf", false, h264_profile());
    }

    /// AMF HEVC Main10 (10-bit / P010) via the zero-copy GPU submit path.
    /// The Windows host probe reports Main10 as Supported *statically* for
    /// every vendor without ever exercising it; if AMF can't encode 10-bit
    /// the live encoder dies with Goodbye(InternalError) at lazy-init. This
    /// is the test that turns that unverified claim into a checked one.
    /// `gpu_export = true`: a Main10 P010 surface has no 8-bit CPU-download
    /// representation, so the GPU shared-handle export is the only valid
    /// path (mirrors the host decode probe).
    #[test]
    #[ignore = "requires AMD GPU with AMF HEVC Main10 (Windows)"]
    fn d3d11_amf_hevc_main10_gpu_encode_decode_roundtrip() {
        gpu_roundtrip_for_vendor(VENDOR_AMD, "hevc_amf", true, hevc_main10_profile());
    }

    /// AV1 via the vendor-agnostic `av1_mf` (Media Foundation) fallback —
    /// the `encode_bgra` path with `vendor_id=0`, mirroring
    /// `d3d11_hevc_encode_decode_roundtrip`. This is the AV1 encoder a host
    /// reaches on any GPU whose vendor isn't AMD/Intel/NVIDIA, AND the only
    /// AV1 encoder left after `av1_amf` fail-fasts on a pre-RDNA-3 AMD card —
    /// so the production fallback needs its own coverage, not just `av1_amf`.
    /// Runs on any Windows GPU with an AV1 MF encoder + D3D11VA AV1 decode.
    #[test]
    #[ignore = "requires Windows GPU with AV1 MF encode + D3D11VA AV1 decode"]
    fn d3d11_av1_mf_encode_decode_roundtrip() {
        let mut enc = D3D11Encoder::new(
            av1_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        )
        .expect("av1_mf encoder construction");
        assert_eq!(enc.name(), "av1_mf", "expected av1_mf; got {}", enc.name());

        let mut dec = D3D11Decoder::new(CodecKind::Av1, false).expect("decoder construction");
        let bgra = vec![96u8; (TEST_WIDTH * TEST_HEIGHT * 4) as usize];

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
        assert!(
            !all_packets.is_empty(),
            "av1_mf produced no packets after 30 frames"
        );
        for pkt in &all_packets {
            dec.submit(&pkt.data).expect("submit failed");
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
        let frame = decoded.expect("decoder never produced a frame from av1_mf output");
        match frame {
            Frame::Cpu(f) => assert_eq!((f.width, f.height), (TEST_WIDTH, TEST_HEIGHT)),
            Frame::Gpu(g) => assert_eq!((g.width, g.height), (TEST_WIDTH, TEST_HEIGHT)),
        }
    }

    /// AMF AV1 (`av1_amf`) via the zero-copy GPU submit path — the AV1
    /// analogue of `d3d11_amf_gpu_encode_decode_roundtrip`. RDNA 3+ does AV1
    /// in hardware; this is the direct exercise of the wired AV1 encode +
    /// D3D11VA AV1 decode chain on AMD. Asserts `av1_amf` opened (not the
    /// `av1_mf` fallback), so a build missing the AV1 AMF encoder fails loud.
    #[test]
    #[ignore = "requires AMD GPU with AMF AV1 (Windows, RDNA 3+)"]
    fn d3d11_amf_av1_gpu_encode_decode_roundtrip() {
        gpu_roundtrip_for_vendor(VENDOR_AMD, "av1_amf", false, av1_profile());
    }

    /// AMF AV1 10-bit (P010) via the zero-copy GPU submit path. AV1 Main
    /// covers both 8 and 10-bit 4:2:0 off one profile pin; a P010 surface
    /// has no 8-bit CPU-download representation, so `gpu_export = true` is
    /// the only valid path (mirrors the Main10 HEVC test).
    #[test]
    #[ignore = "requires AMD GPU with AMF AV1 10-bit (Windows, RDNA 3+)"]
    fn d3d11_amf_av1_10bit_gpu_encode_decode_roundtrip() {
        gpu_roundtrip_for_vendor(VENDOR_AMD, "av1_amf", true, av1_10bit_profile());
    }

    /// On-demand IDR for AV1: `av1_amf` sets `forced_idr=1` and we stamp
    /// `pict_type = I` on forced frames. AV1's AVOption set differs from
    /// hevc_amf (no `gops_per_idr`, enum `latency`), so this confirms the
    /// AV1-specific dict still yields a mid-stream keyframe — the load-
    /// bearing behaviour for loss recovery. Mirrors the HEVC forced-IDR test.
    #[test]
    #[ignore = "requires AMD GPU with AMF AV1 (Windows, RDNA 3+)"]
    fn d3d11_amf_av1_forced_idr_midstream_produces_keyframe() {
        use crate::D3D11TextureFrame;
        use windows::core::Interface;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM;

        let Some((device, context)) =
            amd_video_device_or_skip("d3d11_amf_av1_forced_idr_midstream_produces_keyframe")
        else {
            return;
        };

        let texture = make_bgra_texture(&device, TEST_WIDTH, TEST_HEIGHT);
        let mut enc = D3D11Encoder::new(
            av1_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            device.as_raw() as *mut _,
            context.as_raw() as *mut _,
            VENDOR_AMD,
        )
        .expect("AMF AV1 encoder construction");
        assert_eq!(
            enc.name(),
            "av1_amf",
            "av1_amf unavailable; got {}",
            enc.name()
        );

        let frame = D3D11TextureFrame {
            texture: texture.as_raw() as *mut _,
            device: device.as_raw() as *mut _,
            device_context: context.as_raw() as *mut _,
            width: TEST_WIDTH,
            height: TEST_HEIGHT,
            format: DXGI_FORMAT_B8G8R8A8_UNORM.0 as u32,
        };

        // Warm the stream: frame 0 is an implicit keyframe. With
        // `async_depth=1` each submit drains synchronously, so discarding
        // the returned packets here means the frame-0 keyframe never reaches
        // the forced-IDR scan window below — any keyframe seen there is the
        // mid-stream ForceIdr, not a leftover.
        for pts in 0..30 {
            let _ = enc
                .submit_d3d11_texture(&frame, pts, false)
                .expect("warmup submit");
        }

        let mut saw_forced_keyframe = false;
        for pts in 30..45 {
            let pkts = enc
                .submit_d3d11_texture(&frame, pts, pts == 30)
                .expect("submit");
            if pkts.iter().any(|p| p.keyframe) {
                saw_forced_keyframe = true;
                break;
            }
        }
        assert!(
            saw_forced_keyframe,
            "av1_amf emitted no keyframe after a mid-stream ForceIdr — \
             forced_idr / pict_type=I is being ignored, breaking loss recovery"
        );
    }

    /// Build an AMF encoder, drop it, then build another at different dims
    /// on the SAME D3D11 device. The AMD analogue of
    /// `d3d11_qsv_encoder_rebuild_same_device`, and the direct regression
    /// for `D3D11Encoder::drop`'s `flush_buffers`: AMF has a single hardware
    /// session that releases asynchronously, so without the flush the second
    /// construction can fail the single-session limit. The host rebuilds the
    /// encoder on every viewport change, so this must hold.
    #[test]
    #[ignore = "requires AMD GPU with AMF (Windows)"]
    fn d3d11_amf_encoder_rebuild_same_device() {
        use windows::core::Interface;

        let Some((device, context)) =
            amd_video_device_or_skip("d3d11_amf_encoder_rebuild_same_device")
        else {
            return;
        };
        // `as_raw()` borrows without an AddRef, so these raw pointers are
        // only valid while `device`/`context` live. Both outlive `enc_a` and
        // `enc_b` (dropped at end of function), and `D3D11Encoder::new`
        // AddRefs internally for its own retained reference — so dropping
        // `enc_a` can't free the device out from under `enc_b`. Don't drop
        // `device`/`context` early.
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
            VENDOR_AMD,
        )
        .expect("first AMF encoder construction");
        assert_eq!(
            enc_a.name(),
            "hevc_amf",
            "AMF unavailable; got {}",
            enc_a.name()
        );
        drop(enc_a);

        let enc_b = D3D11Encoder::new(
            hevc_profile(),
            1440,
            896,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            dev_ptr,
            ctx_ptr,
            VENDOR_AMD,
        )
        .expect("rebuilt AMF encoder on same device — was the first session released on drop?");
        assert_eq!(enc_b.name(), "hevc_amf");
    }

    /// The HEVC VPS-reorder fix exists *because AMF* emits SPS→PPS→VPS;
    /// `snapshot_extradata` rewrites it to VPS→SPS→PPS. The existing
    /// `d3d11_hevc_extradata_starts_with_vps` routes through vendor 0 → Media
    /// Foundation, so it never exercises AMF's ordering. This constructs the
    /// real `hevc_amf` encoder and asserts the stored extradata leads with
    /// VPS — covering the actual backend the fix was written for.
    #[test]
    #[ignore = "requires AMD GPU with AMF HEVC encode (Windows)"]
    fn d3d11_amf_hevc_extradata_starts_with_vps() {
        use windows::core::Interface;

        let Some((device, context)) =
            amd_video_device_or_skip("d3d11_amf_hevc_extradata_starts_with_vps")
        else {
            return;
        };

        let enc = D3D11Encoder::new(
            hevc_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            device.as_raw() as *mut _,
            context.as_raw() as *mut _,
            VENDOR_AMD,
        )
        .expect("AMF HEVC encoder construction");
        assert_eq!(
            enc.name(),
            "hevc_amf",
            "AMF unavailable; got {}",
            enc.name()
        );

        let extradata = enc.extradata();
        assert!(extradata.len() > 5, "extradata too short");
        assert_eq!(
            &extradata[..4],
            &[0x00, 0x00, 0x00, 0x01],
            "extradata not Annex-B"
        );
        let nalu_type = (extradata[4] >> 1) & 0x3F;
        assert_eq!(
            nalu_type, 32,
            "first NALU in AMF extradata should be VPS (32), got {nalu_type} — \
             the SPS→PPS→VPS reorder fix regressed on the real AMF backend"
        );
    }

    /// On-demand IDR (`ControlMessage::ForceIdr`) is load-bearing for loss
    /// recovery. The AMF path sets `forced_idr=1` and stamps `pict_type = I`
    /// on the forced frame; this verifies AMF honours it *mid-stream* — a
    /// forced submit after the stream is warm must yield a keyframe packet,
    /// not just at frame 0. If AMF ignored it, recovery would silently fail.
    #[test]
    #[ignore = "requires AMD GPU with AMF HEVC encode (Windows)"]
    fn d3d11_amf_forced_idr_midstream_produces_keyframe() {
        use crate::D3D11TextureFrame;
        use windows::core::Interface;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM;

        let Some((device, context)) =
            amd_video_device_or_skip("d3d11_amf_forced_idr_midstream_produces_keyframe")
        else {
            return;
        };

        let texture = make_bgra_texture(&device, TEST_WIDTH, TEST_HEIGHT);
        let mut enc = D3D11Encoder::new(
            hevc_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            device.as_raw() as *mut _,
            context.as_raw() as *mut _,
            VENDOR_AMD,
        )
        .expect("AMF encoder construction");
        assert_eq!(
            enc.name(),
            "hevc_amf",
            "AMF unavailable; got {}",
            enc.name()
        );

        let frame = D3D11TextureFrame {
            texture: texture.as_raw() as *mut _,
            device: device.as_raw() as *mut _,
            device_context: context.as_raw() as *mut _,
            width: TEST_WIDTH,
            height: TEST_HEIGHT,
            format: DXGI_FORMAT_B8G8R8A8_UNORM.0 as u32,
        };

        // Warm the stream — frame 0 is an implicit IDR (the encoder forces
        // it on `first_frame` regardless of `force_keyframe`), frames 1..29
        // are P-frames — draining packets so the frame-0 keyframe is long
        // gone before we force the next one.
        for pts in 0..30 {
            let _ = enc
                .submit_d3d11_texture(&frame, pts, false)
                .expect("warmup submit");
        }

        // Request a keyframe mid-stream; a keyframe packet must follow.
        let mut saw_forced_keyframe = false;
        for pts in 30..45 {
            let pkts = enc
                .submit_d3d11_texture(&frame, pts, pts == 30)
                .expect("submit");
            if pkts.iter().any(|p| p.keyframe) {
                saw_forced_keyframe = true;
                break;
            }
        }
        assert!(
            saw_forced_keyframe,
            "AMF emitted no keyframe after a mid-stream ForceIdr — \
             forced_idr / pict_type=I is being ignored, breaking loss recovery"
        );
    }

    /// SKIP-with-diagnostic for the AMF latency knobs (`async_depth=1`,
    /// `usage=ultralowlatency`, `latency=1`). amfenc defaults `async_depth`
    /// to 16; if our override silently no-ops the encoder buffers ~16 frames
    /// before the first packet (16 frames of added latency — exactly what
    /// the encoder.rs comment warns about). We can't read the option back,
    /// so we observe its effect: count submits before the first packet.
    /// async_depth=1 ⇒ a frame or two; a double-digit count ⇒ the override
    /// didn't take. Records the negative per the codec-knob rule in CLAUDE.md.
    #[test]
    #[ignore = "requires AMD GPU with AMF HEVC encode (Windows)"]
    fn d3d11_amf_async_depth_one_bounds_output_delay() {
        use crate::D3D11TextureFrame;
        use windows::core::Interface;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM;

        let Some((device, context)) =
            amd_video_device_or_skip("d3d11_amf_async_depth_one_bounds_output_delay")
        else {
            return;
        };

        let texture = make_bgra_texture(&device, TEST_WIDTH, TEST_HEIGHT);
        let mut enc = D3D11Encoder::new(
            hevc_profile(),
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_FPS,
            TEST_BITRATE_KBPS,
            device.as_raw() as *mut _,
            context.as_raw() as *mut _,
            VENDOR_AMD,
        )
        .expect("AMF encoder construction");
        assert_eq!(
            enc.name(),
            "hevc_amf",
            "AMF unavailable; got {}",
            enc.name()
        );

        let frame = D3D11TextureFrame {
            texture: texture.as_raw() as *mut _,
            device: device.as_raw() as *mut _,
            device_context: context.as_raw() as *mut _,
            width: TEST_WIDTH,
            height: TEST_HEIGHT,
            format: DXGI_FORMAT_B8G8R8A8_UNORM.0 as u32,
        };

        let mut frames_to_first = None;
        for pts in 0..32 {
            let pkts = enc
                .submit_d3d11_texture(&frame, pts, pts == 0)
                .expect("submit");
            if !pkts.is_empty() {
                frames_to_first = Some(pts + 1);
                break;
            }
        }
        let n = frames_to_first.expect("AMF produced no packet in 32 frames");
        eprintln!(
            "AMF frames-to-first-packet = {n} (async_depth=1 expects ~1-2; \
             ~16 ⇒ the async_depth override no-op'd)"
        );
        // Threshold rationale: with B-frames disabled (`max_b_frames=0`) and
        // `async_depth=1`, at most one encode is in flight and there is no
        // reorder DPB, so output should appear within ~1-2 submits; 4 is a 2×
        // tolerance for pipeline fill. The amfenc default of 16 (or any value
        // the override failed to apply) lands far above this. Observed 2 on
        // the Radeon 8060S. If a future driver legitimately needs >4 at
        // async_depth=1, revisit — but a double-digit count means the knob
        // no-op'd, which is the regression this guards.
        assert!(
            n <= 4,
            "AMF buffered {n} frames before the first packet — async_depth=1 \
             is likely being ignored (amfenc default is 16)"
        );
    }

    /// NVENC via the zero-copy GPU submit path — the only coverage of the
    /// NVIDIA backend (dynamic hw_frames pool, `delay=0` + `zerolatency`).
    #[test]
    #[ignore = "requires NVIDIA GPU with NVENC (Windows)"]
    fn d3d11_nvenc_gpu_encode_decode_roundtrip() {
        gpu_roundtrip_for_vendor(VENDOR_NVIDIA, "hevc_nvenc", false, hevc_profile());
    }

    /// AV1 via NVIDIA NVENC (`av1_nvenc`) — the AV1 analogue of the HEVC
    /// NVENC round trip (Ada+ does AV1 in hardware). Asserts `av1_nvenc`
    /// opened, not the `av1_mf` fallback. SKIPs on non-NVIDIA GPUs.
    #[test]
    #[ignore = "requires NVIDIA Ada+ GPU with AV1 NVENC (Windows)"]
    fn d3d11_nvenc_av1_gpu_encode_decode_roundtrip() {
        gpu_roundtrip_for_vendor(VENDOR_NVIDIA, "av1_nvenc", false, av1_profile());
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
            ID3D11DeviceContext, ID3D11Texture2D, D3D11_SUBRESOURCE_DATA, D3D11_TEXTURE2D_DESC,
            D3D11_USAGE_DEFAULT,
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
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
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
            avg,
            mx,
            &submit_us[..5]
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
        assert_eq!(
            enc.name(),
            "hevc_qsv",
            "QSV unavailable; got {}",
            enc.name()
        );

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
        assert!(
            !packets.is_empty(),
            "QSV encode_bgra produced no packets after 30 frames"
        );

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
        assert!(
            decoded.is_some(),
            "decoder produced no frame from QSV encode_bgra"
        );
    }

    /// Diagnostic: can a second D3D11 device read content a first device
    /// wrote into a shared NT-handle texture, with only a `Flush` between
    /// them (no keyed mutex / fence)? This is the exact coherency contract
    /// the cross-device renderer relies on — the decoder copies a decoded
    /// plane into a `MISC_SHARED|NTHANDLE` texture on its device, the
    /// renderer opens it on its own device and samples. If this reads back
    /// zero on device B (but the right value on device A), the shared read
    /// is uninitialised/stale and the black-screen loopback is a coherency
    /// failure, not a render bug.
    #[test]
    #[ignore = "diagnostic: cross-device shared-handle read coherency (Windows)"]
    fn d3d11_cross_device_shared_handle_coherency() {
        use windows::core::Interface;
        use windows::Win32::Foundation::{HANDLE, HMODULE};
        use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
        use windows::Win32::Graphics::Direct3D11::{
            D3D11CreateDevice, ID3D11Device, ID3D11Device1, ID3D11DeviceContext, ID3D11Texture2D,
            D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ, D3D11_SDK_VERSION, D3D11_SUBRESOURCE_DATA,
            D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, D3D11_USAGE_STAGING,
        };
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8_UNORM, DXGI_SAMPLE_DESC};
        use windows::Win32::Graphics::Dxgi::{IDXGIResource1, DXGI_SHARED_RESOURCE_READ};

        const D3D11_RESOURCE_MISC_SHARED: u32 = 0x2;
        const D3D11_RESOURCE_MISC_SHARED_NTHANDLE: u32 = 0x800;
        const DIM: u32 = 64;
        const FILL: u8 = 200;

        fn make_device() -> (ID3D11Device, ID3D11DeviceContext) {
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
            (device.unwrap(), context.unwrap())
        }

        // Copy `tex` into a STAGING texture on `device`/`context` and read
        // the center R8 texel.
        fn read_center(
            device: &ID3D11Device,
            context: &ID3D11DeviceContext,
            tex: &ID3D11Texture2D,
        ) -> u8 {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: DIM,
                Height: DIM,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_R8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_STAGING,
                BindFlags: 0,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                MiscFlags: 0,
            };
            let mut staging = None;
            unsafe { device.CreateTexture2D(&desc, None, Some(&mut staging)) }
                .expect("staging CreateTexture2D");
            let staging = staging.unwrap();
            unsafe {
                context.CopyResource(&staging, tex);
                context.Flush();
                let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
                context
                    .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                    .expect("Map staging");
                let row = (DIM / 2) as usize;
                let col = (DIM / 2) as usize;
                let byte = *(mapped.pData as *const u8).add(row * mapped.RowPitch as usize + col);
                context.Unmap(&staging, 0);
                byte
            }
        }

        // Device A (producer / decoder role).
        let (dev_a, ctx_a) = make_device();
        // Source filled with a known value, then GPU-copied into the
        // shared texture — mirrors the decoder's CopySubresourceRegion.
        let src_data = vec![FILL; (DIM * DIM) as usize];
        let src_desc = D3D11_TEXTURE2D_DESC {
            Width: DIM,
            Height: DIM,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let init = D3D11_SUBRESOURCE_DATA {
            pSysMem: src_data.as_ptr().cast(),
            SysMemPitch: DIM,
            SysMemSlicePitch: 0,
        };
        let mut src = None;
        unsafe { dev_a.CreateTexture2D(&src_desc, Some(&init), Some(&mut src)) }
            .expect("source CreateTexture2D");
        let src = src.unwrap();

        let shared_desc = D3D11_TEXTURE2D_DESC {
            MiscFlags: D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
            ..src_desc
        };
        let mut shared = None;
        unsafe { dev_a.CreateTexture2D(&shared_desc, None, Some(&mut shared)) }
            .expect("shared CreateTexture2D");
        let shared = shared.unwrap();

        unsafe {
            ctx_a.CopyResource(&shared, &src);
            ctx_a.Flush();
        }

        // A must see its own write (proves the copy itself worked).
        let a_value = read_center(&dev_a, &ctx_a, &shared);
        assert_eq!(
            a_value, FILL,
            "device A can't read its own write — copy failed, not a coherency issue"
        );

        // Export the NT handle and open it on device B (renderer role).
        let dxgi_res: IDXGIResource1 = shared.cast().expect("IDXGIResource1");
        let handle: HANDLE = unsafe {
            dxgi_res.CreateSharedHandle(
                None,
                DXGI_SHARED_RESOURCE_READ.0,
                windows::core::PCWSTR::null(),
            )
        }
        .expect("CreateSharedHandle");

        let (dev_b, ctx_b) = make_device();
        let dev_b1: ID3D11Device1 = dev_b.cast().expect("ID3D11Device1");
        let opened: ID3D11Texture2D =
            unsafe { dev_b1.OpenSharedResource1(handle) }.expect("OpenSharedResource1");

        let b_value = read_center(&dev_b, &ctx_b, &opened);
        eprintln!("cross-device shared read: A={a_value} B={b_value} (expected {FILL})");
        assert_eq!(
            b_value, FILL,
            "device B read {b_value}, not {FILL}: the cross-device shared-handle read does \
             NOT see device A's GPU write with Flush alone — this is the loopback black screen"
        );
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
        assert_eq!(
            enc_a.name(),
            "hevc_qsv",
            "QSV unavailable; got {}",
            enc_a.name()
        );
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
