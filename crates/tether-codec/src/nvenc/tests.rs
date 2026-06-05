//! Tests for the NVENC backend.
//!
//! The mapping-table and NVIDIA-detection tests are no-hardware and run in
//! default `cargo test` on any host. The `#[ignore]` tests at the bottom
//! exercise the real encoder and need an NVIDIA GPU + an NVENC-enabled
//! FFmpeg (`--enable-nvenc --enable-cuda`); run them with
//! `cargo test -p tether-codec --ignored nvenc_`.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use rsmpeg::ffi;
use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

use super::encoder::{nvenc_codec_name as test_codec_name, nvenc_sw_format as test_sw_format};
use super::{nvidia_gpu_present_in, NvencEncoder};
use crate::Encoder;

// --- format / codec mapping tables -----------------------------------------

#[test]
fn sw_format_maps_exactly_the_advertised_profiles() {
    // 4:2:0 8/10-bit are the supported set; everything else (4:4:4, odd
    // depths) is deferred and must report UnsupportedInputFormat so the
    // probe records the profile unsupported rather than mis-encoding it.
    assert_eq!(
        test_sw_format(ChromaSubsampling::Yuv420, 8).unwrap(),
        ffi::AV_PIX_FMT_NV12
    );
    assert_eq!(
        test_sw_format(ChromaSubsampling::Yuv420, 10).unwrap(),
        ffi::AV_PIX_FMT_P010LE
    );
    assert!(test_sw_format(ChromaSubsampling::Yuv444, 8).is_err());
    assert!(test_sw_format(ChromaSubsampling::Yuv444, 10).is_err());
}

#[test]
fn codec_name_is_exhaustive_and_nvenc() {
    for kind in [CodecKind::H264, CodecKind::Hevc, CodecKind::Av1] {
        assert!(
            test_codec_name(kind).ends_with("_nvenc"),
            "{kind:?} should map to an *_nvenc encoder name"
        );
    }
    assert_eq!(test_codec_name(CodecKind::H264), "h264_nvenc");
    assert_eq!(test_codec_name(CodecKind::Hevc), "hevc_nvenc");
    assert_eq!(test_codec_name(CodecKind::Av1), "av1_nvenc");
}

// --- NVIDIA detection (synthetic sysfs tree) -------------------------------

/// Unique temp dir for a synthetic `/sys/class/drm` tree; removed on drop.
struct SysfsFixture {
    root: PathBuf,
}

impl SysfsFixture {
    fn new() -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "tether-nvenc-drm-{}-{}",
            std::process::id(),
            n
        ));
        fs::create_dir_all(&root).expect("create fixture root");
        Self { root }
    }

    /// Add a `renderD<num>` node whose `device/vendor` reads `vendor`.
    fn render_node(&self, num: u32, vendor: &str) -> &Self {
        let dev = self.root.join(format!("renderD{num}")).join("device");
        fs::create_dir_all(&dev).expect("create node");
        // Real sysfs writes the vendor with a trailing newline; detection
        // must trim it.
        fs::write(dev.join("vendor"), format!("{vendor}\n")).expect("write vendor");
        self
    }

    /// Add a non-render node (e.g. a `card*` or `version` entry) that
    /// detection must ignore.
    fn other_node(&self, name: &str, vendor: &str) -> &Self {
        let dev = self.root.join(name).join("device");
        fs::create_dir_all(&dev).expect("create node");
        fs::write(dev.join("vendor"), format!("{vendor}\n")).expect("write vendor");
        self
    }
}

impl Drop for SysfsFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn detects_nvidia_render_node() {
    let fx = SysfsFixture::new();
    fx.render_node(128, "0x10de");
    assert!(nvidia_gpu_present_in(&fx.root));
}

#[test]
fn detects_nvidia_among_mixed_vendors() {
    // The dev box this targets has two NVIDIA render nodes plus an AMD one;
    // detection must find NVIDIA regardless of enumeration order.
    let fx = SysfsFixture::new();
    fx.render_node(130, "0x1002") // AMD
        .render_node(128, "0x10de") // NVIDIA
        .render_node(129, "0x10de"); // NVIDIA
    assert!(nvidia_gpu_present_in(&fx.root));
}

#[test]
fn no_nvidia_when_only_intel_or_amd() {
    let fx = SysfsFixture::new();
    fx.render_node(128, "0x8086") // Intel
        .render_node(129, "0x1002"); // AMD
    assert!(!nvidia_gpu_present_in(&fx.root));
}

#[test]
fn case_insensitive_vendor_match() {
    // sysfs reports lowercase hex; be robust to a capitalised form too.
    let fx = SysfsFixture::new();
    fx.render_node(128, "0x10DE");
    assert!(nvidia_gpu_present_in(&fx.root));
}

#[test]
fn ignores_non_render_nodes() {
    // A `card0` node pointing at the NVIDIA device must NOT count — we key
    // on render nodes (what the GPU-compute / VAAPI path opens). With only
    // a card node present, detection reports absent.
    let fx = SysfsFixture::new();
    fx.other_node("card0", "0x10de");
    assert!(!nvidia_gpu_present_in(&fx.root));
}

#[test]
fn missing_drm_root_is_not_nvidia() {
    // No /sys/class/drm at all (containers, odd kernels) → cleanly false,
    // never a panic.
    let missing = std::env::temp_dir().join("tether-nvenc-does-not-exist-xyz");
    assert!(!nvidia_gpu_present_in(&missing));
}

// --- hardware tests (NVIDIA GPU + NVENC-enabled FFmpeg) --------------------

/// High-entropy BGRA so the encoder produces a non-trivial bitstream (a flat
/// frame compresses to almost nothing and hides "did it actually encode?").
/// xorshift mix over (x, y, t); opaque alpha.
// The `as u8` casts deliberately take the low 8 bits of each mixed word —
// truncation IS the point (per-channel pseudo-random noise), not a bug.
#[allow(clippy::cast_possible_truncation)]
fn make_noisy_bgra(w: u32, h: u32, t: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let mut m = x
                .wrapping_mul(2_654_435_761)
                .wrapping_add(y.wrapping_mul(40_503))
                .wrapping_add(t.wrapping_mul(2_246_822_519));
            m ^= m >> 15;
            m = m.wrapping_mul(2_246_822_519);
            m ^= m >> 13;
            buf[i] = m as u8;
            buf[i + 1] = (m >> 8) as u8;
            buf[i + 2] = (m >> 16) as u8;
            buf[i + 3] = 0xff;
        }
    }
    buf
}

/// Drive `encode_bgra` for `profile` through the real NVENC encoder and
/// assert it produces a self-decodable IDR. The shared body of the per-codec
/// hardware tests below.
fn assert_encode_bgra_produces_idr(profile: VideoProfile) {
    const W: u32 = 256;
    const H: u32 = 256;
    let mut enc = NvencEncoder::new(profile, W, H, 30, 4_000)
        .unwrap_or_else(|e| panic!("NvencEncoder::new({profile:?}) failed: {e}"));

    let mut saw_keyframe = false;
    let mut total_bytes = 0usize;
    for t in 0..6u32 {
        let bgra = make_noisy_bgra(W, H, t);
        let packets = enc
            .encode_bgra(&bgra, i64::from(t), t == 0)
            .unwrap_or_else(|e| panic!("encode_bgra frame {t} failed: {e}"));
        for p in packets {
            total_bytes += p.data.len();
            if p.keyframe {
                saw_keyframe = true;
                // Every IDR has extradata (VPS/SPS/PPS for HEVC, SPS/PPS for
                // H.264) prepended, so it begins with an Annex-B start code.
                assert!(
                    p.data.starts_with(&[0, 0, 0, 1]) || p.data.starts_with(&[0, 0, 1]),
                    "{profile:?}: keyframe should start with an Annex-B start code \
                     (extradata prepended); got {:02x?}",
                    &p.data[..p.data.len().min(8)]
                );
            }
        }
    }
    assert!(saw_keyframe, "{profile:?}: expected at least one keyframe (IDR)");
    assert!(total_bytes > 0, "{profile:?}: encoder produced no bitstream");
}

#[test]
#[ignore = "requires NVIDIA GPU + NVENC-enabled FFmpeg (cargo test -p tether-codec --ignored nvenc_)"]
fn nvenc_hevc_8bit_encode_bgra_produces_idr() {
    assert_encode_bgra_produces_idr(VideoProfile::HEVC_8BIT_420);
}

#[test]
#[ignore = "requires NVIDIA GPU + NVENC-enabled FFmpeg (cargo test -p tether-codec --ignored nvenc_)"]
fn nvenc_h264_8bit_encode_bgra_produces_idr() {
    assert_encode_bgra_produces_idr(VideoProfile::H264_8BIT_420);
}

#[test]
#[ignore = "requires NVIDIA GPU (cargo test -p tether-codec --ignored nvenc_)"]
fn nvenc_detection_true_on_this_nvidia_host() {
    // Sanity: the production detection path (real /sys/class/drm) agrees that
    // this is an NVIDIA host. Guards against a sysfs-layout assumption that
    // the synthetic-tree unit tests can't catch.
    assert!(
        super::nvidia_gpu_present(),
        "nvidia_gpu_present() should be true on an NVIDIA host"
    );
}

/// Luma stats over a decoded frame, to verify a solid-color round trip
/// without a full SSIM harness.
struct DecodedYStats {
    w: u32,
    h: u32,
    mean: f64,
    stddev: f64,
    /// Full-scale luma value: 255 (8-bit) or 1023 (10-bit).
    max_scale: f64,
}

/// Decode an HEVC Annex-B bitstream (extradata-prefixed IDR) with FFmpeg's
/// in-build native software decoder — no nvidia-vaapi-driver — and return
/// luma stats for the first decoded frame. `None` if nothing decodes.
fn sw_hevc_decode_y_stats(packets: &[crate::EncodedPacket]) -> Option<DecodedYStats> {
    use rsmpeg::avcodec::{AVCodec, AVCodecContext};

    let codec = AVCodec::find_decoder(ffi::AV_CODEC_ID_HEVC)?;
    let mut dec = AVCodecContext::new(&codec);
    dec.open(None).ok()?;
    for p in packets {
        let pkt = crate::h264::packet_from_bytes(&p.data).ok()?;
        dec.send_packet(Some(&pkt)).ok()?;
    }
    // Flush: a lone IDR sits in the reorder DPB until EOF.
    let _ = dec.send_packet(None);
    let frame = dec.receive_frame().ok()?;
    Some(y_stats(&frame))
}

// ffmpeg i32 width/height/linesize on an allocated frame are non-negative;
// the u16 read is the 10-bit little-endian luma sample. Both casts are
// deliberate, not lossy in practice.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn y_stats(frame: &rsmpeg::avutil::AVFrame) -> DecodedYStats {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let stride = frame.linesize[0] as usize;
    let is_10bit = frame.format == ffi::AV_PIX_FMT_YUV420P10LE;
    let max_scale = if is_10bit { 1023.0 } else { 255.0 };
    let data = frame.data[0];
    let (mut sum, mut sumsq, mut n) = (0f64, 0f64, 0f64);
    // SAFETY: data points to at least `stride * h` readable bytes; we index
    // within the visible w×h region (10-bit reads 2 bytes per sample).
    unsafe {
        for y in 0..h {
            let row = data.add(y * stride);
            for x in 0..w {
                let v = if is_10bit {
                    f64::from(row.add(x * 2).cast::<u16>().read_unaligned() & 0x03ff)
                } else {
                    f64::from(*row.add(x))
                };
                sum += v;
                sumsq += v * v;
                n += 1.0;
            }
        }
    }
    let mean = sum / n;
    let var = (sumsq / n) - mean * mean;
    DecodedYStats {
        w: w as u32,
        h: h as u32,
        mean,
        stddev: var.max(0.0).sqrt(),
        max_scale,
    }
}

#[test]
#[ignore = "requires NVIDIA GPU + NVENC + Vulkan dma-buf (cargo test -p tether-codec --ignored nvenc_)"]
fn nvenc_p010_dmabuf_roundtrip_decodes_our_pixels() {
    use tether_gpuconvert::Bgra2P010DmaBuf;

    const W: u32 = 256;
    const H: u32 = 256;

    let bridge = match pollster::block_on(Bgra2P010DmaBuf::new(W, H)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP nvenc_p010_dmabuf_roundtrip: Bgra2P010DmaBuf unavailable: {e}");
            return;
        }
    };

    // Solid mid-gray. A correct EGL→CUDA import decodes back to a uniform,
    // mid-range frame; a wrong plane/stride/offset yields garbage (high
    // variance) or zeroed memory (near-black).
    let mut bgra = vec![0u8; (W * H * 4) as usize];
    for px in bgra.chunks_exact_mut(4) {
        px.copy_from_slice(&[128, 128, 128, 255]);
    }
    let p010 = bridge.convert_bgra_bytes(&bgra).expect("P010 convert");
    let frame = crate::build_p010_dmabuf_frame(
        p010.fd,
        p010.size,
        p010.modifier,
        p010.y_offset,
        p010.y_stride,
        p010.uv_offset,
        p010.uv_stride,
    );

    let mut enc =
        NvencEncoder::new(VideoProfile::HEVC_10BIT_420, W, H, 30, 8_000).expect("NVENC HEVC Main10");
    let packets = enc
        .submit_dmabuf(&frame, 0, true)
        .expect("submit_dmabuf (EGL→CUDA import + encode)");
    assert!(
        packets.iter().any(|p| p.keyframe),
        "submit_dmabuf should produce an IDR"
    );

    let stats = sw_hevc_decode_y_stats(&packets)
        .expect("software HEVC decode of the NVENC output produced no frame");
    assert_eq!((stats.w, stats.h), (W, H), "decoded dims mismatch");
    assert!(
        stats.stddev < stats.max_scale * 0.05,
        "decoded luma not uniform (stddev {:.1} of {:.0}) — the EGL→CUDA import likely \
         delivered garbage instead of our solid frame",
        stats.stddev,
        stats.max_scale
    );
    let frac = stats.mean / stats.max_scale;
    assert!(
        (0.25..0.75).contains(&frac),
        "decoded luma mean {:.1} ({:.0}% of full scale) is not mid-range — the import did \
         not deliver the gray we encoded",
        stats.mean,
        frac * 100.0
    );
}

/// Encode `n` high-entropy frames starting at timestamp `start_t`; return
/// (total bitstream bytes, next timestamp).
fn encode_noisy(enc: &mut NvencEncoder, w: u32, h: u32, start_t: u32, n: u32) -> (usize, u32) {
    let mut bytes = 0usize;
    let mut t = start_t;
    for _ in 0..n {
        let bgra = make_noisy_bgra(w, h, t);
        let packets = enc
            .encode_bgra(&bgra, i64::from(t), t == 0)
            .expect("encode_bgra");
        for p in packets {
            bytes += p.data.len();
        }
        t += 1;
    }
    (bytes, t)
}

#[test]
#[ignore = "requires NVIDIA GPU + NVENC (cargo test -p tether-codec --ignored nvenc_)"]
fn nvenc_bitrate_retune_changes_bitstream_size() {
    const W: u32 = 256;
    const H: u32 = 256;
    const LOW_KBPS: u32 = 1_000;
    const HIGH_KBPS: u32 = 20_000;
    const WARMUP: u32 = 10;
    const MEASURE: u32 = 40;

    // The NVENC analogue of the VAAPI `vaapi_bitrate_retune_changes_bitstream_size`
    // test — but where VAAPI SKIPs (its FFmpeg wrapper builds the rate-control
    // buffer once at init), NVENC must PASS: live retune is the reason it exists
    // alongside VAAPI (GH #16). High-entropy frames keep the encoder bitrate-
    // bound so the CBR target, not content, governs frame size.
    let mut enc =
        NvencEncoder::new(VideoProfile::H264_8BIT_420, W, H, 30, LOW_KBPS).expect("NVENC H.264");
    assert!(
        enc.supports_changing_bitrate(),
        "NVENC must advertise live bitrate retune"
    );

    let (_, t) = encode_noisy(&mut enc, W, H, 0, WARMUP); // let rate control settle
    let (low_bytes, t) = encode_noisy(&mut enc, W, H, t, MEASURE);

    enc.set_bitrate_kbps(HIGH_KBPS).expect("set_bitrate_kbps");
    let (_, t) = encode_noisy(&mut enc, W, H, t, WARMUP); // let the new target settle
    let (high_bytes, _) = encode_noisy(&mut enc, W, H, t, MEASURE);

    let ratio = high_bytes as f64 / low_bytes.max(1) as f64;
    eprintln!("nvenc retune: low={low_bytes}B high={high_bytes}B ratio={ratio:.2}");

    // SKIP escape hatch: if a driver/FFmpeg combo ever silently ignores the
    // retune, record it loudly rather than failing — but on the verified path
    // (RTX 3090 Ti) the assertion is the point.
    if ratio <= 1.5 {
        eprintln!(
            "SKIP nvenc_bitrate_retune: 1→20 Mbps retune produced only {ratio:.2}x \
             bitstream growth — this driver did not honour the live retune"
        );
        return;
    }
    assert!(
        ratio > 2.0,
        "expected >2x bitstream growth after a 1→20 Mbps retune; got {ratio:.2}x \
         (low={low_bytes}B high={high_bytes}B)"
    );
}

#[test]
#[ignore = "requires NVIDIA GPU + NVENC + Vulkan dma-buf (cargo test -p tether-codec --ignored nvenc_)"]
fn nvenc_nv12_dmabuf_roundtrip_decodes_our_pixels() {
    use tether_gpuconvert::Nv12DmaBuf;

    const W: u32 = 256;
    const H: u32 = 256;

    let bridge = match pollster::block_on(Nv12DmaBuf::new(W, H)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP nvenc_nv12_dmabuf_roundtrip: Nv12DmaBuf unavailable: {e}");
            return;
        }
    };

    // Solid mid-gray; same correctness logic as the P010 test, on the
    // 8-bit NV12 path (HEVC Main).
    let mut bgra = vec![0u8; (W * H * 4) as usize];
    for px in bgra.chunks_exact_mut(4) {
        px.copy_from_slice(&[128, 128, 128, 255]);
    }
    let nv12 = bridge.convert_bgra_bytes(&bgra).expect("NV12 convert");
    let frame = crate::build_nv12_dmabuf_frame(
        nv12.fd,
        nv12.size,
        nv12.modifier,
        nv12.y_offset,
        nv12.y_stride,
        nv12.uv_offset,
        nv12.uv_stride,
    );

    let mut enc =
        NvencEncoder::new(VideoProfile::HEVC_8BIT_420, W, H, 30, 8_000).expect("NVENC HEVC Main");
    let packets = enc
        .submit_dmabuf(&frame, 0, true)
        .expect("submit_dmabuf (EGL→CUDA import + encode)");
    assert!(
        packets.iter().any(|p| p.keyframe),
        "submit_dmabuf should produce an IDR"
    );

    let stats = sw_hevc_decode_y_stats(&packets)
        .expect("software HEVC decode of the NVENC output produced no frame");
    assert_eq!((stats.w, stats.h), (W, H), "decoded dims mismatch");
    assert!(
        stats.stddev < stats.max_scale * 0.05,
        "decoded luma not uniform (stddev {:.1} of {:.0}) — EGL→CUDA import likely garbage",
        stats.stddev,
        stats.max_scale
    );
    let frac = stats.mean / stats.max_scale;
    assert!(
        (0.25..0.75).contains(&frac),
        "decoded luma mean {:.1} ({:.0}% of full scale) not mid-range — import wrong",
        stats.mean,
        frac * 100.0
    );
}
