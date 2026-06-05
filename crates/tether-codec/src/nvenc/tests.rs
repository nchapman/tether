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
