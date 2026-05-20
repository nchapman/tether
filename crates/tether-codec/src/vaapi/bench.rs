//! Hardware-backed VAAPI benchmarks. Run with:
//!
//! ```text
//! cargo test -p tether-codec --lib bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` is load-bearing for readable output: parallel
//! cells interleave their stdout lines into nonsense. It also keeps the
//! cells from competing for the same VAAPI device, which would make the
//! per-iteration timing meaningless.
//!
//! Each `#[test] #[ignore]` entry covers one (codec, resolution) cell
//! and exercises three layers: `encode_bgra` (CPU-upload path, what the
//! test pattern uses), `encode_dmabuf` (zero-copy production path), and
//! decode (submit + drain). Output is a uniform `p50 / p99 / max ms`
//! line per scenario so you can scan the table by eye.
//!
//! Bitrates and per-frame budgets are anchored at 60 fps:
//! - 1080p60: 16.6 ms/frame budget, 8 Mbps target
//! - 4K60:    16.6 ms/frame budget, 25 Mbps target (HEVC: × 0.7 in the
//!            host's `derive_bitrate_kbps`; we use the upper number
//!            here to keep the benches comparable across codecs)
//!
//! Source for `encode_dmabuf`: run one frame through VAAPI encode →
//! VAAPI decode → DMA-BUF export to get a real surface at the right
//! dimensions. The encoder under test then re-imports that DMA-BUF for
//! every iteration (ffmpeg dups the fds, so one source surface fans
//! out across the whole bench).
//!
//! Decode bitstream source: VAAPI encode of a generated BGRA pattern
//! at the same dims/codec we're decoding. The encoder is fresh each
//! bench so the bitstream characteristics are deterministic, modulo
//! the encoder's own non-determinism (rate-control state).

use std::time::Instant;

use tether_protocol::control::CodecKind;

use crate::{Decoder, Encoder, Frame, GpuFrame, GpuFrameSource};

use super::{VaapiDecoder, VaapiEncoder};

/// Sorted-percentile summary of a bench loop. Returns `(p50, p99, max)`
/// in milliseconds. Caller is expected to print; this helper just
/// computes.
fn percentiles_ms(mut times_ms: Vec<f64>) -> (f64, f64, f64) {
    times_ms.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in wall-clock measurements"));
    let p50 = times_ms[times_ms.len() / 2];
    let p99 = times_ms[(times_ms.len() * 99) / 100];
    let max = *times_ms.last().expect("at least one iteration");
    (p50, p99, max)
}

/// Print one bench line in the standard format.
fn print_result(label: &str, p50: f64, p99: f64, max: f64, budget_ms: f64) {
    let headroom = budget_ms - p99;
    println!(
        "  {label:<32} p50={p50:>6.2} ms  p99={p99:>6.2} ms  max={max:>6.2} ms  \
         (budget={budget_ms:>5.2} ms, p99 headroom={headroom:+6.2} ms)"
    );
}

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

/// Encode-from-BGRA microbenchmark. Includes BGRA→NV12 swscale and the
/// hwframe_transfer_data upload — i.e. the work the test-pattern path
/// pays and the zero-copy DMA-BUF path skips.
fn bench_encode_bgra(
    codec: CodecKind,
    width: u32,
    height: u32,
    fps: u32,
    bitrate_kbps: u32,
    iters: usize,
    budget_ms: f64,
) {
    let mut enc = match VaapiEncoder::new(codec, width, height, fps, bitrate_kbps) {
        Ok(e) => e,
        Err(e) => {
            println!("  encode_bgra:   SKIP ({codec:?} @ {width}x{height}: {e})");
            return;
        }
    };
    let bgra = make_test_bgra(width, height, 0);
    // Warm up — first frame includes the hwframes pool prealloc and the
    // codec's SPS/PPS buffering, neither of which represents the
    // steady-state cost we're trying to measure.
    let _ = enc.encode_bgra(&bgra, 0, true);

    let mut times_ms = Vec::with_capacity(iters);
    for t in 1..=iters as i64 {
        let start = Instant::now();
        let _ = enc.encode_bgra(&bgra, t, false);
        times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let (p50, p99, max) = percentiles_ms(times_ms);
    print_result("encode_bgra", p50, p99, max, budget_ms);
}

/// Encode-from-DMA-BUF microbenchmark — the production hot path on
/// Linux (PipeWire DMA-BUF → tether-gpuconvert NV12 DMA-BUF → encoder).
/// We synthesise the source DMA-BUF by running one frame through SW
/// encode → VAAPI decode → DMA-BUF export, then re-import it on every
/// iteration. ffmpeg's `av_hwframe_map` dups the fds each call, so the
/// single source surface is safe to reuse.
fn bench_encode_dmabuf(
    codec: CodecKind,
    width: u32,
    height: u32,
    fps: u32,
    bitrate_kbps: u32,
    iters: usize,
    budget_ms: f64,
) {
    let dmabuf_holder = match build_dmabuf_source(width, height) {
        Ok(d) => d,
        Err(msg) => {
            println!("  encode_dmabuf: SKIP ({msg})");
            return;
        }
    };
    let GpuFrameSource::DmaBuf(ref dmabuf) = dmabuf_holder.source;

    let mut enc = match VaapiEncoder::new(codec, width, height, fps, bitrate_kbps) {
        Ok(e) => e,
        Err(e) => {
            println!("  encode_dmabuf: SKIP ({codec:?} @ {width}x{height}: {e})");
            return;
        }
    };
    // Warm-up frame (first import allocates encoder hwframes pool, etc.).
    let _ = enc.submit_dmabuf(dmabuf, 0, true);

    let mut times_ms = Vec::with_capacity(iters);
    for t in 1..=iters as i64 {
        let start = Instant::now();
        let _ = enc.submit_dmabuf(dmabuf, t, false);
        times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let (p50, p99, max) = percentiles_ms(times_ms);
    print_result("encode_dmabuf", p50, p99, max, budget_ms);
}

/// Decode microbenchmark. Pre-builds a bitstream by VAAPI-encoding
/// `iters + warmup` frames at the bench's dims/codec, then measures
/// `submit` + drain per packet. The drain loop catches whatever the
/// decoder emits (including zero frames during warm-up), so the
/// reported time is the steady-state per-frame cost.
fn bench_decode(
    codec: CodecKind,
    width: u32,
    height: u32,
    fps: u32,
    bitrate_kbps: u32,
    iters: usize,
    budget_ms: f64,
) {
    // Build the input bitstream first. We share one encoder; if it
    // fails to construct, the decoder bench has nothing to feed and
    // we skip cleanly.
    let mut enc = match VaapiEncoder::new(codec, width, height, fps, bitrate_kbps) {
        Ok(e) => e,
        Err(e) => {
            println!("  decode:        SKIP (cannot build {codec:?} encoder for bitstream: {e})");
            return;
        }
    };
    let warmup = 4i64;
    let total_frames = warmup + iters as i64;
    let mut packets: Vec<Vec<u8>> = Vec::with_capacity(total_frames as usize);
    for t in 0..total_frames {
        let bgra = make_test_bgra(width, height, t as u32);
        match enc.encode_bgra(&bgra, t, t == 0) {
            Ok(out) => {
                for p in out {
                    packets.push(p.data);
                }
            }
            Err(e) => {
                println!("  decode:        SKIP (bitstream build failed at frame {t}: {e})");
                return;
            }
        }
    }
    drop(enc);

    let mut dec = match VaapiDecoder::new(codec) {
        Ok(d) => d,
        Err(e) => {
            println!("  decode:        SKIP ({codec:?} @ {width}x{height}: {e})");
            return;
        }
    };

    // Drain the first `warmup` packets so the decoder is past
    // SPS/PPS buffering; only measure from there on.
    let mut times_ms = Vec::with_capacity(iters);
    for (i, pkt) in packets.iter().enumerate() {
        let start = Instant::now();
        if dec.submit(pkt).is_err() {
            continue;
        }
        while let Ok(Some(_)) = dec.next_frame() {
            // drop the frame; we just want timing of submit + drain.
        }
        if i >= warmup as usize {
            times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    if times_ms.is_empty() {
        println!("  decode:        SKIP (no timed iterations after warmup)");
        return;
    }
    let (p50, p99, max) = percentiles_ms(times_ms);
    print_result("decode", p50, p99, max, budget_ms);
}

/// Produce one decoded `GpuFrame` carrying a DMA-BUF descriptor at the
/// given dimensions. We pump VAAPI encode → VAAPI decode (both H.264
/// to keep the source path cheap regardless of which codec the bench
/// is measuring) and return the first GPU frame that lands. Caller
/// keeps the returned `GpuFrame` alive for the duration of the bench;
/// dropping it releases the VA surface back to the source decoder's
/// pool.
fn build_dmabuf_source(width: u32, height: u32) -> Result<GpuFrame, String> {
    let mut src_enc = VaapiEncoder::new(CodecKind::H264, width, height, 60, 8_000)
        .map_err(|e| format!("source encoder failed: {e}"))?;
    let mut src_dec = VaapiDecoder::new(CodecKind::H264)
        .map_err(|e| format!("source decoder failed: {e}"))?;

    for t in 0..8i64 {
        let bgra = make_test_bgra(width, height, t as u32);
        let packets = src_enc
            .encode_bgra(&bgra, t, t == 0)
            .map_err(|e| format!("source encode at frame {t} failed: {e}"))?;
        for p in packets {
            src_dec
                .submit(&p.data)
                .map_err(|e| format!("source decoder submit failed: {e}"))?;
            while let Some(f) = src_dec
                .next_frame()
                .map_err(|e| format!("source decoder next_frame failed: {e}"))?
            {
                if let Frame::Gpu(g) = f {
                    return Ok(g);
                }
            }
        }
    }
    Err("source pipeline produced no GPU frame within 8 input frames".to_string())
}

/// Drive all three paths for one (codec, resolution) cell.
fn run_cell(codec: CodecKind, width: u32, height: u32, fps: u32, bitrate_kbps: u32) {
    let iters = 60;
    let budget_ms = 1_000.0 / f64::from(fps);
    println!(
        "\n=== {codec:?} @ {width}x{height} {fps}fps  bitrate={bitrate_kbps}kbps  budget={budget_ms:.2}ms/frame ==="
    );
    bench_encode_bgra(codec, width, height, fps, bitrate_kbps, iters, budget_ms);
    bench_encode_dmabuf(codec, width, height, fps, bitrate_kbps, iters, budget_ms);
    bench_decode(codec, width, height, fps, bitrate_kbps, iters, budget_ms);
}

#[test]
#[ignore = "VAAPI benchmark (1080p60 H.264). Run: cargo test -p tether-codec --lib bench -- --ignored --nocapture"]
fn bench_h264_1080p60() {
    run_cell(CodecKind::H264, 1920, 1080, 60, 8_000);
}

#[test]
#[ignore = "VAAPI benchmark (1080p60 HEVC)."]
fn bench_hevc_1080p60() {
    run_cell(CodecKind::Hevc, 1920, 1080, 60, 8_000);
}

#[test]
#[ignore = "VAAPI benchmark (4K60 H.264)."]
fn bench_h264_4k60() {
    run_cell(CodecKind::H264, 3840, 2160, 60, 25_000);
}

#[test]
#[ignore = "VAAPI benchmark (4K60 HEVC)."]
fn bench_hevc_4k60() {
    run_cell(CodecKind::Hevc, 3840, 2160, 60, 25_000);
}
