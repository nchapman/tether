//! Smoke test for tether-render: opens a window and streams an animated
//! grayscale Y-plane (no chroma) at ~60 FPS. Run with
//! `cargo run --example test_pattern -p tether-render`.

use std::time::Duration;

use crossbeam_channel::bounded;
use tether_render::{CpuFrame, Frame};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let (tx, rx) = bounded::<Frame>(2);

    std::thread::spawn(move || {
        let (w, h) = (800u32, 600u32);
        let mut t: u32 = 0;
        loop {
            let frame = test_pattern(w, h, t);
            t = t.wrapping_add(1);
            if tx.send(frame).is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(16));
        }
    });

    tether_render::run("tether-render test pattern", (800, 600), rx, None)?;
    Ok(())
}

// Animated diagonal gradient. Y carries the luminance signal; U+V are
// pinned to chroma-neutral (128) so the output is greyscale and we
// don't have to do a full RGB→YUV conversion just to demo the pipeline.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn test_pattern(width: u32, height: u32, t: u32) -> Frame {
    let mut y = Vec::with_capacity((width * height) as usize);
    let phase = (t as f32 * 0.05).sin() * 64.0;
    for j in 0..height {
        for i in 0..width {
            let base = ((i + j) * 255 / (width + height).max(1)) as f32;
            let v = (base + phase).clamp(16.0, 235.0) as u8;
            y.push(v);
        }
    }
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    // Neutral grey chroma in NV12 layout: U=128, V=128 interleaved.
    let uv = vec![128u8; (cw * ch * 2) as usize];
    Frame::Cpu(CpuFrame {
        width,
        height,
        y,
        uv,
        t_capture_client_clock: None,
    })
}
