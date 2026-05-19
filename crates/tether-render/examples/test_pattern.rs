//! Smoke test for tether-render: opens a window and streams an animated
//! gradient at ~60 FPS. Run with `cargo run --example test_pattern -p tether-render`.

use std::time::Duration;

use crossbeam_channel::bounded;
use tether_render::RawFrame;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let (tx, rx) = bounded::<RawFrame>(2);

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

// All `as u8` casts in this fn are intentional truncations in math whose
// outputs are bounded to [0, 255] by construction. clippy can't prove that.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn test_pattern(width: u32, height: u32, t: u32) -> RawFrame {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    let phase_b = (t as f32 * 0.05).sin() * 0.5 + 0.5;
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = (phase_b * 255.0) as u8;
            data.extend_from_slice(&[r, g, b, 255]);
        }
    }
    RawFrame {
        width,
        height,
        data,
    }
}
