//! Shared cross-platform colour-validation fixture + assertion.
//!
//! Every platform's round-trip harness (Linux `dmabuf_test`, macOS
//! `iosurface_test`, Windows `d3d11`) drives the **same** four-bar
//! pattern — red / green / blue / white — through its production
//! encode/decode/render chain and asserts the bars survive with the
//! right colour. This is the colour-decode coverage the geometric /
//! SSIM metrics miss: `CoordEncoded` has near-constant chroma, so a
//! hue cast or Cb/Cr swap doesn't move it.
//!
//! What each bar catches:
//!   * **white** — a hue cast on neutrals (a wrong chroma neutral point
//!     renders white as e.g. purple — the live "colors are wrong" bug);
//!   * **red ↔ blue** — a Cb/Cr swap;
//!   * **green** — a dropped or inverted chroma channel.

/// BGRA colour-bar fixture at `dims`: four equal vertical bars
/// red / green / blue / white (BGRA byte order, opaque alpha). The
/// procedural source for the platforms that encode in-test (Linux
/// VAAPI, Windows QSV, macOS VT 4:2:0). macOS 4:4:4 can't encode
/// locally, so it decodes a committed bitstream of the same pattern
/// (`fixtures/colorbars_hevc_yuv444_*.idr`) instead.
///
/// Values are near-but-not-full saturation (235 / 0) so an encoder's
/// limited-range clamp doesn't have to handle super-white / -black
/// excursions — the bars stay unambiguously their colour after the
/// BT.709 limited round-trip.
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub(crate) fn colorbars_bgra(dims: (u32, u32)) -> Vec<u8> {
    let (w, h) = dims;
    let bar = (w / 4).max(1);
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for _y in 0..h {
        for x in 0..w {
            let (r, g, b) = match (x / bar).min(3) {
                0 => (235u8, 0, 0),
                1 => (0, 235, 0),
                2 => (0, 0, 235),
                _ => (235, 235, 235),
            };
            out.extend_from_slice(&[b, g, r, 255]);
        }
    }
    out
}

/// Channel order of a readback buffer. macOS reads back an
/// `Rgba8Unorm` offscreen target; the Linux/Windows production
/// renderers read back `Bgra8UnormSrgb` (matching the live swapchain).
/// Each platform uses exactly one variant, so the other is dead on any
/// single-target build — allow it rather than cfg-gate per platform.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) enum ChannelOrder {
    Rgba,
    Bgra,
}

/// Average (R, G, B) over a rectangle, honouring the buffer's channel
/// order. Shared by the colour-bar assertion and the macOS 4:2:0
/// left/right region check.
// The per-channel results are means of u8 values, so each is ≤ 255 and
// fits in u8 by construction — the final casts can't truncate.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn region_average_rgb(
    px: &[u8],
    width: u32,
    order: ChannelOrder,
    x0: u32,
    y0: u32,
    rw: u32,
    rh: u32,
) -> (u8, u8, u8) {
    let (ri, gi, bi) = match order {
        ChannelOrder::Rgba => (0usize, 1, 2),
        ChannelOrder::Bgra => (2usize, 1, 0),
    };
    let mut sum = [0u64; 3];
    let mut count = 0u64;
    for y in y0..y0 + rh {
        for x in x0..x0 + rw {
            let idx = ((y * width + x) * 4) as usize;
            sum[0] += u64::from(px[idx + ri]);
            sum[1] += u64::from(px[idx + gi]);
            sum[2] += u64::from(px[idx + bi]);
            count += 1;
        }
    }
    let count = count.max(1);
    (
        (sum[0] / count) as u8,
        (sum[1] / count) as u8,
        (sum[2] / count) as u8,
    )
}

/// Average RGB of the centre of colour-bar `index` (0..4): four equal
/// vertical bars. Samples the middle half of the bar to stay clear of
/// chroma bleed at the bar boundaries — resolution-agnostic.
fn bar_rgb(px: &[u8], width: u32, height: u32, order: ChannelOrder, index: u32) -> (u8, u8, u8) {
    let bar_w = width / 4;
    region_average_rgb(
        px,
        width,
        order,
        index * bar_w + bar_w / 4,
        height / 4,
        bar_w / 2,
        height / 2,
    )
}

/// Assert the four bars reconstruct to red / green / blue / white.
///
/// Generous per-channel bounds (an HEVC + BT.709 limited-range
/// round-trip can shift a channel by ~25, and the readback is sRGB- or
/// linear-encoded depending on platform) but tight enough to catch a
/// hue cast (white bar), a Cb/Cr swap (red↔blue), or a lost chroma
/// channel (green). Works for both the macOS linear `Rgba8Unorm`
/// readback and the Linux/Windows sRGB `Bgra8UnormSrgb` readback —
/// saturated primaries and white land high/low in either encoding.
pub(crate) fn assert_colorbars(
    label: &str,
    px: &[u8],
    width: u32,
    height: u32,
    order: ChannelOrder,
) {
    let red = bar_rgb(px, width, height, order, 0);
    let green = bar_rgb(px, width, height, order, 1);
    let blue = bar_rgb(px, width, height, order, 2);
    let white = bar_rgb(px, width, height, order, 3);
    eprintln!("{label}: red={red:?} green={green:?} blue={blue:?} white={white:?}");
    assert!(
        red.0 > 130 && red.1 < 80 && red.2 < 80,
        "{label}: bar 0 should be red; got {red:?}"
    );
    assert!(
        green.1 > 110 && green.0 < 90 && green.2 < 90,
        "{label}: bar 1 should be green; got {green:?}"
    );
    assert!(
        blue.2 > 130 && blue.0 < 80 && blue.1 < 80,
        "{label}: bar 2 should be blue; got {blue:?}"
    );
    let (wr, wg, wb) = white;
    let wmin = wr.min(wg).min(wb);
    let wspread = wr.max(wg).max(wb) - wmin;
    assert!(
        wmin > 150 && wspread < 45,
        "{label}: bar 3 should be neutral white (R≈G≈B, all high) — a hue cast here is the \
         'colors are wrong' bug; got {white:?}"
    );
}
