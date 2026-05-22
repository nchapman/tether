//! Mitchell-Netravali bicubic image scaler in linear-light.
//!
//! Used by both the host (downscale BGRA capture → encoder input at the
//! client's viewport dimensions) and the client (upscale decoded video
//! → window dimensions when the window is larger than the encoded
//! stream). The same WGSL compute shader runs on both ends; the only
//! per-call configuration is the source/destination dimensions.
//!
//! ## Why a separate crate
//!
//! Both [`tether-gpuconvert`](../tether_gpuconvert) (host downscale) and
//! [`tether-render`](../tether_render) (client upscale) need this. A
//! shared crate keeps the WGSL source single-of-truth and lets the
//! quality verification (CPU reference, SSIM/PSNR tests) live next to
//! the implementation it covers.
//!
//! ## Quality story
//!
//! Three things make this measurably better than the bilinear scaling
//! that every comparable project (Sunshine, Apollo, Moonlight,
//! rustdesk) ships with:
//!
//! 1. **Mitchell-Netravali bicubic** with `B = C = 1/3` — the canonical
//!    "balanced" cubic that ImageMagick and fontconfig default to for
//!    screen content. Negative-lobe weights at `|x| ∈ [1, 2]` provide
//!    edge sharpening without the ringing artifacts Lanczos produces
//!    on hard edges (text, window borders).
//! 2. **Linear-light filtering.** sRGB → linear decode on read, filter
//!    in linear, encode back to sRGB on write — using the IEC
//!    61966-2-1 piecewise transfer function, not the `pow(c, 2.2)`
//!    approximation. The approximation is wrong enough in the dark
//!    pixel regime where text antialiasing lives that it would undo the
//!    quality win.
//! 3. **Scale-aware tap count + mipmap prefilter.** A fixed 4-tap
//!    kernel aliases visibly at downscale ratios > 2×. The shader
//!    widens its kernel and its tap count with the scale ratio; for
//!    ratios > 2× the [`Scaler`] runs a box-filter mipmap pass first so
//!    Mitchell's input is always within 2× of its output.
//!
//! Quality is verified as a code property: the CPU reference in
//! [`reference`] is the spec, and `tests/hardware.rs` asserts the
//! shader matches it within an fp16-calibrated SSIM/PSNR bar.

mod pipeline;
pub mod reference;
mod scaler;

pub use scaler::Scaler;

#[derive(Debug, thiserror::Error)]
pub enum ScalerError {
    /// One or both source/destination dimensions were zero.
    #[error("source or destination dimension is zero (src={src:?}, dst={dst:?})")]
    ZeroDim { src: (u32, u32), dst: (u32, u32) },

    /// Destination is `>=` source in both axes. Host callers treat
    /// this as "skip the scaler entirely" (we only downscale on the
    /// host); client callers don't see it (they only construct a
    /// scaler when window > video, i.e. upscale, which is *not* this
    /// case).
    #[error("destination >= source in both axes; no scaling required")]
    NoScaleNeeded,
}
