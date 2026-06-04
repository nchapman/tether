//! macOS input injection via `CGEventPost` (Quartz Event Services).
//!
//! enigo's macOS backend wraps `CGEventCreateMouseEvent` /
//! `CGEventCreateKeyboardEvent` / `CGEventCreateScrollWheelEvent` and
//! posts via `CGEventPost(kCGHIDEventTap, …)`. All the input-shaping
//! logic — modifier reconciliation, HID→`Key` mapping, held-key
//! cleanup on disconnect — is shared with the Linux (libei) backend
//! through [`super::enigo_backend::EnigoBackend`].
//!
//! Permission: the first event injection triggers the macOS
//! Accessibility TCC prompt. The host must be granted "Accessibility"
//! (not "Input Monitoring" — that controls *reading*, not posting).
//! enigo doesn't preflight; if the user denies, posts silently no-op
//! until accessibility is granted and the host is restarted. Same
//! first-run shape as the screen-capture prompt.

use enigo::{Enigo, Settings};

use tether_protocol::cursor::ClientCursorPacket;
use tether_protocol::input::InputEvent;

use super::enigo_backend::EnigoBackend;
use super::{InjectError, Injector, Result};

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    /// The user's main display ID. Cursor injection pins to the primary
    /// display, matching the capture path (which also captures primary).
    fn CGMainDisplayID() -> u32;
    /// Width of the display in the *logical-point* coordinate system of
    /// the active (HiDPI-scaled) mode — NOT backing pixels. This is the
    /// exact space `CGEvent` mouse warps consume, so it's what the
    /// injector must denormalize against. (See the matching note in
    /// `tether-capture`'s `macos.rs`: `CGDisplayPixelsWide`/`High`
    /// return point dims on a Retina display, while
    /// `CGDisplayModeGetPixelWidth` returns backing pixels.)
    fn CGDisplayPixelsWide(display: u32) -> usize;
    /// Height counterpart of [`CGDisplayPixelsWide`], in logical points.
    fn CGDisplayPixelsHigh(display: u32) -> usize;
}

/// Query the main display's logical-point dimensions — the coordinate
/// space `CGEvent` mouse events live in. Returns `None` if the query
/// yields a degenerate 0 dimension (e.g. the display dropped off
/// between calls), so the caller can fall back to capture-pixel dims.
fn main_display_point_dims() -> Option<(u32, u32)> {
    // SAFETY: both calls are pure reads of the current display config
    // and transfer no ownership. In a headless context (no monitor, or
    // a display-sleep race) CGMainDisplayID may return
    // kCGNullDirectDisplayID (0); CGDisplayPixelsWide(0) is well-defined
    // and returns 0, which the degenerate check below converts to None.
    let (w, h) = unsafe {
        let id = CGMainDisplayID();
        (CGDisplayPixelsWide(id), CGDisplayPixelsHigh(id))
    };
    let w = u32::try_from(w).ok()?;
    let h = u32::try_from(h).ok()?;
    (w > 0 && h > 0).then_some((w, h))
}

/// Pick the dimensions the injector denormalizes client `[0,1]` cursor
/// coords against. `capture_px` is the capture-pixel size the pipeline
/// reported (backing pixels on a Retina display); `queried_points` is
/// the display's logical-point size from [`main_display_point_dims`].
/// Prefers the point dims — the space `CGEvent` consumes — and falls
/// back to capture pixels when the point query is unavailable. The
/// non-zero guard is independent of [`main_display_point_dims`] (which
/// already filters zeros): keeping it here means this decision point is
/// safe against a degenerate `Some` from any source, and lets the unit
/// test pin that behaviour directly. Pure so the points-vs-pixels
/// selection is unit-testable.
fn resolve_injection_dims(
    capture_px: (u32, u32),
    queried_points: Option<(u32, u32)>,
) -> (u32, u32) {
    match queried_points {
        Some((w, h)) if w > 0 && h > 0 => (w, h),
        _ => capture_px,
    }
}

pub struct MacOsInjector {
    inner: EnigoBackend,
}

impl MacOsInjector {
    /// Construct the CGEvent-backed injector. Unlike the libei path
    /// there's no portal handshake — enigo opens the system event
    /// source synchronously. We still go through `spawn_blocking` for
    /// symmetry and to keep the async signature.
    pub async fn connect() -> Result<Self> {
        let settings = Settings::default();
        let enigo = tokio::task::spawn_blocking(move || Enigo::new(&settings))
            .await
            .map_err(|e| InjectError::Init(format!("spawn_blocking join: {e}")))?
            .map_err(|e| InjectError::Init(format!("enigo new: {e:?}")))?;
        // CGEvent construction succeeds regardless of the Accessibility
        // TCC grant — `CGEventPost` silently no-ops when the grant is
        // missing rather than returning an error, so an operator who
        // forgot to grant Accessibility sees a happy `connect()` and
        // events that go nowhere. Make the grant a load-bearing
        // post-condition by logging it at info every time, not just on
        // failure.
        tracing::info!(
            session = "macos (CGEvent)",
            "macOS input backend selected. If injected events do not appear, \
             verify the host has Accessibility access in \
             System Settings → Privacy & Security → Accessibility."
        );
        Ok(Self {
            inner: EnigoBackend::new(enigo),
        })
    }
}

impl Injector for MacOsInjector {
    fn inject(&mut self, evt: &InputEvent) -> Result<()> {
        self.inner.inject(evt)
    }
    fn inject_cursor(&mut self, cursor: &ClientCursorPacket) -> Result<()> {
        self.inner.inject_cursor(cursor)
    }
    fn set_display_size(&mut self, width: u32, height: u32) {
        // The capture pipeline reports capture-*pixel* dims here (backing
        // pixels — 2× the logical size on a Retina display). But the
        // enigo macOS backend posts cursor warps straight to `CGEvent`,
        // whose global coordinate space is in *points*. Denormalizing
        // against backing pixels would warp the host cursor by the
        // backing scale factor, leaving the rendered overlay hundreds of
        // points away from the user's local pointer. Substitute the main
        // display's logical-point dims (the actual `CGEvent` space),
        // falling back to the reported capture pixels only if the query
        // fails.
        //
        // ASSUMPTION: CGMainDisplayID is the display the capture path
        // selected. `tether-capture::macos::start` captures
        // `SCShareableContent::displays().next()`, which is the main
        // display today. If capture ever supports a non-primary display,
        // this query must use that same display_id — otherwise the point
        // substitution is only correct when both displays share a logical
        // resolution.
        let (w, h) = resolve_injection_dims((width, height), main_display_point_dims());
        self.inner.set_display_size(w, h);
    }
}

#[cfg(test)]
mod tests {
    use super::{main_display_point_dims, resolve_injection_dims};

    #[test]
    fn prefers_queried_point_dims_over_capture_pixels() {
        // Retina host: capture reports backing pixels (3840×2160) but
        // CGEvent wants logical points (1920×1080). The point query must
        // win — otherwise the injected host cursor lands ~2× off and the
        // overlay drifts hundreds of points from the local pointer.
        assert_eq!(
            resolve_injection_dims((3840, 2160), Some((1920, 1080))),
            (1920, 1080)
        );
    }

    #[test]
    fn non_retina_query_equals_capture_and_is_a_no_op() {
        // 1× display: points == pixels, so substitution changes nothing.
        assert_eq!(
            resolve_injection_dims((2560, 1440), Some((2560, 1440))),
            (2560, 1440)
        );
    }

    #[test]
    fn falls_back_to_capture_pixels_on_degenerate_query() {
        // A 0-dimension query (display dropped off mid-read) is unusable;
        // fall back rather than denormalize into a zero-width space.
        assert_eq!(resolve_injection_dims((2560, 1440), None), (2560, 1440));
        assert_eq!(
            resolve_injection_dims((2560, 1440), Some((0, 1440))),
            (2560, 1440)
        );
        assert_eq!(
            resolve_injection_dims((2560, 1440), Some((2560, 0))),
            (2560, 1440)
        );
    }

    #[test]
    #[ignore = "requires a live macOS main display"]
    fn main_display_query_returns_sane_point_dims() {
        // Sanity anchor for the real query: on any Mac with a display it
        // yields non-zero point dims. Logical points are never larger
        // than backing pixels (scale ≥ 1), but we can't read the backing
        // pixels here without the capture path, so we only assert
        // non-degeneracy. Ignored by default — CI/headless runners may
        // lack a main display.
        // `expect` is the real assertion — `main_display_point_dims`
        // returns None on degenerate dims, so a Some is non-zero by
        // contract and the bounds check below is vacuously true (kept as
        // executable documentation of the post-condition).
        let dims = main_display_point_dims().expect("main display has non-zero point dims");
        assert!(dims.0 > 0 && dims.1 > 0, "degenerate point dims: {dims:?}");
    }
}
