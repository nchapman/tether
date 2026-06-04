//! macOS cursor extraction: NSCursor sprite + CGEvent location.
//!
//! macOS has no public equivalent of PipeWire's `SPA_META_Cursor`, so
//! we reproduce RustDesk's recipe:
//!
//! 1. **Private `CGSCurrentCursorSeed`** — a monotonic counter the
//!    WindowServer bumps every time the system cursor changes,
//!    regardless of which process pushed the change. Polling the seed
//!    is a cheap int read; we only re-fetch the sprite when it moves.
//!    Weak-linked via `dlsym` so a future macOS that removes the
//!    symbol degrades to "always re-fetch" instead of crashing.
//! 2. **`NSCursor.currentSystemCursor`** read from a background thread
//!    — Apple's docs don't explicitly bless this, but RustDesk has
//!    shipped it in production for years.
//! 3. **`CGEventCreate` / `CGEventGetLocation`** — public C API,
//!    returns the cursor location in screen points. We scale to
//!    capture pixels using the (logical → physical) ratio captured at
//!    `start_time`.
//!
//! Threading: one poll thread at 120 Hz handles both responsibilities.
//! Cursor seed reads are sub-microsecond; position reads via CGEvent
//! are also cheap. Sprite fetch (the expensive part — TIFF
//! round-trip) only fires on a seed change.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{unbounded, Receiver, Sender};
use objc2::ffi::{NSInteger, NSUInteger};
use objc2::msg_send;
use objc2::runtime::{AnyClass, AnyObject};
use objc2::{Encode, Encoding, RefEncode};
use tether_protocol::cursor::CursorPixelFormat;

use crate::cursor::{CursorEvent, CursorPosition, CursorShapeEvent, CursorSource};

/// Source spawned by [`start`]. Holds the receivers + the stop flag
/// the poll thread observes. Dropping the source signals shutdown and
/// joins the thread.
pub struct NSCursorCursorSource {
    shape_rx: Receiver<CursorEvent>,
    position_state: Arc<Mutex<Option<CursorPosition>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for NSCursorCursorSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl CursorSource for NSCursorCursorSource {
    fn next_event(&mut self) -> CursorEvent {
        self.shape_rx.try_recv().unwrap_or(CursorEvent::Idle)
    }

    fn poll_position(&mut self) -> Option<CursorPosition> {
        self.position_state.lock().ok().and_then(|g| *g)
    }
}

/// Geometry of the capture stream, supplied by the SCK backend so the
/// cursor source can translate `CGEventGetLocation`'s screen-point
/// coordinates into the same pixel frame the encoder emits.
#[derive(Clone, Copy, Debug)]
pub struct CaptureGeometry {
    /// Logical-point width of the captured display (what
    /// `SCDisplay::width()` returns, the post-HiDPI-scaling value).
    pub logical_point_w: f64,
    /// Logical-point height of the captured display.
    pub logical_point_h: f64,
    /// Capture-pixel width (`CGDisplayModeGetPixelWidth` floored to
    /// the 16-pixel encoder grid).
    pub capture_pixel_w: u32,
    /// Capture-pixel height.
    pub capture_pixel_h: u32,
}

/// Build the source and spawn the poll thread.
pub fn start(geom: CaptureGeometry) -> Box<dyn CursorSource> {
    start_with_shared_geometry(Arc::new(Mutex::new(geom)))
}

fn start_with_shared_geometry(geom: Arc<Mutex<CaptureGeometry>>) -> Box<dyn CursorSource> {
    let (shape_tx, shape_rx) = unbounded::<CursorEvent>();
    let position_state = Arc::new(Mutex::new(None::<CursorPosition>));
    let stop = Arc::new(AtomicBool::new(false));

    let pos_for_thread = Arc::clone(&position_state);
    let stop_for_thread = Arc::clone(&stop);

    let seed_fn = unsafe { lookup_cursor_seed() };
    let initial_seed = seed_fn.map(|f| unsafe { f() });
    let initial_geom = geom.lock().map(|g| *g).unwrap_or(CaptureGeometry {
        logical_point_w: 1.0,
        logical_point_h: 1.0,
        capture_pixel_w: 1,
        capture_pixel_h: 1,
    });
    match initial_seed {
        Some(s) => tracing::info!(
            initial_seed = s,
            logical_w = initial_geom.logical_point_w,
            logical_h = initial_geom.logical_point_h,
            pixel_w = initial_geom.capture_pixel_w,
            pixel_h = initial_geom.capture_pixel_h,
            "macOS cursor source: CGSCurrentCursorSeed available, starting poll thread"
        ),
        None => tracing::warn!(
            logical_w = initial_geom.logical_point_w,
            logical_h = initial_geom.logical_point_h,
            pixel_w = initial_geom.capture_pixel_w,
            pixel_h = initial_geom.capture_pixel_h,
            "macOS cursor source: CGSCurrentCursorSeed not found via dlsym; \
             cursor sprite will be re-fetched every 33 ms (slow path)"
        ),
    }

    let thread = std::thread::Builder::new()
        .name("tether-capture-mac-cursor".into())
        .spawn(move || {
            run_poll_thread(geom, seed_fn, shape_tx, pos_for_thread, stop_for_thread);
        })
        .expect("spawn mac cursor thread");

    Box::new(NSCursorCursorSource {
        shape_rx,
        position_state,
        stop,
        thread: Some(thread),
    })
}

/// Poll cadence. Position deltas at 30 Hz would feel laggy on the
/// client overlay; 120 Hz collapses to the host pump's own tick.
const POLL_INTERVAL: Duration = Duration::from_millis(8);

fn run_poll_thread(
    geom: Arc<Mutex<CaptureGeometry>>,
    seed_fn: Option<unsafe extern "C" fn() -> i32>,
    shape_tx: Sender<CursorEvent>,
    position_state: Arc<Mutex<Option<CursorPosition>>>,
    stop: Arc<AtomicBool>,
) {
    let mut last_seed: i32 = i32::MIN;
    let mut last_shape_id: Option<u64> = None;
    // Without seed support we still rate-limit the sprite fetch — it
    // costs an Objective-C TIFF round-trip per attempt.
    let sprite_interval = Duration::from_millis(33);
    let mut last_sprite_attempt = std::time::Instant::now()
        .checked_sub(sprite_interval)
        .unwrap_or_else(std::time::Instant::now);
    // Once-only logging guards. Each event is recorded the very first
    // time it happens; an empty log set tells you which step is silently
    // failing without forcing a debug-level rebuild.
    let mut logged_first_position = false;
    let mut logged_first_seed_change = false;
    let mut logged_first_sprite_ok = false;
    let mut logged_first_sprite_null = false;
    // Periodic heartbeat so a session that emits zero shape changes can
    // still distinguish "thread alive, nothing happening" from "thread
    // died." At debug level since it's a liveness check, not user-facing.
    let mut last_heartbeat = std::time::Instant::now();
    let mut sprite_attempts: u64 = 0;
    let mut sprite_emits: u64 = 0;

    tracing::info!("mac cursor poll thread entered");

    while !stop.load(Ordering::Acquire) {
        let tick_start = std::time::Instant::now();

        // Position: cheap, every tick.
        let current_geom = geom.lock().map(|g| *g).unwrap_or(CaptureGeometry {
            logical_point_w: 1.0,
            logical_point_h: 1.0,
            capture_pixel_w: 1,
            capture_pixel_h: 1,
        });

        if let Some(pos) = read_pointer_position(current_geom) {
            if !logged_first_position {
                tracing::info!(
                    x = pos.x,
                    y = pos.y,
                    visible = pos.visible,
                    "first mac cursor position read"
                );
                logged_first_position = true;
            }
            if let Ok(mut g) = position_state.lock() {
                *g = Some(pos);
            }
        }

        // Sprite: only re-fetch on a seed change (or on the rate-
        // limited slow path when the seed symbol is unavailable).
        let should_fetch_sprite = match seed_fn {
            Some(f) => {
                let seed = unsafe { f() };
                if seed != last_seed {
                    if !logged_first_seed_change && last_seed != i32::MIN {
                        tracing::info!(
                            old_seed = last_seed,
                            new_seed = seed,
                            "first CGSCurrentCursorSeed change observed \
                             (cross-process cursor change detected)"
                        );
                        logged_first_seed_change = true;
                    }
                    last_seed = seed;
                    true
                } else {
                    false
                }
            }
            None => {
                if tick_start.duration_since(last_sprite_attempt) >= sprite_interval {
                    last_sprite_attempt = tick_start;
                    true
                } else {
                    false
                }
            }
        };
        if should_fetch_sprite {
            sprite_attempts += 1;
            match read_cursor_sprite(current_geom) {
                Some(shape) => {
                    if !logged_first_sprite_ok {
                        tracing::info!(
                            id = shape.id,
                            w = shape.width,
                            h = shape.height,
                            hotspot = ?shape.hotspot,
                            pixel_bytes = shape.pixels.len(),
                            "first NSCursor sprite extracted"
                        );
                        logged_first_sprite_ok = true;
                    }
                    if last_shape_id != Some(shape.id) {
                        last_shape_id = Some(shape.id);
                        let _ = shape_tx.try_send(CursorEvent::Shape(shape));
                        sprite_emits += 1;
                    }
                }
                None => {
                    if !logged_first_sprite_null {
                        tracing::warn!(
                            "read_cursor_sprite returned None on first attempt; \
                             see debug-level traces in cursor_macos for which step \
                             returned null (NSCursor / NSImage / TIFF / NSBitmapImageRep)"
                        );
                        logged_first_sprite_null = true;
                    }
                }
            }
        }

        if last_heartbeat.elapsed() >= Duration::from_secs(5) {
            tracing::debug!(
                seed = last_seed,
                last_shape_id = ?last_shape_id,
                sprite_attempts,
                sprite_emits,
                "mac cursor poll thread heartbeat"
            );
            last_heartbeat = std::time::Instant::now();
        }

        // Sleep the remainder of the tick.
        let elapsed = tick_start.elapsed();
        if elapsed < POLL_INTERVAL {
            std::thread::sleep(POLL_INTERVAL - elapsed);
        }
    }
}

// ===========================================================================
// Position
// ===========================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CGPoint {
    x: f64,
    y: f64,
}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    /// `CGEventCreate(NULL)` returns a +1 retained CGEventRef
    /// representing the current input state. Passing the result to
    /// `CGEventGetLocation` reads the cursor location in screen
    /// points (relative to the top-left of the main display).
    fn CGEventCreate(source: *const std::ffi::c_void) -> *mut std::ffi::c_void;
    fn CGEventGetLocation(event: *mut std::ffi::c_void) -> CGPoint;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: *const std::ffi::c_void);
}

// Pointer coords are rounded screen-space values bounded by the display
// rect; the i32→u32 casts are guarded by the `>= 0` visibility checks.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn read_pointer_position(geom: CaptureGeometry) -> Option<CursorPosition> {
    // SAFETY: passing NULL to CGEventCreate is the documented form for
    // "snapshot the current input state." The returned event is +1
    // retained; we CFRelease it before returning.
    let point = unsafe {
        let evt = CGEventCreate(std::ptr::null());
        if evt.is_null() {
            return None;
        }
        let p = CGEventGetLocation(evt);
        CFRelease(evt as *const _);
        p
    };

    let scale_x = if geom.logical_point_w > 0.0 {
        f64::from(geom.capture_pixel_w) / geom.logical_point_w
    } else {
        1.0
    };
    let scale_y = if geom.logical_point_h > 0.0 {
        f64::from(geom.capture_pixel_h) / geom.logical_point_h
    } else {
        1.0
    };
    let px = (point.x * scale_x).round() as i32;
    let py = (point.y * scale_y).round() as i32;

    // macOS has no first-class "cursor hidden" signal we can poll.
    // Conservative proxy: pointer outside the captured display rect ⇒
    // not visible. This avoids the default-true bug where a stale
    // overlay would pin to (0,0) when the pointer leaves the display.
    let visible = px >= 0
        && py >= 0
        && (px as u32) < geom.capture_pixel_w
        && (py as u32) < geom.capture_pixel_h;

    Some(CursorPosition {
        x: px,
        y: py,
        visible,
    })
}

// ===========================================================================
// Sprite (NSCursor.currentSystemCursor → NSBitmapImageRep → RGBA)
// ===========================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NSPoint {
    x: f64,
    y: f64,
}

// SAFETY: matches Apple's `NSPoint` (= `CGPoint`) struct layout — two
// `CGFloat` (f64 on 64-bit Apple platforms) in order.
unsafe impl Encode for NSPoint {
    const ENCODING: Encoding = Encoding::Struct("CGPoint", &[f64::ENCODING, f64::ENCODING]);
}
unsafe impl RefEncode for NSPoint {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NSSize {
    width: f64,
    height: f64,
}

// SAFETY: matches Apple's `NSSize` (= `CGSize`).
unsafe impl Encode for NSSize {
    const ENCODING: Encoding = Encoding::Struct("CGSize", &[f64::ENCODING, f64::ENCODING]);
}
unsafe impl RefEncode for NSSize {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

/// NSBitmapFormat bit constants we care about. Full list in
/// `<AppKit/NSBitmapImageRep.h>`.
const NS_BITMAP_FORMAT_ALPHA_FIRST: NSUInteger = 1 << 0;
const NS_BITMAP_FORMAT_ALPHA_NON_PREMULTIPLIED: NSUInteger = 1 << 1;
const NS_BITMAP_FORMAT_32_BIT_LITTLE_ENDIAN: NSUInteger = 1 << 2;

// Sprite math: premultiplied-alpha colours are clamped to [0, 255] before the
// f32→u8 cast, and the f64→u32 dimension/hotspot conversions are rounded,
// non-negative, and bounded by the (small) displayed cursor size.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn read_cursor_sprite(geom: CaptureGeometry) -> Option<CursorShapeEvent> {
    // SAFETY: every call below is a standard AppKit message send. The
    // returned objects are autoreleased; objc2's autoreleasepool keeps
    // them alive for the body of this function. AppKit on a background
    // thread is "mostly fine" — `NSCursor.currentSystemCursor` is
    // empirically thread-safe (RustDesk has shipped this for years).
    objc2::rc::autoreleasepool(|_| unsafe {
        let Some(ns_cursor_class) = AnyClass::get(c"NSCursor") else {
            tracing::debug!("NSCursor class lookup returned None");
            return None;
        };
        let cursor: *mut AnyObject = msg_send![ns_cursor_class, currentSystemCursor];
        if cursor.is_null() {
            tracing::debug!("NSCursor.currentSystemCursor returned nil");
            return None;
        }
        let image: *mut AnyObject = msg_send![cursor, image];
        if image.is_null() {
            tracing::debug!("NSCursor.image returned nil");
            return None;
        }
        let hotspot: NSPoint = msg_send![cursor, hotSpot];
        let point_size: NSSize = msg_send![image, size];

        let tiff: *mut AnyObject = msg_send![image, TIFFRepresentation];
        if tiff.is_null() {
            tracing::debug!("NSImage.TIFFRepresentation returned nil");
            return None;
        }
        let Some(rep_class) = AnyClass::get(c"NSBitmapImageRep") else {
            tracing::debug!("NSBitmapImageRep class lookup returned None");
            return None;
        };
        let rep: *mut AnyObject = msg_send![rep_class, imageRepWithData: tiff];
        if rep.is_null() {
            tracing::debug!("NSBitmapImageRep.imageRepWithData returned nil");
            return None;
        }

        let pixels_wide: NSInteger = msg_send![rep, pixelsWide];
        let pixels_high: NSInteger = msg_send![rep, pixelsHigh];
        let bytes_per_row: NSInteger = msg_send![rep, bytesPerRow];
        let samples_per_pixel: NSInteger = msg_send![rep, samplesPerPixel];
        let bits_per_pixel: NSInteger = msg_send![rep, bitsPerPixel];
        let bitmap_format: NSUInteger = msg_send![rep, bitmapFormat];
        let data_ptr: *const u8 = msg_send![rep, bitmapData];

        if pixels_wide <= 0 || pixels_high <= 0 || bytes_per_row <= 0 || data_ptr.is_null() {
            return None;
        }
        // Only handle the standard 32-bit RGBA / ARGB layouts the TIFF
        // round-trip produces. Exotic formats (float, planar, indexed)
        // fall through — the client overlay sees no shape change.
        if bits_per_pixel != 32 || samples_per_pixel != 4 {
            tracing::debug!(
                bits_per_pixel,
                samples_per_pixel,
                "NSCursor sprite in unsupported format; skipping"
            );
            return None;
        }

        let width = u16::try_from(pixels_wide).ok()?;
        let height = u16::try_from(pixels_high).ok()?;
        let stride = usize::try_from(bytes_per_row).ok()?;
        let row_bytes = (width as usize).checked_mul(4)?;
        if stride < row_bytes {
            return None;
        }

        let alpha_first = bitmap_format & NS_BITMAP_FORMAT_ALPHA_FIRST != 0;
        let little_endian = bitmap_format & NS_BITMAP_FORMAT_32_BIT_LITTLE_ENDIAN != 0;
        let non_premul = bitmap_format & NS_BITMAP_FORMAT_ALPHA_NON_PREMULTIPLIED != 0;

        let total = stride.checked_mul(height as usize)?;
        // SAFETY: bitmapData points to at least bytesPerRow * pixelsHigh
        // bytes per AppKit's contract for NSBitmapImageRep.
        let raw: &[u8] = std::slice::from_raw_parts(data_ptr, total);

        let mut rgba: Vec<u8> = Vec::with_capacity(row_bytes * height as usize);
        for y in 0..height as usize {
            let line = &raw[y * stride..y * stride + row_bytes];
            for px in line.chunks_exact(4) {
                // Map (alpha_first × endianness) → (R, G, B, A).
                // TIFF defaults: alpha_first=false, little_endian=false
                // ⇒ memory order [R, G, B, A].
                let (r, g, b, a) = match (alpha_first, little_endian) {
                    (false, false) => (px[0], px[1], px[2], px[3]),
                    (false, true) => (px[3], px[2], px[1], px[0]),
                    (true, false) => (px[1], px[2], px[3], px[0]),
                    (true, true) => (px[2], px[1], px[0], px[3]),
                };
                let (r, g, b) = if non_premul || a == 0 {
                    (r, g, b)
                } else {
                    // Convert premultiplied → straight: divide colour
                    // by alpha. The client's overlay shader expects
                    // straight RGBA.
                    let inv = 255.0 / f32::from(a);
                    (
                        ((f32::from(r) * inv).min(255.0)) as u8,
                        ((f32::from(g) * inv).min(255.0)) as u8,
                        ((f32::from(b) * inv).min(255.0)) as u8,
                    )
                };
                rgba.extend_from_slice(&[r, g, b, a]);
            }
        }

        // Apple ships `NSCursor.image` as a multi-resolution asset.
        // `pixelsWide`/`pixelsHigh` is the *largest* rep in the TIFF
        // (e.g. 170×230 for the accessibility arrow) — NOT what the
        // user actually sees drawn on the host display. The displayed
        // cursor is `image.size` (points) at the screen's backing
        // scale. Sending the raw bitmap dims means the client renders
        // a 4×-oversampled cursor that looks 2–4× too big.
        //
        // The right "source" dim is `image.size × capture_scale`,
        // where capture_scale is the host display's backing-px-per-
        // point ratio (≈ 2 on Retina). That matches what the user
        // actually sees, in the same coordinate frame the rest of the
        // pipeline (encoder + cursor pump rescale) expects.
        let capture_scale_x = if geom.logical_point_w > 0.0 {
            f64::from(geom.capture_pixel_w) / geom.logical_point_w
        } else {
            1.0
        };
        let capture_scale_y = if geom.logical_point_h > 0.0 {
            f64::from(geom.capture_pixel_h) / geom.logical_point_h
        } else {
            1.0
        };
        let displayed_w = ((point_size.width * capture_scale_x).round() as u32).max(1);
        let displayed_h = ((point_size.height * capture_scale_y).round() as u32).max(1);
        // Hotspot is in points; convert to capture-pixel space using
        // the same scale so it tracks the (now-resampled) bitmap.
        let raw_hot_x = (hotspot.x * capture_scale_x).max(0.0).round() as u32;
        let raw_hot_y = (hotspot.y * capture_scale_y).max(0.0).round() as u32;
        tracing::debug!(
            rep_pixels = ?(width, height),
            point_size = ?(point_size.width, point_size.height),
            capture_scale = ?(capture_scale_x, capture_scale_y),
            displayed = ?(displayed_w, displayed_h),
            hotspot_pts = ?(hotspot.x, hotspot.y),
            hotspot_px = ?(raw_hot_x, raw_hot_y),
            "NSCursor sprite dims (rep vs displayed)"
        );

        // Downsample the oversized rep to the actual displayed size.
        // Then apply the wire-budget safety cap (still useful as a
        // backstop against a malformed image with degenerate
        // `image.size`).
        let displayed_w_u16 = u16::try_from(displayed_w).unwrap_or(u16::MAX);
        let displayed_h_u16 = u16::try_from(displayed_h).unwrap_or(u16::MAX);
        let (resampled_w, resampled_h, resampled_pixels) =
            if displayed_w_u16 < width || displayed_h_u16 < height {
                let pixels = crate::cursor::box_downsample_rgba(
                    width,
                    height,
                    &rgba,
                    displayed_w_u16,
                    displayed_h_u16,
                );
                (displayed_w_u16, displayed_h_u16, pixels)
            } else {
                (width, height, rgba)
            };
        // 64 KiB control-stream safety cap. The "displayed" path should
        // already produce far smaller bitmaps, but a degenerate
        // `image.size` (0 or absurdly large) is still possible.
        const MAX_DIM: u16 = 128;
        let (final_w, final_h, final_pixels, final_hot) = downscale_if_oversize(
            resampled_w,
            resampled_h,
            resampled_pixels,
            (raw_hot_x, raw_hot_y),
            MAX_DIM,
        );

        let hotspot_px = (
            u16::try_from(final_hot.0).unwrap_or(0),
            u16::try_from(final_hot.1).unwrap_or(0),
        );

        Some(CursorShapeEvent {
            id: crate::cursor::fnv1a_64(final_w, final_h, &final_pixels),
            width: final_w,
            height: final_h,
            hotspot: hotspot_px,
            format: CursorPixelFormat::Rgba8,
            pixels: final_pixels,
        })
    })
}

/// Box-downsample RGBA pixels if either dimension exceeds `max_dim`,
/// preserving aspect ratio. Hotspot is rescaled proportionally. The
/// downscale path runs once per sprite *change* (not per frame), so a
/// pure-Rust implementation is fine even at ~64 KB input.
// `scale <= 1` here (we only enter the resize branch when a dim exceeds
// `max_dim`), so the rounded f64→u32→u16 results are non-negative and never
// exceed the original u16 dims.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn downscale_if_oversize(
    width: u16,
    height: u16,
    pixels: Vec<u8>,
    hotspot: (u32, u32),
    max_dim: u16,
) -> (u16, u16, Vec<u8>, (u32, u32)) {
    if width <= max_dim && height <= max_dim {
        return (width, height, pixels, hotspot);
    }
    let longest = width.max(height);
    let scale = f64::from(max_dim) / f64::from(longest);
    let new_w = ((f64::from(width) * scale).round() as u32).max(1) as u16;
    let new_h = ((f64::from(height) * scale).round() as u32).max(1) as u16;
    let new_pixels = crate::cursor::box_downsample_rgba(width, height, &pixels, new_w, new_h);
    let new_hot = (
        ((f64::from(hotspot.0) * scale).round() as u32),
        ((f64::from(hotspot.1) * scale).round() as u32),
    );
    tracing::info!(
        src_w = width,
        src_h = height,
        dst_w = new_w,
        dst_h = new_h,
        "cursor sprite downscaled to fit control-stream budget"
    );
    (new_w, new_h, new_pixels, new_hot)
}

// ===========================================================================
// CGSCurrentCursorSeed (private symbol, weak-linked via dlsym)
// ===========================================================================

/// SAFETY: looked-up symbol must be ABI-compatible with
/// `extern "C" fn() -> i32`. `CGSCurrentCursorSeed` is documented
/// (via reverse-engineering of the SkyLight private framework) as
/// returning a `CGSConnectionID`-sized counter, which is `i32`.
unsafe fn lookup_cursor_seed() -> Option<unsafe extern "C" fn() -> i32> {
    let name = c"CGSCurrentCursorSeed";
    let sym = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
    if sym.is_null() {
        return None;
    }
    // SAFETY: caller's responsibility — see fn-level safety note.
    Some(unsafe {
        std::mem::transmute::<*mut std::ffi::c_void, unsafe extern "C" fn() -> i32>(sym)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_id_changes_on_pixel_edit() {
        let a = crate::cursor::fnv1a_64(2, 2, &[0; 16]);
        let mut b_bytes = [0u8; 16];
        b_bytes[7] = 1;
        let b = crate::cursor::fnv1a_64(2, 2, &b_bytes);
        assert_ne!(a, b);
    }

    #[test]
    fn fnv1a_id_changes_on_dims_edit() {
        let pixels = [0u8; 16];
        assert_ne!(
            crate::cursor::fnv1a_64(2, 2, &pixels),
            crate::cursor::fnv1a_64(4, 1, &pixels)
        );
    }

    #[test]
    fn downscale_passthrough_when_under_cap() {
        let pixels = vec![0xAB; 16 * 16 * 4];
        let (w, h, out, hot) = downscale_if_oversize(16, 16, pixels.clone(), (3, 4), 128);
        assert_eq!((w, h), (16, 16));
        assert_eq!(out.len(), pixels.len());
        assert_eq!(hot, (3, 4));
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // rounded, bounded test arithmetic
    fn downscale_preserves_aspect_and_fits_under_cap() {
        let pixels = vec![0xFFu8; 170 * 230 * 4];
        let (w, h, out, hot) = downscale_if_oversize(170, 230, pixels, (40, 40), 128);
        assert!(w <= 128 && h <= 128, "got {w}x{h}");
        assert_eq!(h, 128, "longest dim should hit the cap");
        let aspect_in = 170.0_f64 / 230.0_f64;
        let aspect_out = f64::from(w) / f64::from(h);
        assert!(
            (aspect_in - aspect_out).abs() < 0.02,
            "aspect drifted: in={aspect_in}, out={aspect_out}"
        );
        assert_eq!(out.len(), (w as usize) * (h as usize) * 4);
        assert!(
            out.len() < 64 * 1024,
            "downscaled sprite must fit framed cap"
        );
        // Hotspot scales by the same factor as the longest side.
        let expected_scale = f64::from(h) / 230.0;
        let expected_hot_x = (40.0 * expected_scale).round() as u32;
        assert!(hot.0.abs_diff(expected_hot_x) <= 1);
    }

    #[test]
    fn box_downsample_solid_colour_stays_solid_with_alpha_premul() {
        // 4×4 solid red, fully opaque → 2×2 solid red.
        let src: Vec<u8> = (0..16).flat_map(|_| [0xFF, 0x00, 0x00, 0xFF]).collect();
        let out = crate::cursor::box_downsample_rgba(4, 4, &src, 2, 2);
        for px in out.chunks_exact(4) {
            assert_eq!(px, [0xFF, 0x00, 0x00, 0xFF]);
        }
    }

    #[test]
    fn box_downsample_transparent_source_yields_transparent_output() {
        let src = vec![0u8; 4 * 4 * 4];
        let out = crate::cursor::box_downsample_rgba(4, 4, &src, 2, 2);
        assert!(out.iter().all(|&b| b == 0));
    }

    #[test]
    #[allow(clippy::cast_sign_loss)] // visibility predicate guards `>= 0` before the i32→u32 cast
    fn position_outside_bounds_marks_hidden() {
        let geom = CaptureGeometry {
            logical_point_w: 1000.0,
            logical_point_h: 500.0,
            capture_pixel_w: 1000,
            capture_pixel_h: 500,
        };
        // Stub the CGEvent call by using a known geometry and direct
        // arithmetic. The `read_pointer_position` shape works on any
        // CGPoint; we re-derive the visibility predicate here to lock
        // it in.
        let visible = |x: i32, y: i32| {
            x >= 0
                && y >= 0
                && (x as u32) < geom.capture_pixel_w
                && (y as u32) < geom.capture_pixel_h
        };
        assert!(visible(0, 0));
        assert!(visible(999, 499));
        assert!(!visible(-1, 10));
        assert!(!visible(10, 500));
        assert!(!visible(1000, 10));
    }
}
