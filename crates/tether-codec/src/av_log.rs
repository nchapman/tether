//! Bridge FFmpeg's `av_log` output into `tracing`, plus a process-wide
//! warning-or-above counter that callers (notably the client decoder)
//! can poll to detect "ffmpeg is unhappy" states that don't surface
//! through the `avcodec_send_packet` / `receive_frame` return code.
//!
//! The motivation: HEVC RPS-construction failures ("Could not find ref
//! with POC N") happen entirely inside libavcodec — the decoder skips
//! the undecodable NALU and returns `Ok` from the API, but the user
//! sees a frozen frame until the next IDR arrives. Without this
//! bridge, the only visible signal was raw stderr output and the
//! client's `decode_errs=0` metric was lying. With it, the client
//! can observe the counter advance and proactively request a
//! `ForceIdr` so recovery is fast.
//!
//! Routing: messages at AV_LOG_INFO or above go to tracing at the
//! mapped level (panic/fatal/error → error, warning → warn, info →
//! info). AV_LOG_VERBOSE / DEBUG / TRACE are dropped — they're far
//! too chatty under normal operation and bog down tracing.
//!
//! Dedup: `AV_LOG_SKIP_REPEATED` is enabled at the ffmpeg level so
//! the same "skipping invalid undecodable NALU" line doesn't show up
//! 30 times in a row. The counter still increments once per logged
//! message (after dedup), which is exactly what callers want for
//! "did anything new go wrong since I last checked?".

use std::ffi::{c_char, c_int, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Once, OnceLock};

use crossbeam_channel::{bounded, Sender, TrySendError};
use rsmpeg::ffi;

// FFmpeg's logging callback takes a `va_list`, whose concrete Rust
// type differs per platform (bindgen generates the host C ABI's
// shape). Linux is a pointer to a `__va_list_tag` struct; macOS
// (aarch64) is the bindgen-generated `va_list` typedef. We never
// inspect the variadic args — they get passed straight back into
// `av_log_format_line2` — so a target-conditional alias suffices.
#[cfg(target_os = "linux")]
type AvLogVaList = *mut ffi::__va_list_tag;
#[cfg(not(target_os = "linux"))]
type AvLogVaList = ffi::va_list;

/// Process-wide count of `av_log` messages at AV_LOG_WARNING or
/// above. Use [`warning_or_above_count`] to read; the counter never
/// decreases. Callers diff successive snapshots to detect new
/// problems.
static WARNING_OR_ABOVE: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the warning-or-above counter. Monotonically
/// non-decreasing across the lifetime of the process. A diff between
/// two snapshots tells the caller how many warning/error messages
/// libavcodec emitted in that window.
///
/// **Process-wide scope.** Encoders and decoders share this counter,
/// so a host process running both will conflate the two. Today only
/// the client decoder polls it; if a future host-side diagnostic
/// wants the same signal it should either disambiguate by interval
/// (the host's encoder is quiet in steady state) or refactor to a
/// per-context counter (`Arc<AtomicU64>` registered against an
/// `AVClass*` lookup in the callback).
///
/// **Lower bound on broken frames.** `AV_LOG_SKIP_REPEATED` is
/// enabled at the ffmpeg level, so a 30-frame RPS storm of identical
/// "Could not find ref" lines is one counter increment, not 30. The
/// boolean check "did this advance?" is the intended use; treating
/// the delta as a literal frame-loss count understates by however
/// many repeats ffmpeg deduplicated.
#[must_use]
pub fn warning_or_above_count() -> u64 {
    WARNING_OR_ABOVE.load(Ordering::Relaxed)
}

/// Tracing emit happens on a dedicated drainer thread, not on the
/// decoder thread that libavcodec invokes the callback from. The
/// channel is bounded; on overflow we drop the line (the counter
/// still bumps, which is the load-bearing signal). This isolates
/// the QUIC datagram recv loop (which on the client is the same
/// thread that runs decode) from tracing-subscriber latency.
struct LogRecord {
    level: c_int,
    class: String,
    msg: String,
}

/// Sized to absorb a worst-case dedup-suppressed error storm
/// (a few hundred unique messages over ~1 s) without dropping
/// anything in the common case. A full channel drops the line
/// silently; that's preferable to blocking the decoder.
const LOG_CHANNEL_CAPACITY: usize = 256;

static LOG_SENDER: OnceLock<Sender<LogRecord>> = OnceLock::new();

/// Install the `av_log_set_callback` bridge. Idempotent — safe to call
/// from every codec constructor; the underlying install happens once.
pub(crate) fn install() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let (tx, rx) = bounded::<LogRecord>(LOG_CHANNEL_CAPACITY);
        let _ = LOG_SENDER.set(tx);
        std::thread::Builder::new()
            .name("tether-avlog".into())
            .spawn(move || {
                #[allow(clippy::cast_possible_wrap)]
                let warning_level = ffi::AV_LOG_WARNING as c_int;
                #[allow(clippy::cast_possible_wrap)]
                let error_level = ffi::AV_LOG_ERROR as c_int;
                #[allow(clippy::cast_possible_wrap)]
                let info_level = ffi::AV_LOG_INFO as c_int;
                while let Ok(rec) = rx.recv() {
                    // Wrap in catch_unwind so a buggy subscriber can't
                    // kill the drainer thread.
                    let _ = catch_unwind(AssertUnwindSafe(|| {
                        if rec.level <= warning_level {
                            if rec.level <= error_level {
                                tracing::error!(target: "ffmpeg", class = %rec.class, "{}", rec.msg);
                            } else {
                                tracing::warn!(target: "ffmpeg", class = %rec.class, "{}", rec.msg);
                            }
                        } else if rec.level <= info_level {
                            tracing::info!(target: "ffmpeg", class = %rec.class, "{}", rec.msg);
                        }
                    }));
                }
            })
            .expect("spawn tether-avlog drainer thread");

        // SAFETY: `av_log_set_callback` accepts a thread-safe extern "C"
        // function pointer; ours satisfies that contract (it reads no
        // shared mutable state besides the atomic counter and a
        // OnceLock-installed channel sender, both Sync). Order matters:
        // `LOG_SENDER.set(tx)` above happens-before the callback is
        // registered here, so a callback fired by a later codec
        // construction always observes a populated sender.
        unsafe {
            ffi::av_log_set_callback(Some(tether_log_callback));
            // Dedup consecutive identical lines at the ffmpeg layer so
            // an RPS storm doesn't flood tracing with 30 copies of the
            // same warning. Counter increments still happen once per
            // (deduplicated) message.
            #[allow(clippy::cast_possible_wrap)]
            ffi::av_log_set_flags(ffi::AV_LOG_SKIP_REPEATED as c_int);
        }
    });
}

// Re-entrancy guard for the callback. `av_log_format_line2` itself
// can in principle re-enter `av_log` (it shouldn't for the formats
// we care about, but the cost of guarding is one atomic load) and
// our `tracing::*` calls absolutely can if a subscriber is buggy.
// A re-entrant call would either deadlock the buffer or recurse
// indefinitely; the guard short-circuits the inner call to a no-op.
thread_local! {
    static IN_CALLBACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// C callback installed via `av_log_set_callback`. Formats the line
/// into a stack buffer, routes it to tracing at the mapped level, and
/// bumps the warning/error counter when appropriate.
///
/// # Safety
/// - All pointer arguments are owned by libavcodec for the duration
///   of the call; we do not retain them past return.
/// - `av_log_format_line2` is the documented way to render a
///   `va_list` from inside an av_log callback; we pass `vl` straight
///   back to it without inspecting the variadic args ourselves.
unsafe extern "C" fn tether_log_callback(
    avcl: *mut c_void,
    level: c_int,
    fmt: *const c_char,
    vl: AvLogVaList,
) {
    // Drop verbose/debug/trace entirely — they're firehose-level and
    // not useful for diagnosis at info-level tracing.
    #[allow(clippy::cast_possible_wrap)]
    if level > ffi::AV_LOG_INFO as c_int {
        return;
    }

    let already_in = IN_CALLBACK.with(|c| {
        if c.get() {
            true
        } else {
            c.set(true);
            false
        }
    });
    if already_in {
        return;
    }
    // RAII reset so an early return / panic in the formatter still
    // clears the flag.
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            IN_CALLBACK.with(|c| c.set(false));
        }
    }
    let _guard = Guard;

    let mut buf = [0u8; 1024];
    let mut print_prefix: c_int = 1;
    // SAFETY: ffmpeg-owned pointers used inside their valid window;
    // buf is a fresh local; print_prefix is a stack int.
    let written = unsafe {
        ffi::av_log_format_line2(
            avcl,
            level,
            fmt,
            vl,
            buf.as_mut_ptr().cast::<c_char>(),
            #[allow(clippy::cast_possible_wrap)]
            { buf.len() as c_int },
            &mut print_prefix,
        )
    };
    if written <= 0 {
        return;
    }
    // av_log_format_line2 NUL-terminates and may report a length
    // larger than the buffer if it truncated. Clamp to what's
    // actually present.
    #[allow(clippy::cast_sign_loss)]
    let nul_at = buf
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(buf.len());
    let raw = &buf[..nul_at];
    // Trim trailing newline — av_log lines end with '\n' but tracing
    // adds its own.
    let trimmed = match raw.split_last() {
        Some((&b'\n', rest)) => rest,
        _ => raw,
    };
    let msg = match std::str::from_utf8(trimmed) {
        Ok(s) => s,
        // libavcodec emits ASCII-only messages in practice; on the
        // off chance one isn't valid UTF-8 we drop it rather than
        // lossy-convert (a bytes-replacement here would still be
        // logged at the wrong level and the bytes are not load-
        // bearing for the counter).
        Err(_) => return,
    };

    // The first field of every AVClass-bearing context is a pointer
    // to an AVClass. `av_default_item_name` walks that and returns a
    // short name ("hevc", "h264_videotoolbox", ...). Tag the tracing
    // record with it so a reader can tell which codec instance the
    // line came from.
    let class_name = if avcl.is_null() {
        ""
    } else {
        // SAFETY: when avcl is non-null, ffmpeg promises the first
        // field is an AVClass*. `av_default_item_name` handles the
        // null-class case internally and never returns NULL.
        let p = unsafe { ffi::av_default_item_name(avcl) };
        if p.is_null() {
            ""
        } else {
            // SAFETY: returned string is borrowed from a static
            // AVClass; valid for as long as the AVClass is, which
            // outlives this callback frame.
            unsafe { CStr::from_ptr(p) }.to_str().unwrap_or("")
        }
    };

    #[allow(clippy::cast_possible_wrap)]
    let warning_level = ffi::AV_LOG_WARNING as c_int;
    // Bump the counter first so it lands even if the channel send
    // below fails — a missed warning is worse than a missed log
    // line, and the counter is the load-bearing signal for the
    // client's auto-IDR path.
    if level <= warning_level {
        WARNING_OR_ABOVE.fetch_add(1, Ordering::Relaxed);
    }
    // Hand the formatted message to the drainer thread. The decoder
    // thread never calls into `tracing` directly — that's the whole
    // point of this bridge. `try_send` so a slow subscriber can't
    // backpressure the decoder; full channel drops the line.
    //
    // `catch_unwind` wraps the channel send so a panic (e.g. an
    // allocation failure in `String::from`) can't unwind across the
    // `extern "C"` boundary (UB).
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Some(tx) = LOG_SENDER.get() {
            let record = LogRecord {
                level,
                class: class_name.to_owned(),
                msg: msg.to_owned(),
            };
            match tx.try_send(record) {
                Ok(()) | Err(TrySendError::Full(_)) => {}
                Err(TrySendError::Disconnected(_)) => {
                    // Drainer thread is gone; nothing we can do. The
                    // counter still bumped above, which is what the
                    // auto-IDR path actually needs.
                }
            }
        }
    }));
}
