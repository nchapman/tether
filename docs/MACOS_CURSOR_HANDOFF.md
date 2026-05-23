# macOS cursor extraction handoff

**Status:** research complete, implementation not started.
**Target:** Workstream D of the cursor-delivery plan
(`/home/nchapman/.claude/plans/effervescent-weaving-valiant.md`,
GitHub #21).
**Goal:** lift the OS cursor *out of* the captured video frame on the
macOS host and stream it on the dedicated cursor channel, so the
client overlay can draw it at sub-frame latency. Mirrors the Linux
SPA_META_Cursor work that landed in `0dd1995`.

The shape of the wire and the client overlay are already in place —
`HostCursorPacket::Position`, `ControlMessage::CursorShape` /
`CursorUseShape`, plus the `CursorSource` trait
(`crates/tether-capture/src/cursor.rs`). The only missing piece is a
macOS implementation of `CursorSource` and wiring it into
`tether-capture::macos::start`.

## Why this is harder than it looks

macOS has no documented equivalent of PipeWire's `SPA_META_Cursor`.
The following dead ends got considered first:

- **`NSCursor.currentSystemCursor`** — on its face this returns the
  "current" cursor, but `NSCursor` is documented as a *per-process*
  cursor stack. When the host CLI is the calling process, you don't
  see the text-beam Safari pushed or the hand cursor Slack pushed,
  because those live in those apps' stacks.
- **`SCStreamFrameInfo` cursor fields** (macOS 14+) — undocumented
  beyond the bare attribute names. Neither OBS nor RustDesk uses
  them. We didn't verify whether the fields are populated; treat as
  unknown.
- **Reading composited pixels off the framebuffer** — no public API.
  WindowServer composites the cursor on its own layer.

So `.with_shows_cursor(true)` (the current setting at
`crates/tether-capture/src/macos.rs:286`) and burning the cursor into
the encoded frame is the lazy answer, and it's what OBS Studio's
`mac-capture` plugin does
(`refs/obs-studio/plugins/mac-capture/mac-sck-video-capture.m:202`).
**That's not what we want** — it defeats both the latency win and
the idle-bandwidth win the separation was supposed to buy.

## RustDesk's approach (the answer)

RustDesk solved this in production in September 2025 using a *private*
CoreGraphics symbol plus a careful threading model.

**Source paths in our reference clone** (`~/Code/refs/rustdesk`):

- `src/platform/macos.rs:509–638` — `get_cursor()` + `CGSCurrentCursorSeed()`
  polling + `get_cursor_data()` sprite extraction.
- `src/server/input_service.rs:395–410` — `run_cursor()` service loop
  ticking at 33 ms.

### The trick: `CGSCurrentCursorSeed()`

`CGSCurrentCursorSeed` is a private CoreGraphics symbol (no public
header) that returns a monotonic `int` incremented by WindowServer
every time the *system* cursor changes — regardless of which process
caused the change. Polling it is a cheap int read. Cache the last
seed; only re-fetch the sprite when it changes.

This is what makes the multi-process problem go away:
`NSCursor.currentSystemCursor` from a **background thread** DOES
reflect the system cursor (Apple's docs don't explicitly bless this,
but they don't forbid it, and RustDesk ships it). The seed tells you
*when* the sprite is fresh so you avoid burning CPU on identical
fetches.

### Flow

1. **Sprite poll thread** at 30 Hz (RustDesk uses 33 ms exactly):
   - Read `CGSCurrentCursorSeed()`. If unchanged from last tick, skip.
   - Read `NSCursor.currentSystemCursor`.
   - Pull pixels: `NSCursor.image` → `TIFFRepresentation` →
     `NSBitmapImageRep` → iterate pixel bytes into RGBA.
   - Read hotspot: `NSCursor.hotSpot` (NSPoint, points; scale by
     `NSImage.size` ratio when HiDPI).
   - Emit `CursorEvent::Shape`.
2. **Position poll thread** at 120 Hz:
   - `CGEventCreate(nil)` → `CGEventGetLocation(evt)` → `CGPoint`.
   - Cast to `(i32, i32)` physical pixels (mind the points-vs-pixels
     conversion on Retina — same axis as the video stream uses).
   - Update the `Arc<Mutex<Option<CursorPosition>>>` snapshot the
     host pump reads. RustDesk uses a single 33 ms cadence for both
     sprite and position; tether's plan separates them because
     position deltas at 30 Hz feel laggy. Worth measuring before
     committing to two threads — one thread at 60-120 Hz reading
     both may be enough.
3. **Capture config**: flip
   `crates/tether-capture/src/macos.rs:286` from
   `.with_shows_cursor(true)` to `.with_shows_cursor(false)` so the
   cursor stops being burned into the video. The probe path at
   line 791 is already `false`.
4. **Wire it up**: `tether-capture::macos::start` returns the new
   `NSCursorCursorSource` via `CaptureHandle::with_cursor_source`;
   the host's `pump_cursor` task (already in
   `apps/tether-host/src/main.rs`) takes it from there. Same pattern
   as `PipeWireCursorSource` on Linux.

## What we know works in our codebase already

- The host pump (`apps/tether-host/src/main.rs::pump_cursor`)
  consumes any `Box<dyn CursorSource>` and forwards shape changes +
  position datagrams. No host-side changes needed.
- The client overlay
  (`crates/tether-render/src/cursor_overlay.rs`) renders whatever the
  wire delivers. No client-side changes needed.
- The protocol (`tether-protocol::cursor::HostCursorPacket`,
  `ControlMessage::CursorShape` / `CursorUseShape`) is final and
  already verified by Linux end-to-end.
- The `CursorPosition`, `CursorShapeEvent`, and `CursorSource` trait
  surface in `crates/tether-capture/src/cursor.rs` is the seam — a
  macOS impl needs nothing more than `next_event` returning
  buffered shape events and `poll_position` returning the latest
  position snapshot.

## Concrete shape of the new code

New file: `crates/tether-capture/src/cursor_macos.rs`.

```rust
pub struct NSCursorCursorSource {
    shape_rx: crossbeam_channel::Receiver<CursorEvent>,
    position_state: Arc<Mutex<Option<CursorPosition>>>,
    // Holds the JoinHandles for the two poll threads so dropping the
    // source stops them.
    _threads: Vec<std::thread::JoinHandle<()>>,
}

impl CursorSource for NSCursorCursorSource { /* mirrors PipeWireCursorSource */ }
```

Construction parallels `PipeWireCursorSource` in `linux.rs:74`.

The `CGSCurrentCursorSeed` binding: weak-link via FFI so a future
macOS that removes the symbol falls back to "always re-fetch every
33 ms" instead of crashing on startup. Pattern:

```rust
extern "C" {
    fn CGSCurrentCursorSeed() -> i32;
}
// Wrapped behind a `seed_supported: bool` runtime check via
// `dlsym(RTLD_DEFAULT, "CGSCurrentCursorSeed")` so missing-symbol
// degrades to the slow path.
```

## Risks the mac dev should know about

1. **`CGSCurrentCursorSeed` is private.** Apple may remove it.
   Production apps (RustDesk, others) have shipped against it for
   years, but the weak-link fallback is mandatory — never assume
   the symbol exists at runtime.
2. **`NSCursor.currentSystemCursor` from a background thread is
   *not* officially blessed.** RustDesk does this in production.
   Apple's threading rules for AppKit are "main thread only" by
   default, but the documented exceptions are unclear and
   `NSCursor.currentSystemCursor` empirically works off-main.
   Recommend: log + retry on null result rather than panic; the
   first poll may need a brief delay after the main runloop spins
   up.
3. **HiDPI / Retina cursors.** `NSCursor.image.size` is in *points*;
   the underlying `NSBitmapImageRep.pixelsWide` is in *pixels*. On
   a Retina display these differ by 2×. Use pixels for the wire
   payload (the client renderer scales relative to video dims
   already) and scale hotspot accordingly. Linux's `SPA_META_Cursor`
   delivers physical pixels too — keep them in the same coordinate
   frame end-to-end.
4. **Code-signing / notarization.** Not blocking for our
   distribution model (CLI binary, not App Store). Note for
   posterity if we ever pursue the App Store route.
5. **Cursor visibility / off-screen pointer.** macOS doesn't have a
   first-class "cursor is hidden" signal we can poll. RustDesk
   doesn't appear to handle this case explicitly. Conservative:
   when `CGEventGetLocation` returns a point outside any
   `NSScreen.frame`, mark `visible = false`. The default-true
   bug would manifest as a phantom cursor at `(0,0)` on the client
   when the pointer leaves the host display.
6. **Multi-display hosts.** The Linux path delivers
   capture-frame-relative coordinates, not screen-relative. On
   macOS we capture one `SCDisplay` at a time today. Make sure the
   cursor position you emit is in the same coordinate frame as the
   currently-captured display — subtract the display's
   `frame.origin` from the `CGEventGetLocation` result before
   sending. Wrong coordinate frame is the easiest silent bug.

## Acceptance criteria

1. Host log shows `cursor pump stats positions_sent=...
   shapes_sent=... seen_shape_ids=...` with non-zero counts during
   active pointer motion.
2. `seen_shape_ids` grows from 1 → 3+ as you hover over a text
   field, then a link. (Confirms `CGSCurrentCursorSeed` is actually
   detecting cross-app cursor changes.)
3. Client visibly draws the correct sprite at the host pointer
   location with no video-round-trip lag.
4. Stopping pointer motion drives the send-stats `kbps_out` near
   zero — proves the cursor separation actually buys the
   idle-bandwidth win (the architectural reason this work exists,
   per the plan doc).
5. Pull the plug on `.with_shows_cursor(true)` at
   `crates/tether-capture/src/macos.rs:286` and confirm the
   captured video is cursor-free (no double cursor on the client).

## Hot files

- `crates/tether-capture/src/cursor.rs` — trait + types.
- `crates/tether-capture/src/macos.rs:286` — the `.with_shows_cursor`
  flip, and `start()` is where the source gets attached to
  `CaptureHandle`.
- `crates/tether-capture/src/cursor_macos.rs` — new module to add.
- `apps/tether-host/src/main.rs::pump_cursor` — consumer, untouched.
- Reference: `~/Code/refs/rustdesk/src/platform/macos.rs:509–638`.
