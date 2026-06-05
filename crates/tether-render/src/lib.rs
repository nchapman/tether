//! Client-side display: a winit window driving a wgpu render pipeline,
//! fed from a [`LatestFrame`] slot. The slot keeps exactly one frame at
//! a time — drop-oldest semantics — because a remote-desktop viewer
//! always wants the most recent picture, not a queued backlog.

pub mod color;
mod cursor_overlay;
// Render backend is platform-specialized: wgpu (Vulkan/Metal) on
// Linux/macOS, native D3D11 on Windows (decode + present stay in one
// API — see `d3d11`). The shared `App` event loop drives whichever
// through the `Backend` alias below.
#[cfg(target_os = "windows")]
mod d3d11;
#[cfg(not(target_os = "windows"))]
mod gpu;
pub mod present_policy;
pub mod relative_mouse;

/// Shared cross-platform colour-bar fixture + assertion used by every
/// platform's round-trip test harness.
#[cfg(test)]
mod color_fixture;

#[cfg(all(test, target_os = "linux"))]
mod dmabuf_test;

#[cfg(all(test, target_os = "macos"))]
mod iosurface_test;

#[cfg(all(test, target_os = "linux"))]
mod test_harness;

use std::sync::{Arc, Mutex};

use std::time::{Duration, Instant};
use tether_codec::{GpuFrameGuard, GpuFrameSource};
use tether_protocol::control::CursorMode;
use tether_protocol::MonoNanos;
use tracing::warn;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowAttributes, WindowId};

// The active render backend. Both expose the same method surface
// (`new`, `resize`, `apply_frame`, `render`, `dimensions`) so the `App`
// loop is identical across platforms.
#[cfg(target_os = "windows")]
use d3d11::D3D11RenderState as Backend;
#[cfg(not(target_os = "windows"))]
use gpu::GpuState as Backend;

// Re-exported so tether-input / tether-client can match on render events
// without having to add their own winit dep at a possibly-different
// version. tether-render's version of winit is the workspace version.
pub use winit::event::MouseButton;
pub use winit::keyboard::{KeyCode, ModifiersState};

/// macOS-only — whether the renderer's IOSurface import path accepts
/// the given `(chroma, bit_depth, fourcc)` triple. The predicate itself
/// lives in `tether_codec::macos_interop` (so the probe can consult it
/// without a render dep); re-exported here for renderer callers. The
/// decode-emit ↔ render-accept agreement is asserted in
/// `tether-probe::host::videotoolbox` and the encoder/bridge agreement
/// in `tether-gpuconvert::nv12_iosurface`. Drift between those tables is
/// the family of bug that shipped a broken 10-bit session in commit
/// `621badc` — fast feedback in default CI is cheaper than catching it
/// in a session.
#[cfg(target_os = "macos")]
pub use gpu::accepts_iosurface_fourcc;

#[cfg(target_os = "windows")]
pub use d3d11::supports_10bit_render;
#[cfg(not(target_os = "windows"))]
pub use gpu::supports_10bit_render;
// Windows decode-format accept table, exported for the cross-crate
// consistency test in tether-client (see `decode_plane_srv_formats`).
#[cfg(target_os = "windows")]
pub use d3d11::decode_plane_srv_formats;

/// Shared cursor state for the overlay render pass. Construct one,
/// hand a clone to the wire-receive side (call `with(|s| s.set_host_visible(...))`
/// / `with(|s| s.enqueue_shape(...))`), pass another clone into
/// [`run`] which threads it to the renderer.
pub use cursor_overlay::{CursorChannel, CursorState};

/// One frame ready for display. Either the pixels live in CPU memory
/// and need an `R8` + `Rg8` upload (the SW decoder path), or they live
/// on the GPU as a DMA-BUF exported from a HW decoder surface and the
/// renderer imports them directly.
#[derive(Debug)]
pub enum Frame {
    Cpu(CpuFrame),
    Gpu(GpuFrame),
}

impl Frame {
    pub fn width(&self) -> u32 {
        match self {
            Frame::Cpu(f) => f.width,
            Frame::Gpu(f) => f.width,
        }
    }
    pub fn height(&self) -> u32 {
        match self {
            Frame::Cpu(f) => f.height,
            Frame::Gpu(f) => f.height,
        }
    }
    pub fn t_capture_client_clock(&self) -> Option<MonoNanos> {
        match self {
            Frame::Cpu(f) => f.t_capture_client_clock,
            Frame::Gpu(f) => f.t_capture_client_clock,
        }
    }
}

/// Frame whose pixels live in CPU memory (SW decoder path), in NV12
/// (Y plus interleaved UV) layout — the canonical output of every
/// hardware H.264 decoder, mirrored on the SW side by a cheap byte
/// interleave so the renderer only knows one format. The renderer
/// uploads `y` as an `R8Unorm` texture and `uv` as a half-resolution
/// `Rg8Unorm` texture, then does the limited-range BT.709 matrix in
/// the fragment shader.
#[derive(Clone, Debug)]
pub struct CpuFrame {
    pub width: u32,
    pub height: u32,
    /// Tight Y plane, `width * height` bytes.
    pub y: Vec<u8>,
    /// Tight UV plane in NV12 layout, `chroma_width * chroma_height * 2`
    /// bytes where `chroma_width = (width + 1) / 2` (and same for
    /// height). Each chroma sample is two bytes — U first, V second —
    /// so a single `Rg8` texture sample on the GPU yields both
    /// channels in one read.
    pub uv: Vec<u8>,
    /// Optional client-clock timestamp of when this frame was
    /// captured at the host (translated through the handshake's
    /// `ClockSync::remote_to_local`, so it shares an epoch with
    /// any `MonoNanos::now()` on this side). When set, the render
    /// loop uses it to log capture-to-present latency once per
    /// second — the second segment of the glass-to-glass budget
    /// that the recv-side latency log can't see. `None` for
    /// callers that don't care (the test_pattern example).
    pub t_capture_client_clock: Option<MonoNanos>,
}

impl CpuFrame {
    /// Chroma plane dimensions (in chroma samples) for the 4:2:0
    /// subsampling we assume. The `uv` buffer is twice this wide in
    /// bytes because each sample carries U and V interleaved.
    #[must_use]
    pub fn chroma_dims(&self) -> (u32, u32) {
        (self.width.div_ceil(2), self.height.div_ceil(2))
    }
}

/// Frame whose pixels live on the GPU. Carries the DMA-BUF descriptor
/// the renderer needs to import the surface, plus the decoder-side
/// release guard that returns the underlying VAAPI surface to the
/// hwframes pool when the renderer is done with it. The `source` is
/// borrowed by `Drop` of the imported wgpu textures; pair them.
#[derive(Debug)]
pub struct GpuFrame {
    pub width: u32,
    pub height: u32,
    pub t_capture_client_clock: Option<MonoNanos>,
    pub source: GpuFrameSource,
    /// Backend-side lifetime extender (typically an `AVFrame`). The
    /// renderer parks this alongside the imported textures and drops
    /// both when the next frame replaces them.
    pub guard: GpuFrameGuard,
}

/// Input-side events surfaced from the window's event loop. The render
/// crate already owns the window and its letterbox transform, so it does
/// the cursor-normalisation math once and hands callers either video-
/// region coordinates or `None` (cursor outside the video region — events
/// downstream of this should be suppressed by the consumer).
#[derive(Clone, Debug)]
pub enum RenderEvent {
    Key {
        code: KeyCode,
        pressed: bool,
        repeat: bool,
        /// The text the OS produced for this keypress, as resolved by
        /// the current keyboard layout and IME. `None` for keys that
        /// don't generate text (modifiers, function keys, IME mid-
        /// composition). Lets the translator route unmodified printable
        /// input to a layout-aware text path while keeping HID for
        /// shortcuts and named keys.
        text: Option<String>,
    },
    Modifiers(ModifiersState),
    /// Cursor moved. `video_normalized` is `Some((x, y))` with both
    /// components in `[0.0, 1.0]` when the pointer is inside the video
    /// region, `None` when it sits in a letterbox bar or outside the
    /// window entirely.
    Cursor {
        video_normalized: Option<(f32, f32)>,
    },
    MouseButton {
        button: MouseButton,
        pressed: bool,
    },
    /// Horizontal + vertical scroll deltas. `by_line` is `true` for
    /// notched-wheel input (winit `LineDelta`), `false` for high-resolution
    /// trackpad / Magic Mouse input (winit `PixelDelta`).
    Scroll {
        dx: f32,
        dy: f32,
        by_line: bool,
    },
    Focused(bool),
    /// Device-level pointer delta (after sub-pixel accumulation),
    /// emitted only while [`crate::present_policy`] is mostly
    /// unrelated — see [`relative_mouse`]. Translated to
    /// `InputEventKind::RelativeMouseMove` on the wire while
    /// `CursorMode::Relative` is active.
    RelativeMouseMove {
        dx: i16,
        dy: i16,
    },
    /// User toggled cursor mode (e.g. via the relative-mode
    /// hotkey). Client binary forwards this as
    /// `ControlMessage::SetCursorMode`; consumer-side decision to
    /// actually grab the pointer happens inside the renderer.
    CursorModeChanged(CursorMode),
    /// Window resized to `(width, height)` physical pixels. The client
    /// binary forwards this to the host as `ControlMessage::SetViewportHint`
    /// (after debouncing) so the host can re-encode at the new dims —
    /// without this hook, the host stays at the original capture
    /// resolution regardless of how small the client window is, wasting
    /// encode time and bandwidth.
    Resized {
        width: u32,
        height: u32,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("event loop: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),
    #[error("winit OS: {0}")]
    Os(#[from] winit::error::OsError),
    #[error("no compatible wgpu adapter")]
    NoAdapter,
    #[error("wgpu request device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    #[error("wgpu surface create: {0}")]
    SurfaceCreate(#[from] wgpu::CreateSurfaceError),
    /// DMA-BUF import failed at the hal layer (driver refused the
    /// modifier, fd dup failed, layout mismatch, etc.). The frame is
    /// dropped; the next decoded frame will try again.
    #[error("dma-buf import: {0}")]
    DmaBufImport(String),
    /// A platform graphics-API call failed outside the import path —
    /// device/swapchain creation, shader compilation, present. Used by
    /// the Windows D3D11 backend, whose errors have no DMA-BUF analogue.
    #[error("graphics API: {0}")]
    GraphicsApi(String),
}

pub type Result<T> = std::result::Result<T, RenderError>;

/// Callback invoked on each input-relevant window event. The render loop
/// fires this synchronously inside `window_event`, so the closure must
/// be cheap — push into a channel and process elsewhere rather than
/// blocking on network IO here. Returning is the only contract; errors
/// inside the callback are the caller's to handle.
pub type EventSink = Box<dyn Fn(RenderEvent) + Send>;

/// Run the render loop on the current thread. Blocks until the user
/// closes the window. The frame channel may disconnect; the window stays
/// open showing the last received frame until the user closes it.
///
/// `color_space` is the `VideoColorSpec` the host advertised in the
/// handshake. Drives the renderer's EOTF dispatch so the decoded
/// frame's bytes are interpreted with the right transfer function.
///
/// `on_event` is optional: pass `None` to skip input-event emission.
/// Using a callback rather than a channel lets the caller bridge into
/// whatever sync/async plumbing they already have.
// Public render entrypoint: window config, negotiated video format, and the
// frame/cursor/event seams are all distinct, intrinsic parameters.
#[allow(clippy::too_many_arguments)]
pub fn run(
    title: &str,
    initial_size: (u32, u32),
    color_space: tether_protocol::control::VideoColorSpec,
    chroma: tether_protocol::control::ChromaSubsampling,
    bit_depth: u8,
    frames: LatestFrame,
    cursor_channel: CursorChannel,
    on_event: Option<EventSink>,
) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = App {
        title: title.to_string(),
        initial_size,
        color_space,
        chroma,
        bit_depth,
        window: None,
        gpu: None,
        frames,
        latest: None,
        on_event,
        present_stats: PresentStats::default(),
        last_recorded_t_cap: None,
        refresh_rate_mhz: present_policy::REFRESH_RATE_FALLBACK_HZ.saturating_mul(1000),
        age_tracker: present_policy::FrameAgeTracker::default(),
        health: RenderHealth::default(),
        cursor_mode: CursorMode::Absolute,
        relative_accum: relative_mouse::SubPixelAccum::default(),
        local_cursor_hidden: false,
        ctrl_held: false,
        alt_held: false,
        cursor_channel,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    title: String,
    initial_size: (u32, u32),
    color_space: tether_protocol::control::VideoColorSpec,
    chroma: tether_protocol::control::ChromaSubsampling,
    bit_depth: u8,
    window: Option<Arc<Window>>,
    gpu: Option<Backend>,
    frames: LatestFrame,
    latest: Option<Frame>,
    on_event: Option<EventSink>,
    /// Per-frame present-latency stats: sum / count / start.
    /// Flushed once a second so a long-running session doesn't
    /// silently accumulate floating-point error. `None` until we've
    /// presented at least one timestamped frame.
    present_stats: PresentStats,
    /// Capture timestamp of the most recently *recorded* frame, used
    /// to dedup present-stats samples when the OS fires
    /// `RedrawRequested` without a new frame having arrived (focus
    /// change, expose, etc.). Without this, every OS-initiated
    /// redraw would record the same age again, inflating the sample
    /// count and pulling the average up.
    last_recorded_t_cap: Option<MonoNanos>,
    /// Cached monitor refresh rate, in millihertz. Queried on
    /// resume + monitor change. Falls back to 60 Hz when winit
    /// can't report a real value.
    refresh_rate_mhz: u32,
    /// Frame-age skip policy state. See [`present_policy`].
    age_tracker: present_policy::FrameAgeTracker,
    /// End-to-end present-health counters. Distinct from
    /// `present_stats` (which only ticks when a frame is actually
    /// presented): this is ticked every event-loop turn so a *stall* —
    /// frames decoding but nothing reaching the screen — surfaces as a
    /// loud log line instead of the silent absence of `present stats`.
    health: RenderHealth,
    /// Cursor input model. Toggled by Ctrl+Alt+G; on
    /// transition to `Relative` we grab + hide the cursor and
    /// route `DeviceEvent::MouseMotion` through `relative_accum`.
    cursor_mode: CursorMode,
    relative_accum: relative_mouse::SubPixelAccum,
    /// Whether we've hidden the local OS cursor. In absolute mode we
    /// hide it exactly while the host-cursor overlay draws in its place
    /// (pointer over the video, host shows a cursor), and show it again
    /// over letterbox bars / off-window. Tracked so we only call winit's
    /// `set_cursor_visible` on an actual change, not every motion event.
    local_cursor_hidden: bool,
    /// Modifier-key edge tracker for the hotkey. We watch
    /// `WindowEvent::ModifiersChanged` for the actual state but
    /// also need the Ctrl+Alt+G three-key combo to fire on the
    /// `G` keydown specifically.
    ctrl_held: bool,
    alt_held: bool,
    /// Shared cursor state for the overlay render pass. The client's
    /// wire-receive task writes sprite cache + position; the renderer
    /// reads each frame. Bypassed when no producer ever fires (the
    /// existing test_pattern example passes a detached default).
    cursor_channel: CursorChannel,
}

#[derive(Default)]
struct PresentStats {
    sum_ns: u128,
    samples: u32,
    window_start: Option<Instant>,
}

impl PresentStats {
    fn record_and_maybe_log(&mut self, latency_ns: u64) {
        self.sum_ns += u128::from(latency_ns);
        self.samples += 1;
        let now = Instant::now();
        let start = *self.window_start.get_or_insert(now);
        if now.duration_since(start) >= Duration::from_secs(1) {
            let avg_ms = if self.samples == 0 {
                0.0
            } else {
                #[allow(clippy::cast_precision_loss)]
                let sum_f = self.sum_ns as f64;
                sum_f / f64::from(self.samples) / 1_000_000.0
            };
            tracing::info!(
                samples = self.samples,
                avg_present_latency_ms = avg_ms,
                "present stats"
            );
            self.sum_ns = 0;
            self.samples = 0;
            self.window_start = Some(now);
        }
    }
}

/// End-to-end present-health accounting for one ~1-second window.
///
/// `present_stats` answers "how fast are presented frames?" but only
/// exists while frames *are* being presented. `RenderHealth` answers the
/// orthogonal question "are decoded frames actually reaching the
/// screen?" — and is ticked unconditionally every event-loop turn so the
/// answer is logged even when it's "no." The green-screen-until-resize
/// bug presented exactly this way: 36 fps decoded, 0 fps presented, and
/// every stage counter reading healthy because the un-drawn frames were
/// overwritten in a path nobody counted.
#[derive(Default)]
struct RenderHealth {
    /// Decoded frames handed to the renderer this window.
    arrived: u32,
    /// Frames actually applied + presented this window.
    presented: u32,
    /// Frames overwritten in the single-frame slot before they could be
    /// drawn — a silent drop. Should be ~0 now that `about_to_wait`
    /// presents on arrival; a non-zero value means frames are landing
    /// faster than we can draw, or arriving before the GPU is ready.
    dropped_undrawn: u32,
    window_start: Option<Instant>,
}

impl RenderHealth {
    /// A decoded frame reached the renderer. `displaced_undrawn` is true
    /// if it overwrote a prior frame that was never presented.
    fn frame_arrived(&mut self, displaced_undrawn: bool) {
        self.arrived = self.arrived.saturating_add(1);
        if displaced_undrawn {
            self.dropped_undrawn = self.dropped_undrawn.saturating_add(1);
        }
    }

    /// A new frame was applied and presented to the screen.
    fn frame_presented(&mut self) {
        self.presented = self.presented.saturating_add(1);
    }

    /// Emit at most one health line per second. Called every event-loop
    /// turn; the window clock starts on the first call, not on first
    /// frame arrival. A window with no arrivals classifies as `Idle` and
    /// resets silently (no noise pre-connection or while paused); windows
    /// with activity log every second — loudly when frames are arriving
    /// but none are reaching the screen.
    fn maybe_log(&mut self) {
        let now = Instant::now();
        let start = *self.window_start.get_or_insert(now);
        if now.duration_since(start) < Duration::from_secs(1) {
            return;
        }
        match classify_health(self.arrived, self.presented) {
            HealthVerdict::Idle => {}
            HealthVerdict::Stalled => tracing::warn!(
                arrived = self.arrived,
                dropped_undrawn = self.dropped_undrawn,
                "render stalled: frames decoded but none reached the screen"
            ),
            HealthVerdict::Healthy => tracing::info!(
                presented_fps = self.presented,
                arrived = self.arrived,
                dropped_undrawn = self.dropped_undrawn,
                "render stats"
            ),
        }
        self.arrived = 0;
        self.presented = 0;
        self.dropped_undrawn = 0;
        self.window_start = Some(now);
    }
}

/// Verdict for one completed health window. Split out from
/// [`RenderHealth::maybe_log`] so the decode-vs-present decision is
/// unit-testable without a clock.
#[derive(Debug, PartialEq, Eq)]
enum HealthVerdict {
    /// No frames arrived or presented — idle / pre-connection. No log.
    Idle,
    /// Frames reached the screen this window.
    Healthy,
    /// Frames arrived from decode but none were presented — a stall
    /// (the green-screen signature).
    Stalled,
}

fn classify_health(arrived: u32, presented: u32) -> HealthVerdict {
    match (arrived, presented) {
        (0, 0) => HealthVerdict::Idle,
        (_, 0) => HealthVerdict::Stalled,
        _ => HealthVerdict::Healthy,
    }
}

impl App {
    fn emit(&self, event: RenderEvent) {
        if let Some(cb) = &self.on_event {
            cb(event);
        }
    }

    /// Try `Locked` first (true pointer-lock, supported on most
    /// platforms in 2026), fall back to `Confined` on X11/Wayland
    /// combos that reject Locked. Cursor visibility is handled
    /// separately via [`set_local_cursor_hidden`](Self::set_local_cursor_hidden).
    fn apply_cursor_grab(&self, window: &Window) {
        if window
            .set_cursor_grab(winit::window::CursorGrabMode::Locked)
            .is_err()
        {
            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
        }
    }

    /// Show or hide the local OS cursor, calling winit only on an actual
    /// state change (`WindowEvent::CursorMoved` fires this every motion
    /// event, so the dedup matters).
    fn set_local_cursor_hidden(&mut self, hidden: bool) {
        if self.local_cursor_hidden == hidden {
            return;
        }
        self.local_cursor_hidden = hidden;
        if let Some(window) = &self.window {
            window.set_cursor_visible(!hidden);
        }
    }

    fn toggle_cursor_mode(&mut self) {
        let new_mode = match self.cursor_mode {
            CursorMode::Absolute => CursorMode::Relative,
            CursorMode::Relative => CursorMode::Absolute,
        };
        self.cursor_mode = new_mode;
        // Mirror into the renderer's cursor state: suppress overlay
        // drawing in relative mode (we render our own locked pointer),
        // and clear the local-pointer anchor so a stale over-video
        // position from the prior mode can't briefly draw the overlay
        // before the next `CursorMoved`.
        self.cursor_channel.with(|state| {
            state.set_relative_mode(matches!(new_mode, CursorMode::Relative));
            state.set_local_pointer(None);
        });
        // Drop sub-pixel residue so stale fractional motion from
        // the prior mode doesn't leak into the first delta.
        self.relative_accum.reset();
        if let Some(window) = &self.window {
            match new_mode {
                CursorMode::Relative => self.apply_cursor_grab(window),
                CursorMode::Absolute => {
                    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                }
            }
        }
        match new_mode {
            // Relative locks + hides the pointer; we render our own.
            CursorMode::Relative => self.set_local_cursor_hidden(true),
            // Show the OS cursor now; the next `CursorMoved` re-hides it
            // if the pointer is over the video and the overlay draws.
            CursorMode::Absolute => self.set_local_cursor_hidden(false),
        }
        self.emit(RenderEvent::CursorModeChanged(new_mode));
    }

    /// Apply the freshest pending frame (if any) and present.
    ///
    /// Driven from two places: `about_to_wait` calls this directly the
    /// instant a decoded frame lands (frame-arrival is the pacing
    /// signal), and `RedrawRequested` calls it for OS-initiated repaints
    /// (expose/resize), which re-present the last applied frame without a
    /// new upload. We deliberately do NOT rely on `request_redraw()` to
    /// drive steady-state liveness: on Wayland a redraw request only
    /// turns into `RedrawRequested` when the compositor delivers a frame
    /// callback, and that cadence doesn't reliably bootstrap before the
    /// first real window event — leaving decoded frames undrawn (a green
    /// screen) until the user happens to resize. Presenting here makes
    /// liveness depend on our own decode loop, not the compositor.
    fn draw(&mut self) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        // Apply a new frame at most once per draw. GpuState holds the
        // imported/uploaded textures across draws, so OS-initiated
        // redraws (expose, focus) re-render the most recently applied
        // frame without re-uploading or re-importing.
        //
        // Before applying, run the frame-age policy: a sufficiently-stale
        // frame (>~1.5× refresh period, streak gated) is dropped on the
        // floor so the refresh slot can carry whatever lands next instead
        // of stale content the user has already adjusted to. Frames
        // without a timestamp (test_pattern example) always render — no
        // policy input to evaluate.
        let mut skipped = false;
        let mut applied_new = false;
        if let Some(frame) = self.latest.take() {
            let t_capture = frame.t_capture_client_clock();
            let now = MonoNanos::now();
            if let Some(t_cap) = t_capture {
                let age_ns = now.saturating_sub(t_cap);
                let decision = present_policy::decide_present(
                    age_ns,
                    self.refresh_rate_mhz,
                    &mut self.age_tracker,
                );
                if matches!(decision, present_policy::PresentDecision::Skip) {
                    skipped = true;
                }
            }
            if !skipped {
                // Only count a present in `RenderHealth` if the upload
                // actually succeeded — otherwise a persistent apply
                // failure (DMA-BUF import error) would report Healthy
                // while the screen shows stale content.
                match gpu.apply_frame(frame) {
                    Ok(()) => applied_new = true,
                    Err(e) => warn!(error = ?e, "applying frame failed"),
                }
            }
            if let Some(t_cap) = t_capture.filter(|_| !skipped) {
                // Sample present latency only for frames we actually
                // applied. Dedup against `last_recorded_t_cap` so
                // OS-initiated redraws don't double-count.
                if self.last_recorded_t_cap != t_capture {
                    let latency = MonoNanos::now().saturating_sub(t_cap);
                    self.present_stats.record_and_maybe_log(latency);
                    self.last_recorded_t_cap = Some(t_cap);
                }
            }
        }
        if !skipped {
            if let Err(e) = gpu.render() {
                warn!(error = ?e, "render frame failed");
            }
            if applied_new {
                self.health.frame_presented();
            }
        } else {
            tracing::trace!(
                late_streak = self.age_tracker.late_streak,
                drops = self.age_tracker.drops_in_window,
                "skipped stale frame"
            );
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title(&self.title)
            .with_inner_size(LogicalSize::new(self.initial_size.0, self.initial_size.1));
        let win = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                tracing::error!(error = %e, "failed to create window");
                event_loop.exit();
                return;
            }
        };
        let gpu = match pollster::block_on(Backend::new(
            win.clone(),
            self.color_space,
            self.chroma,
            self.bit_depth,
            self.cursor_channel.clone(),
        )) {
            Ok(g) => g,
            Err(e) => {
                tracing::error!(error = %e, "failed to initialise render backend");
                event_loop.exit();
                return;
            }
        };
        // Cache the monitor refresh rate. Winit returns an
        // `Option<u32>` in millihertz; fall back to 60 Hz when
        // it's unavailable (some Wayland compositors don't expose
        // it on every monitor).
        if let Some(monitor) = self.window.as_ref().and_then(|_| win.current_monitor()) {
            if let Some(mhz) = monitor.refresh_rate_millihertz() {
                self.refresh_rate_mhz = mhz;
            }
        } else if let Some(mhz) = win
            .current_monitor()
            .and_then(|m| m.refresh_rate_millihertz())
        {
            self.refresh_rate_mhz = mhz;
        }
        self.window = Some(win);
        self.gpu = Some(gpu);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Existence guard only — individual arms re-acquire `gpu` locally
        // so `RedrawRequested` can call `self.draw()` (which borrows all
        // of `self`) without colliding with a function-wide `gpu` borrow.
        if self.gpu.is_none() {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size.width, size.height);
                }
                self.emit(RenderEvent::Resized {
                    width: size.width,
                    height: size.height,
                });
            }
            WindowEvent::RedrawRequested => {
                // OS-initiated repaint (expose/resize). Re-present the
                // last applied frame; steady-state liveness is driven by
                // `about_to_wait` instead — see `draw`.
                self.draw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                // PhysicalKey + the resolved text travel together: HID
                // for shortcuts, text for layout-aware typing. The
                // translator decides which path each event takes.
                if let PhysicalKey::Code(code) = event.physical_key {
                    // Hotkey gate: Ctrl+Alt+G toggles cursor mode.
                    // Consume the G keydown on this combo so the
                    // host doesn't receive a phantom "G" press.
                    if code == KeyCode::KeyG
                        && event.state == ElementState::Pressed
                        && !event.repeat
                        && self.ctrl_held
                        && self.alt_held
                    {
                        self.toggle_cursor_mode();
                        return;
                    }
                    self.emit(RenderEvent::Key {
                        code,
                        pressed: event.state == ElementState::Pressed,
                        repeat: event.repeat,
                        text: event.text.as_deref().map(str::to_owned),
                    });
                }
            }
            WindowEvent::ModifiersChanged(m) => {
                self.ctrl_held = m.state().control_key();
                self.alt_held = m.state().alt_key();
                self.emit(RenderEvent::Modifiers(m.state()));
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Suppress absolute-cursor reports while the
                // pointer is grabbed for relative mode — the
                // DeviceEvent::MouseMotion path is authoritative.
                if matches!(self.cursor_mode, CursorMode::Relative) {
                    return;
                }
                let Some(gpu) = self.gpu.as_ref() else {
                    return;
                };
                let (texture, surface) = gpu.dimensions();
                let video_normalized = cursor_to_video_normalized(position, surface, texture);
                // Anchor the host-cursor overlay to the local pointer for
                // zero-latency motion (the host still gets the absolute
                // position below, to move its real cursor). Convert the
                // normalized position into the video-pixel space the
                // overlay shader expects, then hide the local OS cursor
                // exactly when the overlay draws in its place.
                #[allow(clippy::cast_precision_loss)]
                let local_px =
                    video_normalized.map(|(nx, ny)| (nx * texture.0 as f32, ny * texture.1 as f32));
                let overlay_active = self.cursor_channel.with(|state| {
                    state.set_local_pointer(local_px);
                    state.overlay_active()
                });
                let was_hidden = self.local_cursor_hidden;
                // Known edge: if the host flips `visible` false after this
                // `overlay_active` read but before the next render, the OS
                // cursor stays hidden while nothing draws (no cursor) until
                // the next motion event re-evaluates. Self-correcting and
                // rare (the host's cursor stays in-bounds while driven), so
                // not worth cross-thread redraw plumbing.
                self.set_local_cursor_hidden(overlay_active);
                // Cursor-only motion produces no new video frame (the host
                // drops idle frames; the cursor is out-of-band), so the
                // frame-arrival present loop won't redraw the moved sprite.
                // Request a coalesced redraw while the overlay is drawing,
                // plus one final redraw when it just stopped (to erase the
                // last sprite). winit collapses bursts into one present per
                // refresh, so a high-rate mouse can't over-present.
                if overlay_active || was_hidden {
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                self.emit(RenderEvent::Cursor { video_normalized });
            }
            WindowEvent::CursorLeft { .. } => {
                // Pointer left the window: drop the overlay anchor and
                // restore the local OS cursor. (Relative mode keeps the
                // pointer grabbed, so this won't fire there.)
                self.cursor_channel
                    .with(|state| state.set_local_pointer(None));
                let needs_erase = self.local_cursor_hidden;
                self.set_local_cursor_hidden(false);
                if needs_erase {
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                self.emit(RenderEvent::MouseButton {
                    button,
                    pressed: state == ElementState::Pressed,
                });
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let (dx, dy, by_line) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x, y, true),
                    MouseScrollDelta::PixelDelta(p) => {
                        #[allow(clippy::cast_possible_truncation)]
                        let x = p.x as f32;
                        #[allow(clippy::cast_possible_truncation)]
                        let y = p.y as f32;
                        (x, y, false)
                    }
                };
                self.emit(RenderEvent::Scroll { dx, dy, by_line });
            }
            WindowEvent::Focused(b) => {
                // Window blur → release the grab so the user can
                // get their cursor back to switch apps. Don't
                // change `cursor_mode` itself — on focus-regain
                // we re-acquire the grab if still in Relative.
                if let Some(window) = &self.window {
                    if !b {
                        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                    } else if matches!(self.cursor_mode, CursorMode::Relative) {
                        // Re-acquire on focus regain.
                        self.apply_cursor_grab(window);
                    }
                }
                if !b {
                    // Blur: hand the OS cursor back and drop the overlay
                    // anchor, regardless of mode.
                    self.cursor_channel
                        .with(|state| state.set_local_pointer(None));
                    self.set_local_cursor_hidden(false);
                } else if matches!(self.cursor_mode, CursorMode::Relative) {
                    self.set_local_cursor_hidden(true);
                }
                self.emit(RenderEvent::Focused(b));
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        // Device-level pointer motion fires even when the cursor
        // is grabbed (locked or confined), which is exactly the
        // raw-input shape recenter-loop games need. Only emit
        // while we're actually in Relative mode; otherwise the
        // host already gets absolute reports via
        // `WindowEvent::CursorMoved`.
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if matches!(self.cursor_mode, CursorMode::Relative) {
                if let Some((dxi, dyi)) = self.relative_accum.record(dx, dy) {
                    self.emit(RenderEvent::RelativeMouseMove { dx: dxi, dy: dyi });
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // LatestFrame holds at most one frame — if the producer wrote
        // multiple times since we last polled, only the newest is
        // visible. That's the intended drop-oldest semantics: a
        // remote-desktop viewer wants the freshest picture, not a
        // queued backlog.
        //
        // Present directly rather than via `request_redraw()`: under
        // `ControlFlow::Poll` this runs every loop turn, so frame arrival
        // paces presentation. Relying on `request_redraw()` here left
        // decoded frames undrawn on Wayland until the first window event
        // (the green-screen-until-resize bug) because a redraw request
        // only becomes `RedrawRequested` once the compositor delivers a
        // frame callback.
        if let Some(frame) = self.frames.take() {
            // A prior frame still in the slot here is one `draw()` never
            // consumed (e.g. arrived before the GPU was ready) — a silent
            // drop worth counting, not hiding.
            let displaced_undrawn = self.latest.replace(frame).is_some();
            self.health.frame_arrived(displaced_undrawn);
            self.draw();
        }
        // Ticked every turn — independent of whether anything was drawn —
        // so a present stall logs instead of going silent.
        self.health.maybe_log();
    }
}

/// Single-slot, latest-wins frame channel between the decoder thread
/// and the renderer. Replaces a bounded queue: a remote-desktop
/// viewer never benefits from rendering a stale frame when a newer
/// one is available. The producer side `set`s, optionally caring
/// about the displaced frame (e.g. to bump a drop counter); the
/// renderer `take`s once per redraw cycle.
///
/// Cheap clone (`Arc` inside). Both sides hold a clone.
#[derive(Clone, Default)]
pub struct LatestFrame(Arc<Mutex<Option<Frame>>>);

impl LatestFrame {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace whatever frame is currently held. Returns the
    /// displaced frame if one was present — the producer can use
    /// that for a drop count or just drop it.
    #[must_use = "displaced frame should be counted as a render drop or explicitly ignored"]
    pub fn set(&self, frame: Frame) -> Option<Frame> {
        (*self.0.lock().expect("LatestFrame mutex poisoned")).replace(frame)
    }

    /// Take the currently-held frame, leaving the slot empty.
    pub fn take(&self) -> Option<Frame> {
        self.0.lock().expect("LatestFrame mutex poisoned").take()
    }
}

/// Map a window-pixel cursor position onto the video region in `[0,1]^2`,
/// accounting for the same letterbox / pillarbox transform the GPU shader
/// applies. Returns `None` when the cursor sits in a letterbox bar
/// (outside the video region) or when either size is degenerate.
///
/// Mirrors `gpu::letterbox_scale`: when source and surface aspect ratios
/// match, the whole window is the video region; otherwise the video is
/// centered and one axis is shrunk by `min_aspect / max_aspect`.
#[allow(clippy::cast_precision_loss)]
fn cursor_to_video_normalized(
    pos: PhysicalPosition<f64>,
    surface: (u32, u32),
    texture: (u32, u32),
) -> Option<(f32, f32)> {
    if surface.0 == 0 || surface.1 == 0 || texture.0 == 0 || texture.1 == 0 {
        return None;
    }
    let (sx, sy) = letterbox_scale(texture, surface);
    let sw = f64::from(surface.0);
    let sh = f64::from(surface.1);
    let (sx, sy) = (f64::from(sx), f64::from(sy));
    let video_w = sw * sx;
    let video_h = sh * sy;
    let offset_x = (sw - video_w) * 0.5;
    let offset_y = (sh - video_h) * 0.5;
    let nx = (pos.x - offset_x) / video_w;
    let ny = (pos.y - offset_y) / video_h;
    if !(0.0..=1.0).contains(&nx) || !(0.0..=1.0).contains(&ny) {
        return None;
    }
    #[allow(clippy::cast_possible_truncation)]
    Some((nx as f32, ny as f32))
}

/// Aspect-preserving letterbox / pillarbox scale: the `(x, y)` NDC scale
/// that fits a `src`-sized image inside a `dst`-sized surface centered,
/// shrinking one axis by `min_aspect / max_aspect`. Shared by the cursor
/// mapping above and the Windows D3D11 backend's vertex scale so the two
/// can't drift; the wgpu backend keeps its own private copy (it's cfg'd
/// out on Windows, and widening its visibility buys nothing there).
#[allow(clippy::cast_precision_loss)]
pub(crate) fn letterbox_scale(src: (u32, u32), dst: (u32, u32)) -> (f32, f32) {
    let src_aspect = src.0 as f32 / src.1 as f32;
    let dst_aspect = dst.0 as f32 / dst.1 as f32;
    if (src_aspect - dst_aspect).abs() < f32::EPSILON {
        (1.0, 1.0)
    } else if src_aspect > dst_aspect {
        (1.0, dst_aspect / src_aspect)
    } else {
        (src_aspect / dst_aspect, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_centre_maps_to_centre() {
        let n = cursor_to_video_normalized(
            PhysicalPosition::new(640.0, 360.0),
            (1280, 720),
            (1920, 1080),
        )
        .expect("centre is inside the video region");
        assert!((n.0 - 0.5).abs() < 1e-4);
        assert!((n.1 - 0.5).abs() < 1e-4);
    }

    #[test]
    fn cursor_in_letterbox_bar_returns_none() {
        // 1000x1000 window, 1920x1080 source -> top/bottom letterbox
        // (source is wider than the window's square aspect). y=10 lands
        // in the top bar.
        assert!(cursor_to_video_normalized(
            PhysicalPosition::new(500.0, 10.0),
            (1000, 1000),
            (1920, 1080),
        )
        .is_none());
        // 1000x1000 window, 1080x1920 source -> left/right pillarbox.
        // x=10 lands in the left bar.
        assert!(cursor_to_video_normalized(
            PhysicalPosition::new(10.0, 500.0),
            (1000, 1000),
            (1080, 1920),
        )
        .is_none());
    }

    #[test]
    fn cursor_outside_window_returns_none() {
        assert!(cursor_to_video_normalized(
            PhysicalPosition::new(-5.0, 100.0),
            (1280, 720),
            (1280, 720),
        )
        .is_none());
    }

    #[test]
    fn matching_aspect_uses_full_window() {
        let n = cursor_to_video_normalized(
            PhysicalPosition::new(128.0, 72.0),
            (1280, 720),
            (1920, 1080),
        )
        .expect("inside");
        assert!((n.0 - 0.1).abs() < 1e-4);
        assert!((n.1 - 0.1).abs() < 1e-4);
    }

    /// Compile-time assertion that `LatestFrame` actually meets the
    /// `Send + Sync` bound it implicitly relies on for cross-thread
    /// use (decode std::thread → renderer winit thread). A regression
    /// to the inner type that broke this would surface as a build
    /// error here rather than at the use site.
    #[test]
    fn latest_frame_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LatestFrame>();
    }

    #[test]
    fn latest_frame_displaces_previous_on_set() {
        let frames = LatestFrame::new();
        let frame_a = Frame::Cpu(CpuFrame {
            width: 1,
            height: 1,
            y: vec![0xa],
            uv: vec![0, 0],
            t_capture_client_clock: None,
        });
        let frame_b = Frame::Cpu(CpuFrame {
            width: 2,
            height: 2,
            y: vec![0xb],
            uv: vec![0, 0],
            t_capture_client_clock: None,
        });
        // First set: empty slot, no displacement.
        assert!(
            frames.set(frame_a).is_none(),
            "empty slot should return None"
        );
        // Second set: A is displaced; caller can count this as a render drop.
        let displaced = frames.set(frame_b).expect("second set should displace");
        match displaced {
            Frame::Cpu(f) => assert_eq!(f.width, 1, "displaced frame should be A"),
            _ => panic!("expected Cpu frame"),
        }
        // take() yields the latest (B), then empties the slot.
        let latest = frames.take().expect("slot should hold B");
        match latest {
            Frame::Cpu(f) => assert_eq!(f.width, 2),
            _ => panic!("expected Cpu frame"),
        }
        assert!(frames.take().is_none(), "slot should be empty after take");
    }

    #[test]
    fn health_idle_window_is_silent() {
        // No decode activity and nothing presented: pre-connection or a
        // paused stream. Must not log (Idle), or every idle second spams.
        assert_eq!(classify_health(0, 0), HealthVerdict::Idle);
    }

    #[test]
    fn health_frames_presented_is_healthy() {
        assert_eq!(classify_health(30, 30), HealthVerdict::Healthy);
        // Presenting fewer than arrived (drop-oldest under load) is still
        // healthy — pixels are reaching the screen.
        assert_eq!(classify_health(60, 45), HealthVerdict::Healthy);
    }

    #[test]
    fn health_decoding_without_presenting_is_a_stall() {
        // The green-screen signature: frames arriving from decode, none
        // reaching the screen. This is the case that must be loud.
        assert_eq!(classify_health(36, 0), HealthVerdict::Stalled);
    }

    #[test]
    fn health_counts_undrawn_displacement_as_drop() {
        let mut h = RenderHealth::default();
        // Frame arrives into an empty slot: no silent drop.
        h.frame_arrived(false);
        // Next frame overwrites one that was never drawn: a silent drop.
        h.frame_arrived(true);
        assert_eq!(h.arrived, 2);
        assert_eq!(h.dropped_undrawn, 1);
    }
}
