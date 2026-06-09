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
use tether_protocol::control::{ClientDisplayMetrics, CursorMode, DisplayMode};
use tether_protocol::MonoNanos;
use tracing::warn;
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentationMode {
    /// Fit inside the client surface without growing past logical 100%.
    Fit,
    /// Present at logical 100%. A smaller surface clips the centered image.
    ActualSize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisplayScale {
    pub num: u16,
    pub den: u16,
}

impl DisplayScale {
    #[must_use]
    pub const fn one() -> Self {
        Self { num: 1, den: 1 }
    }

    #[must_use]
    pub const fn new(num: u16, den: u16) -> Option<Self> {
        if num == 0 || den == 0 {
            None
        } else {
            Some(Self { num, den })
        }
    }

    #[must_use]
    pub fn as_f64(self) -> f64 {
        debug_assert!(self.num > 0);
        debug_assert!(self.den > 0);
        f64::from(self.num) / f64::from(self.den)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostDisplayGeometry {
    pub size_px: (u32, u32),
    pub scale: DisplayScale,
}

impl HostDisplayGeometry {
    #[must_use]
    pub const fn new(size_px: (u32, u32), scale: DisplayScale) -> Option<Self> {
        if size_px.0 == 0 || size_px.1 == 0 || scale.num == 0 || scale.den == 0 {
            None
        } else {
            Some(Self { size_px, scale })
        }
    }
}

#[derive(Clone, Debug)]
pub struct HostDisplayHandle {
    geometry: Arc<Mutex<HostDisplayGeometry>>,
}

impl HostDisplayHandle {
    #[must_use]
    pub fn new(geometry: HostDisplayGeometry) -> Self {
        Self {
            geometry: Arc::new(Mutex::new(geometry)),
        }
    }

    #[must_use]
    pub fn get(&self) -> HostDisplayGeometry {
        *self
            .geometry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn set(&self, geometry: HostDisplayGeometry) {
        *self
            .geometry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = geometry;
    }
}

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
pub use d3d11::{supports_10bit_render, supports_video_profile_render};
#[cfg(not(target_os = "windows"))]
pub use gpu::{supports_10bit_render, supports_video_profile_render};
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
    /// Window resized. `surface_*` is the winit physical-pixel window size;
    /// `viewport_*` is the density/presentation-mode-corrected rectangle the
    /// client forwards to the host as `ControlMessage::SetViewportHint` after
    /// debouncing.
    Resized {
        surface_width: u32,
        surface_height: u32,
        viewport_width: u32,
        viewport_height: u32,
        presentation_width: u32,
        presentation_height: u32,
        host_width: u32,
        host_height: u32,
        host_scale_num: u16,
        host_scale_den: u16,
        client_scale_factor: f64,
        presentation_mode: PresentationMode,
    },
    /// Physical metrics for the client output hosting the renderer. The client
    /// forwards this to the host as display-mode-matching input; it is not a
    /// host mode-change request.
    ClientDisplayMetrics(ClientDisplayMetrics),
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
    initial_video_size_px: (u32, u32),
    host_display_scale: DisplayScale,
    color_space: tether_protocol::control::VideoColorSpec,
    chroma: tether_protocol::control::ChromaSubsampling,
    bit_depth: u8,
    presentation_mode: PresentationMode,
    frames: LatestFrame,
    cursor_channel: CursorChannel,
    on_event: Option<EventSink>,
) -> Result<()> {
    let geometry = HostDisplayGeometry::new(initial_video_size_px, host_display_scale)
        .expect("tether_render::run requires nonzero host display size and display scale");
    run_with_host_display_handle(
        title,
        HostDisplayHandle::new(geometry),
        color_space,
        chroma,
        bit_depth,
        presentation_mode,
        frames,
        cursor_channel,
        on_event,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_with_host_display_handle(
    title: &str,
    host_display: HostDisplayHandle,
    color_space: tether_protocol::control::VideoColorSpec,
    chroma: tether_protocol::control::ChromaSubsampling,
    bit_depth: u8,
    presentation_mode: PresentationMode,
    frames: LatestFrame,
    cursor_channel: CursorChannel,
    on_event: Option<EventSink>,
) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let last_host_display = host_display.get();
    let mut app = App {
        title: title.to_string(),
        host_display,
        last_host_display,
        client_scale_factor: 1.0,
        color_space,
        chroma,
        bit_depth,
        presentation_mode,
        window: None,
        gpu: None,
        frames,
        latest: None,
        on_event,
        present_stats: PresentStats::default(),
        last_recorded_t_cap: None,
        last_display_metrics: None,
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
    host_display: HostDisplayHandle,
    last_host_display: HostDisplayGeometry,
    client_scale_factor: f64,
    color_space: tether_protocol::control::VideoColorSpec,
    chroma: tether_protocol::control::ChromaSubsampling,
    bit_depth: u8,
    presentation_mode: PresentationMode,
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
    /// Last client-display metrics emitted to the session. Monitor
    /// changes can arrive as resize, move, scale, or focus events
    /// depending on the window system; dedup here so all of those hooks
    /// can cheaply refresh without spamming control messages.
    last_display_metrics: Option<ClientDisplayMetrics>,
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

    fn refresh_client_display_metrics(&mut self) {
        let Some(monitor) = self
            .window
            .as_ref()
            .and_then(|window| window.current_monitor())
        else {
            return;
        };
        self.refresh_client_display_metrics_for_monitor(&monitor);
    }

    fn refresh_client_display_metrics_for_monitor(
        &mut self,
        monitor: &winit::monitor::MonitorHandle,
    ) {
        if let Some(mhz) = monitor.refresh_rate_millihertz() {
            self.refresh_rate_mhz = mhz;
        }
        self.client_scale_factor = sanitize_scale_factor(monitor.scale_factor());
        self.update_presentation_size();
        let metrics = client_display_metrics_for_monitor(monitor);
        if !client_display_metrics_changed(self.last_display_metrics.as_ref(), &metrics) {
            return;
        }
        self.last_display_metrics = Some(metrics.clone());
        self.emit(RenderEvent::ClientDisplayMetrics(metrics));
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
            CursorMode::Relative | CursorMode::Unknown(_) => CursorMode::Absolute,
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
                CursorMode::Absolute | CursorMode::Unknown(_) => {
                    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                }
            }
        }
        match new_mode {
            // Relative locks + hides the pointer; we render our own.
            CursorMode::Relative => self.set_local_cursor_hidden(true),
            // Show the OS cursor now; the next `CursorMoved` re-hides it
            // if the pointer is over the video and the overlay draws.
            CursorMode::Absolute | CursorMode::Unknown(_) => self.set_local_cursor_hidden(false),
        }
        self.emit(RenderEvent::CursorModeChanged(new_mode));
    }

    fn apply_window_resize(&mut self, size: PhysicalSize<u32>) {
        let viewport_size = self.viewport_size_for_surface(size);
        self.apply_window_resize_with_viewport(size, viewport_size);
    }

    fn apply_window_resize_with_viewport(
        &mut self,
        surface_size: PhysicalSize<u32>,
        viewport_size: PhysicalSize<u32>,
    ) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.resize(surface_size.width, surface_size.height);
        }
        let presentation_size_px = self.presentation_size_px();
        let host_display = self.host_display.get();
        self.emit(RenderEvent::Resized {
            surface_width: surface_size.width,
            surface_height: surface_size.height,
            viewport_width: viewport_size.width,
            viewport_height: viewport_size.height,
            presentation_width: presentation_size_px.0,
            presentation_height: presentation_size_px.1,
            host_width: host_display.size_px.0,
            host_height: host_display.size_px.1,
            host_scale_num: host_display.scale.num,
            host_scale_den: host_display.scale.den,
            client_scale_factor: self.client_scale_factor,
            presentation_mode: self.presentation_mode,
        });
        self.refresh_client_display_metrics();
    }

    fn viewport_override_for_surface(&self, size: PhysicalSize<u32>) -> Option<PhysicalSize<u32>> {
        let viewport = viewport_size_for_surface(
            self.presentation_mode,
            (size.width, size.height),
            self.presentation_size_px(),
        );
        (viewport.width != size.width || viewport.height != size.height).then_some(viewport)
    }

    fn viewport_size_for_surface(&self, size: PhysicalSize<u32>) -> PhysicalSize<u32> {
        viewport_size_for_surface(
            self.presentation_mode,
            (size.width, size.height),
            self.presentation_size_px(),
        )
    }

    fn presentation_size_px(&self) -> (u32, u32) {
        let host_display = self.host_display.get();
        logical_actual_size_px(
            host_display.size_px,
            host_display.scale,
            self.client_scale_factor,
        )
    }

    fn refresh_host_display_geometry(&mut self) {
        let geometry = self.host_display.get();
        if geometry == self.last_host_display {
            return;
        }
        let old_presentation_size_px = self.presentation_size_px();
        let pending_window_resize = self.window.as_ref().and_then(|window| {
            let current_size = window.inner_size();
            let monitor_size = window.current_monitor().map(|monitor| {
                let size = monitor.size();
                (size.width, size.height)
            });
            corrected_window_size_for_host_geometry_update(
                current_size,
                old_presentation_size_px,
                logical_actual_size_px(geometry.size_px, geometry.scale, self.client_scale_factor),
                monitor_size,
            )
        });
        self.last_host_display = geometry;
        self.update_presentation_size();
        if let Some(window) = self.window.as_ref() {
            let size = if let Some(size) = pending_window_resize {
                window
                    .request_inner_size(size)
                    .unwrap_or_else(|| window.inner_size())
            } else {
                window.inner_size()
            };
            self.apply_window_resize(size);
        }
    }

    fn update_presentation_size(&mut self) {
        let presentation_size_px = self.presentation_size_px();
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_presentation_size_px(presentation_size_px);
        }
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
        let monitor = event_loop
            .primary_monitor()
            .or_else(|| event_loop.available_monitors().next());
        let monitor_size = monitor.as_ref().map(|monitor| {
            let size = monitor.size();
            (size.width, size.height)
        });
        self.client_scale_factor = monitor
            .as_ref()
            .map_or(1.0, |monitor| sanitize_scale_factor(monitor.scale_factor()));
        let presentation_size_px = self.presentation_size_px();
        let initial_size = initial_window_size_for_monitor(presentation_size_px, monitor_size);
        let attrs = WindowAttributes::default()
            .with_title(&self.title)
            .with_inner_size(PhysicalSize::new(initial_size.0, initial_size.1));
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
            self.presentation_mode,
            presentation_size_px,
            self.cursor_channel.clone(),
        )) {
            Ok(g) => g,
            Err(e) => {
                tracing::error!(error = %e, "failed to initialise render backend");
                event_loop.exit();
                return;
            }
        };
        let size = win.inner_size();
        self.window = Some(win);
        self.gpu = Some(gpu);
        self.apply_window_resize(size);
        self.refresh_client_display_metrics();
        if self.last_display_metrics.is_none() {
            if let Some(monitor) = monitor.as_ref() {
                self.refresh_client_display_metrics_for_monitor(monitor);
            }
        }
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
                if let Some(viewport_size) = self.viewport_override_for_surface(size) {
                    self.apply_window_resize_with_viewport(size, viewport_size);
                    return;
                }
                self.apply_window_resize(size);
            }
            WindowEvent::Moved(_) | WindowEvent::ScaleFactorChanged { .. } => {
                self.refresh_client_display_metrics();
                if let Some(window) = self.window.as_ref() {
                    self.apply_window_resize(window.inner_size());
                }
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
                let video_normalized = cursor_to_video_normalized(
                    position,
                    surface,
                    texture,
                    self.presentation_mode,
                    self.presentation_size_px(),
                );
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
                if b {
                    self.refresh_client_display_metrics();
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
        self.refresh_host_display_geometry();
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

/// Pick the physical-pixel startup window size from the host's advertised
/// capture/display size and the client monitor. The host size is the target for
/// 1:1 presentation; the monitor only caps it when the host would not fit.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn initial_window_size_for_monitor(
    host_px: (u32, u32),
    monitor_px: Option<(u32, u32)>,
) -> (u32, u32) {
    const FALLBACK: (u32, u32) = (1280, 720);
    let (host_w, host_h) = host_px;
    if host_w == 0 || host_h == 0 {
        return FALLBACK;
    }
    let Some((monitor_w, monitor_h)) = monitor_px.filter(|(w, h)| *w > 0 && *h > 0) else {
        return (host_w, host_h);
    };
    if host_w <= monitor_w && host_h <= monitor_h {
        return (host_w, host_h);
    }

    let scale_w = f64::from(monitor_w) / f64::from(host_w);
    let scale_h = f64::from(monitor_h) / f64::from(host_h);
    let scale = scale_w.min(scale_h).min(1.0);
    let width = (f64::from(host_w) * scale).floor() as u32;
    let height = (f64::from(host_h) * scale).floor() as u32;
    (width.max(1), height.max(1))
}

#[must_use]
fn corrected_window_size_for_host_geometry_update(
    current_surface_px: PhysicalSize<u32>,
    old_presentation_px: (u32, u32),
    new_presentation_px: (u32, u32),
    monitor_px: Option<(u32, u32)>,
) -> Option<PhysicalSize<u32>> {
    let old_initial = initial_window_size_for_monitor(old_presentation_px, monitor_px);
    if !approx_size(current_surface_px, old_initial, 1) {
        return None;
    }
    let new_initial = initial_window_size_for_monitor(new_presentation_px, monitor_px);
    if approx_size(current_surface_px, new_initial, 1) {
        return None;
    }
    Some(PhysicalSize::new(new_initial.0, new_initial.1))
}

fn approx_size(size: PhysicalSize<u32>, expected: (u32, u32), tolerance: u32) -> bool {
    size.width.abs_diff(expected.0) <= tolerance && size.height.abs_diff(expected.1) <= tolerance
}

#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn logical_actual_size_px(
    host_physical_px: (u32, u32),
    host_display_scale: DisplayScale,
    client_scale_factor: f64,
) -> (u32, u32) {
    let client_scale_factor = sanitize_scale_factor(client_scale_factor);
    let host_scale = host_display_scale.as_f64();
    if host_physical_px.0 == 0 || host_physical_px.1 == 0 || host_scale <= 0.0 {
        return (1, 1);
    }
    let scale = client_scale_factor / host_scale;
    (
        (f64::from(host_physical_px.0) * scale)
            .round()
            .clamp(1.0, f64::from(u32::MAX)) as u32,
        (f64::from(host_physical_px.1) * scale)
            .round()
            .clamp(1.0, f64::from(u32::MAX)) as u32,
    )
}

#[must_use]
fn viewport_size_for_surface(
    mode: PresentationMode,
    surface: (u32, u32),
    actual_size_px: (u32, u32),
) -> PhysicalSize<u32> {
    let dims = presentation_rect_dims(mode, actual_size_px, surface);
    PhysicalSize::new(dims.0, dims.1)
}

#[must_use]
fn sanitize_scale_factor(scale: f64) -> f64 {
    if !scale.is_finite() || scale <= 0.0 {
        1.0
    } else {
        scale
    }
}

fn client_display_metrics_for_monitor(
    monitor: &winit::monitor::MonitorHandle,
) -> ClientDisplayMetrics {
    let size = monitor.size();
    let refresh_millihz = monitor.refresh_rate_millihertz().unwrap_or(60_000);
    let (scale_num, scale_den) = scale_to_ratio(monitor.scale_factor());
    ClientDisplayMetrics {
        display_id: 0,
        mode: DisplayMode::new(size.width, size.height, refresh_millihz),
        scale_num,
        scale_den,
        safe_area: None,
    }
}

fn client_display_metrics_changed(
    last: Option<&ClientDisplayMetrics>,
    next: &ClientDisplayMetrics,
) -> bool {
    last != Some(next)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn scale_to_ratio(scale: f64) -> (u16, u16) {
    if !scale.is_finite() || scale <= 0.0 {
        return (1, 1);
    }

    const DEN: u32 = 1000;
    let num = (scale * f64::from(DEN))
        .round()
        .clamp(1.0, f64::from(u16::MAX)) as u32;
    let gcd = gcd_u32(num, DEN);
    (
        u16::try_from(num / gcd).unwrap_or(u16::MAX),
        u16::try_from(DEN / gcd).unwrap_or(u16::MAX),
    )
}

const fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    if a == 0 {
        1
    } else {
        a
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
        (*self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()))
        .replace(frame)
    }

    /// Take the currently-held frame, leaving the slot empty.
    pub fn take(&self) -> Option<Frame> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }
}

/// Map a window-pixel cursor position onto the video region in `[0,1]^2`,
/// accounting for the same letterbox / pillarbox transform the GPU shader
/// applies. Returns `None` when the cursor sits in a letterbox bar
/// (outside the video region) or when either size is degenerate.
///
/// Mirrors the active presentation rectangle: `Fit` maps through the
/// fit-or-logical-100% rect, while `ActualSize` maps through a centered logical
/// 100% rect that may extend beyond the surface.
#[allow(clippy::cast_precision_loss)]
fn cursor_to_video_normalized(
    pos: PhysicalPosition<f64>,
    surface: (u32, u32),
    texture: (u32, u32),
    presentation_mode: PresentationMode,
    presentation_size_px: (u32, u32),
) -> Option<(f32, f32)> {
    if surface.0 == 0
        || surface.1 == 0
        || texture.0 == 0
        || texture.1 == 0
        || presentation_size_px.0 == 0
        || presentation_size_px.1 == 0
    {
        return None;
    }
    let (video_w_px, video_h_px) =
        presentation_rect_dims(presentation_mode, presentation_size_px, surface);
    if video_w_px == 0 || video_h_px == 0 {
        return None;
    }
    let sw = f64::from(surface.0);
    let sh = f64::from(surface.1);
    let video_w = f64::from(video_w_px);
    let video_h = f64::from(video_h_px);
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

#[must_use]
pub(crate) fn presentation_rect_dims(
    mode: PresentationMode,
    actual_size_px: (u32, u32),
    surface: (u32, u32),
) -> (u32, u32) {
    if actual_size_px.0 == 0 || actual_size_px.1 == 0 || surface.0 == 0 || surface.1 == 0 {
        return (0, 0);
    }
    match mode {
        PresentationMode::ActualSize => actual_size_px,
        PresentationMode::Fit => {
            if actual_size_px.0 <= surface.0 && actual_size_px.1 <= surface.1 {
                actual_size_px
            } else {
                fit_rect_dims(actual_size_px, surface)
            }
        }
    }
}

#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
pub(crate) fn presentation_scale(
    mode: PresentationMode,
    actual_size_px: (u32, u32),
    surface: (u32, u32),
) -> (f32, f32) {
    if surface.0 == 0 || surface.1 == 0 {
        return (1.0, 1.0);
    }
    let (w, h) = presentation_rect_dims(mode, actual_size_px, surface);
    if w == 0 || h == 0 {
        return (1.0, 1.0);
    }
    (w as f32 / surface.0 as f32, h as f32 / surface.1 as f32)
}

/// Aspect-preserving fit rect in pixels, centered by callers. This may shrink
/// or upscale; [`presentation_rect_dims`] applies the no-upscale policy.
#[must_use]
pub(crate) fn fit_rect_dims(src: (u32, u32), dst: (u32, u32)) -> (u32, u32) {
    if src.0 == 0 || src.1 == 0 || dst.0 == 0 || dst.1 == 0 {
        return (0, 0);
    }
    let src_w = u64::from(src.0);
    let src_h = u64::from(src.1);
    let dst_w = u64::from(dst.0);
    let dst_h = u64::from(dst.1);
    if src_w * dst_h > dst_w * src_h {
        let w = dst.0;
        let h = u32::try_from(((dst_w * src_h) / src_w).clamp(1, dst_h))
            .expect("fit height is clamped to the u32 destination height");
        (w, h)
    } else {
        let h = dst.1;
        let w = u32::try_from(((dst_h * src_w) / src_h).clamp(1, dst_w))
            .expect("fit width is clamped to the u32 destination width");
        (w, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn initial_window_uses_host_size_when_it_fits_client_monitor() {
        assert_eq!(
            initial_window_size_for_monitor((1920, 1080), Some((2560, 1440))),
            (1920, 1080)
        );
    }

    #[test]
    fn initial_window_fits_host_inside_smaller_client_monitor() {
        assert_eq!(
            initial_window_size_for_monitor((3840, 2160), Some((2560, 1440))),
            (2560, 1440)
        );
        assert_eq!(
            initial_window_size_for_monitor((3840, 2160), Some((1920, 1200))),
            (1920, 1080)
        );
    }

    #[test]
    fn initial_window_without_monitor_uses_host_size() {
        assert_eq!(
            initial_window_size_for_monitor((3024, 1952), None),
            (3024, 1952)
        );
    }

    #[test]
    fn initial_window_invalid_host_size_uses_fallback() {
        assert_eq!(
            initial_window_size_for_monitor((0, 1080), Some((2560, 1440))),
            (1280, 720)
        );
        assert_eq!(initial_window_size_for_monitor((0, 0), None), (1280, 720));
    }

    #[test]
    fn host_geometry_update_resizes_window_still_at_bootstrap_size() {
        assert_eq!(
            corrected_window_size_for_host_geometry_update(
                PhysicalSize::new(1920, 1200),
                (1920, 1200),
                (3840, 2400),
                Some((6720, 2836)),
            ),
            Some(PhysicalSize::new(3840, 2400))
        );
    }

    #[test]
    fn host_geometry_update_caps_corrected_window_to_monitor() {
        assert_eq!(
            corrected_window_size_for_host_geometry_update(
                PhysicalSize::new(1920, 1200),
                (1920, 1200),
                (3840, 2400),
                Some((2560, 1440)),
            ),
            Some(PhysicalSize::new(2304, 1440))
        );
    }

    #[test]
    fn host_geometry_update_does_not_resize_after_user_resize() {
        assert_eq!(
            corrected_window_size_for_host_geometry_update(
                PhysicalSize::new(1800, 1000),
                (1920, 1200),
                (3840, 2400),
                Some((6720, 2836)),
            ),
            None
        );
    }

    #[test]
    fn logical_actual_size_scales_between_display_densities() {
        assert_eq!(
            logical_actual_size_px((1920, 1080), DisplayScale::one(), 2.0),
            (3840, 2160)
        );
        assert_eq!(
            logical_actual_size_px((3840, 2160), DisplayScale::new(2, 1).unwrap(), 1.0),
            (1920, 1080)
        );
        assert_eq!(
            logical_actual_size_px((2560, 1440), DisplayScale::new(3, 2).unwrap(), 2.0),
            (3413, 1920)
        );
    }

    #[test]
    fn fit_viewport_caps_to_logical_actual_size() {
        assert_eq!(
            viewport_size_for_surface(PresentationMode::Fit, (2560, 1440), (1920, 1080),),
            PhysicalSize::new(1920, 1080)
        );
    }

    #[test]
    fn fit_viewport_preserves_actual_size_aspect() {
        assert_eq!(
            viewport_size_for_surface(PresentationMode::Fit, (1000, 1000), (1920, 1080)),
            PhysicalSize::new(1000, 562)
        );
        assert_eq!(
            viewport_size_for_surface(PresentationMode::Fit, (1280, 600), (1920, 1080)),
            PhysicalSize::new(1066, 600)
        );
    }

    #[test]
    fn fit_viewport_accepts_already_constrained_size() {
        assert_eq!(
            viewport_size_for_surface(PresentationMode::Fit, (1920, 1080), (1920, 1080),),
            PhysicalSize::new(1920, 1080)
        );
    }

    #[test]
    fn actual_size_viewport_ignores_surface_size() {
        assert_eq!(
            viewport_size_for_surface(PresentationMode::ActualSize, (640, 360), (1920, 1080),),
            PhysicalSize::new(1920, 1080)
        );
    }

    #[test]
    fn viewport_override_reports_density_correct_fit_delta() {
        let app = App {
            title: "test".to_string(),
            host_display: HostDisplayHandle::new(
                HostDisplayGeometry::new((1920, 1080), DisplayScale::one()).unwrap(),
            ),
            last_host_display: HostDisplayGeometry::new((1920, 1080), DisplayScale::one()).unwrap(),
            client_scale_factor: 1.0,
            color_space: tether_protocol::control::VideoColorSpec::sdr_desktop(),
            chroma: tether_protocol::control::ChromaSubsampling::Yuv420,
            bit_depth: 8,
            presentation_mode: PresentationMode::Fit,
            window: None,
            gpu: None,
            frames: LatestFrame::new(),
            latest: None,
            on_event: None,
            present_stats: PresentStats::default(),
            last_recorded_t_cap: None,
            last_display_metrics: None,
            refresh_rate_mhz: present_policy::REFRESH_RATE_FALLBACK_HZ.saturating_mul(1000),
            age_tracker: present_policy::FrameAgeTracker::default(),
            health: RenderHealth::default(),
            cursor_mode: CursorMode::Absolute,
            relative_accum: relative_mouse::SubPixelAccum::default(),
            local_cursor_hidden: false,
            ctrl_held: false,
            alt_held: false,
            cursor_channel: CursorChannel::new(),
        };
        assert_eq!(
            app.viewport_override_for_surface(PhysicalSize::new(2560, 1440)),
            Some(PhysicalSize::new(1920, 1080))
        );
    }

    #[test]
    fn scale_to_ratio_reduces_common_hidpi_values() {
        assert_eq!(scale_to_ratio(1.5), (3, 2));
        assert_eq!(scale_to_ratio(2.0), (2, 1));
        assert_eq!(scale_to_ratio(0.0), (1, 1));
    }

    #[test]
    fn client_display_metrics_changed_dedups_identical_metrics() {
        let base = ClientDisplayMetrics {
            display_id: 0,
            mode: DisplayMode::new(2560, 1440, 60_000),
            scale_num: 2,
            scale_den: 1,
            safe_area: None,
        };
        assert!(client_display_metrics_changed(None, &base));
        assert!(!client_display_metrics_changed(Some(&base), &base));

        let mut scaled = base.clone();
        scaled.scale_num = 3;
        scaled.scale_den = 2;
        assert!(client_display_metrics_changed(Some(&base), &scaled));

        let mut resized = base.clone();
        resized.mode = DisplayMode::new(1920, 1080, 60_000);
        assert!(client_display_metrics_changed(Some(&base), &resized));
    }

    #[test]
    fn fit_presents_at_logical_actual_size_when_it_fits() {
        assert_eq!(
            presentation_rect_dims(PresentationMode::Fit, (1920, 1080), (2560, 1440)),
            (1920, 1080)
        );
        assert_eq!(
            presentation_scale(PresentationMode::Fit, (1920, 1080), (2560, 1440)),
            (0.75, 0.75)
        );
    }

    #[test]
    fn fit_can_scale_up_to_logical_actual_size() {
        assert_eq!(
            presentation_rect_dims(PresentationMode::Fit, (3840, 2160), (2560, 1440)),
            (2560, 1440)
        );
    }

    #[test]
    fn fit_fits_down_when_surface_is_smaller() {
        assert_eq!(
            presentation_rect_dims(PresentationMode::Fit, (3840, 2160), (1280, 1024)),
            (1280, 720)
        );
    }

    #[test]
    fn fit_rect_never_exceeds_destination() {
        let cases = [
            ((1920, 1080), (1000, 1000)),
            ((1920, 1080), (1280, 600)),
            ((3024, 1964), (1512, 982)),
            ((u32::MAX, u32::MAX - 1), (u32::MAX - 3, u32::MAX - 7)),
            ((u32::MAX - 1, u32::MAX), (u32::MAX - 7, u32::MAX - 3)),
        ];
        for (src, dst) in cases {
            let fit = fit_rect_dims(src, dst);
            assert!(fit.0 <= dst.0, "fit width {fit:?} exceeds {dst:?}");
            assert!(fit.1 <= dst.1, "fit height {fit:?} exceeds {dst:?}");
            assert!(
                fit.0 > 0,
                "fit width should stay nonzero for {src:?} in {dst:?}"
            );
            assert!(
                fit.1 > 0,
                "fit height should stay nonzero for {src:?} in {dst:?}"
            );
        }
    }

    #[test]
    fn actual_size_presents_logical_size_even_when_clipped() {
        assert_eq!(
            presentation_rect_dims(PresentationMode::ActualSize, (1920, 1080), (1280, 720)),
            (1920, 1080)
        );
        assert_eq!(
            presentation_scale(PresentationMode::ActualSize, (1920, 1080), (1280, 720)),
            (1.5, 1.5)
        );
    }

    #[test]
    fn cursor_centre_maps_to_centre() {
        let n = cursor_to_video_normalized(
            PhysicalPosition::new(640.0, 360.0),
            (1280, 720),
            (1920, 1080),
            PresentationMode::Fit,
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
            PresentationMode::Fit,
            (1920, 1080),
        )
        .is_none());
        // 1000x1000 window, 1080x1920 source -> left/right pillarbox.
        // x=10 lands in the left bar.
        assert!(cursor_to_video_normalized(
            PhysicalPosition::new(10.0, 500.0),
            (1000, 1000),
            (1080, 1920),
            PresentationMode::Fit,
            (1080, 1920),
        )
        .is_none());
    }

    #[test]
    fn actual_size_cursor_maps_through_clipped_logical_rect() {
        let n = cursor_to_video_normalized(
            PhysicalPosition::new(0.0, 0.0),
            (1280, 720),
            (1920, 1080),
            PresentationMode::ActualSize,
            (1920, 1080),
        )
        .expect("surface centre area is inside the clipped video");
        assert!((n.0 - (320.0 / 1920.0)).abs() < 1e-4);
        assert!((n.1 - (180.0 / 1080.0)).abs() < 1e-4);
    }

    #[test]
    fn cursor_outside_window_returns_none() {
        assert!(cursor_to_video_normalized(
            PhysicalPosition::new(-5.0, 100.0),
            (1280, 720),
            (1280, 720),
            PresentationMode::Fit,
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
            PresentationMode::Fit,
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
    fn latest_frame_recovers_after_poisoned_lock() {
        let frames = LatestFrame::new();
        let poisoned = catch_unwind(AssertUnwindSafe(|| {
            let _guard = frames.0.lock().expect("initial lock succeeds");
            panic!("poison latest-frame slot");
        }));
        assert!(poisoned.is_err());

        let frame = Frame::Cpu(CpuFrame {
            width: 3,
            height: 2,
            y: vec![0; 6],
            uv: vec![128; 4],
            t_capture_client_clock: None,
        });
        assert!(frames.set(frame).is_none());
        let latest = frames.take().expect("slot should recover and hold frame");
        assert_eq!((latest.width(), latest.height()), (3, 2));
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
