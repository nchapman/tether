//! Client-side display: a winit window driving a wgpu render pipeline,
//! fed from a [`LatestFrame`] slot. The slot keeps exactly one frame at
//! a time — drop-oldest semantics — because a remote-desktop viewer
//! always wants the most recent picture, not a queued backlog.

pub mod color;
mod cursor_overlay;
mod gpu;
pub mod present_policy;
pub mod relative_mouse;

#[cfg(test)]
mod dmabuf_test;

#[cfg(test)]
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

use gpu::GpuState;

// Re-exported so tether-input / tether-client can match on render events
// without having to add their own winit dep at a possibly-different
// version. tether-render's version of winit is the workspace version.
pub use winit::event::MouseButton;
pub use winit::keyboard::{KeyCode, ModifiersState};

/// macOS-only — whether the renderer's IOSurface import path accepts
/// the given `(chroma, bit_depth, fourcc)` triple. Exported so
/// cross-crate tests (in `tether-host`) can confirm the renderer's
/// accept set agrees with the encoder's and the VT probe's parallel
/// tables. Drift between any of the three is the family of bug that
/// shipped a broken 10-bit session in commit `621badc` — fast
/// feedback in default CI is cheaper than catching it in a session.
#[cfg(target_os = "macos")]
pub use gpu::accepts_iosurface_fourcc;

pub use gpu::supports_10bit_render;
pub use gpu::supports_d3d11_zero_copy_import;

/// Shared cursor state for the overlay render pass. Construct one,
/// hand a clone to the wire-receive side (call `with(|s| s.set_position(...))`
/// / `with(|s| s.upload_shape(...))`), pass another clone into
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
    /// binary forwards this to the host as `ControlMessage::SetClientViewport`
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
        refresh_rate_mhz: present_policy::REFRESH_RATE_FALLBACK_HZ
            .saturating_mul(1000),
        age_tracker: present_policy::FrameAgeTracker::default(),
        cursor_mode: CursorMode::Absolute,
        relative_accum: relative_mouse::SubPixelAccum::default(),
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
    gpu: Option<GpuState>,
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
    /// Cursor input model. Toggled by Ctrl+Alt+G; on
    /// transition to `Relative` we grab + hide the cursor and
    /// route `DeviceEvent::MouseMotion` through `relative_accum`.
    cursor_mode: CursorMode,
    relative_accum: relative_mouse::SubPixelAccum,
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

impl App {
    fn emit(&self, event: RenderEvent) {
        if let Some(cb) = &self.on_event {
            cb(event);
        }
    }

    /// Try `Locked` first (true pointer-lock, supported on most
    /// platforms in 2026), fall back to `Confined` on X11/Wayland
    /// combos that reject Locked. Hide the cursor either way.
    fn apply_cursor_grab(&self, window: &Window) {
        if window
            .set_cursor_grab(winit::window::CursorGrabMode::Locked)
            .is_err()
        {
            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
        }
        window.set_cursor_visible(false);
    }

    fn toggle_cursor_mode(&mut self) {
        let new_mode = match self.cursor_mode {
            CursorMode::Absolute => CursorMode::Relative,
            CursorMode::Relative => CursorMode::Absolute,
        };
        self.cursor_mode = new_mode;
        // Mirror into the renderer's cursor state so the overlay
        // pass doesn't draw the host pointer while we're rendering
        // our own locked pointer locally.
        self.cursor_channel.with(|state| {
            state.set_relative_mode(matches!(new_mode, CursorMode::Relative));
        });
        // Drop sub-pixel residue so stale fractional motion from
        // the prior mode doesn't leak into the first delta.
        self.relative_accum.reset();
        if let Some(window) = &self.window {
            match new_mode {
                CursorMode::Relative => self.apply_cursor_grab(window),
                CursorMode::Absolute => {
                    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                    window.set_cursor_visible(true);
                }
            }
        }
        self.emit(RenderEvent::CursorModeChanged(new_mode));
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
        let gpu = match pollster::block_on(GpuState::new(
            win.clone(),
            self.color_space,
            self.chroma,
            self.bit_depth,
            self.cursor_channel.clone(),
        )) {
            Ok(g) => g,
            Err(e) => {
                tracing::error!(error = %e, "failed to initialise wgpu");
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
        } else if let Some(mhz) = win.current_monitor().and_then(|m| m.refresh_rate_millihertz()) {
            self.refresh_rate_mhz = mhz;
        }
        self.window = Some(win);
        self.gpu = Some(gpu);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                gpu.resize(size.width, size.height);
                self.emit(RenderEvent::Resized {
                    width: size.width,
                    height: size.height,
                });
            }
            WindowEvent::RedrawRequested => {
                // Apply a new frame at most once per redraw. GpuState
                // holds the imported/uploaded textures across redraws,
                // so OS-initiated redraws (expose, focus) re-render the
                // most recently applied frame without re-uploading or
                // re-importing.
                //
                // Before applying, run the frame-age policy: a
                // sufficiently-stale frame (>~1.5× refresh period,
                // streak gated) is dropped on the floor so the
                // refresh slot can carry whatever lands next instead
                // of stale content the user has already adjusted to.
                // Frames without a timestamp (test_pattern example)
                // always render — no policy input to evaluate.
                let mut skipped = false;
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
                        if let Err(e) = gpu.apply_frame(frame) {
                            warn!(error = ?e, "applying frame failed");
                        }
                    }
                    if !skipped && t_capture.is_some() {
                        // Sample present latency only for frames we
                        // actually applied. Dedup against
                        // `last_recorded_t_cap` so OS-initiated
                        // redraws don't double-count.
                        if self.last_recorded_t_cap != t_capture {
                            let t_cap = t_capture.expect("checked Some above");
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
                } else {
                    tracing::trace!(
                        late_streak = self.age_tracker.late_streak,
                        drops = self.age_tracker.drops_in_window,
                        "skipped stale frame"
                    );
                }
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
                let (texture, surface) = gpu.dimensions();
                self.emit(RenderEvent::Cursor {
                    video_normalized: cursor_to_video_normalized(position, surface, texture),
                });
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
                        window.set_cursor_visible(true);
                    } else if matches!(self.cursor_mode, CursorMode::Relative) {
                        // Re-acquire on focus regain.
                        self.apply_cursor_grab(window);
                    }
                }
                self.emit(RenderEvent::Focused(b));
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _id: DeviceId,
        event: DeviceEvent,
    ) {
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
        if let Some(frame) = self.frames.take() {
            self.latest = Some(frame);
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
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
        std::mem::replace(&mut *self.0.lock().expect("LatestFrame mutex poisoned"), Some(frame))
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
    let (sx, sy) = letterbox_scale_for_cursor(texture, surface);
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

/// Local copy of `gpu::letterbox_scale` — the GPU module's version is
/// private and we don't want to widen its visibility just for the cursor
/// math. If these ever drift, the cursor will land in the wrong place.
#[allow(clippy::cast_precision_loss)]
fn letterbox_scale_for_cursor(src: (u32, u32), dst: (u32, u32)) -> (f32, f32) {
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
            width: 1, height: 1,
            y: vec![0xa], uv: vec![0, 0],
            t_capture_client_clock: None,
        });
        let frame_b = Frame::Cpu(CpuFrame {
            width: 2, height: 2,
            y: vec![0xb], uv: vec![0, 0],
            t_capture_client_clock: None,
        });
        // First set: empty slot, no displacement.
        assert!(frames.set(frame_a).is_none(), "empty slot should return None");
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
}
