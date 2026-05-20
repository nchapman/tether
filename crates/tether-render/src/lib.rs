//! Client-side display: a winit window driving a wgpu render pipeline,
//! fed from a crossbeam channel of [`Frame`]s.

mod gpu;

#[cfg(test)]
mod dmabuf_test;

use std::sync::Arc;

use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use tether_codec::{GpuFrameGuard, GpuFrameSource};
use tether_protocol::MonoNanos;
use tracing::warn;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowAttributes, WindowId};

use gpu::GpuState;

// Re-exported so tether-input / tether-client can match on render events
// without having to add their own winit dep at a possibly-different
// version. tether-render's version of winit is the workspace version.
pub use winit::event::MouseButton;
pub use winit::keyboard::{KeyCode, ModifiersState};

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
/// `on_event` is optional: pass `None` to skip input-event emission.
/// Using a callback rather than a channel lets the caller bridge into
/// whatever sync/async plumbing they already have.
pub fn run(
    title: &str,
    initial_size: (u32, u32),
    frames: Receiver<Frame>,
    on_event: Option<EventSink>,
) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = App {
        title: title.to_string(),
        initial_size,
        window: None,
        gpu: None,
        frames,
        latest: None,
        on_event,
        present_stats: PresentStats::default(),
        last_recorded_t_cap: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}


struct App {
    title: String,
    initial_size: (u32, u32),
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    frames: Receiver<Frame>,
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
        let gpu = match pollster::block_on(GpuState::new(win.clone())) {
            Ok(g) => g,
            Err(e) => {
                tracing::error!(error = %e, "failed to initialise wgpu");
                event_loop.exit();
                return;
            }
        };
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
            WindowEvent::Resized(size) => gpu.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                // Apply a new frame at most once per redraw. GpuState
                // holds the imported/uploaded textures across redraws,
                // so OS-initiated redraws (expose, focus) re-render the
                // most recently applied frame without re-uploading or
                // re-importing.
                let applied_t_capture = if let Some(frame) = self.latest.take() {
                    let t_capture = frame.t_capture_client_clock();
                    if let Err(e) = gpu.apply_frame(frame) {
                        warn!(error = ?e, "applying frame failed");
                    }
                    t_capture
                } else {
                    None
                };
                if let Err(e) = gpu.render() {
                    warn!(error = ?e, "render frame failed");
                }
                // Sample t_present after the present() call inside
                // gpu.render() returns. This isn't the true on-screen
                // time (that's compositor + display latency further
                // down) but it bounds it from below, which is enough
                // to decompose recv-to-present vs network+encode.
                //
                // Only record on a frame we just applied; OS-initiated
                // redraws without a new frame produce no sample.
                if let Some(t_cap) = applied_t_capture {
                    if self.last_recorded_t_cap != Some(t_cap) {
                        let latency = MonoNanos::now().saturating_sub(t_cap);
                        self.present_stats.record_and_maybe_log(latency);
                        self.last_recorded_t_cap = Some(t_cap);
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                // PhysicalKey + the resolved text travel together: HID
                // for shortcuts, text for layout-aware typing. The
                // translator decides which path each event takes.
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.emit(RenderEvent::Key {
                        code,
                        pressed: event.state == ElementState::Pressed,
                        repeat: event.repeat,
                        text: event.text.as_deref().map(str::to_owned),
                    });
                }
            }
            WindowEvent::ModifiersChanged(m) => {
                self.emit(RenderEvent::Modifiers(m.state()));
            }
            WindowEvent::CursorMoved { position, .. } => {
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
                self.emit(RenderEvent::Focused(b));
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let mut received = false;
        loop {
            match self.frames.try_recv() {
                Ok(frame) => {
                    self.latest = Some(frame);
                    received = true;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    // Producer is gone — keep showing the last frame
                    // until the user closes the window.
                    break;
                }
            }
        }
        if received {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
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
}
