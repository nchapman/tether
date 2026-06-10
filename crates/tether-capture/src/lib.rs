//! Screen capture for Tether.
//!
//! Each backend is a free function returning a
//! [`crossbeam_channel::Receiver`] of [`CapturedFrame`]s. Dropping the
//! receiver shuts the producer down. A `Capturer` trait will land once
//! we have multiple real backends that need runtime selection.
//!
//! Backends:
//! - [`linux`] — PipeWire + xdg-desktop-portal; advertises DMA-BUF and
//!   SHM alternatives and produces zero-copy DMA-BUF frames when the
//!   compositor agrees.
//! - [`macos`] — ScreenCaptureKit producing zero-copy IOSurface frames.
//! - [`test_pattern`] is always available and produces synthetic frames
//!   at a fixed cadence so the walking skeleton and headless tests can
//!   exercise the pipeline without a display server.

pub mod cursor;
pub mod damage;
pub mod test_pattern;

#[cfg(feature = "test-support")]
pub mod test_support;

pub use cursor::{
    rescale_shape_to_frame, CursorEvent, CursorPosition, CursorShapeEvent, CursorSource,
    PlaceholderCursorSource,
};
pub use damage::{DamageHint, DamageSignal, HashDamage, NativeDamage};

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use crossbeam_channel::{Receiver, RecvTimeoutError};
use tether_protocol::control::{DisplayDescriptor, DisplayId, DisplayMode};

#[cfg(target_os = "linux")]
use smithay_client_toolkit::{
    delegate_output, delegate_registry,
    output::{OutputHandler, OutputInfo, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
};
#[cfg(target_os = "linux")]
use wayland_client::{globals::registry_queue_init, protocol::wl_output, Connection, QueueHandle};

/// Capture-source handle returned by every backend's `start()`. Bundles
/// the [`CapturedFrame`] receiver with a runtime-mutable target-FPS
/// atomic so the ABR / quality-tier controller can throttle capture
/// without rebuilding the backend.
///
/// **Per-backend honouring is uneven.** [`test_pattern::start`] reads
/// the atomic on every produced frame and adjusts its sleep period
/// immediately. The Linux PipeWire and macOS ScreenCaptureKit backends
/// currently *accept* the atomic so the seam is end-to-end live, but
/// renegotiating the stream's actual frame interval requires backend-
/// specific work (PipeWire format renegotiation; SCK
/// `SCStreamConfiguration.minimumFrameInterval` update) that lands
/// per backend. Until then, `set_target_fps` updates the atomic
/// silently — readers see the new value but the produced cadence
/// does not change. The honouring matrix lands per-backend; check
/// the backend's module docs before assuming a write took effect.
pub struct CaptureHandle {
    /// Frame receiver. Take ownership via [`Self::into_rx`] or borrow
    /// via [`Self::rx`].
    rx: Receiver<CapturedFrame>,
    /// Target FPS the backend should aim for. `Arc` so the ABR
    /// controller can hold a clone and update it from a different
    /// thread.
    target_fps: Arc<AtomicU32>,
    /// Optional per-backend cursor source. Wayland/PipeWire fills
    /// this with a `SPA_META_Cursor` parser; macOS will fill it with
    /// an `NSCursor` poller; the test pattern leaves it `None` and
    /// the host falls back to [`PlaceholderCursorSource`].
    cursor_source: Option<Box<dyn CursorSource>>,
    /// Optional display-coordinate metadata for the capture source.
    /// Linux portals can report monitor stream bounds in compositor
    /// coordinates before PipeWire reports the actual frame pixel grid.
    display_hints: CaptureDisplayHints,
    /// Consumer-liveness token. Moves into the [`FrameReceiver`] on
    /// [`Self::into_rx`] so its strong count tracks whether the consumer
    /// still holds the receiver. A backend producer that keeps its own
    /// receiver clone (Windows drop-oldest eviction masks channel
    /// disconnection) watches [`Self::liveness`] to know when to stop;
    /// backends that rely on channel `Disconnected` ignore it.
    alive: Arc<()>,
}

impl CaptureHandle {
    /// Build a handle owning the receiver and seeded at `initial_fps`.
    /// Backends construct the atomic up front (so they can also read
    /// it from their producer thread) and pass it through.
    #[must_use]
    pub fn from_parts(rx: Receiver<CapturedFrame>, target_fps: Arc<AtomicU32>) -> Self {
        Self {
            rx,
            target_fps,
            cursor_source: None,
            display_hints: CaptureDisplayHints::default(),
            alive: Arc::new(()),
        }
    }

    /// A weak handle to the consumer-liveness token. A backend producer
    /// holds this to detect when the consumer has dropped its
    /// [`FrameReceiver`] — necessary when the producer keeps its own
    /// receiver clone (which would otherwise mask channel disconnection).
    /// `strong_count() == 0` means the consumer is gone.
    #[must_use]
    pub fn liveness(&self) -> Weak<()> {
        Arc::downgrade(&self.alive)
    }

    /// Attach a cursor source to this handle. Called by the backend
    /// after `from_parts` once the producer thread is spawned.
    #[must_use]
    pub fn with_cursor_source(mut self, src: Box<dyn CursorSource>) -> Self {
        self.cursor_source = Some(src);
        self
    }

    /// Attach display-coordinate hints observed by the capture backend.
    #[must_use]
    pub fn with_display_hints(mut self, hints: CaptureDisplayHints) -> Self {
        self.display_hints = hints;
        self
    }

    /// Display-coordinate hints observed by the capture backend.
    #[must_use]
    pub fn display_hints(&self) -> CaptureDisplayHints {
        self.display_hints
    }

    /// Take the cursor source out, leaving `None`. The host pump
    /// consumes this once at session start.
    #[must_use]
    pub fn take_cursor_source(&mut self) -> Option<Box<dyn CursorSource>> {
        self.cursor_source.take()
    }

    /// Borrow the frame receiver — for callers that need to interleave
    /// receiving with shutdown checks.
    #[must_use]
    pub fn rx(&self) -> &Receiver<CapturedFrame> {
        &self.rx
    }

    /// Consume the handle, returning the [`FrameReceiver`]. The FPS atomic
    /// is dropped from the handle's side; clone via [`Self::fps_handle`]
    /// before calling this if the ABR controller needs to keep writing.
    /// The returned receiver carries the consumer-liveness token (see
    /// [`Self::liveness`]); hold it for as long as frames are wanted.
    #[must_use]
    pub fn into_rx(self) -> FrameReceiver {
        FrameReceiver {
            rx: self.rx,
            _alive: self.alive,
        }
    }

    /// Current target FPS as observed by the backend's producer
    /// thread. Reads are `Relaxed` — the value is advisory, not a
    /// synchronization primitive.
    #[must_use]
    pub fn target_fps(&self) -> u32 {
        self.target_fps.load(Ordering::Relaxed)
    }

    /// Update the target FPS. Backends pick up the new value on their
    /// next loop iteration (test_pattern) or via renegotiation
    /// (real backends — currently unimplemented; see struct docs).
    /// Values < 1 are clamped to 1.
    pub fn set_target_fps(&self, fps: u32) {
        self.target_fps.store(fps.max(1), Ordering::Relaxed);
    }

    /// Cheap clone of the atomic for the ABR controller to retain
    /// across the `into_rx` boundary.
    #[must_use]
    pub fn fps_handle(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.target_fps)
    }
}

/// The consuming end of a capture stream, returned by
/// [`CaptureHandle::into_rx`]. Wraps the [`CapturedFrame`] receiver and
/// carries the consumer-liveness token: while this is held, the backend
/// producer's [`CaptureHandle::liveness`] weak handle reports a non-zero
/// strong count. Dropping it both disconnects the channel and drops the
/// token, so producers can detect consumer shutdown either way.
pub struct FrameReceiver {
    rx: Receiver<CapturedFrame>,
    _alive: Arc<()>,
}

/// Optional source geometry reported by a capture backend before or alongside
/// frames. The size/position are in host logical or compositor coordinates, not
/// in captured frame pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CaptureDisplayHints {
    pub host_logical_size: Option<(u32, u32)>,
    pub host_logical_position: Option<(i32, i32)>,
}

/// Live capture geometry used to refresh `DisplayList` after the capture
/// backend reveals the actual frame pixel grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureDisplayGeometry {
    pub capture_width: u32,
    pub capture_height: u32,
    pub refresh_millihz: u32,
    pub host_logical_size: Option<(u32, u32)>,
    pub host_logical_position: Option<(i32, i32)>,
}

impl CaptureDisplayGeometry {
    #[must_use]
    pub fn new(capture_width: u32, capture_height: u32, refresh_millihz: u32) -> Self {
        Self {
            capture_width,
            capture_height,
            refresh_millihz,
            host_logical_size: None,
            host_logical_position: None,
        }
    }

    #[must_use]
    pub fn with_hints(mut self, hints: CaptureDisplayHints) -> Self {
        self.host_logical_size = hints.host_logical_size;
        self.host_logical_position = hints.host_logical_position;
        self
    }
}

impl FrameReceiver {
    /// Wait up to `timeout` for the next frame. Mirrors
    /// [`crossbeam_channel::Receiver::recv_timeout`] exactly.
    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> std::result::Result<CapturedFrame, RecvTimeoutError> {
        self.rx.recv_timeout(timeout)
    }
}

#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "macos")]
pub mod cursor_macos;

#[cfg(target_os = "windows")]
pub mod cursor_windows;

#[cfg(target_os = "windows")]
pub mod windows;

use tether_protocol::MonoNanos;

/// Re-export of [`tether_protocol::GpuResourceGuard`]. Producers (the
/// capture backend) stash whatever they need to keep alive while the
/// consumer reads the buffer; consumers can't downcast or inspect.
pub use tether_protocol::GpuResourceGuard as GpuCapturedGuard;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// 8-bit-per-channel BGRA, little-endian per pixel (B, G, R, A).
    /// What ScreenCaptureKit and DXGI emit by default.
    Bgra8,
    /// 8-bit-per-channel RGBA. Used by `tether-render`'s passthrough.
    Rgba8,
    /// 8-bit Y plane followed by interleaved 8-bit Cb/Cr at half
    /// resolution in each axis. What the HEVC/H.264 hardware encoders
    /// want as input.
    Nv12,
}

/// Best-effort host display topology for the protocol handshake.
///
/// Mode mutation is deliberately not advertised here. A backend should flip
/// `can_set_mode` and expand `available_modes` only once it can apply and
/// restore display modes end to end.
pub fn display_list() -> Result<Vec<DisplayDescriptor>> {
    platform_display_list()
}

/// Synthetic display descriptor for the test-pattern capture source.
#[must_use]
pub fn test_pattern_display(width: u32, height: u32, refresh_millihz: u32) -> DisplayDescriptor {
    let mode = DisplayMode::new(width, height, refresh_millihz);
    DisplayDescriptor {
        id: DisplayId(0),
        name: "test-pattern".to_string(),
        scale_num: 1,
        scale_den: 1,
        primary: true,
        position: (0, 0),
        current_mode: mode,
        available_modes: vec![mode],
        can_set_mode: false,
    }
}

/// Update the display that matches the capture stream's actual frame
/// dimensions, falling back to the primary display when no descriptor matches.
/// This corrects portal/ScreenCaptureKit negotiation details that are only
/// known after the capture stream starts, and lets a user-selected non-primary
/// monitor carry its own density metadata into the live display list.
#[must_use]
pub fn display_list_with_primary_mode(
    displays: Vec<DisplayDescriptor>,
    width: u32,
    height: u32,
    refresh_millihz: u32,
) -> Vec<DisplayDescriptor> {
    display_list_with_capture_geometry(
        displays,
        CaptureDisplayGeometry::new(width, height, refresh_millihz),
    )
}

/// Update the display descriptor that corresponds to the live capture source.
/// If the backend reports host logical/compositor geometry for that source, use
/// it to derive the display scale that belongs to the captured frame pixels.
#[must_use]
pub fn display_list_with_capture_geometry(
    mut displays: Vec<DisplayDescriptor>,
    geometry: CaptureDisplayGeometry,
) -> Vec<DisplayDescriptor> {
    let idx = displays
        .iter()
        .position(|display| {
            display.current_mode.width == geometry.capture_width
                && display.current_mode.height == geometry.capture_height
        })
        .or_else(|| {
            let (pos, size) = (geometry.host_logical_position?, geometry.host_logical_size?);
            displays
                .iter()
                .position(|display| display_matches_host_logical_geometry(display, pos, size))
        })
        .or_else(|| displays.iter().position(|display| display.primary))
        .unwrap_or(0);
    for (display_idx, display) in displays.iter_mut().enumerate() {
        display.primary = display_idx == idx;
    }

    let Some(display) = displays.get_mut(idx) else {
        return vec![test_pattern_display(
            geometry.capture_width,
            geometry.capture_height,
            geometry.refresh_millihz,
        )];
    };

    let mode = DisplayMode::new(
        geometry.capture_width,
        geometry.capture_height,
        geometry.refresh_millihz,
    );
    display.current_mode = mode;
    if let Some((scale_num, scale_den)) = scale_from_capture_and_host_logical(
        geometry.capture_width,
        geometry.capture_height,
        geometry.host_logical_size,
    ) {
        display.scale_num = scale_num;
        display.scale_den = scale_den;
    }
    if display.available_modes.is_empty() {
        display.available_modes.push(mode);
    } else {
        display.available_modes[0] = mode;
    }
    displays
}

fn scale_from_capture_and_host_logical(
    capture_width: u32,
    capture_height: u32,
    host_logical_size: Option<(u32, u32)>,
) -> Option<(u16, u16)> {
    let (logical_width, logical_height) = host_logical_size?;
    if capture_width == 0 || capture_height == 0 || logical_width == 0 || logical_height == 0 {
        return None;
    }
    let scale_x = f64::from(capture_width) / f64::from(logical_width);
    let scale_y = f64::from(capture_height) / f64::from(logical_height);
    let rel_delta = (scale_x - scale_y).abs() / scale_x.max(scale_y);
    (rel_delta <= 0.01).then(|| scale_to_ratio((scale_x + scale_y) * 0.5))
}

#[cfg(any(target_os = "linux", test))]
fn scale_from_mode_and_logical_size(
    mode_size: (u32, u32),
    logical_size: Option<(u32, u32)>,
    fallback_scale: f64,
) -> (u16, u16) {
    scale_from_capture_and_host_logical(mode_size.0, mode_size.1, logical_size)
        .unwrap_or_else(|| scale_to_ratio(fallback_scale))
}

fn display_matches_host_logical_geometry(
    display: &DisplayDescriptor,
    host_logical_position: (i32, i32),
    host_logical_size: (u32, u32),
) -> bool {
    let Some((x, y)) =
        scaled_i32_pair_to_logical(display.position, display.scale_num, display.scale_den)
    else {
        return false;
    };
    let Some((width, height)) = scaled_u32_pair_to_logical(
        (display.current_mode.width, display.current_mode.height),
        display.scale_num,
        display.scale_den,
    ) else {
        return false;
    };

    approx_i32(x, host_logical_position.0, 1)
        && approx_i32(y, host_logical_position.1, 1)
        && approx_u32(width, host_logical_size.0, 1)
        && approx_u32(height, host_logical_size.1, 1)
}

fn scaled_u32_pair_to_logical(
    value: (u32, u32),
    scale_num: u16,
    scale_den: u16,
) -> Option<(u32, u32)> {
    if scale_num == 0 || scale_den == 0 {
        return None;
    }
    Some((
        u32::try_from(
            div_round_u64(
                u64::from(value.0) * u64::from(scale_den),
                u64::from(scale_num),
            )
            .min(u64::from(u32::MAX)),
        )
        .unwrap_or(u32::MAX),
        u32::try_from(
            div_round_u64(
                u64::from(value.1) * u64::from(scale_den),
                u64::from(scale_num),
            )
            .min(u64::from(u32::MAX)),
        )
        .unwrap_or(u32::MAX),
    ))
}

fn scaled_i32_pair_to_logical(
    value: (i32, i32),
    scale_num: u16,
    scale_den: u16,
) -> Option<(i32, i32)> {
    if scale_num == 0 || scale_den == 0 {
        return None;
    }
    Some((
        i32::try_from(
            div_round_i64(
                i64::from(value.0) * i64::from(scale_den),
                i64::from(scale_num),
            )
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)),
        )
        .unwrap_or(if value.0.is_negative() {
            i32::MIN
        } else {
            i32::MAX
        }),
        i32::try_from(
            div_round_i64(
                i64::from(value.1) * i64::from(scale_den),
                i64::from(scale_num),
            )
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)),
        )
        .unwrap_or(if value.1.is_negative() {
            i32::MIN
        } else {
            i32::MAX
        }),
    ))
}

#[cfg(any(target_os = "linux", test))]
fn scaled_i32_pair_from_logical(
    value: (i32, i32),
    scale_num: u16,
    scale_den: u16,
) -> Option<(i32, i32)> {
    if scale_num == 0 || scale_den == 0 {
        return None;
    }
    Some((
        i32::try_from(
            div_round_i64(
                i64::from(value.0) * i64::from(scale_num),
                i64::from(scale_den),
            )
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)),
        )
        .unwrap_or(if value.0.is_negative() {
            i32::MIN
        } else {
            i32::MAX
        }),
        i32::try_from(
            div_round_i64(
                i64::from(value.1) * i64::from(scale_num),
                i64::from(scale_den),
            )
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)),
        )
        .unwrap_or(if value.1.is_negative() {
            i32::MIN
        } else {
            i32::MAX
        }),
    ))
}

const fn div_round_u64(value: u64, divisor: u64) -> u64 {
    (value + (divisor / 2)) / divisor
}

const fn div_round_i64(value: i64, divisor: i64) -> i64 {
    if value >= 0 {
        (value + (divisor / 2)) / divisor
    } else {
        (value - (divisor / 2)) / divisor
    }
}

fn approx_u32(a: u32, b: u32, tolerance: u32) -> bool {
    a.abs_diff(b) <= tolerance
}

fn approx_i32(a: i32, b: i32, tolerance: i32) -> bool {
    a.abs_diff(b) <= u32::try_from(tolerance).unwrap_or(0)
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows", test))]
fn scale_to_ratio(scale: f64) -> (u16, u16) {
    if !scale.is_finite() || scale <= 0.0 {
        return (1, 1);
    }

    const DEN: u32 = 1000;
    let num = f64_to_u32_clamped(scale * f64::from(DEN), 1, u32::from(u16::MAX));
    let gcd = gcd_u32(num, DEN);
    (
        u16::try_from(num / gcd).unwrap_or(u16::MAX),
        u16::try_from(DEN / gcd).unwrap_or(u16::MAX),
    )
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows", test))]
fn f64_to_u32_clamped(value: f64, min: u32, max: u32) -> u32 {
    if !value.is_finite() {
        return min;
    }
    let clamped = value.round().clamp(f64::from(min), f64::from(max));
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        clamped as u32
    }
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows", test))]
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

#[cfg(target_os = "linux")]
fn platform_display_list() -> Result<Vec<DisplayDescriptor>> {
    use std::sync::OnceLock;

    static DISPLAY_LIST: OnceLock<Vec<DisplayDescriptor>> = OnceLock::new();
    if let Some(displays) = DISPLAY_LIST.get() {
        return Ok(displays.clone());
    }

    let displays = wayland_display_list()
        .inspect_err(|e| {
            tracing::debug!(
                error = %e,
                "wayland output topology unavailable; falling back to winit monitor topology"
            );
        })
        .or_else(|_| winit_display_list())?;
    Ok(DISPLAY_LIST.get_or_init(|| displays).clone())
}

#[cfg(target_os = "linux")]
struct WaylandOutputCollector {
    registry_state: RegistryState,
    output_state: OutputState,
}

#[cfg(target_os = "linux")]
impl OutputHandler for WaylandOutputCollector {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }
}

#[cfg(target_os = "linux")]
delegate_output!(WaylandOutputCollector);

#[cfg(target_os = "linux")]
delegate_registry!(WaylandOutputCollector);

#[cfg(target_os = "linux")]
impl ProvidesRegistryState for WaylandOutputCollector {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }

    registry_handlers! {
        OutputState,
    }
}

#[cfg(target_os = "linux")]
fn wayland_display_list() -> Result<Vec<DisplayDescriptor>> {
    let conn = Connection::connect_to_env()
        .map_err(|e| CaptureError::Display(format!("wayland connect: {e}")))?;
    let (globals, mut event_queue) = registry_queue_init(&conn)
        .map_err(|e| CaptureError::Display(format!("wayland registry: {e}")))?;
    let qh = event_queue.handle();
    let registry_state = RegistryState::new(&globals);
    let output_state = OutputState::new(&globals, &qh);
    let mut collector = WaylandOutputCollector {
        registry_state,
        output_state,
    };
    event_queue
        .roundtrip(&mut collector)
        .map_err(|e| CaptureError::Display(format!("wayland output roundtrip: {e}")))?;

    let mut displays = Vec::new();
    for (idx, output) in collector.output_state.outputs().enumerate() {
        let info = collector
            .output_state
            .info(&output)
            .ok_or_else(|| CaptureError::Display("wayland output has no info".into()))?;
        displays.push(wayland_output_to_descriptor(idx, &info)?);
    }

    if displays.is_empty() {
        return Err(CaptureError::Display(
            "no monitors reported by wayland".into(),
        ));
    }
    Ok(displays)
}

#[cfg(target_os = "linux")]
fn wayland_output_to_descriptor(idx: usize, info: &OutputInfo) -> Result<DisplayDescriptor> {
    let mode = info
        .modes
        .iter()
        .find(|mode| mode.current)
        .or_else(|| info.modes.iter().find(|mode| mode.preferred))
        .or_else(|| info.modes.first())
        .ok_or_else(|| CaptureError::Display("wayland output has no modes".into()))?;
    let mode_size = positive_u32_pair(mode.dimensions)
        .ok_or_else(|| CaptureError::Display("wayland output mode has invalid size".into()))?;
    let logical_size = positive_u32_pair(
        info.logical_size
            .ok_or_else(|| CaptureError::Display("wayland output has no logical size".into()))?,
    )
    .ok_or_else(|| CaptureError::Display("wayland output has invalid logical size".into()))?;
    let logical_position = info
        .logical_position
        .ok_or_else(|| CaptureError::Display("wayland output has no logical position".into()))?;
    let refresh_millihz = u32::try_from(mode.refresh_rate)
        .ok()
        .filter(|refresh| *refresh > 0)
        .unwrap_or(60_000);
    let (scale_num, scale_den) = scale_from_mode_and_logical_size(
        mode_size,
        Some(logical_size),
        f64::from(info.scale_factor.max(1)),
    );
    let position =
        scaled_i32_pair_from_logical(logical_position, scale_num, scale_den).unwrap_or((0, 0));
    let current_mode = DisplayMode::new(mode_size.0, mode_size.1, refresh_millihz);
    let name = info
        .name
        .clone()
        .or_else(|| info.description.clone())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| {
            let label = [info.make.as_str(), info.model.as_str()]
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            if label.is_empty() {
                format!("display-{idx}")
            } else {
                label
            }
        });

    tracing::debug!(
        id = info.id,
        name = %name,
        mode_width_px = mode_size.0,
        mode_height_px = mode_size.1,
        logical_width = logical_size.0,
        logical_height = logical_size.1,
        logical_x = logical_position.0,
        logical_y = logical_position.1,
        scale_num,
        scale_den,
        "wayland output topology"
    );

    Ok(DisplayDescriptor {
        id: DisplayId(u32::try_from(idx).unwrap_or(u32::MAX)),
        primary: idx == 0,
        name,
        scale_num,
        scale_den,
        position,
        current_mode,
        available_modes: vec![current_mode],
        can_set_mode: false,
    })
}

#[cfg(target_os = "linux")]
fn positive_u32_pair(value: (i32, i32)) -> Option<(u32, u32)> {
    Some((u32::try_from(value.0).ok()?, u32::try_from(value.1).ok()?))
        .filter(|(width, height)| *width > 0 && *height > 0)
}

#[cfg(target_os = "linux")]
fn winit_display_list() -> Result<Vec<DisplayDescriptor>> {
    use std::sync::OnceLock;
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::window::WindowId;

    static WINIT_DISPLAY_LIST: OnceLock<Vec<DisplayDescriptor>> = OnceLock::new();
    if let Some(displays) = WINIT_DISPLAY_LIST.get() {
        return Ok(displays.clone());
    }

    let mut builder = EventLoop::builder();
    winit::platform::wayland::EventLoopBuilderExtWayland::with_any_thread(&mut builder, true);
    winit::platform::x11::EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
    let event_loop = builder
        .build()
        .map_err(|e| CaptureError::Display(format!("winit event loop: {e}")))?;

    // winit 0.30 exposes monitor enumeration only through `ActiveEventLoop`, which
    // is reachable from the `resumed` callback. Desktop platforms emit `resumed`
    // once at startup, so we collect the monitor list there and exit immediately.
    #[derive(Default)]
    struct DisplayCollector {
        displays: Vec<DisplayDescriptor>,
    }

    impl ApplicationHandler for DisplayCollector {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            // Wayland has no primary-monitor concept, so `primary_monitor()`
            // always returns `None` there; `monitor_to_descriptor` then falls
            // back to flagging index 0. Only X11 (RandR) reports a real primary.
            let primary_name = event_loop
                .primary_monitor()
                .and_then(|monitor| monitor.name());
            self.displays = event_loop
                .available_monitors()
                .enumerate()
                .map(|(idx, monitor)| monitor_to_descriptor(idx, &monitor, primary_name.as_deref()))
                .collect();
            event_loop.exit();
        }

        fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
    }

    let mut collector = DisplayCollector::default();
    event_loop
        .run_app(&mut collector)
        .map_err(|e| CaptureError::Display(format!("winit run_app: {e}")))?;

    if collector.displays.is_empty() {
        return Err(CaptureError::Display(
            "no monitors reported by winit".into(),
        ));
    }
    Ok(WINIT_DISPLAY_LIST
        .get_or_init(|| collector.displays)
        .clone())
}

#[cfg(target_os = "linux")]
fn monitor_to_descriptor(
    idx: usize,
    monitor: &winit::monitor::MonitorHandle,
    primary_name: Option<&str>,
) -> DisplayDescriptor {
    let size = monitor.size();
    let position = monitor.position();
    let refresh_millihz = monitor.refresh_rate_millihertz().unwrap_or(60_000);
    let mode = DisplayMode::new(size.width, size.height, refresh_millihz);
    let name = monitor.name().unwrap_or_else(|| format!("display-{idx}"));
    let (scale_num, scale_den) = scale_to_ratio(monitor.scale_factor());
    DisplayDescriptor {
        id: DisplayId(u32::try_from(idx).unwrap_or(u32::MAX)),
        primary: primary_name.is_some_and(|primary| primary == name)
            || (primary_name.is_none() && idx == 0),
        name,
        scale_num,
        scale_den,
        position: (position.x, position.y),
        current_mode: mode,
        available_modes: vec![mode],
        can_set_mode: false,
    }
}

#[cfg(target_os = "macos")]
fn platform_display_list() -> Result<Vec<DisplayDescriptor>> {
    macos::display_list()
}

#[cfg(target_os = "windows")]
fn platform_display_list() -> Result<Vec<DisplayDescriptor>> {
    windows::display_list()
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn platform_display_list() -> Result<Vec<DisplayDescriptor>> {
    Err(CaptureError::Unsupported)
}

/// A single captured frame from the host's display.
///
/// Two shapes: CPU-side owned bytes (the SHM fallback path), and a
/// platform-specific GPU handle (DMA-BUF on Linux, IOSurface on macOS, D3D11
/// texture on Windows). The host's encode path pattern-matches: GPU frames go
/// through the platform bridge/encoder path; CPU frames fall through to
/// `encode_bgra`.
///
/// Shape mirrors [`tether_codec::Frame`] / [`tether_codec::GpuFrame`]
/// for consistency — the producer and consumer end of the same
/// architectural split.
pub enum CapturedFrame {
    Cpu(CpuFrame),
    Gpu(GpuCapturedFrame),
}

/// CPU-resident captured frame (BGRA / RGBA / NV12 bytes). The SHM
/// fallback path and the test pattern produce these.
pub struct CpuFrame {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub data: Vec<u8>,
    /// Source timestamp from the capture API (e.g. `CVTimeStamp`, PipeWire
    /// `pts`). For backends that don't expose this, falls back to the
    /// userspace timestamp.
    pub t_capture_kernel: MonoNanos,
    /// Monotonic time at which our userspace code first observed the
    /// frame. Always populated.
    pub t_capture_userspace: MonoNanos,
    /// Backend-supplied damage hint, when the capture API exposes one.
    /// `None` means the backend has no opinion; the consumer falls back
    /// to the hash classifier. See [`damage::NativeDamage`].
    pub native_damage: Option<damage::NativeDamage>,
}

/// GPU-resident captured frame. The descriptor varies per platform via
/// [`GpuCapturedSource`]; a release guard keeps backend objects needed by the
/// consumer alive until it is done with them. The exact guarantee is
/// backend-specific: ScreenCaptureKit retains the `CMSampleBuffer` that owns
/// the `IOSurface`, while Linux DMA-BUF capture dup's the fd so the object
/// remains importable after PipeWire requeues the slot. Linux does not promise
/// exclusive content ownership after requeue; consumers rely on the capture
/// pool's latency headroom and should copy/submit promptly.
pub struct GpuCapturedFrame {
    pub width: u32,
    pub height: u32,
    pub source: GpuCapturedSource,
    pub t_capture_kernel: MonoNanos,
    pub t_capture_userspace: MonoNanos,
    /// Opaque "hold this alive while the consumer reads the buffer"
    /// container. Dropped by the consumer once it has either copied
    /// the data into encoder-owned memory (VAAPI dups the dma-buf
    /// internally during `vaCreateSurfaces`) or otherwise no longer
    /// needs the source.
    pub release_guard: GpuCapturedGuard,
    /// Backend-supplied damage hint; mirrors [`CpuFrame::native_damage`].
    pub native_damage: Option<damage::NativeDamage>,
}

/// Per-platform GPU buffer descriptor. Gated on `target_os` so the
/// host's match is exhaustive on each platform without a catch-all
/// that silently swallows future variants — same pattern as
/// [`tether_codec::GpuFrameSource`].
pub enum GpuCapturedSource {
    /// Linux DMA-BUF (typically from PipeWire DMA-BUF buffer-type).
    /// Single-plane BGRx/BGRA from the compositor; multi-plane
    /// negotiation will be a future addition when there's a
    /// compositor known to produce multi-plane capture buffers.
    #[cfg(target_os = "linux")]
    DmaBuf(CapturedDmaBuf),
    /// macOS IOSurface from ScreenCaptureKit's `CMSampleBuffer`. The
    /// real CFRetain on the underlying `IOSurfaceRef` (and the
    /// `CMSampleBuffer` keeping it alive) lives in the parent
    /// [`GpuCapturedFrame::release_guard`] — the pointer here is a
    /// non-owning view, valid until the guard is dropped.
    #[cfg(target_os = "macos")]
    IOSurface(CapturedIOSurface),
    /// Windows D3D11 texture from DXGI Desktop Duplication. The
    /// texture is an owned pool copy (the duplication surface is
    /// released immediately after `CopyResource`). Carries a
    /// reference to the shared device so downstream consumers can
    /// operate without cross-device copies.
    #[cfg(target_os = "windows")]
    D3D11Texture(windows::CapturedD3D11Texture),
}

/// Linux DMA-BUF descriptor for a captured frame. Mirrors what
/// `tether_codec::DmaBufObject + DmaBufLayer` carry for a single-plane
/// surface; kept separate so `tether-capture` doesn't depend on
/// `tether-codec` (capture/encode stay decoupled).
#[cfg(target_os = "linux")]
pub struct CapturedDmaBuf {
    /// DRM fourcc of the source plane (typically `XR24`/`AR24`/`XB24`
    /// etc.) as supplied by PipeWire's negotiated format.
    pub fourcc: u32,
    pub fd: std::os::fd::OwnedFd,
    pub stride: u64,
    pub offset: u64,
    pub modifier: u64,
}

/// macOS IOSurface descriptor for a captured frame. Mirrors the shape
/// `tether_codec::IOSurfaceFrame` carries for the encoder side; kept
/// separate so `tether-capture` doesn't depend on `tether-codec` (same
/// rationale as [`CapturedDmaBuf`] vs `DmaBufFrame`).
///
/// The pointer is a non-owning view; lifetime is the parent
/// [`GpuCapturedFrame::release_guard`], which retains the
/// `CMSampleBuffer` (and transitively the IOSurface). Dropping the
/// guard releases both.
#[cfg(target_os = "macos")]
pub struct CapturedIOSurface {
    /// `IOSurfaceRef` — opaque Apple type, valid until the parent
    /// `release_guard` is dropped.
    pub surface: *mut std::ffi::c_void,
    /// `kCVPixelFormatType_*` fourcc as returned by
    /// `IOSurfaceGetPixelFormat`. Typically NV12
    /// (`420YpCbCr8BiPlanarVideoRange` = `'420v'`).
    pub pixel_format: u32,
    pub width: u32,
    pub height: u32,
}

// No `&mut` access is possible to the IOSurface through this raw
// pointer from Rust; all mutation goes through Apple's IOSurface C
// API, which is itself thread-safe (CF-style refcounted, kernel
// surface thread-shareable). The struct carries no Rust state that
// would conflict with crossing a thread boundary.
#[cfg(target_os = "macos")]
unsafe impl Send for CapturedIOSurface {}

impl CapturedFrame {
    #[must_use]
    pub fn width(&self) -> u32 {
        match self {
            Self::Cpu(f) => f.width,
            Self::Gpu(f) => f.width,
        }
    }
    #[must_use]
    pub fn height(&self) -> u32 {
        match self {
            Self::Cpu(f) => f.height,
            Self::Gpu(f) => f.height,
        }
    }
    /// `(t_capture_kernel, t_capture_userspace)` — populated for both
    /// variants. The host's timing-metric path doesn't care which
    /// shape produced the frame.
    #[must_use]
    pub fn timestamps(&self) -> (MonoNanos, MonoNanos) {
        match self {
            Self::Cpu(f) => (f.t_capture_kernel, f.t_capture_userspace),
            Self::Gpu(f) => (f.t_capture_kernel, f.t_capture_userspace),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("capture backend not available on this platform")]
    Unsupported,
    #[error("portal: {0}")]
    Portal(String),
    #[error("pipewire: {0}")]
    PipeWire(String),
    #[error("display topology: {0}")]
    Display(String),
    /// ScreenCaptureKit (macOS) error — typically a permission denial
    /// (`NSScreenCaptureUsageDescription` missing or TCC denied),
    /// `SCShareableContent::get` failure, or `start_capture` rejection.
    /// Carries the framework error's display form.
    #[error("ScreenCaptureKit: {0}")]
    Sck(String),
    /// DXGI Desktop Duplication (Windows) error — typically
    /// `DuplicateOutput` failure, access lost (driver reset / mode
    /// change), or D3D11 device creation failure.
    #[cfg(target_os = "windows")]
    #[error("DXGI: {0}")]
    Dxgi(String),
}

pub type Result<T> = std::result::Result<T, CaptureError>;

#[cfg(target_os = "linux")]
impl From<ashpd::Error> for CaptureError {
    fn from(e: ashpd::Error) -> Self {
        Self::Portal(e.to_string())
    }
}

#[cfg(target_os = "linux")]
impl From<pipewire::Error> for CaptureError {
    fn from(e: pipewire::Error) -> Self {
        Self::PipeWire(e.to_string())
    }
}

#[cfg(target_os = "macos")]
impl From<screencapturekit::error::SCError> for CaptureError {
    fn from(e: screencapturekit::error::SCError) -> Self {
        Self::Sck(e.to_string())
    }
}

#[cfg(test)]
mod display_tests {
    use super::*;

    #[test]
    fn test_pattern_display_reports_current_mode() {
        let display = test_pattern_display(320, 240, 60_000);
        assert_eq!(display.id, DisplayId(0));
        assert_eq!(display.current_mode, DisplayMode::new(320, 240, 60_000));
        assert_eq!(display.available_modes, vec![display.current_mode]);
        assert!(display.primary);
        assert!(!display.can_set_mode);
    }

    #[test]
    fn primary_mode_update_preserves_topology_metadata() {
        let mut display = test_pattern_display(1280, 720, 60_000);
        display.name = "DP-3".to_string();
        display.position = (1920, 0);
        display.scale_num = 3;
        display.scale_den = 2;

        let updated = display_list_with_primary_mode(vec![display], 2560, 1440, 144_000);
        assert_eq!(updated.len(), 1);
        assert_eq!(updated[0].name, "DP-3");
        assert_eq!(updated[0].position, (1920, 0));
        assert_eq!(updated[0].scale_num, 3);
        assert_eq!(updated[0].scale_den, 2);
        assert_eq!(
            updated[0].current_mode,
            DisplayMode::new(2560, 1440, 144_000)
        );
        assert_eq!(updated[0].available_modes, vec![updated[0].current_mode]);
    }

    #[test]
    fn capture_geometry_derives_scale_from_host_logical_size() {
        let mut display = test_pattern_display(3840, 2400, 60_000);
        display.scale_num = 2;
        display.scale_den = 1;

        let updated = display_list_with_capture_geometry(
            vec![display],
            CaptureDisplayGeometry {
                capture_width: 1920,
                capture_height: 1200,
                refresh_millihz: 60_000,
                host_logical_size: Some((1920, 1200)),
                host_logical_position: Some((0, 0)),
            },
        );

        assert_eq!(
            updated[0].current_mode,
            DisplayMode::new(1920, 1200, 60_000)
        );
        assert_eq!(updated[0].scale_num, 1);
        assert_eq!(updated[0].scale_den, 1);
    }

    #[test]
    fn capture_geometry_keeps_hidpi_scale_when_framebuffer_is_backing_pixels() {
        let mut display = test_pattern_display(3840, 2400, 60_000);
        display.scale_num = 2;
        display.scale_den = 1;

        let updated = display_list_with_capture_geometry(
            vec![display],
            CaptureDisplayGeometry {
                capture_width: 3840,
                capture_height: 2400,
                refresh_millihz: 60_000,
                host_logical_size: Some((1920, 1200)),
                host_logical_position: Some((0, 0)),
            },
        );

        assert_eq!(updated[0].scale_num, 2);
        assert_eq!(updated[0].scale_den, 1);
    }

    #[test]
    fn capture_geometry_can_select_display_by_logical_portal_rect() {
        let primary_mode = DisplayMode::new(3840, 2160, 60_000);
        let secondary_mode = DisplayMode::new(3840, 2400, 60_000);
        let displays = vec![
            DisplayDescriptor {
                id: DisplayId(0),
                name: "DP-1".into(),
                scale_num: 2,
                scale_den: 1,
                primary: true,
                position: (0, 0),
                current_mode: primary_mode,
                available_modes: vec![primary_mode],
                can_set_mode: false,
            },
            DisplayDescriptor {
                id: DisplayId(1),
                name: "DP-2".into(),
                scale_num: 2,
                scale_den: 1,
                primary: false,
                position: (3840, 0),
                current_mode: secondary_mode,
                available_modes: vec![secondary_mode],
                can_set_mode: false,
            },
        ];

        let updated = display_list_with_capture_geometry(
            displays,
            CaptureDisplayGeometry {
                capture_width: 1920,
                capture_height: 1200,
                refresh_millihz: 60_000,
                host_logical_size: Some((1920, 1200)),
                host_logical_position: Some((1920, 0)),
            },
        );

        assert!(!updated[0].primary);
        assert!(updated[1].primary);
        assert_eq!(updated[1].scale_num, 1);
        assert_eq!(updated[1].scale_den, 1);
    }

    #[test]
    fn primary_mode_update_selects_matching_display_scale() {
        let primary_mode = DisplayMode::new(1920, 1080, 60_000);
        let secondary_mode = DisplayMode::new(3840, 2160, 60_000);
        let displays = vec![
            DisplayDescriptor {
                id: DisplayId(0),
                name: "DP-1".into(),
                scale_num: 1,
                scale_den: 1,
                primary: true,
                position: (0, 0),
                current_mode: primary_mode,
                available_modes: vec![primary_mode],
                can_set_mode: false,
            },
            DisplayDescriptor {
                id: DisplayId(1),
                name: "DP-2".into(),
                scale_num: 2,
                scale_den: 1,
                primary: false,
                position: (1920, 0),
                current_mode: secondary_mode,
                available_modes: vec![secondary_mode],
                can_set_mode: false,
            },
        ];

        let updated = display_list_with_primary_mode(displays, 3840, 2160, 120_000);
        assert!(!updated[0].primary);
        assert!(updated[1].primary);
        assert_eq!(updated[1].scale_num, 2);
        assert_eq!(updated[1].scale_den, 1);
        assert_eq!(
            updated[1].current_mode,
            DisplayMode::new(3840, 2160, 120_000)
        );
    }

    #[test]
    fn scale_factor_uses_reduced_rational() {
        assert_eq!(scale_to_ratio(1.5), (3, 2));
        assert_eq!(scale_to_ratio(2.0), (2, 1));
        assert_eq!(scale_to_ratio(0.0), (1, 1));
    }

    #[test]
    fn scale_from_mode_and_logical_size_prefers_direct_logical_geometry() {
        assert_eq!(
            scale_from_mode_and_logical_size((1920, 1200), Some((1536, 960)), 2.0),
            (5, 4)
        );
    }

    #[test]
    fn scale_from_mode_and_logical_size_falls_back_without_logical_geometry() {
        assert_eq!(
            scale_from_mode_and_logical_size((1920, 1200), None, 2.0),
            (2, 1)
        );
    }

    #[test]
    fn logical_position_scales_to_physical_display_position() {
        assert_eq!(
            scaled_i32_pair_from_logical((1536, -960), 5, 4),
            Some((1920, -1200))
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::bounded;

    // The Windows drop-oldest producer keeps its own receiver clone to
    // evict the stale mailbox frame, which masks channel `Disconnected`.
    // It therefore shuts down off the liveness token instead. These
    // tests pin the exact contract that producer relies on.

    #[test]
    fn liveness_tracks_frame_receiver_lifetime() {
        let (_tx, rx) = bounded::<CapturedFrame>(1);
        let handle = CaptureHandle::from_parts(rx, Arc::new(AtomicU32::new(60)));

        let weak = handle.liveness();
        assert_eq!(weak.strong_count(), 1, "handle holds the liveness token");

        let frames = handle.into_rx();
        assert_eq!(
            weak.strong_count(),
            1,
            "token moves into the FrameReceiver — still a live consumer"
        );

        drop(frames);
        assert_eq!(
            weak.strong_count(),
            0,
            "dropping the receiver signals the consumer is gone"
        );
    }

    #[test]
    fn liveness_drops_when_handle_discarded_without_into_rx() {
        let (_tx, rx) = bounded::<CapturedFrame>(1);
        let handle = CaptureHandle::from_parts(rx, Arc::new(AtomicU32::new(60)));
        let weak = handle.liveness();

        drop(handle);
        assert_eq!(
            weak.strong_count(),
            0,
            "a handle dropped without into_rx means no consumer ever materialised"
        );
    }
}
