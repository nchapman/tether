//! Cursor channel — separate from video and input to keep pointer
//! traffic on its own unreliable, latest-wins path.
//!
//! Two directions, two packet types:
//! - [`HostCursorPacket`]: host → client. The host's pointer sprite as
//!   rendered on its own display, so the client can paint a cursor that
//!   tracks the OS pointer at full host frame rate without waiting for
//!   the encoded video to catch up.
//! - [`ClientCursorPacket`]: client → host. The client window's pointer
//!   position, in `[0,1]` video-region coordinates. Routed through the
//!   datagram channel rather than the reliable input stream so a queue
//!   of keystrokes can't head-of-line-block the cursor.

use crate::MonoNanos;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CursorPixelFormat {
    Rgba8,
}

/// Host → client. The host sends `Position` updates at a high rate
/// (250-500 Hz) and emits `Shape` only when the cursor shape changes.
/// Clients keep a small cache of shapes keyed by `id`. Not implemented
/// in v0; the wire shape is reserved.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HostCursorPacket {
    Position {
        t_capture: MonoNanos,
        x: i32,
        y: i32,
        visible: bool,
    },
    Shape {
        id: u64,
        hotspot_x: u16,
        hotspot_y: u16,
        width: u16,
        height: u16,
        format: CursorPixelFormat,
        pixels: Vec<u8>,
    },
    UseShape {
        id: u64,
    },
}

/// Client → host. Pointer position from the client window, normalised
/// to `[0,1]^2` inside the video region (same coordinate space as the
/// original `InputEventKind::MousePosition` had). Carries a per-
/// connection monotonic `seq` so the host can drop reordered datagrams
/// — datagrams can reorder freely and the input stream's `event_id`
/// counter doesn't extend to the cursor path.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClientCursorPacket {
    pub seq: u32,
    pub display_idx: u8,
    pub x: f32,
    pub y: f32,
    /// `false` lets the client signal "pointer left the video region"
    /// so the host can hide its synthetic cursor / stop applying
    /// position updates.
    pub visible: bool,
    pub t_client: MonoNanos,
}
