//! Cursor channel — separate from video and input to keep pointer
//! traffic on its own unreliable, latest-wins path.
//!
//! Two directions, two packet types:
//! - [`HostCursorPacket`]: host → client. The host's pointer position
//!   as rendered on its own display, so the client can paint a cursor
//!   that tracks the OS pointer at full host frame rate without waiting
//!   for the encoded video to catch up. Position only — cursor shapes
//!   live on the reliable control stream (`ControlMessage::CursorShape`,
//!   `ControlMessage::CursorUseShape`) because a 64×64 RGBA sprite
//!   (16 KB) does not fit in a 1200-byte datagram and a corrupted /
//!   partially-delivered shape is worse than no shape at all.
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

/// Host → client. Sent at a high rate (250-500 Hz) while the pointer is
/// over the host's display. Cursor *shapes* are not on this channel —
/// see the module-level doc.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HostCursorPacket {
    Position {
        t_capture: MonoNanos,
        x: i32,
        y: i32,
        visible: bool,
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
