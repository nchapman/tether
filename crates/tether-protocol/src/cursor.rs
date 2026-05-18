//! Cursor channel — separate from video to minimize perceived pointer lag.

use crate::MonoNanos;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CursorPixelFormat {
    Rgba8,
}

/// A datagram on the cursor channel. The host sends `Position` updates at a
/// high rate (250-500 Hz) and emits `Shape` only when the cursor shape
/// changes. Clients keep a small cache of shapes keyed by `id`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CursorPacket {
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
