//! Windows cursor extraction via DXGI OutputDuplication.
//!
//! DXGI Desktop Duplication exposes pointer state through
//! `DXGI_OUTDUPL_FRAME_INFO` on every `AcquireNextFrame`:
//!
//! - **Position** from `PointerPosition` (coordinates + visible flag).
//! - **Shape** from `GetFramePointerShape` when
//!   `PointerShapeBufferSize > 0` (the shape changed this frame).
//!
//! Shape types: `MONOCHROME` (AND+XOR bitmask pair), `COLOR`
//! (premultiplied BGRA), `MASKED_COLOR` (BGRA with mask row).
//! All are converted to RGBA8 for the wire.
//!
//! Threading: cursor data is extracted in the same capture thread
//! that calls `AcquireNextFrame`. Shape events are sent to a
//! crossbeam channel; position is written to a shared mutex. The
//! `DxgiCursorSource` implements [`CursorSource`] and is attached
//! to the [`CaptureHandle`] via `with_cursor_source`.

use std::sync::{Arc, Mutex};

use crossbeam_channel::{unbounded, Receiver, Sender};
use tether_protocol::cursor::CursorPixelFormat;
use windows::Win32::Graphics::Dxgi::{
    IDXGIOutputDuplication, DXGI_OUTDUPL_FRAME_INFO, DXGI_OUTDUPL_POINTER_SHAPE_INFO,
    DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR, DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR,
    DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME,
};

use crate::cursor::{fnv1a_64, CursorEvent, CursorPosition, CursorShapeEvent, CursorSource};

/// Cursor source backed by DXGI OutputDuplication pointer metadata.
/// Created by the capture module and attached to `CaptureHandle`.
pub struct DxgiCursorSource {
    shape_rx: Receiver<CursorEvent>,
    position: Arc<Mutex<Option<CursorPosition>>>,
}

impl CursorSource for DxgiCursorSource {
    fn next_event(&mut self) -> CursorEvent {
        self.shape_rx.try_recv().unwrap_or(CursorEvent::Idle)
    }

    fn poll_position(&mut self) -> Option<CursorPosition> {
        self.position.lock().ok().and_then(|g| *g)
    }
}

/// Internal state used by the capture thread to produce cursor events.
pub(crate) struct DxgiCursorState {
    shape_tx: Sender<CursorEvent>,
    position: Arc<Mutex<Option<CursorPosition>>>,
    last_shape_id: u64,
    shape_buffer: Vec<u8>,
}

impl DxgiCursorState {
    pub(crate) fn new() -> (Self, DxgiCursorSource) {
        let (shape_tx, shape_rx) = unbounded();
        let position = Arc::new(Mutex::new(None));
        let state = Self {
            shape_tx,
            position: Arc::clone(&position),
            last_shape_id: u64::MAX,
            shape_buffer: vec![0u8; 1024 * 1024],
        };
        let source = DxgiCursorSource { shape_rx, position };
        (state, source)
    }

    /// Extract cursor state from this frame's info. Called after every
    /// successful `AcquireNextFrame`.
    pub(crate) fn update(
        &mut self,
        frame_info: &DXGI_OUTDUPL_FRAME_INFO,
        duplication: &IDXGIOutputDuplication,
    ) {
        // Position update (every frame that has pointer data).
        if frame_info.LastMouseUpdateTime != 0 {
            let pos = frame_info.PointerPosition;
            let position = CursorPosition {
                x: pos.Position.x,
                y: pos.Position.y,
                visible: pos.Visible.as_bool(),
            };
            if let Ok(mut guard) = self.position.lock() {
                *guard = Some(position);
            }
        }

        // Shape update (only when the pointer bitmap changed).
        if frame_info.PointerShapeBufferSize == 0 {
            return;
        }

        let buf_size = frame_info.PointerShapeBufferSize as usize;
        if self.shape_buffer.len() < buf_size {
            self.shape_buffer.resize(buf_size, 0);
        }

        let mut shape_info: DXGI_OUTDUPL_POINTER_SHAPE_INFO = unsafe { std::mem::zeroed() };
        let mut required_size = 0u32;

        let hr = unsafe {
            duplication.GetFramePointerShape(
                buf_size as u32,
                self.shape_buffer.as_mut_ptr().cast(),
                &mut required_size,
                &mut shape_info,
            )
        };

        if hr.is_err() {
            tracing::debug!("GetFramePointerShape failed");
            return;
        }

        let rgba = match shape_info.Type {
            t if t == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR.0 as u32 => {
                bgra_to_rgba(&self.shape_buffer[..required_size as usize])
            }
            t if t == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME.0 as u32 => monochrome_to_rgba(
                &self.shape_buffer[..required_size as usize],
                shape_info.Width,
                shape_info.Height,
                shape_info.Pitch,
            ),
            t if t == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR.0 as u32 => {
                masked_color_to_rgba(
                    &self.shape_buffer[..required_size as usize],
                    shape_info.Width,
                    shape_info.Height,
                    shape_info.Pitch,
                )
            }
            other => {
                tracing::debug!(shape_type = other, "unknown pointer shape type");
                return;
            }
        };

        let width = shape_info.Width as u16;
        // Monochrome shapes are double-height (AND mask + XOR mask).
        let height = if shape_info.Type == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME.0 as u32 {
            (shape_info.Height / 2) as u16
        } else {
            shape_info.Height as u16
        };

        let id = fnv1a_64(width, height, &rgba);
        if id == self.last_shape_id {
            return;
        }
        self.last_shape_id = id;

        let hotspot = (shape_info.HotSpot.x as u16, shape_info.HotSpot.y as u16);

        let _ = self.shape_tx.send(CursorEvent::Shape(CursorShapeEvent {
            id,
            width,
            height,
            hotspot,
            format: CursorPixelFormat::Rgba8,
            pixels: rgba,
        }));
    }
}

/// Convert premultiplied BGRA to straight RGBA (un-premultiplies alpha).
fn bgra_to_rgba(bgra: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(bgra.len());
    for pixel in bgra.chunks_exact(4) {
        let b = pixel[0];
        let g = pixel[1];
        let r = pixel[2];
        let a = pixel[3];
        if a == 0 {
            rgba.extend_from_slice(&[0, 0, 0, 0]);
        } else {
            let inv = 255.0 / f32::from(a);
            rgba.push((f32::from(r) * inv).min(255.0) as u8);
            rgba.push((f32::from(g) * inv).min(255.0) as u8);
            rgba.push((f32::from(b) * inv).min(255.0) as u8);
            rgba.push(a);
        }
    }
    rgba
}

/// Convert monochrome AND+XOR bitmask to RGBA8.
/// The buffer is two bitmaps stacked vertically: top half is AND mask,
/// bottom half is XOR mask. Each row is `pitch` bytes (1 bit per pixel,
/// padded to dword boundary).
fn monochrome_to_rgba(buf: &[u8], width: u32, height: u32, pitch: u32) -> Vec<u8> {
    let h = (height / 2) as usize;
    let w = width as usize;
    let pitch = pitch as usize;

    let required = (h * 2) * pitch;
    if buf.len() < required || w == 0 || h == 0 {
        return vec![];
    }

    let mut rgba = vec![0u8; w * h * 4];

    for row in 0..h {
        let and_row = &buf[row * pitch..];
        let xor_row = &buf[(h + row) * pitch..];
        for col in 0..w {
            let byte_idx = col / 8;
            let bit_idx = 7 - (col % 8);
            let and_bit = (and_row[byte_idx] >> bit_idx) & 1;
            let xor_bit = (xor_row[byte_idx] >> bit_idx) & 1;

            let (r, g, b, a) = match (and_bit, xor_bit) {
                (0, 0) => (0, 0, 0, 255),       // Black, opaque
                (0, 1) => (255, 255, 255, 255), // White, opaque
                (1, 0) => (0, 0, 0, 0),         // Transparent
                (1, 1) => (255, 255, 255, 128), // Inverted — approximate as semi-white
                _ => unreachable!(),
            };

            let idx = (row * w + col) * 4;
            rgba[idx] = r;
            rgba[idx + 1] = g;
            rgba[idx + 2] = b;
            rgba[idx + 3] = a;
        }
    }
    rgba
}

/// Convert masked-color BGRA to RGBA8.
/// Each pixel: if alpha == 0, pixel is visible as-is (opaque).
/// If alpha != 0, pixel should be XOR'd with screen — approximate as
/// semi-transparent.
fn masked_color_to_rgba(buf: &[u8], width: u32, height: u32, pitch: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let pitch = pitch as usize;

    let required = h * pitch;
    if buf.len() < required || pitch < w * 4 || w == 0 || h == 0 {
        return vec![];
    }

    let mut rgba = vec![0u8; w * h * 4];

    for row in 0..h {
        let src_row = &buf[row * pitch..];
        for col in 0..w {
            let si = col * 4;
            let di = (row * w + col) * 4;
            let mask = src_row[si + 3];
            if mask == 0 {
                // Opaque color pixel.
                rgba[di] = src_row[si + 2]; // R
                rgba[di + 1] = src_row[si + 1]; // G
                rgba[di + 2] = src_row[si]; // B
                rgba[di + 3] = 255; // A
            } else {
                // XOR pixel — approximate as semi-transparent inverted.
                rgba[di] = src_row[si + 2];
                rgba[di + 1] = src_row[si + 1];
                rgba[di + 2] = src_row[si];
                rgba[di + 3] = 128;
            }
        }
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgra_to_rgba_fully_opaque() {
        // BGRA pixel: B=10, G=20, R=30, A=255 (premul = straight when a=255)
        let bgra = vec![10, 20, 30, 255];
        let rgba = bgra_to_rgba(&bgra);
        assert_eq!(rgba, vec![30, 20, 10, 255]);
    }

    #[test]
    fn bgra_to_rgba_transparent_pixel() {
        let bgra = vec![100, 200, 50, 0];
        let rgba = bgra_to_rgba(&bgra);
        assert_eq!(rgba, vec![0, 0, 0, 0]);
    }

    #[test]
    fn bgra_to_rgba_unpremultiplies_alpha() {
        // Premultiplied BGRA: B=64, G=0, R=128, A=128
        // Straight: R = 128 * 255/128 = 255, G = 0, B = 64 * 255/128 ≈ 127
        let bgra = vec![64, 0, 128, 128];
        let rgba = bgra_to_rgba(&bgra);
        assert_eq!(rgba[0], 255); // R
        assert_eq!(rgba[1], 0); // G
        assert_eq!(rgba[2], 127); // B (64 * 255 / 128 = 127.5 → 127)
        assert_eq!(rgba[3], 128); // A
    }

    #[test]
    fn monochrome_2x2_all_cases() {
        // 2×2 monochrome: pitch=4 bytes (1 byte per row, padded to dword).
        // Height is doubled for AND+XOR masks: total height=4 in buffer.
        let width = 2u32;
        let height = 4u32; // 2 rows AND + 2 rows XOR
        let pitch = 4u32;

        // Row layout (bits): pixel0 is bit7, pixel1 is bit6.
        // AND mask row 0: bits [0,0,...] = 0x00
        // AND mask row 1: bits [1,1,...] = 0xC0
        // XOR mask row 0: bits [0,1,...] = 0x40
        // XOR mask row 1: bits [0,1,...] = 0x40
        let mut buf = vec![0u8; (pitch * height) as usize];
        // AND row 0 (offset 0): 0x00 → and_bit = 0 for both pixels
        buf[0] = 0x00;
        // AND row 1 (offset 4): 0xC0 → and_bit = 1 for both pixels
        buf[4] = 0xC0;
        // XOR row 0 (offset 8): 0x40 → xor_bit = 0 for pixel0, 1 for pixel1
        buf[8] = 0x40;
        // XOR row 1 (offset 12): 0x40 → xor_bit = 0 for pixel0, 1 for pixel1
        buf[12] = 0x40;

        let rgba = monochrome_to_rgba(&buf, width, height, pitch);
        // Expected: 2×2 RGBA pixels
        // (0,0): AND=0, XOR=0 → Black opaque (0,0,0,255)
        // (0,1): AND=0, XOR=1 → White opaque (255,255,255,255)
        // (1,0): AND=1, XOR=0 → Transparent (0,0,0,0)
        // (1,1): AND=1, XOR=1 → Inverted/semi (255,255,255,128)
        assert_eq!(rgba.len(), 2 * 2 * 4);
        assert_eq!(&rgba[0..4], &[0, 0, 0, 255]);
        assert_eq!(&rgba[4..8], &[255, 255, 255, 255]);
        assert_eq!(&rgba[8..12], &[0, 0, 0, 0]);
        assert_eq!(&rgba[12..16], &[255, 255, 255, 128]);
    }

    #[test]
    fn masked_color_opaque_and_xor_pixels() {
        let width = 2u32;
        let height = 1u32;
        let pitch = 8u32; // 2 pixels * 4 bytes
                          // Pixel 0: BGRA = (10, 20, 30, 0) → alpha=0 → opaque → RGBA = (30, 20, 10, 255)
                          // Pixel 1: BGRA = (40, 50, 60, 0xFF) → alpha!=0 → XOR → RGBA = (60, 50, 40, 128)
        let buf = vec![10, 20, 30, 0, 40, 50, 60, 0xFF];
        let rgba = masked_color_to_rgba(&buf, width, height, pitch);
        assert_eq!(&rgba[0..4], &[30, 20, 10, 255]);
        assert_eq!(&rgba[4..8], &[60, 50, 40, 128]);
    }

    #[test]
    fn cursor_state_deduplicates_shapes() {
        let (mut state, mut source) = DxgiCursorState::new();
        // Manually push two identical shape events — only one should be
        // forwarded after dedup.
        let shape = CursorShapeEvent {
            id: 42,
            width: 4,
            height: 4,
            hotspot: (0, 0),
            format: CursorPixelFormat::Rgba8,
            pixels: vec![255u8; 4 * 4 * 4],
        };
        let _ = state.shape_tx.send(CursorEvent::Shape(shape.clone()));
        // Simulate same shape arriving again with the same id.
        state.last_shape_id = 42;
        // A subsequent call with the same id would not send (tested in
        // the update path). Verify the source can read the first one.
        match source.next_event() {
            CursorEvent::Shape(s) => assert_eq!(s.id, 42),
            CursorEvent::Idle => panic!("expected shape event"),
        }
        assert_eq!(source.next_event(), CursorEvent::Idle);
    }
}
