//! Windows screen capture via DXGI Desktop Duplication.
//!
//! Calling [`start`] enumerates adapters, picks the primary output,
//! duplicates the desktop, and spawns a dedicated thread that emits one
//! [`CapturedFrame::Gpu`] per acquired frame carrying a live
//! `ID3D11Texture2D`. The texture is copied from the duplication
//! surface into a pool of owned textures so `ReleaseFrame` can be
//! called promptly (DXGI requires release before the next acquire).
//!
//! The D3D11 device is created once and shared with the encoder via
//! [`D3D11Device`] (an `Arc`-wrapped handle). The encoder must be
//! constructed on the same device to avoid cross-device copies.
//!
//! Shutdown: dropping the returned receiver causes the next loop
//! iteration to see a disconnected channel and exit cleanly.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{bounded, Sender, TrySendError};
use tether_protocol::MonoNanos;
use windows::core::Interface;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_UNKNOWN;
use windows::Win32::Graphics::Direct3D11::{
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_DEFAULT,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1, IDXGIOutput, IDXGIOutput1,
    IDXGIOutputDuplication, IDXGIResource, DXGI_ERROR_ACCESS_LOST,
    DXGI_ERROR_WAIT_TIMEOUT, DXGI_OUTDUPL_FRAME_INFO,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM,
};
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};

use crate::cursor_windows::DxgiCursorState;
use crate::damage::NativeDamage;
use crate::{
    CaptureError, CaptureHandle, CapturedFrame, GpuCapturedFrame, GpuCapturedGuard,
    GpuCapturedSource, Result,
};

const CAPTURE_CHANNEL_DEPTH: usize = 2;
const CAPTURE_FPS: u32 = 60;
const TEXTURE_POOL_SIZE: usize = 3;

/// Shared D3D11 device handle. The capture thread creates it; the
/// encoder receives a clone so both operate on the same device (no
/// cross-device texture copies).
#[derive(Clone)]
pub struct D3D11Device {
    pub device: ID3D11Device,
    pub context: ID3D11DeviceContext,
}

// SAFETY: ID3D11Device and ID3D11DeviceContext are thread-safe COM
// objects when created with D3D11_CREATE_DEVICE_SINGLETHREADED not set
// (our default). The D3D11 runtime serialises concurrent calls
// internally via the implicit multithread contract.
unsafe impl Send for D3D11Device {}
unsafe impl Sync for D3D11Device {}

/// Descriptor for a captured D3D11 texture. Carries the texture
/// handle and a reference to the shared device so downstream
/// consumers (encoder, converter) can operate on it without a
/// separate device-lookup step.
pub struct CapturedD3D11Texture {
    /// The captured frame as an owned `ID3D11Texture2D`. This is a
    /// pool texture that received a `CopyResource` from the
    /// duplication surface — not the duplication surface itself
    /// (which must be released promptly).
    pub texture: ID3D11Texture2D,
    /// Shared device handle. Encoder uses this to avoid cross-device
    /// copies.
    pub device: D3D11Device,
    pub width: u32,
    pub height: u32,
    /// Pixel format of the texture (typically BGRA8).
    pub format: DXGI_FORMAT,
}

// SAFETY: ID3D11Texture2D is a COM object ref-counted and thread-safe
// under the same conditions as the device.
unsafe impl Send for CapturedD3D11Texture {}

/// Start DXGI Desktop Duplication on the primary output.
///
/// Returns a [`CaptureHandle`] whose receiver emits
/// [`CapturedFrame::Gpu`] frames containing D3D11 textures on the
/// shared device.
pub fn start() -> Result<(CaptureHandle, D3D11Device)> {
    let (device, context) = create_d3d11_device()?;
    let shared_device = D3D11Device {
        device: device.clone(),
        context: context.clone(),
    };

    let (duplication, width, height) = create_duplication(&device)?;

    let (tx, rx) = bounded::<CapturedFrame>(CAPTURE_CHANNEL_DEPTH);
    let target_fps = Arc::new(AtomicU32::new(CAPTURE_FPS));
    let target_fps_thread = Arc::clone(&target_fps);
    let device_thread = shared_device.clone();

    let (cursor_state, cursor_source) = DxgiCursorState::new();

    std::thread::Builder::new()
        .name("tether-capture-dxgi".into())
        .spawn(move || {
            run_capture_thread(
                tx,
                device_thread,
                duplication,
                width,
                height,
                target_fps_thread,
                cursor_state,
            );
        })
        .map_err(|e| CaptureError::Io(e))?;

    let handle =
        CaptureHandle::from_parts(rx, target_fps).with_cursor_source(Box::new(cursor_source));
    Ok((handle, shared_device))
}

fn create_d3d11_device() -> Result<(ID3D11Device, ID3D11DeviceContext)> {
    let factory: IDXGIFactory1 =
        unsafe { CreateDXGIFactory1() }.map_err(|e| CaptureError::Io(hresult_io(e)))?;

    let adapter: IDXGIAdapter1 = unsafe { factory.EnumAdapters1(0) }
        .map_err(|e| CaptureError::Io(hresult_io(e)))?;

    let mut device = None;
    let mut context = None;

    unsafe {
        D3D11CreateDevice(
            &adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
    }
    .map_err(|e| CaptureError::Io(hresult_io(e)))?;

    Ok((device.unwrap(), context.unwrap()))
}

fn create_duplication(
    device: &ID3D11Device,
) -> Result<(IDXGIOutputDuplication, u32, u32)> {
    let dxgi_device: windows::Win32::Graphics::Dxgi::IDXGIDevice =
        device.cast().map_err(|e| CaptureError::Io(hresult_io(e)))?;

    let adapter: IDXGIAdapter1 = unsafe { dxgi_device.GetParent() }
        .map_err(|e| CaptureError::Io(hresult_io(e)))?;

    let output: IDXGIOutput = unsafe { adapter.EnumOutputs(0) }
        .map_err(|e| CaptureError::Io(hresult_io(e)))?;

    let output1: IDXGIOutput1 = output
        .cast()
        .map_err(|e| CaptureError::Io(hresult_io(e)))?;

    let duplication = unsafe { output1.DuplicateOutput(device) }
        .map_err(|e| CaptureError::Io(hresult_io(e)))?;

    let desc = unsafe { duplication.GetDesc() };

    let width = desc.ModeDesc.Width;
    let height = desc.ModeDesc.Height;

    tracing::info!(
        width,
        height,
        format = ?desc.ModeDesc.Format,
        "DXGI Desktop Duplication initialized"
    );

    Ok((duplication, width, height))
}

fn create_pool_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> std::result::Result<ID3D11Texture2D, windows::core::Error> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: 0,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut texture = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture)) }?;
    Ok(texture.unwrap())
}

const RECONNECT_BACKOFF: &[Duration] = &[
    Duration::from_millis(50),
    Duration::from_millis(100),
    Duration::from_millis(200),
    Duration::from_millis(500),
    Duration::from_secs(1),
    Duration::from_secs(2),
];
const RECONNECT_MAX_TOTAL: Duration = Duration::from_secs(30);

fn run_capture_thread(
    tx: Sender<CapturedFrame>,
    device: D3D11Device,
    mut duplication: IDXGIOutputDuplication,
    mut width: u32,
    mut height: u32,
    target_fps: Arc<AtomicU32>,
    mut cursor_state: DxgiCursorState,
) {
    let mut pool = match create_texture_pool(&device.device, width, height) {
        Some(p) => p,
        None => return,
    };
    let mut pool_idx = 0usize;

    let qpc_freq = qpc_frequency();

    let mut frame_info: DXGI_OUTDUPL_FRAME_INFO = unsafe { std::mem::zeroed() };

    loop {
        let fps = target_fps.load(Ordering::Relaxed).max(1);
        let frame_interval = Duration::from_nanos(1_000_000_000 / u64::from(fps));

        let timeout_ms = frame_interval.as_millis().min(100) as u32;
        let resource: std::result::Result<IDXGIResource, _> = unsafe {
            let mut resource = None;
            let hr = duplication.AcquireNextFrame(timeout_ms, &mut frame_info, &mut resource);
            hr.map(|()| resource.unwrap())
        };

        let resource = match resource {
            Ok(r) => r,
            Err(e) if e.code() == DXGI_ERROR_WAIT_TIMEOUT => {
                continue;
            }
            Err(e) if e.code() == DXGI_ERROR_ACCESS_LOST => {
                match reconnect_duplication(&device.device) {
                    Some((new_dup, new_w, new_h)) => {
                        duplication = new_dup;
                        if new_w != width || new_h != height {
                            width = new_w;
                            height = new_h;
                            match create_texture_pool(&device.device, width, height) {
                                Some(p) => pool = p,
                                None => break,
                            }
                            pool_idx = 0;
                        }
                        tracing::info!(width, height, "DXGI reconnected after ACCESS_LOST");
                        continue;
                    }
                    None => {
                        tracing::error!("DXGI reconnect failed after 30s; exiting capture");
                        break;
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "AcquireNextFrame failed");
                break;
            }
        };

        let t_userspace = MonoNanos::now();

        let t_kernel = qpc_to_mono_nanos(frame_info.LastPresentTime, qpc_freq)
            .unwrap_or(t_userspace);

        cursor_state.update(&frame_info, &duplication);

        let native_damage = Some(NativeDamage {
            idle: frame_info.TotalMetadataBufferSize == 0,
        });

        if native_damage == Some(NativeDamage { idle: true }) {
            let _ = unsafe { duplication.ReleaseFrame() };
            continue;
        }

        let src_texture: ID3D11Texture2D = match resource.cast() {
            Ok(t) => t,
            Err(e) => {
                tracing::error!(error = %e, "failed to cast IDXGIResource to ID3D11Texture2D");
                let _ = unsafe { duplication.ReleaseFrame() };
                break;
            }
        };

        let dst_texture = &pool[pool_idx % TEXTURE_POOL_SIZE];
        unsafe {
            device.context.CopyResource(dst_texture, &src_texture);
        }

        let _ = unsafe { duplication.ReleaseFrame() };

        let frame_texture = dst_texture.clone();
        pool_idx = pool_idx.wrapping_add(1);

        let frame = CapturedFrame::Gpu(GpuCapturedFrame {
            width,
            height,
            source: GpuCapturedSource::D3D11Texture(CapturedD3D11Texture {
                texture: frame_texture,
                device: device.clone(),
                width,
                height,
                format: DXGI_FORMAT_B8G8R8A8_UNORM,
            }),
            t_capture_kernel: t_kernel,
            t_capture_userspace: t_userspace,
            release_guard: GpuCapturedGuard::new(()),
            native_damage,
        });

        match tx.try_send(frame) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => {
                tracing::debug!("capture receiver dropped; shutting down");
                break;
            }
        }
    }
}

fn create_texture_pool(device: &ID3D11Device, width: u32, height: u32) -> Option<Vec<ID3D11Texture2D>> {
    let mut pool = Vec::with_capacity(TEXTURE_POOL_SIZE);
    for _ in 0..TEXTURE_POOL_SIZE {
        match create_pool_texture(device, width, height) {
            Ok(tex) => pool.push(tex),
            Err(e) => {
                tracing::error!(error = %e, "failed to create texture pool");
                return None;
            }
        }
    }
    Some(pool)
}

/// Attempt to re-acquire DXGI OutputDuplication after ACCESS_LOST.
/// Retries with exponential backoff up to [`RECONNECT_MAX_TOTAL`].
/// First attempt is immediate (fast-user-switch recovers instantly);
/// backoff sleep happens only after a failed attempt.
fn reconnect_duplication(device: &ID3D11Device) -> Option<(IDXGIOutputDuplication, u32, u32)> {
    let start = std::time::Instant::now();
    let mut attempt = 0usize;

    loop {
        if start.elapsed() > RECONNECT_MAX_TOTAL {
            return None;
        }

        match create_duplication(device) {
            Ok(result) => return Some(result),
            Err(e) => {
                tracing::debug!(
                    attempt,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    error = %e,
                    "DXGI reconnect attempt failed"
                );
            }
        }

        let backoff = RECONNECT_BACKOFF[attempt.min(RECONNECT_BACKOFF.len() - 1)];
        std::thread::sleep(backoff);
        attempt += 1;
    }
}

/// Query QPC frequency (ticks per second).
fn qpc_frequency() -> u64 {
    let mut freq = 0i64;
    let _ = unsafe { QueryPerformanceFrequency(&mut freq) };
    freq as u64
}

/// Convert a DXGI `LastPresentTime` (QPC value) to a [`MonoNanos`]
/// relative to the same epoch as [`MonoNanos::now()`]. Works by
/// computing how far in the past the QPC timestamp is relative to the
/// current QPC, then subtracting that delta from `MonoNanos::now()`.
/// Returns `None` if the value is zero (DXGI sets it to 0 when
/// unavailable).
fn qpc_to_mono_nanos(qpc: i64, freq: u64) -> Option<MonoNanos> {
    if qpc <= 0 || freq == 0 {
        return None;
    }
    let mut now_qpc = 0i64;
    let _ = unsafe { QueryPerformanceCounter(&mut now_qpc) };
    let elapsed_ticks = now_qpc.saturating_sub(qpc).max(0) as u128;
    let elapsed_nanos = (elapsed_ticks * 1_000_000_000) / freq as u128;
    let now = MonoNanos::now();
    let kernel_nanos = now.0.saturating_sub(elapsed_nanos as u64);
    Some(MonoNanos(kernel_nanos))
}

fn hresult_io(e: windows::core::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qpc_zero_returns_none() {
        assert!(qpc_to_mono_nanos(0, 10_000_000).is_none());
    }

    #[test]
    fn qpc_negative_returns_none() {
        assert!(qpc_to_mono_nanos(-1, 10_000_000).is_none());
    }

    #[test]
    fn qpc_zero_freq_returns_none() {
        assert!(qpc_to_mono_nanos(100, 0).is_none());
    }

    #[test]
    fn qpc_recent_timestamp_produces_valid_mono_nanos() {
        let freq = qpc_frequency();
        assert!(freq > 0, "QPC frequency should be non-zero on Windows");
        let mut qpc_now = 0i64;
        let _ = unsafe { QueryPerformanceCounter(&mut qpc_now) };
        let result = qpc_to_mono_nanos(qpc_now, freq);
        assert!(result.is_some());
        let mono = result.unwrap();
        let now = MonoNanos::now();
        // The kernel timestamp should be very close to now (within 1ms).
        assert!(
            now.0.saturating_sub(mono.0) < 1_000_000,
            "kernel timestamp should be within 1ms of now, got delta={}ns",
            now.0.saturating_sub(mono.0)
        );
    }

    #[test]
    fn qpc_past_timestamp_produces_earlier_mono_nanos() {
        // Warm up MonoNanos epoch and give enough headroom.
        let _ = MonoNanos::now();
        std::thread::sleep(std::time::Duration::from_millis(50));

        let freq = qpc_frequency();
        // Take a QPC reading, sleep, then take another. The first should
        // produce a MonoNanos earlier than the second.
        let mut qpc_before = 0i64;
        let _ = unsafe { QueryPerformanceCounter(&mut qpc_before) };
        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut qpc_after = 0i64;
        let _ = unsafe { QueryPerformanceCounter(&mut qpc_after) };

        let mono_before = qpc_to_mono_nanos(qpc_before, freq).unwrap();
        let mono_after = qpc_to_mono_nanos(qpc_after, freq).unwrap();
        assert!(
            mono_after.0 > mono_before.0,
            "later QPC should map to later MonoNanos"
        );
        let delta_ns = mono_after.0 - mono_before.0;
        // Should be ~20ms (allow 10-50ms for scheduling).
        assert!(
            delta_ns > 10_000_000 && delta_ns < 50_000_000,
            "expected ~20ms delta between readings, got {}ns",
            delta_ns
        );
    }
}
