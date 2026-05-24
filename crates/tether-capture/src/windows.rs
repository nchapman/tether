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
            );
        })
        .map_err(|e| CaptureError::Io(e))?;

    let handle = CaptureHandle::from_parts(rx, target_fps);
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
        // Phase 2: D3D11_BIND_SHADER_RESOURCE when used as VP input
        // for the BGRA→NV12 blit. No bind flags needed for the
        // current CopyResource-only path.
        BindFlags: 0,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut texture = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture)) }?;
    Ok(texture.unwrap())
}

fn run_capture_thread(
    tx: Sender<CapturedFrame>,
    device: D3D11Device,
    duplication: IDXGIOutputDuplication,
    width: u32,
    height: u32,
    target_fps: Arc<AtomicU32>,
) {
    let mut pool: Vec<ID3D11Texture2D> = Vec::with_capacity(TEXTURE_POOL_SIZE);
    for _ in 0..TEXTURE_POOL_SIZE {
        match create_pool_texture(&device.device, width, height) {
            Ok(tex) => pool.push(tex),
            Err(e) => {
                tracing::error!(error = %e, "failed to create texture pool");
                return;
            }
        }
    }
    let mut pool_idx = 0usize;

    let mut frame_info: DXGI_OUTDUPL_FRAME_INFO = unsafe { std::mem::zeroed() };

    loop {
        let fps = target_fps.load(Ordering::Relaxed).max(1);
        let frame_interval = Duration::from_nanos(1_000_000_000 / u64::from(fps));

        // Acquire next frame with a timeout matching the frame interval.
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
                // ACCESS_LOST is recoverable (mode change, monitor hotplug,
                // fast-user-switch). The correct response is to call
                // DuplicateOutput again. For now we exit the thread — the
                // disconnected channel signals the session to tear down.
                // TODO: reconnect loop (re-enumerate outputs, re-duplicate).
                tracing::error!(
                    "DXGI access lost (display mode change or driver reset); \
                     capture thread exiting — session will stall until reconnect is implemented"
                );
                break;
            }
            Err(e) => {
                tracing::error!(error = %e, "AcquireNextFrame failed");
                break;
            }
        };

        let t_userspace = MonoNanos::now();

        // QueryInterface the resource to ID3D11Texture2D.
        let src_texture: ID3D11Texture2D = match resource.cast() {
            Ok(t) => t,
            Err(e) => {
                tracing::error!(error = %e, "failed to cast IDXGIResource to ID3D11Texture2D");
                let _ = unsafe { duplication.ReleaseFrame() };
                break;
            }
        };

        // Copy into a pool texture so we can release the frame immediately.
        let dst_texture = &pool[pool_idx % TEXTURE_POOL_SIZE];
        unsafe {
            device.context.CopyResource(dst_texture, &src_texture);
        }

        // Release the DXGI frame as soon as the copy is submitted.
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
            t_capture_kernel: t_userspace,
            t_capture_userspace: t_userspace,
            release_guard: GpuCapturedGuard::new(()),
            native_damage: None,
        });

        match tx.try_send(frame) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                // Drop oldest — consumer is behind. This matches the
                // bounded-channel back-pressure strategy on Linux/macOS.
            }
            Err(TrySendError::Disconnected(_)) => {
                tracing::debug!("capture receiver dropped; shutting down");
                break;
            }
        }

        // AcquireNextFrame's timeout_ms provides the frame-rate pacing.
        // No additional sleep — the next call blocks until a new frame
        // arrives or the timeout expires.
    }
}

fn hresult_io(e: windows::core::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
}
