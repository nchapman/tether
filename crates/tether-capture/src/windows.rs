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
//! Shutdown: dropping the [`crate::FrameReceiver`] drops the consumer
//! liveness token; the capture thread sees its strong count hit zero at
//! the top of the loop and exits. (Channel `Disconnected` is no longer
//! the signal — the thread holds an `evict_rx` clone for drop-oldest
//! eviction, which keeps the channel connected.)

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, Sender};
use tether_protocol::MonoNanos;
use windows::core::Interface;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_UNKNOWN};
use windows::Win32::Graphics::Direct3D11::{
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Multithread,
    ID3D11Texture2D, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
    D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1, IDXGIOutput1,
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

/// Capture→encode mailbox depth. One slot: the handoff is single-frame,
/// drop-oldest (freshest-wins) via [`send_latest`], so the encode loop
/// always dequeues the newest captured frame, never a stale backlog —
/// the property that matters when an encode stall (e.g. shared-iGPU
/// contention) briefly lets the capture thread outrun the consumer.
const CAPTURE_MAILBOX_DEPTH: usize = 1;
const CAPTURE_FPS: u32 = 60;
/// Texture pool size. Three is the minimum that lets the capture thread
/// always own a free slot to write into: at most one frame sits in the
/// mailbox and one is held by the encoder mid-flight, leaving a third
/// free. A slot is reused only once its `release_guard` ([`SlotReturn`])
/// drops — the ownership handshake that prevents overwriting a texture
/// the encoder's Video Processor is still sampling (the cause of the
/// progressive-corruption regression).
const TEXTURE_POOL_SIZE: usize = 3;

/// Shared D3D11 device handle. The capture thread creates it; the
/// encoder receives a clone so both operate on the same device (no
/// cross-device texture copies).
#[derive(Clone)]
pub struct D3D11Device {
    pub device: ID3D11Device,
    pub context: ID3D11DeviceContext,
    /// PCI vendor ID from the DXGI adapter (0x8086 = Intel, 0x1002 = AMD,
    /// 0x10de = NVIDIA). Used by the encoder to select the right backend.
    pub vendor_id: u32,
}

impl D3D11Device {
    /// Raw COM pointers for injecting into FFmpeg's hwctx so the encoder
    /// shares the same D3D11 device as capture (zero-copy VP blit).
    pub fn device_ptrs(&self) -> (*mut std::ffi::c_void, *mut std::ffi::c_void) {
        use windows::core::Interface;
        (
            self.device.as_raw() as *mut std::ffi::c_void,
            self.context.as_raw() as *mut std::ffi::c_void,
        )
    }
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

/// Pre-create the DXGI device and output duplication. Must be called
/// BEFORE any AMF/D3D11VA probe activity (the AMF driver corrupts
/// in-process DXGI output enumeration). The returned value is passed
/// to [`start_with`].
pub fn pre_create() -> Result<PreCreatedCapture> {
    let (device, context, duplication, width, height, vendor_id) =
        create_d3d11_device_and_duplication()?;
    Ok(PreCreatedCapture { device, context, duplication, width, height, vendor_id })
}

/// Opaque handle holding pre-created DXGI resources.
pub struct PreCreatedCapture {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    duplication: IDXGIOutputDuplication,
    width: u32,
    height: u32,
    vendor_id: u32,
}

// SAFETY: same conditions as D3D11Device — multithread-protected COM objects.
unsafe impl Send for PreCreatedCapture {}

/// Start DXGI Desktop Duplication using pre-created resources from [`pre_create`].
pub fn start_with(pre: PreCreatedCapture) -> Result<(CaptureHandle, D3D11Device)> {
    let PreCreatedCapture { device, context, duplication, width, height, vendor_id } = pre;
    let shared_device = D3D11Device {
        device: device.clone(),
        context: context.clone(),
        vendor_id,
    };

    let (tx, rx) = bounded::<CapturedFrame>(CAPTURE_MAILBOX_DEPTH);
    // The capture thread keeps a receiver clone purely to evict the
    // unconsumed mailbox frame (drop-oldest); it never consumes frames.
    // Because that clone masks channel disconnection, shutdown is driven
    // off the consumer-liveness weak handle below, not `Disconnected`.
    let evict_rx = rx.clone();
    let target_fps = Arc::new(AtomicU32::new(CAPTURE_FPS));
    let target_fps_thread = Arc::clone(&target_fps);
    let device_thread = shared_device.clone();

    let (cursor_state, cursor_source) = DxgiCursorState::new();

    // Build the handle first so its liveness token exists, then hand the
    // capture thread a weak ref to it. The handle (or, after `into_rx`,
    // the `FrameReceiver`) keeps the strong count non-zero while a
    // consumer is alive.
    let handle =
        CaptureHandle::from_parts(rx, target_fps).with_cursor_source(Box::new(cursor_source));
    let liveness = handle.liveness();

    std::thread::Builder::new()
        .name("tether-capture-dxgi".into())
        .spawn(move || {
            run_capture_thread(
                tx,
                evict_rx,
                liveness,
                device_thread,
                duplication,
                width,
                height,
                target_fps_thread,
                cursor_state,
            );
        })
        .map_err(|e| CaptureError::Io(e))?;

    Ok((handle, shared_device))
}

/// Returns (device, context, duplication, width, height, vendor_id).
fn create_d3d11_device_and_duplication(
) -> Result<(ID3D11Device, ID3D11DeviceContext, IDXGIOutputDuplication, u32, u32, u32)> {
    // First try: create device via D3D_DRIVER_TYPE_HARDWARE (lets D3D11
    // pick the GPU itself, bypassing DXGI factory enumeration which can
    // become stale after AMF/D3D11VA probe activity in-process).
    let mut device = None;
    let mut context = None;
    let hw_ok = unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
    };

    if let (Ok(()), Some(device), Some(context)) = (hw_ok, device, context) {
        if let Ok(mt) = device.cast::<ID3D11Multithread>() {
            let _ = unsafe { mt.SetMultithreadProtected(true) };
        }

        let dxgi_device: windows::Win32::Graphics::Dxgi::IDXGIDevice =
            device.cast().map_err(|e| CaptureError::Io(hresult_io(e)))?;
        let adapter: IDXGIAdapter1 = unsafe { dxgi_device.GetParent() }
            .map_err(|e| CaptureError::Io(hresult_io(e)))?;
        let vendor_id = unsafe { adapter.GetDesc1() }
            .map(|d| d.VendorId)
            .unwrap_or(0);

        let mut output_idx = 0u32;
        while let Ok(output) = unsafe { adapter.EnumOutputs(output_idx) } {
            let output1: IDXGIOutput1 = match output.cast() {
                Ok(o) => o,
                Err(_) => { output_idx += 1; continue; }
            };
            match unsafe { output1.DuplicateOutput(&device) } {
                Ok(duplication) => {
                    let desc = unsafe { duplication.GetDesc() };
                    let width = desc.ModeDesc.Width;
                    let height = desc.ModeDesc.Height;
                    tracing::info!(
                        output_idx,
                        width,
                        height,
                        vendor_id = format_args!("0x{vendor_id:04x}"),
                        format = ?desc.ModeDesc.Format,
                        "DXGI Desktop Duplication initialized (hardware device)"
                    );
                    return Ok((device, context, duplication, width, height, vendor_id));
                }
                Err(e) => {
                    tracing::warn!(output_idx, error = %e, "DuplicateOutput failed on hardware device");
                }
            }
            output_idx += 1;
        }
        tracing::warn!("hardware device has no duplicable outputs; falling back to factory enumeration");
    }

    // Fallback: enumerate all adapters via factory.
    let factory: IDXGIFactory1 =
        unsafe { CreateDXGIFactory1() }.map_err(|e| CaptureError::Io(hresult_io(e)))?;

    let mut adapter_idx = 0u32;
    while let Ok(adapter) = unsafe { factory.EnumAdapters1(adapter_idx) } {
        let adapter_desc = unsafe { adapter.GetDesc1() }
            .map_err(|e| CaptureError::Io(hresult_io(e)))?;
        let adapter_name = String::from_utf16_lossy(
            &adapter_desc.Description[..adapter_desc.Description.iter().position(|&c| c == 0).unwrap_or(adapter_desc.Description.len())]
        );

        let mut output_idx = 0u32;
        while let Ok(output) = unsafe { adapter.EnumOutputs(output_idx) } {
            let output1: IDXGIOutput1 = match output.cast() {
                Ok(o) => o,
                Err(_) => { output_idx += 1; continue; }
            };

            let mut dev = None;
            let mut ctx = None;
            if unsafe {
                D3D11CreateDevice(
                    &adapter,
                    D3D_DRIVER_TYPE_UNKNOWN,
                    HMODULE::default(),
                    D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
                    None,
                    D3D11_SDK_VERSION,
                    Some(&mut dev),
                    None,
                    Some(&mut ctx),
                )
            }.is_err() {
                output_idx += 1;
                continue;
            }

            let dev = dev.unwrap();
            let ctx = ctx.unwrap();

            match unsafe { output1.DuplicateOutput(&dev) } {
                Ok(duplication) => {
                    if let Ok(mt) = dev.cast::<ID3D11Multithread>() {
                        let _ = unsafe { mt.SetMultithreadProtected(true) };
                    }
                    let desc = unsafe { duplication.GetDesc() };
                    let width = desc.ModeDesc.Width;
                    let height = desc.ModeDesc.Height;
                    let vendor_id = adapter_desc.VendorId;
                    tracing::info!(
                        adapter = adapter_name.as_str(),
                        adapter_idx,
                        output_idx,
                        width,
                        height,
                        vendor_id = format_args!("0x{vendor_id:04x}"),
                        format = ?desc.ModeDesc.Format,
                        "DXGI Desktop Duplication initialized (factory fallback)"
                    );
                    return Ok((dev, ctx, duplication, width, height, vendor_id));
                }
                Err(e) => {
                    tracing::warn!(
                        adapter = adapter_name.as_str(),
                        adapter_idx,
                        output_idx,
                        error = %e,
                        "DuplicateOutput failed; trying next output"
                    );
                }
            }
            output_idx += 1;
        }
        adapter_idx += 1;
    }

    Err(CaptureError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "no DXGI adapter has an active output",
    )))
}

fn create_duplication(
    device: &ID3D11Device,
) -> Result<(IDXGIOutputDuplication, u32, u32)> {
    let dxgi_device: windows::Win32::Graphics::Dxgi::IDXGIDevice =
        device.cast().map_err(|e| CaptureError::Io(hresult_io(e)))?;

    let adapter: IDXGIAdapter1 = unsafe { dxgi_device.GetParent() }
        .map_err(|e| CaptureError::Io(hresult_io(e)))?;

    let mut output_idx = 0u32;
    while let Ok(output) = unsafe { adapter.EnumOutputs(output_idx) } {
        let output1: IDXGIOutput1 = match output.cast() {
            Ok(o) => o,
            Err(_) => { output_idx += 1; continue; }
        };
        match unsafe { output1.DuplicateOutput(device) } {
            Ok(duplication) => {
                let desc = unsafe { duplication.GetDesc() };
                let width = desc.ModeDesc.Width;
                let height = desc.ModeDesc.Height;
                tracing::info!(width, height, "DXGI reconnected on output {output_idx}");
                return Ok((duplication, width, height));
            }
            Err(_) => { output_idx += 1; }
        }
    }
    Err(CaptureError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "no DXGI output available for duplication on this adapter",
    )))
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

/// Returns a texture-pool slot to the capture thread's free-list when
/// the consumer (or an evicted mailbox entry) drops the frame. This is
/// the ownership handshake: a slot is only reused once `SlotReturn`
/// drops, so the capture thread never `CopyResource`s a new frame into a
/// texture the encoder's Video Processor is still sampling. Stashed in
/// [`GpuCapturedFrame::release_guard`].
struct SlotReturn {
    free_tx: Sender<usize>,
    slot: usize,
}

impl Drop for SlotReturn {
    fn drop(&mut self) {
        // try_send never blocks: the free-list is sized to the pool, so
        // it can't be full of in-use slots. A failure means the pool was
        // torn down (capture thread exited), in which case the slot is
        // moot.
        let _ = self.free_tx.try_send(self.slot);
    }
}

/// Single-slot, drop-oldest handoff: evict any unconsumed frame before
/// enqueuing the newest, so the consumer always dequeues the freshest
/// (invariant: at most one frame resident, freshest wins). Dropping the
/// evicted frame runs its [`SlotReturn`], freeing the slot it held.
///
/// INVARIANT: single producer only. The evict-then-send is not atomic;
/// a second producer could fill the mailbox in the gap. The DXGI capture
/// thread is the sole sender, so this holds.
fn send_latest<T>(tx: &Sender<T>, evict: &Receiver<T>, item: T) {
    let _ = evict.try_recv();
    let _ = tx.try_send(item);
}

/// Acquire a free pool slot to write the next capture into. A slot is
/// free only once its previous frame's [`SlotReturn`] dropped, so the
/// caller never overwrites a texture the encoder is still sampling. If
/// the free-list is momentarily empty, evict the unconsumed mailbox
/// frame (freshest-wins) to reclaim its slot and retry. Returns `None`
/// only when every slot is held downstream (a consumer stall) — the
/// caller then drops the capture rather than overwrite an in-use texture.
fn acquire_slot<T>(free_rx: &Receiver<usize>, evict_rx: &Receiver<T>) -> Option<usize> {
    if let Ok(slot) = free_rx.try_recv() {
        return Some(slot);
    }
    let _ = evict_rx.try_recv();
    free_rx.try_recv().ok()
}

fn run_capture_thread(
    tx: Sender<CapturedFrame>,
    evict_rx: Receiver<CapturedFrame>,
    liveness: Weak<()>,
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
    // Free-list of pool slot indices, prefilled with every slot. The
    // capture thread acquires a slot to write into; a slot returns only
    // when its frame's `release_guard` ([`SlotReturn`]) drops.
    let (free_tx, free_rx) = bounded::<usize>(TEXTURE_POOL_SIZE);
    for slot in 0..TEXTURE_POOL_SIZE {
        let _ = free_tx.try_send(slot);
    }

    let qpc_freq = qpc_frequency();

    let mut frame_info: DXGI_OUTDUPL_FRAME_INFO = unsafe { std::mem::zeroed() };

    loop {
        // Shut down when the consumer has dropped its `FrameReceiver`.
        // Our `evict_rx` clone keeps the channel connected, so we can't
        // rely on `Disconnected`; the liveness token's strong count
        // hitting zero is the signal. The final `Arc` drop decrements with
        // release ordering and `Weak::strong_count` loads with SeqCst, so
        // a zero observed here is globally ordered after the consumer's
        // drop — no fence needed. Worst-case detection latency is one
        // `AcquireNextFrame` timeout (checked once per loop iteration).
        if liveness.strong_count() == 0 {
            tracing::debug!("capture consumer dropped; shutting down");
            break;
        }
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
                            // Drop any frame still queued from the OLD pool
                            // before swapping pools, so the encoder never
                            // dequeues an old-resolution texture against the
                            // new-resolution pool. The dropped frame's
                            // `SlotReturn` frees its slot immediately. (A
                            // frame already inside the encoder is caught by
                            // the VP input-dimension guard, which triggers an
                            // encoder rebuild.)
                            let _ = evict_rx.try_recv();
                            match create_texture_pool(&device.device, width, height) {
                                Some(p) => pool = p,
                                None => break,
                            }
                            // Free-list indices stay valid across the new
                            // pool (same count); outstanding guards return
                            // indices that now map to the new textures.
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

        // Acquire a free pool slot to write into (see [`acquire_slot`]).
        // `None` means every slot is in flight (a mid-encode stall); drop
        // this capture rather than overwrite a texture the encoder is
        // still sampling — the ownership handshake that fixes the
        // progressive-corruption regression.
        let slot = match acquire_slot(&free_rx, &evict_rx) {
            Some(s) => s,
            None => {
                let _ = unsafe { duplication.ReleaseFrame() };
                continue;
            }
        };

        let dst_texture = &pool[slot];
        unsafe {
            device.context.CopyResource(dst_texture, &src_texture);
        }

        let _ = unsafe { duplication.ReleaseFrame() };

        let frame_texture = dst_texture.clone();

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
            release_guard: GpuCapturedGuard::new(SlotReturn {
                free_tx: free_tx.clone(),
                slot,
            }),
            native_damage,
        });

        // Freshest-wins single-slot handoff: evict any unconsumed frame
        // (returning its slot) before enqueuing the newest. Shutdown is
        // detected via the liveness check at the top of the loop, not a
        // send error, since `evict_rx` keeps the channel connected.
        send_latest(&tx, &evict_rx, frame);
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
    fn slot_return_releases_slot_to_free_list_on_drop() {
        // The ownership handshake: a pool slot is reclaimable only after
        // the frame holding it is dropped.
        let (free_tx, free_rx) = bounded::<usize>(1);
        {
            let _guard = SlotReturn {
                free_tx: free_tx.clone(),
                slot: 7,
            };
            assert!(free_rx.try_recv().is_err(), "slot must not return while held");
        }
        assert_eq!(free_rx.try_recv().ok(), Some(7), "slot returns once guard drops");
    }

    #[test]
    fn send_latest_drops_oldest_and_reclaims_its_slot() {
        // Freshest-wins: enqueuing a second frame onto a full depth-1
        // mailbox evicts the first, the consumer dequeues the newest, and
        // the evicted frame's pool slot returns to the free-list.
        let (free_tx, free_rx) = bounded::<usize>(2);
        free_tx.try_send(0).unwrap();
        free_tx.try_send(1).unwrap();

        let (mtx, mrx) = bounded::<(u32, SlotReturn)>(CAPTURE_MAILBOX_DEPTH);
        let evict = mrx.clone();

        // Frame A grabs slot 0 and goes into the mailbox.
        let s0 = free_rx.try_recv().unwrap();
        send_latest(&mtx, &evict, (100, SlotReturn { free_tx: free_tx.clone(), slot: s0 }));

        // Frame B grabs slot 1; sending it evicts (drops) A → slot 0 frees.
        let s1 = free_rx.try_recv().unwrap();
        send_latest(&mtx, &evict, (200, SlotReturn { free_tx: free_tx.clone(), slot: s1 }));

        assert_eq!(free_rx.try_recv().ok(), Some(s0), "evicted frame's slot returns");
        let (val, _held) = mrx.try_recv().expect("mailbox holds the newest frame");
        assert_eq!(val, 200, "consumer dequeues the freshest frame, not the stale one");
    }

    #[test]
    fn acquire_slot_evicts_mailbox_when_free_list_empty() {
        // When the free-list is momentarily empty, acquiring a slot must
        // evict the unconsumed mailbox frame and reclaim the slot it held.
        let (free_tx, free_rx) = bounded::<usize>(2);
        let (mtx, mrx) = bounded::<(u32, SlotReturn)>(CAPTURE_MAILBOX_DEPTH);
        let evict = mrx.clone();

        // Slot 5 is held by the mailbox frame; the free-list is empty.
        mtx.try_send((1, SlotReturn { free_tx: free_tx.clone(), slot: 5 }))
            .unwrap();

        assert_eq!(acquire_slot(&free_rx, &evict), Some(5), "eviction reclaims the slot");
        assert!(mrx.try_recv().is_err(), "the evicted frame is gone from the mailbox");
    }

    #[test]
    fn acquire_slot_returns_none_when_every_slot_is_in_flight() {
        // Free-list empty and mailbox empty: the consumer holds every
        // slot (a stall). Acquire must refuse rather than overwrite.
        let (_free_tx, free_rx) = bounded::<usize>(2);
        let (_mtx, mrx) = bounded::<(u32, SlotReturn)>(CAPTURE_MAILBOX_DEPTH);
        let evict = mrx.clone();

        assert_eq!(acquire_slot(&free_rx, &evict), None);
    }

    #[test]
    fn producer_outrunning_consumer_keeps_freshest_and_leaks_no_slots() {
        // Drive the real acquire+send cadence: a 3-slot pool, the consumer
        // never reads. Each iteration must find a slot (eviction reclaims
        // one), and after ten produces the mailbox holds only the newest
        // frame with every pool slot still accounted for.
        const POOL: usize = TEXTURE_POOL_SIZE;
        let (free_tx, free_rx) = bounded::<usize>(POOL);
        for s in 0..POOL {
            free_tx.try_send(s).unwrap();
        }
        let (mtx, mrx) = bounded::<(u32, SlotReturn)>(CAPTURE_MAILBOX_DEPTH);
        let evict = mrx.clone();

        for id in 0..10u32 {
            let slot = acquire_slot(&free_rx, &evict)
                .expect("eviction keeps a slot available while only the mailbox holds one");
            send_latest(&mtx, &evict, (id, SlotReturn { free_tx: free_tx.clone(), slot }));
        }

        let (newest, guard) = mrx.try_recv().expect("mailbox holds a frame");
        assert_eq!(newest, 9, "consumer would dequeue the freshest frame");
        drop(guard);

        let mut reclaimed = 0;
        while free_rx.try_recv().is_ok() {
            reclaimed += 1;
        }
        assert_eq!(reclaimed, POOL, "every pool slot is reclaimed; none leaked");
    }

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
