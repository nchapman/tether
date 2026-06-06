//! Process-global physical-GPU pin for the NVIDIA path (GitHub issue #16).
//!
//! On a multi-GPU host every NVIDIA subsystem must land on one physical GPU:
//! NVENC's and NVDEC's CUDA contexts, the EGL→CUDA dma-buf importer, and the
//! NVDEC surface pool's Vulkan device. The dma-buf *producer* (a gpuconvert
//! bridge) leads — the host reads the GPU its `HighPerformance` wgpu adapter
//! will pick (`tether_gpuconvert::gpu_select::preferred_device_uuid`) and calls
//! [`pin_gpu_uuid`] once at startup. Everything built afterward (the capability
//! probe and the live session alike) reads this pin and binds to the same GPU.
//!
//! Why a process-global rather than a threaded parameter: the EGL importer is
//! itself a lazily-initialised process-global (one `EGLDisplay` per process),
//! so its device choice *must* come from shared state; routing the same pin
//! through the encoder, decoder, and surface pool keeps all four in agreement
//! from one source of truth. Unset (the default) means "driver-default device"
//! — exactly the single-GPU behavior that shipped before pinning, so a host
//! that never calls [`pin_gpu_uuid`] is unchanged.

use std::ffi::CString;
use std::sync::OnceLock;

use super::ffi::{cuda_ordinal_for_uuid, GpuUuid};

/// The pinned target GPU UUID. `None` until the host pins one; the first pin
/// wins (a host pins exactly once at startup).
static TARGET_UUID: OnceLock<GpuUuid> = OnceLock::new();

/// Pin every NVIDIA subsystem to the physical GPU with this 16-byte device
/// UUID (the Vulkan `deviceUUID` of the dma-buf producer, which equals the
/// CUDA UUID on NVIDIA). Call once at host startup, before constructing any
/// encoder/decoder or running the capability probe, so they all bind here.
///
/// Idempotent in practice: the first pin wins. A second call with a *different*
/// UUID is a host bug (two GPUs chosen) — it's logged and ignored rather than
/// silently re-pointed, since the EGL importer may already be bound.
pub fn pin_gpu_uuid(uuid: GpuUuid) {
    match TARGET_UUID.set(uuid) {
        Ok(()) => {
            tracing::info!(uuid = ?uuid, "pinned NVIDIA subsystems to GPU");
        }
        Err(_) => {
            let existing = TARGET_UUID.get().copied();
            if existing != Some(uuid) {
                tracing::warn!(
                    requested = ?uuid,
                    existing = ?existing,
                    "GPU already pinned to a different UUID; ignoring re-pin"
                );
            }
        }
    }
}

/// The pinned target GPU UUID, or `None` when unpinned (driver default).
pub(crate) fn pinned_uuid() -> Option<GpuUuid> {
    TARGET_UUID.get().copied()
}

/// The CUDA device ordinal the pinned UUID maps to, or `None` when unpinned
/// (use the driver default) or the UUID has no live CUDA device. Used by the
/// EGL importer to pick the matching `EGL_CUDA_DEVICE_NV` display.
pub(crate) fn pinned_cuda_ordinal() -> Option<i32> {
    cuda_ordinal_for_uuid(pinned_uuid()?)
}

/// The CUDA device string for [`rsmpeg::avutil::AVHWDeviceContext::create`]
/// (FFmpeg takes the device as a decimal ordinal), or `None` to let FFmpeg use
/// its default device. `None` whenever [`pinned_cuda_ordinal`] is `None`.
pub(crate) fn cuda_device_cstring() -> Option<CString> {
    // ordinal.to_string() is decimal digits — never contains an interior NUL,
    // so CString::new can't fail; map to None defensively all the same.
    CString::new(pinned_cuda_ordinal()?.to_string()).ok()
}
