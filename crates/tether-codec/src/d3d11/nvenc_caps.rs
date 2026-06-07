//! NVENC AV1 hardware-encode capability gate for Windows NVIDIA hosts.
//!
//! AV1 hardware encode arrived with NVIDIA Ada Lovelace (compute capability
//! 8.9). Ampere (8.0 / 8.6), Turing (7.5), and older NVIDIA GPUs have **no**
//! AV1 encode engine. Asking FFmpeg's `av1_nvenc` to open on such a GPU does
//! not fail cleanly: NVENC reports "Codec not supported", but FFmpeg then
//! faults (`STATUS_ACCESS_VIOLATION`) inside its nvenc failure-cleanup path
//! instead of returning the error. The crash takes down the whole process —
//! the live host encoder on an NVIDIA-primary Ampere box, and the hardware
//! test binary alike.
//!
//! So `av1_nvenc` must be refused *before* FFmpeg init when the GPU can't do
//! AV1. The encoder then falls through to `av1_mf` (Media Foundation AV1,
//! which works on any modern Windows GPU), so a session that negotiated AV1
//! still encodes, and the verified-negative test
//! `d3d11_nvenc_av1_gpu_encode_decode_roundtrip` SKIPs on pre-Ada hardware
//! rather than crashing.
//!
//! We read the GPU's compute capability through the CUDA driver API
//! (`nvcuda.dll`, which ships with every NVIDIA driver alongside NVENC), so
//! the check costs nothing on non-NVIDIA hosts and degrades to "unsupported"
//! when CUDA is absent. The CUDA device is matched to the encoder's exact
//! D3D11 adapter by LUID (`cuDeviceGetLuid` returns the same LUID as
//! `IDXGIAdapter::GetDesc`), so a multi-GPU box — e.g. an Intel iGPU driving
//! the panel plus an NVIDIA eGPU — checks the right GPU.

use std::ffi::{c_int, c_uint, c_void};
use std::sync::OnceLock;

use libloading::Library;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D11::ID3D11Device;
use windows::Win32::Graphics::Dxgi::{IDXGIAdapter, IDXGIDevice};

const CUDA_SUCCESS: c_int = 0;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

/// `true` if a GPU at compute capability `cc` carries an AV1 hardware encoder.
///
/// This is an explicit allowlist of *verified* capabilities, NOT a floor in
/// either direction — a floor is unsafe both below and above. AV1 NVENC is
/// present on Ada Lovelace (8.9 — RTX 40 / L4 / L40) and consumer Blackwell
/// (12.0 — RTX 50). The capabilities between them belong to datacenter parts
/// with **no NVENC engine at all** — Hopper (9.0, H100) and datacenter
/// Blackwell (10.0, B100/B200) — and NVIDIA has repeatedly shipped encoder-less
/// datacenter parts between consumer generations, so an open `>= 12` upper
/// floor would just as easily admit a future encoder-less major and reproduce
/// the crash. We therefore match only the exact capabilities with a confirmed
/// encoder and fall back to `av1_mf` for everything else (Turing 7.5, Ampere
/// 8.0/8.6, the 9.0–11.x gap, and any not-yet-validated future arch). Erring
/// toward exclusion is safe — a missed `av1_nvenc` just uses `av1_mf`; a false
/// positive crashes the process. Add new entries here as they are verified.
fn compute_capability_has_av1_nvenc(cc: (i32, i32)) -> bool {
    matches!(cc, (8, 9) | (12, 0))
}

// CUDA driver API. CUDA declares its entry points `__cdecl`; on Win64 every
// calling convention collapses to the single Microsoft x64 ABI, so
// `extern "system"` is equivalent here (tether-codec is 64-bit only). All
// entry points return `CUresult` (0 == `CUDA_SUCCESS`).
type CuInit = unsafe extern "system" fn(c_uint) -> c_int;
type CuDeviceGetCount = unsafe extern "system" fn(*mut c_int) -> c_int;
type CuDeviceGet = unsafe extern "system" fn(*mut c_int, c_int) -> c_int;
type CuDeviceGetLuid = unsafe extern "system" fn(*mut u8, *mut c_uint, c_int) -> c_int;
type CuDeviceGetAttribute = unsafe extern "system" fn(*mut c_int, c_int, c_int) -> c_int;

struct Cuda {
    // Kept alive so the function pointers below stay valid.
    _lib: Library,
    device_get_count: CuDeviceGetCount,
    device_get: CuDeviceGet,
    device_get_luid: CuDeviceGetLuid,
    device_get_attribute: CuDeviceGetAttribute,
}

impl Cuda {
    /// Load `nvcuda.dll` and `cuInit` once per process. Returns `None` when
    /// CUDA is absent (no NVIDIA driver) or initialization fails — both mean
    /// "can't confirm AV1 support", handled as unsupported by the caller.
    fn get() -> Option<&'static Cuda> {
        static CUDA: OnceLock<Option<Cuda>> = OnceLock::new();
        CUDA.get_or_init(|| unsafe {
            let lib = Library::new("nvcuda.dll").ok()?;
            let cu_init = *lib.get::<CuInit>(b"cuInit\0").ok()?;
            let device_get_count = *lib.get::<CuDeviceGetCount>(b"cuDeviceGetCount\0").ok()?;
            let device_get = *lib.get::<CuDeviceGet>(b"cuDeviceGet\0").ok()?;
            let device_get_luid = *lib.get::<CuDeviceGetLuid>(b"cuDeviceGetLuid\0").ok()?;
            let device_get_attribute = *lib
                .get::<CuDeviceGetAttribute>(b"cuDeviceGetAttribute\0")
                .ok()?;
            // `cuInit` must succeed before any other driver-API call.
            if cu_init(0) != CUDA_SUCCESS {
                return None;
            }
            Some(Cuda {
                _lib: lib,
                device_get_count,
                device_get,
                device_get_luid,
                device_get_attribute,
            })
        })
        .as_ref()
    }

    /// Compute capability `(major, minor)` of the CUDA device whose LUID
    /// matches `target`, or `None` if no CUDA device matches.
    fn compute_capability_for_luid(&self, target: [u8; 8]) -> Option<(i32, i32)> {
        unsafe {
            let mut count: c_int = 0;
            if (self.device_get_count)(&mut count) != CUDA_SUCCESS || count <= 0 {
                return None;
            }
            for ordinal in 0..count {
                let mut dev: c_int = 0;
                if (self.device_get)(&mut dev, ordinal) != CUDA_SUCCESS {
                    continue;
                }
                let mut luid = [0u8; 8];
                let mut node_mask: c_uint = 0;
                if (self.device_get_luid)(luid.as_mut_ptr(), &mut node_mask, dev) != CUDA_SUCCESS {
                    continue;
                }
                if luid != target {
                    continue;
                }
                let mut major: c_int = 0;
                let mut minor: c_int = 0;
                if (self.device_get_attribute)(
                    &mut major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    dev,
                ) != CUDA_SUCCESS
                    || (self.device_get_attribute)(
                        &mut minor,
                        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        dev,
                    ) != CUDA_SUCCESS
                {
                    return None;
                }
                return Some((major, minor));
            }
            None
        }
    }
}

/// 8-byte adapter LUID of the `ID3D11Device` at `device_ptr`, in the same
/// byte layout `cuDeviceGetLuid` reports (`LowPart` then `HighPart`).
fn adapter_luid(device_ptr: *mut c_void) -> Option<[u8; 8]> {
    if device_ptr.is_null() {
        return None;
    }
    unsafe {
        // Borrow without taking ownership — we don't Release this device.
        let device = ID3D11Device::from_raw_borrowed(&device_ptr)?;
        let dxgi: IDXGIDevice = device.cast().ok()?;
        let adapter: IDXGIAdapter = dxgi.GetAdapter().ok()?;
        let desc = adapter.GetDesc().ok()?;
        let mut bytes = [0u8; 8];
        bytes[..4].copy_from_slice(&desc.AdapterLuid.LowPart.to_ne_bytes());
        bytes[4..].copy_from_slice(&desc.AdapterLuid.HighPart.to_ne_bytes());
        Some(bytes)
    }
}

/// `true` if the NVIDIA GPU backing `device_ptr` (an `ID3D11Device`) has an
/// AV1 hardware encoder. Returns `false` — i.e. "fall back to `av1_mf`" — when
/// the device is older than Ada, when CUDA is unavailable, or when the adapter
/// can't be matched, so we never hand `av1_nvenc` a GPU it would crash on.
pub(crate) fn nvidia_gpu_supports_av1_encode(device_ptr: *mut c_void) -> bool {
    let Some(luid) = adapter_luid(device_ptr) else {
        return false;
    };
    let Some(cuda) = Cuda::get() else {
        return false;
    };
    cuda.compute_capability_for_luid(luid)
        .is_some_and(compute_capability_has_av1_nvenc)
}

#[cfg(test)]
mod tests {
    use super::compute_capability_has_av1_nvenc;

    /// Pin the AV1-NVENC-by-compute-capability mapping to real NVIDIA parts.
    /// The load-bearing cases are the verified negatives just above the Ada
    /// floor: Hopper (9.0) and datacenter Blackwell (10.0) have no NVENC engine
    /// at all, so a naive `>= 8.9` floor would admit them and crash. If this
    /// test ever fails because the predicate was simplified to a floor, that
    /// simplification is the bug.
    #[test]
    fn av1_nvenc_capability_matches_known_architectures() {
        // Have an AV1 encoder.
        assert!(
            compute_capability_has_av1_nvenc((8, 9)),
            "Ada Lovelace (RTX 40)"
        );
        assert!(
            compute_capability_has_av1_nvenc((12, 0)),
            "Blackwell consumer (RTX 50)"
        );

        // No AV1 encoder (older consumer / no encoder at all).
        assert!(!compute_capability_has_av1_nvenc((7, 5)), "Turing (RTX 20)");
        assert!(!compute_capability_has_av1_nvenc((8, 0)), "Ampere A100");
        assert!(!compute_capability_has_av1_nvenc((8, 6)), "Ampere (RTX 30)");

        // Above the Ada floor but NO NVENC engine — the floor trap.
        assert!(
            !compute_capability_has_av1_nvenc((9, 0)),
            "Hopper H100 (no NVENC)"
        );
        assert!(
            !compute_capability_has_av1_nvenc((10, 0)),
            "datacenter Blackwell B100/B200 (no NVENC)"
        );

        // Unverified capabilities above the highest known entry must be
        // excluded, not admitted by an open upper floor — a future
        // encoder-less major would otherwise crash on av1_nvenc.
        assert!(
            !compute_capability_has_av1_nvenc((12, 1)),
            "unverified Blackwell minor"
        );
        assert!(
            !compute_capability_has_av1_nvenc((13, 0)),
            "unverified future arch"
        );
    }
}
