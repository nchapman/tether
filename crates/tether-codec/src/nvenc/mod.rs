//! NVENC hardware video encoder (Linux / NVIDIA).
//!
//! A native NVENC backend parallel to [`crate::vaapi`], selected first on
//! NVIDIA hosts with VAAPI as the universal fallback (see
//! `crate::probe::build_encoder`). NVENC exists because FFmpeg's `*_vaapi`
//! wrappers build the rate-control / picture-param misc buffers once at
//! `init()` and never re-read them, so live bitrate retune, qmin, and
//! intra-refresh are silent no-ops on VAAPI (including NVIDIA's
//! nvidia-vaapi-driver shim, which still routes through that wrapper). The
//! `*_nvenc` wrappers expose real per-frame plumbing — that's the whole point.
//!
//! Pipeline: capture DMA-BUF → import into CUDA zero-copy via EGLImage
//! interop (`eglCreateImage(EGL_LINUX_DMA_BUF_EXT)` →
//! `cuGraphicsEGLRegisterImage`, see [`encoder::NvencEncoder::submit_dmabuf`]
//! and [`ffi`]) → `*_nvenc`. There is no `av_hwframe_map(DRM_PRIME → CUDA)`
//! path in FFmpeg (that helper is registered only for VAAPI/Vulkan
//! importers), and `cuImportExternalMemory(OPAQUE_FD)` does **not** accept a
//! Vulkan-exported dma-buf (it isn't a CUDA opaque-fd handle); the EGLImage
//! path is what Sunshine uses. The import runs against the same `CUcontext`
//! FFmpeg's `AVCUDADeviceContext` holds, so the device pointers it produces
//! are dereferenceable by the encoder.
//!
//! Requires an FFmpeg built with `--enable-nvenc --enable-cuda` and the runtime
//! `libnvidia-encode.so` / `libcuda.so` present. Construction returns a clean
//! `CodecError` when either is missing, so a non-NVIDIA host degrades to VAAPI.

pub use encoder::NvencEncoder;

mod encoder;
// `pub(crate)` so the NVDEC decoder (`crate::nvdec`) can reuse the same
// EGL→CUDA importer for its reverse (CUDA → dma-buf) copy — the import
// machinery is direction-agnostic, so there's one home for it.
pub(crate) mod ffi;

#[cfg(test)]
mod tests;

/// CUDA frames pool size. NVENC with `delay=0` drains synchronously
/// (one input surface in flight at a time), so a small pool is plenty;
/// 8 matches [`crate::vaapi::VAAPI_POOL_SIZE`] for a like-for-like
/// surface budget across the two Linux backends.
pub(crate) const NVENC_POOL_SIZE: i32 = 8;

/// NVIDIA PCI vendor id, as reported by `/sys/class/drm/renderD*/device/vendor`
/// and DXGI. Same constant the Windows D3D11 path keys on.
const VENDOR_NVIDIA: &str = "0x10de";

/// Whether an NVIDIA GPU is present on this host.
///
/// The single source of truth consulted by **both** `build_encoder`
/// (live encoder dispatch) and the host probe, so the two never disagree
/// about which backend a profile will be served by. Detection is pure
/// sysfs — it does not load libcuda or create a CUDA context — so it's
/// cheap enough to call on every session setup and safe on a box with no
/// NVIDIA driver at all.
///
/// Scans the DRM render nodes for a device whose PCI vendor is NVIDIA.
/// This mirrors the Windows path's DXGI `VendorId` check (NVIDIA `0x10de`)
/// rather than probing `/dev/nvidia*`, so it reports the GPU that actually
/// backs a render node (the one wgpu/VAAPI will use), not merely that the
/// kernel module is loaded.
#[must_use]
pub fn nvidia_gpu_present() -> bool {
    nvidia_gpu_present_in(std::path::Path::new("/sys/class/drm"))
}

/// [`nvidia_gpu_present`] with an injectable DRM sysfs root, so the
/// detection logic is unit-testable against a synthetic tree without
/// real hardware.
pub(crate) fn nvidia_gpu_present_in(drm_root: &std::path::Path) -> bool {
    let Ok(entries) = std::fs::read_dir(drm_root) else {
        return false;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        // Render nodes only (renderD128, renderD129, …). card* nodes
        // point at the same devices but the render node is what the
        // GPU-compute / VAAPI path opens, so keying on it matches what
        // the encoder will actually use.
        if !name.starts_with("renderD") {
            continue;
        }
        let vendor_path = entry.path().join("device/vendor");
        let Ok(vendor) = std::fs::read_to_string(&vendor_path) else {
            continue;
        };
        if vendor.trim().eq_ignore_ascii_case(VENDOR_NVIDIA) {
            return true;
        }
    }
    false
}
