//! Minimal libva bindings — only what Tether needs.
//!
//! No upstream Rust crate wraps `vaExportSurfaceHandle`, which is the
//! single VA-API call that lets us hand a decoded surface to wgpu as a
//! DMA-BUF instead of reading it back through `av_hwframe_transfer_data`.
//! Rather than pull in a large auto-generated `libva-sys` and pin around
//! its churn, we declare the five symbols we actually call.
//!
//! Pattern reference: Sunshine's `va_t::set_frame` in
//! `src/platform/linux/vaapi.cpp` — constants, struct definitions, and
//! the call site live in one file there.

// VAAPI is a Linux-only API. The crate is a workspace member, so a
// `cargo build --workspace` on macOS/Windows still compiles it — gate the
// whole crate to Linux so it resolves to an empty rlib elsewhere instead
// of failing on the Unix-only `std::os::fd` import below. `tether-codec`
// only depends on it under `cfg(target_os = "linux")`, so nothing
// references these symbols off-Linux anyway.
#![cfg(target_os = "linux")]
#![allow(non_camel_case_types)]

mod ffi;

use std::ffi::CStr;
use std::os::fd::{FromRawFd, OwnedFd};

pub use ffi::{
    VADisplay, VAGenericID, VASurfaceID, VA_EXPORT_SURFACE_READ_ONLY,
    VA_EXPORT_SURFACE_SEPARATE_LAYERS, VA_EXPORT_SURFACE_WRITE_ONLY,
    VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
};

/// `VA_STATUS_SUCCESS` is 0; anything else is an error code reportable by
/// `vaErrorStr`.
const VA_STATUS_SUCCESS: ffi::VAStatus = 0;

/// libva-reported failure. We carry both the numeric status and the
/// human-readable string from `vaErrorStr` so callers don't have to do the
/// FFI dance themselves to log a useful message.
#[derive(Debug, thiserror::Error)]
#[error("libva call failed: {message} (status = 0x{status:x})")]
pub struct VaError {
    pub status: i32,
    pub message: String,
}

impl VaError {
    fn from_status(status: ffi::VAStatus) -> Self {
        // SAFETY: vaErrorStr returns a static C string for any status,
        // including unknown ones (it falls back to "unknown libva error").
        let msg = unsafe {
            let ptr = ffi::vaErrorStr(status);
            if ptr.is_null() {
                "unknown libva error (null vaErrorStr)".to_string()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Self { status, message: msg }
    }
}

/// One DMA-BUF object backing (part of) a VA surface. The fd is `OwnedFd`
/// so close-exactly-once is enforced by the type system, not docs.
#[derive(Debug)]
pub struct PrimeObject {
    pub fd: OwnedFd,
    pub size: u32,
    pub drm_format_modifier: u64,
}

/// One image-plane layer described in DRM fourcc terms. Indices in
/// `object_index` point into the parent `DrmPrimeSurface::objects` slice.
/// Layout mirrors `va_drmcommon.h::VADRMPRIMESurfaceDescriptor::layers[]`.
#[derive(Debug, Copy, Clone)]
pub struct PrimeLayer {
    pub drm_format: u32,
    pub num_planes: u32,
    pub object_index: [u32; 4],
    pub offset: [u32; 4],
    pub pitch: [u32; 4],
}

impl From<ffi::PrimeLayer> for PrimeLayer {
    fn from(l: ffi::PrimeLayer) -> Self {
        Self {
            drm_format: l.drm_format,
            num_planes: l.num_planes,
            object_index: l.object_index,
            offset: l.offset,
            pitch: l.pitch,
        }
    }
}

/// Owned PRIME descriptor. Each FD lives in an `OwnedFd` and is closed
/// when the surface is dropped; `into_parts()` lets a downstream importer
/// (wgpu/Vulkan) take ownership.
#[derive(Debug)]
pub struct DrmPrimeSurface {
    pub fourcc: u32,
    pub width: u32,
    pub height: u32,
    pub objects: Vec<PrimeObject>,
    pub layers: Vec<PrimeLayer>,
}

impl DrmPrimeSurface {
    /// Decompose into fields so callers can hand fds to a wgpu/Vulkan
    /// import that takes ownership.
    pub fn into_parts(self) -> (u32, u32, u32, Vec<PrimeObject>, Vec<PrimeLayer>) {
        (self.fourcc, self.width, self.height, self.objects, self.layers)
    }
}

/// Block until all in-flight operations on `surface` complete. Cheap
/// when the surface is already done (microseconds); guarantees the
/// next consumer (a wgpu/Vulkan import) sees decode-complete pixels.
///
/// We always call this before [`export_surface_handle`] because dma-buf
/// implicit sync via the reservation object only works when the
/// producer attaches a write fence — Mesa+Intel does, but proprietary
/// stacks (e.g. NVIDIA's libva backends) historically do not, and a
/// race window has been observed on the Intel `iHD` driver around
/// export. The cost of always syncing is negligible compared to the
/// cost of a silently torn frame with no log signal.
///
/// # Safety
/// Same contract as [`export_surface_handle`]: `display` must be live
/// and `surface` must be owned by it.
pub unsafe fn sync_surface(display: VADisplay, surface: VASurfaceID) -> Result<(), VaError> {
    // SAFETY: forwarding caller's invariants on display+surface.
    let status = unsafe { ffi::vaSyncSurface(display, surface) };
    if status == VA_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(VaError::from_status(status))
    }
}

/// Export the given VA surface as a DRM PRIME descriptor.
///
/// `mem_type` is typically [`VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2`].
/// `flags` is a bitset of `VA_EXPORT_SURFACE_*` constants — most callers
/// want `WRITE_ONLY | SEPARATE_LAYERS` to mirror Sunshine.
///
/// The `VADisplay`/`VASurfaceID` pair is taken raw. The decoder rewrite
/// will introduce a borrowed `VaDisplay<'_>` newtype tied to the
/// `AVHWDeviceContext`; until that exists, the safety contract lives in
/// the `unsafe` requirement and the doc below.
///
/// # Safety
/// `display` must be a live `VADisplay` (i.e. the one your decoder is
/// using), and `surface` must be a surface ID currently owned by that
/// display. Both invariants hold trivially when you pass values pulled
/// from the same `AVHWDeviceContext`.
pub unsafe fn export_surface_handle(
    display: VADisplay,
    surface: VASurfaceID,
    mem_type: u32,
    flags: u32,
) -> Result<DrmPrimeSurface, VaError> {
    let mut desc = ffi::DRMPRIMESurfaceDescriptor::zeroed();
    // SAFETY: forwarding caller's invariants on display+surface; `desc`
    // is a valid writable struct of the right type.
    let status = unsafe {
        ffi::vaExportSurfaceHandle(display, surface, mem_type, flags, &mut desc)
    };
    if status != VA_STATUS_SUCCESS {
        return Err(VaError::from_status(status));
    }

    let mut objects = Vec::with_capacity(desc.num_objects as usize);
    for raw in &desc.objects[..desc.num_objects as usize] {
        // SAFETY: vaExportSurfaceHandle returned this fd open and
        // transferred ownership to us. We wrap it immediately so any
        // subsequent panic still closes it.
        let fd = unsafe { OwnedFd::from_raw_fd(raw.fd) };
        objects.push(PrimeObject {
            fd,
            size: raw.size,
            drm_format_modifier: raw.drm_format_modifier,
        });
    }

    let layers = desc.layers[..desc.num_layers as usize]
        .iter()
        .copied()
        .map(PrimeLayer::from)
        .collect();

    Ok(DrmPrimeSurface {
        fourcc: desc.fourcc,
        width: desc.width,
        height: desc.height,
        objects,
        layers,
    })
}
