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
const VA_STATUS_INVALID_DESCRIPTOR: ffi::VAStatus = -1;

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
        Self {
            status,
            message: msg,
        }
    }

    fn invalid_descriptor(message: impl Into<String>) -> Self {
        Self {
            status: VA_STATUS_INVALID_DESCRIPTOR,
            message: message.into(),
        }
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
        (
            self.fourcc,
            self.width,
            self.height,
            self.objects,
            self.layers,
        )
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
    let status =
        unsafe { ffi::vaExportSurfaceHandle(display, surface, mem_type, flags, &mut desc) };
    if status != VA_STATUS_SUCCESS {
        return Err(VaError::from_status(status));
    }

    drm_prime_surface_from_descriptor(desc)
}

fn drm_prime_surface_from_descriptor(
    desc: ffi::DRMPRIMESurfaceDescriptor,
) -> Result<DrmPrimeSurface, VaError> {
    let object_count = usize::try_from(desc.num_objects)
        .map_err(|_| VaError::invalid_descriptor("object count does not fit usize"))?;
    let raw_objects = desc.objects.get(..object_count).ok_or_else(|| {
        VaError::invalid_descriptor(format!(
            "object count {} exceeds descriptor capacity {}",
            desc.num_objects,
            desc.objects.len()
        ))
    })?;

    let mut objects = Vec::with_capacity(object_count);
    for raw in raw_objects {
        if raw.fd < 0 {
            return Err(VaError::invalid_descriptor(format!(
                "object fd must be non-negative, got {}",
                raw.fd
            )));
        }
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

    let layer_count = usize::try_from(desc.num_layers)
        .map_err(|_| VaError::invalid_descriptor("layer count does not fit usize"))?;
    let raw_layers = desc.layers.get(..layer_count).ok_or_else(|| {
        VaError::invalid_descriptor(format!(
            "layer count {} exceeds descriptor capacity {}",
            desc.num_layers,
            desc.layers.len()
        ))
    })?;

    let layers = raw_layers.iter().copied().map(PrimeLayer::from).collect();

    Ok(DrmPrimeSurface {
        fourcc: desc.fourcc,
        width: desc.width,
        height: desc.height,
        objects,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::os::fd::IntoRawFd;

    fn descriptor_with_fd(fd: std::os::raw::c_int) -> ffi::DRMPRIMESurfaceDescriptor {
        let mut desc = ffi::DRMPRIMESurfaceDescriptor::zeroed();
        desc.fourcc = u32::from_le_bytes(*b"NV12");
        desc.width = 1920;
        desc.height = 1080;
        desc.num_objects = 1;
        desc.objects[0] = ffi::PrimeObject {
            fd,
            size: 4096,
            drm_format_modifier: 0,
        };
        desc.num_layers = 1;
        desc.layers[0] = ffi::PrimeLayer {
            drm_format: u32::from_le_bytes(*b"NV12"),
            num_planes: 2,
            object_index: [0, 0, 0, 0],
            offset: [0, 2048, 0, 0],
            pitch: [1920, 1920, 0, 0],
        };
        desc
    }

    #[test]
    fn descriptor_conversion_takes_fd_ownership() {
        let fd = File::open("/dev/null")
            .expect("/dev/null should open")
            .into_raw_fd();
        let surface = drm_prime_surface_from_descriptor(descriptor_with_fd(fd))
            .expect("valid descriptor should convert");

        assert_eq!(surface.fourcc, u32::from_le_bytes(*b"NV12"));
        assert_eq!(surface.width, 1920);
        assert_eq!(surface.height, 1080);
        assert_eq!(surface.objects.len(), 1);
        assert_eq!(surface.objects[0].size, 4096);
        assert_eq!(surface.layers.len(), 1);
        assert_eq!(surface.layers[0].num_planes, 2);
    }

    #[test]
    fn descriptor_conversion_rejects_too_many_objects() {
        let mut desc = ffi::DRMPRIMESurfaceDescriptor::zeroed();
        desc.num_objects = 5;

        let err = drm_prime_surface_from_descriptor(desc).expect_err("count should be rejected");

        assert!(err.message.contains("object count 5 exceeds"));
    }

    #[test]
    fn descriptor_conversion_rejects_too_many_layers() {
        let mut desc = ffi::DRMPRIMESurfaceDescriptor::zeroed();
        desc.num_layers = 5;

        let err = drm_prime_surface_from_descriptor(desc).expect_err("count should be rejected");

        assert!(err.message.contains("layer count 5 exceeds"));
    }

    #[test]
    fn descriptor_conversion_closes_fd_when_later_validation_fails() {
        let fd = File::open("/dev/null")
            .expect("/dev/null should open")
            .into_raw_fd();
        let mut desc = descriptor_with_fd(fd);
        desc.num_layers = 5;

        let err = drm_prime_surface_from_descriptor(desc).expect_err("count should be rejected");

        assert!(err.message.contains("layer count 5 exceeds"));
        assert!(
            std::fs::read_link(format!("/proc/self/fd/{fd}")).is_err(),
            "descriptor conversion should close owned fd on later validation failure"
        );
    }

    #[test]
    fn descriptor_conversion_rejects_negative_fd() {
        let err = drm_prime_surface_from_descriptor(descriptor_with_fd(-1))
            .expect_err("negative fd should be rejected");

        assert!(err.message.contains("object fd must be non-negative"));
    }
}
