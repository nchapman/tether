//! Raw libva symbols and structs. Keep this file tiny — anything that
//! isn't a direct C-ABI declaration belongs in `lib.rs` (safe wrappers).

use std::ffi::c_void;
use std::os::raw::{c_char, c_int};

pub type VADisplay = *mut c_void;
pub type VAProfile = c_int;
pub type VAEntrypoint = c_int;
pub type VAStatus = c_int;
pub type VAGenericID = u32;
pub type VASurfaceID = VAGenericID;

// From <va/va_drmcommon.h>. Names match upstream so a grep against the
// header (or Sunshine's vaapi.cpp) lands on the same identifier.
pub const VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2: u32 = 0x4000_0000;
pub const VA_EXPORT_SURFACE_READ_ONLY: u32 = 0x0001;
pub const VA_EXPORT_SURFACE_WRITE_ONLY: u32 = 0x0002;
pub const VA_EXPORT_SURFACE_SEPARATE_LAYERS: u32 = 0x0004;

// From <va/va.h>. Values are ABI-stable enum discriminants in libva 2.x.
pub const VA_PROFILE_H264_MAIN: VAProfile = 6;
pub const VA_PROFILE_HEVC_MAIN: VAProfile = 17;
pub const VA_PROFILE_HEVC_MAIN10: VAProfile = 18;
pub const VA_PROFILE_HEVC_MAIN444: VAProfile = 26;
pub const VA_PROFILE_HEVC_MAIN444_10: VAProfile = 27;
pub const VA_PROFILE_AV1_PROFILE0: VAProfile = 32;

pub const VA_ENTRYPOINT_ENC_SLICE: VAEntrypoint = 6;
pub const VA_ENTRYPOINT_ENC_SLICE_LP: VAEntrypoint = 8;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PrimeObject {
    pub fd: c_int,
    pub size: u32,
    pub drm_format_modifier: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PrimeLayer {
    pub drm_format: u32,
    pub num_planes: u32,
    pub object_index: [u32; 4],
    pub offset: [u32; 4],
    pub pitch: [u32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DRMPRIMESurfaceDescriptor {
    pub fourcc: u32,
    pub width: u32,
    pub height: u32,
    pub num_objects: u32,
    pub objects: [PrimeObject; 4],
    pub num_layers: u32,
    pub layers: [PrimeLayer; 4],
}

impl DRMPRIMESurfaceDescriptor {
    pub fn zeroed() -> Self {
        // SAFETY: all fields are POD (integers / fixed arrays of integers).
        unsafe { std::mem::zeroed() }
    }
}

extern "C" {
    pub fn vaMaxNumProfiles(dpy: VADisplay) -> c_int;

    pub fn vaMaxNumEntrypoints(dpy: VADisplay) -> c_int;

    pub fn vaQueryConfigProfiles(
        dpy: VADisplay,
        profile_list: *mut VAProfile,
        num_profiles: *mut c_int,
    ) -> VAStatus;

    pub fn vaQueryConfigEntrypoints(
        dpy: VADisplay,
        profile: VAProfile,
        entrypoint_list: *mut VAEntrypoint,
        num_entrypoints: *mut c_int,
    ) -> VAStatus;

    pub fn vaExportSurfaceHandle(
        dpy: VADisplay,
        surface_id: VASurfaceID,
        mem_type: u32,
        flags: u32,
        descriptor: *mut DRMPRIMESurfaceDescriptor,
    ) -> VAStatus;

    pub fn vaErrorStr(status: VAStatus) -> *const c_char;

    /// Block until all pending operations on `surface_id` complete.
    /// Used before `vaExportSurfaceHandle` so the consumer sees a
    /// fully-decoded surface even on driver/kernel combinations
    /// that don't reliably attach the decoder's write fence to the
    /// dma-buf reservation object.
    pub fn vaSyncSurface(dpy: VADisplay, surface_id: VASurfaceID) -> VAStatus;
}

// Layout assertions: pin both sizes AND field offsets so a libva struct
// change that happens to fit existing padding can't sneak past us.
// Numbers come from <va/va_drmcommon.h> on libva 2.x.
const _: () = {
    // PrimeObject: int + u32 + u64 with 8-byte alignment.
    assert!(std::mem::size_of::<PrimeObject>() == 16);
    assert!(std::mem::offset_of!(PrimeObject, fd) == 0);
    assert!(std::mem::offset_of!(PrimeObject, size) == 4);
    assert!(std::mem::offset_of!(PrimeObject, drm_format_modifier) == 8);

    // PrimeLayer: all u32, no padding.
    assert!(std::mem::size_of::<PrimeLayer>() == 56);
    assert!(std::mem::offset_of!(PrimeLayer, drm_format) == 0);
    assert!(std::mem::offset_of!(PrimeLayer, num_planes) == 4);
    assert!(std::mem::offset_of!(PrimeLayer, object_index) == 8);
    assert!(std::mem::offset_of!(PrimeLayer, offset) == 24);
    assert!(std::mem::offset_of!(PrimeLayer, pitch) == 40);

    // Descriptor: 16B header + 64B objects + 4B num_layers + 224B layers
    // + 4B trailing pad to the struct's 8-byte alignment = 312.
    assert!(std::mem::size_of::<DRMPRIMESurfaceDescriptor>() == 312);
    assert!(std::mem::offset_of!(DRMPRIMESurfaceDescriptor, fourcc) == 0);
    assert!(std::mem::offset_of!(DRMPRIMESurfaceDescriptor, objects) == 16);
    assert!(std::mem::offset_of!(DRMPRIMESurfaceDescriptor, num_layers) == 80);
    assert!(std::mem::offset_of!(DRMPRIMESurfaceDescriptor, layers) == 84);
};
