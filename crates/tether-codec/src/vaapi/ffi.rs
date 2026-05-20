//! Hand-rolled FFI bindings for the libavutil hwcontext_vaapi and
//! hwcontext_drm headers that rusty-ffmpeg's bindgen step skips
//! (it doesn't pull in libva).
//!
//! Field layouts are stable since libavutil 56.x. The size/offset
//! `const _: ()` block at the bottom catches both FFmpeg ABI drift and
//! the LP64-vs-ILP32-vs-LLP64 trap (`ptrdiff_t` / `size_t` change
//! width); numbers are for LP64 (x86_64 + aarch64 Linux). If we ever
//! cross-compile to 32-bit ARM the const eval fails and forces a
//! per-arch fix rather than silently corrupting layout.

/// `AVVAAPIDeviceContext` from `<libavutil/hwcontext_vaapi.h>`. Two
/// fields, stable since libavutil 56.x — if FFmpeg ever extends it,
/// the new fields land *after* `driver_quirks` so our access of
/// `display` stays correct.
#[repr(C)]
pub(super) struct AVVAAPIDeviceContext {
    pub(super) display: tether_vaapi::VADisplay,
    pub(super) driver_quirks: std::os::raw::c_uint,
}

/// DRM PRIME frame descriptors from `<libavutil/hwcontext_drm.h>`.
///
/// `ptrdiff_t` / `size_t` map to `isize` / `usize` — one machine word
/// regardless of arch, matching what the C header expands to. (The
/// signed type for offsets/pitches reflects the kernel API's signature;
/// VAAPI's PRIME_2 import path rejects negative pitches, so we don't
/// actually accept flipped buffers either way.)
pub(super) const AV_DRM_MAX_PLANES: usize = 4;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct AVDRMPlaneDescriptor {
    pub(super) object_index: std::os::raw::c_int,
    pub(super) offset: isize,
    pub(super) pitch: isize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct AVDRMLayerDescriptor {
    pub(super) format: u32,
    pub(super) nb_planes: std::os::raw::c_int,
    pub(super) planes: [AVDRMPlaneDescriptor; AV_DRM_MAX_PLANES],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct AVDRMObjectDescriptor {
    pub(super) fd: std::os::raw::c_int,
    pub(super) size: usize,
    pub(super) format_modifier: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct AVDRMFrameDescriptor {
    pub(super) nb_objects: std::os::raw::c_int,
    pub(super) objects: [AVDRMObjectDescriptor; AV_DRM_MAX_PLANES],
    pub(super) nb_layers: std::os::raw::c_int,
    pub(super) layers: [AVDRMLayerDescriptor; AV_DRM_MAX_PLANES],
}

const _: () = {
    // AVDRMPlaneDescriptor: int + 4 pad + isize + isize = 24.
    assert!(std::mem::size_of::<AVDRMPlaneDescriptor>() == 24);
    assert!(std::mem::offset_of!(AVDRMPlaneDescriptor, object_index) == 0);
    assert!(std::mem::offset_of!(AVDRMPlaneDescriptor, offset) == 8);
    assert!(std::mem::offset_of!(AVDRMPlaneDescriptor, pitch) == 16);

    // AVDRMLayerDescriptor: u32 + int + 4 planes * 24 = 104.
    assert!(std::mem::size_of::<AVDRMLayerDescriptor>() == 104);
    assert!(std::mem::offset_of!(AVDRMLayerDescriptor, format) == 0);
    assert!(std::mem::offset_of!(AVDRMLayerDescriptor, nb_planes) == 4);
    assert!(std::mem::offset_of!(AVDRMLayerDescriptor, planes) == 8);

    // AVDRMObjectDescriptor: int + 4 pad + usize + u64 = 24.
    assert!(std::mem::size_of::<AVDRMObjectDescriptor>() == 24);
    assert!(std::mem::offset_of!(AVDRMObjectDescriptor, fd) == 0);
    assert!(std::mem::offset_of!(AVDRMObjectDescriptor, size) == 8);
    assert!(std::mem::offset_of!(AVDRMObjectDescriptor, format_modifier) == 16);

    // AVDRMFrameDescriptor: int + 4 pad + 4 objects * 24 + int + 4 pad
    // + 4 layers * 104 = 528.
    assert!(std::mem::size_of::<AVDRMFrameDescriptor>() == 528);
    assert!(std::mem::offset_of!(AVDRMFrameDescriptor, nb_objects) == 0);
    assert!(std::mem::offset_of!(AVDRMFrameDescriptor, objects) == 8);
    assert!(std::mem::offset_of!(AVDRMFrameDescriptor, nb_layers) == 104);
    assert!(std::mem::offset_of!(AVDRMFrameDescriptor, layers) == 112);
};
