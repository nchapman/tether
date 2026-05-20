//! CoreVideo / CoreFoundation FFI for the VideoToolbox encoder's
//! IOSurface input path.
//!
//! Scope is deliberately narrow: wrapping a raw `IOSurfaceRef` in a
//! fresh `CVPixelBufferRef` (so FFmpeg's `videotoolbox_enc` reads from
//! it via `frame->data[3]`), plus `CFRetain`/`CFRelease` for managing
//! the buffer's lifetime against an `AVBufferRef` free callback.
//!
//! Hand-rolled FFI rather than pulling in `apple-cf` / `core-video`
//! matches the `tether-vaapi` convention — codec-side platform interop
//! lives in this crate, capture-side in `tether-capture`.

use std::ffi::c_void;

/// Apple's `CVReturn` — 32-bit signed status code. Zero on success.
#[allow(non_camel_case_types)]
pub type CVReturn = i32;
pub const K_CV_RETURN_SUCCESS: CVReturn = 0;

/// Opaque pointer types. `CVPixelBufferRef` and `IOSurfaceRef` are both
/// `*mut c_void` at the Objective-C runtime layer; this module treats
/// them as such (no wrappers — call sites handle lifetime explicitly
/// against the encoder's `AVBufferRef`).
pub type IOSurfaceRef = *mut c_void;
pub type CVPixelBufferRef = *mut c_void;
pub type CFAllocatorRef = *const c_void;
pub type CFDictionaryRef = *const c_void;
pub type CFTypeRef = *const c_void;

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    /// Wraps an `IOSurfaceRef` in a fresh `CVPixelBufferRef`. Returns a
    /// +1 retained `CVPixelBufferRef` on success; the caller is
    /// responsible for `CFRelease` (or stashing in an `AVBufferRef`
    /// whose free callback does it).
    pub fn CVPixelBufferCreateWithIOSurface(
        allocator: CFAllocatorRef,
        surface: IOSurfaceRef,
        pixel_buffer_attributes: CFDictionaryRef,
        pixel_buffer_out: *mut CVPixelBufferRef,
    ) -> CVReturn;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    pub fn CFRelease(cf: CFTypeRef);
}
