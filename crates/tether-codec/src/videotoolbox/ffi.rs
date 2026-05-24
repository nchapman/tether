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
pub type CFStringRef = *const c_void;

/// `CVAttachmentMode` — controls whether the attachment propagates to
/// derived buffers (copies, wrappers). We use `ShouldPropagate` so
/// any downstream consumer (e.g. a later CVPixelBuffer wrapper) still
/// sees the color tags we set.
pub type CVAttachmentMode = u32;
pub const K_CV_ATTACHMENT_MODE_SHOULD_PROPAGATE: CVAttachmentMode = 1;

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

    /// Return the `IOSurfaceRef` backing a `CVPixelBufferRef`, or NULL
    /// if the buffer is not IOSurface-backed. The returned IOSurface's
    /// lifetime is the parent CVPixelBuffer's — no extra retain.
    pub fn CVPixelBufferGetIOSurface(pixel_buffer: CVPixelBufferRef) -> IOSurfaceRef;

    /// `kCVPixelFormatType_*` fourcc (e.g. `'420v'` for NV12 video
    /// range) of the buffer's pixel data.
    pub fn CVPixelBufferGetPixelFormatType(pixel_buffer: CVPixelBufferRef) -> u32;

    pub fn CVPixelBufferGetWidth(pixel_buffer: CVPixelBufferRef) -> usize;
    pub fn CVPixelBufferGetHeight(pixel_buffer: CVPixelBufferRef) -> usize;

    /// Attach a CoreFoundation value to a CVBuffer under the given
    /// key. Used here to pin BT.709 color attachments on the wrapped
    /// CVPixelBuffer so the IOSurface's incoming tags (which on
    /// pre-Sequoia macOS have been observed to default to BT.601 for
    /// small capture regions) don't slip through and mis-match the
    /// encoder's hardcoded VUI.
    pub fn CVBufferSetAttachment(
        buffer: CVPixelBufferRef,
        key: CFStringRef,
        value: CFTypeRef,
        attachment_mode: CVAttachmentMode,
    );

    /// CFStringRef keys for the color attachments.
    pub static kCVImageBufferYCbCrMatrixKey: CFStringRef;
    pub static kCVImageBufferColorPrimariesKey: CFStringRef;
    pub static kCVImageBufferTransferFunctionKey: CFStringRef;

    /// CFStringRef values for BT.709 limited-range. Pinned to match
    /// the encoder VUI tags set in `VideoToolboxEncoder::new`.
    pub static kCVImageBufferYCbCrMatrix_ITU_R_709_2: CFStringRef;
    pub static kCVImageBufferColorPrimaries_ITU_R_709_2: CFStringRef;
    pub static kCVImageBufferTransferFunction_ITU_R_709_2: CFStringRef;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    pub fn CFRelease(cf: CFTypeRef);
}
