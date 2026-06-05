//! Hand-rolled CUDA driver + EGL bindings for the zero-copy
//! DMA-BUF → CUDA import path NVENC consumes (GitHub issue #16).
//!
//! Both libraries are loaded with `dlopen` (via `libloading`) rather
//! than linked, so a host with no NVIDIA stack — no `libcuda.so.1`, no
//! `libEGL.so.1`, or the EGL device-enumeration extensions absent —
//! degrades gracefully: [`importer`] returns `None` and `submit_dmabuf`
//! reports `NoHardwareCodec` so the host falls back to the CPU upload
//! path. There are no link-time dependencies on either library.
//!
//! The bridge mirrors Sunshine's approach (`src/platform/linux/cuda.cpp`,
//! `graphics.cpp`): a Vulkan-exported dma-buf is **not** a CUDA opaque-fd
//! external-memory handle, so `cuImportExternalMemory(OPAQUE_FD)` fails.
//! Instead each plane's dma-buf is wrapped in an `EGLImage`
//! (`eglCreateImage` with `EGL_LINUX_DMA_BUF_EXT`), registered into CUDA
//! with `cuGraphicsEGLRegisterImage`, and the mapped EGL frame's pitched
//! device pointer is handed to NVENC inside an `AV_PIX_FMT_CUDA` AVFrame.
//!
//! Field layouts of the hand-rolled `#[repr(C)]` structs are pinned with
//! `const _: ()` size asserts, matching `crate::vaapi::ffi`. Numbers are
//! for LP64 (x86_64 + aarch64 Linux); a 32-bit cross-compile fails the
//! const eval rather than silently corrupting layout.

use std::ffi::{c_char, c_void};
use std::os::fd::AsRawFd;
use std::sync::OnceLock;

use libloading::Library;

use crate::{DmaBufFrame, DmaBufLayer};

// ---------------------------------------------------------------------------
// CUDA driver API (libcuda.so.1)
// ---------------------------------------------------------------------------

/// `CUcontext` — opaque handle to a CUDA context.
pub(crate) type CuContext = *mut c_void;
/// `CUgraphicsResource` — opaque handle to a registered graphics resource.
type CuGraphicsResource = *mut c_void;
/// `CUresult` — CUDA driver-API status code.
type CuResult = i32;

/// `CUDA_SUCCESS`.
const CUDA_SUCCESS: CuResult = 0;

/// `CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY` — NVENC only reads the input
/// surface, so register read-only to let the driver skip write-back
/// bookkeeping.
const CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY: u32 = 0x01;

/// `CU_EGL_FRAME_TYPE_PITCH` — the mapped EGL frame is a pitched linear
/// allocation (device pointer + pitch), the layout NVENC's CUDA input
/// expects. The alternative (`CU_EGL_FRAME_TYPE_ARRAY = 0`) hands back a
/// `CUarray` we'd have to copy out of.
const CU_EGL_FRAME_TYPE_PITCH: u32 = 1;

/// `CU_MEMORYTYPE_DEVICE` / `CU_MEMORYTYPE_ARRAY` — a `CUDA_MEMCPY2D`
/// operand in linear device memory / a `CUarray`. The mapped EGL frame is
/// one or the other depending on `frame_type`.
const CU_MEMORYTYPE_DEVICE: u32 = 2;
const CU_MEMORYTYPE_ARRAY: u32 = 3;
/// `CU_MEMORYTYPE_HOST` — a `CUDA_MEMCPY2D` operand in host memory
/// (test-only readback; see `ImportedCudaFrame::copy_plane_to_host`).
#[cfg(test)]
const CU_MEMORYTYPE_HOST: u32 = 1;

/// `CUDA_MEMCPY2D_v2` from `<cuda.h>`. Used to copy each EGL-imported plane
/// (a standalone device allocation) into the contiguous NV12/P010 layout of
/// a frame from the encoder's own CUDA pool — NVENC registers a single
/// pool surface per input frame and cannot consume separate per-plane
/// external pointers, so a device→device copy is the bridge (this is the
/// hop Sunshine does with `cuMemcpy2D` too). Unused operand fields stay zero.
#[repr(C)]
struct CudaMemcpy2D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: u32,
    src_host: *const c_void,
    src_device: u64,
    src_array: *mut c_void,
    src_pitch: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: u32,
    dst_host: *mut c_void,
    dst_device: u64,
    dst_array: *mut c_void,
    dst_pitch: usize,
    width_in_bytes: usize,
    height: usize,
}

const _: () = {
    // Two symmetric 56-byte operand blocks (each: 2 size_t + u32+pad +
    // host ptr + device u64 + array ptr + size_t pitch) then WidthInBytes +
    // Height = 128 B on LP64. A mismatch means the CUDA ABI shifted.
    assert!(std::mem::size_of::<CudaMemcpy2D>() == 128);
    assert!(std::mem::offset_of!(CudaMemcpy2D, src_device) == 32);
    assert!(std::mem::offset_of!(CudaMemcpy2D, dst_device) == 88);
    assert!(std::mem::offset_of!(CudaMemcpy2D, width_in_bytes) == 112);
};

/// `AVCUDADeviceContext` from `<libavutil/hwcontext_cuda.h>`. We read
/// `cuda_ctx` (FFmpeg's `CUcontext`) so the EGL→CUDA registration happens
/// against the very context the `*_nvenc` encoder allocates surfaces in —
/// registering against a different context would hand NVENC pointers it
/// can't dereference.
///
/// The struct has trailing fields after `internal` in some FFmpeg builds
/// (a `CUstream` array, push/pop callbacks); they live after the three
/// fields we touch, so reading `cuda_ctx` at offset 0 stays correct. The
/// size assert below pins the prefix we depend on, not the whole struct.
#[repr(C)]
pub(crate) struct AvCudaDeviceContext {
    pub(crate) cuda_ctx: CuContext,
    pub(crate) stream: *mut c_void,
    pub(crate) internal: *mut c_void,
}

/// `CUeglFrame` from `<cudaEGL.h>`. The leading union is the per-plane
/// storage: `frame.pArray[i]` for `ARRAY` frames, `frame.pPitch[i]` for
/// `PITCH` frames. We only consume `pPitch[0]` (the plane's base device
/// pointer) and `pitch` for `frameType == CU_EGL_FRAME_TYPE_PITCH`.
///
/// The union of two `[*mut c_void; 3]` arrays is 24 bytes; the nine
/// trailing `u32` scalars take it to 64 with no tail padding on LP64.
#[repr(C)]
#[derive(Clone, Copy)]
union CuEglFramePtr {
    p_array: [*mut c_void; 3],
    p_pitch: [*mut c_void; 3],
}

// The mapping FFI writes every field; we only read `frame`, `width`,
// `pitch`, and `frame_type`. The rest are part of the ABI layout the
// size/offset asserts pin — keep them named for documentation and let
// the allow cover the unread tail rather than scattering `_` prefixes
// (which don't suppress dead_code on named struct fields).
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct CuEglFrame {
    frame: CuEglFramePtr,
    width: u32,
    height: u32,
    depth: u32,
    pitch: u32,
    plane_count: u32,
    num_channels: u32,
    frame_type: u32,
    egl_color_format: u32,
    cu_format: u32,
}

const _: () = {
    // CUcontext / stream / internal are three pointers (8 B each on LP64).
    assert!(std::mem::size_of::<AvCudaDeviceContext>() == 24);
    assert!(std::mem::offset_of!(AvCudaDeviceContext, cuda_ctx) == 0);
    assert!(std::mem::offset_of!(AvCudaDeviceContext, stream) == 8);
    assert!(std::mem::offset_of!(AvCudaDeviceContext, internal) == 16);

    // Union of 3 pointers = 24 B, then 9 * u32 = 36 B → 60, padded to a
    // pointer-alignment multiple = 64.
    assert!(std::mem::size_of::<CuEglFramePtr>() == 24);
    assert!(std::mem::size_of::<CuEglFrame>() == 64);
    assert!(std::mem::offset_of!(CuEglFrame, frame) == 0);
    assert!(std::mem::offset_of!(CuEglFrame, width) == 24);
    assert!(std::mem::offset_of!(CuEglFrame, pitch) == 36);
    assert!(std::mem::offset_of!(CuEglFrame, frame_type) == 48);
};

// CUDA driver function-pointer signatures (the `_v2` ABI suffix is the
// symbol name; the C call has no suffix). `extern "C"` matches the CUDA
// driver's calling convention on every target we build for.
type FnCuCtxPushCurrent = unsafe extern "C" fn(CuContext) -> CuResult;
type FnCuCtxPopCurrent = unsafe extern "C" fn(*mut CuContext) -> CuResult;
type FnCuGraphicsEglRegisterImage =
    unsafe extern "C" fn(*mut CuGraphicsResource, *mut c_void, u32) -> CuResult;
type FnCuGraphicsResourceGetMappedEglFrame =
    unsafe extern "C" fn(*mut CuEglFrame, CuGraphicsResource, u32, u32) -> CuResult;
type FnCuGraphicsUnregisterResource = unsafe extern "C" fn(CuGraphicsResource) -> CuResult;
type FnCuMemcpy2D = unsafe extern "C" fn(*const CudaMemcpy2D) -> CuResult;

/// The subset of CUDA driver entry points the import path needs.
/// `Copy` so [`ImportGuard`] can keep its own snapshot of the freeing
/// calls without borrowing the importer.
#[derive(Clone, Copy)]
struct CudaFns {
    ctx_push_current: FnCuCtxPushCurrent,
    ctx_pop_current: FnCuCtxPopCurrent,
    graphics_egl_register_image: FnCuGraphicsEglRegisterImage,
    graphics_resource_get_mapped_egl_frame: FnCuGraphicsResourceGetMappedEglFrame,
    graphics_unregister_resource: FnCuGraphicsUnregisterResource,
    memcpy_2d: FnCuMemcpy2D,
}

impl CudaFns {
    /// Resolve every entry point from an open `libcuda.so.1`. Returns
    /// `None` if any symbol is absent (an ancient/partial driver).
    ///
    /// # Safety
    /// The transmuted symbol types must match the CUDA driver ABI; they
    /// do, per `<cuda.h>` / `<cudaEGL.h>`.
    unsafe fn load(lib: &Library) -> Option<Self> {
        // SAFETY: each symbol name is a real entry point in libcuda.so.1
        // and the bound signature matches the driver header. We copy the
        // function pointer out of the `Symbol` guard, which is sound for
        // as long as `lib` stays loaded — the importer keeps it resident
        // for the process lifetime (see `EglCudaImporter::_libs`).
        unsafe {
            Some(Self {
                ctx_push_current: *lib.get(b"cuCtxPushCurrent_v2\0").ok()?,
                ctx_pop_current: *lib.get(b"cuCtxPopCurrent_v2\0").ok()?,
                graphics_egl_register_image: *lib.get(b"cuGraphicsEGLRegisterImage\0").ok()?,
                graphics_resource_get_mapped_egl_frame: *lib
                    .get(b"cuGraphicsResourceGetMappedEglFrame\0")
                    .ok()?,
                graphics_unregister_resource: *lib.get(b"cuGraphicsUnregisterResource\0").ok()?,
                memcpy_2d: *lib.get(b"cuMemcpy2D_v2\0").ok()?,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// EGL (libEGL.so.1)
// ---------------------------------------------------------------------------

type EglDisplay = *mut c_void;
type EglImage = *mut c_void;
type EglDeviceExt = *mut c_void;
type EglContext = *mut c_void;
/// `EGLClientBuffer` — always null on the dma-buf import path.
type EglClientBuffer = *mut c_void;
/// `EGLAttrib` — `intptr_t`. The attrib list passed to `eglCreateImage`
/// is `EGLAttrib`-wide (the `KHR_image_base`/`EXT_image_dma_buf_import`
/// "modifiers" variant), so a 32-bit fd / offset / pitch all widen
/// losslessly into it.
type EglAttrib = isize;
type EglInt = i32;
type EglEnum = u32;
type EglBoolean = u32;

const EGL_NONE: EglAttrib = 0x3038;
const EGL_WIDTH: EglAttrib = 0x3057;
const EGL_HEIGHT: EglAttrib = 0x3056;
const EGL_LINUX_DRM_FOURCC_EXT: EglAttrib = 0x3271;
const EGL_LINUX_DMA_BUF_EXT: EglEnum = 0x3270;
const EGL_DMA_BUF_PLANE0_FD_EXT: EglAttrib = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: EglAttrib = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: EglAttrib = 0x3274;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: EglAttrib = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: EglAttrib = 0x3444;
const EGL_DMA_BUF_PLANE1_FD_EXT: EglAttrib = 0x3275;
const EGL_DMA_BUF_PLANE1_OFFSET_EXT: EglAttrib = 0x3276;
const EGL_DMA_BUF_PLANE1_PITCH_EXT: EglAttrib = 0x3277;
const EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT: EglAttrib = 0x3445;
const EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT: EglAttrib = 0x3446;
const EGL_DMA_BUF_PLANE2_FD_EXT: EglAttrib = 0x3278;
const EGL_DMA_BUF_PLANE2_OFFSET_EXT: EglAttrib = 0x3279;
const EGL_DMA_BUF_PLANE2_PITCH_EXT: EglAttrib = 0x327A;
const EGL_DMA_BUF_PLANE2_MODIFIER_LO_EXT: EglAttrib = 0x3447;
const EGL_DMA_BUF_PLANE2_MODIFIER_HI_EXT: EglAttrib = 0x3448;
const EGL_PLATFORM_DEVICE_EXT: EglEnum = 0x313F;
const EGL_CUDA_DEVICE_NV: EglInt = 0x323A;
const EGL_TRUE: EglBoolean = 1;
const EGL_NO_CONTEXT: EglContext = std::ptr::null_mut();
const EGL_NO_DISPLAY: EglDisplay = std::ptr::null_mut();
const EGL_NO_IMAGE: EglImage = std::ptr::null_mut();

/// `DRM_FORMAT_MOD_INVALID` — sentinel meaning "no explicit modifier".
/// When the dma-buf carries this we omit the per-plane modifier attribs
/// so EGL falls back to the driver's implicit-modifier import path.
const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

// Core EGL entry points (resolved by name from libEGL.so.1) and the
// device/platform extension entry points (resolved via eglGetProcAddress,
// which is how EGL exposes `EXT`/`KHR` functions).
type FnEglGetProcAddress = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type FnEglInitialize = unsafe extern "C" fn(EglDisplay, *mut EglInt, *mut EglInt) -> EglBoolean;
type FnEglCreateImage = unsafe extern "C" fn(
    EglDisplay,
    EglContext,
    EglEnum,
    EglClientBuffer,
    *const EglAttrib,
) -> EglImage;
type FnEglDestroyImage = unsafe extern "C" fn(EglDisplay, EglImage) -> EglBoolean;
type FnEglGetError = unsafe extern "C" fn() -> EglInt;
type FnEglQueryDevicesExt =
    unsafe extern "C" fn(EglInt, *mut EglDeviceExt, *mut EglInt) -> EglBoolean;
type FnEglQueryDeviceAttribExt =
    unsafe extern "C" fn(EglDeviceExt, EglInt, *mut EglAttrib) -> EglBoolean;
type FnEglGetPlatformDisplayExt =
    unsafe extern "C" fn(EglEnum, *mut c_void, *const EglAttrib) -> EglDisplay;

/// The subset of EGL entry points the import path needs. `Copy` so the
/// drop guard can hold its own snapshot of `destroy_image` + `get_error`.
#[derive(Clone, Copy)]
struct EglFns {
    initialize: FnEglInitialize,
    create_image: FnEglCreateImage,
    destroy_image: FnEglDestroyImage,
    get_error: FnEglGetError,
    query_devices: FnEglQueryDevicesExt,
    query_device_attrib: FnEglQueryDeviceAttribExt,
    get_platform_display: FnEglGetPlatformDisplayExt,
}

impl EglFns {
    /// Resolve every EGL entry point. Core functions come straight from
    /// the library; the `EXT` device/platform functions are looked up via
    /// `eglGetProcAddress` (the EGL-mandated way to reach extensions).
    /// Returns `None` if any symbol or extension is missing.
    ///
    /// # Safety
    /// The transmuted symbol/proc-address types must match the EGL ABI;
    /// they do, per `<EGL/egl.h>` and the `EGL_EXT_device_*` /
    /// `EGL_EXT_platform_device` extension specs.
    unsafe fn load(lib: &Library) -> Option<Self> {
        // SAFETY: core EGL symbols present in any libEGL.so.1; bound
        // signatures match <EGL/egl.h>. Copied out of the Symbol guard;
        // sound while `lib` stays resident (held by the importer).
        let get_proc_address: FnEglGetProcAddress = *unsafe { lib.get(b"eglGetProcAddress\0") }.ok()?;
        let initialize: FnEglInitialize = *unsafe { lib.get(b"eglInitialize\0") }.ok()?;
        let create_image: FnEglCreateImage = *unsafe { lib.get(b"eglCreateImage\0") }.ok()?;
        let destroy_image: FnEglDestroyImage = *unsafe { lib.get(b"eglDestroyImage\0") }.ok()?;
        let get_error: FnEglGetError = *unsafe { lib.get(b"eglGetError\0") }.ok()?;

        // SAFETY: eglGetProcAddress returns a function pointer of the
        // queried entry point's type, or null when the extension is
        // absent. We transmute the non-null result to the matching
        // signature and bail on null (extension unavailable). Calling
        // through the resolved get_proc_address is the documented EGL
        // mechanism for reaching extension entry points.
        let query_devices: FnEglQueryDevicesExt = unsafe {
            std::mem::transmute::<*mut c_void, Option<FnEglQueryDevicesExt>>(get_proc_address(
                c"eglQueryDevicesEXT".as_ptr(),
            ))
        }?;
        let query_device_attrib: FnEglQueryDeviceAttribExt = unsafe {
            std::mem::transmute::<*mut c_void, Option<FnEglQueryDeviceAttribExt>>(get_proc_address(
                c"eglQueryDeviceAttribEXT".as_ptr(),
            ))
        }?;
        let get_platform_display: FnEglGetPlatformDisplayExt = unsafe {
            std::mem::transmute::<*mut c_void, Option<FnEglGetPlatformDisplayExt>>(get_proc_address(
                c"eglGetPlatformDisplayEXT".as_ptr(),
            ))
        }?;

        Some(Self {
            initialize,
            create_image,
            destroy_image,
            get_error,
            query_devices,
            query_device_attrib,
            get_platform_display,
        })
    }
}

// ---------------------------------------------------------------------------
// High-level importer
// ---------------------------------------------------------------------------

/// Process-global EGL/CUDA importer. Holds the dlopen'd libraries
/// resident, the resolved CUDA + EGL entry points, and an `eglInitialize`d
/// display bound to CUDA device 0 (the same physical GPU the default CUDA
/// context — and therefore NVENC — runs on).
pub(crate) struct EglCudaImporter {
    // The libraries are kept alive for the process lifetime so the raw
    // function pointers copied out of them stay valid. Never read directly.
    _libs: (Library, Library),
    cuda: CudaFns,
    egl: EglFns,
    display: EglDisplay,
}

// SAFETY: every field is either a function pointer (process-global,
// thread-agnostic) or an `EGLDisplay` (an opaque process-global handle —
// EGL and the CUDA driver are themselves thread-safe). The importer is
// only ever reached through `&'static EglCudaImporter`, and the per-frame
// CUDA context push/pop that the import performs is serialised by the
// `&mut self` on the encoder. Sending/sharing the handles is sound.
unsafe impl Send for EglCudaImporter {}
unsafe impl Sync for EglCudaImporter {}

/// One imported plane, as the mapped CUeglFrame exposes it. A `PITCH`
/// frame hands back a linear device pointer (+ row pitch); an `ARRAY`
/// frame hands back a `CUarray`. Exactly one of `device_ptr` / `array` is
/// non-null; [`ImportedCudaFrame::copy_plane_into`] dispatches on which.
struct ImportedPlane {
    device_ptr: *mut u8,
    array: *mut c_void,
    pitch: usize,
}

/// The result of [`EglCudaImporter::import`]: per-plane device pointers
/// for an `AV_PIX_FMT_CUDA` AVFrame, plus a guard that frees the EGL
/// images + CUDA graphics resources when dropped.
pub(crate) struct ImportedCudaFrame {
    planes: Vec<ImportedPlane>,
    // Drop order is declaration order: planes (plain pointers, no
    // cleanup) before the guard, which unregisters/destroys the backing
    // resources. Kept last so the pointers are valid for the AVFrame's
    // whole lifetime up to the explicit drop after send_frame.
    _guard: ImportGuard,
}

impl ImportedCudaFrame {
    /// Number of imported planes (1 per dma-buf layer).
    pub(crate) fn plane_count(&self) -> usize {
        self.planes.len()
    }

    /// Copy imported plane `i` (a standalone device allocation) into a
    /// destination device pointer with `dst_pitch`, `width_bytes` × `height`.
    /// Device→device, on the import's CUDA context — no CPU bounce. NVENC
    /// registers one contiguous pool surface per input frame, so the caller
    /// uses this to assemble the planes into a pool frame's NV12/P010 layout.
    pub(crate) fn copy_plane_into(
        &self,
        i: usize,
        dst_device_ptr: *mut u8,
        dst_pitch: usize,
        width_bytes: usize,
        height: usize,
    ) -> Result<(), String> {
        let plane = &self.planes[i];
        let src_is_array = !plane.array.is_null();
        let cpy = CudaMemcpy2D {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: if src_is_array {
                CU_MEMORYTYPE_ARRAY
            } else {
                CU_MEMORYTYPE_DEVICE
            },
            src_host: std::ptr::null(),
            src_device: plane.device_ptr as u64,
            src_array: plane.array,
            // cuMemcpy2D ignores srcPitch for an ARRAY source.
            src_pitch: plane.pitch,
            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_memory_type: CU_MEMORYTYPE_DEVICE,
            dst_host: std::ptr::null_mut(),
            dst_device: dst_device_ptr as u64,
            dst_array: std::ptr::null_mut(),
            dst_pitch,
            width_in_bytes: width_bytes,
            height,
        };
        // SAFETY: both operands are device allocations in `cuda_ctx` (the
        // imported plane and a pool frame allocated against the same FFmpeg
        // CUDA context). We push that context for the copy and pop it after.
        // cuMemcpy2D is synchronous on the default stream, so the data is in
        // place when this returns.
        unsafe {
            let push_rc = (self._guard.cuda.ctx_push_current)(self._guard.cuda_ctx);
            if push_rc != CUDA_SUCCESS {
                return Err(format!("cuCtxPushCurrent (copy) failed: {push_rc}"));
            }
            let rc = (self._guard.cuda.memcpy_2d)(&cpy);
            let mut popped: CuContext = std::ptr::null_mut();
            (self._guard.cuda.ctx_pop_current)(&mut popped);
            if rc != CUDA_SUCCESS {
                return Err(format!("cuMemcpy2D(plane {i}) failed: {rc}"));
            }
        }
        Ok(())
    }

    /// Copy a SOURCE device pointer (`src_device_ptr` with `src_pitch`) INTO
    /// imported plane `i`, `width_bytes` × `height`. The exact reverse of
    /// [`Self::copy_plane_into`]: device→device on the import's CUDA context,
    /// no CPU bounce. The NVDEC decoder uses this to fill an EGL-imported
    /// dma-buf surface plane from the NVDEC-decoded `AV_PIX_FMT_CUDA` frame's
    /// device pointers, so the renderer can import the dma-buf zero-copy.
    ///
    /// The imported (destination) plane may be `PITCH` (a linear device
    /// pointer + row pitch) or `ARRAY` (a `CUarray`) — exactly the dispatch
    /// `copy_plane_into` does on the source side, mirrored here onto the dst.
    pub(crate) fn copy_plane_from(
        &self,
        i: usize,
        src_device_ptr: *mut u8,
        src_pitch: usize,
        width_bytes: usize,
        height: usize,
    ) -> Result<(), String> {
        let plane = &self.planes[i];
        let dst_is_array = !plane.array.is_null();
        let cpy = CudaMemcpy2D {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: CU_MEMORYTYPE_DEVICE,
            src_host: std::ptr::null(),
            src_device: src_device_ptr as u64,
            src_array: std::ptr::null_mut(),
            src_pitch,
            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_memory_type: if dst_is_array {
                CU_MEMORYTYPE_ARRAY
            } else {
                CU_MEMORYTYPE_DEVICE
            },
            dst_host: std::ptr::null_mut(),
            dst_device: plane.device_ptr as u64,
            dst_array: plane.array,
            // cuMemcpy2D ignores dstPitch for an ARRAY destination.
            dst_pitch: plane.pitch,
            width_in_bytes: width_bytes,
            height,
        };
        // SAFETY: both operands are device allocations in `cuda_ctx` (the
        // source is the NVDEC-decoded CUDA frame's plane, the destination is
        // the EGL-imported dma-buf surface plane registered against the same
        // FFmpeg CUDA context). We push that context for the copy and pop it
        // after. cuMemcpy2D is synchronous on the default stream, so the data
        // is in place when this returns.
        unsafe {
            let push_rc = (self._guard.cuda.ctx_push_current)(self._guard.cuda_ctx);
            if push_rc != CUDA_SUCCESS {
                return Err(format!("cuCtxPushCurrent (copy_from) failed: {push_rc}"));
            }
            let rc = (self._guard.cuda.memcpy_2d)(&cpy);
            let mut popped: CuContext = std::ptr::null_mut();
            (self._guard.cuda.ctx_pop_current)(&mut popped);
            if rc != CUDA_SUCCESS {
                return Err(format!("cuMemcpy2D(into plane {i}) failed: {rc}"));
            }
        }
        Ok(())
    }

    /// Copy imported plane `i` (device or array) out to a host buffer with
    /// `dst_pitch`, `width_bytes` × `height`. Test-only readback used to
    /// verify a decoded surface holds the right pixels without a full GPU
    /// import. `dst` must be at least `dst_pitch * height` bytes.
    #[cfg(test)]
    pub(crate) fn copy_plane_to_host(
        &self,
        i: usize,
        dst: *mut u8,
        dst_pitch: usize,
        width_bytes: usize,
        height: usize,
    ) -> Result<(), String> {
        let plane = &self.planes[i];
        let src_is_array = !plane.array.is_null();
        let cpy = CudaMemcpy2D {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: if src_is_array {
                CU_MEMORYTYPE_ARRAY
            } else {
                CU_MEMORYTYPE_DEVICE
            },
            src_host: std::ptr::null(),
            src_device: plane.device_ptr as u64,
            src_array: plane.array,
            src_pitch: plane.pitch,
            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_memory_type: CU_MEMORYTYPE_HOST,
            dst_host: dst.cast::<c_void>(),
            dst_device: 0,
            dst_array: std::ptr::null_mut(),
            dst_pitch,
            width_in_bytes: width_bytes,
            height,
        };
        // SAFETY: src is the imported surface plane in `cuda_ctx`; dst is a
        // host buffer of at least dst_pitch*height bytes. Push/pop the context
        // around the synchronous copy.
        unsafe {
            let push_rc = (self._guard.cuda.ctx_push_current)(self._guard.cuda_ctx);
            if push_rc != CUDA_SUCCESS {
                return Err(format!("cuCtxPushCurrent (to_host) failed: {push_rc}"));
            }
            let rc = (self._guard.cuda.memcpy_2d)(&cpy);
            let mut popped: CuContext = std::ptr::null_mut();
            (self._guard.cuda.ctx_pop_current)(&mut popped);
            if rc != CUDA_SUCCESS {
                return Err(format!("cuMemcpy2D(plane {i} → host) failed: {rc}"));
            }
        }
        Ok(())
    }
}

/// Owns the EGL images + CUDA graphics resources backing an imported
/// frame and releases them on drop. Holds its own copy of the CUDA
/// context + the EGL/CUDA freeing functions so it doesn't borrow the
/// importer — the guard outlives the `import()` call by design (it must
/// stay alive until NVENC has consumed the input).
///
/// With NVENC's `delay=0`, the encoder consumes the input surface within
/// `send_frame`, so in practice this guard drops immediately after the
/// encoder's `submit_dmabuf` returns. Keeping the resources registered
/// until then is what guarantees the device pointers stay mapped while
/// NVENC reads them.
struct ImportGuard {
    cuda_ctx: CuContext,
    display: EglDisplay,
    cuda: CudaFns,
    egl: EglFns,
    // (egl_image, cu_resource) pairs, freed in reverse registration order.
    resources: Vec<(EglImage, CuGraphicsResource)>,
}

impl Drop for ImportGuard {
    fn drop(&mut self) {
        // SAFETY: cuda_ctx is the live FFmpeg CUDA context; pushing it
        // current is required before the per-resource unregister, which must
        // run on the context the resource was registered against. Each
        // `cu_resource` came from cuGraphicsEGLRegisterImage and each
        // `egl_image` from eglCreateImage on `display`; both are freed exactly
        // once here. Teardown return codes are ignored (no recovery), but the
        // pop is gated on a SUCCESSFUL push — popping without a matching push
        // would corrupt the calling thread's context stack for every later
        // CUDA call. A failed push can't happen with a live `_hw_device`
        // backing `cuda_ctx`, but if it did we accept the resource leak over
        // stack corruption.
        unsafe {
            if (self.cuda.ctx_push_current)(self.cuda_ctx) != CUDA_SUCCESS {
                tracing::warn!(
                    leaked = self.resources.len(),
                    "NVENC import guard: cuCtxPushCurrent failed in drop; leaking EGL/CUDA \
                     resources to avoid context-stack corruption"
                );
                return;
            }
            for (image, resource) in self.resources.drain(..) {
                (self.cuda.graphics_unregister_resource)(resource);
                (self.egl.destroy_image)(self.display, image);
            }
            let mut popped: CuContext = std::ptr::null_mut();
            (self.cuda.ctx_pop_current)(&mut popped);
        }
    }
}

impl EglCudaImporter {
    /// Load the libraries, resolve the entry points, and bind an EGL
    /// display to CUDA device 0. Returns `None` on any failure so a
    /// non-NVIDIA / partial host degrades to the CPU upload path.
    fn init() -> Option<Self> {
        // SAFETY: dlopen of the system CUDA/EGL runtimes by soname. The
        // libraries are stored in `_libs` and never unloaded, so every
        // function pointer copied out of them below stays valid.
        let cuda_lib = unsafe { Library::new("libcuda.so.1") }.ok()?;
        let egl_lib = unsafe { Library::new("libEGL.so.1") }.ok()?;

        // SAFETY: signatures match the CUDA / EGL ABIs; see CudaFns::load
        // and EglFns::load.
        let cuda = unsafe { CudaFns::load(&cuda_lib) }?;
        let egl = unsafe { EglFns::load(&egl_lib) }?;

        let display = egl_display_for_cuda_device0(&egl)?;

        Some(Self {
            _libs: (cuda_lib, egl_lib),
            cuda,
            egl,
            display,
        })
    }

    /// Import a capture DMA-BUF into CUDA, returning per-plane device
    /// pointers wrapped in a guard that frees the EGL/CUDA resources on
    /// drop. `cuda_ctx` must be the FFmpeg encoder's CUDA context (read
    /// from its `AVCUDADeviceContext`); `width`/`height` are the full
    /// luma dimensions.
    ///
    /// Returns `Err(String)` (a human-readable diagnostic the caller maps
    /// to `NoHardwareCodec`) on any EGL/CUDA failure.
    //
    // `layer.pitch[0] as i32` is a dma-buf row pitch in bytes: it fits
    // in i32 for any realistic surface (a 32 Ki-wide 4-byte plane is
    // 128 KiB), and AVFrame::linesize is i32-typed at the ffmpeg ABI. The
    // allow documents that the same-width u32→i32 reinterpret is intended.
    #[allow(clippy::cast_possible_wrap)]
    pub(crate) fn import(
        &self,
        cuda_ctx: CuContext,
        frame: &DmaBufFrame,
        width: u32,
        height: u32,
    ) -> Result<ImportedCudaFrame, String> {
        if frame.objects.len() != 1 {
            return Err(format!(
                "EGL/CUDA import expects a single dma-buf object, got {}",
                frame.objects.len()
            ));
        }
        if frame.layers.is_empty() || frame.layers.len() > 4 {
            return Err(format!(
                "EGL/CUDA import expects 1..=4 layers, got {}",
                frame.layers.len()
            ));
        }
        let object = &frame.objects[0];

        // Push the encoder's CUDA context: cuGraphicsEGLRegisterImage and
        // the mapped-frame lookup must run with it current. We pop it on
        // every return path below.
        //
        // SAFETY: cuda_ctx is FFmpeg's live CUDA context for this encoder.
        let push_rc = unsafe { (self.cuda.ctx_push_current)(cuda_ctx) };
        if push_rc != CUDA_SUCCESS {
            return Err(format!("cuCtxPushCurrent failed: {push_rc}"));
        }

        let mut guard = ImportGuard {
            cuda_ctx,
            display: self.display,
            cuda: self.cuda,
            egl: self.egl,
            resources: Vec::with_capacity(frame.layers.len()),
        };
        let mut planes: Vec<ImportedPlane> = Vec::with_capacity(frame.layers.len());

        // Import the WHOLE surface as ONE multi-plane EGLImage keyed on the
        // surface fourcc (NV12 / P010), not one image per plane. CUDA's
        // cuGraphicsEGLRegisterImage rejects a standalone single-channel 8-bit
        // (R8) Y-plane image with INVALID_VALUE; the canonical YUV import is a
        // single semi-planar image, from which the mapped CUeglFrame exposes
        // both planes.
        let attribs = build_frame_attribs(
            width,
            height,
            frame.fourcc,
            object.fd.as_raw_fd(),
            &frame.layers,
            object.drm_format_modifier,
        );

        // SAFETY: `attribs` is a properly terminated EGLAttrib list (EGL_NONE
        // last); EGL_NO_CONTEXT + null client buffer are the
        // EXT_image_dma_buf_import convention. The returned image, if non-null,
        // is owned by us and freed via the guard.
        let image = unsafe {
            (self.egl.create_image)(
                self.display,
                EGL_NO_CONTEXT,
                EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(),
                attribs.as_ptr(),
            )
        };
        if image == EGL_NO_IMAGE {
            // SAFETY: get_error is a no-arg EGL query.
            let egl_err = unsafe { (self.egl.get_error)() };
            self.pop_ctx();
            return Err(format!(
                "eglCreateImage(fourcc 0x{:08x}) failed: eglGetError 0x{:04x}",
                frame.fourcc, egl_err
            ));
        }

        let mut resource: CuGraphicsResource = std::ptr::null_mut();
        // SAFETY: `image` is a live EGLImage from eglCreateImage above;
        // READ_ONLY matches NVENC's read-only access. On success the resource
        // is stored in the guard for later unregistration.
        let reg_rc = unsafe {
            (self.cuda.graphics_egl_register_image)(
                &mut resource,
                image,
                CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY,
            )
        };
        if reg_rc != CUDA_SUCCESS {
            // SAFETY: image is live and not yet handed to CUDA; free it.
            unsafe { (self.egl.destroy_image)(self.display, image) };
            self.pop_ctx();
            return Err(format!("cuGraphicsEGLRegisterImage failed: {reg_rc}"));
        }
        // Registered: hand both to the guard so an error past here still frees.
        guard.resources.push((image, resource));

        let mut egl_frame: CuEglFrame = unsafe { std::mem::zeroed() };
        // SAFETY: `resource` was just registered; index 0 / mipLevel 0 is the
        // only sub-resource a registered EGLImage exposes.
        let map_rc = unsafe {
            (self.cuda.graphics_resource_get_mapped_egl_frame)(&mut egl_frame, resource, 0, 0)
        };
        if map_rc != CUDA_SUCCESS {
            self.pop_ctx();
            return Err(format!(
                "cuGraphicsResourceGetMappedEglFrame failed: {map_rc}"
            ));
        }

        // One ImportedPlane per CUeglFrame plane (2 for NV12/P010), tagged by
        // frame type: PITCH hands back device pointers + a row pitch; ARRAY
        // hands back CUarrays (the copy path handles both). Cap at the union's
        // 3 slots and the surface's layer count.
        let plane_count = (egl_frame.plane_count as usize).min(3).min(frame.layers.len());
        if plane_count == 0 {
            // A misbehaving driver / format CUDA doesn't model: with no planes
            // the pool frame would ship to NVENC holding stale surface memory
            // (silent corruption). Fail loudly instead.
            self.pop_ctx();
            return Err(format!(
                "cuGraphicsResourceGetMappedEglFrame reported plane_count=0 for fourcc \
                 0x{:08x} (expected 2 for NV12/P010)",
                frame.fourcc
            ));
        }
        let pitch = usize::try_from(egl_frame.pitch).unwrap_or(0);
        for i in 0..plane_count {
            // SAFETY: the active union arm is selected by `frame_type`; both
            // arms are `[*mut c_void; 3]` and `i < plane_count <= 3`.
            let plane = unsafe {
                if egl_frame.frame_type == CU_EGL_FRAME_TYPE_PITCH {
                    ImportedPlane {
                        device_ptr: egl_frame.frame.p_pitch[i].cast::<u8>(),
                        array: std::ptr::null_mut(),
                        pitch,
                    }
                } else {
                    ImportedPlane {
                        device_ptr: std::ptr::null_mut(),
                        array: egl_frame.frame.p_array[i],
                        pitch,
                    }
                }
            };
            planes.push(plane);
        }

        self.pop_ctx();

        Ok(ImportedCudaFrame {
            planes,
            _guard: guard,
        })
    }

    /// Pop the CUDA context pushed at the start of `import`. Errors are
    /// unrecoverable and ignored — the alternative is unwinding through a
    /// half-popped context stack.
    fn pop_ctx(&self) {
        // SAFETY: balances the cuCtxPushCurrent in `import`. The out
        // pointer receives the popped context, which we discard.
        unsafe {
            let mut popped: CuContext = std::ptr::null_mut();
            (self.cuda.ctx_pop_current)(&mut popped);
        }
    }
}

/// Build the `eglCreateImage` attribute list for one dma-buf plane.
///
/// Factored out (and kept verbose) because the per-plane geometry is the
/// most likely thing to need a hardware tweak: `width`/`height` here are
/// the **subsampled plane** dimensions (see [`plane_dimensions`]) and
/// `fourcc` is the per-plane DRM format (`R8`/`GR88` for NV12, `R16`/`RG32`
/// for P010), not the surface fourcc.
///
/// Mirrors Sunshine's `surface_descriptor_to_egl_attribs`
/// (`graphics.cpp`): width, height, fourcc, then plane-0 fd/offset/pitch,
/// then the modifier lo/hi pair only when an explicit modifier is present,
/// terminated by `EGL_NONE`. Every value widens losslessly into the
/// `EGLAttrib` (intptr) list.
//
// `EGLAttrib` is `intptr_t` (64-bit on LP64): the 32-bit width/height/
// fourcc/fd/offset/pitch and the two 32-bit modifier halves all fit. The
// allow covers those intentional ABI widenings; the `& 0xFFFF_FFFF` /
// `>> 32` masking makes the modifier-half truncation explicit and lossless.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless
)]
fn build_frame_attribs(
    width: u32,
    height: u32,
    fourcc: u32,
    fd: i32,
    layers: &[DmaBufLayer],
    modifier: u64,
) -> Vec<EglAttrib> {
    let mut attribs: Vec<EglAttrib> = vec![
        EGL_WIDTH,
        width as EglAttrib,
        EGL_HEIGHT,
        height as EglAttrib,
        EGL_LINUX_DRM_FOURCC_EXT,
        fourcc as EglAttrib,
    ];
    // One PLANEn_* triple (+ modifier pair) per dma-buf layer, all sharing
    // the single object's fd. EGL_EXT_image_dma_buf_import supports up to 3
    // planes for the import target.
    for (i, layer) in layers.iter().enumerate().take(3) {
        let (fd_tok, off_tok, pitch_tok, lo_tok, hi_tok) = plane_attr_tokens(i);
        attribs.push(fd_tok);
        attribs.push(fd as EglAttrib);
        attribs.push(off_tok);
        attribs.push(layer.offset[0] as EglAttrib);
        attribs.push(pitch_tok);
        attribs.push(layer.pitch[0] as EglAttrib);
        // Pass the tiling modifier only when the buffer carries an explicit
        // one. With DRM_FORMAT_MOD_INVALID the driver picks the implicit
        // modifier; passing the sentinel through would be rejected.
        if modifier != DRM_FORMAT_MOD_INVALID {
            attribs.push(lo_tok);
            attribs.push((modifier & 0xFFFF_FFFF) as EglAttrib);
            attribs.push(hi_tok);
            attribs.push((modifier >> 32) as EglAttrib);
        }
    }
    attribs.push(EGL_NONE);
    attribs
}

/// The five `EGL_DMA_BUF_PLANE{n}_*` attribute tokens (fd, offset, pitch,
/// modifier-lo, modifier-hi) for plane index `n` (0..=2).
fn plane_attr_tokens(n: usize) -> (EglAttrib, EglAttrib, EglAttrib, EglAttrib, EglAttrib) {
    match n {
        0 => (
            EGL_DMA_BUF_PLANE0_FD_EXT,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT,
            EGL_DMA_BUF_PLANE0_PITCH_EXT,
            EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
            EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
        ),
        1 => (
            EGL_DMA_BUF_PLANE1_FD_EXT,
            EGL_DMA_BUF_PLANE1_OFFSET_EXT,
            EGL_DMA_BUF_PLANE1_PITCH_EXT,
            EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT,
            EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT,
        ),
        _ => (
            EGL_DMA_BUF_PLANE2_FD_EXT,
            EGL_DMA_BUF_PLANE2_OFFSET_EXT,
            EGL_DMA_BUF_PLANE2_PITCH_EXT,
            EGL_DMA_BUF_PLANE2_MODIFIER_LO_EXT,
            EGL_DMA_BUF_PLANE2_MODIFIER_HI_EXT,
        ),
    }
}

/// Find the EGL device whose `EGL_CUDA_DEVICE_NV` attribute is CUDA
/// device 0, open a platform display on it, and `eglInitialize` it.
///
/// Picking device 0 keeps the EGL→CUDA import on the same physical GPU as
/// the default CUDA context FFmpeg's nvenc wrapper creates. (Multi-GPU
/// device selection is deferred — the encoder's `new()` also pins CUDA
/// device 0; the two must agree.) Returns `None` if no CUDA-backed EGL
/// device exists or initialisation fails.
fn egl_display_for_cuda_device0(egl: &EglFns) -> Option<EglDisplay> {
    // Enumerate up to a generous fixed cap of EGL devices.
    const MAX_DEVICES: EglInt = 32;
    let mut devices: [EglDeviceExt; MAX_DEVICES as usize] = [std::ptr::null_mut(); 32];
    let mut num_devices: EglInt = 0;

    // SAFETY: query_devices fills up to MAX_DEVICES entries and writes the
    // count to num_devices; both out buffers are valid for that span.
    let ok = unsafe {
        (egl.query_devices)(MAX_DEVICES, devices.as_mut_ptr(), &mut num_devices)
    };
    if ok != EGL_TRUE || num_devices <= 0 {
        return None;
    }

    // num_devices > 0 per the guard above, so the conversion can't fail;
    // clamp to the buffer length defensively against a misbehaving driver.
    let count = usize::try_from(num_devices).unwrap_or(0).min(devices.len());
    for &device in &devices[..count] {
        let mut cuda_device: EglAttrib = -1;
        // SAFETY: device is a valid EGLDeviceEXT from query_devices; the
        // attribute out pointer is valid. A false return means the device
        // doesn't expose EGL_CUDA_DEVICE_NV — skip it.
        let ok = unsafe {
            (egl.query_device_attrib)(device, EGL_CUDA_DEVICE_NV, &mut cuda_device)
        };
        if ok != EGL_TRUE || cuda_device != 0 {
            continue;
        }

        // SAFETY: device is the CUDA-device-0 EGLDeviceEXT; a null attrib
        // list requests the default platform display configuration.
        let display = unsafe {
            (egl.get_platform_display)(
                EGL_PLATFORM_DEVICE_EXT,
                device,
                std::ptr::null(),
            )
        };
        if display == EGL_NO_DISPLAY {
            continue;
        }

        // SAFETY: display is a fresh platform display; null major/minor
        // out pointers are permitted (we don't need the version).
        let init_ok = unsafe {
            (egl.initialize)(display, std::ptr::null_mut(), std::ptr::null_mut())
        };
        if init_ok == EGL_TRUE {
            return Some(display);
        }
    }
    None
}

/// Process-global importer, lazily initialised on first call. `None` means
/// the EGL/CUDA stack isn't usable on this host (no NVIDIA driver, missing
/// EGL device extensions, etc.) and the encoder falls back to CPU upload.
pub(crate) fn importer() -> Option<&'static EglCudaImporter> {
    static IMPORTER: OnceLock<Option<EglCudaImporter>> = OnceLock::new();
    IMPORTER.get_or_init(EglCudaImporter::init).as_ref()
}
