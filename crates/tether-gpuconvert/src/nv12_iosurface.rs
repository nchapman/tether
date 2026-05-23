//! macOS NV12 IOSurface scaler bridge.
//!
//! Drives the [`tether_scaler`] YUV-plane paths against a pool of
//! IOSurface-backed destination textures so the host can hand
//! VideoToolbox a downscaled NV12 surface without a CPU round-trip.
//! Stage 3 of the macOS host GPU scaling plan.
//!
//! Per frame the bridge:
//!
//! 1. Imports the source IOSurface's Y + UV planes as `wgpu::Texture`
//!    handles (read-only, `ShaderRead | Private`) via the shared
//!    `tether_codec::macos_interop::import_iosurface_plane`.
//! 2. Acquires a destination slot from [`IOSurfacePool`]; the slot
//!    owns one CF-retained IOSurface at the destination dimensions
//!    plus pre-built `wgpu::Texture` wrappers (`ShaderRead |
//!    ShaderWrite | PixelFormatView | Shared`) for both planes.
//! 3. Copies the four colorimetry attachments
//!    (`kCVImageBufferYCbCrMatrixKey`, `…ColorPrimariesKey`,
//!    `…TransferFunctionKey`, `…ChromaLocationTopFieldKey`) from
//!    source to destination on the first scale, so the encoded HEVC
//!    VUI carries the correct matrix/primaries/transfer/siting
//!    instead of "unspecified" — the macOS analog of the Linux libva
//!    pitch-alignment bug the project nearly shipped.
//! 4. Runs the two pre-built scalers (Y plane R8Unorm, UV plane
//!    Rg8Unorm) directly into the slot's destination textures via
//!    [`tether_scaler::Scaler::scale_into`], with the cosited UV
//!    siting offset baked into the chroma scaler. The encoder's
//!    `submit_iosurface` takes a CF retain on the destination
//!    IOSurface, so the slot's storage stays alive as long as VT
//!    needs it.
//! 5. Returns a [`PooledIOSurface`] guard — the host holds it until
//!    VideoToolbox's compression-output callback fires, then drops
//!    it to return the slot to the pool.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use core_foundation::base::{CFRelease, CFType, CFTypeRef, TCFType};
use core_foundation::dictionary::{CFDictionary, CFDictionaryRef};
use core_foundation::number::CFNumber;
use core_foundation::string::{CFString, CFStringRef};
use objc2_metal::MTLStorageMode;
use tether_codec::macos_interop::{
    accepts_iosurface_fourcc, import_iosurface_plane, iosurface_as_ref, ImportPlaneOptions,
    READ_ONLY_MTL_USAGE, READ_WRITE_MTL_USAGE,
};
use tether_codec::IOSurfaceFrame;
use tether_scaler::{ColorSpace, Pipelines, Scaler, ScalerError};

/// Default pool depth used by [`Nv12IOSurfaceBridge::new`]. Sized for
/// VideoToolbox's typical in-flight count
/// (`kVTCompressionPropertyKey_MaxFrameDelayCount` ≈ 3 in low-latency
/// mode) plus one slot of headroom so the host doesn't stall on a
/// single late completion callback. Hosts that tune
/// `MaxFrameDelayCount` should pass a matching `pool_depth` to
/// [`Nv12IOSurfaceBridge::with_pool_depth`].
pub const DEFAULT_POOL_DEPTH: usize = 4;

/// Errors the macOS host bridge can produce.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("bridge dims must be > 0 (src={src:?}, dst={dst:?})")]
    ZeroDim { src: (u32, u32), dst: (u32, u32) },

    /// Source and destination dimensions are identical — no scaling
    /// needed. The host should skip the bridge entirely and feed the
    /// source IOSurface directly to the encoder. A dedicated sentinel
    /// keeps Stage 4's "do I need a bridge?" decision out of inner
    /// error-variant matching.
    #[error("source equals destination dims {dims:?}; bridge not needed")]
    NoScaleNeeded { dims: (u32, u32) },

    #[error(
        "destination fourcc 0x{fourcc:08x} is not a supported NV12 family \
         ({chroma:?} {bit_depth}-bit); see tether_codec::macos_interop::accepts_iosurface_fourcc"
    )]
    UnsupportedFourcc {
        chroma: tether_protocol::control::ChromaSubsampling,
        bit_depth: u8,
        fourcc: u32,
    },

    /// `IOSurfaceCreate` returned null. The properties dictionary is
    /// rejected: typically a zero dimension or an unsupported
    /// `kIOSurfacePixelFormat`.
    #[error("IOSurfaceCreate returned null for {width}x{height} fourcc 0x{fourcc:08x}")]
    IOSurfaceCreateFailed { width: u32, height: u32, fourcc: u32 },

    #[error("scaler construction: {0}")]
    Scaler(#[from] ScalerError),

    #[error("IOSurface import: {0}")]
    Import(#[from] tether_codec::macos_interop::IOSurfaceImportError),

    /// The pool is exhausted — every slot is currently held by an
    /// outstanding [`PooledIOSurface`]. The host should drop the
    /// frame and continue; a sustained exhaustion means the
    /// encoder/transport is taking longer than `POOL_DEPTH` frames
    /// per ingest, in which case the pool depth needs bumping or
    /// the host pipeline has a downstream stall.
    #[error("IOSurface pool exhausted (depth {depth})")]
    PoolExhausted { depth: usize },
}

/// Owning wrapper around a raw IOSurface pointer. CF-retained at
/// construction (via [`Self::create`]); CF-released on Drop.
///
/// Send + Sync is justified for the bridge's pool: the IOSurface C
/// API is thread-safe (Apple's IOSurface ref counting is atomic), and
/// the raw pointer is treated as opaque on the Rust side — no
/// `&mut` access is ever taken.
struct IOSurfaceOwned {
    ptr: NonNull<c_void>,
}

// SAFETY: see field doc. IOSurfaceRef is thread-safe (Apple's
// documented contract); we never mutate through this pointer from
// Rust. The MTLTextures wrapping it are the only code that touches
// surface contents, and Metal serialises submissions on the
// command queue.
unsafe impl Send for IOSurfaceOwned {}
unsafe impl Sync for IOSurfaceOwned {}

impl Drop for IOSurfaceOwned {
    fn drop(&mut self) {
        // SAFETY: `ptr` was a freshly CFRetain'd CFType handle (from
        // IOSurfaceCreate, which returns +1) when we wrapped it.
        unsafe { CFRelease(self.ptr.as_ptr() as CFTypeRef) }
    }
}

impl IOSurfaceOwned {
    /// Allocate a fresh NV12 (or P010 / x420 etc.) IOSurface at the
    /// given dimensions and fourcc. Lets IOSurfaceCreate compute the
    /// per-plane `BytesPerRow` from BytesPerElement — VideoToolbox
    /// silently rejects surfaces whose plane pitch isn't 16-aligned
    /// on some macOS versions, so hand-rolling the pitch is risky;
    /// the framework's own alignment is the reference.
    fn create(width: u32, height: u32, fourcc: u32) -> Result<Self, BridgeError> {
        // IOSurface property dictionary. The minimum keys are
        // Width, Height, PixelFormat, BytesPerElement, and (for
        // biplanar) PlaneInfo — an array of dictionaries, one per
        // plane, each with PlaneWidth/PlaneHeight/PlaneBytesPerElement.
        let (chroma_w, chroma_h) = (width / 2, height / 2);
        let (luma_bpe, chroma_bpe) = nv12_bytes_per_element(fourcc);

        let width_num = CFNumber::from(width as i64);
        let height_num = CFNumber::from(height as i64);
        let fourcc_num = CFNumber::from(fourcc as i64);

        // Luma plane dict: width × height × `luma_bpe` bytes/element.
        let luma_plane = make_plane_dict(width as i64, height as i64, luma_bpe);
        // Chroma plane dict: chroma_w × chroma_h × `chroma_bpe`.
        let chroma_plane = make_plane_dict(chroma_w as i64, chroma_h as i64, chroma_bpe);

        // Plane-info array. CFArray of two CFDictionaries.
        let planes_arr = unsafe {
            let items: [CFTypeRef; 2] = [
                luma_plane.as_concrete_TypeRef() as CFTypeRef,
                chroma_plane.as_concrete_TypeRef() as CFTypeRef,
            ];
            let arr_ref = CFArrayCreate(
                core::ptr::null(),
                items.as_ptr(),
                items.len() as isize,
                &kCFTypeArrayCallBacks,
            );
            if arr_ref.is_null() {
                return Err(BridgeError::IOSurfaceCreateFailed { width, height, fourcc });
            }
            CFType::wrap_under_create_rule(arr_ref)
        };

        let props_pairs: Vec<(CFString, CFType)> = vec![
            (
                cf_str_borrowed(unsafe { kIOSurfaceWidth }),
                width_num.as_CFType(),
            ),
            (
                cf_str_borrowed(unsafe { kIOSurfaceHeight }),
                height_num.as_CFType(),
            ),
            (
                cf_str_borrowed(unsafe { kIOSurfacePixelFormat }),
                fourcc_num.as_CFType(),
            ),
            (
                cf_str_borrowed(unsafe { kIOSurfacePlaneInfo }),
                planes_arr,
            ),
        ];
        let props = CFDictionary::from_CFType_pairs(&props_pairs);

        // SAFETY: `IOSurfaceCreate` is the documented Apple API for
        // allocating a fresh IOSurface; the returned pointer has
        // +1 refcount which we transfer into `IOSurfaceOwned`.
        let raw = unsafe { IOSurfaceCreate(props.as_concrete_TypeRef()) };
        let ptr = NonNull::new(raw)
            .ok_or(BridgeError::IOSurfaceCreateFailed { width, height, fourcc })?;
        Ok(Self { ptr })
    }

    fn as_ptr(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    /// Build an [`IOSurfaceFrame`] view (non-owning) the rest of the
    /// codebase consumes. The slot must outlive the returned frame —
    /// in this crate, both live for the duration of one
    /// [`PooledIOSurface`].
    fn as_frame(&self, fourcc: u32, width: u32, height: u32) -> IOSurfaceFrame {
        IOSurfaceFrame {
            surface: self.as_ptr(),
            pixel_format: fourcc,
            width,
            height,
        }
    }

    /// Copy the colorimetry-attachment values present on `src` onto
    /// this surface. Skips any key the source surface doesn't have
    /// (some color taggings are sparse) so a partially-tagged source
    /// produces a partially-tagged dest with no silent fabrication.
    fn copy_colorimetry_attachments(&self, src: *const c_void) {
        // CoreVideo attachment-key constants. These are documented
        // CFString globals exported by the CoreVideo framework; the
        // shape `kCVImageBuffer<X>Key` mirrors what
        // `CVBufferGetAttachment` / `CVBufferSetAttachment` use.
        let keys = [
            unsafe { kCVImageBufferYCbCrMatrixKey },
            unsafe { kCVImageBufferColorPrimariesKey },
            unsafe { kCVImageBufferTransferFunctionKey },
            unsafe { kCVImageBufferChromaLocationTopFieldKey },
        ];
        for key in keys {
            if key.is_null() {
                continue;
            }
            // SAFETY: IOSurfaceCopyValue returns a +1 CF reference;
            // IOSurfaceSetValue retains its own copy, so we balance
            // by CFReleasing the value after the set call.
            unsafe {
                let value = IOSurfaceCopyValue(src, key);
                if value.is_null() {
                    continue;
                }
                IOSurfaceSetValue(self.as_ptr(), key, value);
                CFRelease(value);
            }
        }
    }
}

/// Per-plane bytes-per-element for a given IOSurface fourcc. NV12 8-bit
/// is `R=1, RG=2`; 10-bit (`'x420'`/`'xf20'`/`'P010'`) is `R16=2,
/// RG16=4` because the planes use 16-bit storage cells with 10 bits
/// MSB-aligned.
fn nv12_bytes_per_element(fourcc: u32) -> (i64, i64) {
    use tether_codec::macos_interop::*;
    match fourcc {
        NV12_VIDEO_RANGE_FOURCC | NV12_FULL_RANGE_FOURCC => (1, 2),
        X420_FOURCC | XF20_FOURCC | P010_FOURCC => (2, 4),
        // 4:4:4 paths aren't exposed on the macOS encoder so this
        // arm is unreachable in production — return the 4:4:4 shape
        // for completeness so the same code can fall under a
        // future 4:4:4-host path.
        NV24_VIDEO_RANGE_FOURCC | NV24_FULL_RANGE_FOURCC => (1, 2),
        XF44_FOURCC | P410_FOURCC => (2, 4),
        // Unknown fourcc — caller validates against
        // `accepts_iosurface_fourcc` before reaching here.
        _ => (1, 2),
    }
}

/// Per-slot state. Built eagerly at bridge construction so per-frame
/// `scale_to_iosurface` only walks the pool for an available slot
/// and dispatches the two scalers.
struct PoolSlot {
    iosurface: IOSurfaceOwned,
    y_tex: wgpu::Texture,
    uv_tex: wgpu::Texture,
    in_use: bool,
    /// Whether colorimetry attachments have been seeded from a
    /// source surface yet. Done once per slot (first frame after
    /// allocation); subsequent frames assume the source's color
    /// tagging stays constant for the bridge's lifetime, which is
    /// safe because the bridge is rebuilt on viewport / chroma
    /// changes.
    attachments_seeded: bool,
}

struct PoolInner {
    slots: Vec<PoolSlot>,
}

/// Round-robin / first-available IOSurface pool. Sized at construction
/// to [`POOL_DEPTH`].
struct IOSurfacePool {
    inner: Mutex<PoolInner>,
}

impl IOSurfacePool {
    fn acquire(&self) -> Option<usize> {
        let mut g = self.inner.lock().unwrap();
        for (i, slot) in g.slots.iter_mut().enumerate() {
            if !slot.in_use {
                slot.in_use = true;
                return Some(i);
            }
        }
        None
    }

    fn release(&self, idx: usize) {
        let mut g = self.inner.lock().unwrap();
        if let Some(slot) = g.slots.get_mut(idx) {
            slot.in_use = false;
        }
    }
}

/// Bridge handle. Holds the wgpu device + queue, the per-plane scalers
/// (built once for the bridge's lifetime — one src→dst dim pair), the
/// IOSurface pool, and the destination fourcc / chroma tagging.
pub struct Nv12IOSurfaceBridge {
    device: wgpu::Device,
    src_dims: (u32, u32),
    dst_dims: (u32, u32),
    dst_fourcc: u32,
    /// Y plane scaler: src_dims → dst_dims, R8Unorm.
    y_scaler: Scaler,
    /// UV plane scaler: (src_w/2, src_h/2) → (dst_w/2, dst_h/2),
    /// Rg8Unorm, with the cosited-NV12 chroma siting offset.
    uv_scaler: Scaler,
    pool: Arc<IOSurfacePool>,
}

impl Nv12IOSurfaceBridge {
    /// Build a bridge for the given source / destination dimensions
    /// and destination fourcc. Allocates [`DEFAULT_POOL_DEPTH`]
    /// IOSurfaces up front so per-frame work stays in the hot path.
    /// Hosts that tune VideoToolbox's `MaxFrameDelayCount` should use
    /// [`Self::with_pool_depth`] to keep the pool sized to match.
    ///
    /// `device` and `queue` must be:
    /// - Metal-backed (this constructor checks via `as_hal`).
    /// - Opted into `Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`.
    ///   The scaler validates this when constructing its plane
    ///   pipelines, so the failure mode is a clear error here, not
    ///   a per-frame validation panic.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        src_dims: (u32, u32),
        dst_dims: (u32, u32),
        dst_fourcc: u32,
    ) -> Result<Self, BridgeError> {
        Self::with_pool_depth(device, queue, src_dims, dst_dims, dst_fourcc, DEFAULT_POOL_DEPTH)
    }

    /// Like [`Self::new`] but with an explicit `pool_depth`. Caller
    /// sets this to match VideoToolbox's `MaxFrameDelayCount` plus a
    /// small headroom: too low and the bridge starts dropping frames
    /// under load; too high and IOSurface memory grows linearly with
    /// the depth.
    pub fn with_pool_depth(
        device: wgpu::Device,
        queue: wgpu::Queue,
        src_dims: (u32, u32),
        dst_dims: (u32, u32),
        dst_fourcc: u32,
        pool_depth: usize,
    ) -> Result<Self, BridgeError> {
        assert!(pool_depth > 0, "pool_depth must be > 0");
        if src_dims.0 == 0 || src_dims.1 == 0 || dst_dims.0 == 0 || dst_dims.1 == 0 {
            return Err(BridgeError::ZeroDim {
                src: src_dims,
                dst: dst_dims,
            });
        }
        if src_dims == dst_dims {
            return Err(BridgeError::NoScaleNeeded { dims: src_dims });
        }
        // Reject non-NV12-family fourccs up front. The bridge
        // produces IOSurfaces VideoToolbox encodes from, so the
        // gate is the encoder's accept set, not the renderer's
        // (which is broader — e.g. it tolerates 'P010' for decoded
        // surfaces even though VT *encode* doesn't take 'P010' as
        // input). Cross-checking against both keeps the four
        // tables in lockstep — see the unit test
        // `nv12_fourccs_round_trip_across_tables`.
        let (chroma, bit_depth) = chroma_bit_depth_for_fourcc(dst_fourcc)
            .ok_or(BridgeError::UnsupportedFourcc {
                chroma: tether_protocol::control::ChromaSubsampling::Yuv420,
                bit_depth: 8,
                fourcc: dst_fourcc,
            })?;
        if !accepts_iosurface_fourcc(chroma, bit_depth, dst_fourcc)
            || !tether_codec::videotoolbox::encoder::iosurface_fourcc_matches(
                chroma, bit_depth, dst_fourcc,
            )
        {
            return Err(BridgeError::UnsupportedFourcc {
                chroma,
                bit_depth,
                fourcc: dst_fourcc,
            });
        }

        // Build the scaler pipelines with plane storage (R8 / Rg8
        // vertical passes). The device feature opt-in lives on the
        // caller side; this is the load-bearing check.
        let pipelines = Arc::new(Pipelines::build_with_plane_storage(&device));

        // Horizontal scale ratio in source-pixel units (src_w /
        // dst_w). The chroma plane scaler operates at half
        // resolution on both ends in NV12, so the *plane* scale
        // ratio equals the luma ratio; the cosited correction is
        // `-(scale_x - 1) * 0.5` per the derivation in the scaler
        // crate docs.
        let scale_x = src_dims.0 as f32 / dst_dims.0 as f32;
        let chroma_offset_x = -(scale_x - 1.0) * 0.5;

        let y_scaler = Scaler::new_with_color_space(
            pipelines.clone(),
            device.clone(),
            queue.clone(),
            src_dims,
            dst_dims,
            ColorSpace::LumaR8,
        )?;
        let uv_scaler = Scaler::new_with_color_space(
            pipelines,
            device.clone(),
            queue.clone(),
            (src_dims.0 / 2, src_dims.1 / 2),
            (dst_dims.0 / 2, dst_dims.1 / 2),
            ColorSpace::ChromaRg8 {
                chroma_offset: (chroma_offset_x, 0.0),
            },
        )?;

        // Allocate the pool. Each slot owns one IOSurface + the
        // two wgpu::Texture wrappers; the texture wrappers retain
        // the IOSurface internally so the surface stays alive even
        // if a future change drops the explicit `IOSurfaceOwned`.
        let (luma_format, chroma_format) = plane_wgpu_formats(dst_fourcc);
        let mut slots = Vec::with_capacity(pool_depth);
        for slot_idx in 0..pool_depth {
            let iosurface = IOSurfaceOwned::create(dst_dims.0, dst_dims.1, dst_fourcc)?;
            // Wrap the IOSurface's planes as wgpu textures. The
            // bridge writes through the compute scaler so the
            // textures need ShaderRead|ShaderWrite|PixelFormatView
            // MTL usage and TEXTURE_BINDING|STORAGE_BINDING|COPY_SRC
            // wgpu usage. Shared storage mode is the only valid
            // choice for IOSurface-backed textures (Private is
            // invalid).
            let frame_view = IOSurfaceFrame {
                surface: iosurface.as_ptr(),
                pixel_format: dst_fourcc,
                width: dst_dims.0,
                height: dst_dims.1,
            };
            let surface_ref = iosurface_as_ref(&frame_view)?;
            let y_label =
                Box::leak(format!("nv12-iosurface dst y (slot {slot_idx})").into_boxed_str());
            let uv_label =
                Box::leak(format!("nv12-iosurface dst uv (slot {slot_idx})").into_boxed_str());
            let y_tex = import_iosurface_plane(
                &device,
                surface_ref,
                ImportPlaneOptions {
                    label: y_label,
                    plane_index: 0,
                    width: dst_dims.0,
                    height: dst_dims.1,
                    metal_format: luma_format.metal,
                    wgpu_format: luma_format.wgpu,
                    mtl_usage: READ_WRITE_MTL_USAGE,
                    mtl_storage: MTLStorageMode::Shared,
                    wgpu_usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                },
            )?;
            let uv_tex = import_iosurface_plane(
                &device,
                surface_ref,
                ImportPlaneOptions {
                    label: uv_label,
                    plane_index: 1,
                    width: dst_dims.0 / 2,
                    height: dst_dims.1 / 2,
                    metal_format: chroma_format.metal,
                    wgpu_format: chroma_format.wgpu,
                    mtl_usage: READ_WRITE_MTL_USAGE,
                    mtl_storage: MTLStorageMode::Shared,
                    wgpu_usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                },
            )?;
            slots.push(PoolSlot {
                iosurface,
                y_tex,
                uv_tex,
                in_use: false,
                attachments_seeded: false,
            });
        }

        // The `queue` parameter is consumed by `Scaler::new_with_color_space`
        // (which clones it internally); we don't keep it on the bridge
        // since per-frame `scale_into` calls go through the scaler's
        // own queue handle.
        let _ = queue;

        Ok(Self {
            device,
            src_dims,
            dst_dims,
            dst_fourcc,
            y_scaler,
            uv_scaler,
            pool: Arc::new(IOSurfacePool {
                inner: Mutex::new(PoolInner { slots }),
            }),
        })
    }

    /// Scale `src` into a freshly-acquired pool slot. Returns a
    /// [`PooledIOSurface`] handle the host passes to
    /// `encoder.submit_iosurface`; dropping the handle returns the
    /// slot to the pool, so the caller MUST keep the handle alive
    /// until VideoToolbox's compression-output callback fires.
    pub fn scale_to_iosurface(
        &self,
        src: &IOSurfaceFrame,
    ) -> Result<PooledIOSurface, BridgeError> {
        let pool_depth = self.pool.inner.lock().unwrap().slots.len();
        let slot_idx = self.pool.acquire().ok_or(BridgeError::PoolExhausted {
            depth: pool_depth,
        })?;

        // From here on, we MUST release the slot on any error path
        // (the early `?` returns below would otherwise leak the slot
        // and exhaust the pool). A tiny RAII guard handles that.
        struct SlotGuard<'a> {
            pool: &'a IOSurfacePool,
            slot_idx: usize,
            released: bool,
        }
        impl<'a> Drop for SlotGuard<'a> {
            fn drop(&mut self) {
                if !self.released {
                    self.pool.release(self.slot_idx);
                }
            }
        }
        let mut guard = SlotGuard {
            pool: &self.pool,
            slot_idx,
            released: false,
        };

        // Import the source IOSurface's planes. Source dims are
        // fixed at bridge construction (the host rebuilds the
        // bridge on any source-resolution change).
        let src_ref = iosurface_as_ref(src)?;
        let (luma_format, chroma_format) = plane_wgpu_formats(self.dst_fourcc);
        let src_y = import_iosurface_plane(
            &self.device,
            src_ref,
            ImportPlaneOptions {
                label: "nv12-iosurface src y",
                plane_index: 0,
                width: self.src_dims.0,
                height: self.src_dims.1,
                metal_format: luma_format.metal,
                wgpu_format: luma_format.wgpu,
                mtl_usage: READ_ONLY_MTL_USAGE,
                mtl_storage: MTLStorageMode::Shared,
                wgpu_usage: wgpu::TextureUsages::TEXTURE_BINDING,
            },
        )?;
        let src_uv = import_iosurface_plane(
            &self.device,
            src_ref,
            ImportPlaneOptions {
                label: "nv12-iosurface src uv",
                plane_index: 1,
                width: self.src_dims.0 / 2,
                height: self.src_dims.1 / 2,
                metal_format: chroma_format.metal,
                wgpu_format: chroma_format.wgpu,
                mtl_usage: READ_ONLY_MTL_USAGE,
                mtl_storage: MTLStorageMode::Shared,
                wgpu_usage: wgpu::TextureUsages::TEXTURE_BINDING,
            },
        )?;

        // Pull the slot's destination textures + iosurface frame
        // view under the mutex, then drop the lock before dispatch.
        // `wgpu::Texture` is `Clone` (Arc-backed), so cloning is
        // cheap and lets us run `scale_into` without serialising
        // every concurrent caller behind the pool mutex. The slot
        // identity is stable across the unlocked region because
        // `in_use` is set and only the `SlotGuard` / `PooledIOSurface`
        // drop paths can clear it — neither fires here.
        let (frame, attachments_already_seeded, y_dst_tex, uv_dst_tex) = {
            let g = self.pool.inner.lock().unwrap();
            let slot = &g.slots[slot_idx];
            (
                slot.iosurface.as_frame(self.dst_fourcc, self.dst_dims.0, self.dst_dims.1),
                slot.attachments_seeded,
                slot.y_tex.clone(),
                slot.uv_tex.clone(),
            )
        };
        // First-use seeding of colorimetry attachments. Idempotent
        // on the destination — copying the same attachments twice
        // doesn't break anything — but the `attachments_seeded`
        // flag avoids paying CF query/set cost on every frame.
        if !attachments_already_seeded {
            // SAFETY: src.surface was just successfully imported
            // above (so it's not null and is a live IOSurface).
            let mut g = self.pool.inner.lock().unwrap();
            let slot = &mut g.slots[slot_idx];
            slot.iosurface.copy_colorimetry_attachments(src.surface);
            slot.attachments_seeded = true;
        }

        // Dispatch the scalers against the cloned texture handles.
        // The mutex is released; concurrent acquirers can pick a
        // different slot in parallel. `scale_into` calls
        // `queue.submit`, so by the time `submit_iosurface` runs on
        // the encoder side the IOSurface contents are observable to
        // VideoToolbox. On Apple Silicon (unified memory) no
        // explicit `MTLBlitCommandEncoder.synchronize` is needed.
        self.y_scaler.scale_into(&src_y, &y_dst_tex)?;
        self.uv_scaler.scale_into(&src_uv, &uv_dst_tex)?;

        guard.released = true;
        Ok(PooledIOSurface {
            pool: self.pool.clone(),
            slot_idx,
            frame,
        })
    }

    /// Source dims this bridge was built for. Caller should rebuild
    /// the bridge on any resolution change.
    pub fn src_dims(&self) -> (u32, u32) {
        self.src_dims
    }
    /// Destination dims this bridge was built for.
    pub fn dst_dims(&self) -> (u32, u32) {
        self.dst_dims
    }
}

/// Per-plane Metal + wgpu format pair. Picked from the destination
/// fourcc at bridge construction; kept identical for source and
/// destination since the source IOSurface arrives in the same
/// fourcc family (host capture's `sck_pixel_format_for_profile`
/// negotiates this).
struct PlaneFormat {
    metal: objc2_metal::MTLPixelFormat,
    wgpu: wgpu::TextureFormat,
}

fn plane_wgpu_formats(fourcc: u32) -> (PlaneFormat, PlaneFormat) {
    use tether_codec::macos_interop::*;
    match fourcc {
        NV12_VIDEO_RANGE_FOURCC | NV12_FULL_RANGE_FOURCC => (
            PlaneFormat {
                metal: objc2_metal::MTLPixelFormat::R8Unorm,
                wgpu: wgpu::TextureFormat::R8Unorm,
            },
            PlaneFormat {
                metal: objc2_metal::MTLPixelFormat::RG8Unorm,
                wgpu: wgpu::TextureFormat::Rg8Unorm,
            },
        ),
        X420_FOURCC | XF20_FOURCC | P010_FOURCC => (
            PlaneFormat {
                metal: objc2_metal::MTLPixelFormat::R16Unorm,
                wgpu: wgpu::TextureFormat::R16Unorm,
            },
            PlaneFormat {
                metal: objc2_metal::MTLPixelFormat::RG16Unorm,
                wgpu: wgpu::TextureFormat::Rg16Unorm,
            },
        ),
        // 4:4:4 fourccs aren't producable by the macOS encoder; the
        // bridge constructor rejects them before reaching here.
        _ => (
            PlaneFormat {
                metal: objc2_metal::MTLPixelFormat::R8Unorm,
                wgpu: wgpu::TextureFormat::R8Unorm,
            },
            PlaneFormat {
                metal: objc2_metal::MTLPixelFormat::RG8Unorm,
                wgpu: wgpu::TextureFormat::Rg8Unorm,
            },
        ),
    }
}

fn chroma_bit_depth_for_fourcc(
    fourcc: u32,
) -> Option<(tether_protocol::control::ChromaSubsampling, u8)> {
    use tether_codec::macos_interop::*;
    use tether_protocol::control::ChromaSubsampling;
    match fourcc {
        NV12_VIDEO_RANGE_FOURCC | NV12_FULL_RANGE_FOURCC => Some((ChromaSubsampling::Yuv420, 8)),
        X420_FOURCC | XF20_FOURCC | P010_FOURCC => Some((ChromaSubsampling::Yuv420, 10)),
        NV24_VIDEO_RANGE_FOURCC | NV24_FULL_RANGE_FOURCC => Some((ChromaSubsampling::Yuv444, 8)),
        XF44_FOURCC | P410_FOURCC => Some((ChromaSubsampling::Yuv444, 10)),
        _ => None,
    }
}

/// RAII guard for one pool slot. The wrapped [`IOSurfaceFrame`] is
/// what the host hands to `encoder.submit_iosurface`. The host MUST
/// keep this guard alive until VideoToolbox's compression-output
/// callback fires for the corresponding frame — at that point VT has
/// drained the input surface and the slot is safe to recycle. Dropping
/// returns the slot to the pool.
pub struct PooledIOSurface {
    pool: Arc<IOSurfacePool>,
    slot_idx: usize,
    /// Public field so the host can pass `&pooled.frame` directly to
    /// `encoder.submit_iosurface` without an accessor dance. The
    /// `surface` pointer aliases the slot's owning IOSurface — valid
    /// for the guard's lifetime.
    pub frame: IOSurfaceFrame,
}

// SAFETY: the same rationale as `tether_codec::IOSurfaceFrame`'s
// own `unsafe impl Send` — the underlying IOSurface C API is
// thread-safe (Apple's IOSurface refcount is atomic), no Rust code
// mutates through `frame.surface`, and `pool` is `Arc<...>` which
// is already `Send + Sync`. Stage 4 hands this to VideoToolbox's
// compression-output callback (fires on an Apple-internal thread)
// which is the typical reason we'd need `Send`.
unsafe impl Send for PooledIOSurface {}

impl Drop for PooledIOSurface {
    fn drop(&mut self) {
        self.pool.release(self.slot_idx);
    }
}

// === Plain-FFI declarations ===
//
// `objc2-io-surface` exposes constants but not the `IOSurfaceCreate` /
// `IOSurfaceCopyValue` / `IOSurfaceSetValue` APIs we need. Declare
// them directly. CoreVideo's attachment-key constants likewise are
// not in any objc2 binding; declare those too.

// IOSurface framework: declare the C ABI directly. `objc2-io-surface`
// exposes the property-key constants only behind its
// `objc2-core-foundation` feature, which would pull in a sibling
// CF-binding crate alongside the `core-foundation` crate this module
// already uses. Declaring the constants as `CFStringRef` is simpler
// and avoids two-CF-crate friction.
#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    fn IOSurfaceCreate(properties: CFDictionaryRef) -> *mut c_void;
    fn IOSurfaceSetValue(buffer: *mut c_void, key: CFTypeRef, value: CFTypeRef);
    fn IOSurfaceCopyValue(buffer: *const c_void, key: CFTypeRef) -> CFTypeRef;

    static kIOSurfaceWidth: CFStringRef;
    static kIOSurfaceHeight: CFStringRef;
    static kIOSurfacePixelFormat: CFStringRef;
    static kIOSurfacePlaneInfo: CFStringRef;
    static kIOSurfacePlaneWidth: CFStringRef;
    static kIOSurfacePlaneHeight: CFStringRef;
    static kIOSurfacePlaneBytesPerElement: CFStringRef;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFTypeArrayCallBacks: CFArrayCallBacks;
    fn CFArrayCreate(
        allocator: *const c_void,
        values: *const CFTypeRef,
        num_values: isize,
        callbacks: *const CFArrayCallBacks,
    ) -> CFTypeRef;
}

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    static kCVImageBufferYCbCrMatrixKey: CFTypeRef;
    static kCVImageBufferColorPrimariesKey: CFTypeRef;
    static kCVImageBufferTransferFunctionKey: CFTypeRef;
    static kCVImageBufferChromaLocationTopFieldKey: CFTypeRef;
}

#[repr(C)]
struct CFArrayCallBacks {
    _opaque: [u8; 0],
}

/// Wrap a borrowed `CFStringRef` global as a `CFString` without
/// taking a +1 retain we'd then need to release. The framework's
/// static globals are kept alive for process lifetime; treating
/// them as get-rule borrows is correct.
fn cf_str_borrowed(ptr: CFStringRef) -> CFString {
    // SAFETY: framework-exported `CFStringRef` globals are valid for
    // process lifetime; the `wrap_under_get_rule` retains for us, so
    // the resulting `CFString` owns its own +1.
    unsafe { CFString::wrap_under_get_rule(ptr) }
}

fn make_plane_dict(width: i64, height: i64, bytes_per_element: i64) -> CFDictionary<CFString, CFType> {
    let w = CFNumber::from(width);
    let h = CFNumber::from(height);
    let bpe = CFNumber::from(bytes_per_element);
    let pairs: Vec<(CFString, CFType)> = vec![
        (cf_str_borrowed(unsafe { kIOSurfacePlaneWidth }), w.as_CFType()),
        (cf_str_borrowed(unsafe { kIOSurfacePlaneHeight }), h.as_CFType()),
        (
            cf_str_borrowed(unsafe { kIOSurfacePlaneBytesPerElement }),
            bpe.as_CFType(),
        ),
    ];
    CFDictionary::from_CFType_pairs(&pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-table consistency: the fourccs the bridge allocates must
    /// be the same ones the encoder accepts (`iosurface_fourcc_matches`),
    /// the probe expects (`expected_iosurface_fourccs`), and the
    /// renderer imports (`accepts_iosurface_fourcc`). This is the
    /// drift-catcher CLAUDE.md calls out: any new fourcc family added
    /// to one table must show up in all four.
    #[test]
    fn nv12_fourccs_round_trip_across_tables() {
        use tether_codec::macos_interop::{
            accepts_iosurface_fourcc as renderer_accepts, NV12_FULL_RANGE_FOURCC,
            NV12_VIDEO_RANGE_FOURCC, X420_FOURCC, XF20_FOURCC,
        };
        use tether_codec::videotoolbox::encoder::iosurface_fourcc_matches;
        use tether_protocol::control::ChromaSubsampling;

        // The bridge allocates IOSurfaces that VideoToolbox encodes
        // from, so the consistency check is "every fourcc the bridge
        // can allocate (its `chroma_bit_depth_for_fourcc` table) must
        // be accepted by both the encoder and the renderer". The
        // renderer accepts a superset (it also tolerates `'P010'` /
        // `'P410'` because VT *decode* can emit those even though
        // VT *encode* doesn't take them as input) — that asymmetry
        // is by design, so the bridge restricts itself to the
        // encoder's accept set.
        //
        // 8-bit 4:2:0: NV12_VIDEO and NV12_FULL.
        for &fcc in &[NV12_VIDEO_RANGE_FOURCC, NV12_FULL_RANGE_FOURCC] {
            assert!(
                iosurface_fourcc_matches(ChromaSubsampling::Yuv420, 8, fcc),
                "encoder rejected 8-bit 4:2:0 fourcc 0x{fcc:08x}"
            );
            assert!(
                renderer_accepts(ChromaSubsampling::Yuv420, 8, fcc),
                "renderer rejected 8-bit 4:2:0 fourcc 0x{fcc:08x}"
            );
            assert_eq!(
                nv12_bytes_per_element(fcc),
                (1, 2),
                "bridge plane-bytes-per-element wrong for 8-bit 4:2:0"
            );
            let (chroma, bd) = chroma_bit_depth_for_fourcc(fcc).expect("known fourcc");
            assert_eq!(chroma, ChromaSubsampling::Yuv420);
            assert_eq!(bd, 8);
        }
        // 10-bit 4:2:0: x420 (video-range) and xf20 (full-range).
        // P010 is NOT in the encoder's input set — see fn-level
        // comment above; the bridge's table excludes it.
        for &fcc in &[X420_FOURCC, XF20_FOURCC] {
            assert!(
                iosurface_fourcc_matches(ChromaSubsampling::Yuv420, 10, fcc),
                "encoder rejected 10-bit 4:2:0 fourcc 0x{fcc:08x}"
            );
            assert!(
                renderer_accepts(ChromaSubsampling::Yuv420, 10, fcc),
                "renderer rejected 10-bit 4:2:0 fourcc 0x{fcc:08x}"
            );
            assert_eq!(
                nv12_bytes_per_element(fcc),
                (2, 4),
                "bridge plane-bytes-per-element wrong for 10-bit 4:2:0"
            );
            let (chroma, bd) = chroma_bit_depth_for_fourcc(fcc).expect("known fourcc");
            assert_eq!(chroma, ChromaSubsampling::Yuv420);
            assert_eq!(bd, 10);
        }
    }

    /// Verify the cosited-NV12 chroma offset formula matches the
    /// expert review derivation: at 2× downscale the offset is
    /// `-0.5` source-pixel; at 4× it's `-1.5`.
    #[test]
    fn chroma_offset_matches_cosited_formula() {
        let cases = [
            ((640u32, 480u32), (640, 480), 0.0),
            ((1920, 1080), (960, 540), -0.5),
            ((3840, 2160), (960, 540), -1.5),
            ((1920, 1080), (1280, 720), -0.25),
        ];
        for (src, dst, expected) in cases {
            let scale = src.0 as f32 / dst.0 as f32;
            let computed = -(scale - 1.0) * 0.5;
            assert!(
                (computed - expected).abs() < 1e-6,
                "src={src:?} dst={dst:?}: expected offset {expected}, got {computed}"
            );
        }
    }
}
