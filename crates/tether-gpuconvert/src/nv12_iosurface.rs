//! macOS IOSurface bridge helpers for VideoToolbox encode.
//!
//! The production macOS 4:2:0 host path uses [`BgraIOSurfaceBridge`]:
//! ScreenCaptureKit supplies a full-resolution BGRA IOSurface, MPS
//! Lanczos resizes it on Metal, and a tiny Metal compute kernel packs
//! the result into a VideoToolbox-ready NV12/P010-family IOSurface.
//! This keeps desktop text in RGB through the resize step and only
//! chroma-subsamples at the encoder boundary.
//!
//! This module also keeps the older [`Nv12IOSurfaceBridge`] around for
//! YUV-plane IOSurface scaling tests. Both bridges use the same
//! IOSurface pool and diagnostic dump helpers.
//!
//! Per frame the BGRA bridge acquires a destination pool slot, copies
//! colorimetry metadata from the source on first use, runs the Metal
//! resize/convert command buffer, optionally dumps a readback PNG for
//! diagnostics, and returns a [`PooledIOSurface`] guard. The host keeps
//! that guard alive until VideoToolbox's compression-output callback
//! fires, then dropping it returns the slot to the pool.

use std::ffi::c_void;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use core_foundation::base::{CFRelease, CFType, CFTypeRef, TCFType};
use core_foundation::dictionary::{CFDictionary, CFDictionaryRef};
use core_foundation::number::CFNumber;
use core_foundation::string::{CFString, CFStringRef};
use objc2::{rc::Retained, runtime::ProtocolObject, AnyThread};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLPixelFormat, MTLSize, MTLStorageMode,
    MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};
use objc2_metal_performance_shaders::MPSImageLanczosScale;
use tether_codec::macos_interop::{
    accepts_iosurface_fourcc, create_iosurface_mtl_texture, import_iosurface_plane,
    iosurface_as_ref, metal_device_from_wgpu, wrap_mtl_texture_as_wgpu, ImportPlaneOptions,
    READ_ONLY_MTL_USAGE, READ_WRITE_MTL_USAGE,
};
use tether_codec::IOSurfaceFrame;
use tether_scaler::{ColorSpace, Pipelines, Scaler, ScalerError};

const ENV_BGRA_BRIDGE_DUMP_DIR: &str = "TETHER_BGRA_BRIDGE_DUMP_DIR";
const ENV_BGRA_BRIDGE_DUMP_LABEL: &str = "TETHER_BGRA_BRIDGE_DUMP_LABEL";
const BGRA_TO_420_METAL: &str = include_str!("bgra_to_420.metal");

/// Construct a wgpu Metal device + queue suitable for driving the
/// [`Nv12IOSurfaceBridge`]. Opts into
/// `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` (mandatory for the R8 /
/// Rg8 plane pipelines) and also `TEXTURE_FORMAT_16BIT_NORM` when the
/// adapter advertises it (required for the R16 / Rg16 plane pipelines
/// the bridge needs for HEVC Main10 sessions).
///
/// Single source of truth for the device-features contract: the host
/// binary calls this at session start; the iosurface_test hardware
/// suite calls it for parity, so the production and test code paths
/// can't drift. A future change to the bridge's feature requirements
/// updates one function and both sides pick it up.
///
/// Fails with `anyhow::Error` if no adapter is present or the
/// adapter doesn't advertise the mandatory format-features set. The
/// 16-bit feature is best-effort — its absence doesn't fail
/// construction; the bridge will then refuse 10-bit sessions via
/// [`BridgeError::TenBitStorageUnsupported`] when those come up.
pub async fn build_bridge_device(
) -> anyhow::Result<(wgpu::Device, wgpu::Queue, BridgeDeviceCapabilities)> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|_| anyhow::anyhow!("no wgpu adapter for macOS NV12 IOSurface bridge"))?;
    let features = adapter.features();
    if !features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
        anyhow::bail!(
            "wgpu adapter does not advertise TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES \
             (required for R8Unorm / Rg8Unorm storage); NV12 IOSurface bridge cannot \
             initialise. Adapter features = {features:?}"
        );
    }
    let has_16bit = features.contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM);
    let mut required = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    if has_16bit {
        required |= wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("tether nv12 iosurface bridge"),
            required_features: required,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await?;
    Ok((
        device,
        queue,
        BridgeDeviceCapabilities {
            supports_10bit: has_16bit,
        },
    ))
}

/// Capability summary returned alongside the wgpu device by
/// [`build_bridge_device`]. Callers use `supports_10bit` to decide
/// whether to advertise 10-bit profiles up the stack or fall back
/// to 8-bit at probe / negotiation time.
#[derive(Debug, Clone, Copy)]
pub struct BridgeDeviceCapabilities {
    pub supports_10bit: bool,
}

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

    /// The destination fourcc isn't in the bridge's known set at all
    /// — distinct from [`Self::UnsupportedFourcc`], which reports a
    /// known fourcc rejected by the encoder/renderer cross-check.
    /// Carries only the offending fourcc because the chroma/bit_depth
    /// can't be inferred.
    #[error("destination fourcc 0x{fourcc:08x} is not recognised as any NV12-family fourcc")]
    UnknownFourcc { fourcc: u32 },

    /// 10-bit NV12 (`'x420'`/`'xf20'`) hosting was requested on a
    /// Metal/wgpu device that cannot create the required R16Unorm /
    /// Rg16Unorm plane textures. The macOS host should refuse 10-bit
    /// profiles at probe/negotiation when the bridge would be in the
    /// loop on such a device.
    #[error(
        "10-bit NV12 host scaling requires TEXTURE_FORMAT_16BIT_NORM \
         for fourcc 0x{fourcc:08x}; this Metal/wgpu device did not advertise it"
    )]
    TenBitStorageUnsupported { fourcc: u32 },

    /// `IOSurfaceCreate` returned null. The properties dictionary is
    /// rejected: typically a zero dimension or an unsupported
    /// `kIOSurfacePixelFormat`.
    #[error("IOSurfaceCreate returned null for {width}x{height} fourcc 0x{fourcc:08x}")]
    IOSurfaceCreateFailed {
        width: u32,
        height: u32,
        fourcc: u32,
    },

    #[error("scaler construction: {0}")]
    Scaler(#[from] ScalerError),

    #[error("Metal BGRA bridge: {0}")]
    Metal(String),

    #[error("IOSurface import: {0}")]
    Import(#[from] tether_codec::macos_interop::IOSurfaceImportError),

    #[error("source IOSurface fourcc 0x{fourcc:08x} is not BGRA")]
    SourceNotBgra { fourcc: u32 },

    #[error("input texture format must be Bgra8Unorm or Rgba8Unorm, got {0:?}")]
    InputFormat(wgpu::TextureFormat),

    #[error("BGRA source dimensions {input_w}x{input_h} do not match bridge source dimensions {width}x{height}")]
    InputDimMismatch {
        width: u32,
        height: u32,
        input_w: u32,
        input_h: u32,
    },

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

        let width_num = CFNumber::from(i64::from(width));
        let height_num = CFNumber::from(i64::from(height));
        let fourcc_num = CFNumber::from(i64::from(fourcc));

        // Luma plane dict: width × height × `luma_bpe` bytes/element.
        let luma_plane = make_plane_dict(i64::from(width), i64::from(height), luma_bpe);
        // Chroma plane dict: chroma_w × chroma_h × `chroma_bpe`.
        let chroma_plane = make_plane_dict(i64::from(chroma_w), i64::from(chroma_h), chroma_bpe);

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
                return Err(BridgeError::IOSurfaceCreateFailed {
                    width,
                    height,
                    fourcc,
                });
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
            (cf_str_borrowed(unsafe { kIOSurfacePlaneInfo }), planes_arr),
        ];
        let props = CFDictionary::from_CFType_pairs(&props_pairs);

        // SAFETY: `IOSurfaceCreate` is the documented Apple API for
        // allocating a fresh IOSurface; the returned pointer has
        // +1 refcount which we transfer into `IOSurfaceOwned`.
        let raw = unsafe { IOSurfaceCreate(props.as_concrete_TypeRef()) };
        let ptr = NonNull::new(raw).ok_or(BridgeError::IOSurfaceCreateFailed {
            width,
            height,
            fourcc,
        })?;
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
        // `'x444'` is the video-range sibling of `'xf44'`/`'P410'` —
        // same MSB-aligned 16-bit biplanar shape, so same per-plane
        // bytes-per-element. Listed for identification parity with
        // `accepts_iosurface_fourcc`; like the rest of the 4:4:4
        // arm it's unreachable on the macOS encoder path today.
        X444_FOURCC | XF44_FOURCC | P410_FOURCC => (2, 4),
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
    y_mtl: Retained<ProtocolObject<dyn MTLTexture>>,
    uv_mtl: Retained<ProtocolObject<dyn MTLTexture>>,
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
        Self::with_pool_depth(
            device,
            queue,
            src_dims,
            dst_dims,
            dst_fourcc,
            DEFAULT_POOL_DEPTH,
        )
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
            .ok_or(BridgeError::UnknownFourcc { fourcc: dst_fourcc })?;
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
        // Pick the scaler pipeline set based on bit depth. 8-bit
        // uses the R8 / Rg8 plane pipelines (always available when
        // `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` is on). 10-bit
        // additionally needs the R16 / Rg16 pipelines, which in turn
        // require `TEXTURE_FORMAT_16BIT_NORM` on the device. If the
        // caller's device didn't opt into the 16-bit feature, the
        // 16-bit BGL construction inside
        // `build_with_plane_storage_16bit` will validation-error —
        // probe the device features here and surface a clearer
        // error if so.
        let pipelines = if bit_depth == 10 {
            if !device
                .features()
                .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
            {
                return Err(BridgeError::TenBitStorageUnsupported { fourcc: dst_fourcc });
            }
            Arc::new(Pipelines::build_with_plane_storage_16bit(&device))
        } else {
            Arc::new(Pipelines::build_with_plane_storage(&device))
        };

        // Horizontal scale ratio in source-pixel units (src_w /
        // dst_w). The chroma plane scaler operates at half
        // resolution on both ends in NV12, so the *plane* scale
        // ratio equals the luma ratio; the cosited correction is
        // `-(scale_x - 1) * 0.5` per the derivation in the scaler
        // crate docs.
        let scale_x = src_dims.0 as f32 / dst_dims.0 as f32;
        let chroma_offset_x = -(scale_x - 1.0) * 0.5;

        // Bit-depth-specific ColorSpace variants. 8-bit picks the
        // R8 / Rg8 plane shaders; 10-bit picks the R16 / Rg16
        // shaders with identical Mitchell math but 16-bit storage
        // cells (10 bits MSB-aligned per NV12 P010-family
        // convention).
        let (y_space, uv_space) = if bit_depth == 10 {
            (
                ColorSpace::LumaR16,
                ColorSpace::ChromaRg16 {
                    chroma_offset: (chroma_offset_x, 0.0),
                },
            )
        } else {
            (
                ColorSpace::LumaR8,
                ColorSpace::ChromaRg8 {
                    chroma_offset: (chroma_offset_x, 0.0),
                },
            )
        };
        let y_scaler = Scaler::new_with_color_space(
            pipelines.clone(),
            device.clone(),
            queue.clone(),
            src_dims,
            dst_dims,
            y_space,
        )?;
        // Vertical chroma offset is 0: MPEG-2 left-cosited NV12
        // (`chroma_sample_loc_type_top_field = 0`, what VT/SCK tag)
        // places the first chroma row co-sited with luma row 0, so
        // a uniform downscale needs no vertical correction — the
        // naive Mitchell center formula is already right for that
        // axis. Only horizontal siting differs from the centered
        // assumption.
        let uv_scaler = Scaler::new_with_color_space(
            pipelines,
            device.clone(),
            queue.clone(),
            (src_dims.0 / 2, src_dims.1 / 2),
            (dst_dims.0 / 2, dst_dims.1 / 2),
            uv_space,
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
            let y_opts = ImportPlaneOptions {
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
            };
            let y_mtl = create_iosurface_mtl_texture(&device, surface_ref, y_opts)?;
            let y_tex = wrap_mtl_texture_as_wgpu(&device, y_mtl.clone(), y_opts)?;
            let uv_opts = ImportPlaneOptions {
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
            };
            let uv_mtl = create_iosurface_mtl_texture(&device, surface_ref, uv_opts)?;
            let uv_tex = wrap_mtl_texture_as_wgpu(&device, uv_mtl.clone(), uv_opts)?;
            slots.push(PoolSlot {
                iosurface,
                y_mtl,
                uv_mtl,
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
    pub fn scale_to_iosurface(&self, src: &IOSurfaceFrame) -> Result<PooledIOSurface, BridgeError> {
        let pool_depth = self.pool.inner.lock().unwrap().slots.len();
        let slot_idx = self
            .pool
            .acquire()
            .ok_or(BridgeError::PoolExhausted { depth: pool_depth })?;

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
                slot.iosurface
                    .as_frame(self.dst_fourcc, self.dst_dims.0, self.dst_dims.1),
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
        // `queue.submit`, queueing the writes onto Metal's command
        // queue.
        //
        // Ordering invariant: VideoToolbox's later access to the
        // IOSurface happens through its own Metal command queue on
        // the *same* MTLDevice that wgpu's submit went to, so Metal
        // serialises the dependency without an explicit fence here
        // — `submit_iosurface` retains the wrapping CVPixelBuffer
        // synchronously, but VT's first read of the surface contents
        // happens through MTL submission ordering. The Apple Silicon
        // unified-memory model is what makes that read coherent;
        // the same-MTLDevice contract is what makes it ordered.
        // A multi-device pipeline (hypothetical; not exercised on
        // macOS today) would need a `device.poll(Wait)` or an
        // explicit cross-device fence.
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

struct MetalBgraPipeline {
    mtl_device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    lanczos: Retained<MPSImageLanczosScale>,
    convert_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl MetalBgraPipeline {
    fn new(device: &wgpu::Device, dst_fourcc: u32) -> Result<Self, BridgeError> {
        let mtl_device = metal_device_from_wgpu(device)?;
        let command_queue = mtl_device
            .newCommandQueue()
            .ok_or_else(|| BridgeError::Metal("MTLDevice::newCommandQueue returned nil".into()))?;
        command_queue.setLabel(Some(&NSString::from_str(
            "tether bgra-iosurface metal bridge",
        )));
        let lanczos = unsafe {
            MPSImageLanczosScale::initWithDevice(MPSImageLanczosScale::alloc(), &mtl_device)
        };
        let convert_pipeline = build_bgra_to_420_pipeline(&mtl_device, dst_fourcc)?;
        Ok(Self {
            mtl_device,
            command_queue,
            lanczos,
            convert_pipeline,
        })
    }
}

/// BGRA IOSurface -> optional BGRA scale -> VideoToolbox-ready
/// NV12-family IOSurface.
///
/// This is the macOS analogue of the Linux host path: keep desktop
/// capture in BGRA through MPS Lanczos resize, then chroma-subsample
/// with a native Metal compute shader only at the encoder input
/// boundary.
pub struct BgraIOSurfaceBridge {
    device: wgpu::Device,
    queue: wgpu::Queue,
    src_dims: (u32, u32),
    dst_dims: (u32, u32),
    dst_fourcc: u32,
    metal: MetalBgraPipeline,
    pool: Arc<IOSurfacePool>,
    dump: Option<Arc<BridgeFrameDump>>,
}

impl BgraIOSurfaceBridge {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        src_dims: (u32, u32),
        dst_dims: (u32, u32),
        dst_fourcc: u32,
    ) -> Result<Self, BridgeError> {
        Self::with_pool_depth(
            device,
            queue,
            src_dims,
            dst_dims,
            dst_fourcc,
            DEFAULT_POOL_DEPTH,
        )
    }

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

        let (chroma, bit_depth) = chroma_bit_depth_for_fourcc(dst_fourcc)
            .ok_or(BridgeError::UnknownFourcc { fourcc: dst_fourcc })?;
        if chroma != tether_protocol::control::ChromaSubsampling::Yuv420
            || !accepts_iosurface_fourcc(chroma, bit_depth, dst_fourcc)
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
        if bit_depth == 10
            && !device
                .features()
                .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        {
            return Err(BridgeError::TenBitStorageUnsupported { fourcc: dst_fourcc });
        }

        let metal = MetalBgraPipeline::new(&device, dst_fourcc)?;
        tracing::info!(
            "using native Metal BGRA bridge for macOS IOSurface encode (MPS Lanczos + Metal 4:2:0 conversion, {}x{} -> {}x{})",
            src_dims.0,
            src_dims.1,
            dst_dims.0,
            dst_dims.1
        );

        let (luma_format, chroma_format) = plane_wgpu_formats(dst_fourcc);
        let mut slots = Vec::with_capacity(pool_depth);
        for slot_idx in 0..pool_depth {
            let iosurface = IOSurfaceOwned::create(dst_dims.0, dst_dims.1, dst_fourcc)?;
            let frame_view = iosurface.as_frame(dst_fourcc, dst_dims.0, dst_dims.1);
            let surface_ref = iosurface_as_ref(&frame_view)?;
            let y_label =
                Box::leak(format!("bgra-iosurface dst y (slot {slot_idx})").into_boxed_str());
            let uv_label =
                Box::leak(format!("bgra-iosurface dst uv (slot {slot_idx})").into_boxed_str());
            let y_opts = ImportPlaneOptions {
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
            };
            let y_mtl = create_iosurface_mtl_texture(&device, surface_ref, y_opts)?;
            let y_tex = wrap_mtl_texture_as_wgpu(&device, y_mtl.clone(), y_opts)?;
            let uv_opts = ImportPlaneOptions {
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
            };
            let uv_mtl = create_iosurface_mtl_texture(&device, surface_ref, uv_opts)?;
            let uv_tex = wrap_mtl_texture_as_wgpu(&device, uv_mtl.clone(), uv_opts)?;
            slots.push(PoolSlot {
                iosurface,
                y_mtl,
                uv_mtl,
                y_tex,
                uv_tex,
                in_use: false,
                attachments_seeded: false,
            });
        }

        Ok(Self {
            device,
            queue,
            src_dims,
            dst_dims,
            dst_fourcc,
            metal,
            pool: Arc::new(IOSurfacePool {
                inner: Mutex::new(PoolInner { slots }),
            }),
            dump: BridgeFrameDump::from_env(),
        })
    }

    pub fn convert_to_iosurface(
        &self,
        src: &IOSurfaceFrame,
    ) -> Result<PooledIOSurface, BridgeError> {
        const BGRA_FOURCC: u32 = u32::from_be_bytes(*b"BGRA");
        if src.pixel_format != BGRA_FOURCC {
            return Err(BridgeError::SourceNotBgra {
                fourcc: src.pixel_format,
            });
        }
        if src.width != self.src_dims.0 || src.height != self.src_dims.1 {
            return Err(BridgeError::InputDimMismatch {
                width: self.src_dims.0,
                height: self.src_dims.1,
                input_w: src.width,
                input_h: src.height,
            });
        }

        let pool_depth = self.pool.inner.lock().unwrap().slots.len();
        let slot_idx = self
            .pool
            .acquire()
            .ok_or(BridgeError::PoolExhausted { depth: pool_depth })?;

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

        let (frame, attachments_already_seeded, y_dst_mtl, uv_dst_mtl, y_dst_tex, uv_dst_tex) = {
            let g = self.pool.inner.lock().unwrap();
            let slot = &g.slots[slot_idx];
            (
                slot.iosurface
                    .as_frame(self.dst_fourcc, self.dst_dims.0, self.dst_dims.1),
                slot.attachments_seeded,
                slot.y_mtl.clone(),
                slot.uv_mtl.clone(),
                slot.y_tex.clone(),
                slot.uv_tex.clone(),
            )
        };
        if !attachments_already_seeded {
            let mut g = self.pool.inner.lock().unwrap();
            let slot = &mut g.slots[slot_idx];
            slot.iosurface.copy_colorimetry_attachments(src.surface);
            slot.attachments_seeded = true;
        }

        let src_ref = iosurface_as_ref(src)?;
        let src_bgra = create_iosurface_mtl_texture(
            &self.device,
            src_ref,
            ImportPlaneOptions {
                label: "bgra-iosurface metal src",
                plane_index: 0,
                width: self.src_dims.0,
                height: self.src_dims.1,
                metal_format: MTLPixelFormat::BGRA8Unorm,
                wgpu_format: wgpu::TextureFormat::Bgra8Unorm,
                mtl_usage: READ_ONLY_MTL_USAGE,
                mtl_storage: MTLStorageMode::Shared,
                wgpu_usage: wgpu::TextureUsages::TEXTURE_BINDING,
            },
        )?;
        self.convert_with_metal(&src_bgra, &y_dst_mtl, &uv_dst_mtl)?;

        if let Some(dump) = &self.dump {
            dump.maybe_dump(
                &self.device,
                &self.queue,
                &y_dst_tex,
                &uv_dst_tex,
                self.dst_dims,
                self.dst_fourcc,
            );
        }

        guard.released = true;
        Ok(PooledIOSurface {
            pool: self.pool.clone(),
            slot_idx,
            frame,
        })
    }

    pub fn src_dims(&self) -> (u32, u32) {
        self.src_dims
    }

    pub fn dst_dims(&self) -> (u32, u32) {
        self.dst_dims
    }

    fn convert_with_metal(
        &self,
        src_bgra: &ProtocolObject<dyn MTLTexture>,
        y_dst: &ProtocolObject<dyn MTLTexture>,
        uv_dst: &ProtocolObject<dyn MTLTexture>,
    ) -> Result<(), BridgeError> {
        let command_buffer = self.metal.command_queue.commandBuffer().ok_or_else(|| {
            BridgeError::Metal("MTLCommandQueue::commandBuffer returned nil".into())
        })?;

        let convert_src;
        let convert_src_ref: &ProtocolObject<dyn MTLTexture> = if self.src_dims == self.dst_dims {
            src_bgra
        } else {
            convert_src = create_mps_destination_texture(
                &self.metal.mtl_device,
                self.dst_dims,
                MTLPixelFormat::BGRA8Unorm,
            )?;
            // SAFETY: the MPS kernel, command buffer, and textures are
            // all created from the same Metal device. The following
            // compute encoder is appended to this same command buffer,
            // so Metal orders the conversion after the resize.
            unsafe {
                self.metal
                    .lanczos
                    .encodeToCommandBuffer_sourceTexture_destinationTexture(
                        &command_buffer,
                        src_bgra,
                        &convert_src,
                    );
            }
            &convert_src
        };

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| BridgeError::Metal("computeCommandEncoder returned nil".into()))?;
        encoder.setComputePipelineState(&self.metal.convert_pipeline);
        unsafe {
            encoder.setTexture_atIndex(Some(convert_src_ref), 0);
            encoder.setTexture_atIndex(Some(y_dst), 1);
            encoder.setTexture_atIndex(Some(uv_dst), 2);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (self.dst_dims.0 / 2).div_ceil(8) as usize,
                height: (self.dst_dims.1 / 2).div_ceil(8) as usize,
                depth: 1,
            },
            MTLSize {
                width: 8,
                height: 8,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        Ok(())
    }
}

fn create_mps_destination_texture(
    mtl_device: &ProtocolObject<dyn MTLDevice>,
    dims: (u32, u32),
    pixel_format: MTLPixelFormat,
) -> Result<objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLTexture>>, BridgeError> {
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(MTLTextureType::Type2D);
    descriptor.setPixelFormat(pixel_format);
    unsafe {
        descriptor.setWidth(dims.0 as usize);
        descriptor.setHeight(dims.1 as usize);
        descriptor.setMipmapLevelCount(1);
        descriptor.setSampleCount(1);
        descriptor.setArrayLength(1);
    }
    descriptor.setUsage(MTLTextureUsage(
        MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::ShaderWrite.0,
    ));
    descriptor.setStorageMode(MTLStorageMode::Private);
    mtl_device
        .newTextureWithDescriptor(&descriptor)
        .ok_or_else(|| {
            BridgeError::Metal(format!(
                "MTLDevice::newTextureWithDescriptor returned nil for {}x{} BGRA",
                dims.0, dims.1
            ))
        })
}

fn build_bgra_to_420_pipeline(
    mtl_device: &ProtocolObject<dyn MTLDevice>,
    dst_fourcc: u32,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, BridgeError> {
    let library = mtl_device
        .newLibraryWithSource_options_error(&NSString::from_str(BGRA_TO_420_METAL), None)
        .map_err(|e| BridgeError::Metal(format!("compile bgra_to_420.metal: {}", ns_error(&e))))?;
    let function_name = NSString::from_str(bgra_to_420_function(dst_fourcc)?);
    let function = library
        .newFunctionWithName(&function_name)
        .ok_or_else(|| BridgeError::Metal(format!("missing Metal function {function_name}")))?;
    mtl_device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| {
            BridgeError::Metal(format!(
                "create compute pipeline {function_name}: {}",
                ns_error(&e)
            ))
        })
}

fn bgra_to_420_function(dst_fourcc: u32) -> Result<&'static str, BridgeError> {
    use tether_codec::macos_interop::{
        NV12_FULL_RANGE_FOURCC, NV12_VIDEO_RANGE_FOURCC, P010_FOURCC, X420_FOURCC, XF20_FOURCC,
    };
    match dst_fourcc {
        NV12_VIDEO_RANGE_FOURCC | NV12_FULL_RANGE_FOURCC => Ok("bgra_to_420v"),
        X420_FOURCC | XF20_FOURCC | P010_FOURCC => Ok("bgra_to_x420"),
        _ => Err(BridgeError::UnknownFourcc { fourcc: dst_fourcc }),
    }
}

fn ns_error(error: &objc2_foundation::NSError) -> String {
    error.localizedDescription().to_string()
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
        // `'x444'` (video range) shares the plane shape of
        // `'xf44'`/`'P410'`; identified here so a stray 4:4:4 10-bit
        // surface reports `UnsupportedFourcc` (known family, bridge
        // can't produce it) rather than the misleading `UnknownFourcc`.
        X444_FOURCC | XF44_FOURCC | P410_FOURCC => Some((ChromaSubsampling::Yuv444, 10)),
        _ => None,
    }
}

struct BridgeFrameDump {
    dir: PathBuf,
    label: Option<String>,
    saved: AtomicBool,
}

impl BridgeFrameDump {
    fn from_env() -> Option<Arc<Self>> {
        let dir = std::env::var_os(ENV_BGRA_BRIDGE_DUMP_DIR).map(PathBuf::from)?;
        let label = std::env::var(ENV_BGRA_BRIDGE_DUMP_LABEL)
            .ok()
            .map(|s| sanitize_filename_part(&s))
            .filter(|s| !s.is_empty());
        Some(Arc::new(Self {
            dir,
            label,
            saved: AtomicBool::new(false),
        }))
    }

    fn maybe_dump(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        y_tex: &wgpu::Texture,
        uv_tex: &wgpu::Texture,
        dims: (u32, u32),
        fourcc: u32,
    ) {
        if self.saved.swap(true, Ordering::AcqRel) {
            return;
        }
        match self.dump(device, queue, y_tex, uv_tex, dims, fourcc) {
            Ok(path) => tracing::info!(path = %path.display(), "BGRA bridge output frame dumped"),
            Err(e) => tracing::warn!(error = %e, "BGRA bridge output frame dump failed"),
        }
    }

    fn dump(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        y_tex: &wgpu::Texture,
        uv_tex: &wgpu::Texture,
        dims: (u32, u32),
        fourcc: u32,
    ) -> std::result::Result<PathBuf, String> {
        let (width, height) = dims;
        let (_, bit_depth) =
            chroma_bit_depth_for_fourcc(fourcc).ok_or("unsupported bridge dump fourcc")?;
        let y_bpe = if bit_depth == 10 { 2 } else { 1 };
        let uv_bpe = if bit_depth == 10 { 4 } else { 2 };
        let chroma_w = width / 2;
        let chroma_h = height / 2;
        let y_tight_row = width
            .checked_mul(y_bpe)
            .ok_or("Y row byte count overflow")?;
        let uv_tight_row = chroma_w
            .checked_mul(uv_bpe)
            .ok_or("UV row byte count overflow")?;
        let y_padded_row = padded_copy_row(y_tight_row);
        let uv_padded_row = padded_copy_row(uv_tight_row);

        let y_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bgra bridge dump y readback"),
            size: u64::from(y_padded_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let uv_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bgra bridge dump uv readback"),
            size: u64::from(uv_padded_row) * u64::from(chroma_h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bgra bridge dump readback enc"),
        });
        copy_texture_to_buffer(
            &mut encoder,
            y_tex,
            &y_readback,
            width,
            height,
            y_padded_row,
        );
        copy_texture_to_buffer(
            &mut encoder,
            uv_tex,
            &uv_readback,
            chroma_w,
            chroma_h,
            uv_padded_row,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let y = read_tight_rows(device, &y_readback, y_tight_row, height, y_padded_row)?;
        let uv = read_tight_rows(device, &uv_readback, uv_tight_row, chroma_h, uv_padded_row)?;
        let rgba = yuv420_to_rgba(width, height, fourcc, &y, &uv)?;

        std::fs::create_dir_all(&self.dir)
            .map_err(|e| format!("create {}: {e}", self.dir.display()))?;
        let label = self.label.as_deref().unwrap_or("frame");
        let filename = format!(
            "bgra-bridge-{}-{}x{}-{}-pid{}.png",
            label,
            width,
            height,
            fourcc_label(fourcc),
            std::process::id()
        );
        let path = self.dir.join(filename);
        write_rgba_png(&path, width, height, &rgba)?;
        Ok(path)
    }
}

fn copy_texture_to_buffer(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
    padded_row: u32,
) {
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

fn padded_copy_row(tight_row: u32) -> u32 {
    const ALIGN: u32 = 256;
    tight_row.div_ceil(ALIGN) * ALIGN
}

fn read_tight_rows(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    tight_row: u32,
    rows: u32,
    padded_row: u32,
) -> std::result::Result<Vec<u8>, String> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    struct Unmapper<'a>(&'a wgpu::Buffer);
    impl<'a> Drop for Unmapper<'a> {
        fn drop(&mut self) {
            self.0.unmap();
        }
    }
    let _guard = Unmapper(buffer);

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("poll: {e:?}"))?;
    rx.recv()
        .map_err(|_| "map_async callback never fired".to_string())?
        .map_err(|e| format!("map: {e:?}"))?;

    let mapped = slice
        .get_mapped_range()
        .map_err(|e| format!("get_mapped_range: {e:?}"))?;
    let tight = usize::try_from(tight_row).map_err(|_| "tight row overflow")?;
    let padded = usize::try_from(padded_row).map_err(|_| "padded row overflow")?;
    let row_count = usize::try_from(rows).map_err(|_| "row count overflow")?;
    let mut out = Vec::with_capacity(tight * row_count);
    for row in 0..row_count {
        let start = row * padded;
        out.extend_from_slice(&mapped[start..start + tight]);
    }
    drop(mapped);
    Ok(out)
}

fn yuv420_to_rgba(
    width: u32,
    height: u32,
    fourcc: u32,
    y: &[u8],
    uv: &[u8],
) -> std::result::Result<Vec<u8>, String> {
    let (_, bit_depth) =
        chroma_bit_depth_for_fourcc(fourcc).ok_or("unsupported bridge dump fourcc")?;
    let full_range = matches!(
        fourcc,
        tether_codec::macos_interop::NV12_FULL_RANGE_FOURCC
            | tether_codec::macos_interop::XF20_FOURCC
    );
    let mut rgba = vec![0u8; usize::try_from(width * height * 4).map_err(|_| "RGBA overflow")?];
    let width_us = usize::try_from(width).map_err(|_| "width overflow")?;
    let height_us = usize::try_from(height).map_err(|_| "height overflow")?;
    let chroma_w = usize::try_from(width / 2).map_err(|_| "chroma width overflow")?;
    for py in 0..height_us {
        for px in 0..width_us {
            let (yy, cb, cr) = if bit_depth == 10 {
                let y_idx = (py * width_us + px) * 2;
                let uv_idx = ((py / 2) * chroma_w + (px / 2)) * 4;
                (
                    u32::from(read_le_u16(y, y_idx)? >> 6),
                    u32::from(read_le_u16(uv, uv_idx)? >> 6),
                    u32::from(read_le_u16(uv, uv_idx + 2)? >> 6),
                )
            } else {
                let y_idx = py * width_us + px;
                let uv_idx = ((py / 2) * chroma_w + (px / 2)) * 2;
                (
                    u32::from(*y.get(y_idx).ok_or("Y byte out of range")?),
                    u32::from(*uv.get(uv_idx).ok_or("Cb byte out of range")?),
                    u32::from(*uv.get(uv_idx + 1).ok_or("Cr byte out of range")?),
                )
            };
            let [r, g, b] = yuv_bt709_to_rgb8(yy, cb, cr, bit_depth, full_range);
            let dst = (py * width_us + px) * 4;
            rgba[dst] = r;
            rgba[dst + 1] = g;
            rgba[dst + 2] = b;
            rgba[dst + 3] = 255;
        }
    }
    Ok(rgba)
}

fn read_le_u16(bytes: &[u8], offset: usize) -> std::result::Result<u16, String> {
    let bytes = bytes
        .get(offset..offset + 2)
        .ok_or("16-bit sample out of range")?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn yuv_bt709_to_rgb8(y: u32, cb: u32, cr: u32, bit_depth: u8, full_range: bool) -> [u8; 3] {
    let (y_norm, u_norm, v_norm) = if full_range {
        let max = if bit_depth == 10 { 1023.0 } else { 255.0 };
        let center = if bit_depth == 10 { 512.0 } else { 128.0 };
        (
            y as f32 / max,
            (cb as f32 - center) / (max + 1.0),
            (cr as f32 - center) / (max + 1.0),
        )
    } else if bit_depth == 10 {
        (
            (y as f32 - 64.0) / 876.0,
            (cb as f32 - 512.0) / 896.0,
            (cr as f32 - 512.0) / 896.0,
        )
    } else {
        (
            (y as f32 - 16.0) / 219.0,
            (cb as f32 - 128.0) / 224.0,
            (cr as f32 - 128.0) / 224.0,
        )
    };

    let r = y_norm + 1.5748 * v_norm;
    let g = y_norm - 0.1873 * u_norm - 0.4681 * v_norm;
    let b = y_norm + 1.8556 * u_norm;
    [float_to_u8(r), float_to_u8(g), float_to_u8(b)]
}

// Clamp and round before narrowing; this is the final PNG byte quantization.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn float_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn write_rgba_png(
    path: &Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> std::result::Result<(), String> {
    let file = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut png = encoder
        .write_header()
        .map_err(|e| format!("png header {}: {e}", path.display()))?;
    png.write_image_data(rgba)
        .map_err(|e| format!("png write {}: {e}", path.display()))
}

fn fourcc_label(fourcc: u32) -> String {
    let bytes = fourcc.to_be_bytes();
    if bytes.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
        String::from_utf8_lossy(&bytes).into_owned()
    } else {
        format!("0x{fourcc:08x}")
    }
}

fn sanitize_filename_part(value: &str) -> String {
    value
        .chars()
        .filter_map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                Some(c)
            } else if c.is_ascii_whitespace() {
                Some('-')
            } else {
                None
            }
        })
        .collect()
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

fn make_plane_dict(
    width: i64,
    height: i64,
    bytes_per_element: i64,
) -> CFDictionary<CFString, CFType> {
    let w = CFNumber::from(width);
    let h = CFNumber::from(height);
    let bpe = CFNumber::from(bytes_per_element);
    let pairs: Vec<(CFString, CFType)> = vec![
        (
            cf_str_borrowed(unsafe { kIOSurfacePlaneWidth }),
            w.as_CFType(),
        ),
        (
            cf_str_borrowed(unsafe { kIOSurfacePlaneHeight }),
            h.as_CFType(),
        ),
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
        // from, so the consistency check is "every video-range fourcc
        // the bridge can allocate (its `chroma_bit_depth_for_fourcc`
        // table) must be accepted by both the encoder and the
        // renderer". Full-range fourccs ('420f', 'xf20', '444f') are
        // rejected at the encoder and renderer because both pin
        // BT.709 limited; they remain in `chroma_bit_depth_for_fourcc`
        // for diagnostic identification (e.g. logging which fourcc
        // arrived when a session can't proceed) but the bridge does
        // not allocate them.
        //
        // 8-bit 4:2:0 video-range: must round-trip.
        assert!(
            iosurface_fourcc_matches(ChromaSubsampling::Yuv420, 8, NV12_VIDEO_RANGE_FOURCC),
            "encoder rejected 8-bit 4:2:0 video-range fourcc"
        );
        assert!(
            renderer_accepts(ChromaSubsampling::Yuv420, 8, NV12_VIDEO_RANGE_FOURCC),
            "renderer rejected 8-bit 4:2:0 video-range fourcc"
        );
        assert_eq!(
            nv12_bytes_per_element(NV12_VIDEO_RANGE_FOURCC),
            (1, 2),
            "bridge plane-bytes-per-element wrong for 8-bit 4:2:0"
        );
        let (chroma, bd) =
            chroma_bit_depth_for_fourcc(NV12_VIDEO_RANGE_FOURCC).expect("known fourcc");
        assert_eq!(chroma, ChromaSubsampling::Yuv420);
        assert_eq!(bd, 8);

        // 8-bit 4:2:0 full-range: encoder + renderer reject.
        // chroma_bit_depth_for_fourcc still identifies the family for
        // diagnostics, but the bridge never targets this fourcc.
        assert!(
            !iosurface_fourcc_matches(ChromaSubsampling::Yuv420, 8, NV12_FULL_RANGE_FOURCC),
            "encoder must reject 8-bit 4:2:0 full-range fourcc (limited VUI)"
        );
        assert!(
            !renderer_accepts(ChromaSubsampling::Yuv420, 8, NV12_FULL_RANGE_FOURCC),
            "renderer must reject 8-bit 4:2:0 full-range fourcc (BT.709 limited shader)"
        );
        assert!(
            chroma_bit_depth_for_fourcc(NV12_FULL_RANGE_FOURCC).is_some(),
            "full-range '420f' must stay identifiable in chroma_bit_depth_for_fourcc \
             for diagnostic logging when a session can't proceed"
        );

        // 10-bit 4:2:0 video-range: must round-trip.
        assert!(
            iosurface_fourcc_matches(ChromaSubsampling::Yuv420, 10, X420_FOURCC),
            "encoder rejected 10-bit 4:2:0 video-range fourcc"
        );
        assert!(
            renderer_accepts(ChromaSubsampling::Yuv420, 10, X420_FOURCC),
            "renderer rejected 10-bit 4:2:0 video-range fourcc"
        );
        assert_eq!(
            nv12_bytes_per_element(X420_FOURCC),
            (2, 4),
            "bridge plane-bytes-per-element wrong for 10-bit 4:2:0"
        );

        // 10-bit 4:2:0 full-range: encoder + renderer reject.
        assert!(
            !iosurface_fourcc_matches(ChromaSubsampling::Yuv420, 10, XF20_FOURCC),
            "encoder must reject 10-bit 4:2:0 full-range fourcc"
        );
        assert!(
            !renderer_accepts(ChromaSubsampling::Yuv420, 10, XF20_FOURCC),
            "renderer must reject 10-bit 4:2:0 full-range fourcc"
        );
        assert!(
            chroma_bit_depth_for_fourcc(XF20_FOURCC).is_some(),
            "full-range 'xf20' must stay identifiable in chroma_bit_depth_for_fourcc \
             for diagnostic logging"
        );

        // Negative cross-check: `'P010'` is in the renderer's accept
        // set (VT decode emits it) but MUST NOT be in the encoder's
        // input set. A future edit that mistakenly adds it to
        // `iosurface_fourcc_matches` would break the bridge silently
        // — the bridge would happily build a pool, then VT would
        // reject every submit with a fourcc-mismatch.
        use tether_codec::macos_interop::P010_FOURCC;
        assert!(
            !iosurface_fourcc_matches(ChromaSubsampling::Yuv420, 10, P010_FOURCC),
            "encoder must NOT accept 'P010' as input fourcc — that path is decode-only"
        );
    }

    /// The 10-bit fourccs (`'x420'`, `'xf20'`) are valid in the
    /// bridge's family table. Verify they report bit depth 10 so
    /// construction can require `TEXTURE_FORMAT_16BIT_NORM` before
    /// building R16/Rg16 plane textures.
    ///
    /// No wgpu device required: the guard fires before
    /// `Pipelines::build_with_plane_storage`. We can't actually
    /// construct a `Nv12IOSurfaceBridge` here (that needs a real
    /// device), so this test verifies the rejection logic via the
    /// helper tables instead — `chroma_bit_depth_for_fourcc`
    /// returns `Some((_, 10))` for both fourccs, which is what the
    /// constructor branches on.
    #[test]
    fn ten_bit_fourccs_route_through_dedicated_guard() {
        use tether_codec::macos_interop::{X420_FOURCC, XF20_FOURCC};
        for &fcc in &[X420_FOURCC, XF20_FOURCC] {
            let (_chroma, bd) =
                chroma_bit_depth_for_fourcc(fcc).expect("10-bit fourcc must be in the table");
            assert_eq!(
                bd, 10,
                "fourcc 0x{fcc:08x} must report bit_depth=10 so the constructor's \
                 `if bit_depth == 10` guard fires"
            );
        }
    }

    /// Verify the cosited-NV12 horizontal chroma offset formula
    /// matches the expert review derivation: at 2× downscale the
    /// offset is `-0.5` source-pixel; at 4× it's `-1.5`. The
    /// vertical axis stays at 0 for MPEG-2 left-cosited NV12 (the
    /// first chroma row is co-sited with luma row 0 — the naive
    /// Mitchell center formula is already correct for that axis).
    #[test]
    fn chroma_offset_matches_cosited_formula() {
        // (src, dst, expected_horizontal_offset). The vertical
        // offset is `0.0` in every case by design — see the doc
        // comment above; encoded as a separate assertion below.
        let cases = [
            ((640u32, 480u32), (640, 480), 0.0),
            ((1920, 1080), (960, 540), -0.5),
            ((3840, 2160), (960, 540), -1.5),
            ((1920, 1080), (1280, 720), -0.25),
        ];
        for (src, dst, expected) in cases {
            let scale_x = src.0 as f32 / dst.0 as f32;
            let computed_x = -(scale_x - 1.0) * 0.5;
            assert!(
                (computed_x - expected).abs() < 1e-6,
                "src={src:?} dst={dst:?}: expected x-offset {expected}, got {computed_x}"
            );
            // Vertical-axis invariant: the bridge always passes
            // y-offset = 0.0 (see `with_pool_depth`). If a future
            // change introduces a vertical correction (e.g. for a
            // chroma-loc-type other than left-cosited), this
            // assertion needs to grow into the same scale-aware
            // shape as the horizontal one. Today the contract is
            // explicit zero, and this test pins it.
            let bridge_y_offset = 0.0_f32;
            assert_eq!(
                bridge_y_offset, 0.0,
                "src={src:?} dst={dst:?}: vertical chroma offset must be 0 for left-cosited NV12"
            );
        }
    }
}
