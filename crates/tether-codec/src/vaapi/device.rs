//! VAAPI render-node selection for Linux multi-GPU hosts.
//!
//! FFmpeg's VAAPI device default is unsafe on NVIDIA systems using
//! `nvidia-vaapi-driver`: the render node exists and advertises decode
//! entrypoints, but encoder/decoder construction can fault inside FFmpeg
//! instead of returning a normal error. Production NVIDIA sessions route
//! through NVENC/NVDEC, but the VAAPI hardware tests construct this backend
//! directly, so direct VAAPI use needs its own guard.

use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use rsmpeg::avutil::AVHWDeviceContext;
use rsmpeg::ffi;

use crate::{CodecError, Result};
use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

const ENV_VAAPI_DEVICE: &str = "TETHER_VAAPI_DEVICE";
const NVIDIA_VENDOR: &str = "0x10de";

pub(super) fn create_hw_device() -> Result<AVHWDeviceContext> {
    let selection = select_vaapi_device(Path::new("/sys/class/drm"), Path::new("/dev/dri"));
    match selection {
        DeviceSelection::Explicit(path) | DeviceSelection::Preferred(path) => {
            let device = cstring_path(&path)?;
            tracing::debug!(device = %path.display(), "opening explicit VAAPI render node");
            Ok(AVHWDeviceContext::create(
                ffi::AV_HWDEVICE_TYPE_VAAPI,
                Some(device.as_c_str()),
                None,
                0,
            )?)
        }
        DeviceSelection::Default => Ok(AVHWDeviceContext::create(
            ffi::AV_HWDEVICE_TYPE_VAAPI,
            None,
            None,
            0,
        )?),
        DeviceSelection::NvidiaOnly => Err(CodecError::NoHardwareCodec(
            "VAAPI is disabled on NVIDIA-only Linux hosts because nvidia-vaapi-driver can fault \
             inside FFmpeg during hardware codec initialization; use NVENC/NVDEC instead"
                .to_string(),
        )),
    }
}

pub(super) fn ensure_encode_entrypoint(
    hw_device: &AVHWDeviceContext,
    profile: VideoProfile,
) -> Result<()> {
    let va_profile = va_profile_for_encode(profile)?;
    let supported = unsafe {
        let buf_ref = hw_device.as_ptr();
        let device_ctx = (*buf_ref).data as *const ffi::AVHWDeviceContext;
        let vaapi_device_ctx = (*device_ctx).hwctx as *const super::ffi::AVVAAPIDeviceContext;
        let display = (*vaapi_device_ctx).display;
        tether_vaapi::supports_encode_entrypoint(display, va_profile)
    }
    .map_err(|e| {
        CodecError::NoHardwareCodec(format!(
            "VAAPI encode capability query failed for {profile:?}: {e}"
        ))
    })?;

    if supported {
        return Ok(());
    }

    Err(CodecError::NoHardwareCodec(format!(
        "VAAPI device does not advertise an encode entrypoint for {profile:?}; \
         refusing before FFmpeg encoder initialization"
    )))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DeviceSelection {
    Explicit(PathBuf),
    Preferred(PathBuf),
    Default,
    NvidiaOnly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RenderNode {
    number: u32,
    vendor: Option<String>,
    dev_path: PathBuf,
}

fn select_vaapi_device(drm_root: &Path, dev_root: &Path) -> DeviceSelection {
    if let Ok(path) = std::env::var(ENV_VAAPI_DEVICE) {
        if !path.trim().is_empty() {
            return DeviceSelection::Explicit(PathBuf::from(path));
        }
    }
    choose_render_node(enumerate_render_nodes(drm_root, dev_root))
}

fn enumerate_render_nodes(drm_root: &Path, dev_root: &Path) -> Vec<RenderNode> {
    let Ok(entries) = std::fs::read_dir(drm_root) else {
        return Vec::new();
    };
    let mut nodes = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(number) = name.to_str().and_then(render_node_number) else {
            continue;
        };
        let dev_path = dev_root.join(name);
        if !dev_path.exists() {
            continue;
        }
        let vendor_path = entry.path().join("device/vendor");
        let vendor = std::fs::read_to_string(vendor_path)
            .ok()
            .map(|s| s.trim().to_ascii_lowercase());
        nodes.push(RenderNode {
            number,
            vendor,
            dev_path,
        });
    }
    nodes
}

fn choose_render_node(mut nodes: Vec<RenderNode>) -> DeviceSelection {
    nodes.sort_by_key(|node| node.number);

    let mut saw_nvidia = false;
    for node in &nodes {
        let Some(vendor) = node.vendor.as_deref() else {
            continue;
        };
        if vendor.eq_ignore_ascii_case(NVIDIA_VENDOR) {
            saw_nvidia = true;
            continue;
        }
        return DeviceSelection::Preferred(node.dev_path.clone());
    }

    if saw_nvidia {
        DeviceSelection::NvidiaOnly
    } else {
        DeviceSelection::Default
    }
}

fn render_node_number(name: &str) -> Option<u32> {
    name.strip_prefix("renderD")?.parse().ok()
}

fn cstring_path(path: &Path) -> Result<CString> {
    CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        CodecError::NoHardwareCodec(format!(
            "{ENV_VAAPI_DEVICE} / VAAPI render-node path contains an interior NUL byte"
        ))
    })
}

fn va_profile_for_encode(profile: VideoProfile) -> Result<tether_vaapi::VAProfile> {
    match (profile.codec, profile.chroma, profile.bit_depth) {
        (CodecKind::H264, ChromaSubsampling::Yuv420, 8) => Ok(tether_vaapi::VA_PROFILE_H264_MAIN),
        (CodecKind::Hevc, ChromaSubsampling::Yuv420, 8) => Ok(tether_vaapi::VA_PROFILE_HEVC_MAIN),
        (CodecKind::Hevc, ChromaSubsampling::Yuv420, 10) => {
            Ok(tether_vaapi::VA_PROFILE_HEVC_MAIN10)
        }
        (CodecKind::Hevc, ChromaSubsampling::Yuv444, 8) => {
            Ok(tether_vaapi::VA_PROFILE_HEVC_MAIN444)
        }
        (CodecKind::Hevc, ChromaSubsampling::Yuv444, 10) => {
            Ok(tether_vaapi::VA_PROFILE_HEVC_MAIN444_10)
        }
        (CodecKind::Av1, ChromaSubsampling::Yuv420, 8 | 10) => {
            Ok(tether_vaapi::VA_PROFILE_AV1_PROFILE0)
        }
        _ => Err(CodecError::UnsupportedInputFormat),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(number: u32, vendor: Option<&str>) -> RenderNode {
        RenderNode {
            number,
            vendor: vendor.map(str::to_string),
            dev_path: PathBuf::from(format!("/dev/dri/renderD{number}")),
        }
    }

    #[test]
    fn prefers_non_nvidia_render_node_on_mixed_hosts() {
        assert_eq!(
            choose_render_node(vec![
                node(130, Some("0x1002")),
                node(128, Some("0x10de")),
                node(129, Some("0x10de")),
            ]),
            DeviceSelection::Preferred(PathBuf::from("/dev/dri/renderD130"))
        );
    }

    #[test]
    fn nvidia_only_is_rejected_before_ffmpeg_device_init() {
        assert_eq!(
            choose_render_node(vec![node(128, Some("0x10de")), node(129, Some("0x10de"))]),
            DeviceSelection::NvidiaOnly
        );
    }

    #[test]
    fn unknown_sysfs_shape_leaves_ffmpeg_default_available() {
        assert_eq!(
            choose_render_node(vec![node(128, None)]),
            DeviceSelection::Default
        );
        assert_eq!(choose_render_node(Vec::new()), DeviceSelection::Default);
    }
}
