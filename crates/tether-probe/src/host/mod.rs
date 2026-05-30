//! Host-side probe orchestration. Dispatches per-platform to the
//! backend [`ProfileProbe`] impl and rolls each result up into the
//! crate-level [`ProfileSupport`] verdict.

#[cfg(target_os = "linux")]
pub(crate) mod vaapi;

#[cfg(target_os = "macos")]
pub(crate) mod videotoolbox;

#[cfg(target_os = "windows")]
pub(crate) mod d3d11;

#[cfg(target_os = "linux")]
pub(crate) use vaapi::VaapiProbe as ActiveProbe;

#[cfg(target_os = "macos")]
pub(crate) use videotoolbox::VideoToolboxProbe as ActiveProbe;

#[cfg(target_os = "windows")]
pub(crate) use d3d11::D3D11Probe as ActiveProbe;
