//! Host-side probe orchestration. Dispatches per-platform to the
//! backend [`ProfileProbe`] impl and rolls each result up into the
//! crate-level [`crate::ProfileSupport`] verdict.
//!
//! macOS and Windows have a single host encode backend, aliased as
//! [`ActiveProbe`]. Linux has two — NVENC (NVIDIA) and VAAPI — selected at
//! runtime by [`probe_encode`] rather than compile time, because both are
//! compiled in and which one serves a profile depends on the live GPU. The
//! encode dispatch mirrors `tether_codec::build_encoder`: an NVIDIA host uses
//! NVENC exclusively (no VAAPI encode fallback — the default VAAPI device
//! there is the decode-only nvidia-vaapi-driver, whose encoder construction
//! SIGSEGVs), and a non-NVIDIA host uses VAAPI. Matching the live dispatch
//! keeps the advertised capability set from disagreeing with what the session
//! encoder actually picks. Decode is always the platform decoder backend
//! ([`ActiveProbe`]) — an NVENC host decodes its own round-trip fixtures
//! through VAAPI (nvidia-vaapi-driver's decode path).

#[cfg(target_os = "linux")]
pub(crate) mod nvenc;
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

/// Run the host encode probe for `profile`, returning `Ok(())` when a real
/// frame round-trips through the chosen backend.
///
/// macOS: the single VideoToolbox backend. Linux: NVENC exclusively on an
/// NVIDIA host (matching `build_encoder`) — VAAPI is **not** tried there, so
/// a profile NVENC can't serve (AV1 on pre-Ada, 4:4:4 not yet wired) reports
/// unsupported rather than dispatching to the decode-only nvidia-vaapi-driver
/// whose `VaapiEncoder::new` SIGSEGVs. Non-NVIDIA Linux uses VAAPI.
#[cfg(target_os = "macos")]
pub(crate) fn probe_encode(profile: tether_protocol::control::VideoProfile) -> crate::profile_probe::Result<()> {
    use crate::profile_probe::ProfileProbe;
    ActiveProbe::probe_encode(profile)
}

#[cfg(target_os = "linux")]
pub(crate) fn probe_encode(
    profile: tether_protocol::control::VideoProfile,
) -> crate::profile_probe::Result<()> {
    use crate::profile_probe::ProfileProbe;
    if tether_codec::nvenc::nvidia_gpu_present() {
        return nvenc::NvencProbe::probe_encode(profile);
    }
    vaapi::VaapiProbe::probe_encode(profile)
}

/// Run the host decode probe for `profile` against the platform decoder
/// backend ([`ActiveProbe`]). Used by the host round-trip (encode → decode)
/// and by the client decode advertisement. On Linux this is always VAAPI —
/// including on an NVENC host, which has no NVDEC path.
pub(crate) fn probe_decode(
    profile: tether_protocol::control::VideoProfile,
    fixture: &[u8],
) -> crate::profile_probe::Result<()> {
    use crate::profile_probe::ProfileProbe;
    ActiveProbe::probe_decode(profile, fixture)
}
