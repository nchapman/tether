//! D3D11 (Windows) implementation of the [`ProfileProbe`] contract.
//!
//! Windows implements only `probe_decode` — `construct decoder, submit
//! fixture IDR, poll for a frame back`. There is no `probe_encode`: the
//! AMF single-session limit makes a destructive per-profile encode probe
//! unsafe, so `probe_host` reports encode support statically and the
//! trait drops the method on Windows.

use tether_codec::d3d11::{expected_decode_dxgi_format, D3D11Decoder};
use tether_codec::Decoder;
use tether_protocol::control::VideoProfile;

use crate::profile_probe::{ProbeError, ProfileProbe, Result};
use crate::PipelineStage;

pub(crate) struct D3D11Probe;

impl ProfileProbe for D3D11Probe {
    fn probe_decode(profile: VideoProfile, fixture: &[u8]) -> Result<()> {
        tether_codec::av_log::with_probe_suppression(|| probe_decode_inner(profile, fixture))
    }
}

fn probe_decode_inner(profile: VideoProfile, fixture: &[u8]) -> Result<()> {
    // Gate on a renderer-importable output format BEFORE the live decode. The
    // D3D11VA codec can decode more than the native renderer can display — Arc
    // hardware-decodes HEVC 4:4:4 to a 4:4:4 surface, but the renderer imports
    // only NV12 / P010 (`expected_decode_dxgi_format`). Without this gate the
    // probe would report 4:4:4 decode "Supported" (a frame came back), the
    // client would advertise it, and a host that *can* encode HEVC 4:4:4 (top
    // of PROFILE_PREFERENCE) would pick it — the client then decodes but can't
    // render, breaking the session. Refusing here keeps advertised decode caps
    // a subset of what the renderer accepts, which is the contract the
    // `decoder_output_is_subset_of_renderer_accept` test assumes but can't
    // enforce against a *live* probe.
    if expected_decode_dxgi_format(profile).is_none() {
        return Err(ProbeError::new(
            PipelineStage::Decode,
            "no renderer-importable surface format (D3D11 renderer is 4:2:0 NV12/P010 only)",
        ));
    }

    // Probe with `gpu_export = true` so it exercises the production export
    // path (`export_gpu_frame`), not the 8-bit-only CPU download. This is
    // load-bearing for 10-bit: a Main10 fixture decodes to a P010 surface,
    // which the CPU path can't represent (it would fail) — only the GPU
    // staging export handles P010. The export is self-contained in the
    // decoder's own D3D11 device; no renderer is needed here.
    let mut dec = D3D11Decoder::new(profile.codec, true)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Construct, e))?;
    dec.submit(fixture)
        .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
    // D3D11VA buffers a single IDR until either another packet or EOF
    // arrives (same as VideoToolbox). Signal EOF to drain.
    dec.signal_eof()
        .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?;
    // A decoded frame (now a shared-handle `Frame::Gpu`) means the codec +
    // export path work for this profile.
    //
    // Render-acceptance is enforced by the `expected_decode_dxgi_format` gate
    // above (only NV12 / P010 reach this point) rather than by consulting the
    // renderer directly: that predicate lives in `tether-render`, which
    // `tether-probe` must not depend on (client-only crate; the edge would be
    // a cycle). The pure-logic `decoder_output_is_subset_of_renderer_accept`
    // test (apps/tether-client) proves `expected_decode_dxgi_format`'s set ⊆
    // the renderer's `decode_plane_srv_formats`, so the gate is sound.
    for _ in 0..4 {
        match dec
            .next_frame()
            .map_err(|e| ProbeError::from_codec(PipelineStage::Decode, e))?
        {
            Some(_) => return Ok(()),
            None => continue,
        }
    }
    Err(ProbeError::new(
        PipelineStage::Decode,
        "decoder produced no frames after submit + eof + 4 polls",
    ))
}
