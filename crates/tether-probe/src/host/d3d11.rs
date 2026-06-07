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

#[cfg(test)]
mod decode_pixel_tests {
    use tether_codec::d3d11::{D3D11Decoder, DecodedPlanes};
    use tether_codec::Decoder;
    use tether_protocol::control::{ChromaSubsampling, CodecKind, VideoProfile};

    use crate::profile_probe::fixture_for;

    /// Read one 10-bit sample (P010 little-endian, MSB-aligned) as its
    /// 10-bit code, or an 8-bit sample directly.
    fn sample(plane: &[u8], idx: usize, bps: usize) -> u32 {
        if bps == 1 {
            u32::from(plane[idx])
        } else {
            let lo = u32::from(plane[idx * 2]);
            let hi = u32::from(plane[idx * 2 + 1]);
            ((lo | (hi << 8)) >> 6) & 0x3ff
        }
    }

    /// `(min, max, mean)` over a plane of single-sample texels.
    fn luma_stats(p: &DecodedPlanes) -> (u32, u32, f64) {
        let n = (p.width * p.height) as usize;
        let (mut min, mut max, mut sum) = (u32::MAX, 0u32, 0u64);
        for i in 0..n {
            let v = sample(&p.y, i, p.bytes_per_sample);
            min = min.min(v);
            max = max.max(v);
            sum += u64::from(v);
        }
        (min, max, sum as f64 / n as f64)
    }

    /// `(cb_mean, cr_mean, c_min, c_max)` over the interleaved chroma plane.
    fn chroma_stats(p: &DecodedPlanes) -> (f64, f64, u32, u32) {
        let chroma_w = (p.width as usize).div_ceil(2);
        let chroma_h = (p.height as usize).div_ceil(2);
        let n = chroma_w * chroma_h;
        let (mut cb_sum, mut cr_sum) = (0u64, 0u64);
        let (mut min, mut max) = (u32::MAX, 0u32);
        for i in 0..n {
            let cb = sample(&p.uv, i * 2, p.bytes_per_sample);
            let cr = sample(&p.uv, i * 2 + 1, p.bytes_per_sample);
            cb_sum += u64::from(cb);
            cr_sum += u64::from(cr);
            for v in [cb, cr] {
                min = min.min(v);
                max = max.max(v);
            }
        }
        (cb_sum as f64 / n as f64, cr_sum as f64 / n as f64, min, max)
    }

    fn decode_fixture_planes(profile: VideoProfile) -> Option<DecodedPlanes> {
        let fixture = fixture_for(profile).expect("fixture shipped");
        // `next_frame_planes` downloads directly (independent of gpu_export),
        // so no GPU export/fence is needed for this decode-correctness check.
        let mut dec = D3D11Decoder::new(profile.codec, false).expect("decoder construction");
        dec.submit(fixture).expect("submit fixture");
        dec.signal_eof().expect("signal_eof");
        for _ in 0..8 {
            if let Some(planes) = dec.next_frame_planes().expect("next_frame_planes") {
                return Some(planes);
            }
        }
        None
    }

    /// Decode each checked-in 4:2:0 fixture and assert the pixels are
    /// coherent: the fixtures are a flat 256×256 grey, so a correct decode is
    /// near-uniform luma with neutral chroma. This validates that the GPU's
    /// hardware decoder *produces correct pixels* for the profile, not merely
    /// that a frame came back — the gap that let AMD's AV1 decode (which the
    /// plain probe accepted) reach negotiation. Runs on any D3D11VA GPU,
    /// including AMD, where the Intel-QSV-gated render roundtrips skip.
    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows); run with: cargo test -p tether-probe -- --ignored"]
    fn fixtures_decode_to_coherent_grey() {
        let profiles = [
            (CodecKind::H264, 8),
            (CodecKind::Hevc, 8),
            (CodecKind::Hevc, 10),
            (CodecKind::Av1, 8),
            (CodecKind::Av1, 10),
        ];
        for (codec, bit_depth) in profiles {
            let profile = VideoProfile {
                codec,
                chroma: ChromaSubsampling::Yuv420,
                bit_depth,
            };
            let Some(planes) = decode_fixture_planes(profile) else {
                // Profile not decodable on this GPU (e.g. no AV1 decode) —
                // not this test's concern; the probe gates advertisement.
                eprintln!("SKIP {codec:?} {bit_depth}-bit: no frame decoded on this GPU");
                continue;
            };
            assert_coherent_grey(&planes, &format!("{codec:?} {bit_depth}-bit"));
        }
    }

    /// Assert a decoded flat-grey fixture is near-uniform with neutral
    /// chroma. Full-scale for the bit depth: 8-bit → 255, 10-bit → 1023.
    /// Grey is ~mid luma and chroma at the neutral center; the AV1 garbling
    /// blows luma range wide open and pulls chroma far off neutral (the
    /// green-block / black-band corruption).
    fn assert_coherent_grey(p: &DecodedPlanes, label: &str) {
        let full_scale: u32 = if p.bytes_per_sample == 1 { 255 } else { 1023 };
        let center = full_scale.div_ceil(2);
        // Generous tolerances: a flat field after lossy encode is essentially
        // constant (a few codes of quantization), so anything beyond ~10% of
        // full scale is corruption, not codec noise.
        let uniformity_tol = full_scale / 10;
        let neutral_tol = full_scale / 8;

        let (ymin, ymax, ymean) = luma_stats(p);
        assert!(
            ymax - ymin <= uniformity_tol,
            "{label}: luma not uniform (min={ymin} max={ymax} range={} > {uniformity_tol}) — \
             decode corruption",
            ymax - ymin,
        );
        assert!(
            ymean > f64::from(full_scale) * 0.2 && ymean < f64::from(full_scale) * 0.8,
            "{label}: luma mean {ymean:.1} implausible for grey (full scale {full_scale})",
        );

        let (cb, cr, cmin, cmax) = chroma_stats(p);
        assert!(
            cmax - cmin <= uniformity_tol,
            "{label}: chroma not uniform (min={cmin} max={cmax}) — decode corruption",
        );
        let off_cb = (cb - f64::from(center)).abs();
        let off_cr = (cr - f64::from(center)).abs();
        assert!(
            off_cb <= f64::from(neutral_tol) && off_cr <= f64::from(neutral_tol),
            "{label}: chroma not neutral (cb={cb:.1} cr={cr:.1}, center={center}) — \
             grey must decode to neutral chroma; off-neutral is the green-cast corruption",
        );
    }
}
