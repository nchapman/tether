//! Authoritative capability discovery for the Tether pipeline.
//!
//! The probe answers one question per `VideoProfile`: *can this host
//! actually produce a frame, encode it, and round-trip it through the
//! decoder?* and *can this client construct the decoder and round-trip
//! a known fixture?*. The result drives the host's
//! typed `ServerHello::video` selection and the client's typed
//! `ClientHello::decode_profiles` advertisement.
//!
//! ## Why this crate exists
//!
//! `tether-codec::probe::supported_profiles()` already runs a per-profile
//! round-trip via the `ProfileProbe` trait — but `tether-codec` can't
//! depend on `tether-gpuconvert` (the producer side of the Linux hot
//! path) without inverting the project's dependency graph. So the
//! codec-internal probe is *construction-only* for 10-bit: it builds a
//! P010LE encoder and confirms `avcodec_open2` accepts it, but never
//! feeds the encoder a real dma-buf. The Intel iHD + Mesa + FFmpeg 8.1
//! combination accepts construction and then rejects the actual
//! `av_hwframe_map(DRM_PRIME → VAAPI)` at submit time — so the host
//! used to advertise Main10 it couldn't serve, and clients hit a
//! silent freeze after handshake.
//!
//! We patched it with a second probe layer in `tether-host`
//! (`warm_gpuconvert_capability_cache` + `probe_p010_submit_round_trip`)
//! and a third filter layer (`capture_filtered_encode_profiles`). Three
//! caches, three platform cfg arms — workaround scaffolding for one
//! architectural fact: the probe needs to round-trip the production
//! chain *including the bridge*. The bridge lives one crate up from
//! tether-codec, so the probe orchestration belongs here.
//!
//! After this lands, `host_supported_profiles()` is the single source
//! of truth. Adding a new hardware combo means none of the host's
//! caches change — the round-trip is the answer.
//!
//! ## Public surface
//!
//! - [`host_supported_profiles`] — what the host's encoder pipeline
//!   can actually deliver, end-to-end. Cached for the process lifetime.
//! - [`client_supported_profiles`] — what the client's decoder
//!   pipeline can construct + round-trip. Cached for the process
//!   lifetime.
//! - [`pick_supported_profile`] — host-authoritative negotiation:
//!   first preference-list entry that appears in both sets.
//! - [`ProfileSupport`] / [`SupportStatus`] / [`PipelineStage`] —
//!   structured per-profile diagnostic so a rejection log line names
//!   the exact stage that failed (Capture, EncoderConstruct,
//!   EncoderSubmit, Decoder).

use std::sync::OnceLock;

use tether_protocol::control::VideoProfile;

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
mod host;
mod preference;
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
mod profile_probe;

pub use preference::{
    forced_video_profile_from_env, parse_forced_video_profile, pick_supported_profile,
    pick_supported_profile_with_force, FORCE_VIDEO_PROFILE_ENV, PROFILE_PREFERENCE,
};

/// Per-profile capability verdict. Carries an `encode` and a `decode`
/// status independently because the host cares about encode and the
/// client cares about decode — and on some profiles one half works
/// while the other doesn't (e.g. a Mac client decoding HEVC 4:4:4 10-bit
/// even though VideoToolbox encode for the same profile doesn't exist).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileSupport {
    pub profile: VideoProfile,
    pub encode: SupportStatus,
    pub decode: SupportStatus,
}

impl ProfileSupport {
    /// True iff the encode-side end-to-end round trip succeeded.
    #[must_use]
    pub fn is_encode_supported(&self) -> bool {
        matches!(self.encode, SupportStatus::Supported)
    }

    /// True iff the decode-side fixture round trip succeeded.
    #[must_use]
    pub fn is_decode_supported(&self) -> bool {
        matches!(self.decode, SupportStatus::Supported)
    }
}

/// Per-profile verdict on one side of the pipeline (encode *or*
/// decode). The `Unsupported` variant names which stage in the
/// pipeline rejected the probe + the underlying reason, so a host
/// operator reading the startup log gets "Main10 unsupported because
/// EncoderSubmit: av_hwframe_map rejected P010 dma-buf" instead of a
/// stack of "10-bit encode probe failed" log lines from three layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupportStatus {
    /// Full round trip succeeded. The host/client may advertise this
    /// profile and the session is guaranteed not to fail at that
    /// specific stage of the pipeline at runtime.
    Supported,
    /// Round trip rejected at `stage`. `reason` is free-form English
    /// suitable for logs.
    Unsupported {
        stage: PipelineStage,
        reason: String,
    },
}

impl SupportStatus {
    /// `Unsupported` with no reason text — uncommon, mostly for tests.
    #[must_use]
    pub fn unsupported(stage: PipelineStage, reason: impl Into<String>) -> Self {
        SupportStatus::Unsupported {
            stage,
            reason: reason.into(),
        }
    }
}

/// Identifies which stage of the pipeline rejected a probe. Shared
/// across host (encode) and client (decode) probes — not every variant
/// is reachable on every side:
///
/// | Stage | Host encode probe | Client decode probe |
/// |-------|-------------------|---------------------|
/// | [`Capture`](Self::Capture) | Linux: gpuconvert bridge unbuildable. macOS: SCK can't deliver fourcc. | — |
/// | [`Construct`](Self::Construct) | Encoder construction (`VaapiEncoder::new`/`VtEncoder::new`) failed. | Decoder construction (`VaapiDecoder::new`/`VtDecoder::new`) failed. |
/// | [`Submit`](Self::Submit) | `submit_dmabuf`/`submit_iosurface` rejected a real frame. This catches the Intel iHD P010 gap. | — |
/// | [`Decode`](Self::Decode) | Round-trip decode of the encoder's own output failed. | Fixture submit + `next_frame` couldn't return a `Frame::Gpu`. |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    /// Capture-layer rejection. Linux: gpuconvert bridge can't be
    /// constructed for this profile's chroma/bit-depth (e.g. driver
    /// doesn't advertise R16 storage modifiers for the 10-bit path).
    /// macOS: SCK probe says the pixel format isn't deliverable on this
    /// Mac (`ScreenCaptureKit` refused to make a stream with that
    /// format).
    Capture,
    /// Encoder or decoder constructor returned an error. Most often
    /// "driver doesn't claim this profile at all" — `avcodec_open2`
    /// rejected the parameters, or VT refused the session.
    Construct,
    /// Encoder construction succeeded but a real submit failed. This
    /// is the architecturally important stage: the codec layer alone
    /// can't reach it because submitting needs a real bridge-produced
    /// frame, which is why the codec-internal probe used to be a
    /// half-credit answer.
    Submit,
    /// Decode round-trip failed: either the host's encoder produced a
    /// packet the decoder rejected, or the client's fixture-submit
    /// didn't yield a hardware `Frame::Gpu`.
    Decode,
}

/// Process-lifetime cached host-side capability set. First call runs
/// the full per-profile round trip; subsequent calls return the
/// cached slice without re-probing. Cost is paid once at startup;
/// driver caps don't change at runtime.
///
/// Eager warming: in `tether-host::main()`, call
/// `let _ = tether_probe::host_supported_profiles();` after the server
/// binds but before the listener accepts, so the probe doesn't
/// straddle a handshake's clock-sync measurement.
///
/// Per profile, the probe runs the full production chain end-to-end:
/// capture-stage bridge construction (Linux gpuconvert / macOS SCK
/// capability), encoder construction, real `submit_dmabuf` /
/// `submit_iosurface` of a synthesised frame, and decoder round trip.
/// `SupportStatus::Unsupported` carries a [`PipelineStage`] naming the
/// first stage that rejected the profile.
#[must_use]
pub fn host_supported_profiles() -> &'static [ProfileSupport] {
    static CACHED: OnceLock<Vec<ProfileSupport>> = OnceLock::new();
    CACHED.get_or_init(probe_host).as_slice()
}

/// Process-lifetime cached client-side capability set. Decode probe
/// per profile via the bundled fixtures in `tether-codec`. Same
/// caching semantics as [`host_supported_profiles`].
#[must_use]
pub fn client_supported_profiles() -> &'static [ProfileSupport] {
    static CACHED: OnceLock<Vec<ProfileSupport>> = OnceLock::new();
    CACHED.get_or_init(probe_client).as_slice()
}

/// Convenience filter for the host's advertise step: returns just the
/// `VideoProfile`s where the encoder probe came back `Supported`.
#[must_use]
pub fn host_encode_profiles() -> Vec<VideoProfile> {
    host_supported_profiles()
        .iter()
        .filter(|s| s.is_encode_supported())
        .map(|s| s.profile)
        .collect()
}

/// Convenience filter for the client's advertise step: returns just
/// the `VideoProfile`s where the decoder probe came back `Supported`.
#[must_use]
pub fn client_decode_profiles() -> Vec<VideoProfile> {
    client_supported_profiles()
        .iter()
        .filter(|s| s.is_decode_supported())
        .map(|s| s.profile)
        .collect()
}

/// Run the per-profile encode + decode round trip and roll the
/// results up into [`ProfileSupport`]. The backend's
/// [`profile_probe::ProbeError`] carries the [`PipelineStage`] tag
/// for each rejection, so a log dump names exactly which stage
/// rejected each profile.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn probe_host() -> Vec<ProfileSupport> {
    // Windows reports host encode/decode support statically (see the
    // per-branch comments below), so it never touches the live probe
    // backend or fixtures.
    #[cfg(not(target_os = "windows"))]
    use host::ActiveProbe;
    #[cfg(not(target_os = "windows"))]
    use profile_probe::{fixture_for, ProfileProbe};

    // On Windows/AMF, the encoder has a single-session limit and releases
    // asynchronously on drop. Probe H.264 first so the floor codec always
    // gets a fresh session; higher profiles can fail without breaking
    // negotiation. Probe order doesn't affect negotiation priority (that's
    // PROFILE_PREFERENCE in pick_supported_profile).
    #[cfg(target_os = "windows")]
    let probe_order = {
        let mut order: Vec<_> = PROFILE_PREFERENCE.to_vec();
        // Move H.264 to front.
        if let Some(pos) = order
            .iter()
            .position(|p| p.codec == tether_protocol::control::CodecKind::H264)
        {
            let h264 = order.remove(pos);
            order.insert(0, h264);
        }
        order
    };
    #[cfg(not(target_os = "windows"))]
    let probe_order = PROFILE_PREFERENCE.to_vec();

    probe_order
        .iter()
        .copied()
        .map(|profile| {
            // On Windows/AMF, the deep encode probe is destructive: AMF
            // has a single-session limit and doesn't reliably release
            // sessions on drop, so probing N profiles exhausts AMF for
            // the live encoder. Skip the encode probe and report
            // H.264/HEVC as supported (AMF hardware is known-good if
            // the driver loaded). The live encoder will fail-fast with
            // a clear error if the hardware genuinely can't encode.
            //
            // 4:4:4 is a code-path absence (not a hardware question) that the
            // static probe must still exclude, so negotiation never picks a
            // profile the live encoder would reject at construction and then
            // fail the session: the D3D11 Video Processor only outputs 4:2:0
            // (NV12/P010), so there is no 4:4:4 encode path at all (cf. VAAPI
            // H.264 4:4:4). See `D3D11Encoder::new`.
            //
            // AV1 4:2:0 IS advertised (like HEVC): `backends_for_vendor` now
            // wires `av1_{amf,qsv,nvenc,mf}`. AV1 hardware encode needs a
            // recent GPU (RDNA 3+, Ada, Arc); a card too old fail-fasts in the
            // live encoder, exactly as an AMF card that can't do Main10 would.
            // The client's decode probe is the real per-GPU AV1 gate.
            #[cfg(target_os = "windows")]
            let encode = if profile.chroma == tether_protocol::control::ChromaSubsampling::Yuv444 {
                SupportStatus::Unsupported {
                    stage: PipelineStage::Construct,
                    reason: "D3D11 Video Processor has no 4:4:4 encode path (NV12/P010 only)"
                        .into(),
                }
            } else {
                SupportStatus::Supported
            };
            #[cfg(not(target_os = "windows"))]
            let encode = match ActiveProbe::probe_encode(profile) {
                Ok(()) => SupportStatus::Supported,
                Err(e) => {
                    tracing::debug!(
                        ?profile,
                        stage = ?e.stage,
                        reason = %e.reason,
                        "encode probe rejected profile"
                    );
                    SupportStatus::Unsupported {
                        stage: e.stage,
                        reason: e.reason,
                    }
                }
            };
            // On Windows, skip the *live* decode probe on the host side.
            // Creating multiple D3D11VA decoder contexts triggers
            // DXGI_ERROR_DEVICE_REMOVED on the pre-created capture
            // device (AMD RDNA 4 driver bug). The client runs its own
            // decode probe independently. We still gate the static verdict
            // on a renderer-importable format (a pure, no-hardware check) so
            // 4:4:4 — which the codec can decode but the D3D11 renderer can't
            // display — never reports decode-Supported here either.
            #[cfg(target_os = "windows")]
            let decode = if tether_codec::d3d11::expected_decode_dxgi_format(profile).is_some() {
                SupportStatus::Supported
            } else {
                SupportStatus::Unsupported {
                    stage: PipelineStage::Decode,
                    reason: "D3D11 renderer imports only 4:2:0 NV12/P010".into(),
                }
            };
            #[cfg(not(target_os = "windows"))]
            let decode = match fixture_for(profile) {
                Some(fixture) => match ActiveProbe::probe_decode(profile, fixture) {
                    Ok(()) => SupportStatus::Supported,
                    Err(e) => {
                        tracing::debug!(
                            ?profile,
                            stage = ?e.stage,
                            reason = %e.reason,
                            "decode probe rejected profile"
                        );
                        SupportStatus::Unsupported {
                            stage: e.stage,
                            reason: e.reason,
                        }
                    }
                },
                None => SupportStatus::Unsupported {
                    stage: PipelineStage::Decode,
                    reason: format!(
                        "no decode fixture shipped for {profile:?}; \
                         add a fixture and extend fixture_for"
                    ),
                },
            };
            ProfileSupport {
                profile,
                encode,
                decode,
            }
        })
        .collect()
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn probe_host() -> Vec<ProfileSupport> {
    PROFILE_PREFERENCE
        .iter()
        .copied()
        .map(|profile| ProfileSupport {
            profile,
            encode: SupportStatus::Unsupported {
                stage: PipelineStage::Construct,
                reason: "no hardware backend on this platform".into(),
            },
            decode: SupportStatus::Unsupported {
                stage: PipelineStage::Construct,
                reason: "no hardware backend on this platform".into(),
            },
        })
        .collect()
}

/// Client-side probe. **Does not alias `probe_host`** — the Mac 4:4:4
/// decode asymmetry (M-series silicon decodes HEVC 4:4:4 even though
/// VideoToolbox can't encode it) means encode-false on the host probe
/// must not poison the decode advertisement. So we drive only the
/// decode half of the trait per profile, and report encode as
/// `Unsupported{Construct, "client-side encode not probed"}` — callers
/// asking the client about encode are asking the wrong question.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn probe_client() -> Vec<ProfileSupport> {
    use host::ActiveProbe;
    use profile_probe::{fixture_for, ProfileProbe};

    PROFILE_PREFERENCE
        .iter()
        .copied()
        .map(|profile| {
            let decode = match fixture_for(profile) {
                Some(fixture) => match ActiveProbe::probe_decode(profile, fixture) {
                    Ok(()) => SupportStatus::Supported,
                    Err(e) => {
                        tracing::debug!(
                            ?profile,
                            stage = ?e.stage,
                            reason = %e.reason,
                            "client decode probe rejected profile"
                        );
                        SupportStatus::Unsupported {
                            stage: e.stage,
                            reason: e.reason,
                        }
                    }
                },
                None => SupportStatus::Unsupported {
                    stage: PipelineStage::Decode,
                    reason: format!(
                        "no decode fixture shipped for {profile:?}; \
                         add a fixture and extend fixture_for"
                    ),
                },
            };
            ProfileSupport {
                profile,
                encode: SupportStatus::Unsupported {
                    stage: PipelineStage::Construct,
                    reason: "client-side encode not probed".into(),
                },
                decode,
            }
        })
        .collect()
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn probe_client() -> Vec<ProfileSupport> {
    PROFILE_PREFERENCE
        .iter()
        .copied()
        .map(|profile| ProfileSupport {
            profile,
            encode: SupportStatus::Unsupported {
                stage: PipelineStage::Construct,
                reason: "client-side encode not probed".into(),
            },
            decode: SupportStatus::Unsupported {
                stage: PipelineStage::Construct,
                reason: "no hardware backend on this platform".into(),
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tether_protocol::control::{ChromaSubsampling, CodecKind};

    /// Regression: the D3D11 renderer imports only 4:2:0 (NV12 / P010), so
    /// the Windows probe must never report 4:4:4 *decode* as Supported. The
    /// codec can hardware-decode HEVC 4:4:4 (a frame comes back), but the
    /// renderer can't display it — so advertising it would let a 4:4:4-capable
    /// host (Mac/Linux, where 4:4:4 tops PROFILE_PREFERENCE) negotiate it, and
    /// the client would decode but fail to render. Guards both the host static
    /// verdict and the client live-probe gate (which rejects on format before
    /// constructing a decoder, so this needs no GPU).
    #[cfg(target_os = "windows")]
    #[test]
    fn windows_does_not_advertise_444_decode() {
        use crate::host::d3d11::D3D11Probe;
        use crate::profile_probe::ProfileProbe;
        use tether_protocol::control::VideoProfile;

        // Host static path (pure on Windows — no live decode probe).
        for s in host_supported_profiles() {
            if s.profile.chroma == ChromaSubsampling::Yuv444 {
                assert!(
                    !s.is_decode_supported(),
                    "host probe must not advertise 4:4:4 decode: {:?}",
                    s.profile
                );
            }
        }

        // Client live-probe building block: the format gate rejects 4:4:4
        // before any decoder construction, so no hardware/fixture is needed.
        for profile in [VideoProfile::HEVC_8BIT_444, VideoProfile::HEVC_10BIT_444] {
            let err =
                D3D11Probe::probe_decode(profile, &[]).expect_err("4:4:4 decode must be rejected");
            assert_eq!(err.stage, PipelineStage::Decode);
        }
    }

    /// Pin the public surface: every variant of `PipelineStage` has a
    /// stable name. The match here will fail to compile if a new
    /// variant is added without the test being updated — that's
    /// intentional: any new stage should consciously decide which
    /// reachable-from column it lands in.
    #[test]
    fn pipeline_stage_exhaustive() {
        for stage in [
            PipelineStage::Capture,
            PipelineStage::Construct,
            PipelineStage::Submit,
            PipelineStage::Decode,
        ] {
            let s = SupportStatus::unsupported(stage, "test");
            match s {
                SupportStatus::Supported => unreachable!(),
                SupportStatus::Unsupported { stage: s, .. } => match s {
                    PipelineStage::Capture
                    | PipelineStage::Construct
                    | PipelineStage::Submit
                    | PipelineStage::Decode => {}
                },
            }
        }
    }

    /// Diagnostic dump: run the full per-profile encode/decode probe on
    /// real hardware and print the verdict matrix. Not an assertion —
    /// `mise run probe` runs this with `--nocapture` so an operator can see
    /// which codecs this host can actually build + round-trip, and the
    /// `PipelineStage` reason for any it can't. Ignored by default
    /// because it constructs real encoders/decoders.
    #[test]
    #[ignore = "requires a working host codec backend (VAAPI on Linux, VideoToolbox on macOS, D3D11 on Windows)"]
    fn print_host_supported_profiles() {
        println!("Host codec capability matrix:");
        for support in host_supported_profiles() {
            println!("  {:?}", support.profile);
            println!("    encode: {:?}", support.encode);
            println!("    decode: {:?}", support.decode);
        }
    }

    #[test]
    fn support_status_helpers_match() {
        let supp = ProfileSupport {
            profile: VideoProfile::H264_8BIT_420,
            encode: SupportStatus::Supported,
            decode: SupportStatus::Unsupported {
                stage: PipelineStage::Decode,
                reason: "no fixture".into(),
            },
        };
        assert!(supp.is_encode_supported());
        assert!(!supp.is_decode_supported());
    }

    #[test]
    fn pick_supported_profile_returns_first_match_in_preference_order() {
        let host = vec![
            VideoProfile::H264_8BIT_420,
            VideoProfile::HEVC_8BIT_420,
            VideoProfile::HEVC_8BIT_444,
        ];
        let client = vec![
            VideoProfile::HEVC_8BIT_444,
            VideoProfile::HEVC_8BIT_420,
            VideoProfile::H264_8BIT_420,
        ];
        assert_eq!(
            pick_supported_profile(&host, &client),
            Some(VideoProfile::HEVC_8BIT_444)
        );
    }

    #[test]
    fn pick_supported_profile_returns_none_when_disjoint() {
        let host = vec![VideoProfile::HEVC_8BIT_420];
        let client = vec![VideoProfile {
            codec: CodecKind::Av1,
            chroma: ChromaSubsampling::Yuv420,
            bit_depth: 8,
        }];
        assert_eq!(pick_supported_profile(&host, &client), None);
    }

    /// Pin the Mac 4:4:4 decode asymmetry: `probe_client` must report
    /// encode as not-applicable (Unsupported with a "not probed"
    /// reason), independent of the host-side `c.encode` bit. The bug
    /// this guards against: aliasing `probe_client = probe_host` would
    /// surface a Mac client decoding HEVC 4:4:4 (which works) as
    /// decode-Unsupported because the codec probe's encode bit is
    /// false for that profile on Mac. CLAUDE.md calls out this
    /// asymmetry as an explicit invariant.
    #[test]
    fn probe_client_does_not_mirror_encode_bit_into_decode_field() {
        for support in client_supported_profiles() {
            match &support.encode {
                SupportStatus::Unsupported { reason, .. }
                    if reason == "client-side encode not probed" => {}
                other => panic!(
                    "client probe should mark encode as not-probed for \
                     {:?}; got {other:?}",
                    support.profile
                ),
            }
        }
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn probe_host_includes_h264_encode() {
        let profiles = host_supported_profiles();
        let h264 = profiles.iter().find(|s| {
            s.profile.codec == tether_protocol::control::CodecKind::H264 && s.profile.bit_depth == 8
        });
        assert!(
            h264.is_some(),
            "H.264 8-bit not found in host probe results: {profiles:?}"
        );
        assert!(
            h264.unwrap().is_encode_supported(),
            "H.264 8-bit encode should be supported; got: {:?}",
            h264.unwrap().encode
        );
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn probe_host_includes_hevc_420_encode() {
        let profiles = host_supported_profiles();
        let hevc = profiles.iter().find(|s| {
            s.profile.codec == tether_protocol::control::CodecKind::Hevc
                && s.profile.chroma == tether_protocol::control::ChromaSubsampling::Yuv420
                && s.profile.bit_depth == 8
        });
        assert!(
            hevc.is_some(),
            "HEVC 4:2:0 8-bit not found in host probe results"
        );
        assert!(
            hevc.unwrap().is_encode_supported(),
            "HEVC 4:2:0 8-bit encode should be supported; got: {:?}",
            hevc.unwrap().encode
        );
    }

    /// The D3D11 Video Processor can't output 4:4:4, so the host must
    /// never advertise a 4:4:4 encode profile — negotiation would pick it
    /// and the encoder would silently downsample (or now, reject). On
    /// Windows `probe_host()` is pure logic (encode/decode are not
    /// hardware-probed), so this needs no GPU.
    #[cfg(target_os = "windows")]
    #[test]
    fn windows_host_never_advertises_444_encode() {
        use tether_protocol::control::ChromaSubsampling;
        let profiles = host_supported_profiles();
        for s in profiles {
            if s.profile.chroma == ChromaSubsampling::Yuv444 {
                assert!(
                    !s.is_encode_supported(),
                    "host must not advertise 4:4:4 encode (no VP path); got: {:?}",
                    s.profile
                );
            }
        }
        assert!(
            profiles
                .iter()
                .any(|s| s.profile.chroma == ChromaSubsampling::Yuv420 && s.is_encode_supported()),
            "host should still advertise at least one 4:2:0 encode profile"
        );
    }

    /// D3D11 now wires AV1 encode (`backends_for_vendor` returns
    /// `av1_{amf,qsv,nvenc,mf}`), so the host advertises AV1 4:2:0
    /// statically — like HEVC Main10 — and relies on the live encoder to
    /// fail-fast on a GPU too old to AV1-encode. 4:4:4 AV1 (if any
    /// PROFILE_PREFERENCE ever lists it) must still be excluded, since the
    /// Video Processor has no 4:4:4 path. Pure logic on Windows; no GPU.
    #[cfg(target_os = "windows")]
    #[test]
    fn windows_host_advertises_av1_420_encode() {
        use tether_protocol::control::{ChromaSubsampling, CodecKind};
        let profiles = host_supported_profiles();
        let mut saw_av1_420 = false;
        for s in profiles {
            if s.profile.codec != CodecKind::Av1 {
                continue;
            }
            match s.profile.chroma {
                ChromaSubsampling::Yuv420 => {
                    saw_av1_420 = true;
                    assert!(
                        s.is_encode_supported(),
                        "host should advertise AV1 4:2:0 encode (backend wired); got: {:?}",
                        s.encode
                    );
                }
                ChromaSubsampling::Yuv444 => assert!(
                    !s.is_encode_supported(),
                    "host must not advertise AV1 4:4:4 encode (no VP path); got: {:?}",
                    s.profile
                ),
            }
        }
        assert!(
            saw_av1_420,
            "PROFILE_PREFERENCE should surface AV1 4:2:0 for this test to be meaningful"
        );
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn probe_client_includes_h264_decode() {
        let profiles = client_supported_profiles();
        let h264 = profiles.iter().find(|s| {
            s.profile.codec == tether_protocol::control::CodecKind::H264 && s.profile.bit_depth == 8
        });
        assert!(
            h264.is_some(),
            "H.264 8-bit not found in client probe results"
        );
        assert!(
            h264.unwrap().is_decode_supported(),
            "H.264 8-bit decode should be supported; got: {:?}",
            h264.unwrap().decode
        );
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn probe_client_includes_hevc_420_decode() {
        let profiles = client_supported_profiles();
        let hevc = profiles.iter().find(|s| {
            s.profile.codec == tether_protocol::control::CodecKind::Hevc
                && s.profile.chroma == tether_protocol::control::ChromaSubsampling::Yuv420
                && s.profile.bit_depth == 8
        });
        assert!(
            hevc.is_some(),
            "HEVC 4:2:0 8-bit not found in client probe results"
        );
        assert!(
            hevc.unwrap().is_decode_supported(),
            "HEVC 4:2:0 8-bit decode should be supported; got: {:?}",
            hevc.unwrap().decode
        );
    }

    #[test]
    #[ignore = "requires D3D11VA-capable GPU (Windows)"]
    fn probe_host_and_client_intersect_on_at_least_one_profile() {
        let host = host_encode_profiles();
        let client: Vec<_> = client_supported_profiles()
            .iter()
            .filter(|s| s.is_decode_supported())
            .map(|s| s.profile)
            .collect();
        let intersection = host.iter().any(|p| client.contains(p));
        assert!(
            intersection,
            "host and client must share at least one profile for sessions to work.\n\
             host encode: {host:?}\n\
             client decode: {client:?}"
        );
    }

    /// On a GPU that decodes HEVC Main10, the client must advertise it.
    /// Regression guard for the 10-bit unlock: the Windows decode probe
    /// routes through the production P010 GPU-export path, so a Main10
    /// fixture decodes to P010 and comes back as a `Frame::Gpu`. The old
    /// CPU-download probe forced NV12 on that P010 surface, silently
    /// dropping Main10 from the advert before it could ever negotiate.
    #[cfg(target_os = "windows")]
    #[test]
    #[ignore = "requires D3D11 GPU with HEVC Main10 decode (Windows)"]
    fn client_offers_hevc_main10() {
        let profiles = client_decode_profiles();
        eprintln!("client decode profiles: {profiles:?}");
        assert!(
            profiles.iter().any(|p| p.codec == CodecKind::Hevc
                && p.chroma == ChromaSubsampling::Yuv420
                && p.bit_depth == 10),
            "HEVC Main10 should be offered on a Main10-capable GPU; got {profiles:?}"
        );
    }

    /// On a GPU that decodes AV1, the client must advertise AV1 4:2:0.
    /// Regression guard for the AV1 unlock: the client decode probe runs
    /// the AV1 fixture through `D3D11Decoder::new(Av1, ..)` → the per-GPU
    /// `hw_config` D3D11VA gate → GPU-export path. This is the decode-side
    /// analogue of `client_offers_hevc_main10`, and the only test that
    /// catches a regression silently flipping AV1 back off (e.g. reverting
    /// `d3d11_av_codec_id`). SKIPs implicitly on a GPU without AV1 decode
    /// (the assert would fail there — run only on RDNA 2+ / Ada / Arc).
    #[cfg(target_os = "windows")]
    #[test]
    #[ignore = "requires D3D11 GPU with AV1 decode (RDNA 2+ / Ada / Arc, Windows)"]
    fn client_offers_av1_420() {
        let profiles = client_decode_profiles();
        eprintln!("client decode profiles: {profiles:?}");
        assert!(
            profiles
                .iter()
                .any(|p| p.codec == CodecKind::Av1 && p.chroma == ChromaSubsampling::Yuv420),
            "AV1 4:2:0 should be offered on an AV1-decode-capable GPU; got {profiles:?}"
        );
    }
}
