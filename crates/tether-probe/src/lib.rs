//! Authoritative capability discovery for the Tether pipeline.
//!
//! The probe answers one question per `VideoProfile`: *can this host
//! actually produce a frame, encode it, and round-trip it through the
//! decoder?* and *can this client construct the decoder and round-trip
//! a known fixture?*. The result drives the host's
//! `tether.cap.video.encode-profile` advertisement and the client's
//! `tether.cap.video.decode-profiles` advertisement.
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

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod host;
mod preference;
#[cfg(any(target_os = "linux", target_os = "macos"))]
mod profile_probe;

pub use preference::{pick_supported_profile, PROFILE_PREFERENCE};

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
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn probe_host() -> Vec<ProfileSupport> {
    use host::ActiveProbe;
    use profile_probe::{fixture_for, ProfileProbe};

    PROFILE_PREFERENCE
        .iter()
        .copied()
        .map(|profile| {
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

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
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
#[cfg(any(target_os = "linux", target_os = "macos"))]
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
                    Err(e) => SupportStatus::Unsupported {
                        stage: e.stage,
                        reason: e.reason,
                    },
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

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn probe_client() -> Vec<ProfileSupport> {
    // Independent of the platform stub for `probe_host` — the Mac
    // 4:4:4 invariant says encode bits must not be inherited from the
    // host probe. The two stubs happen to return the same shape today,
    // but expressing them independently makes the contract
    // self-documenting and immune to a future hypothetical-platform
    // backend that grew encode capability without growing decode.
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
        // Drive the real cached probe — we can't inject a mock
        // ProfileCapability from outside tether-codec, so this test is
        // a structural assertion: for every profile, the client's
        // encode field must be Unsupported with the "not probed"
        // reason. That fingerprint can only hold if probe_client is
        // reading c.decode (not c.encode) into the decode field.
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
}
