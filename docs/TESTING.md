# Tether — Testing conventions

The foundation review (Q2 2026) made the test surface deliberately
small and predictable. This document codifies what we have, what
each layer covers, and how to extend it.

## Headline counts

`cargo test --workspace` runs **332 default-on tests** (including
integration tests; lib-only is 255). Hardware-gated `#[ignore]`
tests total **~70** across `tether-codec`, `tether-render`,
`tether-gpuconvert`, `tether-scaler`, and `tether-audio`. Numbers below
are authoritative — `docs/ARCHITECTURE.md` and `CLAUDE.md` defer here.

## Unit and integration tests by crate

| Crate | Tests | Covers |
| --- | --- | --- |
| `tether-protocol` | 85 lib + 2 integration | Wire round-trips for every control variant (handshake, codec negotiation incl. 10-bit `VideoProfile` constants + forward-compat `u8` bit_depth probe, video packets + `stream_epoch>u16` widening, `VideoFrameMetaEnvelope`, cursor position + control-stream cursor shapes, multi-monitor `DisplayList`, stream lifecycle, `ClientStats`, `ControlMessage::Extension`, audio `Opus` incl. near-max-payload datagram-decode, `PixelFormat` hello extension incl. `P010` / `P410`, `InputEvent::device_id`), fragmenter / reassembler invariants (out-of-order, stale eviction, wall-clock-timeout eviction, cross-epoch rejection, duplicate-fragment idempotency, continuation-before-First, unified IDR-keyframe datagram path), `HostFrameTimingBuilder` typestate, forward-compat probes for every tagged enum, clock-sync edges, **receiver-side wire-validation** (`video::validation_tests`: oversized / zero fragment_count, continuation index, shard_size, total_body_len, total_body_len above shard capacity, legitimate-accept, `fragments_lost` bump on reject; constants `MAX_FRAGMENTS_PER_FRAME = 4096`, `MAX_FRAME_BODY_BYTES = MAX_FRAGMENTS_PER_FRAME * MAX_DATAGRAM_PAYLOAD`), **multi-block Reed-Solomon FEC** around `VideoPacket::Parity` (fragment + per-block parity emission + RS recovery under simulated loss, multi-block split for large IDRs, every-datagram-fits-budget across input-echo sizes; `FEC_MAX_PRIMARY_SHARDS = 212` per block). Integration: `tests/fragmenter_property.rs` — 2 proptest cases (256 iterations each) over fragmenter ↔ reassembler under random loss + reorder. |
| `tether-transport` | 10 lib (test-support) + 6 integration (`tests/roundtrip.rs`) | QUIC handshake, control + datagram round-trip, fingerprint pinning, oversized-datagram local reject, video-keyframe-stream round-trip, oversized-keyframe local reject. `test_support` (feature-gated): `DuplexControlChannel` handshake round-trip, post-handshake control message exchange, dropped-peer surfaces `StreamClosed`; `HostHandshake` → `ClientHelloReceived` typestate routes recv-then-send. |
| `tether-codec` | 19 lib + 14 `#[ignore]` | `validate_chosen_profile` accept/reject/empty paths; SW H264 round-trip (test-only); VT `codec_name` map. Hardware: VAAPI encoder/decoder/dma-buf import incl. `hevc_main444_dmabuf_roundtrip` (8-bit XYUV) and `hevc_main444_10bit_xv30_dmabuf_roundtrip` (10-bit XV30); per-codec × per-resolution `vaapi::bench` (4 cells × 3 paths); VideoToolbox encoder, H.264 BGRA round-trip, HEVC 10-bit / 4:4:4 probe matrix, `videotoolbox_round_trip_chroma_matrix` cross-check, self-decodable-IDR mid-session recovery. **New encoder-knob tests** (all use the SKIP-with-diagnostic pattern — see below): `vaapi_intra_refresh_round_trip`, `vaapi_min_qp_floor_reduces_bitstream`, `vaapi_bitrate_retune_changes_bitstream_size`. **Windows D3D11** (cfg-gated to `target_os = "windows"`, so not in the counts above): backend-selection pure-logic tests (`unknown_vendor_falls_back_to_mf_only` — the MF-only fallback that prevents foreign-vendor construction faults; `known_vendors_lead_with_hardware_then_mf`), `d3d11_rejects_444_at_construction_no_silent_downsample`, and the vendor-gated GPU encode→decode round-trips `d3d11_{qsv,amf,nvenc}_gpu_encode_decode_roundtrip` (`#[ignore]`; each gates on the present GPU vendor and SKIPs off-vendor — see SKIP section). |
| `tether-probe` | 10 (+ `#[ignore]` hardware) | `PipelineStage` exhaustiveness, `ProfileSupport` helpers, preference order (5-entry, 10-bit-first), `pick_supported_profile` (best mutual / fallback / disjoint / empty / forward-compat unknown bit_depth), Mac 4:4:4 decode-without-encode invariant. Windows (`#[ignore]`): `client_offers_hevc_main10` — regression guard that the P010 GPU-export decode probe keeps Main10 in the advert. |
| `tether-input` | 27 Linux / 15 macOS-Windows | 11 cross-platform translator tests (modifier tracking, HID routing, cursor normalization). On Linux, `inject/uinput.rs` adds 16 unit tests: HID→evdev mapping, ASCII char→keystroke, scroll-detent sign/quantisation/overflow, normalised→ABS coordinates, pointer-button↔device cross-table, and modifier-reconcile release-variant logic. On macOS/Windows the 4 `clamp_relative_delta` (±1000 px) tests live in `inject/enigo_backend.rs`. |
| `tether-session` | 34 lib + 20 integration | Lib: `IdrSignal` coalescing + clone-share; `EncodeStatsWindow` emit / idle / accumulate; `HostSession::accept` decode-profile-extension parsing (missing / oversize / malformed / well-formed / unknown bit_depth filter / no-match server-hello / chosen-profile echo); `ClientSession::connect` resolve_negotiated_profile (absent → 8-bit synth, present + decodes → authoritative, undecodable → error, unknown → reject); `paced_sender` pacing math + `Arc<AtomicU64>` retune (8 tests). Integration (via `tether-transport`'s `test-support` feature): `loopback.rs` (10) — full handshake, RTT/offset bounds, Goodbye on no-mutual-profile, `ProfileNotAdvertised`, `UnknownBitDepth`, host filters unknown depths, legacy host inline-field synthesis, dropped-peer Transport error; `decoder_thread_loopback.rs` (3) — `run_thread` + `DuplexVideoChannel` + `LatestFrame` smoke; `video_loopback.rs` (2); `video_loopback_with_loss.rs` (3) via `LossyChannel`; `epoch_bump_invalidates_inflight.rs` (1); `input_loopback.rs` (1). |
| `tether-render` | 37 lib + 15 `#[ignore]` (Linux) + 4 `#[ignore]` (macOS) | Cursor letterbox/aspect; `LatestFrame` Send+Sync + drop-oldest; `transfer_kind` / `range_kind_for` / `render_layout_for` dispatch-table pins (incl. 10-bit limited-range breakpoint algebra and Yuv420 10-bit → Biplanar16); `PresentPolicy` / `FrameAgeTracker` (8 tests on pure `decide_present` logic); **relative-mouse sub-pixel accumulator** (6 tests in `relative_mouse.rs`: whole-pixel passthrough, sub-pixel-held, long-run convergence, mixed-sign no-stall, i16 saturation, reset). Hardware: end-to-end roundtrip harness (`test_harness.rs` + `dmabuf_test.rs`) drives the production multi-pass renderer (`Gpu::new_headless`) through 14 cells covering identity / host-scaler / client-upscale / surface-below-video / full-chain / repro-shape across H.264 4:2:0, HEVC Main, HEVC Main 4:4:4, HEVC Main 10, HEVC Main 4:4:4 10-bit. Primary metric is geometric residual on `Fixture::CoordEncoded`; SSIM + BT.709 Y-PSNR are the secondary catch-net. macOS IOSurface zero-copy across HEVC Main / Main10 / Main 4:4:4 8 + 10-bit. **Windows D3D11** (cfg-gated, `#[ignore]`): native-renderer headless roundtrips — synthetic-NV12 gray, `cursor_overlay_composites_over_video` (alpha-blend over video, no blend leak), and `d3d11_coord_fixture_decode_render_roundtrip_{8bit,10bit}` (full QSV encode → D3D11VA decode → shared-handle import → render across NV12 8-bit + P010 10-bit). |
| `tether-gpuconvert` | 6 lib + 18 `#[ignore]` lib + 7 `#[ignore]` integration (`tests/scaler_roundtrip.rs`) | `drm_fourcc_to_vk_format` table coverage (8 + 10-bit biplanar + packed XV30 → A2B10G10R10_UNORM_PACK32 + unknown rejection); BGRA→NV12 + DMA-BUF round-trip; `convert_solid_{white,red}_roundtrip_packed_xv30` (10-bit channel-mapping + BT.709 math); `storable_probe_returns_linear_for_xv30`; structural alignment regression `convert_reports_64_aligned_y_stride_at_unaligned_width`. Integration: `bgra_dmabuf_roundtrip_{1920×1200,2880×1920}` + `imported_bgra_then_scaler_2880×1920_to_2160×1440` — bisect entry points splitting (scaler isolation) ↔ (BGRA dma-buf import/export) ↔ (scaler-on-dma-buf). |
| `tether-scaler` | 8 lib + 6 integration (`tests/quality.rs`) + 15 `#[ignore]` integration (`tests/hardware.rs`) | Mitchell-Netravali reference vs CPU parity, fp16 linear-light, mip prefilter, asymmetric scale; `matches_reference_coord_encoded_left_edge` (2880×1920 → 2160×1440 left-edge regression — bottom of the bisect stack). |
| `tether-capture` | 15 default / 18 with `test-support` | `HashDamage::classify` policy incl. native-damage short-circuit; `native_damage_for_frame_status` (macOS) maps all six `SCFrameStatus`; `native_damage_from_region_count` (Linux) empty-list ⇒ idle; `video_damage_meta_pod_has_choice_range_size` pod-shape snapshot. Test-pattern producer lifecycle + `set_target_fps_changes_cadence_mid_stream` + `set_target_fps_clamps_zero_to_one`. SCK pixel-format probe (`#[ignore]` on macOS). `test_support`: `ScriptedSource` for precise-timing scenarios. **Windows D3D11** (cfg-gated): the capture→encode ownership handshake + freshest-wins handoff — `slot_return_releases_slot_to_free_list_on_drop`, `send_latest_drops_oldest_and_reclaims_its_slot`, `acquire_slot_evicts_mailbox_when_free_list_empty`, `acquire_slot_returns_none_when_every_slot_is_in_flight`, `producer_outrunning_consumer_keeps_freshest_and_leaks_no_slots`, plus the consumer-liveness shutdown contract (`liveness_tracks_frame_receiver_lifetime`, `liveness_drops_when_handle_discarded_without_into_rx`). |
| `tether-decode` | 0 lib + 18 integration (`tests/run_thread.rs`, requires `test-support`) | `run_thread` (extracted from `apps/tether-client/src/main.rs`) under fault injection: decode success → `LatestFrame`, hard-error → IDR callback, soft-error → IDR callback, rate-limiting, build failure, dropped sender, watchdog escalation, rebuild budget exhaustion, plus the first-IDR decode gate (green-screen-on-connect fix). Exercised via `FakeDecoder` (`one_frame_then_idle`, scriptable submit/next_frame outcomes, `flush_count` field). |
| `tether-audio` | 22 Linux / 19 macOS / 33 Windows lib + 2 integration (`audio_loopback.rs`) | Lib: Opus encode/decode + config hardening against untrusted `OpusConfig` (5, `codec.rs`); lock-free jitter ring drop-oldest + cap-and-drop under overrun (7, `playback/ring.rs`); `playback::policy` prebuffer / starve / resync decisions (6); test-pattern producer (1); Linux PipeWire interleave / truncate / empty-frame adapters (3, `capture/linux.rs`); on Windows the WASAPI `FormatConverter` remix (stereo / 5.1 / 7.1 → stereo) + resampler continuity + mix-format clamping (14, `capture/windows.rs`). Integration: `audio_loopback.rs` (2) — Opus round-trip through the real `Datagram::Audio` unreliable channel, lossless + 1%-loss with gap-driven PLC concealment. Hardware (`#[ignore]`, one per platform): `{linux,macos,windows}_audio_roundtrip.rs` — real system-output capture → Opus → playback (needs a live audio device + capture permission/daemon). |
| `tether-vaapi` | 0 | Hand-rolled libva FFI bindings; tested transitively through `tether-codec/vaapi/tests.rs`. |

## Test infrastructure (`test-support` features)

Several crates expose loopback + fault-injection primitives behind a
`test-support` cargo feature so downstream crates can compose them
without pulling test-only types into production builds.

- **`tether-transport::test_support`** — `DuplexControlChannel`
  (tokio `duplex` backed `ControlChannel`), `DuplexVideoChannel`
  (mpsc for the single video datagram channel),
  `DuplexInputChannel`, `LossyChannel<V: VideoChannel>` with
  `LossyConfig { drop_probability, reorder_window, seed }` for
  deterministic loss/reorder fuzzing.
- **`tether-decode::test_support`** — `FakeDecoder` (constructor
  `one_frame_then_idle`, scriptable submit/next_frame outcomes,
  `flush_count`).
- **`tether-capture::test_support`** — `ScriptedSource` for precise
  per-frame timing scenarios.

Consumers: `tether-session/tests/{loopback,decoder_thread_loopback,
video_loopback,video_loopback_with_loss,epoch_bump_invalidates_inflight,
input_loopback}.rs`, `tether-decode/tests/run_thread.rs`.

## Test categories

We tag tests by *what hardware they need*, encoded via `#[ignore]`
with an explanatory string:

- **Default-on** — run on every `cargo test`. No hardware assumptions
  beyond what the workstation already has (file I/O, in-process QUIC
  loopback via tokio).
- **`#[ignore = "requires …"]`** — needs real GPU, VAAPI, or
  VideoToolbox. The ignore message names the requirement (`vainfo`,
  the Vulkan extension, `requires macOS + VideoToolbox`, etc.).
- **Proptest** — `tether-protocol/tests/fragmenter_property.rs` runs
  256-iteration property cases over fragmenter ↔ reassembler under
  random loss + reorder. New invariants on fragmenter shape go here,
  not in lib unit tests.

The benchmark cells (one per codec × resolution) live in
`crates/tether-codec/src/vaapi/bench.rs`. Run with:

```text
cargo test -p tether-codec --lib bench -- --ignored --nocapture --test-threads=1
```

`--test-threads=1` is required: parallel cells contend for the same
VAAPI device.

## SKIP-with-diagnostic pattern (hardware tests)

When a hardware test asserts a *behaviour* whose backing driver
support is patchy across vendors, the test SKIPs (prints a diagnostic
+ returns) on drivers where the feature is known-missing rather than
hard-failing. SKIP requires a **verified-negative**: a diagnostic
that confirms the feature was not honoured (e.g. ratio in a known
no-op regime, or option present in `unused_avoptions()`). Without
the verified-negative we hard-fail.

Used today in:

- `vaapi_intra_refresh_round_trip` — SKIPs when
  `intra_refresh_period` lands in `unused_avoptions()` (Intel iHD);
  otherwise asserts ≤ 1 IDR + ≥ 56/60 decoded.
- `vaapi_min_qp_floor_reduces_bitstream` — encodes noisy content
  at qmin=1 vs qmin=45; ratio < 0.70 passes, ≥ 0.95 SKIPs (Intel
  iHD reality), 0.70..0.95 fails-with-diagnostic.
- `vaapi_bitrate_retune_changes_bitstream_size` — live retune
  1 Mbps → 20 Mbps; SKIPs at ratio ≤ 1.5 (Intel iHD reality).
- `d3d11_{qsv,amf,nvenc}_gpu_encode_decode_roundtrip` (Windows) —
  gate on the present GPU's PCI vendor (`device_vendor_id`) and SKIP
  before constructing the encoder when it doesn't match the target.
  Here the "verified-negative" is structural, not a no-op ratio:
  constructing a foreign vendor's encoder faults inside that vendor's
  runtime (`STATUS_ACCESS_VIOLATION`), so the test *must* skip on the
  wrong GPU rather than try-and-fall-back. On matching hardware it
  hard-asserts the intended backend opened (not the `hevc_mf` fallback).

When to use SKIP: the property is real *behaviour* (not API shape)
and known to be silently no-op on a specific driver. When to
hard-fail: API-shape contract (`set_bitrate_kbps` returns Ok) or
behaviour we expect to work everywhere.

## Conventions for new tests

- **Wire round-trip first.** Any new protocol message gets a unit
  test in `tether-protocol/src/lib.rs#mod tests` that encodes +
  decodes and asserts every field round-trips.
- **Forward-compat probes for tagged enums.** When you add a new
  `*Hello` body field or variant, hand-craft a "future" wire byte
  sequence and assert older code errors cleanly. See
  `unknown_client_hello_variant_fails_decode`.
- **Stamp every timing.** New stages in the host pipeline that
  touch `HostFrameTimingBuilder` add a `should_panic` test for the
  skip case — the panic is the contract.
- **Hardware tests are `#[ignore]`, not deleted.** Gate on
  `#[ignore = "requires …"]` with a specific message.
- **No scaffolding for hypothetical futures.** Tests that exist
  only to assert "this hook exists" without an in-tree caller get
  deleted, not kept.

## CI shape

- Implemented in `.github/workflows/ci.yml` (push/PR) across Linux, macOS,
  and Windows, with shared setup in `.github/actions/setup`. `make ci`
  reproduces the no-hardware checks locally.
- Default: `cargo build --workspace --all-targets && cargo test --workspace`.
  The build runs with `RUSTFLAGS=-D warnings`, so any new warning is a hard
  gate. The Tauri shell (excluded from the workspace) is typechecked +
  backend-tested in a separate `shell-check` job.
- Hardware tests (VAAPI/Vulkan/Metal) are **not** run in CI — GitHub-hosted
  runners have no usable GPU. Run `make test-hw` locally on a hardware runner;
  a self-hosted GPU runner that adds `cargo test --workspace -- --ignored` is
  a documented follow-up.
- Releases: `.github/workflows/release.yml` on `v*` tags produces Tauri
  installers + a signed updater manifest (see `docs/RELEASING.md`).
- Clippy is a blocking gate: `cargo clippy --workspace --all-targets -- -D
  warnings`, plus the excluded Tauri shell. Intentional numeric/pixel casts
  (the workspace opts into `cast_possible_truncation` / `cast_sign_loss` /
  `cast_lossless`) are suppressed with scoped `#[allow(...)]` + a justifying
  comment — never by disabling the lints. New warnings must be resolved the
  same way (fix, or scoped allow with reason) in the change that introduces
  them.

## What's deliberately untested today

- **End-to-end host↔client glass-to-glass.** Validated by hand on a
  Linux↔Linux LAN and Mac→Linux LAN; no automation. The handshake,
  video, input, decoder-thread, and epoch-bump layers are now
  loopback-tested in-process (see `tether-session` integration tests
  above). What still requires a manual session: real QUIC across a
  real link, real capture backend producing real frames, real
  display present, glass-to-glass latency.
- **`tether-vaapi` directly.** Tested transitively through
  `tether-codec/vaapi/tests.rs`. Direct tests would exercise libva.
- **`VaapiDecoder` SW-fallback rejection.** The hard-error path
  when the driver hands back a non-VAAPI AVFrame mid-stream isn't
  unit-tested — reaching it requires real hardware that bails or a
  substantial test-seam refactor. Verified by inspection; failure
  mode is observably loud (`Err(UnsupportedInputFormat)` + auto-IDR).
- **Encoder parameter-set repetition.** The `drain_encoder` prepend
  of extradata onto every keyframe is exercised by hardware VAAPI
  encoder tests; no fake-AVCodecContext seam exists to unit-test
  the prepend in isolation.
- **`max_concurrent_uni_streams` enforcement.** Quinn enforces the
  limit; trusted to quinn's own test suite.
- **Capture-backend FFI surface (PipeWire pod negotiation, SCK
  attachment reads).** We carve testable kernels out of each backend
  (`native_damage_for_frame_status`, `native_damage_from_region_count`,
  `video_damage_meta_pod_has_choice_range_size`) so policy and pod
  *encoding* are unit-testable without hardware. What stays
  unverified in-tree: whether a real PipeWire server attaches the
  meta in response to our `SPA_PARAM_Meta` request (compositor /
  xdg-desktop-portal version dependent), whether real SCK fires
  `Idle`/`Stopped` at the rate we expect, and whether `find_meta`
  returns the same pointer shape across libpipewire minor versions.
  Any new capture-backend FFI surface (a new SPA meta type, a new
  SCK attachment) follows the same shape: pure helper + pod-shape
  snapshot + manual verification note.
