# Tether — Testing conventions

The foundation review (Q2 2026) made the test surface deliberately
small and predictable. This document codifies what we have, what
each layer covers, and how to extend it.

## Headline counts

`cargo test --workspace` is the authoritative no-hardware suite; use
`cargo test --workspace -- --list` when you need the current count for a
release note or audit. Counts are intentionally not treated as a protocol
contract because platform cfgs and hardware-gated cells move often. At this
point the workspace lists hundreds of tests across unit, integration, doc, and
`#[ignore]` hardware binaries; the table below records the current shape and
the coverage that matters.

## Unit and integration tests by crate

| Crate | Tests | Covers |
| --- | --- | --- |
| `tether-protocol` | 100 lib + 4 integration | Wire round-trips for every control variant (handshake, codec negotiation incl. 10-bit `VideoProfile` constants + forward-compat `u8` bit_depth probe, video packets + `stream_epoch>u16` widening, `VideoFrameMetaEnvelope`, cursor position + control-stream cursor shapes, multi-monitor `DisplayList`, stream lifecycle, `ClientStats` named-field mapping, `Goodbye` final session summary incl. video/audio stats, `ControlMessage::Extension`, audio `Opus` incl. near-max-payload datagram-decode, `PixelFormat` hello extension incl. `P010` / `P410`, `InputEvent::device_id`), fragmenter / reassembler invariants (out-of-order, stale eviction, wall-clock-timeout eviction, cross-epoch rejection, duplicate-fragment idempotency, continuation-before-First, unified IDR-keyframe datagram path), `HostFrameTimingBuilder` typestate, forward-compat probes for every tagged enum, clock-sync burst edges, reliable-control field caps (`ExtensionMessage.payload`, cursor-shape pixels), **receiver-side wire-validation** (`video::validation_tests`: oversized / zero fragment_count, continuation index, shard_size, total_body_len, total_body_len above shard capacity, legitimate-accept, `fragments_lost` bump on reject; constants `MAX_FRAGMENTS_PER_FRAME = 4096`, `MAX_FRAME_BODY_BYTES = MAX_FRAGMENTS_PER_FRAME * MAX_DATAGRAM_PAYLOAD`), **multi-block Reed-Solomon FEC** around `VideoPacket::Parity` (fragment + per-block parity emission + RS recovery under simulated loss, multi-block split for large IDRs, every-datagram-fits-budget across input-echo sizes; `FEC_MAX_PRIMARY_SHARDS = 212` per block). Integration: `tests/fragmenter_property.rs` — 4 proptest cases (256 iterations each) over fragmenter ↔ reassembler under random loss + reorder. |
| `tether-transport` | 17 lib + 8 integration in workspace CI | QUIC handshake, control + datagram round-trip, fingerprint pinning, oversized-datagram local reject, live-MTU video fragmentation, input wrong-role errors, clean datagram shutdown classification, datagram variant round-trips incl. audio, and pairing accept/reject. `test_support` (feature-gated): `DuplexControlChannel` handshake round-trip, post-handshake control message exchange, dropped-peer surfaces `StreamClosed`; `HostHandshake` → `ClientHelloReceived` typestate routes recv-then-send. |
| `tether-codec` | 85 Linux-listed tests incl. hardware/bench cells | `validate_chosen_profile` accept/reject/empty paths; SW H264 round-trip (test-only); VT `codec_name` map; NVENC/NVDEC table and bridge tests. Hardware: VAAPI encoder/decoder/dma-buf import incl. `av1_main_dmabuf_roundtrip`, `hevc_main444_dmabuf_roundtrip` (8-bit XYUV), and `hevc_main444_10bit_xv30_dmabuf_roundtrip` (10-bit XV30); per-codec × per-resolution `vaapi::bench`; VideoToolbox encoder, H.264 BGRA round-trip, HEVC 10-bit / 4:4:4 probe matrix, `videotoolbox_round_trip_chroma_matrix` cross-check, self-decodable-IDR mid-session recovery. **Encoder-knob tests** (all use the SKIP-with-diagnostic pattern — see below): `vaapi_intra_refresh_round_trip`, `vaapi_min_qp_floor_reduces_bitstream`, `vaapi_bitrate_retune_changes_bitstream_size`. **Windows D3D11** (cfg-gated to `target_os = "windows"`): backend-selection pure-logic tests (`unknown_vendor_falls_back_to_mf_only`, `known_vendors_lead_with_hardware_then_mf`), `d3d11_rejects_444_at_construction_no_silent_downsample`, and the vendor-gated GPU encode→decode round-trips `d3d11_{qsv,amf,nvenc}_gpu_encode_decode_roundtrip` (`#[ignore]`; each gates on the present GPU vendor and SKIPs off-vendor — see SKIP section). |
| `tether-probe` | 22 lib + platform `#[ignore]` hardware | `PipelineStage` exhaustiveness, `ProfileSupport` helpers, seven-entry preference order (HEVC 4:4:4, AV1 4:2:0, HEVC 4:2:0, H.264 floor), forced-profile parser/selector, `pick_supported_profile` (best mutual / fallback / disjoint / empty / forward-compat unknown bit_depth), Mac 4:4:4 decode-without-encode invariant, host/client probe intersection, and decode advert guards for H.264/HEVC/AV1 paths. Windows (`#[ignore]`): `client_offers_hevc_main10` — regression guard that the P010 GPU-export decode probe keeps Main10 in the advert. |
| `tether-input` | 27 Linux / 15 macOS-Windows | 11 cross-platform translator tests (modifier tracking, HID routing, cursor normalization). On Linux, `inject/uinput.rs` adds 16 unit tests: HID→evdev mapping, ASCII char→keystroke, scroll-detent sign/quantisation/overflow, normalised→ABS coordinates, pointer-button↔device cross-table, and modifier-reconcile release-variant logic. On macOS/Windows the 4 `clamp_relative_delta` (±1000 px) tests live in `inject/enigo_backend.rs`. |
| `tether-session` | 21 lib + 24 integration | Lib: `IdrSignal` coalescing + clone-share; `EncodeStatsWindow` emit / idle / accumulate; host-side typed `ServerHello` construction + unknown bit-depth filtering; client-side unknown bit-depth rejection helper. Integration (via `tether-transport`'s `test-support` feature): `loopback.rs` covers successful typed handshake + clock-probe burst, typed no-profile rejection, unadvertised-profile rejection + client `Goodbye(ProtocolError)`, unknown host bit-depth rejection + client `Goodbye(ProtocolError)`, host filtering of unknown client bit-depths, Goodbye during clock probe, initial viewport wire field, post-handshake `ClientStats`, `SetViewportHint`, `SetDisplayMode -> Unsupported`, dropped-peer Transport error, and the double-send corruption guard; `decoder_thread_loopback.rs` (3) — `run_thread` + `DuplexVideoChannel` + `LatestFrame` smoke; `video_loopback.rs` (2); `video_loopback_with_loss.rs` (3) via `LossyChannel`; `epoch_bump_invalidates_inflight.rs` (1); `input_loopback.rs` (1). |
| `tether-render` | 77 Linux-listed tests incl. hardware cells | Cursor letterbox/aspect; `LatestFrame` Send+Sync + drop-oldest; render health/drop accounting; `transfer_kind` / `range_kind_for` / `render_layout_for` dispatch-table pins (incl. 10-bit limited-range breakpoint algebra and Yuv420 10-bit → Biplanar16); `PresentPolicy` / `FrameAgeTracker` (8 tests on pure `decide_present` logic); **relative-mouse sub-pixel accumulator** (6 tests in `relative_mouse.rs`: whole-pixel passthrough, sub-pixel-held, long-run convergence, mixed-sign no-stall, i16 saturation, reset). Hardware: end-to-end roundtrip harness (`test_harness.rs` + `dmabuf_test.rs`) drives the production multi-pass renderer (`Gpu::new_headless`) through cells covering identity / host-scaler / client-upscale / surface-below-video / full-chain / repro-shape across H.264 4:2:0, HEVC Main, HEVC Main 4:4:4, HEVC Main 10, HEVC Main 4:4:4 10-bit, AV1 4:2:0 8-bit + 10-bit. Primary metric is geometric residual on `Fixture::CoordEncoded`; SSIM + BT.709 Y-PSNR are the secondary catch-net. A parallel set of `Fixture::ColorBars` cells asserts each codec reconstructs true colour. macOS IOSurface zero-copy covers HEVC Main / Main10 / Main 4:4:4 8 + 10-bit; BGRA-bridge cells cover the macOS host Metal bridge output for 4:2:0 and 4:4:4 at 8/10-bit; `iosurface_bgra_bridge_videotoolbox_encode_chroma_matrix` records the current VT encode negative for 4:4:4. **Windows D3D11** (cfg-gated, `#[ignore]`): native-renderer headless roundtrips cover synthetic NV12, cursor overlay, HEVC/H.264/AV1 coord and colour fixtures, and non-identity scaling; 4:4:4 is rejected at construction (D3D11 VP is 4:2:0-only). |
| `tether-gpuconvert` | 6 lib + 18 `#[ignore]` lib + 7 `#[ignore]` integration (`tests/scaler_roundtrip.rs`) | `drm_fourcc_to_vk_format` table coverage (8 + 10-bit biplanar + packed XV30 → A2B10G10R10_UNORM_PACK32 + unknown rejection); BGRA→NV12 + DMA-BUF round-trip; `convert_solid_{white,red}_roundtrip_packed_xv30` (10-bit channel-mapping + BT.709 math); `storable_probe_returns_linear_for_xv30`; structural alignment regression `convert_reports_64_aligned_y_stride_at_unaligned_width`. Integration: `bgra_dmabuf_roundtrip_{1920×1200,2880×1920}` + `imported_bgra_then_scaler_2880×1920_to_2160×1440` — bisect entry points splitting (scaler isolation) ↔ (BGRA dma-buf import/export) ↔ (scaler-on-dma-buf). |
| `tether-scaler` | 8 lib + 6 integration (`tests/quality.rs`) + 15 `#[ignore]` integration (`tests/hardware.rs`) | Mitchell-Netravali reference vs CPU parity, fp16 linear-light, mip prefilter, asymmetric scale; `matches_reference_coord_encoded_left_edge` (2880×1920 → 2160×1440 left-edge regression — bottom of the bisect stack). |
| `tether-capture` | 41 Linux lib | `HashDamage::classify` policy incl. native-damage short-circuit; cursor source/shape policy; restore-token file permission hardening; `native_damage_for_frame_status` (macOS) maps all six `SCFrameStatus`; `native_damage_from_region_count` (Linux) empty-list ⇒ idle; `video_damage_meta_pod_has_choice_range_size` pod-shape snapshot. Test-pattern producer lifecycle + `set_target_fps_changes_cadence_mid_stream` + `set_target_fps_clamps_zero_to_one`. SCK pixel-format probe (`#[ignore]` on macOS). `test_support`: `ScriptedSource` for precise-timing scenarios. **Windows D3D11** (cfg-gated): the capture→encode ownership handshake + freshest-wins handoff, plus the consumer-liveness shutdown contract. |
| `tether-decode` | 21 feature-gated integration (`tests/run_thread.rs`) | `run_thread` (extracted from `apps/tether-client/src/main.rs`) under fault injection: decode success → `LatestFrame`, hard-error → IDR callback, soft-error → IDR callback, rate-limiting, build failure, dropped sender, watchdog escalation, rebuild budget exhaustion, render-drop counting, epoch-advance rebuild and throttle counters, AV1 empty-extradata resume, and the first-IDR decode gate. Exercised via `FakeDecoder` (`one_frame_then_idle`, scriptable submit/next_frame outcomes, `flush_count` field). |
| `tether-audio` | 38 Linux lib + 3 integration + 1 Linux `#[ignore]` hardware | Lib: Opus encode/decode + config hardening against untrusted `OpusConfig`; lock-free jitter ring drop-oldest + cap-and-drop under overrun; `playback::policy` prebuffer / starve / resync decisions; RED recovery classification; test-pattern producer; Linux PipeWire interleave / truncate / empty-frame adapters; on Windows the WASAPI `FormatConverter` remix (stereo / 5.1 / 7.1 → stereo) + resampler continuity + mix-format clamping. Integration: `audio_loopback.rs` — Opus round-trip through the real `Datagram::Audio` unreliable channel, isolated-loss RED recovery, and 1%-loss concealment. Hardware (`#[ignore]`, one per platform): `{linux,macos,windows}_audio_roundtrip.rs` — real system-output capture → Opus → playback (needs a live audio device + capture permission/daemon). |
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
  and Windows, with shared setup in `.github/actions/setup`. `mise run ci`
  reproduces the no-hardware checks locally.
- Default: `cargo build --workspace --all-targets && cargo test --workspace`.
  The build runs with `RUSTFLAGS=-D warnings`, so any new warning is a hard
  gate. The Tauri shell (excluded from the workspace) is typechecked +
  backend-tested in a separate `shell-check` job.
- Hardware tests (VAAPI/Vulkan/Metal/D3D11) are **not** run in CI — GitHub-hosted
  runners have no usable GPU. Run `mise run test-hw` locally on a hardware
  runner. It is platform-symmetric by construction: `cargo test --workspace
  --exclude tether-audio --tests -- --ignored` runs every `#[ignore]` hardware
  test for the present host, no matter which crate it lives in. The per-platform
  backends are cfg-gated (VAAPI/VideoToolbox/D3D11) and the cross-platform GPU
  tests (`tether-scaler`, `tether-probe`) run everywhere, so the *same command*
  covers Linux, macOS, and Windows with no platform privileged. Companions:
  `mise run test-audio` (the per-platform system-audio round-trip; needs an
  audio device with sound playing) and `mise run bench` (per-platform benchmark
  cells in release — VAAPI matrix on Linux, the scaler microbench everywhere).
  macOS builds `test-hw` with `--release` (the IOSurface/Metal round-trips are
  too slow in debug for their comparison thresholds); Linux/Windows run debug.
  Expect a non-zero ignored-test count per platform — a run that reports `0`
  hardware tests means a `#[cfg]`/`#[ignore]` gate is misconfigured, not that
  the suite is clean. A self-hosted GPU runner that wires it into CI is a
  documented follow-up.

  Note the suites are symmetric in *infrastructure* (one command, every crate,
  every platform) but not yet in *cell coverage*: the Linux VAAPI/dma-buf path
  is the most mature (the 23-cell render harness, 4:4:4 + AV1 dma-buf, encoder-
  knob SKIP tests). The Windows D3D11 render matrix now covers the shapes it can
  — HEVC + H.264 + AV1 coord/colour (8/10-bit where the codec supports it),
  client-upscale, and surface-below-video — with AV1 verified on Arc/Lunar-Lake-
  class encode; its only remaining gap is 4:4:4 (no D3D11 Video Processor path),
  a hardware limit rather than a missing test. The macOS VideoToolbox/IOSurface
  matrix is the next to broaden. Closing that is tracked separately; the runner
  no longer assumes Linux is the reference.
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

## Live session checklist

Use this when validating a real host↔client pair after protocol/session
changes. Capture host and client logs for every run; `RUST_LOG=info` is the
baseline, `RUST_LOG=debug` is appropriate when investigating recovery, cursor,
or teardown behaviour.

Minimum matrix before calling a protocol/session change live-ready:

- Linux host → Linux client on LAN.
- macOS host → Linux client.
- Linux host → macOS client.
- Windows host → Windows client loopback.
- At least one cross-device Windows client or host run when D3D11 shared-handle
  ownership changed.
- One audio-enabled run per platform pair where both sides support audio, plus
  one host `--no-audio` run and one client `--no-audio` run to verify
  negotiation disables it cleanly.

Before each cross-device run, capture a basic network sanity baseline on the
same path: sustained throughput, packet loss, jitter, and the effective MTU. A
quiet LAN should show no sustained loss and low jitter; if those are bad, keep
that result attached to the session logs so protocol symptoms are not diagnosed
in isolation.

For each run, verify lifecycle events in order:

1. `event="handshake_start"` appears on both peers.
2. Either `event="handshake_accepted"` appears on both peers with the same
   codec/chroma/bit-depth, or one side logs `event="handshake_rejected"` with a
   typed reason.
3. Client logs `event="stream_ready"` and the host logs matching
   `event="stream_ready"` before host `send stats` begin.
4. Normal close logs exactly one typed shutdown reason:
   `event="session_teardown"` on the initiator and `event="peer_goodbye"` on the
   peer. Fatal exits should use `GoodbyeCode::InternalError`; protocol
   violations should use `GoodbyeCode::ProtocolError`.
5. A peer that receives `Goodbye` logs `event="peer_session_summary"` with the
   sender's final video/audio counters (`duration_ms`, codec/chroma/bit-depth,
   frames/bytes sent or received, loss/drop/decode counters).

For steady-state video, collect at least 30 seconds of logs after stream-ready:

- Host `send stats`: `frames`, `avg_capture_age_ms`,
  `min_capture_age_ms`, `max_capture_age_ms`, `avg_encode_ms`,
  `min_encode_ms`, `max_encode_ms`, `avg_send_ms`, `min_send_ms`,
  `max_send_ms`, `kbps_out`, `keyframes_per_s`, `datagrams_sent`,
  `parity_datagrams_sent`, `max_datagrams_per_frame`, `max_frame_bytes`,
  `max_keyframe_bytes`, `forced_idr_misses`, `transient_send_drop_frames`.
- Client `frame stats`: `frames`, `fps`, `avg_latency_ms`,
  `min_latency_ms`, `max_latency_ms`, `avg_network_ms`, `min_network_ms`,
  `max_network_ms`, `avg_decode_ms`, `min_decode_ms`, `max_decode_ms`,
  `kbps_in`, `decode_errors`, `render_drop_frames`, `idr_requests`,
  `decode_queue_drop_frames`, `decode_stale_epoch_drop_frames`,
  `decode_epoch_throttle_drop_frames`, `fec_recovered_frames`,
  `fec_recovered_fragments`.
- Host `client stats`: `window_ms`, `frames_received`, `incomplete_frames`,
  `fragment_loss_events`, `rtt_us`, `fec_recovered_frames`,
  `fec_recovered_fragments`.

Red flags that should block sign-off until explained:

- Missing `stream_ready` on either side.
- `send stats` present but no client `frame stats`.
- Sustained `incomplete_frames > 0` or `fragment_loss_events > 0` on a quiet LAN.
- Sustained FEC recovery on a quiet LAN, or `fec_recovered_frames` rising
  immediately before `incomplete_frames`; this means FEC is masking real loss
  until bursts exceed parity.
- `decode_errors`, `decode_queue_drop_frames`,
  `decode_stale_epoch_drop_frames`, `decode_epoch_throttle_drop_frames`, or
  `render_drop_frames` rising in every window.
- Audio `decode_queue_drop_packets`, `dropout_frames`, or sustained
  `concealed_frames` rising on a quiet LAN.
- `forced_idr_misses > 0`; a hardware encoder ignored a recovery request.
- Large jumps in `max_datagrams_per_frame`, `max_frame_bytes`, or
  `max_keyframe_bytes` that coincide with RTT/loss spikes.
- `kbps_out` and `kbps_in` diverging materially without matching loss counters.
- More than one teardown reason for one session, or a generic clean teardown
  after an earlier fatal/protocol error.
