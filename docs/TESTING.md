# Tether — Testing conventions

The foundation review (Q2 2026) made the test surface deliberately
small and predictable. This document codifies what we have, what
each layer covers, and how to extend it.

## Unit tests by crate

| Crate | Tests | Covers |
| --- | --- | --- |
| `tether-protocol` | 50 | Wire round-trips for every control variant (handshake, codec negotiation incl. 10-bit `VideoProfile` constants + forward-compat `u8` bit_depth probe, video packets + `stream_epoch>u16` widening, `VideoFrameMetaEnvelope`, cursor position + control-stream cursor shapes, multi-monitor `DisplayList`, stream lifecycle, `ClientStats`, `ControlMessage::Extension`, audio `Opus`, `PixelFormat` hello extension incl. `P010` / `P410`, `InputEvent::device_id`), fragmenter / reassembler invariants (out-of-order, stale eviction, wall-clock-timeout eviction, cross-epoch rejection, duplicate-fragment idempotency, continuation-before-First, `single_packet` reliable-IDR path), `HostFrameTimingBuilder` typestate, forward-compat probes for every wire-serialised tagged enum (`ClientHello`, `ServerHello`, `ControlMessage`, `VideoPacket`, `VideoFrameMetaEnvelope`), clock-sync edges (zero-RTT, negative-processing, near-i64::MAX offset). |
| `tether-transport` | 6 integration tests in `tests/roundtrip.rs` + 3 in `test_support` + 1 in `handshake` (the last two under the `test-support` feature) | QUIC handshake, control + datagram round-trip, fingerprint pinning, oversized-datagram local reject, video-keyframe-stream round-trip, oversized-keyframe local reject. `test_support`: `DuplexControlChannel` handshake round-trip, post-handshake control message exchange, dropped-peer surfaces `StreamClosed`. `handshake`: `HostHandshake` → `ClientHelloReceived` typestate routes recv-then-send and returns the channel for post-handshake use. |
| `tether-codec` | (count) | `validate_chosen_profile` accept/reject/empty-advertised paths; SW H264 round-trip (test-only); VT `codec_name` map; VAAPI encoder/decoder/dma-buf-import on Linux hardware; per-codec × per-resolution benchmarks (`vaapi::bench`, 4 cells × 3 paths); VideoToolbox: encoder construct, H.264 BGRA round-trip, HEVC 10-bit / 4:4:4 probe matrix, `videotoolbox_round_trip_chroma_matrix` cross-checks the encode side against an independent encode→decode→IOSurface-fourcc check (catches silent downsample regressions), self-decodable-IDR mid-session recovery. Capability discovery and preference-list negotiation tests live in `tether-probe`. |
| `tether-probe` | 10 default + 0 `#[ignore]` (today) | `PipelineStage` exhaustiveness, `ProfileSupport` helper accessors, preference order (5-entry list, 10-bit-first), `pick_supported_profile` (best mutual / fallback / disjoint / empty / forward-compat unknown bit_depth), `probe_client_does_not_mirror_encode_bit_into_decode_field` (the Mac 4:4:4 decode-without-encode invariant). |
| `tether-input` | 9 | Modifier tracking, HID routing, cursor normalization. |
| `tether-session` | 15 unit + 7 integration in `tests/loopback.rs` | Unit: `IdrSignal` coalescing + clone-share; `EncodeStatsWindow` emit / idle / accumulate; `HostSession::accept` decode-profile-extension parsing (missing / oversize / malformed / well-formed / unknown bit_depth filter / no-match server-hello shape / chosen-profile echo); `ClientSession::connect` resolve_negotiated_profile (extension absent → 8-bit synth, present + decodes → authoritative, undecodable → error, unknown bit_depth → reject). Integration (via `tether-transport`'s `test-support` feature `DuplexControlChannel`): happy-path handshake with RTT/offset bounds, no-mutual-profile sends Goodbye after placeholder hello, host picks unadvertised profile → client `ProfileNotAdvertised`, host picks unknown bit_depth → client `UnknownBitDepth`, host filters unknown depths from advert keeping known ones, legacy host with no encode-profile extension synthesizes 8-bit profile from inline fields, dropped client during handshake → `Transport(_)`. |
| `tether-render` | 23 default + 14 `#[ignore]` (Linux); + 4 `#[ignore]` (macOS) | Cursor letterbox clipping, aspect ratio; `LatestFrame` Send+Sync + drop-oldest displacement; transfer_kind dispatch table pin; `range_kind_for(bit_depth, layout)` dispatch table pin + algebraic check that 10-bit limited-range breakpoints land white at 1.0 and black at 0.0; `render_layout_for(chroma, bit_depth)` dispatch table pin (incl. Yuv420 10-bit → Biplanar16 which the import path relies on for UV dimensioning); the **end-to-end roundtrip harness** (`test_harness.rs` + `dmabuf_test.rs`) drives the production multi-pass renderer (`Gpu::new_headless`) through 13 hardware cells covering identity / host-scaler / client-upscale / surface-below-video / full-chain / repro-shape rows at H.264 4:2:0, HEVC Main, HEVC Main 4:4:4, and HEVC Main 10 (10-bit cells SKIP on Intel iHD + Meteor Lake driver gap). Primary metric is **geometric residual on a coordinate-encoded fixture** (`Fixture::CoordEncoded`); SSIM and BT.709 Y-PSNR are the secondary catch-net. On failure the harness dumps readback/reference/diff/SSIM-heatmap PNGs + `metrics.txt` to `target/roundtrip-diagnostics/<case>/`. IOSurface zero-copy on macOS hardware (HEVC 4:2:0 8-bit + 10-bit via encode→decode→render; HEVC 4:4:4 8-bit + 10-bit via fixture-decode→render since VT lacks Main444 encode). |
| `tether-gpuconvert` | 4 default + 15 `#[ignore]` lib + 7 `#[ignore]` integration (`tests/scaler_roundtrip.rs`) | `drm_fourcc_to_vk_format` table coverage incl. 10-bit biplanar plane fourccs (R16/GR32 → R16_UNORM/R16G16_UNORM) + 8-bit family regression + unknown-fourcc rejection; BGRA→NV12 + DMA-BUF round-trip with real Vulkan adapter; structural alignment-regression `convert_reports_64_aligned_y_stride_at_unaligned_width` (lib unit test in `src/nv12_dmabuf.rs`) asserts the iHD-VAAPI 64-byte luma row pitch fix at 2160×1440 without needing encoder/decoder/renderer in the loop. Integration tests `bgra_dmabuf_roundtrip_{1920×1200,2880×1920}` and `imported_bgra_then_scaler_2880×1920_to_2160×1440` are bisect entry points: split a failing roundtrip into (scaler isolation) ↔ (BGRA dma-buf import/export) ↔ (scaler-on-dma-buf) without running the full chain. |
| `tether-scaler` | 18 default lib + 12 `#[ignore]` (hardware) + 6 `quality` integration | Mitchell-Netravali reference vs CPU implementation parity, fp16 linear-light path, mip prefilter, asymmetric scale; `matches_reference_coord_encoded_left_edge` (the harness-companion left-edge regression at 2880×1920 → 2160×1440 — bottom of the bisect stack for any future left-edge-corruption bug). |
| `tether-capture` | 1 default + 1 `#[ignore]` macOS | SCK pixel-format probe records `420v`/`420f`/`'444v'`/`'444f'`/`xf44` acceptance via real `start_capture` + frame-arrival check for the Unknown-fourcc cases. |

## Test categories

We tag tests by *what hardware they need*, encoded via `#[ignore]`
with an explanatory string:

- **Default-on** — run on every `cargo test`. No hardware assumptions
  beyond what the workstation already has (file I/O, in-process QUIC
  loopback via tokio).
- **`#[ignore = "requires …"]`** — needs real GPU, VAAPI, or
  VideoToolbox. Run via `cargo test -- --ignored` on a host with the
  right capabilities. The ignore message names the requirement
  (`vainfo`, the Vulkan extension, `requires macOS + VideoToolbox`,
  etc.) so the next person can see at a glance whether their box
  should run them.

Today there are **~58 ignored tests** (count varies by platform-cfg):
`tether-codec` (10 — VAAPI + VideoToolbox correctness, plus the
`bench` matrix cells), `tether-gpuconvert` (15 lib + 7 integration —
includes the BGRA dma-buf import/export bisect helpers + the
structural alignment-regression test), `tether-render` (14 — the
roundtrip-harness matrix in `dmabuf_test.rs`), and `tether-scaler`
(12 — Mitchell reference parity at production dims). They are real
and load-bearing on hardware; they are not abandoned.

The benchmark cells (one per codec × resolution) live in
`crates/tether-codec/src/vaapi/bench.rs`. Run with:

```text
cargo test -p tether-codec --lib bench -- --ignored --nocapture --test-threads=1
```

`--test-threads=1` is required: parallel cells interleave output and
contend for the same VAAPI device, which makes the per-iteration
timing meaningless. Results print as `p50 / p99 / max ms` with a
budget headroom annotation for the 60 fps frame budget (16.67 ms).
See `docs/ARCHITECTURE.md` for the current baseline on Intel Arc.

## Conventions for new tests

- **Wire round-trip first.** Any new protocol message gets a unit test
  in `tether-protocol/src/lib.rs#mod tests` that encodes + decodes and
  asserts every field round-trips.
- **Forward-compat probes for tagged enums.** When you add a new
  `*Hello` body field or a new `*Hello` variant, add a test that
  hand-crafts a "future" wire byte sequence and asserts older code
  errors cleanly. See `unknown_client_hello_variant_fails_decode`
  for the template.
- **Stamp every timing.** New stages in the host pipeline that touch
  `HostFrameTimingBuilder` should add a `should_panic` test for the
  skip case — the panic is the contract.
- **Hardware tests are `#[ignore]`, not deleted.** If a test needs a
  GPU, gate it on `#[ignore = "requires …"]` with a specific message,
  not silently skipped via runtime checks. Make the requirement
  obvious.

## CI shape

- Default: `cargo build --workspace --all-targets && cargo test --workspace`.
  `cargo build --workspace --all-targets` is warning-free today —
  treat any new warning as a gate.
- Hardware runner (separate job, when one exists): same plus
  `cargo test --workspace -- --ignored`.
- `cargo clippy --workspace --all-targets` is currently advisory
  (pre-existing cast warnings in `tether-codec` and
  `tether-scaler/src/reference.rs`; `#[allow]`s pending). New
  clippy warnings in files under active edit should be addressed
  in the same change.

## What's deliberately untested today

- **End-to-end host↔client glass-to-glass.** Validated by hand on a
  Linux↔Linux LAN and Mac→Linux LAN; no automation. The handshake
  layer is now loopback-tested in-process via
  `tether-transport::test_support::DuplexControlChannel` —
  `crates/tether-session/tests/loopback.rs` runs `HostSession::accept`
  and `ClientSession::connect` against each other through a
  `tokio::io::duplex` pair, covering the clock-sync RTT/offset math,
  Goodbye-on-no-match, unknown bit-depth refusal, and the host-
  lenient / client-strict bit-depth asymmetry. The video / input /
  datagram layers don't yet have duplex fakes — `InputChannel` and
  `VideoChannel` traits are defined in `tether-transport` but
  duplex impls land when the first test that needs one does. The
  remaining glass-to-glass gap is everything past `StreamReady`:
  fragmenter under loss, IDR signalling latency, decoder restart
  recovery, render-thread drop-oldest under backpressure.
- ~~**The actual rendered pixels.**~~ As of the
  `test_harness.rs` + `dmabuf_test.rs` work, the production
  multi-pass renderer (`Gpu::new_headless`) is exercised against a
  coordinate-encoded fixture across a 13-cell `(capture × encode ×
  surface)` matrix and compared to a CPU Mitchell reference with
  three metrics (geometric residual, SSIM, BT.709 Y-PSNR). On-
  failure diagnostic dumps land in `target/roundtrip-diagnostics/`.
  Gap that remains: the **macOS** harness sibling (`iosurface_test.rs`
  uses fixed dims and asserts the IOSurface→Metal→wgpu import, but
  doesn't drive the full multi-pass renderer across a (capture ×
  encode × surface) matrix yet).
- **`tether-vaapi` directly.** The hand-rolled libva FFI bindings
  are tested transitively through `tether-codec/vaapi/tests.rs`.
  Direct tests would just exercise libva itself.
- **`VaapiDecoder` SW-fallback rejection.** The hard-error path that
  fires when the driver hands back a non-VAAPI AVFrame mid-stream
  isn't unit-tested — reaching it requires either real hardware that
  bails, or a substantial refactor to extract a test seam. The check
  is short enough to verify by inspection and the failure mode is
  observably loud (returns `Err(UnsupportedInputFormat)`, auto-IDR
  fires from the client).
- **Encoder parameter-set repetition.** The `drain_encoder` prepend of
  extradata onto every keyframe is exercised by the hardware VAAPI
  encoder tests; there's no fake-AVCodecContext path that would let
  us unit-test the prepend logic in isolation. Verified by inspection
  + hardware tests.
- **`max_concurrent_uni_streams` enforcement.** Quinn enforces the
  limit; we don't have a test that opens more than the cap to confirm
  the rejection happens at the connection layer. Trusted to quinn's
  own test suite.
