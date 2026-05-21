# Tether — Testing conventions

The foundation review (Q2 2026) made the test surface deliberately
small and predictable. This document codifies what we have, what
each layer covers, and how to extend it.

## Unit tests by crate

| Crate | Tests | Covers |
| --- | --- | --- |
| `tether-protocol` | 46 | Wire round-trips for every control variant (handshake, codec negotiation, video packets + `stream_epoch>u16` widening, `VideoFrameMetaEnvelope`, cursor position + control-stream cursor shapes, multi-monitor `DisplayList`, stream lifecycle, `ClientStats`, `ControlMessage::Extension`, audio `Opus`, `PixelFormat` hello extension, `InputEvent::device_id`), fragmenter / reassembler invariants (out-of-order, stale eviction, wall-clock-timeout eviction, cross-epoch rejection, duplicate-fragment idempotency, continuation-before-First, `single_packet` reliable-IDR path), `HostFrameTimingBuilder` typestate, forward-compat probes for every wire-serialised tagged enum (`ClientHello`, `ServerHello`, `ControlMessage`, `VideoPacket`, `VideoFrameMetaEnvelope`), clock-sync edges (zero-RTT, negative-processing, near-i64::MAX offset). |
| `tether-transport` | 6 integration tests in `tests/roundtrip.rs` | QUIC handshake, control + datagram round-trip, fingerprint pinning, oversized-datagram local reject, video-keyframe-stream round-trip, oversized-keyframe local reject. |
| `tether-codec` | 2 default + 8 `#[ignore]` Linux + 2 `#[ignore]` macOS | SW H264 round-trip (test-only); codec_name map sanity; VAAPI encoder/decoder/dma-buf-import on Linux hardware; HEVC + H.264 probe smoke check; per-codec × per-resolution benchmarks (`vaapi::bench`, 4 cells × 3 paths); VideoToolbox encoder construct + H.264 BGRA round-trip on macOS hardware. |
| `tether-input` | 9 | Modifier tracking, HID routing, cursor normalization. |
| `tether-session` | 5 | `IdrSignal` coalescing + clone-share; `EncodeStatsWindow` emit / idle / accumulate. |
| `tether-render` | 13 + 1 `#[ignore]` | Cursor letterbox clipping, aspect ratio; `LatestFrame` Send+Sync + drop-oldest displacement; dma-buf zero-copy on hardware. |
| `tether-gpuconvert` | 11 `#[ignore]` | BGRA→NV12 + DMA-BUF round-trip with real Vulkan adapter. |

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

Today there are **~22 ignored tests** (count varies by platform-cfg):
`tether-codec/vaapi` (8 — 5 correctness, plus 4 cells of the `bench`
matrix that each exercise encode_bgra + encode_dmabuf + decode),
`tether-codec/videotoolbox` (2 — encoder smoke + HEVC constructs),
`tether-gpuconvert` (11), and `tether-render` (1). They are real and
load-bearing on hardware; they are not abandoned.

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

- Default: `cargo build --workspace && cargo test --workspace`.
- Hardware runner (separate job, when one exists): same plus
  `cargo test --workspace -- --ignored`.
- `cargo clippy --workspace --all-targets` is currently advisory
  (40 pre-existing warnings, mostly in `tether-gpuconvert` cast
  sites with documented `#[allow]`s pending).

## What's deliberately untested today

- **End-to-end host↔client glass-to-glass.** Validated by hand on a
  Linux↔Linux LAN and Mac→Linux LAN; no automation. A `tether-session`
  loopback integration test that runs `HostSession` and
  `ClientSession` over a `tokio::io::duplex` shim is the natural next
  addition once those session-level scaffolds exist.
- **The actual rendered pixels.** `tether-render`'s shader output
  is validated by eye, not by image-diff. A headless `wgpu::Surface`
  + image diff against a checked-in fixture is a worthwhile
  follow-up but not load-bearing today.
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
- **VAAPI 4:4:4 decode probe.** `probe_decoder_profile` on Linux is
  codec-keyed, not profile-keyed: a successful `VaapiDecoder::new` for
  HEVC means we advertise HEVC 4:4:4 decode capability even though the
  driver may only support 4:2:0 decode. In practice consumer VAAPI
  drivers expose decode for the same chromas they expose encode for, so
  this hasn't bitten anyone — but a driver mismatch would surface as a
  first-frame `UnsupportedInputFormat` rather than a clean startup
  failure. Closing the gap means encoding a tiny 4:4:4 bitstream with
  `VaapiEncoder` and round-tripping it through `VaapiDecoder` at probe
  time. The macOS path has no equivalent gap because
  `probe_decoder_profile` gates on `ChromaSubsampling::Yuv420` before
  constructing.
