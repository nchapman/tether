# Tether — Testing conventions

The foundation review (Q2 2026) made the test surface deliberately
small and predictable. This document codifies what we have, what
each layer covers, and how to extend it.

## Unit tests by crate

| Crate | Tests | Covers |
| --- | --- | --- |
| `tether-protocol` | 50 | Wire round-trips for every control variant (handshake, codec negotiation incl. 10-bit `VideoProfile` constants + forward-compat `u8` bit_depth probe, video packets + `stream_epoch>u16` widening, `VideoFrameMetaEnvelope`, cursor position + control-stream cursor shapes, multi-monitor `DisplayList`, stream lifecycle, `ClientStats`, `ControlMessage::Extension`, audio `Opus`, `PixelFormat` hello extension incl. `P010` / `P410`, `InputEvent::device_id`), fragmenter / reassembler invariants (out-of-order, stale eviction, wall-clock-timeout eviction, cross-epoch rejection, duplicate-fragment idempotency, continuation-before-First, `single_packet` reliable-IDR path), `HostFrameTimingBuilder` typestate, forward-compat probes for every wire-serialised tagged enum (`ClientHello`, `ServerHello`, `ControlMessage`, `VideoPacket`, `VideoFrameMetaEnvelope`), clock-sync edges (zero-RTT, negative-processing, near-i64::MAX offset). |
| `tether-transport` | 6 integration tests in `tests/roundtrip.rs` | QUIC handshake, control + datagram round-trip, fingerprint pinning, oversized-datagram local reject, video-keyframe-stream round-trip, oversized-keyframe local reject. |
| `tether-codec` | (count) | `validate_chosen_profile` accept/reject/empty-advertised paths; SW H264 round-trip (test-only); VT `codec_name` map; VAAPI encoder/decoder/dma-buf-import on Linux hardware; per-codec × per-resolution benchmarks (`vaapi::bench`, 4 cells × 3 paths); VideoToolbox: encoder construct, H.264 BGRA round-trip, HEVC 10-bit / 4:4:4 probe matrix, `videotoolbox_round_trip_chroma_matrix` cross-checks the encode side against an independent encode→decode→IOSurface-fourcc check (catches silent downsample regressions), self-decodable-IDR mid-session recovery. Capability discovery and preference-list negotiation tests live in `tether-probe`. |
| `tether-probe` | 10 default + 0 `#[ignore]` (today) | `PipelineStage` exhaustiveness, `ProfileSupport` helper accessors, preference order (5-entry list, 10-bit-first), `pick_supported_profile` (best mutual / fallback / disjoint / empty / forward-compat unknown bit_depth), `probe_client_does_not_mirror_encode_bit_into_decode_field` (the Mac 4:4:4 decode-without-encode invariant). |
| `tether-input` | 9 | Modifier tracking, HID routing, cursor normalization. |
| `tether-session` | 5 | `IdrSignal` coalescing + clone-share; `EncodeStatsWindow` emit / idle / accumulate. |
| `tether-render` | 16 + 1 `#[ignore]` | Cursor letterbox clipping, aspect ratio; `LatestFrame` Send+Sync + drop-oldest displacement; transfer_kind dispatch table pin; `range_kind_for(bit_depth, layout)` dispatch table pin + algebraic check that 10-bit limited-range breakpoints land white at 1.0 and black at 0.0; `render_layout_for(chroma, bit_depth)` dispatch table pin (incl. Yuv420 10-bit → Biplanar16 which the import path relies on for UV dimensioning); dma-buf zero-copy on hardware. |
| `tether-gpuconvert` | 3 default + 11 `#[ignore]` | `drm_fourcc_to_vk_format` table coverage incl. 10-bit biplanar plane fourccs (R16/GR32 → R16_UNORM/R16G16_UNORM) + 8-bit family regression + unknown-fourcc rejection; BGRA→NV12 + DMA-BUF round-trip with real Vulkan adapter. |
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
