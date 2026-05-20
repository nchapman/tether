# Tether — Testing conventions

The foundation review (Q2 2026) made the test surface deliberately
small and predictable. This document codifies what we have, what
each layer covers, and how to extend it.

## Unit tests by crate

| Crate | Tests | Covers |
| --- | --- | --- |
| `tether-protocol` | 20 | Wire round-trips (ClientHello envelope, video packets, cursor, input), fragmenter / reassembler invariants, `HostFrameTimingBuilder` typestate, handshake forward-compat (unknown variant fails decode; trailing bytes fail decode). |
| `tether-transport` | 4 integration tests in `tests/roundtrip.rs` | QUIC handshake, control + datagram round-trip, fingerprint pinning. |
| `tether-codec` | 2 unit + 3 `#[ignore]` | SW H264 round-trip (test-only); VAAPI encoder/decoder/dma-buf-import on hardware. |
| `tether-input` | 9 | Modifier tracking, HID routing, cursor normalization. |
| `tether-session` | 5 | `IdrSignal` coalescing + clone-share; `EncodeStatsWindow` emit / idle / accumulate. |
| `tether-render` | 4 + 1 `#[ignore]` | Cursor letterbox clipping, aspect ratio; dma-buf zero-copy on hardware. |
| `tether-gpuconvert` | 11 `#[ignore]` | BGRA→NV12 + DMA-BUF round-trip with real Vulkan adapter. |

## Test categories

We tag tests by *what hardware they need*, encoded via `#[ignore]`
with an explanatory string:

- **Default-on** — run on every `cargo test`. No hardware assumptions
  beyond what the workstation already has (file I/O, in-process QUIC
  loopback via tokio).
- **`#[ignore = "requires …"]`** — needs real GPU or VAAPI. Run via
  `cargo test -- --ignored` on a host with the right capabilities.
  The ignore message names the requirement (`vainfo`, the Vulkan
  extension, etc.) so the next person can see at a glance whether
  their box should run them.

Today there are **15 ignored tests** across `tether-codec/vaapi`
(3), `tether-gpuconvert` (11), and `tether-render` (1). They are
real and load-bearing on hardware; they are not abandoned.

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
  Linux↔Linux LAN; no automation. A `tether-session` loopback
  integration test that runs `HostSession` and `ClientSession`
  over a `tokio::io::duplex` shim is the natural next addition
  once those session-level scaffolds exist.
- **The actual rendered pixels.** `tether-render`'s shader output
  is validated by eye, not by image-diff. A headless `wgpu::Surface`
  + image diff against a checked-in fixture is a worthwhile
  follow-up but not load-bearing today.
- **`tether-vaapi` directly.** The hand-rolled libva FFI bindings
  are tested transitively through `tether-codec/vaapi/tests.rs`.
  Direct tests would just exercise libva itself.
