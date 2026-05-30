# Tether

Low-latency, open-source remote desktop in Rust. Inspired by
[Parsec](https://parsec.app/), [Moonlight](https://moonlight-stream.org/),
and [Sunshine](https://github.com/LizardByte/Sunshine).

**Status:** pre-MVP, three platforms wired end-to-end over QUIC:

- **Linux** host → client — VAAPI hardware encode, PipeWire DMA-BUF
  capture, wgpu present (the most-exercised path; negotiates up to
  HEVC Main 4:4:4 10-bit).
- **macOS** host and client — ScreenCaptureKit capture + VideoToolbox
  encode + CGEvent input; VideoToolbox decode + Metal render.
- **Windows** host and client — DXGI Desktop Duplication capture +
  vendor-selected D3D11 encode (QSV / AMF / NVENC, Media Foundation
  fallback) + D3D11VA decode; loopback-verified, 4:2:0 only.

The audio pipeline is deferred on all platforms.

## Layout

- `apps/tether-host` — capture + encode + send
- `apps/tether-client` — receive + decode + present
- `crates/tether-protocol` — wire types, hello/handshake, control messages
- `crates/tether-transport` — QUIC datagrams + reliable streams
- `crates/tether-capture`, `tether-codec`, `tether-vaapi`,
  `tether-gpuconvert`, `tether-render`, `tether-input`, `tether-session`

## Build

```
make build       # cargo build --workspace
make test        # fast unit tests, no hardware
make test-hw     # VAAPI + render + gpuconvert correctness + benchmarks
make help        # full target list
```

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — channels, handshake,
  forward-compat hooks
- [docs/TESTING.md](docs/TESTING.md) — what the test matrix covers

## License

MIT — see [LICENSE](LICENSE).
