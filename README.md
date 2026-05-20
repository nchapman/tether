# Tether

Low-latency, open-source remote desktop in Rust. Inspired by
[Parsec](https://parsec.app/), [Moonlight](https://moonlight-stream.org/),
and [Sunshine](https://github.com/LizardByte/Sunshine).

**Status:** pre-MVP. Linux host → Linux client works end-to-end over QUIC
with VAAPI hardware encode, DMA-BUF capture, and wgpu present. macOS host
and the audio pipeline are next.

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
