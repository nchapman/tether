# Tether

Low-latency, open-source remote desktop in Rust. Inspired by
[Parsec](https://parsec.app/), [Moonlight](https://moonlight-stream.org/),
and [Sunshine](https://github.com/LizardByte/Sunshine).

**Status:** early (v0.1.0). Three platforms are wired end-to-end over QUIC:

- **Linux** host → client — VAAPI hardware encode, PipeWire DMA-BUF
  capture, wgpu present (the most-exercised path; negotiates up to
  HEVC Main 4:4:4 10-bit).
- **macOS** host and client — ScreenCaptureKit capture + VideoToolbox
  encode + CGEvent input; VideoToolbox decode + Metal render.
- **Windows** host and client — DXGI Desktop Duplication capture +
  vendor-selected D3D11 encode (QSV / AMF / NVENC, Media Foundation
  fallback) + D3D11VA decode; loopback-verified, 4:2:0 only.

Audio is deferred on all platforms.

## Install

Tether ships as a single **Tether shell** installer per platform — a
system-tray app that supervises the native host/client engines. Installers
are published to [GitHub Releases](https://github.com/nchapman/tether/releases).

> OS code signing isn't set up yet, so first launch needs a manual override:
> **macOS** right-click → Open (ad-hoc signed); **Windows** dismiss the
> SmartScreen prompt. Auto-update is signed independently and verified.

## The shell

The user-facing product is `apps/tether-shell`: a Tauri (React + TS) tray app
that supervises the engines (`tether-host`, `tether-client`) over the
`tether-ipc` protocol. The webview is chrome only — connection forms, status,
tray; the video session is the engine's own native winit/wgpu window in a
separate process.

## Layout

- `apps/tether-host` — capture + encode + send
- `apps/tether-client` — receive + decode + present
- `apps/tether-shell` — Tauri tray UI that supervises the engines
- `crates/` — `tether-protocol`, `tether-transport`, `tether-capture`,
  `tether-codec`, `tether-decode`, `tether-gpuconvert`, `tether-vaapi`,
  `tether-render`, `tether-scaler`, `tether-input`, `tether-session`,
  `tether-probe`, `tether-pairing`, `tether-ipc`

## Build

First build on a fresh clone fetches the pinned static FFmpeg once:

```
make ffmpeg      # download + stage FFmpeg (idempotent; build targets call it)
make build       # cargo build --workspace
make test        # fast unit tests, no hardware
make test-hw     # VAAPI + render + gpuconvert correctness + benchmarks
make shell       # run the Tauri shell in dev mode
make package     # build local installer bundles
make help        # full target list
```

## Linux host setup

A Linux host needs one-time access to `/dev/uinput` so it can inject
keyboard/mouse input without an xdg-desktop-portal prompt on every
connection. Run once:

```
tether-host --setup-input    # installs a udev rule via pkexec (one auth prompt)
```

or, equivalently, `make install-udev`. After this, input works with no
per-session permission dialog. The screen-capture grant persists the same
way automatically — Tether stores the portal restore token under
`~/.tether/` and replays it, so the "share your screen" dialog appears
only on the first connection (until you revoke it in your desktop's
privacy settings).

Headless hosts with no active local seat must additionally add the user
to the `input` group (`sudo usermod -aG input $USER`) and log back in.

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — channels, handshake,
  forward-compat hooks
- [docs/CODEC_CAPABILITIES.md](docs/CODEC_CAPABILITIES.md) — per-backend
  codec/chroma/bit-depth matrix
- [docs/TESTING.md](docs/TESTING.md) — what the test matrix covers
- [docs/RELEASING.md](docs/RELEASING.md) — how releases are cut

## License

MIT — see [LICENSE](LICENSE).
