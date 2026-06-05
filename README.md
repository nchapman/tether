# Tether

Low-latency, open-source remote desktop in Rust. Inspired by
[Parsec](https://parsec.app/), [Moonlight](https://moonlight-stream.org/),
and [Sunshine](https://github.com/LizardByte/Sunshine).

**Status:** early (v0.2.1). Three platforms are wired end-to-end over QUIC:

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

Dev tasks run through [mise](https://mise.jdx.dev/) (already required — it
pins the Rust/Node toolchain), so the same commands work on Linux, macOS, and
Windows. First build on a fresh clone fetches the pinned static FFmpeg once:

```
mise run ffmpeg    # download + stage FFmpeg (idempotent; build tasks call it)
mise run build     # cargo build --workspace
mise run test      # fast unit tests, no hardware
mise run test-hw   # this platform's hardware tests (codec + render + gpuconvert)
mise run probe     # print this host's codec capability matrix
mise run shell     # run the Tauri shell in dev mode
mise run package   # build local installer bundles
mise tasks         # full task list
```

## Linux host permissions

A Linux host asks for two permissions, both **once, on first use** — no
terminal commands required:

- **Screen capture** is granted through the xdg-desktop-portal "share your
  screen" dialog on the first connection. Tether stores the portal restore
  token under `~/.tether/` and replays it, so the dialog won't appear again
  (until you revoke it in your desktop's privacy settings).
- **Input injection** needs access to `/dev/uinput`, gated by a udev rule:
  - The **deb/rpm packages install the rule at install time**, so input
    just works — no prompt at all.
  - The **AppImage** (and running from source) request it the first time a
    client connects: a system **PolicyKit dialog** appears to authorize the
    one-time setup, exactly like the screen-share prompt. Approve it once
    and input works from then on, no per-session dialog.

Headless hosts with no active local seat must add the user to the `input`
group (`sudo usermod -aG input $USER`) and log back in — `uaccess` only
applies to the user of an active seat.

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — channels, handshake,
  forward-compat hooks
- [docs/CODEC_CAPABILITIES.md](docs/CODEC_CAPABILITIES.md) — per-backend
  codec/chroma/bit-depth matrix
- [docs/TESTING.md](docs/TESTING.md) — what the test matrix covers
- [docs/RELEASING.md](docs/RELEASING.md) — how releases are cut

## License

MIT — see [LICENSE](LICENSE).
