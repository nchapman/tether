# Tether

Tether is a low-latency remote desktop project written in Rust.

It is pre-MVP. The core host/client engines are wired end-to-end on Linux,
macOS, and Windows, but the project still expects hardware video codecs and a
LAN-style direct connection. There is no software codec fallback.

## Platforms

**Linux**

- Host: PipeWire capture, DMA-BUF GPU frames, VAAPI or NVENC encode.
- Client: hardware decode, dma-buf export, wgpu render.
- Best-tested path. Can negotiate HEVC Main 4:4:4 8-bit and 10-bit when the
  live probe passes. AV1 4:2:0 8-bit and 10-bit are preferred above HEVC 4:2:0
  when supported.

**macOS**

- Host: ScreenCaptureKit capture, Metal conversion, VideoToolbox HEVC encode,
  CGEvent input.
- Client: VideoToolbox decode, IOSurface import, Metal/wgpu render.
- Hosts currently advertise HEVC 4:2:0 Main/Main10. Clients can decode and
  render Linux-host HEVC Main 4:4:4 8-bit and 10-bit.

**Windows**

- Host: DXGI Desktop Duplication capture, D3D11 video processing, hardware
  encode selected by GPU vendor: QSV, AMF, NVENC, then Media Foundation.
- Client: D3D11VA decode and native D3D11 render.
- Supports H.264, HEVC, and AV1 4:2:0 NV12/P010 where the hardware supports it.
  Windows does not advertise 4:4:4.

System-output audio is wired on all three platforms: platform capture,
Opus over unreliable datagrams, jitter-buffered playback through `cpal`.
Audio is on when both peers support it; host and client `--no-audio` flags opt
out.

## User-facing app

`apps/tether-shell` is the desktop shell: a Tauri tray app that starts and
supervises `tether-host` and `tether-client` over the `tether-ipc` protocol.
The shell is control-plane only. Video sessions use the native engine window.

Installers are published on
[GitHub Releases](https://github.com/nchapman/tether/releases). OS code signing
is not set up yet, so first launch may need a manual override on macOS or
Windows.

## Repository layout

- `apps/tether-host` - capture, encode, send, input injection.
- `apps/tether-client` - receive, decode, render, input capture.
- `apps/tether-shell` - Tauri tray UI and engine supervisor.
- `crates/tether-protocol` - wire types and media fragmentation.
- `crates/tether-transport` - QUIC transport and loopback test channels.
- `crates/tether-session` - host/client session policy.
- `crates/tether-capture` - PipeWire, ScreenCaptureKit, DXGI, test pattern.
- `crates/tether-codec` / `crates/tether-decode` - hardware encode/decode.
- `crates/tether-gpuconvert`, `tether-render`, `tether-scaler` - GPU format
  conversion, presentation, and scaling.
- `crates/tether-audio`, `tether-input`, `tether-probe`, `tether-pairing`,
  `tether-ipc`, `tether-vaapi` - focused support crates.

## Build

Use `mise`; it pins the Rust and Node toolchains and owns the common tasks.
On a fresh clone, the build tasks fetch the pinned static FFmpeg package into
`vendor/ffmpeg/`.

```sh
mise run ffmpeg     # fetch the pinned FFmpeg artifact
mise run build      # cargo build --workspace
mise run test       # fast lib tests, no hardware
mise run test-all   # full no-hardware workspace tests
mise run test-hw    # current platform's ignored GPU/video tests
mise run probe      # print this host's codec capability matrix
mise run shell      # run the Tauri shell in dev mode
mise run package    # build local installer bundles
mise tasks          # list all tasks
```

## Linux host permissions

Linux screen capture uses the desktop portal. The first connection may show the
standard screen-share dialog; Tether stores the portal restore token under
`~/.tether/` for release builds and `~/.tether-dev/` for `mise run shell`.

Linux input injection uses `/dev/uinput`. Deb/rpm packages install the udev rule
at install time. The AppImage and source builds can request the one-time setup
through PolicyKit when a client connects. Headless hosts with no active local
seat need the user in the `input` group.

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - system design and invariants.
- [docs/CODEC_CAPABILITIES.md](docs/CODEC_CAPABILITIES.md) - platform codec,
  chroma, and bit-depth matrix.
- [docs/PROTOCOL_V1.md](docs/PROTOCOL_V1.md) - current wire protocol.
- [docs/TESTING.md](docs/TESTING.md) - test matrix and conventions.
- [docs/RELEASING.md](docs/RELEASING.md) - release and packaging flow.

## License

See [LICENSE](LICENSE).
