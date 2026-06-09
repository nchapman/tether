# tether-shell

The Tether control-plane UI: a Tauri app (React + TS + Vite) that lives in the
system tray and **supervises the native engine processes** (`tether-host`,
`tether-client`). The webview is chrome only — connection forms, status, tray.
The actual video session is the engine's own native winit/wgpu window in a
separate process; nothing renders video through the webview.

## How it fits together

```
tether-shell (this app)            spawns + stdio JSON-lines
  ├─ webview UI (host/client panels)  ────────────────┐
  └─ supervisor (src-tauri)                            ▼
                                   tether-host --ipc / tether-client --ipc
                                   (engine: capture/encode or decode/present)
```

The shell speaks the `tether-ipc` protocol to each engine: it reads lifecycle
`EngineEvent`s from the child's **stdout** and re-emits them to the UI as
`engine-status` events; it writes `{"cmd":"stop"}` to **stdin** to tear an
engine down. Closing the shell drops each child's stdin → EOF → the engine
stops on its own (so a crashed shell never orphans an engine).

`src-tauri` is **excluded from the Cargo workspace** on purpose — it keeps
`cargo build --workspace` and Linux CI free of the webview/Tauri dependency
tree. Build it through its own pipeline (below).

## Prerequisites

`mise install` (pins node + pnpm + rust). On Windows, WebView2 ships with
Win11. Then `pnpm install` in this directory.

## Run (dev, Windows loopback)

```sh
# Build the engine binaries, then launch the dev-channel shell.
mise run shell
```

`mise run shell` runs the shell as **Tether Dev** (`app.tether.shell.dev`), uses
the dev trust store (`~/.tether-dev` unless `TETHER_CERT_DIR` overrides it), and
starts the host on `0.0.0.0:7384` so it can coexist with an installed release
host on `7374`. Bare addresses in the dev shell also default to `:7384`; release
builds default to `:7374`. The supervisor finds the engines in the workspace
`target/debug` (override with `TETHER_ENGINE_DIR`). To exercise the full
loopback on one machine:

1. **Start hosting** (leave "Test pattern" **off** — real capture negotiates
   HEVC, which works on the Windows/QSV path; the test pattern forces the H.264
   floor, whose only Windows encoder fails the self-decodable-IDR check and
   shows a black window).
2. On the host, **Add a device** → a PIN appears with a countdown.
3. In the client panel, enter the host **address + PIN** → **Pair & connect**.
   The client opens its own native video window.
4. Later runs are one-click: enter just the **address** (no PIN) and
   **Connect** — the host is pinned in known-hosts from the first pairing.

**Disconnect** / closing the shell tears the engines down. **Revoke** a device
on the host to drop its access (and any live session) immediately.

## Layout

- `src/` — React frontend (`App.tsx` = host + client panels).
- `src-tauri/src/lib.rs` — Tauri commands (`start_host`, `connect_client`,
  `stop_engine`, `start_pairing`, `revoke_peer`, `list_peers`) + tray + exit
  cleanup.
- `src-tauri/src/supervisor.rs` — engine spawn / stdout-reader / stop.

## Linux prerequisites

The shell lives in the system tray (`TrayIconBuilder` in `src-tauri/src/lib.rs`),
so Tauri `dlopen`s the Ayatana app-indicator library at startup. If it's missing,
`mise run shell` builds and launches but then panics with:

```
Failed to load ayatana-appindicator3 or appindicator3 dynamic library
libayatana-appindicator3.so.1: cannot open shared object file: No such file or directory
```

Install the library before running the shell:

- **Arch / CachyOS:** `sudo pacman -S libayatana-appindicator`
- **Debian / Ubuntu:** `sudo apt install libayatana-appindicator3-1`
- **Fedora:** `sudo dnf install libayatana-appindicator-gtk3`

(The standard Tauri Linux prerequisites — `webkit2gtk-4.1`, `libsoup-3.0`, etc. —
are also required; see the Tauri docs.)
