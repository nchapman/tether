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
# 1. Build the engine binaries the shell will spawn.
cargo build -p tether-host -p tether-client

# 2. Launch the shell (Vite dev server + Tauri window).
cd apps/tether-shell && pnpm tauri dev
```

The supervisor finds the engines in the workspace `target/debug` (override with
`TETHER_ENGINE_DIR`). To exercise the full loopback on one machine: **Start
hosting** with "Test pattern" on → copy the fingerprint → paste the address +
fingerprint into the client panel → **Connect**. The client opens its own
native video window. **Disconnect** / closing the shell tears the engines down.

## Layout

- `src/` — React frontend (`App.tsx` = host + client panels).
- `src-tauri/src/lib.rs` — Tauri commands (`start_host`, `connect_client`,
  `stop_engine`) + tray + exit cleanup.
- `src-tauri/src/supervisor.rs` — engine spawn / stdout-reader / stop.
