# Tether Shell Prototype — Plan

Branch: `windows-shell-prototype`. Goal: stand up the **multi-process UI
architecture** so we can drive a full host↔client session from a single
installable app. Prototype on **Windows**; port to macOS then Linux after the
boundary is proven.

## Architecture (decided)

```
┌─────────────────────────────────────────────┐
│  Tether shell  (Tauri: webview UI + tray)    │  control plane only, no video
│  - host panel / client panel / settings      │  one per machine
│  - spawns + supervises engine processes       │
└───────────────┬──────────────────┬───────────┘
        spawns  │                  │  spawns
                ▼                  ▼
   ┌─────────────────────┐  ┌──────────────────────────────┐
   │  Host engine        │  │  Client engine               │
   │  (tether-host)      │  │  (tether-client)             │
   │  headless, no window│  │  winit + wgpu video window    │
   └─────────────────────┘  └──────────────────────────────┘
        stdout/stdin ▲            stdout/stdin ▲
        JSON-lines IPC (shell ↔ engine)
```

- The **shell is the one constant**; "host" vs "client" is just which engine it
  spawns. Running both roles at once = shell supervising one of each.
- The **video window stays winit/wgpu in the engine process** — never inside the
  webview. Separate process because winit's event loop can't coexist with
  Tauri's `tao` loop in one process (single main run loop on macOS). Bonus: an
  engine crash can't take down the shell/tray, and the host can run headless
  after the UI window closes.
- **In-window overlays** (stats, disconnect button, reconnect spinner) are
  **egui composited onto the engine's existing wgpu surface** — not this
  prototype's scope, but the reason we don't need the webview over the video.

## Explicitly out of scope for the prototype

- Pairing/auth crypto (addr + fingerprint entered manually). The IPC `connect`
  command carries an optional `token` field so adding it later is additive.
- Code signing / notarization.
- Audio, clipboard, gamepad — no new engine features at all.
- Windows service / session-0 secure desktop.
- Pretty styling. Functional UI only.

## IPC contract (stdio JSON-lines)

One JSON object per line. **stdout = IPC only; logs move to stderr.** Chosen over
named pipe / UDS because stdio is identical on all three platforms and child
lifecycle is free (pipe closes when child dies → disconnect detection).

Shell → engine (stdin):
- `{"cmd":"stop"}` — graceful shutdown. (Stdin EOF when the shell dies also
  triggers shutdown.)

Engine → shell (stdout), `event` tagged:
- host: `listening` `{addr, fingerprint}`, `client_connected` `{peer}`,
  `client_disconnected` `{reason}`
- client: `connecting`, `connected` `{negotiated_profile, host}`, `disconnected`
  `{reason}`
- both: `stats` `{fps, bitrate_kbps, frames_dropped, ...}`, `error` `{message}`

Shared serde types live in a tiny new crate **`crates/tether-ipc`** so the shell
and both engines can't drift. Round-trip unit test per the repo's
"wire round-trip first" convention.

## Work breakdown

### Phase 1 — Make engines shell-drivable (minimal surgery) — DONE
Both mains are large (host 3478 / client 1005 lines); touched only the edges.

1. **`crates/tether-ipc`** — `EngineEvent` (listening / peer_connected /
   peer_disconnected / connecting / connected / disconnected / error),
   `ShellCommand::Stop`, and a `Reporter` (`Human` | `Json`). Round-trip +
   flat-tag tests. ✓
2. **Status routed through `Reporter`.** Host emits listening / peer_connected /
   peer_disconnected; client emits connecting / connected / disconnected / error.
   Human mode reproduces the old `println!` text. ✓
3. **Logs off stdout in IPC mode.** Both `init_tracing(ipc)` calls switch the
   non-blocking writer to **stderr** when `--ipc`; stdout is reserved for
   JSON-lines. Verified live: host stdout was pure JSON, logs on stderr. ✓
4. **`--ipc` flag + shutdown.** Host races a stdin `Stop`/EOF notify in both the
   accept loop and the in-session select; client mirrors the Ctrl-C handler
   (Goodbye + process exit) since the render loop owns the main thread. Stdin
   EOF = implicit stop (crashed shell never orphans an engine), verified live.
   Without `--ipc` the CLIs behave exactly as before. ✓

**Deferred to a Phase 1 fast-follow:** the `Stats` event (fps / bitrate /
drops). Held back to honor the no-scaffolding rule — it lands when the shell's UI
panel actually consumes it. The host already logs a "client stats" cadence to
hook.

### Phase 2 — Tauri shell skeleton
5. **New app `apps/tether-shell`** (Tauri 2). Cargo crate at
   `apps/tether-shell/src-tauri` added to the workspace `members`. Frontend:
   **vanilla TS + Vite** (no UI framework — keep the prototype light).
6. **Supervisor module (Rust backend).** `std::process::Command` (full control
   over stdin/stdout, simpler than the shell plugin): spawn engine, reader task
   parses stdout JSON-lines → `app.emit("engine-status", …)`, writer holds stdin
   for `stop`, tracks the child handle, kills on stop + on app exit.
7. **Tauri commands:** `start_host(opts)`, `connect_client(addr, fp)`,
   `stop(role)`. Binary discovery: dev → `target/debug/`; bundled → Tauri
   **sidecar/externalBin** alongside the shell exe (note for bundling, not
   wired yet).

### Phase 3 — Tray
8. Tauri 2 built-in tray: menu = Show/Hide window, hosting/connection status
   line, Quit. Tooltip/icon reflects state from `engine-status` events. Quit
   kills supervised engines.

### Phase 4 — Frontend (functional only)
9. **Host panel:** Start/Stop hosting; show listening addr + fingerprint
   (copyable) + connected-peer status.
10. **Client panel:** addr + fingerprint fields; Connect/Disconnect; live
    status + stats. Subscribe to `engine-status`, render state.

### Phase 5 — Loopback validation (Windows, one machine)
11. Start host in shell → copy fingerprint → connect client in same shell →
    native video window appears (start with `--test-pattern`, then real capture)
    → stats flow back into the UI → Disconnect closes the window → Quit shell
    kills both engines. Document the steps in `docs/SHELL.md`.

### Phase 6 — Tidy / CI note
12. Ensure `cargo build -p tether-host -p tether-client` (with `--ipc`) and the
    Tauri build both succeed on Windows. Document prereqs (WebView2 ships on
    Win11; node for the frontend). Full installer/CI pipeline is a follow-up.

## De-risking order (what the spike actually proves)
The cheapest first slice: **Connect button → spawn `tether-client --ipc` →
stream its JSON status into the webview + tray.** That exercises the exact
shell↔engine boundary everything else hangs off. Build that end of Phase 2
first, before the host panel or styling.

## Open questions to settle while building
- Binary discovery path for dev vs bundled (sidecar layout).
- Whether `stats` should be throttled in the engine before hitting IPC (the host
  already logs "client stats" periodically — reuse that cadence).
- Frontend: confirm vanilla TS is enough, or pull in a tiny component lib if the
  two panels get fiddly.
