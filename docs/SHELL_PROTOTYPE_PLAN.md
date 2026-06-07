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

- ~~Pairing/auth crypto~~ — **now implemented** (landed after this plan was
  written): PIN-based first-contact pairing over the `start_pairing` /
  `revoke_peer` shell commands and the client's `--pin` flag, backed by SPAKE2 +
  mutual TLS. There is no `token` field; see `crates/tether-pairing`.
- Code signing / notarization.
- Clipboard, gamepad, and shell-level audio controls. The audio engine has since
  landed; this prototype still does not add mute/volume/device controls.
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

### Phase 2 — Tauri shell skeleton — DONE
5. **`apps/tether-shell`** (Tauri 2, **React + TS + Vite**, pnpm). The
   `src-tauri` crate is **excluded** from the Cargo workspace (not a member) so
   `cargo build --workspace` and Linux CI stay free of the webview/Tauri tree;
   it depends on `tether-ipc` by path. node/pnpm are pinned in `mise.toml`. ✓
6. **Supervisor module** (`src-tauri/src/supervisor.rs`). `tokio::process`:
   spawns an engine, a reader task parses stdout JSON-lines → `EngineEvent` →
   `app.emit("engine-status", {role, …})`; stop sends `{"cmd":"stop"}` on stdin
   + closes it (EOF backstop) + reaps with a 3 s force-kill; `kill_all` on app
   exit. Engine stderr inherited (logs in the dev terminal). ✓
7. **Tauri commands:** `start_host(test_pattern)`, `connect_client(addr, fp)`,
   `stop_engine(role)`. Binary discovery: `TETHER_ENGINE_DIR` env or the
   workspace `target/debug` relative to the `tauri dev` CWD. Bundled sidecar
   resolution is a follow-up. ✓

### Phase 3 — Tray — DONE
8. Tauri tray with Show/Quit menu and a "Tether" tooltip; Quit triggers
   `RunEvent::Exit` → `kill_all`. (Live status-in-tooltip is a small
   follow-up — the state already flows via `engine-status`.) ✓

### Phase 4 — Frontend (functional) — DONE
9. **Host panel:** test-pattern toggle, Start/Stop hosting, shows listening
   addr + fingerprint (mono) + connected-peer status, surfaces errors. ✓
10. **Client panel:** addr + fingerprint fields, Connect/Disconnect, live
    connecting/connected/disconnected/error status. ✓ (Stats deferred with the
    engine-side `Stats` event.)

### Phase 5 — Loopback validation (Windows, one machine) — DONE
11. Verified end-to-end on one Windows box: `pnpm tauri dev`, **Start hosting
    (real capture)** → copy addr + fingerprint → Connect → native client window
    shows live desktop (HEVC/QSV) → close window exits cleanly → engines reaped.
    Full IPC lifecycle (`listening` → `connecting` → `peer_connected` →
    `connected` → `disconnected`) round-tripped through the supervisor into the
    UI. See `apps/tether-shell/README.md`.

    Two bugs found and fixed during validation:
    - **Client hung on window close.** The `--ipc` stdin stop-watcher parks a
      `tokio::io::stdin()` blocking read that can't be cancelled, so returning
      from `main` hung the runtime drop. Fixed: the normal-close + render-error
      paths now `std::process::exit` (matching the Ctrl-C handler). The host
      bind-failure path exits explicitly in IPC mode for the same reason.
    - **Black window with test pattern.** Test pattern forces the H.264 floor;
      this box's only H.264 encoder (`h264_mf`) fails the self-decodable-IDR
      check, so the host sent `Goodbye(InternalError)` immediately. Not a shell
      bug — real capture negotiates HEVC/QSV and works. The UI's test-pattern
      toggle now defaults **off**, and host bind failures surface as an
      `error` IPC event instead of dying silently.

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
