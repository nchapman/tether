# Tether — UX Design (0.3.0)

This document is the reference for the 0.3.0 UX redesign of the Tether shell
(`apps/tether-shell`). It supersedes the prototype layout described in
`docs/SHELL_PROTOTYPE_PLAN.md` for everything user-facing. The engine
binaries (`tether-host`, `tether-client`), the IPC contract (`tether-ipc`),
and the multi-process supervision model
(`apps/tether-shell/src-tauri/src/supervisor.rs`) are unchanged in spirit —
this is a redesign of the *control-plane UI*, not the streaming pipeline.

The goal: make Tether feel like a real, well-crafted remote-desktop product a
tester would keep installed — not a developer test harness. It stays
deliberately simple, but every surface should feel like a native desktop app,
not a webpage in a window.

This plan reflects a design review by a senior product designer; the decisions
below are settled.

---

## The problem we're fixing

The 0.2 shell renders both roles at once in a single window: a "Host this
machine" panel stacked on a "Connect to a host" panel. That conflates two
jobs with opposite rhythms and leaks developer vocabulary ("host", "client",
"fingerprint", "test pattern", `127.0.0.1:7654`). Three concrete gaps:

1. **The client has no visible memory.** `known_hosts.json` already persists
   every host the client has paired with (address, cert fingerprint, label,
   pairing time) and the engine can one-click reconnect via the `Resume` path
   — but the UI never surfaces it, so the human retypes an address and PIN
   every session. This is the most fundamental missing piece and the cheapest
   to fix.
2. **The tray is inert.** It offers only Show / Quit — no status, no quick
   actions — even though the supervisor keeps engines alive when the window is
   closed, which makes the tray the natural home for a set-and-forget host.
3. **No separation of concerns.** When connecting *out* to another machine,
   you should not have to think about hosting *this* one.

---

## Organizing principle: two roles, two rhythms

| | Host ("share this computer") | Client ("connect to a computer") |
|---|---|---|
| Frequency | Set once, runs for weeks | Every session, actively |
| Attention | Background / ambient | Foreground / primary |
| Natural home | **The tray** | **The window** |
| Question it answers | "Who can reach me, and is anyone here now?" | "Which machine do I want, and let me in" |

Hosting is a **background utility that lives in the tray** (with a small
settings sheet for the rare change); connecting is the **foreground app**. The
window *is* the client.

### Settled product decisions

- **Connect-only window.** The main window is the client (a saved-connections
  address book). Hosting has no tab in the main window — it's toggled from the
  tray and configured in a settings sheet reached from a gear button.
- **Sharing remembers last state.** OFF on first run (no machine is reachable
  until its owner opts in); sticky across restarts after the user turns it on.
- **Single-click connects.** Clicking a saved-computer row's name region
  reconnects immediately (the pinned-cert `Resume` path). Rename / Forget /
  Copy live in a separate `⋯` overflow target so they never mis-launch a
  connection.
- **Hide-on-Connected handoff.** On `EngineEvent::Connected`, the shell hides
  its own window so the engine's native video window is alone on screen. We
  accept the brief (~0.3–0.8s) gap before the video window paints rather than
  add a first-frame IPC event; revisit if the gap feels rough in testing.
- **Reachability is deferred to 0.3.1.** Tether hosts listen on QUIC (UDP),
  not TCP, so a cheap TCP-connect probe would report every host as offline;
  an accurate probe means a real QUIC handshake (e.g. a `tether-client --probe`
  mode). Not worth blocking 0.3.0 — rows lead with name + recency, and the
  leading dot reflects *session* state only (idle / connecting / connected),
  not network reachability.
- **Teal "signal" identity.** Accent `#2DD4BF`, doubling as the "connected"
  status color — in a connection product, connected *is* the brand color.

---

## Surface 1 — The window (Connect)

The window is the client. Nothing about hosting competes for attention.

```
┌──────────────────────────────────────────────┐
│  Tether                            [ ⚙ ] [ ＋ ]│
├──────────────────────────────────────────────┤
│     Office desktop                      ⌘1  ⋯ │  ← click name = connect (Resume)
│     192.168.1.10 · 2h ago                     │
│     Living room PC                      ⌘2  ⋯ │
│     10.0.0.5 · yesterday                      │
│     Work laptop                             ⋯ │
│     paired Mar 28                             │
├──────────────────────────────────────────────┤
│  ◉ Streaming HEVC Main · Office desktop  [Disconnect] │  ← live-session bar, only while connected
└──────────────────────────────────────────────┘
```

- **The address book is the home screen.** Each row is a known host from
  `known_hosts.json`. The name leads; address + relative time are a single dim
  subtitle. A leading dot reflects *session* state only — hidden when idle, a
  spinner while connecting, accent-filled when this row is the live session.
  (Network reachability, a green/grey "online" dot, is deferred to 0.3.1; see
  decisions above.)
- **Single-click the name region → connect** (pinned-cert `Resume`, no PIN).
  The clicked row IS the progress indicator: its dot/chevron becomes an
  indeterminate spinner during `Connecting`, then the shell hides on
  `Connected` (the handoff above). `⌘1`–`⌘9` quick-connect to the first rows
  (optional; ship arrows + Enter first).
- **`⋯` overflow** (a separate target that never connects): **Rename**,
  **Forget** (with confirm — it drops the pinned cert), **Copy address**.
- **`＋ Add a computer`** is the *only* first-contact entry point (see Surface
  2b). It's the visually dominant titlebar action; `⚙` is a quiet ghost icon.
- **Live-session bar** (bottom, now-playing style, accent-tinted, persistent
  while a session is live) shows the negotiated `profile` from
  `Connected{profile}` plus **Disconnect**. Testers want to confirm "am I
  actually on HEVC?" — show it.
- **Switching hosts mid-session:** only one client session exists (the
  supervisor enforces one client engine). Clicking another row while connected
  confirms: "Disconnect from Office desktop and connect to Living room PC?"

**Empty state:**

```
        🖥  (single neutral line glyph, not emoji)

        No computers yet.
        Add one to connect.

            [ ＋ Add a computer ]

   You'll need its address and a pairing PIN
   from that computer.
```

---

## Surface 2a — The sharing-settings sheet (Host)

Reached from `⚙`. Reads like a system preference: one switch and minimal
reference info. The fingerprint ("safety code") is **not** on the default
path — the PIN flow handles the trust ceremony, and surfacing the fingerprint
re-introduces the vocabulary/clutter the redesign retires. It lives under
Advanced for debugging.

```
   ┌─ Sharing ─────────────────────────────────┐
   │  Allow remote connections      [ ●═══ ON ] │
   │                                            │
   │  This computer:  Nick's MacBook            │
   │  Address:        192.168.1.42:7654  [Copy] │
   │  ● Waiting for someone to connect…         │  ← live status
   │                                            │
   │  Paired devices                  [+ Add]   │
   │   Jane's MacBook   today                ⋯  │  ← Revoke lives in ⋯
   │   Work laptop      Mar 28               ⋯  │
   │                                            │
   │  ▸ Advanced  (safety code, test pattern)   │
   └────────────────────────────────────────────┘
```

- Toggle drives `start_host` / `stop_engine` **and** persists the
  sharing-enabled preference. The tray's "Share this computer" toggle and this
  one are **the same state** (one source of truth: the pref + live engine
  state) — they must never disagree.
- A host operator needs exactly two things to onboard a new device: the
  **address** (to tell the other person) and the **PIN** (generated on demand
  via Add a device). Both have one-click Copy.
- Paired devices, **Add a device** (`StartPairing` → `PairingPin`), and
  **Revoke** (in `⋯`, with confirm) map to existing IPC.
- **Test pattern** and the safety code move under **Advanced**.

## Surface 2b — Add a computer (first contact)

First contact is the highest-friction, highest-stakes, rarest action — a
focused sheet, single screen (not a wizard), that frames the trust ceremony.

```
┌─ Add a computer ─────────────────────────┐
│  Address                                  │
│  ┌──────────────────────────┬─────────┐  │
│  │ 192.168.1.10             │ :7654   │  │  ← port pre-filled + dimmed
│  └──────────────────────────┴─────────┘  │
│                                           │
│  Pairing PIN                              │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┐                │  ← segmented OTP-style, auto-advance,
│  │  │  │  │  │  │  │  │  │   8 digits      │     paste distributes across boxes
│  └──┴──┴──┴──┴──┴──┴──┴──┘                │
│  Open Tether on the computer you want to  │
│  reach, turn on sharing, and read the PIN │  ← directional copy: PIN is on the OTHER machine
│  it shows under "Add a device."           │
│                                           │
│  Name (optional)      Office desktop      │
│                                           │
│              [ Cancel ]  [ Pair → ]       │
└────────────────────────────────────────────┘
```

- Segmented PIN field signals "one-time code / ceremony." `PIN_DIGITS = 8`
  (see `tether_pairing`).
- Never show `127.0.0.1` as the example in a shipped build — use
  `192.168.1.10`. Pre-fill `:7654` dimmed so users type just the address.
- The sheet shows its **own** connecting/error state in place (don't dismiss
  and hope). On success it dismisses and the new row appears in the address
  book in its connecting→connected beat. On failure the error appears in the
  sheet against the relevant field (see Edge states).

---

## Surface 3 — The tray (host home + quick re-entry)

The tray is where the host role lives day to day and the fastest path back
into a connection.

```
  ● Sharing on · Jane connected      ← status header (host + client state)
  ─────────
  Connect to            ▸  Office desktop / Living room PC / New connection…
  ─────────
  ☑ Share this computer              ← toggles start_host / stop_engine (shared state)
  Add a device…                      ← when sharing; force-shows the window
  ─────────
  Open Tether
  Quit
```

- **Icon encodes host state only** — three variants: idle, sharing-idle (dot),
  sharing-active (accent-filled dot). Client-session state is conveyed by the
  header text, not the icon (a client session has its own visible video
  window; don't overload the tray glyph with two orthogonal booleans).
- **Connect to ▸** lists up to ~5 recents by `last_connected_unix` for
  one-click reconnect without opening the window; **New connection…** opens the
  Add sheet.
- **Add a device…** must **force-show the main window** so the `PairingPin`
  event has a surface to display (pairing initiated from a closed window
  otherwise has nowhere to show the PIN).
- Tray menu, tooltip, and icon are rebuilt from live engine state: the
  supervisor already receives every `engine-status` event; it fans them out to
  both the webview and a tray-state updater.

---

## Status, errors, and edge states

The shell must **never echo a raw engine error** (`String(e)`) — that's the
biggest web-tell and a tester-confusion machine. Map `EngineEvent::Error`
messages to human copy via a small table in the shell, with a generic
fallback. States to handle:

| State | Behavior |
|---|---|
| Wrong PIN | In Add sheet, under the PIN field: "That PIN didn't match. Check it and try again." |
| Expired PIN | "That PIN expired. Ask for a new one on the other computer." |
| Host not sharing / refused | "Office desktop didn't answer. Make sure sharing is on there." |
| Host unreachable (timeout) | Row returns to idle with an inline transient "Couldn't reach Office desktop" — not a global banner. Never strand the spinner. |
| Revoked / forgotten by host | Resume fails → "Office desktop no longer recognizes this computer." Action: **Pair again** (opens Add sheet pre-filled with that address). |
| Sharing toggled OFF while a peer connected | Confirm: "Stop sharing? Jane is connected and will be disconnected." (Use `PeerConnected`/`PeerDisconnected` to know.) |
| Host-side PIN expiry | The displayed PIN visibly counts down, then "expired — Add a device again." |
| Engine fails to launch | One-time setup error (binary missing / broken install), not a transient toast. The supervisor returns a rich message. |
| Forget a saved computer | Confirm: "Forget Office desktop? You'll need the PIN to connect again." |

---

## Native feel — it must not feel like a webpage

A first-class requirement. Tether may have its own look (it need not imitate
any OS), but it must feel like a desktop app. Kill these web-tells:

- **Raw error strings** — map to human copy (above).
- **`cursor: pointer` on buttons** — native apps keep the arrow cursor on
  buttons; remove it (the 0.2 CSS has it).
- **Fuzzy focus glow** — replace the browser box-shadow with a tight **2px
  accent outline, 2px offset**, on `:focus-visible` only.
- **Chunky default scrollbar** — thin overlay scrollbar.
- **Slow transitions** — 120–160ms, not 200ms+.
- **Emoji icons** (`🖥`) — use one consistent line-icon set (Lucide or
  Phosphor), 1.5px stroke, 16px, monochrome.
- **Text selection / drag** on non-input chrome — disable. Disable the
  webview's default right-click context menu.
- **Overscroll bounce** — disable.
- **Faded disabled buttons** (`opacity: 0.45`) — use a desaturated-but-crisp
  disabled style.
- **Centered web column** — the address book is full-bleed, edge-to-edge rows;
  no `max-width` centered column.

Also: instant in-place view transitions (no page loads / FOUC); Esc closes the
sheet, Enter submits, ↑/↓ move the list selection; remember window
position/size; `-webkit-font-smoothing: antialiased`; respect
`prefers-reduced-motion`; honor system dark/light (dark ships first and
polished, light must at minimum not look broken).

---

## Visual spec

Dark-first identity. Light mode is a tint inversion of the same ramp; ship
dark first.

### Color — dark (primary)
```
Background (window)      #16181D
Surface / sheet          #1E2127
Surface raised (hover)   #252932
Border / hairline        #2C313B
Border strong            #3A404C

Text primary             #E6E9EF
Text secondary           #9BA1AD
Text tertiary / hint     #646B78

Accent (teal/signal)     #2DD4BF
Accent hover             #5EEAD4
Accent pressed           #14B8A6
Accent tint (bg)         rgba(45,212,191,0.12)

Status online/active     #2DD4BF   (connected = brand color)
Status connecting/wait   #F5B947   warm amber
Status error/offline     #F2645A   warm red
Success tick             #34D399
```

### Color — light (follow-up)
```
Background #F7F8FA   Surface #FFFFFF   Border #E4E7EC
Text #1A1D23 / #5A6270 / #98A0AD
Accent #0D9488   amber #C97A12   red #DC4438
```

### Typography
- UI: `-apple-system, "SF Pro Text", "Segoe UI Variable", "Segoe UI", system-ui, sans-serif`.
  Mono (address/PIN): `ui-monospace, "SF Mono", "Cascadia Code", Menlo, monospace`.
- Scale:
  ```
  Title (window/sheet header)   16px / 600 / -0.01em
  Row primary (host name)       14px / 550
  Row secondary (addr · time)   12px / 450  text-secondary
  Section label (PAIRED)        11px / 600 / 0.05em uppercase  text-tertiary
  Button                        13px / 550
  PIN display (host)            28px / 600 / 0.12em  mono
  ```

### Spacing & layout
- 4px grid: 4 / 8 / 12 / 16 / 20 / 24.
- Window min **360 × 480**, default **420 × 560**; remember position/size.
- Rows: full-bleed, **48px** (52px with two-line subtitle), 16px horizontal
  padding, 1px bottom hairline (`#2C313B`), no per-row card. List is flat on
  the window background.

### Radii / elevation
```
Inputs / buttons     8px
Row hover/selected   6px (inset, 4px horizontal margin)
Sheet / popover      12px
Status dot           full
List      flat, no shadow
Sheet     0 8px 32px rgba(0,0,0,0.45), slides down from titlebar
Live bar  no shadow, accent-tint bg, 1px top hairline
```

### Motion
```
Row hover / button       120ms ease-out
Sheet slide-in           180ms cubic-bezier(0.32, 0.72, 0, 1)
Sheet dismiss            140ms ease-in
Status dot cross-fade    160ms
Connecting spinner       900ms linear, indeterminate
Window-hide-on-connect   instant (no animation)
```
No motion on first paint; respect `prefers-reduced-motion` (drop slides, keep
fades).

---

## What's in / out for 0.3.0

**In:** connect-only window; saved-connections address book (the headline)
with single-click reconnect and `⋯` overflow; the Add-a-computer sheet with
segmented PIN; the sharing-settings sheet; tray status + quick-connect +
sharing toggle; the error/edge-state taxonomy; the native-feel pass; the teal
visual identity; vocabulary cleanup.

**Out / later:** network reachability dots (needs a QUIC handshake probe —
0.3.1); in-window video, multi-monitor pickers, file transfer/clipboard UI,
audio, fully-tuned light mode, `⌘1–9` numbered hints (keep arrows + Enter),
per-host OS-specific icons (one neutral glyph).

---

## Implementation phases

Each phase is independently shippable and testable.

### Phase 1 — Saved-connections address book (headline)

- Surface `known_hosts.json` to the shell. The client engine is transient, so
  the **shell reads/writes the file directly** via new Tauri commands:
  `list_known_hosts`, `rename_known_host`, `forget_known_host`. Resolve the
  client config dir the same way `tether-client` does (`TETHER_CERT_DIR`
  override, else `$HOME/.tether` / `$USERPROFILE`) — factor that resolver into
  a shared location so both agree.
- Add optional `last_connected_unix` to `tether_pairing::HostEntry` (serde
  `default`, backward compatible); the client engine stamps it on a successful
  connect. UI shows "2h ago", falling back to "paired <date>" when absent.
- Rebuild the window as the Connect address book: rows (single-click connect,
  status dot, name + dim subtitle), `⋯` overflow (Rename / Forget-with-confirm
  / Copy), empty state, live-session bar, hide-on-Connected, switch-host
  confirm.
- The Add-a-computer sheet (first contact): address + segmented PIN +
  optional name, directional copy, in-sheet connecting/error state.
- Tests: round-trip extended `HostEntry`; unit-test the shared config-dir
  resolver; test list/rename/forget against a temp known-hosts file.

### Phase 2 — Hosting moves to the tray + settings sheet — DONE

- Host UI lives in the `⚙` sharing-settings sheet (landed with Phase 1). Safety
  code under Advanced; Copy on address; Revoke into `⋯`; test pattern demoted.
- Persist a `sharing_enabled` shell preference (`prefs.rs`, stored next to the
  trust store). On launch the webview reads it via `get_prefs` — after the
  engine-event listeners are live — and re-spawns the host when true. First run
  = false. The posture is written through one path: `stop_engine(host)` clears
  it (explicit "sharing off", idempotent), and the **host's confirmed
  `listening` event** sets it on (persisting at spawn would leave a bind-failure
  auto-restarting every launch). A crash exits via the supervisor's EOF path,
  not `stop_engine`, so the posture stays sticky-on and auto-restore retries.
  The Phase 3 tray toggle reuses `start_host` / `stop_engine`, so it inherits
  the same single source of truth for free.
- Reframe "fingerprint" → "safety code".

### Phase 3 — Tray richness

- Tray-state updater driven by `engine-status`: status header, icon/badge
  states (host state only), tooltip.
- **Connect to ▸** recents submenu and **Share this computer** toggle;
  **Add a device…** force-shows the window.

### Phase 4 — Native-feel + visual polish pass

- Apply the visual spec and native-feel checklist end to end.
- Error/edge-state copy, empty states, confirms, motion, focus, keyboard.

---

## Invariants to preserve

- **The shell is chrome only.** Video is the engine's own native winit/wgpu
  window in a separate process; nothing renders video through the webview.
- **Pairing is host-authoritative and unchanged.** First contact is SPAKE2 +
  channel-bound confirmation; reconnect is the pinned-cert `Resume`. The UI
  reframes vocabulary but never weakens the trust model.
- **`known_hosts.json` is a client-side trust root.** It keeps its atomic,
  owner-only persistence (`tether_pairing::store`); shell writes go through the
  same `save`, never an ad-hoc file write.
- **IPC additions follow the repo convention.** Any new `tether-ipc` message
  gets a round-trip test; variants land only alongside the behavior that
  exercises them (no speculative variants). 0.3.0 adds no new `EngineEvent`
  (the hide-on-Connected handoff deliberately avoids a first-frame event).
