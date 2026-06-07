//! The system-tray menu and its live status.
//!
//! The shell lives in the tray: the window can be closed while a host keeps
//! sharing in the background. This module owns the tray's dynamic menu — a
//! status header, a "Connect to" quick-connect submenu of recent computers, a
//! "Share this computer" toggle, a "Pair a device…" entry, and the
//! show/quit items — plus the hover tooltip, and keeps them in sync with the
//! engines' lifecycle.
//!
//! The tray is a *second consumer* of the same `engine-status` /
//! `engine-exited` events the webview reads: it subscribes on the Rust side and
//! rebuilds itself, so it stays decoupled from both the supervisor internals
//! and the React state. Status is shown as menu text + tooltip only — no
//! per-state icon art (that's a later polish pass).

use std::sync::{Mutex, MutexGuard};

use serde::Deserialize;
use tauri::menu::{CheckMenuItemBuilder, Menu, MenuBuilder, MenuItemBuilder, SubmenuBuilder};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Listener, Manager};

use crate::known_hosts::{self, SavedHost};
use crate::supervisor::{Supervisor, ROLE_CLIENT, ROLE_HOST};

/// Stable id so the listeners can find the tray via `tray_by_id` to update it.
const TRAY_ID: &str = "main";
/// Menu-item id prefix for a quick-connect entry; the suffix is the host
/// address (which may itself contain `:`, so the suffix is taken by length).
const CONNECT_PREFIX: &str = "connect:";
/// How many recent computers the quick-connect submenu shows.
const MAX_RECENTS: usize = 8;

const ID_SHARE: &str = "share_toggle";
const ID_ADD_DEVICE: &str = "add_device";
const ID_DISCONNECT: &str = "disconnect";
const ID_SHOW: &str = "show";
const ID_QUIT: &str = "quit";

/// Tray-relevant slice of session state, mirrored from engine events. Kept in
/// managed state so the event listeners can update it and the rebuild can read
/// it back. Equality drives change-detection so we don't rebuild on every
/// unrelated event.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct TrayStatus {
    /// The host engine is listening — we're sharing.
    sharing: bool,
    /// A client connected to our host (its label/addr), or None while waiting.
    host_peer: Option<String>,
    /// The address we're connected to as a client, or None.
    client_target: Option<String>,
}

/// Newtype so it can be `manage`d without colliding with other `Mutex` state.
pub struct TrayState(Mutex<TrayStatus>);

impl Default for TrayState {
    fn default() -> Self {
        TrayState(Mutex::new(TrayStatus::default()))
    }
}

fn lock_status(state: &TrayState) -> MutexGuard<'_, TrayStatus> {
    state.0.lock().unwrap_or_else(|e| e.into_inner())
}

/// Build the tray, seed its menu, and subscribe to engine events so it stays
/// live. Called once from the Tauri `setup` hook.
pub fn init(app: &AppHandle) -> tauri::Result<()> {
    app.manage(TrayState::default());

    let menu = build_menu(
        app,
        &TrayStatus::default(),
        &known_hosts::recent_hosts(MAX_RECENTS),
    )?;
    TrayIconBuilder::with_id(TRAY_ID)
        .icon(app.default_window_icon().unwrap().clone())
        .tooltip("Tether")
        .menu(&menu)
        .on_menu_event(|app, event| handle_menu_event(app, event.id().as_ref()))
        .build(app)?;

    // The tray rebuilds itself off the same lifecycle events the webview reads.
    let status_app = app.clone();
    app.listen("engine-status", move |event| {
        if let Ok(ev) = serde_json::from_str::<StatusEvent>(event.payload()) {
            if apply_status(&status_app, &ev) {
                rebuild(&status_app);
            }
        }
    });
    let exit_app = app.clone();
    app.listen("engine-exited", move |event| {
        if let Ok(ev) = serde_json::from_str::<ExitEvent>(event.payload()) {
            apply_exit(&exit_app, &ev);
            rebuild(&exit_app);
        }
    });
    // A rename/forget in the address book changes the quick-connect submenu and
    // the label `status_line` resolves; refresh the menu (no state change).
    let hosts_app = app.clone();
    app.listen(known_hosts::HOSTS_CHANGED, move |_| rebuild(&hosts_app));

    Ok(())
}

/// The `engine-status` payload fields the tray cares about. Unknown fields
/// (profile, fingerprint, …) are ignored; absent optionals default to None.
#[derive(Deserialize)]
struct StatusEvent {
    role: String,
    event: String,
    #[serde(default)]
    host: Option<String>,
    #[serde(default)]
    peer: Option<String>,
}

#[derive(Deserialize)]
struct ExitEvent {
    role: String,
}

/// Fold a status event into the tray state. Returns whether anything the tray
/// shows actually changed, so the caller can skip a no-op rebuild.
fn apply_status(app: &AppHandle, ev: &StatusEvent) -> bool {
    let state = app.state::<TrayState>();
    let mut s = lock_status(state.inner());
    let before = s.clone();
    match (ev.role.as_str(), ev.event.as_str()) {
        ("host", "listening") => s.sharing = true,
        ("host", "peer_connected") => s.host_peer = ev.peer.clone(),
        ("host", "peer_disconnected") => s.host_peer = None,
        ("host", "error") => {
            s.sharing = false;
            s.host_peer = None;
        }
        ("client", "connected") => s.client_target = ev.host.clone(),
        ("client", "disconnected") | ("client", "error") => s.client_target = None,
        _ => {}
    }
    *s != before
}

/// Fold an engine exit into the tray state (a crash or a clean stop).
fn apply_exit(app: &AppHandle, ev: &ExitEvent) {
    let state = app.state::<TrayState>();
    let mut s = lock_status(state.inner());
    match ev.role.as_str() {
        ROLE_HOST => {
            s.sharing = false;
            s.host_peer = None;
        }
        ROLE_CLIENT => s.client_target = None,
        _ => {}
    }
}

/// Rebuild the tray menu and tooltip from current state. Best-effort: a failure
/// to rebuild leaves the prior menu in place rather than tearing the tray down.
fn rebuild(app: &AppHandle) {
    let Some(tray) = app.tray_by_id(TRAY_ID) else {
        return;
    };
    let state = app.state::<TrayState>();
    let status = lock_status(state.inner()).clone();
    let recents = known_hosts::recent_hosts(MAX_RECENTS);
    match build_menu(app, &status, &recents) {
        // Only update the tooltip once the menu actually swapped, so the two
        // can't disagree if a rebuild fails.
        Ok(menu) => match tray.set_menu(Some(menu)) {
            Ok(()) => {
                let _ = tray.set_tooltip(Some(tooltip(&status, &recents)));
            }
            Err(e) => tracing::warn!(error = %e, "tray set_menu failed"),
        },
        Err(e) => tracing::warn!(error = %e, "tray menu rebuild failed"),
    }
}

/// One-line current state, used for both the disabled header item and the
/// tooltip. A live client connection takes precedence over the sharing line.
fn status_line(status: &TrayStatus, recents: &[SavedHost]) -> String {
    if let Some(addr) = &status.client_target {
        return format!("Connected to {}", label_for(addr, recents));
    }
    if status.sharing {
        match &status.host_peer {
            Some(peer) => format!("Sharing · {peer} connected"),
            None => "Sharing · waiting for a connection".to_string(),
        }
    } else {
        "Not sharing".to_string()
    }
}

fn tooltip(status: &TrayStatus, recents: &[SavedHost]) -> String {
    format!("Tether — {}", status_line(status, recents))
}

/// Resolve a saved host's display label from the recents, falling back to the
/// raw address (e.g. if it aged out of the capped list).
fn label_for(addr: &str, recents: &[SavedHost]) -> String {
    recents
        .iter()
        .find(|h| h.addr == addr)
        .map(|h| h.label.clone())
        .unwrap_or_else(|| addr.to_string())
}

/// Assemble the full tray menu for the given state.
fn build_menu(
    app: &AppHandle,
    status: &TrayStatus,
    recents: &[SavedHost],
) -> tauri::Result<Menu<tauri::Wry>> {
    let header = MenuItemBuilder::with_id("status", status_line(status, recents))
        .enabled(false)
        .build(app)?;

    // "Connect to ▸" — recents, most-recently-used first. Each entry connects
    // immediately (pinned-cert resume); errors force-show the window.
    let mut connect = SubmenuBuilder::with_id(app, "connect_menu", "Connect to");
    if recents.is_empty() {
        let none = MenuItemBuilder::with_id("connect_none", "No saved computers")
            .enabled(false)
            .build(app)?;
        connect = connect.item(&none);
    } else {
        for host in recents {
            connect = connect.text(format!("{CONNECT_PREFIX}{}", host.addr), &host.label);
        }
    }
    let connect = connect.build()?;

    let disconnect = MenuItemBuilder::with_id(ID_DISCONNECT, "Disconnect").build(app)?;
    let share = CheckMenuItemBuilder::with_id(ID_SHARE, "Share this computer")
        .checked(status.sharing)
        .build(app)?;
    let add_device = MenuItemBuilder::with_id(ID_ADD_DEVICE, "Pair a device…").build(app)?;
    let show = MenuItemBuilder::with_id(ID_SHOW, "Show Tether").build(app)?;
    let quit = MenuItemBuilder::with_id(ID_QUIT, "Quit").build(app)?;

    let mut builder = MenuBuilder::new(app)
        .item(&header)
        .separator()
        .item(&connect);
    if status.client_target.is_some() {
        builder = builder.item(&disconnect);
    }
    builder
        .separator()
        .item(&share)
        .item(&add_device)
        .separator()
        .item(&show)
        .item(&quit)
        .build()
}

/// Route a tray menu click. Connect/share/disconnect spawn onto the async
/// runtime; show/quit/add-device act synchronously.
fn handle_menu_event(app: &AppHandle, id: &str) {
    match id {
        ID_SHOW => crate::show_main_window(app),
        ID_QUIT => app.exit(0),
        ID_ADD_DEVICE => {
            // Force-show the window and ask the webview to open the Sharing
            // sheet, where pairing lives.
            crate::show_main_window(app);
            let _ = app.emit("open-sharing", ());
        }
        ID_SHARE => toggle_sharing(app),
        ID_DISCONNECT => {
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                crate::stop_role(&app, app.state::<Supervisor>().inner(), ROLE_CLIENT).await;
            });
        }
        other if other.starts_with(CONNECT_PREFIX) => {
            connect_to(app, other[CONNECT_PREFIX.len()..].to_string());
        }
        _ => {}
    }
}

/// Flip the sharing posture. Reads the current state, then starts or stops the
/// host engine; the resulting `listening`/exit event rebuilds the menu (and the
/// supervisor persists the posture on `listening` — see `crate::prefs`).
fn toggle_sharing(app: &AppHandle) {
    let state = app.state::<TrayState>();
    let sharing = lock_status(state.inner()).sharing;
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let supervisor = app.state::<Supervisor>();
        if sharing {
            crate::stop_role(&app, supervisor.inner(), ROLE_HOST).await;
        } else if let Err(e) = supervisor
            .spawn(
                &app,
                ROLE_HOST,
                &["--ipc".to_string(), crate::HOST_BIND_ADDR.to_string()],
            )
            .await
        {
            tracing::warn!(error = %e, "tray: start sharing failed");
        }
    });
}

/// Connect to a saved host from the tray (pinned-cert resume, no PIN).
///
/// Headless when idle — the engine's video window is what the user wants, not
/// our chrome. But if a client session is already live, don't silently tear it
/// down: force-show the window and hand the request to the webview, which runs
/// the "Switch computer?" confirm. On a spawn failure, force-show the window so
/// the user isn't left wondering; engine-emitted errors after spawn surface
/// through the webview's handler.
fn connect_to(app: &AppHandle, addr: String) {
    let state = app.state::<TrayState>();
    let in_session = lock_status(state.inner()).client_target.is_some();
    if in_session {
        crate::show_main_window(app);
        let _ = app.emit("request-connect", addr);
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let args = vec!["--ipc".to_string(), addr];
        if let Err(e) = app
            .state::<Supervisor>()
            .spawn(&app, ROLE_CLIENT, &args)
            .await
        {
            tracing::warn!(error = %e, "tray: connect failed");
            crate::show_main_window(&app);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(addr: &str, label: &str) -> SavedHost {
        SavedHost {
            addr: addr.to_string(),
            label: label.to_string(),
            paired_at_unix: 0,
            last_connected_unix: None,
        }
    }

    #[test]
    fn status_line_reflects_each_state() {
        let recents = vec![host("work.local:7654", "Work")];
        assert_eq!(status_line(&TrayStatus::default(), &recents), "Not sharing");

        let sharing = TrayStatus {
            sharing: true,
            ..Default::default()
        };
        assert_eq!(
            status_line(&sharing, &recents),
            "Sharing · waiting for a connection"
        );

        let with_peer = TrayStatus {
            sharing: true,
            host_peer: Some("Laptop".to_string()),
            ..Default::default()
        };
        assert_eq!(
            status_line(&with_peer, &recents),
            "Sharing · Laptop connected"
        );
    }

    #[test]
    fn client_connection_takes_precedence_and_resolves_label() {
        // A live client connection is shown even while sharing, and the address
        // is rendered as its saved label.
        let recents = vec![host("work.local:7654", "Work")];
        let status = TrayStatus {
            sharing: true,
            client_target: Some("work.local:7654".to_string()),
            ..Default::default()
        };
        assert_eq!(status_line(&status, &recents), "Connected to Work");
    }

    #[test]
    fn unknown_target_falls_back_to_address() {
        let status = TrayStatus {
            client_target: Some("10.0.0.5:7654".to_string()),
            ..Default::default()
        };
        assert_eq!(status_line(&status, &[]), "Connected to 10.0.0.5:7654");
    }

    #[test]
    fn tray_status_lock_recovers_from_poisoned_mutex() {
        let state = TrayState::default();
        let poison = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = state.0.lock().unwrap();
            panic!("poison tray status lock");
        }));
        assert!(poison.is_err());

        assert_eq!(*lock_status(&state), TrayStatus::default());
    }
}
