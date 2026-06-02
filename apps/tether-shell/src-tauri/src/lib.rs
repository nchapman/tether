//! Tether shell — the control-plane UI and engine supervisor.
//!
//! A Tauri app that lives in the system tray and supervises the native
//! `tether-host` / `tether-client` engine processes (see [`supervisor`]).
//! The webview is chrome only — connection forms, status, settings. The
//! actual video session is the engine's own native window in its own
//! process; nothing renders video through the webview.

mod known_hosts;
mod prefs;
mod supervisor;
mod tray;

use supervisor::{ExitedPayload, Supervisor, ROLE_CLIENT, ROLE_HOST};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};
use tauri_plugin_updater::UpdaterExt;
use tether_ipc::ShellCommand;

/// Start hosting: spawn `tether-host --ipc`. `test_pattern` swaps real
/// capture for the synthetic gradient (useful for one-machine loopback).
///
/// The sharing posture is persisted as on only once the host actually reaches
/// `listening` (in the supervisor reader), not here at spawn time — so a host
/// that spawns but fails to bind never leaves a broken "sharing on" posture
/// that would auto-restart and fail every launch.
#[tauri::command]
async fn start_host(
    app: AppHandle,
    supervisor: State<'_, Supervisor>,
    test_pattern: bool,
) -> Result<(), String> {
    let mut args = vec!["--ipc".to_string()];
    if test_pattern {
        args.push("--test-pattern".to_string());
    }
    supervisor.spawn(&app, ROLE_HOST, &args).await
}

/// Connect as a client: spawn `tether-client --ipc [--pin P] [--label L] <addr>
/// [fingerprint]`. A `pin` selects first-contact pairing; absent a pin (and a
/// fingerprint), the client reconnects via its known-hosts pin for `addr`. If a
/// `pin` is given, any `fingerprint` is accepted on the command line but ignored
/// — the PIN selects first-contact mode regardless.
#[tauri::command]
async fn connect_client(
    app: AppHandle,
    supervisor: State<'_, Supervisor>,
    addr: String,
    pin: Option<String>,
    label: Option<String>,
    fingerprint: Option<String>,
) -> Result<(), String> {
    let mut args = vec!["--ipc".to_string()];
    if let Some(pin) = pin.filter(|p| !p.is_empty()) {
        args.push("--pin".to_string());
        args.push(pin);
    }
    if let Some(label) = label.filter(|l| !l.is_empty()) {
        args.push("--label".to_string());
        args.push(label);
    }
    args.push(addr);
    if let Some(fp) = fingerprint.filter(|f| !f.is_empty()) {
        args.push(fp);
    }
    supervisor.spawn(&app, ROLE_CLIENT, &args).await
}

/// Stop the engine in `role` ("host" or "client").
///
/// Stopping the *host* is an explicit "turn sharing off", so it clears the
/// persisted posture — the shell won't re-enable hosting next launch. A host
/// that *crashes* exits through the supervisor's EOF path instead (it emits
/// `engine-exited`, not this command), so a crash leaves the posture sticky-on
/// and auto-restore retries on the next launch.
#[tauri::command]
async fn stop_engine(
    app: AppHandle,
    supervisor: State<'_, Supervisor>,
    role: String,
) -> Result<(), String> {
    stop_role(&app, supervisor.inner(), &role).await;
    Ok(())
}

/// Explicitly stop the engine in `role` and tell every consumer it's gone.
///
/// The supervisor's explicit stop removes the engine handle before its stdout
/// hits EOF, so the reader emits no `engine-exited` for this path — we emit one
/// here so the webview *and* the tray (both purely event-driven) don't keep
/// showing stale "sharing"/"connected" state. Stopping the host also clears the
/// persisted sharing posture (an explicit "sharing off"); doing so even when no
/// host was running is intentional and keeps the call safe to make blind.
///
/// Shared by the `stop_engine` command and the tray's Share/Disconnect items so
/// every explicit stop goes through one place.
pub(crate) async fn stop_role(app: &AppHandle, supervisor: &Supervisor, role: &str) {
    supervisor.stop(role).await;
    if role == ROLE_HOST {
        prefs::set_sharing_enabled(false);
    }
    let _ = app.emit(
        "engine-exited",
        ExitedPayload {
            role: role.to_string(),
        },
    );
}

/// Open a host pairing window for a new device with display name `label`. The
/// host replies with an `engine-status` `pairing_pin` event carrying the PIN.
#[tauri::command]
async fn start_pairing(supervisor: State<'_, Supervisor>, label: String) -> Result<(), String> {
    supervisor
        .send_command(ROLE_HOST, &ShellCommand::StartPairing { label })
        .await
}

/// Revoke a paired device by its tagged fingerprint; the host drops any live
/// session from it and pushes a refreshed `peer_list`.
#[tauri::command]
async fn revoke_peer(supervisor: State<'_, Supervisor>, fingerprint: String) -> Result<(), String> {
    supervisor
        .send_command(ROLE_HOST, &ShellCommand::RevokePeer { fingerprint })
        .await
}

/// Ask the host for its paired-device list (replies with a `peer_list` event).
#[tauri::command]
async fn list_peers(supervisor: State<'_, Supervisor>) -> Result<(), String> {
    supervisor
        .send_command(ROLE_HOST, &ShellCommand::ListPeers)
        .await
}

/// Download, verify, install the latest release, then restart into it.
/// Invoked by the "update available" banner. Re-checks the endpoint (cheap)
/// to get a fresh `Update` handle, applies the signature-verified bundle, then
/// restarts — so on success this never returns.
#[tauri::command]
async fn install_update(app: AppHandle) -> Result<(), String> {
    let updater = app.updater().map_err(|e| e.to_string())?;
    let Some(update) = updater.check().await.map_err(|e| e.to_string())? else {
        return Err("no update is available".to_string());
    };
    update
        .download_and_install(|_, _| {}, || {})
        .await
        .map_err(|e| e.to_string())?;
    // Relaunch into the freshly installed version; does not return.
    app.restart()
}

/// Return the persisted shell preferences. The webview reads this on mount to
/// re-apply the sharing posture (auto-start hosting when it was last left on).
#[tauri::command]
fn get_prefs() -> prefs::ShellPrefs {
    prefs::load()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        // Remember the window's position/size between launches.
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .manage(Supervisor::default())
        .setup(|app| {
            // System tray: the shell keeps running here even when the window is
            // closed, so the host engine can stay up headless. The tray owns its
            // own dynamic menu + status (see `tray`).
            tray::init(app.handle())?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_host,
            connect_client,
            stop_engine,
            start_pairing,
            revoke_peer,
            list_peers,
            known_hosts::list_known_hosts,
            known_hosts::rename_known_host,
            known_hosts::forget_known_host,
            hide_window,
            show_window,
            get_prefs,
            check_for_updates,
            install_update
        ])
        .build(tauri::generate_context!())
        .expect("error while building tether-shell")
        .run(|app, event| {
            // On exit, force-kill any engines. They'd also stop on their
            // own (their stdin closes → EOF), but this makes it prompt.
            if let RunEvent::Exit = event {
                if let Some(sup) = app.try_state::<Supervisor>() {
                    sup.kill_all();
                }
            }
        });
}

/// Check the configured GitHub Releases endpoint for a newer signed bundle.
/// Returns the available version (so the webview can show the "Install &
/// restart" banner that calls [`install_update`]), or `None` if up to date.
///
/// Invoked by the webview on mount rather than pushed from `setup`: a startup
/// emit can fire before the React listener mounts, and Tauri doesn't buffer
/// events for late subscribers — the request/response shape has no such race.
/// Any failure (offline, no release yet, dev build) is returned to the caller,
/// which logs and swallows it; a failed check must never block the shell.
#[tauri::command]
async fn check_for_updates(app: AppHandle) -> Result<Option<String>, String> {
    let updater = app.updater().map_err(|e| e.to_string())?;
    match updater.check().await.map_err(|e| e.to_string())? {
        Some(update) => {
            tracing::info!(version = %update.version, "update available");
            Ok(Some(update.version))
        }
        None => {
            tracing::info!("shell is up to date");
            Ok(None)
        }
    }
}

/// Hide the main window. The webview calls this when a client session goes
/// live: the engine's native video window is what the user wants in front, so
/// the control-plane chrome gets out of the way (see the hide-on-Connected
/// handoff in `docs/UX.md`). The window is re-shown via [`show_window`] when
/// the session ends, or from the tray's "Show Tether".
#[tauri::command]
fn hide_window(app: AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.hide();
    }
}

/// Reveal and focus the main window from the webview (e.g. a session ended and
/// we want the address book back in front). Shares the tray "Show" path.
#[tauri::command]
fn show_window(app: AppHandle) {
    show_main_window(&app);
}

/// Reveal and focus the main window (from the tray "Show" item).
pub(crate) fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
    }
}
