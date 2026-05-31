//! Tether shell — the control-plane UI and engine supervisor.
//!
//! A Tauri app that lives in the system tray and supervises the native
//! `tether-host` / `tether-client` engine processes (see [`supervisor`]).
//! The webview is chrome only — connection forms, status, settings. The
//! actual video session is the engine's own native window in its own
//! process; nothing renders video through the webview.

mod supervisor;

use supervisor::{Supervisor, ROLE_CLIENT, ROLE_HOST};
use tauri::menu::{MenuBuilder, MenuItemBuilder};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};
use tauri_plugin_updater::UpdaterExt;
use tether_ipc::ShellCommand;

/// Start hosting: spawn `tether-host --ipc`. `test_pattern` swaps real
/// capture for the synthetic gradient (useful for one-machine loopback).
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
#[tauri::command]
async fn stop_engine(supervisor: State<'_, Supervisor>, role: String) -> Result<(), String> {
    supervisor.stop(&role).await;
    Ok(())
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
        .manage(Supervisor::default())
        .setup(|app| {
            // Best-effort startup update check; never blocks the shell.
            let update_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                if let Err(e) = check_for_updates(update_handle).await {
                    tracing::warn!(error = %e, "update check failed");
                }
            });

            // System tray: the shell keeps running here even when the
            // window is closed, so the host engine can stay up headless.
            let show = MenuItemBuilder::with_id("show", "Show Tether").build(app)?;
            let quit = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
            let menu = MenuBuilder::new(app).items(&[&show, &quit]).build()?;
            TrayIconBuilder::new()
                .icon(app.default_window_icon().unwrap().clone())
                .tooltip("Tether")
                .menu(&menu)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "show" => show_main_window(app),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_host,
            connect_client,
            stop_engine,
            start_pairing,
            revoke_peer,
            list_peers,
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

/// Best-effort startup update check. Queries the configured GitHub Releases
/// endpoint and, if a newer signed bundle exists, emits `update-available`
/// (carrying the new version) so the webview can show the banner whose
/// "Install & restart" button calls [`install_update`]. Any failure (offline,
/// no release yet, dev build) is returned to the caller, which logs and
/// swallows it; a failed check must never block the shell from starting.
async fn check_for_updates(app: AppHandle) -> Result<(), String> {
    let updater = app.updater().map_err(|e| e.to_string())?;
    match updater.check().await.map_err(|e| e.to_string())? {
        Some(update) => {
            tracing::info!(version = %update.version, "update available");
            let _ = app.emit("update-available", update.version.clone());
        }
        None => tracing::info!("shell is up to date"),
    }
    Ok(())
}

/// Reveal and focus the main window (from the tray "Show" item).
fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
    }
}
