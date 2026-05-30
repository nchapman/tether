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
use tauri::{AppHandle, Manager, RunEvent, State};

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

/// Connect as a client: spawn `tether-client --ipc <addr> <fingerprint>`.
#[tauri::command]
async fn connect_client(
    app: AppHandle,
    supervisor: State<'_, Supervisor>,
    addr: String,
    fingerprint: String,
) -> Result<(), String> {
    let args = vec!["--ipc".to_string(), addr, fingerprint];
    supervisor.spawn(&app, ROLE_CLIENT, &args).await
}

/// Stop the engine in `role` ("host" or "client").
#[tauri::command]
async fn stop_engine(supervisor: State<'_, Supervisor>, role: String) -> Result<(), String> {
    supervisor.stop(&role).await;
    Ok(())
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
        .manage(Supervisor::default())
        .setup(|app| {
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
            stop_engine
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

/// Reveal and focus the main window (from the tray "Show" item).
fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
    }
}
