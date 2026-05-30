//! Engine-process supervision.
//!
//! The shell spawns the native `tether-host` / `tether-client` engines as
//! child processes and speaks the [`tether_ipc`] JSON-lines protocol to
//! them: it reads lifecycle [`EngineEvent`]s from each child's stdout and
//! re-emits them to the webview as `engine-status` events; it writes a
//! [`ShellCommand::Stop`] to a child's stdin to tear it down. Engine
//! stderr is inherited so logs surface in the dev terminal.
//!
//! The whole point of the multi-process design lives here: the video
//! window is the engine's own native winit/wgpu surface in a separate
//! process, so a decoder/render stall can't take down this UI, and a
//! crashed shell drops the children's stdin (→ EOF → graceful engine
//! stop, see the engine-side stdin watchers).

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tether_ipc::{EngineEvent, ShellCommand};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};

/// Which engine a child process is. Used as the map key and echoed to the
/// frontend so one `engine-status` listener can route both panels.
pub const ROLE_HOST: &str = "host";
pub const ROLE_CLIENT: &str = "client";

/// A live engine child: its stdin (to send `Stop`) and the handle (to
/// reap / force-kill). Stdout was taken at spawn time by the reader task.
struct EngineHandle {
    stdin: ChildStdin,
    child: Child,
}

/// Shell state: at most one host and one client engine at a time, keyed by
/// role. Behind a `Mutex` because Tauri commands run concurrently.
#[derive(Default)]
pub struct Supervisor {
    engines: Mutex<HashMap<String, EngineHandle>>,
}

/// `engine-status` payload: the engine's [`EngineEvent`] flattened
/// (`{"event":"connected",...}`) plus the `role` that produced it.
#[derive(Clone, Serialize)]
struct StatusPayload {
    role: String,
    #[serde(flatten)]
    event: EngineEvent,
}

/// `engine-exited` payload: the child's stdout reached EOF (it exited).
#[derive(Clone, Serialize)]
struct ExitedPayload {
    role: String,
}

impl Supervisor {
    /// Spawn an engine binary for `role` with `args`, wire up its stdout
    /// reader, and register it. Replaces any existing engine in that role
    /// (the old one is stopped first). Returns an error string the UI can
    /// surface if the binary is missing or won't launch.
    pub async fn spawn(
        &self,
        app: &AppHandle,
        role: &str,
        args: &[String],
    ) -> Result<(), String> {
        // Tear down a prior engine in this role so we never leak one.
        self.stop(role).await;

        let bin = engine_binary(role)?;
        tracing::info!(?bin, role, ?args, "spawning engine");

        let mut child = Command::new(&bin)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Inherit stderr so engine logs land in the dev terminal.
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("failed to launch {}: {e}", bin.display()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "engine stdout was not piped".to_string())?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "engine stdin was not piped".to_string())?;

        // Reader task: forward each JSON-line event to the webview, then
        // emit `engine-exited` on EOF.
        let app_for_reader = app.clone();
        let role_owned = role.to_string();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<EngineEvent>(line) {
                            Ok(event) => {
                                let _ = app_for_reader.emit(
                                    "engine-status",
                                    StatusPayload {
                                        role: role_owned.clone(),
                                        event,
                                    },
                                );
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, line, role = role_owned, "unparseable engine line");
                            }
                        }
                    }
                    Ok(None) | Err(_) => break,
                }
            }
            tracing::info!(role = role_owned, "engine stdout closed; engine exited");
            let _ = app_for_reader.emit(
                "engine-exited",
                ExitedPayload {
                    role: role_owned.clone(),
                },
            );
            // Drop our handle so the role reads as idle again.
            if let Some(sup) = app_for_reader.try_state::<Supervisor>() {
                sup.engines.lock().unwrap().remove(&role_owned);
            }
        });

        self.engines
            .lock()
            .unwrap()
            .insert(role.to_string(), EngineHandle { stdin, child });
        Ok(())
    }

    /// Stop the engine in `role` if present: send `Stop` on its stdin,
    /// close stdin (a second EOF stop signal), then reap with a short
    /// force-kill backstop. No-op if nothing is running in that role.
    pub async fn stop(&self, role: &str) {
        let handle = self.engines.lock().unwrap().remove(role);
        let Some(EngineHandle {
            mut stdin,
            mut child,
        }) = handle
        else {
            return;
        };

        if let Ok(line) = serde_json::to_string(&ShellCommand::Stop) {
            let _ = stdin.write_all(line.as_bytes()).await;
            let _ = stdin.write_all(b"\n").await;
            let _ = stdin.flush().await;
        }
        drop(stdin); // EOF: backup stop signal even if the write was lost.

        tokio::spawn(async move {
            // Give the engine a moment to exit gracefully, then make sure.
            let _ = tokio::time::timeout(Duration::from_secs(3), child.wait()).await;
            let _ = child.start_kill();
        });
    }

    /// Force-kill every engine synchronously. Called on shell exit as a
    /// backstop — the children would also see their stdin close and stop
    /// on their own, but this makes teardown prompt.
    pub fn kill_all(&self) {
        let mut engines = self.engines.lock().unwrap();
        for (role, mut handle) in engines.drain() {
            tracing::info!(role, "killing engine on shell exit");
            let _ = handle.child.start_kill();
        }
    }
}

/// Directory holding the engine binaries. `TETHER_ENGINE_DIR` overrides;
/// the default is the workspace `target/debug` relative to the
/// `tauri dev` working directory (`apps/tether-shell/src-tauri`).
/// Bundling will replace this with Tauri sidecar resolution.
fn engine_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("TETHER_ENGINE_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from("../../../target/debug")
}

/// Resolve the platform binary path for an engine role.
fn engine_binary(role: &str) -> Result<PathBuf, String> {
    let stem = match role {
        ROLE_HOST => "tether-host",
        ROLE_CLIENT => "tether-client",
        other => return Err(format!("unknown engine role: {other}")),
    };
    let path = engine_dir().join(format!("{stem}{}", std::env::consts::EXE_SUFFIX));
    if !path.exists() {
        return Err(format!(
            "engine binary not found at {} — build it with `cargo build -p {stem}` \
             or set TETHER_ENGINE_DIR",
            path.display()
        ));
    }
    Ok(path)
}
