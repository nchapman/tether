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
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tether_ipc::{EngineEvent, ShellCommand};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::Mutex as AsyncMutex;

/// Which engine a child process is. Used as the map key and echoed to the
/// frontend so one `engine-status` listener can route both panels.
pub const ROLE_HOST: &str = "host";
pub const ROLE_CLIENT: &str = "client";

/// A live engine child: its stdin (to send `Stop`) and the handle (to
/// reap / force-kill). Stdout was taken at spawn time by the reader task.
/// `generation` distinguishes successive engines in the same role so a
/// slow-exiting old reader can't clobber a freshly spawned replacement.
struct EngineHandle {
    generation: u64,
    /// Behind an `AsyncMutex` + `Arc` so [`Supervisor::send_command`] can write
    /// a line to a *running* engine without removing its handle, while `spawn`
    /// / `stop` still own the rest of the struct under the outer `std::Mutex`.
    stdin: Arc<AsyncMutex<ChildStdin>>,
    child: Child,
}

/// Shell state: at most one host and one client engine at a time, keyed by
/// role. Behind a `Mutex` because Tauri commands run concurrently.
#[derive(Default)]
pub struct Supervisor {
    engines: Mutex<HashMap<String, EngineHandle>>,
    /// Monotonic id stamped on each spawn; see [`EngineHandle::generation`].
    next_generation: AtomicU64,
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
    pub async fn spawn(&self, app: &AppHandle, role: &str, args: &[String]) -> Result<(), String> {
        // Tear down a prior engine in this role so we never leak one.
        self.stop(role).await;

        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
        let bin = engine_binary(role)?;
        tracing::info!(?bin, role, generation, ?args, "spawning engine");

        let mut command = Command::new(&bin);
        command
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Inherit stderr so engine logs land in the dev terminal.
            .stderr(Stdio::inherit());
        // The engines are console-subsystem binaries; spawning one from this
        // GUI shell would otherwise pop a console window on Windows. We still
        // pipe stdin/stdout for the IPC channel — CREATE_NO_WINDOW only
        // suppresses the console allocation, it doesn't touch the std handles.
        #[cfg(windows)]
        {
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = command
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

        // Register the handle *before* starting the reader. A child that exits
        // immediately (e.g. a bind error) can reach stdout EOF before we'd
        // otherwise get to the insert below; registering first guarantees the
        // reader's generation check sees this engine, so it can emit
        // `engine-exited` and reap it instead of silently leaving the UI think
        // the role is still running.
        self.engines.lock().unwrap().insert(
            role.to_string(),
            EngineHandle {
                generation,
                stdin: Arc::new(AsyncMutex::new(stdin)),
                child,
            },
        );

        // Reader task: forward each JSON-line event to the webview, then
        // reap the child and emit `engine-exited` on EOF.
        let app_for_reader = app.clone();
        let role_owned = role.to_string();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            // Drain stdout line-by-line; Ok(None) (EOF) or an Err ends the loop.
            while let Ok(Some(line)) = lines.next_line().await {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                tracing::debug!(role = role_owned, line, "engine stdout line");
                match serde_json::from_str::<EngineEvent>(line) {
                    Ok(event) => {
                        if let Err(e) = app_for_reader.emit(
                            "engine-status",
                            StatusPayload {
                                role: role_owned.clone(),
                                event,
                            },
                        ) {
                            tracing::error!(error = %e, role = role_owned, "emit engine-status failed");
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, line, role = role_owned, "unparseable engine line");
                    }
                }
            }
            tracing::info!(
                role = role_owned,
                generation,
                "engine stdout closed; engine exited"
            );
            // Only retire the role if *we* are still the current engine.
            // A replacement may have been spawned while our process was
            // still draining its stdout; clobbering its handle (or emitting
            // a spurious `engine-exited`) would orphan it in the UI.
            let removed = app_for_reader.try_state::<Supervisor>().and_then(|sup| {
                let mut engines = sup.engines.lock().unwrap();
                if engines.get(&role_owned).map(|h| h.generation) == Some(generation) {
                    engines.remove(&role_owned)
                } else {
                    None
                }
            });
            if let Some(mut handle) = removed {
                // Reap the exited child so it doesn't linger as a zombie until
                // the shell exits, then tell the UI the role is gone. An
                // explicit `stop()` removes the handle itself, so this branch
                // won't fire for that path — no double `wait()`.
                let _ = handle.child.wait().await;
                let _ = app_for_reader.emit(
                    "engine-exited",
                    ExitedPayload {
                        role: role_owned.clone(),
                    },
                );
            }
        });

        Ok(())
    }

    /// Write one [`ShellCommand`] to a running engine's stdin without tearing it
    /// down (the channel for `StartPairing` / `RevokePeer` / `ListPeers`).
    /// Errors if no engine is running in that role or the write fails.
    pub async fn send_command(&self, role: &str, cmd: &ShellCommand) -> Result<(), String> {
        // Clone out the stdin handle under the brief std lock, then write under
        // the async lock — never hold the std mutex across an await.
        let stdin = {
            let engines = self.engines.lock().unwrap();
            engines.get(role).map(|h| h.stdin.clone())
        };
        let Some(stdin) = stdin else {
            return Err(format!("no {role} engine is running"));
        };
        let line = serde_json::to_string(cmd).map_err(|e| format!("serialize command: {e}"))?;
        let mut guard = stdin.lock().await;
        guard
            .write_all(line.as_bytes())
            .await
            .map_err(|e| format!("write to {role} stdin: {e}"))?;
        guard.write_all(b"\n").await.map_err(|e| e.to_string())?;
        guard.flush().await.map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Stop the engine in `role` if present: send `Stop` on its stdin,
    /// close stdin (a second EOF stop signal), then reap with a short
    /// force-kill backstop. No-op if nothing is running in that role.
    pub async fn stop(&self, role: &str) {
        let handle = self.engines.lock().unwrap().remove(role);
        let Some(EngineHandle {
            stdin, mut child, ..
        }) = handle
        else {
            return;
        };

        if let Ok(line) = serde_json::to_string(&ShellCommand::Stop) {
            let mut guard = stdin.lock().await;
            let _ = guard.write_all(line.as_bytes()).await;
            let _ = guard.write_all(b"\n").await;
            let _ = guard.flush().await;
        }
        // Dropping our handle closes ChildStdin (EOF) as a backup stop signal.
        // The explicit `Stop` line above is the *reliable* signal; if a
        // concurrent `send_command` still holds an Arc clone, EOF is merely
        // delayed until that write finishes — harmless, the engine already
        // received Stop.
        drop(stdin);

        tokio::spawn(async move {
            // Give the engine a moment to exit gracefully; if it overruns, kill
            // it and then `wait()` so the process is reaped rather than left a
            // zombie until the shell exits. (A clean exit within the grace
            // window is already reaped by the `wait()` that the timeout drove.)
            if tokio::time::timeout(Duration::from_secs(3), child.wait())
                .await
                .is_err()
            {
                let _ = child.start_kill();
                let _ = child.wait().await;
            }
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

/// The binary file name for an engine role, including the platform's
/// executable suffix (`.exe` on Windows, empty elsewhere).
fn engine_file_name(role: &str) -> Result<String, String> {
    let stem = match role {
        ROLE_HOST => "tether-host",
        ROLE_CLIENT => "tether-client",
        other => return Err(format!("unknown engine role: {other}")),
    };
    Ok(format!("{stem}{}", std::env::consts::EXE_SUFFIX))
}

/// Directories to search for an engine binary, in priority order:
/// 1. `TETHER_ENGINE_DIR` — explicit override (dev / `make shell`).
/// 2. The directory of the running shell binary — Tauri installs the
///    `externalBin` sidecars next to the app binary in a packaged build.
/// 3. `../../../target/debug` relative to `tauri dev`'s working directory —
///    the dev fallback, where sidecars aren't copied next to the dev binary.
fn engine_search_dirs(override_dir: Option<PathBuf>, exe_dir: Option<PathBuf>) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(dir) = override_dir {
        dirs.push(dir);
    }
    if let Some(dir) = exe_dir {
        dirs.push(dir);
    }
    dirs.push(PathBuf::from("../../../target/debug"));
    dirs
}

/// Pure resolver: the first directory whose `file_name` exists wins. On a
/// miss, returns the full candidate list so the error can show where we
/// looked. Factored out of [`engine_binary`] so the precedence is unit-tested
/// without touching the environment or filesystem.
fn resolve_engine_path(
    dirs: &[PathBuf],
    file_name: &str,
    exists: impl Fn(&Path) -> bool,
) -> Result<PathBuf, Vec<PathBuf>> {
    let candidates: Vec<PathBuf> = dirs.iter().map(|dir| dir.join(file_name)).collect();
    match candidates.iter().find(|path| exists(path)) {
        Some(path) => Ok(path.clone()),
        None => Err(candidates),
    }
}

/// Resolve the platform binary path for an engine role against the live
/// environment, current-exe location, and filesystem.
fn engine_binary(role: &str) -> Result<PathBuf, String> {
    let file_name = engine_file_name(role)?;
    let override_dir = std::env::var_os("TETHER_ENGINE_DIR").map(PathBuf::from);
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    let dirs = engine_search_dirs(override_dir, exe_dir);

    resolve_engine_path(&dirs, &file_name, |path| path.exists()).map_err(|candidates| {
        let looked: Vec<String> = candidates.iter().map(|p| p.display().to_string()).collect();
        format!(
            "{file_name} not found — looked in [{}]. Build it with `cargo build -p {}` \
             or set TETHER_ENGINE_DIR",
            looked.join(", "),
            file_name.trim_end_matches(std::env::consts::EXE_SUFFIX),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_role_is_rejected() {
        assert!(engine_file_name("bogus").is_err());
        // Includes the platform suffix (`.exe` on Windows, empty elsewhere) so
        // this passes on the Windows CI runner too.
        assert_eq!(
            engine_file_name(ROLE_HOST).unwrap(),
            format!("tether-host{}", std::env::consts::EXE_SUFFIX)
        );
    }

    #[test]
    fn search_dirs_are_ordered_override_then_exe_then_fallback() {
        let override_dir = PathBuf::from("/override");
        let exe_dir = PathBuf::from("/app");
        let dirs = engine_search_dirs(Some(override_dir.clone()), Some(exe_dir.clone()));
        assert_eq!(
            dirs,
            vec![
                override_dir,
                exe_dir,
                PathBuf::from("../../../target/debug")
            ]
        );
    }

    #[test]
    fn search_dirs_skip_absent_sources() {
        // With no override and no resolvable exe dir, only the dev fallback
        // remains — never a stray empty path.
        assert_eq!(
            engine_search_dirs(None, None),
            vec![PathBuf::from("../../../target/debug")]
        );
    }

    #[test]
    fn resolution_prefers_first_existing_candidate() {
        let dirs = engine_search_dirs(
            Some(PathBuf::from("/override")),
            Some(PathBuf::from("/app")),
        );

        // Override present → override wins even though the others "exist" too.
        let got = resolve_engine_path(&dirs, "tether-host", |_| true).unwrap();
        assert_eq!(got, PathBuf::from("/override/tether-host"));

        // Override missing → fall through to the exe (sidecar) directory.
        let got = resolve_engine_path(&dirs, "tether-host", |p| p.starts_with("/app")).unwrap();
        assert_eq!(got, PathBuf::from("/app/tether-host"));
    }

    #[test]
    fn resolution_miss_reports_every_candidate() {
        let dirs = engine_search_dirs(
            Some(PathBuf::from("/override")),
            Some(PathBuf::from("/app")),
        );
        let candidates = resolve_engine_path(&dirs, "tether-host", |_| false).unwrap_err();
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0], PathBuf::from("/override/tether-host"));
    }
}
