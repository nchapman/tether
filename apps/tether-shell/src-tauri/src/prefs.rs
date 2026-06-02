//! Shell-level user preferences that persist across launches.
//!
//! Currently this is just the *sharing posture*: whether "Allow remote
//! connections" was last left on. The shell re-applies it on launch — off on
//! first run, sticky after — so a machine the user wants reachable doesn't need
//! the window re-opened every boot. Both the Sharing sheet toggle and (later)
//! the tray toggle drive this through `start_host` / `stop_engine`, so the
//! persisted posture has a single writer path and can't drift from them.
//!
//! Stored next to the trust store in the shared config dir. Unlike
//! `known_hosts.json` this is non-secret UI state, so a plain atomic write
//! (temp file + rename) is enough — no owner-only permission ceremony. A
//! missing or unreadable file is treated as "sharing off", never a hard error:
//! a corrupt pref must not keep the shell from starting.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// Serializes the read-modify-write in [`set_sharing_enabled`]. Tauri commands
/// run on a thread pool, so a rapid toggle (or the sheet and the future tray
/// toggle at once) could otherwise interleave a load with another writer's
/// rename and drop an update.
static WRITE_LOCK: Mutex<()> = Mutex::new(());

/// Persisted shell preferences. New fields must use `#[serde(default)]` so an
/// older on-disk file keeps deserializing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellPrefs {
    /// Whether hosting should be active. When true, the shell auto-starts the
    /// host engine on launch with real screen capture.
    #[serde(default)]
    pub sharing_enabled: bool,
}

/// Location of the prefs file in the shared config dir.
fn prefs_path() -> Result<PathBuf, String> {
    Ok(tether_pairing::config_dir()
        .map_err(|e| e.to_string())?
        .join("shell-prefs.json"))
}

/// Read prefs from `path`. A missing file (first run) or any read/parse failure
/// yields the default (sharing off) so a bad file can't block startup.
fn load_from(path: &Path) -> ShellPrefs {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            tracing::warn!(error = %e, path = %path.display(), "ignoring unreadable shell prefs");
            ShellPrefs::default()
        }),
        Err(_) => ShellPrefs::default(),
    }
}

/// Write prefs to `path` atomically: serialize to a per-process temp file, then
/// rename over the target so a concurrent reader never sees a partial file.
fn save_to(path: &Path, prefs: &ShellPrefs) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create prefs dir {}: {e}", parent.display()))?;
    }
    let json = serde_json::to_vec_pretty(prefs).map_err(|e| format!("serialize prefs: {e}"))?;
    // Per-process *and* per-call unique temp name: the pid keeps two processes
    // from colliding; the counter keeps two in-process writers apart.
    static TMP_SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("json.{}.{seq}.tmp", std::process::id()));
    std::fs::write(&tmp, &json).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("rename into {}: {e}", path.display()))?;
    Ok(())
}

/// Load the shell prefs from their shared on-disk location.
pub fn load() -> ShellPrefs {
    match prefs_path() {
        Ok(path) => load_from(&path),
        Err(e) => {
            tracing::warn!(error = %e, "can't resolve prefs path; using defaults");
            ShellPrefs::default()
        }
    }
}

/// Persist the sharing posture. A no-op when the on-disk value already matches,
/// to avoid rewriting the file on every redundant start/stop. Best-effort: a
/// write failure is logged, never surfaced — failing to remember the posture
/// must not fail the start/stop the user actually asked for.
pub fn set_sharing_enabled(enabled: bool) {
    // Hold the lock across the whole load/compare/save so concurrent toggles
    // can't lose a write. The critical section is plain sync filesystem I/O —
    // no await inside — so a std mutex is fine even though callers are async.
    let _guard = WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut prefs = load();
    if prefs.sharing_enabled == enabled {
        return;
    }
    prefs.sharing_enabled = enabled;
    if let Ok(path) = prefs_path() {
        if let Err(e) = save_to(&path, &prefs) {
            tracing::warn!(error = %e, "failed to persist sharing preference");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    // A unique temp path per test, so the suite stays parallel-safe without
    // touching the real config dir.
    fn temp_path() -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "tether-prefs-test.{}.{seq}.json",
            std::process::id()
        ))
    }

    #[test]
    fn missing_file_is_sharing_off() {
        let path = temp_path();
        assert_eq!(load_from(&path), ShellPrefs::default());
        assert!(!load_from(&path).sharing_enabled);
    }

    #[test]
    fn round_trips_sharing_enabled() {
        let path = temp_path();
        save_to(&path, &ShellPrefs { sharing_enabled: true }).unwrap();
        assert!(load_from(&path).sharing_enabled);
        save_to(&path, &ShellPrefs { sharing_enabled: false }).unwrap();
        assert!(!load_from(&path).sharing_enabled);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn corrupt_file_falls_back_to_default() {
        let path = temp_path();
        std::fs::write(&path, b"not json at all").unwrap();
        assert_eq!(load_from(&path), ShellPrefs::default());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_object_uses_field_defaults() {
        // Forward/backward compat: a file written before a field existed still
        // deserializes, with the missing field defaulting.
        let prefs: ShellPrefs = serde_json::from_slice(b"{}").unwrap();
        assert_eq!(prefs, ShellPrefs::default());
    }
}
