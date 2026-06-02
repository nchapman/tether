//! The Connect window's address book, backed by the client's
//! `known_hosts.json`.
//!
//! The native `tether-client` engine only exists for the lifetime of a
//! session, so the shell can't ask a running engine for the saved-host list.
//! Instead it reads and writes the same trust store the engine uses, through
//! [`tether_pairing::KnownHosts`] — so the atomic, owner-only persistence and
//! the on-disk format stay in one place. These Tauri commands back the
//! address book: list rows, rename a saved computer, and forget one.
//!
//! Connecting to a saved row is *not* here — that reuses the existing
//! `connect_client` command with just an address (the pinned-cert resume
//! path). This module is only the list-management half.

use serde::Serialize;
use tether_pairing::KnownHosts;

/// One saved computer, as the Connect window renders it. The cert fingerprint
/// is deliberately omitted: it's a trust-internal detail the redesigned UI no
/// longer surfaces (reconnect pins it invisibly). `last_connected_unix` is
/// `None` for hosts paired before that field existed; the UI falls back to
/// `paired_at_unix` for its recency line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SavedHost {
    /// Socket address the host was reached at — the key the row connects to
    /// and the value behind "Copy address".
    pub addr: String,
    /// Human-chosen display name.
    pub label: String,
    pub paired_at_unix: u64,
    pub last_connected_unix: Option<u64>,
}

/// Project a store into the rows the UI shows, most-recently-used first.
/// Ordering key is `last_connected_unix` falling back to `paired_at_unix`
/// (descending), with the label as a stable tiebreaker so equal timestamps
/// don't reorder between calls. Pure so it can be unit-tested without the
/// filesystem.
fn sorted_rows(store: &KnownHosts) -> Vec<SavedHost> {
    let mut rows: Vec<SavedHost> = store
        .iter()
        .map(|(addr, entry)| SavedHost {
            addr: addr.to_string(),
            label: entry.label.clone(),
            paired_at_unix: entry.paired_at_unix,
            last_connected_unix: entry.last_connected_unix,
        })
        .collect();
    rows.sort_by(|a, b| {
        let a_key = a.last_connected_unix.unwrap_or(a.paired_at_unix);
        let b_key = b.last_connected_unix.unwrap_or(b.paired_at_unix);
        b_key.cmp(&a_key).then_with(|| a.label.cmp(&b.label))
    });
    rows
}

/// Load the known-hosts store from its shared on-disk location. A missing file
/// is an empty store (first run); a corrupt one is an error the UI surfaces.
fn load_store() -> Result<KnownHosts, String> {
    let path = tether_pairing::known_hosts_path().map_err(|e| e.to_string())?;
    KnownHosts::load(&path).map_err(|e| format!("couldn't read saved computers: {e}"))
}

/// Persist the store back to the shared location, atomically and owner-only.
fn save_store(store: &KnownHosts) -> Result<(), String> {
    let path = tether_pairing::known_hosts_path().map_err(|e| e.to_string())?;
    store
        .save(&path)
        .map_err(|e| format!("couldn't save changes: {e}"))
}

/// List the saved computers for the Connect window's address book.
#[tauri::command]
pub fn list_known_hosts() -> Result<Vec<SavedHost>, String> {
    Ok(sorted_rows(&load_store()?))
}

/// Longest accepted display label. Labels are non-authoritative chrome; this
/// cap just stops a malformed/oversized IPC call from bloating the trust store
/// (which is re-read in full on every load).
const MAX_LABEL_BYTES: usize = 255;

/// Rename a saved computer (display label only — never the pinned identity).
/// A blank label is rejected so a row can't lose its name; an unknown address
/// (e.g. forgotten in another window) is an error the UI can show.
#[tauri::command]
pub fn rename_known_host(addr: String, label: String) -> Result<(), String> {
    let label = label.trim();
    if label.is_empty() {
        return Err("a name can't be empty".to_string());
    }
    if label.len() > MAX_LABEL_BYTES {
        return Err(format!(
            "a name must be {MAX_LABEL_BYTES} characters or fewer"
        ));
    }
    let mut store = load_store()?;
    if !store.rename(&addr, label.to_string()) {
        return Err(format!("no saved computer at {addr}"));
    }
    save_store(&store)
}

/// Forget a saved computer: drop its pinned cert so it disappears from the
/// address book. Reconnecting afterward needs a fresh PIN. Forgetting an
/// already-absent host is a no-op success (idempotent for double-clicks).
#[tauri::command]
pub fn forget_known_host(addr: String) -> Result<(), String> {
    let mut store = load_store()?;
    store.remove(&addr);
    save_store(&store)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with(entries: &[(&str, &str, u64, Option<u64>)]) -> KnownHosts {
        let mut store = KnownHosts::default();
        let fp = [0u8; 32];
        for (addr, label, paired_at, last) in entries {
            store.insert((*addr).to_string(), &fp, (*label).to_string(), *paired_at);
            if let Some(when) = last {
                store.set_last_connected(addr, *when);
            }
        }
        store
    }

    #[test]
    fn rows_are_ordered_most_recently_used_first() {
        let store = store_with(&[
            ("a:7654", "alpha", 100, Some(500)), // last-connected 500
            ("b:7654", "bravo", 900, None),      // never reconnected → paired_at 900
            ("c:7654", "charlie", 100, Some(300)),
        ]);
        let rows = sorted_rows(&store);
        let order: Vec<&str> = rows.iter().map(|r| r.label.as_str()).collect();
        // bravo (900) > alpha (500) > charlie (300).
        assert_eq!(order, ["bravo", "alpha", "charlie"]);
    }

    #[test]
    fn equal_recency_breaks_ties_by_label() {
        let store = store_with(&[
            ("z:7654", "zeta", 100, Some(200)),
            ("a:7654", "alpha", 100, Some(200)),
        ]);
        let rows = sorted_rows(&store);
        let order: Vec<&str> = rows.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(order, ["alpha", "zeta"]);
    }

    #[test]
    fn rename_rejects_blank_and_oversized_labels() {
        // Both checks short-circuit before any filesystem access.
        assert!(rename_known_host("a:7654".to_string(), "   ".to_string()).is_err());
        let huge = "x".repeat(MAX_LABEL_BYTES + 1);
        assert!(rename_known_host("a:7654".to_string(), huge).is_err());
    }

    #[test]
    fn row_carries_addr_and_recency_fields() {
        let store = store_with(&[("host.local:7654", "work", 100, Some(250))]);
        let rows = sorted_rows(&store);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].addr, "host.local:7654");
        assert_eq!(rows[0].label, "work");
        assert_eq!(rows[0].paired_at_unix, 100);
        assert_eq!(rows[0].last_connected_unix, Some(250));
    }
}
