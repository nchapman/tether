//! Persistent allowlist of paired peers.
//!
//! Both ends keep one of these on disk: the host's `paired_clients.json`
//! authorizes which client fingerprints may open a session (and inject
//! input), and the client's `known_hosts.json` records which host cert it has
//! pinned. Keyed by an **algorithm-tagged** fingerprint (`"sha256:<hex>"`) so
//! the hash function can be migrated later without an unparseable bare-hex
//! store. Labels are non-authoritative display strings — never trust them or
//! key on them; the fingerprint is the identity.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Fingerprint;

/// One paired peer. `label` is a human-chosen display name (not a trust
/// input); `paired_at_unix` is the wall-clock pairing time in Unix seconds,
/// stamped by the caller so the store stays deterministic for tests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerEntry {
    pub label: String,
    pub paired_at_unix: u64,
}

/// Allowlist of paired peers, serialized as JSON. Keyed by tagged fingerprint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PairedStore {
    peers: BTreeMap<String, PeerEntry>,
}

impl PairedStore {
    /// Load from `path`. A missing file is an empty store (first run), not an
    /// error. A present-but-corrupt file *is* an error — refusing to silently
    /// treat a damaged allowlist as "trust nobody" surfaces the problem.
    pub fn load(path: &Path) -> io::Result<Self> {
        match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e),
        }
    }

    /// Persist atomically: write a sibling temp file, then rename over the
    /// target (rename replaces on both Unix and Windows). On Unix the file is
    /// created `0o600` — it's a trust root, readable only by its owner.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        // Single-writer assumption: the store is written only by the one
        // engine process that owns this role, so a fixed sibling temp name is
        // safe. If concurrent writers ever appear, switch to a unique temp
        // name (e.g. tempfile::NamedTempFile) to avoid clobbering.
        let tmp = path.with_extension("json.tmp");
        write_private(&tmp, &json)?;
        std::fs::rename(&tmp, path)
    }

    /// Whether `fp` is paired.
    pub fn contains(&self, fp: &Fingerprint) -> bool {
        self.peers.contains_key(&tag_fingerprint(fp))
    }

    /// Add or replace the entry for `fp`.
    pub fn insert(&mut self, fp: &Fingerprint, label: String, paired_at_unix: u64) {
        self.peers.insert(
            tag_fingerprint(fp),
            PeerEntry {
                label,
                paired_at_unix,
            },
        );
    }

    /// Remove the entry for `fp`; returns whether it was present.
    pub fn remove(&mut self, fp: &Fingerprint) -> bool {
        self.peers.remove(&tag_fingerprint(fp)).is_some()
    }

    /// Remove by tagged-fingerprint string (the form the UI round-trips).
    /// Returns whether it was present.
    pub fn remove_tagged(&mut self, tagged: &str) -> bool {
        self.peers.remove(tagged).is_some()
    }

    /// Number of paired peers.
    pub fn len(&self) -> usize {
        self.peers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    /// Iterate `(tagged_fingerprint, entry)` for listing in the UI.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &PeerEntry)> {
        self.peers.iter().map(|(k, v)| (k.as_str(), v))
    }
}

/// Render a fingerprint as `"sha256:<lowercase-hex>"`.
pub fn tag_fingerprint(fp: &Fingerprint) -> String {
    format!("sha256:{}", hex::encode(fp))
}

/// Parse a `"sha256:<hex>"` string back to a fingerprint. Returns `None` for
/// any other algorithm tag, malformed hex, or wrong length.
pub fn parse_tagged_fingerprint(tagged: &str) -> Option<Fingerprint> {
    let hex_part = tagged.strip_prefix("sha256:")?;
    let bytes = hex::decode(hex_part).ok()?;
    bytes.try_into().ok()
}

#[cfg(unix)]
fn write_private(path: &Path, bytes: &[u8]) -> io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    f.write_all(bytes)
}

// SECURITY: on Windows we rely on the inherited ACL of the per-user profile
// directory (`%USERPROFILE%\.tether`), which by default grants access only to
// the owner, SYSTEM, and Administrators — not other standard users. We do not
// yet set an explicit owner-only DACL, so on a machine whose profile ACL has
// been loosened this trust-root file could be read or written by another local
// user (who could then self-authorize for input injection). Acceptable for the
// single-user alpha; harden with SetNamedSecurityInfoW before multi-user use.
#[cfg(not(unix))]
fn write_private(path: &Path, bytes: &[u8]) -> io::Result<()> {
    std::fs::write(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_format_is_sha256_prefixed_hex() {
        let fp = [0u8; 32];
        let tagged = tag_fingerprint(&fp);
        assert_eq!(tagged, format!("sha256:{}", "00".repeat(32)));
        assert!(tagged.starts_with("sha256:"));
        assert_eq!(tagged.len(), "sha256:".len() + 64);
    }

    #[test]
    fn tagged_fingerprint_round_trips() {
        let fp: Fingerprint = std::array::from_fn(|i| i as u8);
        let parsed = parse_tagged_fingerprint(&tag_fingerprint(&fp)).expect("parses");
        assert_eq!(parsed, fp);
    }

    #[test]
    fn parse_rejects_bad_input() {
        assert!(parse_tagged_fingerprint("md5:abcd").is_none()); // wrong algorithm
        assert!(parse_tagged_fingerprint("sha256:zz").is_none()); // bad hex
        assert!(parse_tagged_fingerprint(&format!("sha256:{}", "00".repeat(16))).is_none()); // short
        assert!(parse_tagged_fingerprint("deadbeef").is_none()); // no tag
    }

    #[test]
    fn insert_contains_remove() {
        let mut store = PairedStore::default();
        let fp = [3u8; 32];
        assert!(!store.contains(&fp));
        store.insert(&fp, "laptop".to_string(), 1_700_000_000);
        assert!(store.contains(&fp));
        assert_eq!(store.len(), 1);
        assert!(store.remove(&fp));
        assert!(!store.contains(&fp));
        assert!(!store.remove(&fp)); // idempotent
    }

    #[test]
    fn save_then_load_preserves_entries() {
        let dir = std::env::temp_dir().join(format!("tether-pairing-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("paired_clients.json");

        let mut store = PairedStore::default();
        let fp1 = [1u8; 32];
        let fp2 = [2u8; 32];
        store.insert(&fp1, "desktop".to_string(), 1_700_000_001);
        store.insert(&fp2, "phone".to_string(), 1_700_000_002);
        store.save(&path).expect("save");

        let loaded = PairedStore::load(&path).expect("load");
        assert!(loaded.contains(&fp1));
        assert!(loaded.contains(&fp2));
        assert_eq!(loaded.len(), 2);
        let labels: Vec<_> = loaded.iter().map(|(_, e)| e.label.clone()).collect();
        assert!(labels.contains(&"desktop".to_string()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_file_is_an_error_not_empty() {
        // A damaged allowlist must surface as an error — silently treating it
        // as an empty store would be a fail-open: an attacker who zeroes the
        // file should not turn "trust these peers" into "trust nobody and
        // re-pair anyone." (Fail closed: the host refuses to start instead.)
        let path =
            std::env::temp_dir().join(format!("tether-pairing-corrupt-{}.json", std::process::id()));
        std::fs::write(&path, b"this is not json").expect("write corrupt file");
        let result = PairedStore::load(&path);
        assert!(result.is_err(), "corrupt file must be Err, not an empty store");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_missing_file_is_empty() {
        let path = std::env::temp_dir().join("tether-pairing-does-not-exist-xyz.json");
        let _ = std::fs::remove_file(&path);
        let store = PairedStore::load(&path).expect("missing file is empty store");
        assert!(store.is_empty());
    }

    #[test]
    fn remove_tagged_matches_insert() {
        let mut store = PairedStore::default();
        let fp = [5u8; 32];
        store.insert(&fp, "x".to_string(), 0);
        let tagged = tag_fingerprint(&fp);
        assert!(store.remove_tagged(&tagged));
        assert!(!store.contains(&fp));
    }
}
