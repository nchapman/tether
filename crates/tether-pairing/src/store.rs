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

use serde::de::DeserializeOwned;
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
        load_json(path)
    }

    /// Persist atomically with owner-only permissions. See [`save_json_private`].
    pub fn save(&self, path: &Path) -> io::Result<()> {
        save_json_private(path, self)
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

/// One known host the client has paired with. `fingerprint` is the
/// algorithm-tagged (`"sha256:<hex>"`) host cert fingerprint the client pins on
/// reconnect; `label` is a display name; `paired_at_unix` is when it was first
/// paired (caller-stamped, so the store is deterministic for tests).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostEntry {
    pub fingerprint: String,
    pub label: String,
    pub paired_at_unix: u64,
}

/// The client's pinned-host list, serialized as JSON, keyed by the host's
/// socket-address string. Unlike the host's fingerprint-keyed [`PairedStore`],
/// the client looks hosts up by the address it dials, so reconnecting to a
/// known address can pin the host cert without the user re-entering a PIN or
/// fingerprint. It is a client-side trust root (a tampered entry could redirect
/// a reconnect to an attacker's cert), so it gets the same atomic, owner-only
/// persistence as [`PairedStore`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnownHosts {
    hosts: BTreeMap<String, HostEntry>,
}

impl KnownHosts {
    /// Load from `path`; missing file is empty, corrupt file is an error.
    pub fn load(path: &Path) -> io::Result<Self> {
        load_json(path)
    }

    /// Persist atomically with owner-only permissions. See [`save_json_private`].
    pub fn save(&self, path: &Path) -> io::Result<()> {
        save_json_private(path, self)
    }

    /// The pinned fingerprint for `addr`, decoded from its tagged form, if the
    /// host is known (and its stored tag parses).
    pub fn fingerprint(&self, addr: &str) -> Option<Fingerprint> {
        self.hosts
            .get(addr)
            .and_then(|e| parse_tagged_fingerprint(&e.fingerprint))
    }

    /// Whether `addr` is a known host.
    pub fn contains(&self, addr: &str) -> bool {
        self.hosts.contains_key(addr)
    }

    /// Record (or replace) the host pinned at `addr`.
    pub fn insert(&mut self, addr: String, fp: &Fingerprint, label: String, paired_at_unix: u64) {
        self.hosts.insert(
            addr,
            HostEntry {
                fingerprint: tag_fingerprint(fp),
                label,
                paired_at_unix,
            },
        );
    }

    /// Forget the host at `addr`; returns whether it was present.
    pub fn remove(&mut self, addr: &str) -> bool {
        self.hosts.remove(addr).is_some()
    }

    /// Iterate `(addr, entry)` for listing in the UI.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &HostEntry)> {
        self.hosts.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn len(&self) -> usize {
        self.hosts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hosts.is_empty()
    }
}

/// Load a JSON file into `T`. A missing file yields `T::default()` (first run);
/// a present-but-corrupt file is an error — both stores fail closed rather than
/// silently discarding a damaged trust root.
fn load_json<T: Default + DeserializeOwned>(path: &Path) -> io::Result<T> {
    match std::fs::read(path) {
        Ok(bytes) => {
            serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(T::default()),
        Err(e) => Err(e),
    }
}

/// Persist `value` as pretty JSON to `path` atomically and owner-only: write a
/// sibling temp file (created `0o600` on Unix), then rename over the target
/// (rename replaces on both Unix and Windows). Single-writer assumption: each
/// file path is written by at most one writer process (host → its allowlist,
/// client → its known-hosts, in separate files), so the fixed sibling temp name
/// (`<file>.tmp`) is safe; switch to a unique temp name if two writers ever
/// share a path.
fn save_json_private<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json =
        serde_json::to_vec_pretty(value).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let tmp = path.with_extension("json.tmp");
    write_private(&tmp, &json)?;
    std::fs::rename(&tmp, path)
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

    #[test]
    fn known_hosts_insert_lookup_remove() {
        let mut hosts = KnownHosts::default();
        let fp = [6u8; 32];
        assert!(!hosts.contains("192.168.1.5:7654"));
        hosts.insert("192.168.1.5:7654".to_string(), &fp, "desktop".to_string(), 1_700_000_000);
        assert!(hosts.contains("192.168.1.5:7654"));
        // Lookup returns the decoded fingerprint to pin on reconnect.
        assert_eq!(hosts.fingerprint("192.168.1.5:7654"), Some(fp));
        assert_eq!(hosts.fingerprint("10.0.0.1:7654"), None);
        assert!(hosts.remove("192.168.1.5:7654"));
        assert!(!hosts.contains("192.168.1.5:7654"));
        assert!(!hosts.remove("192.168.1.5:7654")); // idempotent
    }

    #[test]
    fn known_hosts_save_then_load_round_trips() {
        let dir =
            std::env::temp_dir().join(format!("tether-known-hosts-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("known_hosts.json");

        let mut hosts = KnownHosts::default();
        let fp = [7u8; 32];
        hosts.insert("host.local:7654".to_string(), &fp, "work".to_string(), 1_700_000_010);
        hosts.save(&path).expect("save");

        let loaded = KnownHosts::load(&path).expect("load");
        assert_eq!(loaded.fingerprint("host.local:7654"), Some(fp));
        assert_eq!(loaded.len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn known_hosts_corrupt_file_is_an_error() {
        let path = std::env::temp_dir()
            .join(format!("tether-known-hosts-corrupt-{}.json", std::process::id()));
        std::fs::write(&path, b"not json at all").expect("write corrupt");
        assert!(KnownHosts::load(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
