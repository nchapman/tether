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
use std::sync::atomic::{AtomicU64, Ordering};

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
/// `last_connected_unix` is the most recent successful connect, stamped by the
/// client after every session so the UI can show recency ("connected 2h ago");
/// `None` on entries written before this field existed (serde default), so the
/// UI falls back to `paired_at_unix`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostEntry {
    pub fingerprint: String,
    pub label: String,
    pub paired_at_unix: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_connected_unix: Option<u64>,
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

    /// Record (or replace) the host pinned at `addr`. `last_connected_unix`
    /// starts unset; the caller stamps it via [`set_last_connected`] after the
    /// session is up.
    ///
    /// [`set_last_connected`]: KnownHosts::set_last_connected
    pub fn insert(&mut self, addr: String, fp: &Fingerprint, label: String, paired_at_unix: u64) {
        self.hosts.insert(
            addr,
            HostEntry {
                fingerprint: tag_fingerprint(fp),
                label,
                paired_at_unix,
                last_connected_unix: None,
            },
        );
    }

    /// Stamp the most-recent-connect time for `addr` if it's known; returns
    /// whether the host existed. The client calls this after every successful
    /// connect (first-contact or resume) so the UI can show recency.
    pub fn set_last_connected(&mut self, addr: &str, when_unix: u64) -> bool {
        match self.hosts.get_mut(addr) {
            Some(entry) => {
                entry.last_connected_unix = Some(when_unix);
                true
            }
            None => false,
        }
    }

    /// Rename the host at `addr` (display label only — not a trust input);
    /// returns whether it was present.
    pub fn rename(&mut self, addr: &str, label: String) -> bool {
        match self.hosts.get_mut(addr) {
            Some(entry) => {
                entry.label = label;
                true
            }
            None => false,
        }
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
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(T::default()),
        Err(e) => Err(e),
    }
}

/// Process-local counter making each in-flight temp file name distinct even
/// when two threads in the same process save concurrently.
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Persist `value` as pretty JSON to `path` atomically and owner-only: write a
/// sibling temp file (created `0o600` on Unix), then rename over the target
/// (rename replaces on both Unix and Windows).
///
/// The temp name is made unique per writer — `<file>.json.<pid>.<seq>.tmp` —
/// because `known_hosts.json` now has **two** writer processes: the
/// `tether-client` engine (stamping `last_connected_unix` on connect) and the
/// Tauri shell (rename/forget from the address book). A shared fixed temp name
/// risks one writer truncating/renaming the other's half-written temp into
/// place, corrupting the store. Each writer's own temp + atomic rename means
/// the last rename always installs a complete, valid file. (A logical
/// lost-update — e.g. a forget racing a connect-time stamp — can still drop one
/// change, but never corrupts the file or weakens a pin.)
fn save_json_private<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_vec_pretty(value)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("json.{}.{seq}.tmp", std::process::id()));
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

// SECURITY: the Windows analogue of the `0o600` above. We create the file with
// an explicit owner-only, inheritance-protected DACL rather than inheriting the
// profile directory's ACL — so even on a machine whose `%USERPROFILE%\.tether`
// ACL has been loosened, this trust-root file can't be read or written by
// another local user (who could otherwise self-authorize for input injection).
// Building the descriptor at CreateFile time (not tightening after a normal
// write) means the file never exists with a looser ACL, even for an instant.
// The DACL grants *only* the current user — no SYSTEM/Administrators ACE,
// mirroring how Unix `0o600` excludes root from the explicit mode bits (admins
// can still take ownership, so restricting them would be theater; the threat is
// another standard user). A side effect, as on Unix, is that a backup agent
// running as SYSTEM can't read the file without a privilege that bypasses DACLs.
//
// NOTE: byte-for-byte identical to `tether_transport::tls::write_key_file`'s
// Windows arm; the two live in separate crates (this one stays free of quinn,
// that one of spake2) so the shim is duplicated rather than shared. Keep in sync.
#[cfg(windows)]
fn write_private(path: &Path, bytes: &[u8]) -> io::Result<()> {
    use std::io::Write;
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::io::FromRawHandle;

    use windows::core::PCWSTR;
    use windows::Win32::Foundation::{LocalFree, HLOCAL};
    use windows::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
    };
    use windows::Win32::Security::{PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES};
    use windows::Win32::Storage::FileSystem::{
        CreateFileW, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, FILE_GENERIC_WRITE, FILE_SHARE_NONE,
    };

    // The DACL grants exactly the current user, so resolve their SID first.
    // `D:` DACL, `P` protected (don't inherit looser ACEs from the parent dir),
    // `(A;;FA;;;<sid>)` allow full access to just this user.
    let sid = current_user_sid_string()?;
    let sddl: Vec<u16> = format!("D:P(A;;FA;;;{sid})\0").encode_utf16().collect();
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();

    unsafe {
        let mut psd = PSECURITY_DESCRIPTOR(std::ptr::null_mut());
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            PCWSTR(sddl.as_ptr()),
            SDDL_REVISION_1,
            &mut psd,
            None,
        )
        .map_err(io::Error::other)?;

        let sa = SECURITY_ATTRIBUTES {
            nLength: u32::try_from(std::mem::size_of::<SECURITY_ATTRIBUTES>())
                .expect("SECURITY_ATTRIBUTES size fits u32"),
            lpSecurityDescriptor: psd.0,
            bInheritHandle: false.into(),
        };
        let handle = CreateFileW(
            PCWSTR(wide.as_ptr()),
            FILE_GENERIC_WRITE.0,
            FILE_SHARE_NONE,
            Some(&sa),
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            None,
        );
        // Free the descriptor regardless of how CreateFile fared.
        let _ = LocalFree(Some(HLOCAL(psd.0)));
        let handle = handle.map_err(io::Error::other)?;

        // Adopt the raw handle into a std `File` so the write + close are
        // ordinary std I/O (and the handle is closed on drop).
        let mut file = std::fs::File::from_raw_handle(handle.0 as *mut _);
        file.write_all(bytes)
    }
}

/// The current process user's SID in string form (`S-1-5-21-…`), for building
/// the owner-only DACL above.
#[cfg(windows)]
fn current_user_sid_string() -> io::Result<String> {
    use windows::core::PWSTR;
    use windows::Win32::Foundation::{
        CloseHandle, LocalFree, ERROR_INSUFFICIENT_BUFFER, HANDLE, HLOCAL,
    };
    use windows::Win32::Security::Authorization::ConvertSidToStringSidW;
    use windows::Win32::Security::{GetTokenInformation, TokenUser, TOKEN_QUERY, TOKEN_USER};
    use windows::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    unsafe {
        let mut token = HANDLE::default();
        OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token).map_err(io::Error::other)?;

        // First call sizes the buffer; it must fail with ERROR_INSUFFICIENT_BUFFER.
        // Any other outcome (a restricted token, or an unexpected success that
        // leaves `len` at 0) we surface rather than pressing on and dereferencing
        // a zero-length allocation.
        let mut len = 0u32;
        if let Err(e) = GetTokenInformation(token, TokenUser, None, 0, &mut len) {
            if e.code() != ERROR_INSUFFICIENT_BUFFER.to_hresult() {
                let _ = CloseHandle(token);
                return Err(io::Error::other(e));
            }
        }
        if len == 0 {
            let _ = CloseHandle(token);
            return Err(io::Error::other("could not size token information"));
        }

        // Back the buffer with `u64` so it's 8-byte aligned for the `TOKEN_USER`
        // cast below — the struct holds a pointer, and a `Vec<u8>` would only be
        // 1-byte aligned (an unaligned read of the pointer field is UB).
        let mut buf = vec![0u64; (len as usize).div_ceil(8)];
        let filled = GetTokenInformation(
            token,
            TokenUser,
            Some(buf.as_mut_ptr() as *mut _),
            len,
            &mut len,
        );
        let _ = CloseHandle(token);
        filled.map_err(io::Error::other)?;

        let token_user = &*(buf.as_ptr() as *const TOKEN_USER);
        let mut sid_w = PWSTR::null();
        ConvertSidToStringSidW(token_user.User.Sid, &mut sid_w).map_err(io::Error::other)?;
        // Free the LocalAlloc'd SID string on every path, including a to_string error.
        let result = sid_w.to_string().map_err(io::Error::other);
        let _ = LocalFree(Some(HLOCAL(sid_w.0 as *mut _)));
        result
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
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
        let path = std::env::temp_dir().join(format!(
            "tether-pairing-corrupt-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, b"this is not json").expect("write corrupt file");
        let result = PairedStore::load(&path);
        assert!(
            result.is_err(),
            "corrupt file must be Err, not an empty store"
        );
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
        hosts.insert(
            "192.168.1.5:7654".to_string(),
            &fp,
            "desktop".to_string(),
            1_700_000_000,
        );
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
        hosts.insert(
            "host.local:7654".to_string(),
            &fp,
            "work".to_string(),
            1_700_000_010,
        );
        hosts.save(&path).expect("save");

        let loaded = KnownHosts::load(&path).expect("load");
        assert_eq!(loaded.fingerprint("host.local:7654"), Some(fp));
        assert_eq!(loaded.len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_last_connected_only_touches_existing_hosts() {
        let mut hosts = KnownHosts::default();
        let fp = [8u8; 32];
        hosts.insert("a:7654".to_string(), &fp, "a".to_string(), 100);
        // New entries start with no last-connected stamp.
        let entry = hosts.iter().next().unwrap().1;
        assert_eq!(entry.last_connected_unix, None);

        assert!(hosts.set_last_connected("a:7654", 200));
        assert_eq!(
            hosts.iter().next().unwrap().1.last_connected_unix,
            Some(200)
        );
        // Unknown address is a no-op, reported as not-present.
        assert!(!hosts.set_last_connected("b:7654", 300));
    }

    #[test]
    fn rename_changes_label_not_fingerprint() {
        let mut hosts = KnownHosts::default();
        let fp = [9u8; 32];
        hosts.insert("a:7654".to_string(), &fp, "old".to_string(), 100);
        assert!(hosts.rename("a:7654", "new".to_string()));
        let entry = hosts.iter().next().unwrap().1;
        assert_eq!(entry.label, "new");
        // Rename must not disturb the pinned identity.
        assert_eq!(hosts.fingerprint("a:7654"), Some(fp));
        assert!(!hosts.rename("missing:7654", "x".to_string()));
    }

    /// A `known_hosts.json` written before `last_connected_unix` existed must
    /// still load — the field defaults to `None` rather than failing the parse
    /// (which would fail-closed and drop the user's whole address book).
    #[test]
    fn host_entry_without_last_connected_field_deserializes() {
        let legacy = r#"{
            "hosts": {
                "host.local:7654": {
                    "fingerprint": "sha256:0000",
                    "label": "work",
                    "paired_at_unix": 1700000000
                }
            }
        }"#;
        let hosts: KnownHosts = serde_json::from_str(legacy).expect("legacy parses");
        let entry = hosts.iter().next().unwrap().1;
        assert_eq!(entry.last_connected_unix, None);
        assert_eq!(entry.label, "work");
    }

    // The hardening this test guards is the *DACL*, not the write: a plain
    // `fs::write` would still round-trip bytes. Verify the file's DACL is
    // inheritance-protected (`SE_DACL_PROTECTED`), so a loosened parent-dir ACL
    // cannot grant another local user access to this trust root.
    #[cfg(windows)]
    #[test]
    fn write_private_sets_protected_dacl() {
        use std::os::windows::ffi::OsStrExt;

        use windows::core::PCWSTR;
        use windows::Win32::Foundation::{LocalFree, HLOCAL};
        use windows::Win32::Security::Authorization::{GetNamedSecurityInfoW, SE_FILE_OBJECT};
        use windows::Win32::Security::{
            GetSecurityDescriptorControl, DACL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR,
            SE_DACL_PROTECTED,
        };

        let path =
            std::env::temp_dir().join(format!("tether-dacl-test-{}.bin", std::process::id()));
        let _ = std::fs::remove_file(&path);
        write_private(&path, b"secret").expect("write");
        assert_eq!(std::fs::read(&path).expect("read"), b"secret");

        unsafe {
            let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
            let mut psd = PSECURITY_DESCRIPTOR(std::ptr::null_mut());
            GetNamedSecurityInfoW(
                PCWSTR(wide.as_ptr()),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                None,
                None,
                None,
                None,
                &mut psd,
            )
            .ok()
            .expect("GetNamedSecurityInfo");

            let mut control: u16 = 0;
            let mut revision = 0u32;
            GetSecurityDescriptorControl(psd, &mut control, &mut revision).expect("control");
            let _ = LocalFree(Some(HLOCAL(psd.0)));
            // Clean up before asserting so a failure doesn't strand the file.
            let _ = std::fs::remove_file(&path);
            assert!(
                control & SE_DACL_PROTECTED.0 != 0,
                "file DACL must be inheritance-protected"
            );
        }
    }

    #[test]
    fn known_hosts_corrupt_file_is_an_error() {
        let path = std::env::temp_dir().join(format!(
            "tether-known-hosts-corrupt-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, b"not json at all").expect("write corrupt");
        assert!(KnownHosts::load(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
