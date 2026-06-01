//! Self-signed certificate generation + fingerprint-pinning verifier.
//!
//! Trust model: the host generates a fresh self-signed cert at
//! startup. Its SHA-256 fingerprint is exchanged out of band (user pastes
//! it into the client, scans a QR code, etc.). The client pins that
//! fingerprint via [`PinnedCertVerifier`] and accepts no other cert.
//!
//! Signature verification is delegated to
//! [`rustls::crypto::verify_tls12_signature`] /
//! [`rustls::crypto::verify_tls13_signature`] using the ring provider's
//! `WebPkiSupportedAlgorithms`, so the cryptographic handshake is checked
//! the same way standard rustls does it — we only override the *trust*
//! decision (fingerprint match instead of PKI chain).

use std::path::Path;
use std::sync::Arc;

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::WebPkiSupportedAlgorithms;
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::{DigitallySignedStruct, DistinguishedName, SignatureScheme};
use sha2::{Digest, Sha256};

use crate::Result;

/// SHA-256 of an X.509 cert (DER-encoded bytes).
pub type CertFingerprint = [u8; 32];

pub struct SelfSignedCert {
    pub chain: Vec<CertificateDer<'static>>,
    pub key: PrivatePkcs8KeyDer<'static>,
    pub fingerprint: CertFingerprint,
}

pub fn generate_self_signed(subject_alt_names: Vec<String>) -> Result<SelfSignedCert> {
    let cert = rcgen::generate_simple_self_signed(subject_alt_names)?;
    let der: CertificateDer<'static> = cert.cert.der().clone();
    let fingerprint = sha256(&der);
    let key_der = cert.signing_key.serialize_der();
    Ok(SelfSignedCert {
        chain: vec![der],
        key: PrivatePkcs8KeyDer::from(key_der),
        fingerprint,
    })
}

pub(crate) fn sha256(bytes: &[u8]) -> CertFingerprint {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

/// Load a previously-generated self-signed cert from `dir`, or generate
/// a fresh one and write it there if no cert exists. Stable fingerprint
/// across runs — the user only has to copy it to the client once.
///
/// Files written: `host_cert.der` (PKIX DER) and `host_key.der` (PKCS8
/// DER). On unix, the key is created with mode `0o600`. A future
/// `--regenerate-cert` flag would delete both files and call this
/// again; today the operator can just `rm` them.
///
/// Returns the same shape `generate_self_signed` does, so callers
/// can't tell whether the cert was loaded or freshly generated — that
/// distinction lives in the (optional) log line at the call site.
pub fn load_or_generate_persistent(
    dir: &Path,
    subject_alt_names: Vec<String>,
) -> Result<SelfSignedCert> {
    load_or_generate_named(dir, "host_cert.der", "host_key.der", subject_alt_names)
}

/// The client's analogue of [`load_or_generate_persistent`]: a stable
/// self-signed *client* identity loaded from (or written to) `dir` as
/// `client_cert.der` / `client_key.der`. With mutual TLS the host captures
/// this cert's fingerprint and checks it against its paired-clients allowlist,
/// so the client must present the *same* identity across reconnects for
/// one-click pairing to hold.
pub fn load_or_generate_client_identity(dir: &Path) -> Result<SelfSignedCert> {
    load_or_generate_named(
        dir,
        "client_cert.der",
        "client_key.der",
        vec!["tether-client".into()],
    )
}

/// Shared loader for a persistent self-signed identity under `dir`, keyed by
/// the cert/key filenames. Stable fingerprint across runs; `rm` the two files
/// to rotate.
fn load_or_generate_named(
    dir: &Path,
    cert_file: &str,
    key_file: &str,
    subject_alt_names: Vec<String>,
) -> Result<SelfSignedCert> {
    let cert_path = dir.join(cert_file);
    let key_path = dir.join(key_file);

    if cert_path.exists() && key_path.exists() {
        let der_bytes = std::fs::read(&cert_path)?;
        let key_bytes = std::fs::read(&key_path)?;
        let fingerprint = sha256(&der_bytes);
        return Ok(SelfSignedCert {
            chain: vec![CertificateDer::from(der_bytes)],
            key: PrivatePkcs8KeyDer::from(key_bytes),
            fingerprint,
        });
    }

    let fresh = generate_self_signed(subject_alt_names)?;
    std::fs::create_dir_all(dir)?;
    // Write both files to temp siblings, then rename into place. A crash
    // between the two renames must not leave a fresh cert paired with a stale
    // (or missing) key: on the next run both `exists()` checks would pass and
    // we'd load a mismatched pair that fails every TLS handshake until the
    // operator manually deletes the files. We only reach here when at least one
    // final file is missing; remove any surviving one first so a mid-rename
    // crash can leave *at most one* fresh file — which the both-exist guard
    // above treats as "incomplete" and regenerates — never a mixed pair.
    let tmp_cert = dir.join(format!("{cert_file}.tmp"));
    let tmp_key = dir.join(format!("{key_file}.tmp"));
    // The cert is a public artifact (it carries only the public key + SAN, and
    // its fingerprint is exchanged on the wire anyway), so it's written with
    // default permissions on both platforms; only the private key gets the
    // owner-only treatment. Tampering with the public cert under a loosened dir
    // ACL can at worst desync it from the protected key (a load-time failure),
    // not forge an identity — that needs the key, which is locked down.
    std::fs::write(&tmp_cert, fresh.chain[0].as_ref())?;
    write_key_file(&tmp_key, fresh.key.secret_pkcs8_der())?;
    // Clear any stale survivor from an earlier incomplete run before publishing.
    let _ = std::fs::remove_file(&cert_path);
    let _ = std::fs::remove_file(&key_path);
    std::fs::rename(&tmp_cert, &cert_path)?;
    std::fs::rename(&tmp_key, &key_path)?;
    Ok(fresh)
}

#[cfg(unix)]
fn write_key_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
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

// SECURITY: the Windows analogue of the `0o600` above. We create the key file
// with an explicit owner-only, inheritance-protected DACL rather than inheriting
// the cert directory's ACL — so even on a machine whose `%USERPROFILE%\.tether`
// ACL has been loosened, a different local user can't read this private key and
// impersonate the identity. Building the descriptor at CreateFile time (not
// tightening after a normal write) means the key never exists with a looser ACL.
// The DACL grants *only* the current user — no SYSTEM/Administrators ACE,
// mirroring how Unix `0o600` excludes root from the explicit mode bits (admins
// can still take ownership, so restricting them would be theater). A side effect,
// as on Unix, is that a backup agent running as SYSTEM can't read the key without
// a privilege that bypasses DACLs.
//
// NOTE: byte-for-byte identical to `tether_pairing::store::write_private`'s
// Windows arm; the two live in separate crates (this one stays free of spake2,
// that one of quinn) so the shim is duplicated rather than shared. Keep in sync.
#[cfg(windows)]
fn write_key_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
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
        .map_err(std::io::Error::other)?;

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
        let handle = handle.map_err(std::io::Error::other)?;

        // Adopt the raw handle into a std `File` so the write + close are
        // ordinary std I/O (and the handle is closed on drop).
        let mut file = std::fs::File::from_raw_handle(handle.0 as *mut _);
        file.write_all(bytes)
    }
}

/// The current process user's SID in string form (`S-1-5-21-…`), for building
/// the owner-only DACL above.
#[cfg(windows)]
fn current_user_sid_string() -> std::io::Result<String> {
    use windows::core::PWSTR;
    use windows::Win32::Foundation::{
        CloseHandle, LocalFree, ERROR_INSUFFICIENT_BUFFER, HANDLE, HLOCAL,
    };
    use windows::Win32::Security::Authorization::ConvertSidToStringSidW;
    use windows::Win32::Security::{GetTokenInformation, TokenUser, TOKEN_QUERY, TOKEN_USER};
    use windows::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    unsafe {
        let mut token = HANDLE::default();
        OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token)
            .map_err(std::io::Error::other)?;

        // First call sizes the buffer; it must fail with ERROR_INSUFFICIENT_BUFFER.
        // Any other outcome (a restricted token, or an unexpected success that
        // leaves `len` at 0) we surface rather than pressing on and dereferencing
        // a zero-length allocation.
        let mut len = 0u32;
        if let Err(e) = GetTokenInformation(token, TokenUser, None, 0, &mut len) {
            if e.code() != ERROR_INSUFFICIENT_BUFFER.to_hresult() {
                let _ = CloseHandle(token);
                return Err(std::io::Error::other(e));
            }
        }
        if len == 0 {
            let _ = CloseHandle(token);
            return Err(std::io::Error::other("could not size token information"));
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
        filled.map_err(std::io::Error::other)?;

        let token_user = &*(buf.as_ptr() as *const TOKEN_USER);
        let mut sid_w = PWSTR::null();
        ConvertSidToStringSidW(token_user.User.Sid, &mut sid_w).map_err(std::io::Error::other)?;
        // Free the LocalAlloc'd SID string on every path, including a to_string error.
        let result = sid_w.to_string().map_err(std::io::Error::other);
        let _ = LocalFree(Some(HLOCAL(sid_w.0 as *mut _)));
        result
    }
}

#[derive(Debug)]
pub struct PinnedCertVerifier {
    expected: CertFingerprint,
    supported_schemes: WebPkiSupportedAlgorithms,
}

impl PinnedCertVerifier {
    pub fn new(expected: CertFingerprint) -> Arc<Self> {
        let supported_schemes =
            rustls::crypto::ring::default_provider().signature_verification_algorithms;
        Arc::new(Self {
            expected,
            supported_schemes,
        })
    }
}

impl ServerCertVerifier for PinnedCertVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, rustls::Error> {
        let got = sha256(end_entity);
        if got == self.expected {
            Ok(ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::General(
                "certificate fingerprint mismatch".into(),
            ))
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.supported_schemes)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.supported_schemes)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.supported_schemes.supported_schemes()
    }
}

/// Client-side server-cert verifier that accepts **any** well-formed server
/// certificate. Used only during **first-contact pairing**, when the client
/// does not yet know the host's fingerprint — the SPAKE2 key-confirmation
/// (bound to the TLS exporter) is what authenticates the host, after which the
/// client persists the observed cert fingerprint to known-hosts and switches
/// to [`PinnedCertVerifier`] on every reconnect. Like the permissive
/// client-cert verifier, it still proves key-possession via the signature
/// delegation, so it's "any host that owns its cert", not "any bytes".
#[derive(Debug)]
pub struct PermissiveServerCertVerifier {
    supported_schemes: WebPkiSupportedAlgorithms,
}

impl PermissiveServerCertVerifier {
    pub fn new() -> Arc<Self> {
        let supported_schemes =
            rustls::crypto::ring::default_provider().signature_verification_algorithms;
        Arc::new(Self { supported_schemes })
    }
}

impl ServerCertVerifier for PermissiveServerCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.supported_schemes)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.supported_schemes)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.supported_schemes.supported_schemes()
    }
}

/// Server-side client-cert verifier that accepts **any** well-formed client
/// certificate, deferring the actual authorization decision to the
/// application layer (the paired-clients allowlist).
///
/// Why "permissive": the host's trust anchor is a fingerprint allowlist, not
/// a CA, and a brand-new client pairing for the first time is by definition
/// not yet on that list — so the TLS verifier has no basis to accept or reject
/// on identity. Instead it requires *a* cert and verifies the holder actually
/// owns its private key (the `verify_tls1{2,3}_signature` delegation below, on
/// the CertificateVerify message), which is what lets the host read a stable
/// `peer_identity()` fingerprint. Authorization — "is this fingerprint paired,
/// or is a pairing window open?" — happens after the handshake, default-deny.
/// `client_auth_mandatory` stays at its `true` default: every client must
/// present a cert.
#[derive(Debug)]
pub struct PermissiveClientCertVerifier {
    supported_schemes: WebPkiSupportedAlgorithms,
}

impl PermissiveClientCertVerifier {
    pub fn new() -> Arc<Self> {
        let supported_schemes =
            rustls::crypto::ring::default_provider().signature_verification_algorithms;
        Arc::new(Self { supported_schemes })
    }
}

impl ClientCertVerifier for PermissiveClientCertVerifier {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        // Empty: we anchor trust on fingerprints (app-layer allowlist), not
        // CA subjects, so we offer the client no CA hints.
        &[]
    }

    fn verify_client_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _now: UnixTime,
    ) -> std::result::Result<ClientCertVerified, rustls::Error> {
        // Accept any cert as a trust anchor; the app layer authorizes by
        // fingerprint. Key-possession is still proven via the signature
        // checks below, so this is "any client that owns its cert", not
        // "any bytes".
        Ok(ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.supported_schemes)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.supported_schemes)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.supported_schemes.supported_schemes()
    }
}

/// Ensure the ring crypto provider is installed as the rustls default.
/// Idempotent; calling more than once is a no-op.
pub(crate) fn ensure_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique temp dir for one test, removed on drop. Avoids pulling in the
    /// `tempfile` crate just for these two tests.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "tether-tls-test-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn client_identity_fingerprint_is_stable_across_loads() {
        // The pairing feature relies on a client presenting the *same*
        // fingerprint across reconnects; a regression here would silently
        // break one-click reconnect (the host wouldn't recognize the client).
        let dir = TempDir::new("client-stable");
        let first = load_or_generate_client_identity(dir.path()).expect("first load");
        let second = load_or_generate_client_identity(dir.path()).expect("second load");
        assert_eq!(
            first.fingerprint, second.fingerprint,
            "fingerprint must not change between loads from the same directory"
        );
    }

    #[test]
    fn host_and_client_identities_use_separate_files() {
        // Both identities can coexist in one directory without clobbering each
        // other (host uses host_*.der, client uses client_*.der).
        let dir = TempDir::new("coexist");
        let host = load_or_generate_persistent(dir.path(), vec!["tether-host".into()])
            .expect("host identity");
        let client = load_or_generate_client_identity(dir.path()).expect("client identity");
        assert_ne!(
            host.fingerprint, client.fingerprint,
            "host and client identities must be distinct certs"
        );
        assert!(dir.path().join("host_cert.der").exists());
        assert!(dir.path().join("client_cert.der").exists());
    }
}
