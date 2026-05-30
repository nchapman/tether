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

use rustls::client::danger::{
    HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier,
};
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
    // between the two writes must not leave a fresh cert paired with a stale
    // (or missing) key: on the next run both `exists()` checks would pass and
    // we'd load a mismatched pair that fails every TLS handshake until the
    // operator manually deletes the files.
    let tmp_cert = dir.join(format!("{cert_file}.tmp"));
    let tmp_key = dir.join(format!("{key_file}.tmp"));
    std::fs::write(&tmp_cert, fresh.chain[0].as_ref())?;
    write_key_file(&tmp_key, fresh.key.secret_pkcs8_der())?;
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

// SECURITY: on Windows the private key relies on the inherited ACL of the
// per-user cert directory (`%USERPROFILE%\.tether`), which by default excludes
// other standard users. We do not yet set an explicit owner-only DACL, so a
// process running as a *different* local user on a machine with a loosened
// profile ACL could read the key and impersonate this identity. Acceptable for
// the single-user alpha; harden with SetNamedSecurityInfoW before multi-user
// use. Same gap applies to the paired-peer store in tether-pairing.
#[cfg(not(unix))]
fn write_key_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::write(path, bytes)
}

#[derive(Debug)]
pub struct PinnedCertVerifier {
    expected: CertFingerprint,
    supported_schemes: WebPkiSupportedAlgorithms,
}

impl PinnedCertVerifier {
    pub fn new(expected: CertFingerprint) -> Arc<Self> {
        let supported_schemes = rustls::crypto::ring::default_provider()
            .signature_verification_algorithms;
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
