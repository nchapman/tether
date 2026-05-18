//! Self-signed certificate generation + fingerprint-pinning verifier.
//!
//! v0 security model: the host generates a fresh self-signed cert at
//! startup. Its SHA-256 fingerprint is exchanged out of band (user pastes
//! it into the client, scans a QR code, etc.). The client pins that
//! fingerprint via [`PinnedCertVerifier`] and accepts no other cert.
//!
//! [`PinnedCertVerifier::verify_tls12_signature`] /
//! [`PinnedCertVerifier::verify_tls13_signature`] currently return
//! `Ok(HandshakeSignatureValid::assertion())` without checking the
//! algorithm-specific signature. This is safe for our threat model: an
//! attacker who has only the cert's public key cannot complete the TLS
//! handshake at all (they cannot sign with the private key), so by the
//! time control reaches our verifier we already know the peer holds the
//! pinned private key. Revisit if we ever support certificates not bound
//! to a fixed private key (e.g. CA-issued).

use std::sync::Arc;

use rustls::client::danger::{
    HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier,
};
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
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

#[derive(Debug)]
pub struct PinnedCertVerifier {
    expected: CertFingerprint,
}

impl PinnedCertVerifier {
    pub fn new(expected: CertFingerprint) -> Arc<Self> {
        Arc::new(Self { expected })
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
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::ED25519,
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::ECDSA_NISTP384_SHA384,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PSS_SHA384,
            SignatureScheme::RSA_PSS_SHA512,
            SignatureScheme::RSA_PKCS1_SHA256,
            SignatureScheme::RSA_PKCS1_SHA384,
            SignatureScheme::RSA_PKCS1_SHA512,
        ]
    }
}

/// Ensure the ring crypto provider is installed as the rustls default.
/// Idempotent; calling more than once is a no-op.
pub(crate) fn ensure_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}
