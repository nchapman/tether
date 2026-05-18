//! Self-signed certificate generation + fingerprint-pinning verifier.
//!
//! v0 security model: the host generates a fresh self-signed cert at
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

use std::sync::Arc;

use rustls::client::danger::{
    HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier,
};
use rustls::crypto::WebPkiSupportedAlgorithms;
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

/// Ensure the ring crypto provider is installed as the rustls default.
/// Idempotent; calling more than once is a no-op.
pub(crate) fn ensure_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}
