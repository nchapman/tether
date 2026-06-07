use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use socket2::{Domain, Protocol, Socket, Type};
use tracing::{trace, warn};

use crate::{
    connection::{Connection, STREAM_PREAMBLE},
    pending::PendingConnection,
    tls::{
        ensure_crypto_provider, generate_self_signed, load_or_generate_client_identity,
        CertFingerprint, PermissiveServerCertVerifier, PinnedCertVerifier, SelfSignedCert,
    },
    Result, TransportError,
};

/// How the client authenticates the host's certificate.
pub enum ServerAuth {
    /// Reconnect to a known host: pin its cert to this fingerprint (from
    /// known-hosts). The normal, post-pairing path.
    Pinned(CertFingerprint),
    /// First-contact pairing: the host's fingerprint isn't known yet, so accept
    /// any cert at the TLS layer. The SPAKE2 key-confirmation (bound to the TLS
    /// exporter) authenticates the host; the caller then persists the observed
    /// fingerprint to known-hosts for future [`ServerAuth::Pinned`] connects.
    TrustOnFirstPair,
}

/// Target size for the UDP receive buffer (`SO_RCVBUF`). Linux's
/// default of ~208 KB overflows in milliseconds when a bursty H.265
/// keyframe (tens of KB at 5–8 Mbps) arrives faster than the recv
/// loop drains. The kernel silently drops packets at that point —
/// before quinn's own datagram buffer ever sees them — and the
/// decoder gets a fragment-loss storm. 16 MiB gives multi-second
/// headroom on a fast LAN.
///
/// The kernel caps the effective size at `net.core.rmem_max`. If
/// `set_recv_buffer_size` returns a much smaller value, we log a
/// warning telling the operator how to raise the ceiling rather
/// than failing startup — the smaller buffer still works, just
/// with less margin.
///
/// Linux quirk: `getsockopt(SO_RCVBUF)` returns roughly 2× the
/// kernel's true allocation (the doubling accounts for protocol
/// overhead). Our readback check compares against `_BYTES / 4` to
/// account for that — anything below ~4 MiB true allocation
/// (~8 MiB reported) warrants a warning.
const UDP_RECV_BUFFER_BYTES: usize = 16 * 1024 * 1024;

pub struct Client {
    endpoint: quinn::Endpoint,
    /// The client's own self-signed identity, presented to the host under
    /// mutual TLS. The host fingerprints it for the paired-clients allowlist.
    identity: SelfSignedCert,
}

impl Client {
    /// Create a new client endpoint bound to an OS-chosen local UDP port with
    /// a fresh **ephemeral** client identity. The identity changes on every
    /// call — fine for tests and one-shot processes. Long-running clients that
    /// want stable pairing should use [`Self::with_identity`].
    pub fn new() -> Result<Self> {
        let identity = generate_self_signed(vec!["tether-client".into()])?;
        Self::with_identity_cert(identity)
    }

    /// Create a client endpoint with a **persistent** identity loaded from (or
    /// generated into) `identity_dir` as `client_cert.der` / `client_key.der`.
    /// Stable fingerprint across runs, so a host that paired this client once
    /// keeps recognizing it.
    pub fn with_identity(identity_dir: &Path) -> Result<Self> {
        let identity = load_or_generate_client_identity(identity_dir)?;
        Self::with_identity_cert(identity)
    }

    /// SHA-256 fingerprint of this client's certificate — the identity the
    /// host stores in its allowlist when pairing.
    pub fn fingerprint(&self) -> CertFingerprint {
        self.identity.fingerprint
    }

    fn with_identity_cert(identity: SelfSignedCert) -> Result<Self> {
        ensure_crypto_provider();
        let bind = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0));
        let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
        socket.bind(&bind.into())?;
        // Best-effort recv-buffer bump. Failure here isn't fatal —
        // we just lose the headroom. Same for a kernel-imposed cap
        // below our request.
        if let Err(e) = socket.set_recv_buffer_size(UDP_RECV_BUFFER_BYTES) {
            warn!(
                error = %e,
                requested = UDP_RECV_BUFFER_BYTES,
                "set_recv_buffer_size failed; using kernel default. Raise net.core.rmem_max for headroom."
            );
        } else {
            match socket.recv_buffer_size() {
                Ok(actual) if actual < UDP_RECV_BUFFER_BYTES / 4 => {
                    warn!(
                        requested = UDP_RECV_BUFFER_BYTES,
                        actual,
                        "kernel capped UDP recv buffer well below requested size; \
                         consider `sysctl -w net.core.rmem_max={}` to reduce LAN \
                         freeze risk under bursty keyframes",
                        UDP_RECV_BUFFER_BYTES,
                    );
                }
                Ok(actual) => trace!(actual, "UDP recv buffer sized for headroom"),
                Err(e) => trace!(error = %e, "recv_buffer_size readback failed"),
            }
        }
        let std_socket: std::net::UdpSocket = socket.into();
        let runtime = quinn::default_runtime()
            .ok_or_else(|| std::io::Error::other("no async runtime found"))?;
        let endpoint =
            quinn::Endpoint::new(quinn::EndpointConfig::default(), None, std_socket, runtime)?;
        Ok(Self { endpoint, identity })
    }

    /// Connect to a server. `server_name` is the SNI / TLS server name —
    /// our self-signed cert is issued with SANs `["tether-host",
    /// "localhost"]` so either works. `expected_fingerprint` is the
    /// SHA-256 of the server's DER cert, exchanged out of band.
    /// Connect and return a [`PendingConnection`] — the TLS handshake is
    /// complete and the pairing stream is wired, but no session exists yet. The
    /// caller drives the pairing exchange (or a `Resume`) and then promotes it
    /// via [`PendingConnection::into_connection`]. `server_auth` selects
    /// whether the host cert is pinned (reconnect) or trusted-on-first-pair.
    pub async fn connect_pending(
        &self,
        addr: SocketAddr,
        server_name: &str,
        server_auth: ServerAuth,
    ) -> Result<PendingConnection> {
        // Present our identity under mutual TLS. Rebuild the key from its DER
        // bytes (PrivatePkcs8KeyDer isn't trivially cloneable) so this can be
        // called more than once on the same Client.
        let client_chain = self.identity.chain.clone();
        let client_key = PrivatePkcs8KeyDer::from(self.identity.key.secret_pkcs8_der().to_vec());
        let client_config = make_client_config(server_auth, client_chain, client_key.into())?;
        let connecting = self
            .endpoint
            .connect_with(client_config, addr, server_name)?;
        let conn = connecting.await?;
        trace!(remote = %conn.remote_address(), "client connection established");
        // Open the pairing stream (stream #1) and write its preamble. In QUIC,
        // opening a stream is local-only — the peer doesn't see it until the
        // first STREAM frame arrives — so the four-byte preamble forces a frame
        // that unblocks the host's `accept_bi`. Control and input open later,
        // in `into_connection`, only once authorized.
        let (mut pairing_send, pairing_recv) = conn.open_bi().await?;
        pairing_send.write_all(STREAM_PREAMBLE).await?;
        Ok(PendingConnection::new_client(
            conn,
            pairing_send,
            pairing_recv,
        ))
    }

    /// Convenience: connect and immediately promote to a session with **no
    /// pairing exchange**, pinning the host cert to `expected_fingerprint`.
    ///
    /// For transport-mechanics tests and simple peers only; the production
    /// client uses [`Self::connect_pending`] so it can pair or resume first.
    /// The host must use the matching convenience path ([`Server::accept`]) —
    /// pointing this at a host that runs a pairing exchange will hang (the host
    /// waits for pairing messages this path never sends).
    pub async fn connect(
        &self,
        addr: SocketAddr,
        server_name: &str,
        expected_fingerprint: CertFingerprint,
    ) -> Result<Connection> {
        let pending = self
            .connect_pending(addr, server_name, ServerAuth::Pinned(expected_fingerprint))
            .await?;
        pending.into_connection().await
    }

    /// Close the endpoint immediately, without waiting for in-flight
    /// CONNECTION_CLOSE frames to flush. Use [`Self::close_and_wait`] for
    /// graceful shutdown.
    pub fn close(&self, code: u32, reason: &[u8]) {
        self.endpoint.close(code.into(), reason);
    }

    /// Close the endpoint and await the flush of all open connections.
    /// Prefer this when the calling process is about to exit.
    pub async fn close_and_wait(&self, code: u32, reason: &[u8]) {
        self.endpoint.close(code.into(), reason);
        self.endpoint.wait_idle().await;
    }
}

fn make_client_config(
    server_auth: ServerAuth,
    client_chain: Vec<CertificateDer<'static>>,
    client_key: PrivateKeyDer<'static>,
) -> Result<quinn::ClientConfig> {
    // Pin TLS 1.3 explicitly (matching the server). QUIC already mandates 1.3
    // at the protocol layer, but being explicit keeps the exporter's RFC 8446
    // channel-binding guarantee from silently weakening if this config is ever
    // reused over a non-QUIC transport.
    let dangerous =
        rustls::ClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .dangerous();
    let crypto = match server_auth {
        ServerAuth::Pinned(fingerprint) => {
            dangerous.with_custom_certificate_verifier(PinnedCertVerifier::new(fingerprint))
        }
        ServerAuth::TrustOnFirstPair => {
            dangerous.with_custom_certificate_verifier(PermissiveServerCertVerifier::new())
        }
    }
    .with_client_auth_cert(client_chain, client_key)?;
    let quic = quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
        .map_err(|e| TransportError::Rustls(rustls::Error::General(format!("{e}"))))?;
    let mut config = quinn::ClientConfig::new(Arc::new(quic));
    config.transport_config(Arc::new(transport_config()));
    Ok(config)
}

fn transport_config() -> quinn::TransportConfig {
    let mut t = quinn::TransportConfig::default();
    t.keep_alive_interval(Some(Duration::from_secs(5)));
    t.max_idle_timeout(Some(
        Duration::from_secs(30)
            .try_into()
            .expect("30s fits in IdleTimeout"),
    ));
    // Match the server's headroom; see server.rs for rationale. The
    // datagram send buffer is currently unused on the client (control
    // and input both ride reliable streams today; no client→host
    // datagrams) but is pre-provisioned for the cursor datagram channel
    // that lands with tether-input. Total budget is ~12 MiB per
    // connection — fine on desktop targets, would need trimming on
    // embedded.
    t.datagram_receive_buffer_size(Some(8 * 1024 * 1024));
    t.datagram_send_buffer_size(4 * 1024 * 1024);
    // Cap peer-initiated uni streams. The host opens none (all video,
    // keyframes included, rides datagrams); the only uni stream is the
    // client→host input stream. The cap is purely defensive — without it a
    // hostile host could open streams to pin receive-side accept slots.
    t.max_concurrent_uni_streams(crate::MAX_CONCURRENT_UNI_STREAMS.into());
    // Cap host→client bidi streams. A well-behaved host opens none —
    // both bidi streams (pairing, control) are client-initiated — so
    // this is pure defense against a hostile host. Shares the
    // client→host constant; see MAX_CONCURRENT_BIDI_STREAMS.
    t.max_concurrent_bidi_streams(crate::MAX_CONCURRENT_BIDI_STREAMS.into());
    t
}
