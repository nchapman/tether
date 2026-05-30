use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tracing::{info, trace};

use crate::{
    connection::{Connection, STREAM_PREAMBLE_LEN},
    pending::PendingConnection,
    tls::{
        ensure_crypto_provider, generate_self_signed, load_or_generate_persistent,
        CertFingerprint, PermissiveClientCertVerifier, SelfSignedCert,
    },
    Result, TransportError,
};

pub struct Server {
    endpoint: quinn::Endpoint,
    fingerprint: CertFingerprint,
}

impl Server {
    /// Bind a server to the given UDP address with a freshly-generated
    /// ephemeral certificate. The fingerprint changes on every call;
    /// suitable for tests and one-shot processes that don't need
    /// stable identity. Long-running host processes should prefer
    /// [`Self::bind_persistent`] so the operator only copies the
    /// fingerprint to the client once.
    pub async fn bind(addr: SocketAddr) -> Result<Self> {
        let cert = generate_self_signed(default_subject_alt_names())?;
        Self::bind_with_cert(addr, cert).await
    }

    /// Bind with a self-signed cert loaded from `cert_dir`, generating
    /// and persisting a new one if no cert is present. Stable
    /// fingerprint across restarts — `rm` the files in `cert_dir` to
    /// rotate. Same on-wire behaviour as [`Self::bind`].
    pub async fn bind_persistent(addr: SocketAddr, cert_dir: &Path) -> Result<Self> {
        let cert = load_or_generate_persistent(cert_dir, default_subject_alt_names())?;
        Self::bind_with_cert(addr, cert).await
    }

    async fn bind_with_cert(addr: SocketAddr, cert: SelfSignedCert) -> Result<Self> {
        ensure_crypto_provider();

        let SelfSignedCert {
            chain,
            key,
            fingerprint,
        } = cert;

        // Mutual TLS: require a client cert so the host can read the peer's
        // fingerprint via `peer_identity()`. The verifier accepts any
        // well-formed cert; authorization against the paired-clients allowlist
        // happens at the app layer (default-deny). TLS 1.3 only, as quinn
        // requires.
        let crypto =
            rustls::ServerConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
                .with_client_cert_verifier(PermissiveClientCertVerifier::new())
                .with_single_cert(chain, key.into())?;
        let quic = quinn::crypto::rustls::QuicServerConfig::try_from(crypto)
            .map_err(|e| TransportError::Rustls(rustls::Error::General(format!("{e}"))))?;
        let mut server_config = quinn::ServerConfig::with_crypto(Arc::new(quic));
        server_config.transport_config(Arc::new(transport_config()));

        let endpoint = quinn::Endpoint::server(server_config, addr)?;
        info!(addr = %endpoint.local_addr()?, "tether-transport server bound");
        Ok(Self {
            endpoint,
            fingerprint,
        })
    }

    pub fn fingerprint(&self) -> CertFingerprint {
        self.fingerprint
    }

    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.endpoint.local_addr()?)
    }

    /// Accept the next incoming connection as a [`PendingConnection`] — the
    /// TLS handshake is complete and the pairing stream is wired, but no
    /// session exists yet. The caller authorizes the peer (allowlist check
    /// and/or a pairing exchange) and then either
    /// [`PendingConnection::into_connection`] or
    /// [`PendingConnection::reject`]s it. The host binary uses this path so an
    /// unauthorized client never reaches a session. Returns `None` when the
    /// endpoint has been closed.
    pub async fn accept_pending(&self) -> Option<Result<PendingConnection>> {
        let incoming = self.endpoint.accept().await?;
        Some(handle_incoming_pending(incoming).await)
    }

    /// Convenience: accept the next connection and immediately promote it to a
    /// session with **NO authentication and no pairing exchange**.
    ///
    /// This performs no allowlist check — any peer that completes the TLS
    /// handshake gets a session and can inject input. It exists only for
    /// transport-mechanics tests and simple peers. **Production code must use
    /// [`Self::accept_pending`]** and authorize the peer before
    /// [`PendingConnection::into_connection`]. The peer must use the matching
    /// convenience path ([`Client::connect`]): mixing a convenience side with a
    /// `*_pending` side that runs a pairing exchange will hang, because one
    /// side waits for pairing messages the other never sends. Returns `None`
    /// when the endpoint has been closed.
    pub async fn accept(&self) -> Option<Result<Connection>> {
        match self.accept_pending().await? {
            Ok(pending) => Some(pending.into_connection().await),
            Err(e) => Some(Err(e)),
        }
    }

    /// Stop accepting new connections and close the endpoint immediately,
    /// without waiting for in-flight CONNECTION_CLOSE frames to flush. Use
    /// [`Self::close_and_wait`] for graceful shutdown.
    pub fn close(&self, code: u32, reason: &[u8]) {
        self.endpoint.close(code.into(), reason);
    }

    /// Close the endpoint and await the flush of all open connections.
    /// Callers that exit the process right after shutting down should
    /// prefer this method over [`Self::close`] so the peer sees a clean
    /// CONNECTION_CLOSE instead of a silent timeout.
    pub async fn close_and_wait(&self, code: u32, reason: &[u8]) {
        self.endpoint.close(code.into(), reason);
        self.endpoint.wait_idle().await;
    }
}

async fn handle_incoming_pending(incoming: quinn::Incoming) -> Result<PendingConnection> {
    let conn = incoming.await?;
    trace!(remote = %conn.remote_address(), "incoming connection accepted");
    // The client opens streams in a fixed order, each preceded by a four-byte
    // preamble (unused bytes — see Client) that forces a STREAM frame so our
    // `accept_*` completes. Stream #1 is the pairing stream; control and input
    // come later, only once the peer is authorized (see
    // `PendingConnection::into_connection`).
    let (pairing_send, mut pairing_recv) = conn.accept_bi().await?;
    let mut preamble = [0u8; STREAM_PREAMBLE_LEN];
    pairing_recv
        .read_exact(&mut preamble)
        .await
        .map_err(crate::TransportError::from_read_exact)?;
    Ok(PendingConnection::new_host(conn, pairing_send, pairing_recv))
}

fn default_subject_alt_names() -> Vec<String> {
    vec!["tether-host".into(), "localhost".into()]
}

fn transport_config() -> quinn::TransportConfig {
    let mut t = quinn::TransportConfig::default();
    t.keep_alive_interval(Some(Duration::from_secs(5)));
    t.max_idle_timeout(Some(
        Duration::from_secs(30)
            .try_into()
            .expect("30s fits in IdleTimeout"),
    ));
    // Headroom for video: a 1080p H.264 keyframe at 4 Mbps is ~50 KB, and
    // we'd like the receive queue to absorb several frames of bursting
    // without dropping fragments mid-frame (a single lost fragment
    // corrupts every P-frame until the next IDR). 8 MiB is generous on
    // LAN; tune down once we have telemetry.
    t.datagram_receive_buffer_size(Some(8 * 1024 * 1024));
    t.datagram_send_buffer_size(4 * 1024 * 1024);
    // Cap client→host uni streams. Today only the input stream is
    // opened in this direction at handshake time; the cap denies a
    // hostile client the ability to open many concurrent streams.
    t.max_concurrent_uni_streams(crate::MAX_CONCURRENT_UNI_STREAMS.into());
    // Cap client→host bidi streams. The client opens two (pairing,
    // then control) that transiently overlap during into_connection;
    // see MAX_CONCURRENT_BIDI_STREAMS for why the cap must be ≥ 2 and
    // how it still bounds a hostile client's stream count.
    t.max_concurrent_bidi_streams(crate::MAX_CONCURRENT_BIDI_STREAMS.into());
    t
}
