use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tracing::trace;

use crate::{
    connection::{Connection, STREAM_PREAMBLE},
    tls::{ensure_crypto_provider, CertFingerprint, PinnedCertVerifier},
    Result, TransportError,
};

pub struct Client {
    endpoint: quinn::Endpoint,
}

impl Client {
    /// Create a new client endpoint bound to an OS-chosen local UDP port.
    pub fn new() -> Result<Self> {
        ensure_crypto_provider();
        let bind: SocketAddr = "0.0.0.0:0".parse().expect("static literal");
        let endpoint = quinn::Endpoint::client(bind)?;
        Ok(Self { endpoint })
    }

    /// Connect to a server. `server_name` is the SNI / TLS server name —
    /// our self-signed cert is issued with SANs `["tether-host",
    /// "localhost"]` so either works in v0. `expected_fingerprint` is the
    /// SHA-256 of the server's DER cert, exchanged out of band.
    pub async fn connect(
        &self,
        addr: SocketAddr,
        server_name: &str,
        expected_fingerprint: CertFingerprint,
    ) -> Result<Connection> {
        let client_config = make_client_config(expected_fingerprint)?;
        let connecting = self.endpoint.connect_with(client_config, addr, server_name)?;
        let conn = connecting.await?;
        trace!(remote = %conn.remote_address(), "client connection established");
        // Match the stream-open order the server expects:
        //   1. bidirectional control stream
        //   2. unidirectional input stream
        //
        // In QUIC, opening a stream is local-only — the peer doesn't see
        // the stream until the first STREAM frame arrives. We write a
        // four-byte zero "length prefix" on each newly opened stream as a
        // preamble, which the server consumes and discards. This unblocks
        // the server's accept_*() calls so the Connection is fully wired
        // by the time both constructors return, even when no real
        // application data is ready yet.
        let (mut control_send, control_recv) = conn.open_bi().await?;
        control_send.write_all(STREAM_PREAMBLE).await?;
        let mut input_send = conn.open_uni().await?;
        input_send.write_all(STREAM_PREAMBLE).await?;
        Ok(Connection::new_client(
            conn,
            control_send,
            control_recv,
            input_send,
        ))
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

fn make_client_config(fingerprint: CertFingerprint) -> Result<quinn::ClientConfig> {
    let crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(PinnedCertVerifier::new(fingerprint))
        .with_no_client_auth();
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
    // Match the server's headroom; see server.rs for rationale. The send
    // buffer is currently unused on the client (v0 only sends control
    // stream traffic; no client→host datagrams yet) but is pre-provisioned
    // for the input + cursor datagram channels that land with
    // tether-input. Total budget is ~12 MiB per connection — fine on
    // desktop targets, would need trimming on embedded.
    t.datagram_receive_buffer_size(Some(8 * 1024 * 1024));
    t.datagram_send_buffer_size(4 * 1024 * 1024);
    t
}
