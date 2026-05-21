use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tracing::{info, trace};

use crate::{
    connection::{Connection, STREAM_PREAMBLE_LEN},
    tls::{
        ensure_crypto_provider, generate_self_signed, load_or_generate_persistent,
        CertFingerprint, SelfSignedCert,
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

        let mut server_config = quinn::ServerConfig::with_single_cert(chain, key.into())
            .map_err(|e| TransportError::Rustls(rustls::Error::General(format!("{e}"))))?;
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

    /// Accept the next incoming connection. Returns `None` when the
    /// endpoint has been closed.
    pub async fn accept(&self) -> Option<Result<Connection>> {
        let incoming = self.endpoint.accept().await?;
        Some(handle_incoming(incoming).await)
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

async fn handle_incoming(incoming: quinn::Incoming) -> Result<Connection> {
    let conn = incoming.await?;
    trace!(remote = %conn.remote_address(), "incoming connection accepted");
    // Client opens streams in a fixed order, each followed by a
    // four-byte preamble (the bytes themselves are unused — see Client).
    //   1. bidirectional control stream
    //   2. unidirectional input stream
    let (control_send, mut control_recv) = conn.accept_bi().await?;
    let mut preamble = [0u8; STREAM_PREAMBLE_LEN];
    control_recv
        .read_exact(&mut preamble)
        .await
        .map_err(crate::TransportError::from_read_exact)?;
    let mut input_recv = conn.accept_uni().await?;
    input_recv
        .read_exact(&mut preamble)
        .await
        .map_err(crate::TransportError::from_read_exact)?;
    Ok(Connection::new_host(
        conn,
        control_send,
        control_recv,
        input_recv,
    ))
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
    // TODO(latency): switch to a no-pacer / BBR controller for LAN once we
    // have measurements showing Cubic adds visible jitter. Per the expert
    // review, the right place is
    // `TransportConfig::congestion_controller_factory`.
    t
}
