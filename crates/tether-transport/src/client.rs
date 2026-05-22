use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use socket2::{Domain, Protocol, Socket, Type};
use tracing::{trace, warn};

use crate::{
    connection::{Connection, STREAM_PREAMBLE},
    tls::{ensure_crypto_provider, CertFingerprint, PinnedCertVerifier},
    Result, TransportError,
};

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
}

impl Client {
    /// Create a new client endpoint bound to an OS-chosen local UDP port.
    pub fn new() -> Result<Self> {
        ensure_crypto_provider();
        let bind: SocketAddr = "0.0.0.0:0".parse().expect("static literal");
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
        let endpoint = quinn::Endpoint::new(
            quinn::EndpointConfig::default(),
            None,
            std_socket,
            runtime,
        )?;
        Ok(Self { endpoint })
    }

    /// Connect to a server. `server_name` is the SNI / TLS server name —
    /// our self-signed cert is issued with SANs `["tether-host",
    /// "localhost"]` so either works. `expected_fingerprint` is the
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
    // Match the server's headroom; see server.rs for rationale. The
    // datagram send buffer is currently unused on the client (control
    // and input both ride reliable streams today; no client→host
    // datagrams) but is pre-provisioned for the cursor datagram channel
    // that lands with tether-input. Total budget is ~12 MiB per
    // connection — fine on desktop targets, would need trimming on
    // embedded.
    t.datagram_receive_buffer_size(Some(8 * 1024 * 1024));
    t.datagram_send_buffer_size(4 * 1024 * 1024);
    // Cap host→client uni streams. The client accepts these for the
    // reliable-keyframe protocol; without an explicit limit a hostile
    // host could open enough streams to pin 2 MiB of receive buffer
    // per stream (see MAX_VIDEO_STREAM_MESSAGE).
    t.max_concurrent_uni_streams(crate::MAX_CONCURRENT_UNI_STREAMS.into());
    t
}
