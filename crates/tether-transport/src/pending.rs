//! Half-open connection awaiting authorization.
//!
//! A [`PendingConnection`] is what `Server::accept_pending` /
//! `Client::connect_pending` return *before* a session exists: the QUIC + TLS
//! handshake is complete and a single **pairing stream** is wired, but the
//! control and input streams are not. This is the structural authorization
//! gate — the only way to a session-capable [`Connection`] is
//! [`into_connection`](PendingConnection::into_connection), which the host
//! calls *only* after the peer is authorized (allowlist hit or a successful
//! pairing exchange). An unauthorized peer is [`reject`](PendingConnection::reject)ed
//! and never gets control/input streams, so the host's input-injection path is
//! unreachable for it.
//!
//! The pairing exchange itself (SPAKE2 + confirmation) is driven by the
//! application over [`send_pairing`](PendingConnection::send_pairing) /
//! [`recv_pairing`](PendingConnection::recv_pairing) using the
//! [`tether_protocol::pairing`] message types; the crypto lives in
//! `tether-pairing`. This layer only moves bytes and exposes the two handles
//! the confirmation binds to: the peer cert fingerprint and the TLS exporter.

use std::net::SocketAddr;

use crate::connection::{
    export_keying_material, peer_cert_fingerprint, read_framed, write_framed, Connection,
    STREAM_PREAMBLE, STREAM_PREAMBLE_LEN,
};
use crate::tls::CertFingerprint;
use crate::{Result, TransportError};

/// Which end opened the connection — determines whether
/// [`PendingConnection::into_connection`] accepts (host) or opens (client) the
/// control and input streams.
pub(crate) enum Role {
    Host,
    Client,
}

/// A connection that has completed its TLS handshake and wired its pairing
/// stream, but has not yet been authorized into a full [`Connection`]. See the
/// module docs for the authorization-gate rationale.
pub struct PendingConnection {
    conn: quinn::Connection,
    pairing_send: quinn::SendStream,
    pairing_recv: quinn::RecvStream,
    role: Role,
}

impl PendingConnection {
    pub(crate) fn new_host(
        conn: quinn::Connection,
        pairing_send: quinn::SendStream,
        pairing_recv: quinn::RecvStream,
    ) -> Self {
        Self {
            conn,
            pairing_send,
            pairing_recv,
            role: Role::Host,
        }
    }

    pub(crate) fn new_client(
        conn: quinn::Connection,
        pairing_send: quinn::SendStream,
        pairing_recv: quinn::RecvStream,
    ) -> Self {
        Self {
            conn,
            pairing_send,
            pairing_recv,
            role: Role::Client,
        }
    }

    pub fn remote_address(&self) -> SocketAddr {
        self.conn.remote_address()
    }

    /// SHA-256 fingerprint of the peer's leaf cert (the identity to authorize),
    /// or `None` if the peer presented none. A caller treating `None` as
    /// anything but "reject" would defeat the auth gate.
    pub fn peer_cert_fingerprint(&self) -> Option<CertFingerprint> {
        peer_cert_fingerprint(&self.conn)
    }

    /// Export RFC 5705 keying material bound to this connection — the channel
    /// binding the pairing confirmation MAC mixes in. See
    /// [`Connection::export_keying_material`].
    pub fn export_keying_material(
        &self,
        label: &[u8],
        context: &[u8],
        len: usize,
    ) -> Result<Vec<u8>> {
        export_keying_material(&self.conn, label, context, len)
    }

    /// Send one length-framed pairing message (a [`tether_protocol::pairing`]
    /// type) on the pairing stream.
    pub async fn send_pairing<T: serde::Serialize>(&mut self, msg: &T) -> Result<()> {
        let bytes = tether_protocol::encode(msg)?;
        write_framed(&mut self.pairing_send, &bytes).await
    }

    /// Receive one length-framed pairing message from the pairing stream.
    pub async fn recv_pairing<T: for<'de> serde::Deserialize<'de>>(&mut self) -> Result<T> {
        let bytes = read_framed(&mut self.pairing_recv).await?;
        Ok(tether_protocol::decode(&bytes)?)
    }

    /// Refuse this peer: close the QUIC connection without ever wiring the
    /// control or input streams. No [`Connection`] is produced, so no session
    /// or input task can exist for it.
    pub fn reject(self, code: u32, reason: &[u8]) {
        self.conn.close(code.into(), reason);
    }

    /// Promote an authorized pending connection to a full [`Connection`] by
    /// wiring the control and input streams (host accepts; client opens), in
    /// the order the peer expects. Call this only after authorization.
    pub async fn into_connection(self) -> Result<Connection> {
        let PendingConnection {
            conn,
            mut pairing_send,
            pairing_recv,
            role,
        } = self;
        // The pairing exchange is done; finish our half cleanly and drop the
        // recv half so the peer sees the stream close. A finish error only
        // means the peer already closed/reset — in which case the stream
        // accepts/opens below fail anyway — so log and proceed.
        if let Err(e) = pairing_send.finish() {
            tracing::trace!(error = %e, "pairing stream finish failed; peer may have closed");
        }
        drop(pairing_recv);

        match role {
            Role::Host => {
                let (control_send, mut control_recv) = conn.accept_bi().await?;
                read_preamble(&mut control_recv).await?;
                let mut input_recv = conn.accept_uni().await?;
                read_preamble(&mut input_recv).await?;
                Ok(Connection::new_host(
                    conn,
                    control_send,
                    control_recv,
                    input_recv,
                ))
            }
            Role::Client => {
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
        }
    }
}

/// Read and discard a stream preamble (see [`STREAM_PREAMBLE`]).
async fn read_preamble(recv: &mut quinn::RecvStream) -> Result<()> {
    let mut preamble = [0u8; STREAM_PREAMBLE_LEN];
    recv.read_exact(&mut preamble)
        .await
        .map_err(TransportError::from_read_exact)
}
