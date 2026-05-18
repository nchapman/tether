//! QUIC transport for Tether.
//!
//! Provides a [`Server`] (host side) and [`Client`] (client side) that
//! together produce a [`Connection`] multiplexing four logical channels
//! over one QUIC connection:
//!
//! - **Datagrams** (unreliable, unordered) carrying
//!   [`tether_protocol::video::VideoPacket`] and
//!   [`tether_protocol::cursor::CursorPacket`], demuxed via the
//!   [`Datagram`] enum.
//! - A **bidirectional control stream** (reliable, ordered) for
//!   [`tether_protocol::control::ControlMessage`] —
//!   handshake, clock-sync probes, IDR requests, shutdown.
//! - A **unidirectional input stream** (reliable, ordered) for
//!   [`tether_protocol::input::InputEvent`] from client to host.
//!
//! **v0 security model:** the host generates a fresh self-signed cert at
//! startup; its SHA-256 fingerprint must be exchanged with the client out
//! of band. The client pins that fingerprint via
//! [`tls::PinnedCertVerifier`]. See [`tls`] for the rationale.

mod client;
mod connection;
mod server;
pub mod tls;

pub use client::Client;
pub use connection::{Connection, Datagram};
pub use server::Server;
pub use tls::CertFingerprint;

use tether_protocol::CodecError;

/// Hard cap on the size of a single framed message on the control or input
/// stream. Guards [`tether_protocol::decode`] against forged length prefixes
/// from a hostile peer (the QUIC datagram path is already capped by
/// [`tether_protocol::MAX_DATAGRAM_PAYLOAD`]).
pub const MAX_FRAMED_MESSAGE: usize = 64 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("quic connect: {0}")]
    Connect(#[from] quinn::ConnectError),
    #[error("quic connection: {0}")]
    Connection(#[from] quinn::ConnectionError),
    #[error("quic write: {0}")]
    Write(#[from] quinn::WriteError),
    #[error("quic read: {0}")]
    Read(#[from] quinn::ReadError),
    #[error("quic send datagram: {0}")]
    SendDatagram(#[from] quinn::SendDatagramError),
    #[error("rustls: {0}")]
    Rustls(#[from] rustls::Error),
    #[error("rcgen: {0}")]
    Rcgen(#[from] rcgen::Error),
    #[error("protocol codec: {0}")]
    Codec(#[from] CodecError),
    #[error("stream closed mid-message")]
    StreamClosed,
    #[error("frame exceeds maximum size: {size} bytes (max {max})")]
    FrameTooLarge { size: usize, max: usize },
    #[error("input stream not available on this side of the connection")]
    InputStreamWrongRole,
}

pub type Result<T> = std::result::Result<T, TransportError>;

impl TransportError {
    pub(crate) fn from_read_exact(e: quinn::ReadExactError) -> Self {
        match e {
            quinn::ReadExactError::FinishedEarly(_) => TransportError::StreamClosed,
            quinn::ReadExactError::ReadError(re) => TransportError::Read(re),
        }
    }
}
