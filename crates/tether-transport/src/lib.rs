//! QUIC transport for Tether.
//!
//! Provides a [`Server`] (host side) and [`Client`] (client side) that
//! together produce a [`Connection`] multiplexing four logical channels
//! over one QUIC connection:
//!
//! - **Datagrams** (unreliable, unordered) carrying
//!   [`tether_protocol::video::VideoPacket`],
//!   [`tether_protocol::cursor::HostCursorPacket`], and
//!   [`tether_protocol::cursor::ClientCursorPacket`], demuxed via the
//!   [`Datagram`] enum.
//! - A **bidirectional control stream** (reliable, ordered) for
//!   [`tether_protocol::control::ControlMessage`] —
//!   handshake, clock-sync probes, IDR requests, shutdown.
//! - A **unidirectional input stream** (reliable, ordered) for
//!   [`tether_protocol::input::InputEvent`] from client to host.
//!
//! **Trust model:** the host generates a fresh self-signed cert at
//! startup; its SHA-256 fingerprint must be exchanged with the client out
//! of band. The client pins that fingerprint via
//! [`tls::PinnedCertVerifier`]. See [`tls`] for the rationale.

mod channel;
mod client;
mod connection;
mod handshake;
mod pending;
mod server;
pub mod tls;

#[cfg(feature = "test-support")]
#[doc(hidden)]
pub mod test_support;

pub use channel::{AbrSnapshot, ConnectionInfo, ControlChannel, InputChannel, VideoChannel};
pub use client::{Client, ServerAuth};
pub use connection::{Connection, Datagram};
pub use handshake::{ClientHelloReceived, HostHandshake};
pub use pending::PendingConnection;
pub use server::Server;
pub use tls::CertFingerprint;

use tether_protocol::CodecError;

/// Hard cap on the size of a single framed message on the control or input
/// stream. Guards [`tether_protocol::decode`] against forged length prefixes
/// from a hostile peer (the QUIC datagram path is already capped by
/// [`tether_protocol::MAX_DATAGRAM_PAYLOAD`]).
pub const MAX_FRAMED_MESSAGE: usize = 64 * 1024;

/// Hard cap on the size of a single video-keyframe message delivered on
/// the per-IDR unidirectional stream. Sized for a worst-case 4K HEVC
/// IDR plus modest headroom (~2 MiB total). Separate from
/// [`MAX_FRAMED_MESSAGE`] so the control-stream cap stays tight against
/// hostile peers while video streams can carry the legitimately-large
/// keyframe payload. Pre-allocated by `read_framed_with_max` on stream
/// arrival, so any growth here directly widens a peer-controlled
/// allocation; only bump if a real keyframe overruns.
pub const MAX_VIDEO_STREAM_MESSAGE: usize = 2 * 1024 * 1024;

/// Per-connection cap on concurrent host→client unidirectional streams.
/// The reliable-keyframe protocol needs O(1) streams alive at once
/// (the previous keyframe's stream finishes before the next is opened
/// today, but quinn permits in-flight overlap). 4 leaves headroom for
/// a brief overlap during a back-to-back IDR burst while denying a
/// malicious peer the ability to open thousands of streams and pin
/// the receive-side allocations.
pub const MAX_CONCURRENT_UNI_STREAMS: u32 = 4;

/// Cap on concurrent peer-initiated bidirectional streams. Tether opens
/// two client-initiated bidi streams over a connection's life: the
/// pairing stream at connect time, then the control stream in
/// `PendingConnection::into_connection`. The two transiently overlap —
/// `into_connection` calls `open_bi()` for control immediately after the
/// pairing stream's `finish()`/drop, before it has retired and freed its
/// flow-control slot — so the cap must be at least 2 or the control
/// `open_bi()` deadlocks waiting for a `MAX_STREAMS` credit the peer can't
/// grant yet. 2 still closes the low-grade resource-exhaustion surface
/// where a hostile peer could open the QUIC default of 100 half-open bidi
/// streams and pin connection-table state until the idle timeout fires.
pub const MAX_CONCURRENT_BIDI_STREAMS: u32 = 2;

// Deadlock mechanics are on the constant's doc comment above. Guarded at
// compile time because the only runtime coverage — the `client_pairing`
// success-path tests — catches a regression by hanging (30 s timeout)
// rather than failing cleanly.
const _: () = assert!(
    MAX_CONCURRENT_BIDI_STREAMS >= 2,
    "bidi cap must cover the pairing + control streams (see into_connection)"
);

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
    #[error("TLS keying-material exporter unavailable")]
    ExporterUnavailable,
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
