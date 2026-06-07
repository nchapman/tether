//! QUIC transport for Tether.
//!
//! Provides a [`Server`] (host side) and [`Client`] (client side) that
//! together produce a [`Connection`] multiplexing four logical channels
//! over one QUIC connection:
//!
//! - **Datagrams** (unreliable, unordered) carrying
//!   [`tether_protocol::video::VideoPacket`],
//!   [`tether_protocol::cursor::HostCursorPacket`],
//!   [`tether_protocol::cursor::ClientCursorPacket`], and
//!   [`tether_protocol::audio::AudioPacket`], demuxed via the [`Datagram`]
//!   enum.
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

/// Per-connection cap on concurrent peer-initiated unidirectional streams.
/// Tether opens exactly one uni stream over a connection's life: the
/// client→host input event stream (video, IDR keyframes included, rides
/// datagrams — there is no reliable video stream). 2 covers the single
/// long-lived input stream with headroom for a transient reconnect overlap
/// while denying a malicious peer the ability to open thousands of streams
/// and pin the receive-side allocations.
pub const MAX_CONCURRENT_UNI_STREAMS: u32 = 2;

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
    /// True when a receive-side terminal error is the expected result of either
    /// side cleanly closing the QUIC connection during teardown.
    #[must_use]
    pub fn is_clean_shutdown_recv(&self) -> bool {
        fn is_clean_connection_error(e: &quinn::ConnectionError) -> bool {
            match e {
                quinn::ConnectionError::LocallyClosed => true,
                quinn::ConnectionError::ApplicationClosed(close) => close.error_code == 0u32.into(),
                _ => false,
            }
        }

        match self {
            TransportError::Connection(e) => is_clean_connection_error(e),
            TransportError::Read(quinn::ReadError::ConnectionLost(e)) => {
                is_clean_connection_error(e)
            }
            TransportError::StreamClosed => true,
            _ => false,
        }
    }

    /// True when a `send_datagram` failure affects only the current frame, so
    /// the caller should drop the frame and keep the session alive rather than
    /// tear it down.
    ///
    /// The motivating case is MTU shrinkage mid-stream: quinn's
    /// `max_datagram_size()` can drop (PLPMTUD black-hole detection, a Wi-Fi →
    /// cellular handoff, NAT rebinding to a lower-MTU path) *after* a frame has
    /// already been fragmented against a larger budget. The remaining shards
    /// then exceed the new limit — `FrameTooLarge` (our guard) or quinn's
    /// `TooLarge`. That is transient: the next frame re-fragments against the
    /// smaller MTU and recovers. Tearing the session down on either is a
    /// self-inflicted disconnect on a recoverable event. (A full datagram send
    /// buffer is *not* surfaced here: quinn's buffered `send_datagram` drops
    /// the oldest queued datagram instead of erroring.)
    ///
    /// Connection-level failures (peer gone, datagrams unsupported or disabled,
    /// connection lost) are *not* transient: continuing would spin uselessly,
    /// so the caller should end the loop.
    pub fn is_transient_send(&self) -> bool {
        matches!(
            self,
            TransportError::FrameTooLarge { .. }
                | TransportError::SendDatagram(quinn::SendDatagramError::TooLarge)
        )
    }

    pub(crate) fn from_read_exact(e: quinn::ReadExactError) -> Self {
        match e {
            quinn::ReadExactError::FinishedEarly(_) => TransportError::StreamClosed,
            quinn::ReadExactError::ReadError(re) => TransportError::Read(re),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_send_errors_drop_the_frame_not_the_session() {
        // Per-frame failures the send loop should survive by dropping the
        // current frame's remaining shards (MTU shrank mid-frame).
        assert!(TransportError::FrameTooLarge {
            size: 2000,
            max: 1200
        }
        .is_transient_send());
        assert!(
            TransportError::SendDatagram(quinn::SendDatagramError::TooLarge).is_transient_send()
        );
    }

    #[test]
    fn connection_level_send_errors_are_fatal() {
        // A peer that can't or won't receive datagrams will never accept video;
        // continuing would spin. These must end the send loop.
        assert!(
            !TransportError::SendDatagram(quinn::SendDatagramError::UnsupportedByPeer)
                .is_transient_send()
        );
        assert!(
            !TransportError::SendDatagram(quinn::SendDatagramError::Disabled).is_transient_send()
        );
        assert!(!TransportError::StreamClosed.is_transient_send());
    }

    #[test]
    fn clean_receive_shutdown_errors_are_not_warnings() {
        let app_closed = quinn::ConnectionError::ApplicationClosed(quinn::ApplicationClose {
            error_code: 0u32.into(),
            reason: bytes::Bytes::from_static(b"client closing"),
        });
        assert!(TransportError::Connection(app_closed.clone()).is_clean_shutdown_recv());
        assert!(
            TransportError::Read(quinn::ReadError::ConnectionLost(app_closed))
                .is_clean_shutdown_recv()
        );
        assert!(TransportError::StreamClosed.is_clean_shutdown_recv());
    }
}
