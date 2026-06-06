//! Role-shaped traits that abstract the wire surface of a [`Connection`].
//!
//! Each trait mirrors how a specific consumer uses the connection:
//!
//! - [`ControlChannel`] — the application-layer handshake plus
//!   post-handshake `ControlMessage` exchange. Consumed by
//!   `tether_session::{HostSession, ClientSession}` and the per-task
//!   recv loops in both binaries.
//! - [`InputChannel`] — the unidirectional input stream
//!   (client → host).
//! - [`VideoChannel`] — datagrams (unreliable) plus per-IDR
//!   unidirectional streams (reliable).
//! - [`ConnectionInfo`] — observability handles (`rtt`,
//!   `remote_address`, `max_datagram_size`). Kept off the channel
//!   traits because it isn't a channel role — it's about the
//!   connection as a whole.
//!
//! Splitting into role traits — rather than one fat `Connection`
//! trait — means a test double can implement only the roles its test
//! cares about. The duplex-backed loopback fake in [`test_support`]
//! implements [`ControlChannel`] only; future tests that need video
//! or input will get their own focused fakes without dragging in
//! stubs for channels they don't exercise.
//!
//! The handshake methods are split so `tether_session::HostSession::accept`
//! can receive the client hello, run application negotiation, and then send
//! either an accepted `ServerHello` or a typed handshake rejection. Clock sync
//! is a separate post-handshake `ClockProbeRequest` / `ClockProbeResponse`
//! exchange on the control channel.
//!
//! [`Connection`]: crate::Connection
//! [`test_support`]: crate::test_support

use std::net::SocketAddr;
use std::time::Duration;

use async_trait::async_trait;
use tether_protocol::control::{ClientHello, ControlMessage, ServerHandshake, ServerHello};
use tether_protocol::input::InputEvent;

use crate::{Datagram, Result};

/// The control stream + handshake. The handshake is split into two
/// halves so the application-layer orchestration in
/// `tether_session::HostSession` can pick a profile between
/// [`recv_client_hello`] and [`send_server_hello`] without forcing the
/// trait to know about `VideoProfile`.
///
/// **Ordering contract** on the host side: [`recv_client_hello`] must
/// be called exactly once before [`send_server_hello`] is called. The
/// trait does not enforce this on its own — orchestration code must
/// route through [`crate::HostHandshake`] / [`crate::ClientHelloReceived`],
/// which encode the ordering as a typestate that makes the misuse
/// uncompilable. The trait methods stay public because the
/// [`crate::Connection`] impl is public; the runtime regression test
/// `double_send_server_hello_corrupts_the_stream` (in
/// `tether-session/tests/loopback.rs`) covers the trait-direct path
/// that a misbehaving impl could still hit.
///
/// [`recv_client_hello`]: ControlChannel::recv_client_hello
/// [`send_server_hello`]: ControlChannel::send_server_hello
#[async_trait]
pub trait ControlChannel: Send + Sync {
    /// Post-handshake control message send. Reliable, ordered.
    async fn send_control(&self, msg: &ControlMessage) -> Result<()>;

    /// Post-handshake control message receive. Reliable, ordered.
    async fn recv_control(&self) -> Result<ControlMessage>;

    /// Client side of the handshake. Sends the supplied [`ClientHello`]
    /// and awaits the [`ServerHello`]. Clock sync is an explicit
    /// post-handshake [`ControlMessage::ClockProbeRequest`] exchange.
    async fn client_handshake(&self, hello: ClientHello) -> Result<ServerHandshake>;

    /// Host side of the handshake, first half. Awaits the [`ClientHello`].
    /// The caller orchestrates between this and [`Self::send_server_hello`] or
    /// [`Self::send_server_handshake_rejection`] — that's where profile
    /// negotiation runs.
    async fn recv_client_hello(&self) -> Result<ClientHello>;

    /// Host side of the handshake, second half. Sends the supplied
    /// [`ServerHello`] after application-layer negotiation.
    async fn send_server_hello(&self, server: ServerHello) -> Result<()>;

    /// Host side of the handshake, rejection path. Sends a typed failure
    /// instead of a successful [`ServerHello`].
    async fn send_server_handshake_rejection(
        &self,
        failure: tether_protocol::control::HandshakeFailure,
    ) -> Result<()>;
}

/// Unidirectional input event stream. Reliable, ordered. The trait
/// is implemented on both sides of the connection but each instance
/// is uni-directional: a client-side `Connection` only supports
/// `send`, a host-side only `recv`. Mixing them returns
/// [`crate::TransportError::InputStreamWrongRole`].
#[async_trait]
pub trait InputChannel: Send + Sync {
    async fn send_input(&self, evt: &InputEvent) -> Result<()>;
    async fn recv_input(&self) -> Result<InputEvent>;
}

/// The unreliable datagram channel. All video — IDR keyframes and P-frames
/// alike — rides datagrams, sliced into `VideoPacket`s and FEC-protected; there
/// is no separate reliable keyframe stream (see `tether_protocol::video`).
///
/// [`send_datagram`] is **sync** by design: the underlying QUIC
/// datagram send is non-blocking, and the host's encode thread sends
/// fragments via `Handle::block_on`. Making it async would force an
/// executor round-trip per fragment for no benefit, and the
/// drop-on-overflow semantics are preserved (the sync `Err` from a
/// full queue is exactly the signal the caller wants).
///
/// [`send_datagram`]: VideoChannel::send_datagram
#[async_trait]
pub trait VideoChannel: Send + Sync {
    /// Drop-on-overflow datagram send. Sync; see trait-level docs.
    fn send_datagram(&self, d: &Datagram) -> Result<()>;

    /// Await the next datagram off the unreliable channel.
    async fn recv_datagram(&self) -> Result<Datagram>;
}

/// Observability handles. Not a channel role; lives separately so
/// channel traits stay focused on data movement.
pub trait ConnectionInfo: Send + Sync {
    fn rtt(&self) -> Duration;
    fn max_datagram_size(&self) -> Option<usize>;
    fn remote_address(&self) -> SocketAddr;
    /// Path-level transport stats relevant to adaptive bitrate. Kept
    /// behind a small `Copy` struct so the trait doesn't leak
    /// `quinn::ConnectionStats` — that would force every implementer
    /// (including the duplex test fake) to construct a quinn type
    /// they have no underlying state for. The fields are cumulative
    /// counters; the ABR controller takes deltas itself.
    fn quinn_stats(&self) -> AbrSnapshot;
}

/// Cumulative path-level transport counters for the ABR controller.
///
/// All fields are monotonic from connection start. The caller takes
/// deltas against the previous snapshot before feeding the controller.
#[derive(Debug, Clone, Copy, Default)]
pub struct AbrSnapshot {
    /// Quinn's current RTT estimate.
    pub rtt: Duration,
    /// Cumulative count of congestion events on the active path.
    pub congestion_events: u64,
    /// Cumulative count of packets quinn has marked lost on the
    /// active path.
    pub lost_packets: u64,
}
