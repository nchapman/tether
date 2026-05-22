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
//! The handshake methods are split — [`ControlChannel::recv_client_hello`]
//! returns both the hello and the `t1_server_recv` timestamp; the
//! orchestration in `tether_session::HostSession::accept` then builds
//! the application half of the `ServerHello` and calls
//! [`ControlChannel::send_server_hello`], which stamps in `t0_echo` /
//! `t1` / `t2_server_send` immediately before serializing the wire
//! bytes. The clock-sync stamps are a wire-protocol concern (they
//! govern the `ClockSync::from_probe` estimator's accuracy); profile
//! negotiation is an application concern (`tether-probe` policy).
//! Keeping the split aligned to that boundary means the transport
//! crate doesn't see `VideoProfile` and the session crate doesn't
//! see `MonoNanos::now`.
//!
//! [`Connection`]: crate::Connection
//! [`test_support`]: crate::test_support

use std::net::SocketAddr;
use std::time::Duration;

use async_trait::async_trait;
use tether_protocol::control::{ClientHello, ClockSync, ControlMessage, ServerHello};
use tether_protocol::input::InputEvent;
use tether_protocol::video::VideoPacket;
use tether_protocol::MonoNanos;

use crate::{Datagram, Result};

/// The control stream + handshake. The handshake is split into two
/// halves so the application-layer orchestration in
/// `tether_session::HostSession` can pick a profile between
/// [`recv_client_hello`] and [`send_server_hello`] without forcing the
/// trait to know about `VideoProfile`.
///
/// **Ordering contract** on the host side: [`recv_client_hello`] must
/// be called exactly once before [`send_server_hello`] is called. The
/// trait does not enforce this — the orchestrator wraps the channel in
/// a typestate inside `tether_session` so callers can't misuse the
/// trait. Test doubles are free to assume the contract holds and
/// panic if it doesn't.
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
    /// after stamping its `clock_probe_t0` with a fresh local
    /// timestamp, awaits the [`ServerHello`], and returns both the
    /// parsed hello and a [`ClockSync`] computed from the four probe
    /// stamps. Single method because there's no application-level
    /// callback on this side — the client builds its hello up front.
    async fn client_handshake(&self, hello: ClientHello) -> Result<(ServerHello, ClockSync)>;

    /// Host side of the handshake, first half. Awaits the
    /// [`ClientHello`] and captures the receive timestamp
    /// `t1_server_recv`. The caller orchestrates between this and
    /// [`Self::send_server_hello`] — that's where profile negotiation
    /// runs.
    ///
    /// **Performance contract**: the duration between this returning
    /// and [`Self::send_server_hello`] being called biases the
    /// `ClockSync` offset estimator by exactly half its wall-clock
    /// length. Keep the work in between fast (sub-millisecond);
    /// cache anything expensive at process startup.
    async fn recv_client_hello(&self) -> Result<(ClientHello, MonoNanos)>;

    /// Host side of the handshake, second half. Stamps `t0_echo`
    /// (from `client_t0`), `t1_server_recv` (from `t1`), and
    /// `t2_server_send` (a fresh `MonoNanos::now()` right before the
    /// serialize+write) into the supplied [`ServerHello`] body, then
    /// sends it.
    ///
    /// Stamping happens inside the trait method so the wire-time
    /// stamp is as close to the actual write as possible. Doing it
    /// in the application layer would let mutex contention and
    /// serialization overhead between stamp-time and write-time
    /// bias the offset estimator.
    async fn send_server_hello(
        &self,
        server: ServerHello,
        client_t0: MonoNanos,
        t1_server_recv: MonoNanos,
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

/// Datagrams + per-IDR reliable unidirectional streams.
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

    /// Open a fresh QUIC uni stream and write one length-framed
    /// keyframe `VideoPacket`. Reliable, retransmitted on loss.
    async fn send_video_keyframe(&self, packet: &VideoPacket) -> Result<()>;

    /// Accept the next host-opened uni stream and read one
    /// length-framed keyframe.
    async fn accept_video_keyframe(&self) -> Result<VideoPacket>;
}

/// Observability handles. Not a channel role; lives separately so
/// channel traits stay focused on data movement.
pub trait ConnectionInfo: Send + Sync {
    fn rtt(&self) -> Duration;
    fn max_datagram_size(&self) -> Option<usize>;
    fn remote_address(&self) -> SocketAddr;
}
