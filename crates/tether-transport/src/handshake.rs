//! Typestate wrappers that make the host-side handshake ordering
//! contract uncompilable to violate.
//!
//! The raw [`ControlChannel`] trait methods [`recv_client_hello`] and
//! [`send_server_hello`] carry an unenforced ordering invariant: `recv`
//! must be called exactly once before `send` is called exactly once. A
//! double-`send` corrupts the wire (the client decodes the second
//! `ServerHello` body as the next `ControlMessage`); a `send` without a
//! prior `recv` sends stamps based on uninitialized `MonoNanos` values.
//!
//! [`HostHandshake`] and [`ClientHelloReceived`] enforce that ordering
//! at the type level. Each state owns the `Arc<dyn ControlChannel>` by
//! value and consumes itself on transition, so:
//!
//! - `recv_client_hello` is only callable on `HostHandshake` — once.
//! - `send_server_hello` is only callable on `ClientHelloReceived` —
//!   once, only after `recv` has fired.
//! - The channel comes back out of `send_server_hello` so the caller
//!   (`tether_session::HostSession::accept`) can keep using it for the
//!   post-handshake control-message exchange.
//!
//! The trait methods remain public because the [`crate::Connection`]
//! impl is public; orchestration code should route through this module
//! and the runtime ordering test in `test_support` covers the
//! trait-direct path that a misbehaving impl could still hit.
//!
//! [`recv_client_hello`]: crate::ControlChannel::recv_client_hello
//! [`send_server_hello`]: crate::ControlChannel::send_server_hello

use std::sync::Arc;

use tether_protocol::control::{ClientHello, ServerHello};
use tether_protocol::MonoNanos;

use crate::{ControlChannel, Result};

/// Initial state. Holds the channel; the only operation is to receive
/// the [`ClientHello`].
pub struct HostHandshake {
    channel: Arc<dyn ControlChannel>,
}

/// Post-`recv` state. Carries the two clock-sync stamps captured
/// during `recv` (`client_t0` echoed from the wire, `t1_server_recv`
/// from the local clock at recv-return). The only operation is to
/// send the [`ServerHello`].
///
/// The parsed [`ClientHello`] is returned separately from
/// [`HostHandshake::recv_client_hello`] so orchestration code can move
/// it into application-layer state (extension parsing, logging,
/// session state) without having to clone it out of the typestate.
pub struct ClientHelloReceived {
    channel: Arc<dyn ControlChannel>,
    client_t0: MonoNanos,
    t1_server_recv: MonoNanos,
}

impl HostHandshake {
    pub fn new(channel: Arc<dyn ControlChannel>) -> Self {
        Self { channel }
    }

    /// Awaits the [`ClientHello`] and captures `t1_server_recv`. The
    /// returned [`ClientHelloReceived`] is the only path forward —
    /// there is no way to call `send_server_hello` on the trait without
    /// going through it.
    ///
    /// **Performance contract**: keep work between this returning and
    /// the follow-up [`ClientHelloReceived::send_server_hello`] under
    /// 1 ms. The duration biases the NTP-style clock-sync offset
    /// estimator by half its wall-clock length. See
    /// [`crate::ControlChannel::recv_client_hello`] for the detailed
    /// bound.
    pub async fn recv_client_hello(self) -> Result<(ClientHello, ClientHelloReceived)> {
        let (client_hello, t1_server_recv) = self.channel.recv_client_hello().await?;
        let client_t0 = match &client_hello {
            ClientHello::V1(body) => body.clock_probe_t0,
        };
        let pending = ClientHelloReceived {
            channel: self.channel,
            client_t0,
            t1_server_recv,
        };
        Ok((client_hello, pending))
    }
}

impl ClientHelloReceived {
    /// Stamps the supplied [`ServerHello`] with `t0_echo` /
    /// `t1_server_recv` / `t2_server_send` and writes it. Returns the
    /// channel so the caller can resume post-handshake control-message
    /// exchange.
    pub async fn send_server_hello(self, server: ServerHello) -> Result<Arc<dyn ControlChannel>> {
        self.channel
            .send_server_hello(server, self.client_t0, self.t1_server_recv)
            .await?;
        Ok(self.channel)
    }
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use super::*;
    use crate::test_support::duplex_pair;
    use std::collections::BTreeMap;
    use tether_protocol::control::{
        ChromaSubsampling, ClientHelloV1, CodecKind, ServerHelloV1, VideoColorSpec,
    };

    fn hello() -> ClientHello {
        ClientHello::V1(ClientHelloV1 {
            client_name: "t".into(),
            preferred_codecs: vec![CodecKind::H264],
            max_resolution: None,
            viewport: None,
            clock_probe_t0: MonoNanos::now(),
            extensions: BTreeMap::new(),
            resume_token: None,
        })
    }

    fn server() -> ServerHello {
        ServerHello::V1(ServerHelloV1 {
            server_name: "s".into(),
            chosen_codec: CodecKind::H264,
            chosen_chroma: ChromaSubsampling::Yuv420,
            color_space: VideoColorSpec::sdr_desktop(),
            resolution: (0, 0),
            clock_probe_t0_echo: MonoNanos::ZERO,
            t1_server_recv: MonoNanos::ZERO,
            t2_server_send: MonoNanos::ZERO,
            extensions: BTreeMap::new(),
            resume_token: None,
        })
    }

    #[tokio::test]
    async fn typestate_routes_recv_then_send_and_returns_channel() {
        let (host, client) = duplex_pair();
        let host_fut = async move {
            let (_hello, pending) = HostHandshake::new(host).recv_client_hello().await.unwrap();
            let channel = pending.send_server_hello(server()).await.unwrap();
            // Channel comes back usable for post-handshake exchange —
            // we don't send anything, but we should be able to await
            // a recv against the still-open stream without panicking.
            // Drop it to release the duplex end.
            drop(channel);
        };
        let client_fut = async move {
            let (_server, _sync) = client.client_handshake(hello()).await.unwrap();
        };
        tokio::join!(host_fut, client_fut);
    }
}
