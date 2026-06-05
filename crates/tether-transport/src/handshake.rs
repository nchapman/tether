//! Typestate wrappers that make the host-side handshake ordering
//! contract uncompilable to violate.
//!
//! The raw [`ControlChannel`] trait methods [`recv_client_hello`] and
//! [`send_server_hello`] carry an unenforced ordering invariant: `recv`
//! must be called exactly once before `send` is called exactly once. A
//! double-`send` corrupts the wire (the client decodes the second
//! `ServerHello` body as the next `ControlMessage`).
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

use crate::{ControlChannel, Result};
use tether_protocol::control::{ClientHello, HandshakeFailure, ServerHello};

/// Initial state. Holds the channel; the only operation is to receive
/// the [`ClientHello`].
pub struct HostHandshake {
    channel: Arc<dyn ControlChannel>,
}

/// Post-`recv` state. The only operation is to send the [`ServerHello`].
///
/// The parsed [`ClientHello`] is returned separately from
/// [`HostHandshake::recv_client_hello`] so orchestration code can move
/// it into application-layer state (extension parsing, logging,
/// session state) without having to clone it out of the typestate.
pub struct ClientHelloReceived {
    channel: Arc<dyn ControlChannel>,
}

impl HostHandshake {
    pub fn new(channel: Arc<dyn ControlChannel>) -> Self {
        Self { channel }
    }

    /// Awaits the [`ClientHello`]. The returned [`ClientHelloReceived`] is the
    /// only path forward — there is no way to call `send_server_hello` on the
    /// trait without going through it.
    pub async fn recv_client_hello(self) -> Result<(ClientHello, ClientHelloReceived)> {
        let client_hello = self.channel.recv_client_hello().await?;
        let pending = ClientHelloReceived {
            channel: self.channel,
        };
        Ok((client_hello, pending))
    }
}

impl ClientHelloReceived {
    /// Sends the supplied [`ServerHello`]. Returns the channel so the caller
    /// can resume post-handshake control-message exchange.
    pub async fn send_server_hello(self, server: ServerHello) -> Result<Arc<dyn ControlChannel>> {
        self.channel.send_server_hello(server).await?;
        Ok(self.channel)
    }

    /// Sends a typed handshake rejection. Returns the channel so callers can
    /// close or drain deliberately if needed.
    pub async fn send_rejection(
        self,
        failure: HandshakeFailure,
    ) -> Result<Arc<dyn ControlChannel>> {
        self.channel
            .send_server_handshake_rejection(failure)
            .await?;
        Ok(self.channel)
    }
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use super::*;
    use crate::test_support::duplex_pair;
    use tether_protocol::control::{
        DisplayDescriptor, DisplayId, DisplayMode, InputCapabilities, NegotiatedVideo, PixelFormat,
        VideoColorSpec, VideoProfile, VideoStreamId,
    };

    fn hello() -> ClientHello {
        ClientHello {
            client_name: "t".into(),
            decode_profiles: vec![VideoProfile::H264_8BIT_420],
            initial_viewport: None,
            input_capabilities: InputCapabilities::default(),
            requested_features: Vec::new(),
        }
    }

    fn server() -> ServerHello {
        let mode = DisplayMode::new(1280, 720, 60_000);
        ServerHello {
            server_name: "s".into(),
            video: NegotiatedVideo {
                stream_id: VideoStreamId(0),
                display_id: DisplayId(0),
                profile: VideoProfile::H264_8BIT_420,
                pixel_format: PixelFormat::Nv12,
                color_space: VideoColorSpec::sdr_desktop(),
            },
            audio: None,
            displays: vec![DisplayDescriptor {
                id: DisplayId(0),
                name: "test".into(),
                scale_num: 1,
                scale_den: 1,
                primary: true,
                position: (0, 0),
                current_mode: mode,
                available_modes: vec![mode],
                can_set_mode: false,
            }],
            accepted_features: Vec::new(),
        }
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
            let _server = client.client_handshake(hello()).await.unwrap();
        };
        tokio::join!(host_fut, client_fut);
    }
}
