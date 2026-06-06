//! Client-side application handshake + negotiated session state.

use std::sync::Arc;

use tether_protocol::control::{
    is_known_bit_depth, ClientHello, ClockSync, ControlMessage, GoodbyeCode, InputCapabilities,
    ServerHandshake, ServerHello, VideoProfile,
};
use tether_protocol::MonoNanos;
use tether_transport::{ControlChannel, TransportError};
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct ClientSessionConfig {
    pub client_name: String,
    pub client_decode_profiles: Vec<VideoProfile>,
    pub viewport: Option<tether_protocol::control::Viewport>,
}

pub struct ClientSession {
    pub channel: Arc<dyn ControlChannel>,
    pub negotiated: VideoProfile,
    pub negotiated_video: tether_protocol::control::NegotiatedVideo,
    pub server_hello: ServerHello,
    pub clock_sync: ClockSync,
    pub client_decode_profiles: Vec<VideoProfile>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    #[error("transport: {0}")]
    Transport(#[from] TransportError),

    #[error(
        "host chose profile {chosen:?} which this client did not advertise \
         ({} entries in client_decode_profiles)", .advertised.len()
    )]
    ProfileNotAdvertised {
        chosen: VideoProfile,
        advertised: Vec<VideoProfile>,
    },

    #[error("host chose profile with unknown bit_depth {0}; expected one of {1:?}")]
    UnknownBitDepth(u8, &'static [u8]),

    #[error("host rejected handshake: {reason}")]
    HandshakeRejected { code: GoodbyeCode, reason: String },

    #[error("host ended session during clock probe: {reason}")]
    PeerGoodbyeDuringClockProbe { code: GoodbyeCode, reason: String },
}

impl ClientSession {
    pub async fn connect(
        channel: Arc<dyn ControlChannel>,
        cfg: ClientSessionConfig,
    ) -> Result<Self, ConnectError> {
        info!(
            client_decode_profiles = ?cfg.client_decode_profiles,
            "advertising video decode capabilities to host"
        );

        let advertised = cfg.client_decode_profiles.clone();
        let hello = ClientHello {
            client_name: cfg.client_name,
            decode_profiles: advertised.clone(),
            initial_viewport: cfg.viewport,
            input_capabilities: InputCapabilities::default(),
            requested_features: Vec::new(),
        };

        let server_hello = match channel.client_handshake(hello).await? {
            ServerHandshake::Accepted(server_hello) => server_hello,
            ServerHandshake::Rejected(failure) => {
                return Err(ConnectError::HandshakeRejected {
                    code: failure.code,
                    reason: failure.reason,
                });
            }
        };
        let negotiated = server_hello.video.profile;
        if !is_known_bit_depth(negotiated.bit_depth) {
            send_protocol_error_goodbye(
                channel.as_ref(),
                format!(
                    "host chose profile with unknown bit_depth {}; expected one of {:?}",
                    negotiated.bit_depth,
                    tether_protocol::control::KNOWN_BIT_DEPTHS
                ),
            )
            .await;
            return Err(ConnectError::UnknownBitDepth(
                negotiated.bit_depth,
                tether_protocol::control::KNOWN_BIT_DEPTHS,
            ));
        }
        if !advertised.contains(&negotiated) {
            send_protocol_error_goodbye(
                channel.as_ref(),
                format!("host chose unadvertised video profile {negotiated:?}"),
            )
            .await;
            return Err(ConnectError::ProfileNotAdvertised {
                chosen: negotiated,
                advertised,
            });
        }

        let clock_sync = run_clock_probe(channel.as_ref()).await?;
        info!(
            server = %server_hello.server_name,
            negotiated_codec = ?negotiated.codec,
            negotiated_chroma = ?negotiated.chroma,
            negotiated_bit_depth = negotiated.bit_depth,
            rtt_us = clock_sync.rtt_nanos / 1_000,
            clock_offset_us = clock_sync.offset_nanos / 1_000,
            "video profile negotiated; handshake complete"
        );

        if let Err(e) = channel.send_control(&ControlMessage::ForceIdr).await {
            warn!(error = ?e, "initial ForceIdr send failed; continuing anyway");
        }

        Ok(Self {
            channel,
            negotiated,
            negotiated_video: server_hello.video.clone(),
            server_hello,
            clock_sync,
            client_decode_profiles: advertised,
        })
    }
}

async fn send_protocol_error_goodbye(channel: &dyn ControlChannel, reason: String) {
    warn!(
        %reason,
        "host sent invalid handshake selection; sending Goodbye(ProtocolError)"
    );
    if let Err(e) = channel
        .send_control(&ControlMessage::Goodbye {
            reason,
            code: GoodbyeCode::ProtocolError,
        })
        .await
    {
        warn!(error = ?e, "failed to send Goodbye after invalid handshake selection");
    }
}

async fn run_clock_probe(channel: &dyn ControlChannel) -> Result<ClockSync, ConnectError> {
    let t0 = MonoNanos::now();
    channel
        .send_control(&ControlMessage::ClockProbeRequest { t0_sender: t0 })
        .await?;
    loop {
        let msg = channel.recv_control().await?;
        let t3 = MonoNanos::now();
        match msg {
            ControlMessage::ClockProbeResponse(probe) if probe.t0_sender == t0 => {
                return Ok(ClockSync::from_probe(
                    t0,
                    probe.t1_receiver_recv,
                    probe.t2_receiver_send,
                    t3,
                ));
            }
            ControlMessage::ClockProbeResponse(_) => {
                tracing::debug!("ignoring stale ClockProbeResponse during handshake probe");
            }
            other => {
                if let ControlMessage::Goodbye { reason, code } = other {
                    tracing::warn!(%reason, ?code, "peer ended session during clock probe");
                    return Err(ConnectError::PeerGoodbyeDuringClockProbe { reason, code });
                }
                tracing::debug!(?other, "ignoring non-clock message during handshake probe");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn unknown_bit_depth_is_rejected_via_helper() {
        assert!(tether_protocol::control::is_known_bit_depth(8));
        assert!(tether_protocol::control::is_known_bit_depth(10));
        assert!(!tether_protocol::control::is_known_bit_depth(12));
    }
}
