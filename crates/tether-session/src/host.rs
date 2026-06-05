//! Host-side application handshake + per-session lifecycle state.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use tether_protocol::audio::AudioConfig;
use tether_protocol::control::{
    is_known_bit_depth, ChromaSubsampling, ClientHello, DisplayDescriptor, DisplayId, GoodbyeCode,
    HandshakeFailure, NegotiatedVideo, PixelFormat, ServerHello, VideoColorSpec, VideoProfile,
    VideoStreamId,
};
use tether_transport::{ControlChannel, HostHandshake, TransportError};
use tracing::{info, warn};

use crate::idr::IdrSignal;

#[derive(Debug, Clone)]
pub struct HostSessionConfig {
    pub server_name: String,
    pub audio_config: Option<AudioConfig>,
    pub displays: Vec<DisplayDescriptor>,
}

pub struct HostSession {
    pub channel: Arc<dyn ControlChannel>,
    pub negotiated: VideoProfile,
    pub negotiated_video: NegotiatedVideo,
    pub client_hello: ClientHello,
    pub client_decode_profiles: Vec<VideoProfile>,
    pub idr_signal: IdrSignal,
    pub stream_ready: Arc<AtomicBool>,
}

#[derive(Debug, thiserror::Error)]
pub enum AcceptError {
    #[error("transport: {0}")]
    Transport(#[from] TransportError),

    #[error(
        "no video profile intersects host encode + client decode capabilities \
         (client: {client:?})"
    )]
    NoProfileIntersection { client: Vec<VideoProfile> },
}

impl HostSession {
    pub async fn accept<S>(
        channel: Arc<dyn ControlChannel>,
        cfg: HostSessionConfig,
        selector: S,
    ) -> Result<Self, AcceptError>
    where
        S: FnOnce(&[VideoProfile]) -> Option<VideoProfile>,
    {
        let (client_hello, pending) = HostHandshake::new(channel).recv_client_hello().await?;
        let client_decode_profiles = parse_client_decode_profiles(&client_hello);
        info!(
            client_decode_profiles = ?client_decode_profiles,
            "client video decode capabilities"
        );

        let Some(chosen_profile) = selector(&client_decode_profiles) else {
            warn!(
                client_decode_profiles = ?client_decode_profiles,
                "no video profile intersects host encode + client decode capabilities; ending session"
            );
            let failure = HandshakeFailure {
                reason: "host and client video capabilities do not intersect".to_string(),
                code: GoodbyeCode::InternalError,
            };
            let _ = pending.send_rejection(failure).await?;
            return Err(AcceptError::NoProfileIntersection {
                client: client_decode_profiles,
            });
        };

        let server_hello = build_server_hello(&cfg, chosen_profile);
        let negotiated_video = server_hello.video.clone();
        let channel = pending.send_server_hello(server_hello).await?;

        info!(
            client = %client_hello.client_name,
            negotiated_codec = ?chosen_profile.codec,
            negotiated_chroma = ?chosen_profile.chroma,
            chosen_bit_depth = chosen_profile.bit_depth,
            "video profile negotiated; handshake complete"
        );

        Ok(Self {
            channel,
            negotiated: chosen_profile,
            negotiated_video,
            client_hello,
            client_decode_profiles,
            idr_signal: IdrSignal::new(),
            stream_ready: Arc::new(AtomicBool::new(false)),
        })
    }
}

fn parse_client_decode_profiles(body: &ClientHello) -> Vec<VideoProfile> {
    let total = body.decode_profiles.len();
    let filtered: Vec<VideoProfile> = body
        .decode_profiles
        .iter()
        .copied()
        .filter(|p| is_known_bit_depth(p.bit_depth))
        .collect();
    if filtered.len() != total {
        warn!(
            dropped = total - filtered.len(),
            kept = filtered.len(),
            "client advertised profiles with unknown bit_depth values; dropped them"
        );
    }
    filtered
}

fn build_server_hello(cfg: &HostSessionConfig, profile: VideoProfile) -> ServerHello {
    ServerHello {
        server_name: cfg.server_name.clone(),
        video: NegotiatedVideo {
            stream_id: VideoStreamId(0),
            display_id: cfg
                .displays
                .first()
                .map_or(DisplayId(0), |display| display.id),
            profile,
            pixel_format: pixel_format_for(profile),
            color_space: VideoColorSpec::sdr_desktop(),
        },
        audio: cfg.audio_config.clone(),
        displays: cfg.displays.clone(),
        accepted_features: Vec::new(),
    }
}

fn pixel_format_for(profile: VideoProfile) -> PixelFormat {
    match (profile.chroma, profile.bit_depth) {
        (ChromaSubsampling::Yuv444, 10) => PixelFormat::P410,
        (ChromaSubsampling::Yuv444, _) => PixelFormat::Yuv444p,
        (ChromaSubsampling::Yuv420, 10) => PixelFormat::P010,
        _ => PixelFormat::Nv12,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tether_protocol::control::{DisplayMode, InputCapabilities};

    fn display() -> DisplayDescriptor {
        let mode = DisplayMode::new(1280, 720, 60_000);
        DisplayDescriptor {
            id: DisplayId(0),
            name: "test".into(),
            scale_num: 1,
            scale_den: 1,
            primary: true,
            position: (0, 0),
            current_mode: mode,
            available_modes: vec![mode],
            can_set_mode: false,
        }
    }

    fn hello(profiles: Vec<VideoProfile>) -> ClientHello {
        ClientHello {
            client_name: "test".into(),
            decode_profiles: profiles,
            initial_viewport: None,
            input_capabilities: InputCapabilities::default(),
            requested_features: Vec::new(),
        }
    }

    #[test]
    fn unknown_depths_are_filtered() {
        let profiles = parse_client_decode_profiles(&hello(vec![
            VideoProfile::H264_8BIT_420,
            VideoProfile {
                codec: tether_protocol::control::CodecKind::Hevc,
                chroma: ChromaSubsampling::Yuv420,
                bit_depth: 12,
            },
        ]));
        assert_eq!(profiles, vec![VideoProfile::H264_8BIT_420]);
    }

    #[test]
    fn build_server_hello_carries_typed_video_and_displays() {
        let cfg = HostSessionConfig {
            server_name: "host".into(),
            audio_config: None,
            displays: vec![display()],
        };
        let hello = build_server_hello(&cfg, VideoProfile::HEVC_10BIT_420);
        assert_eq!(hello.video.profile, VideoProfile::HEVC_10BIT_420);
        assert_eq!(hello.video.pixel_format, PixelFormat::P010);
        assert_eq!(hello.displays.len(), 1);
    }
}
