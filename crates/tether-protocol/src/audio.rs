//! Audio datagram format — host → client.
//!
//! Tether ships *the wire shape* for audio in V1 so adding the Opus
//! pipeline later (capture, encode, decode, render) is a self-contained
//! change with no protocol bump. The implementation deliberately stays
//! out of scope here: no capture backend integration, no Opus codec
//! wiring, no client-side audio output device. The pipeline is its own
//! workstream once we want sound.
//!
//! ## Channel
//!
//! Audio rides an unreliable datagram channel parallel to the video
//! channel. Like video, packets share `stream_epoch` so a host audio
//! restart (sample-rate change, output device switch) invalidates older
//! packets cleanly. Unlike video there's no fragmentation today: Opus
//! frames at typical 60 ms-or-less packetisation fit well under the
//! 1200-byte budget at any reasonable bitrate.
//!
//! ## Format negotiation
//!
//! Sample rate, channel count, and Opus stream-config bytes are carried
//! through the hello extensions map under key `tether.audio` —
//! reverse-DNS-style same as every other extension. A future revision
//! that wants a typed audio-config field on `ServerHelloV1` can promote
//! the extension to a typed addition in `ServerHelloV2`.

use crate::MonoNanos;
use serde::{Deserialize, Serialize};

/// One host → client audio datagram.
///
/// `stream_epoch` matches the video epoch concept: bumped whenever the
/// host's audio encoder is restarted. `frame_seq` is the per-epoch
/// monotonic counter. `t_capture` lets the client compute glass-to-ear
/// latency the same way it does glass-to-glass for video.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioPacket {
    Opus {
        stream_epoch: u32,
        frame_seq: u32,
        t_capture: MonoNanos,
        /// Opus-encoded payload. For multistream Opus (surround), this
        /// is the concatenated multistream packet — the
        /// `tether.audio` extension carries the stream/coupled count
        /// the decoder needs.
        payload: Vec<u8>,
    },
}

/// Extension-map key for hello audio format negotiation. Value is a
/// bincode-encoded [`AudioConfig`].
pub const AUDIO_CONFIG_EXTENSION_KEY: &str = "tether.audio";

/// Host-advertised audio configuration. Lives in the hello extensions
/// map keyed by [`AUDIO_CONFIG_EXTENSION_KEY`] so it can be added today
/// without a typed hello field — the client decodes it if present,
/// ignores it if absent (no audio).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioConfig {
    /// 48000 is the only Opus-native rate worth shipping; the field
    /// exists for future-proofing (custom hardware, downsampled
    /// links).
    pub sample_rate_hz: u32,
    pub channels: u8,
    /// Multistream Opus: number of independent streams. Mono / stereo
    /// = 1; 5.1 = 4 streams + 1 coupled. See RFC 7845 §5.
    pub streams: u8,
    pub coupled_streams: u8,
    /// Opaque Opus stream-mapping table (channel index per output
    /// channel). Sized `channels` bytes. Receiver passes this verbatim
    /// to the multistream decoder constructor.
    pub channel_mapping: Vec<u8>,
}
