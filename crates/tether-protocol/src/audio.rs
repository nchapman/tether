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
//! ## Loss recovery (RED)
//!
//! Audio has no transport-level FEC (the video [`FrameFragmenter`] does),
//! and Opus in-band FEC doesn't fit our config (LBRR is SILK-only; our
//! CELT-only `RESTRICTED_LOWDELAY` mode — and even `OPUS_APPLICATION_AUDIO`
//! at 128 kbps fullband music — emits little). So each datagram carries a
//! small RFC-2198-style RED tail:
//! the *previous* frames' Opus payloads in [`AudioPacket::Opus::redundant`].
//! When a datagram is lost but a later one arrives, the client recovers the
//! missing frame from the redundant copy and decodes it in order — no PLC
//! click. The copies are tiny (~80 B/frame at the default 5 ms / 128 kbps),
//! so even a couple stay far under the per-datagram MTU. `redundant` is empty when
//! redundancy is disabled or at stream start; a client that ignores it
//! still decodes the primary `payload` exactly as before.
//!
//! [`FrameFragmenter`]: crate::video::FrameFragmenter
//!
//! ## Format negotiation
//!
//! Sample rate, channel count, and Opus stream-config bytes are carried
//! through the hello extensions map under key `tether.audio` —
//! reverse-DNS-style same as every other extension. A future revision
//! that wants a typed audio-config field on `ServerHelloV1` can promote
//! the extension to a typed addition in `ServerHelloV2`.

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::MonoNanos;

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
        /// the decoder needs. `Bytes` (refcounted) like the video wire,
        /// so the encoder's output rides through to the datagram with no
        /// copy. Serialized as a byte sequence (identical wire shape to
        /// `Vec<u8>`).
        payload: Bytes,
        /// RED loss-recovery tail: the Opus payloads of the frames
        /// immediately preceding this one, **newest-first** — so
        /// `redundant[k]` is the payload for `frame_seq - (k + 1)`. Empty
        /// when redundancy is off or at stream start. Lets the client
        /// recover an isolated lost datagram from a later one without a
        /// concealment click. Bounded by the datagram MTU on send and by
        /// [`crate::MAX_DATAGRAM_DECODE_BYTES`] on receive.
        redundant: Vec<Bytes>,
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
