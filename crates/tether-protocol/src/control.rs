//! Control stream: handshake, clock sync, IDR requests, shutdown.

use crate::MonoNanos;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodecKind {
    H264,
    Hevc,
    Av1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChromaSubsampling {
    Yuv420,
    Yuv444,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpace {
    /// BT.709 limited range. The only color space supported in v0.
    Bt709Limited,
}

/// NTP-style three-way clock probe. The receiver records `t3` locally on
/// arrival; the offset between the two monotonic clocks is then
/// `((t1 - t0) + (t2 - t3)) / 2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockProbe {
    pub t0_sender: MonoNanos,
    pub t1_receiver_recv: MonoNanos,
    pub t2_receiver_send: MonoNanos,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientHello {
    pub protocol_version: u32,
    pub client_name: String,
    /// Codecs the client can decode, ordered by preference.
    pub preferred_codecs: Vec<CodecKind>,
    /// Client's maximum displayable resolution (host decides actual).
    pub max_resolution: Option<(u32, u32)>,
    /// First leg of the handshake clock probe — client's monotonic time at
    /// the moment of send.
    pub clock_probe_t0: MonoNanos,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerHello {
    pub protocol_version: u32,
    pub server_name: String,
    pub chosen_codec: CodecKind,
    pub chosen_chroma: ChromaSubsampling,
    pub color_space: ColorSpace,
    pub resolution: (u32, u32),
    /// Echo of the client's `clock_probe_t0` so the client can match
    /// the response to the request it sent.
    pub clock_probe_t0_echo: MonoNanos,
    /// Server-monotonic time when `ClientHello` was received.
    pub t1_server_recv: MonoNanos,
    /// Server-monotonic time when this message is sent.
    pub t2_server_send: MonoNanos,
}

/// Messages exchanged on the reliable control stream after handshake.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlMessage {
    /// Client requests an immediate IDR (e.g., after detected packet loss).
    ForceIdr,
    /// Periodic clock-sync re-probe (either side may initiate).
    ClockProbeRequest { t0_sender: MonoNanos },
    ClockProbeResponse(ClockProbe),
    Goodbye { reason: String },
}
