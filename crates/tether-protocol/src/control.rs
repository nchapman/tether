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

/// First message a client sends after the QUIC handshake completes.
///
/// **Version policy (v0):** the receiver compares
/// [`Self::protocol_version`] against the local [`crate::PROTOCOL_VERSION`]
/// and treats any mismatch as fatal — no negotiation, no minimum-version
/// fallback. The connection is closed with `Goodbye { reason: "protocol
/// version mismatch" }`. We can revisit once we have multiple shipped
/// versions in the wild.
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

/// Result of a successful three-way handshake probe. Computed from
/// `t0_client_send`, `t1_server_recv`, `t2_server_send`, `t3_client_recv`
/// using the standard NTP formula: `offset = ((t1 - t0) + (t2 - t3)) / 2`,
/// `rtt = (t3 - t0) - (t2 - t1)`.
///
/// Convention: `offset_nanos` is signed and reads as "remote clock minus
/// local clock". To translate a host (remote) timestamp into the client's
/// (local) clock: `client_equiv = host_time - offset_nanos`. Positive
/// offset means the remote was ahead at sample time. Cheap to clone;
/// passes around the connection's owned latency view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockSync {
    pub offset_nanos: i64,
    pub rtt_nanos: u64,
    /// Local-clock time at which this sync was sampled. Useful for
    /// detecting drift in long-running connections (re-probe on a timer
    /// and warn if the offset has moved much).
    pub sampled_at_local: MonoNanos,
}

impl ClockSync {
    /// Build a `ClockSync` from the four probe timestamps. `t0` and `t3`
    /// are in the local clock, `t1` and `t2` are in the remote clock.
    /// `rtt` saturates at zero in the (impossible-without-clock-jump)
    /// case where the remote claims it spent more time processing than
    /// the local observed in total.
    #[must_use]
    pub fn from_probe(
        t0_local_send: MonoNanos,
        t1_remote_recv: MonoNanos,
        t2_remote_send: MonoNanos,
        t3_local_recv: MonoNanos,
    ) -> Self {
        let t0 = i128::from(t0_local_send.0);
        let t1 = i128::from(t1_remote_recv.0);
        let t2 = i128::from(t2_remote_send.0);
        let t3 = i128::from(t3_local_recv.0);
        let offset = ((t1 - t0) + (t2 - t3)) / 2;
        let total = t3 - t0;
        let remote_processing = t2 - t1;
        let rtt = (total - remote_processing).max(0);
        Self {
            offset_nanos: i64::try_from(offset.clamp(
                i128::from(i64::MIN),
                i128::from(i64::MAX),
            ))
            .expect("clamped to i64 range"),
            rtt_nanos: u64::try_from(rtt.clamp(0, i128::from(u64::MAX)))
                .expect("clamped to u64 range"),
            sampled_at_local: t3_local_recv,
        }
    }

    /// Translate a timestamp from the remote clock into the local clock.
    /// Saturates at zero rather than overflowing if the offset would
    /// drive the result negative (only possible early in a connection
    /// before both clocks have advanced past zero).
    #[must_use]
    pub fn remote_to_local(&self, remote: MonoNanos) -> MonoNanos {
        let v = i128::from(remote.0) - i128::from(self.offset_nanos);
        MonoNanos(
            u64::try_from(v.clamp(0, i128::from(u64::MAX)))
                .expect("clamped to u64 range"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_sync_recovers_offset_and_rtt() {
        // Client send at t0=100; remote (host) clock is +400ns ahead;
        // one-way delay is 5ns; remote spends 10ns processing.
        let t0 = MonoNanos(100);
        let t1 = MonoNanos(505); // 100 + 5 (one-way) + 400 (offset)
        let t2 = MonoNanos(515); // t1 + 10 (processing)
        let t3 = MonoNanos(120); // t0 + 5 + 10 + 5
        let sync = ClockSync::from_probe(t0, t1, t2, t3);
        assert_eq!(sync.offset_nanos, 400);
        assert_eq!(sync.rtt_nanos, 10);
    }

    #[test]
    fn remote_to_local_subtracts_offset() {
        let sync = ClockSync {
            offset_nanos: 400,
            rtt_nanos: 10,
            sampled_at_local: MonoNanos(120),
        };
        assert_eq!(sync.remote_to_local(MonoNanos(1000)), MonoNanos(600));
    }

    #[test]
    fn remote_to_local_saturates_on_underflow() {
        let sync = ClockSync {
            offset_nanos: 1000,
            rtt_nanos: 10,
            sampled_at_local: MonoNanos(0),
        };
        assert_eq!(sync.remote_to_local(MonoNanos(100)), MonoNanos(0));
    }
}
