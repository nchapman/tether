//! Control stream: handshake, clock sync, IDR requests, shutdown.
//!
//! # Versioning policy
//!
//! Both handshake messages ([`ClientHello`], [`ServerHello`]) are
//! `enum`-shaped wire envelopes whose variants tag a body struct
//! ([`ClientHelloV1`], [`ServerHelloV1`]). New protocol revisions land
//! as additional variants (`V2(ClientHelloV2)` etc.). Bincode encodes
//! enum variants with a varint discriminator, so a future V2 sent to
//! a V1-only receiver fails decode cleanly (`DecodeError::UnexpectedVariant`)
//! rather than silently misinterpreting the bytes. The transport
//! surfaces that as a `Result::Err` to the caller, which then closes
//! the connection — there's no in-band Goodbye round-trip on a
//! decode failure (we'd have to know what the peer can parse, which
//! is exactly what failed).
//!
//! Inside a body struct, **only new variants are wire-additive — never
//! appended fields**. Bincode is strictly positional within a struct
//! with no length prefix, so a V1 encoder that grew a new field at the
//! end would either leave trailing bytes attributed to the *next*
//! framed message or hit EOF in an older decoder. Adding anything to
//! `ClientHelloV1` after this point requires a `ClientHelloV2`.
//! The `#[serde(default)]` attributes on `extensions` and `resume_token`
//! are no-ops under bincode (positional, never sees a missing field) but
//! stay correct under serde formats with field-name tagging, in case
//! we ever export to JSON for telemetry.
//!
//! Forward-compatible *opt-in features* go through [`ClientHelloV1::extensions`]
//! / [`ServerHelloV1::extensions`] instead. The map shape lets either
//! side advertise a feature key (and its value payload) without a wire
//! revision; unknown keys are ignored. **Key naming convention:**
//! reverse-DNS-style `vendor.feature` (`tether.adaptive-bitrate`,
//! `tether.av1-preferred`) so first-party and future third-party
//! extensions don't collide.
//!
//! [`ControlMessage`] is closed: new variants on it are NOT
//! wire-additive (the discriminator collides with future codepoints
//! the peer might already understand differently). Adding a new
//! `ControlMessage` variant requires landing a `ClientHelloV2`
//! alongside it.
//!
//! # The `Extension` escape
//!
//! [`ControlMessage::Extension`] exists so that future features which
//! need a new control message *do not* force a `ClientHelloV2`. A
//! sender publishes a reverse-DNS-keyed payload; receivers that
//! don't recognise the key log and drop it without erroring. Typical
//! lifecycle of a new control feature:
//!
//! 1. Ship the feature as `ControlMessage::Extension { key:
//!    "tether.feature-x", payload: <bincode-encoded body> }`.
//! 2. Soak it until the shape stabilises across deployments.
//! 3. When promoting to a typed variant becomes worthwhile (compile-
//!    time field checking, no String overhead per message), land it
//!    as part of the next `ClientHelloV{N+1}` bump.
//!
//! Receivers MUST log unknown extension keys at `debug` so an
//! operator can see "peer is speaking a feature this build doesn't
//! support yet."

use std::collections::BTreeMap;

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

/// Describes one host display, carried in
/// [`ControlMessage::DisplayList`]. Today the host always sends a
/// one-element list (single-monitor); the field shape is here so adding
/// multi-monitor support later is purely additive — no new
/// `ControlMessage` variant, no V2 hello bump. `refresh_mhz` and the
/// `scale_*` pair also cover what would otherwise need their own
/// extension keys (display geometry / HiDPI hints).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisplayDescriptor {
    /// Host-assigned display id. Stable for the lifetime of a session;
    /// matches the `display: u8` on `VideoPacket` / cursor packets.
    pub id: u8,
    /// Human-readable name as the host's compositor reports it
    /// (`HDMI-A-1`, `DP-3`, `Built-in Retina Display`, etc.). May be
    /// empty if the host's capture backend doesn't expose names.
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Refresh rate in millihertz so 60 Hz = 60_000 and 59.94 Hz =
    /// 59_940 round-trip without floating point.
    pub refresh_mhz: u32,
    /// Logical-to-physical scale as a rational `num / den`. `(1, 1)`
    /// for a 1× display, `(2, 1)` for a Retina-style 2×, `(3, 2)` for
    /// 150% Windows scaling. Rational so common factors round-trip
    /// without `f32` precision loss; the client computes the final
    /// scale factor as needed.
    pub scale_num: u16,
    pub scale_den: u16,
    pub primary: bool,
    /// Origin of this display in the virtual desktop, in physical
    /// pixels. Lets the client place multi-display windows correctly
    /// without re-deriving the topology.
    pub position: (i32, i32),
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

// --- Versioned handshake envelopes --------------------------------------

/// Wire envelope for the client's opening message.
///
/// New protocol revisions add additional variants alongside `V1`. See
/// the module-level docs for the full versioning policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClientHello {
    V1(ClientHelloV1),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientHelloV1 {
    pub client_name: String,
    /// Codecs the client can decode, ordered by preference.
    pub preferred_codecs: Vec<CodecKind>,
    /// Client's maximum displayable resolution (host decides actual).
    pub max_resolution: Option<(u32, u32)>,
    /// First leg of the handshake clock probe — client's monotonic time at
    /// the moment of send.
    pub clock_probe_t0: MonoNanos,
    /// Opt-in extension envelope for forward-compatible feature flags
    /// (AV1 advertise, adaptive bitrate hint, future channels). Both
    /// sides ignore keys they don't recognise — adding a feature here
    /// does not require a protocol revision.
    #[serde(default)]
    pub extensions: BTreeMap<String, Vec<u8>>,
    /// If the client is attempting to resume a previous session, the
    /// token the host issued in that session's [`ServerHelloV1::resume_token`].
    /// `None` on a fresh session. The host may accept (preserving
    /// `stream_epoch` so the renderer can continue without a black
    /// frame) or reject (issue a fresh epoch); semantics are TBD. The
    /// protocol shape is here now so adding the implementation later
    /// doesn't require a wire bump.
    #[serde(default)]
    pub resume_token: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServerHello {
    V1(ServerHelloV1),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerHelloV1 {
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
    /// Opt-in extension envelope — same purpose as
    /// [`ClientHelloV1::extensions`].
    #[serde(default)]
    pub extensions: BTreeMap<String, Vec<u8>>,
    /// Opaque token the client can stash and present in a future
    /// [`ClientHelloV1::resume_token`] to attempt session resume.
    /// `None` if the host doesn't (yet) implement resume.
    #[serde(default)]
    pub resume_token: Option<Vec<u8>>,
}

// --- Goodbye ------------------------------------------------------------

/// Machine-readable reason for a [`ControlMessage::Goodbye`]. Lets the
/// peer distinguish "host shutting down cleanly" from "protocol
/// violation" without parsing the human-readable `reason` string.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoodbyeCode {
    /// Local side initiated a clean shutdown (user quit, system
    /// suspend, etc.). The peer should not retry.
    Clean,
    /// Local side observed a protocol violation in the peer's
    /// messages. Retrying with the same binary is unlikely to help.
    ProtocolError,
    /// Handshake's variant tag didn't decode, i.e. the peer is
    /// speaking a newer protocol revision than this build knows.
    UnsupportedVersion,
    /// Catch-all for internal errors not otherwise classified. The
    /// peer may retry; a transient failure here doesn't preclude
    /// reconnect.
    InternalError,
}

/// Messages exchanged on the reliable control stream after handshake.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlMessage {
    /// Client requests an immediate IDR (e.g., after detected packet loss).
    ForceIdr,
    /// Periodic clock-sync re-probe (either side may initiate).
    ClockProbeRequest { t0_sender: MonoNanos },
    ClockProbeResponse(ClockProbe),
    Goodbye {
        reason: String,
        code: GoodbyeCode,
    },
    /// Escape hatch for features that don't yet warrant a typed
    /// variant. Payload is opaque to the protocol — the convention
    /// is reverse-DNS-style keys (`tether.clipboard`,
    /// `tether.gamepad-rumble`) with a bincode-encoded body. Unknown
    /// keys are logged + dropped. See the module-level doc.
    Extension {
        key: String,
        payload: Vec<u8>,
    },
    /// Host → client. Cursor sprite (RGBA pixels). Routed on the
    /// reliable control stream rather than the cursor datagram channel
    /// because a 64×64 RGBA sprite exceeds the 1200-byte datagram
    /// budget and reassembly would be more complex than the win.
    /// Clients cache shapes by `id` and switch via [`Self::CursorUseShape`].
    CursorShape {
        id: u64,
        hotspot: (u16, u16),
        width: u16,
        height: u16,
        format: crate::cursor::CursorPixelFormat,
        pixels: Vec<u8>,
    },
    /// Host → client. Activate a previously-sent [`Self::CursorShape`]
    /// by id. Separate from `CursorShape` so the host can switch
    /// between cached cursors without re-sending the pixels.
    CursorUseShape { id: u64 },
    /// Host → client. Full display topology. Sent post-handshake and
    /// again on hotplug (display added/removed) — the client treats
    /// each message as the authoritative replacement. Single-monitor
    /// hosts send a one-element list; the field shape supports
    /// multi-monitor without a V2.
    DisplayList { displays: Vec<DisplayDescriptor> },
    /// Client → host. Subscribe to a subset of host displays. Empty
    /// vec means "the primary" (today, the only one). Sent on the
    /// client's user-driven display switch; host stops emitting video
    /// for displays not in the set.
    SetActiveDisplays { displays: Vec<u8> },
    /// Client → host. Sent once after the client has finished building
    /// its decoders, so the host doesn't start blasting video before
    /// the receive side is ready. Booleans indicate which streams the
    /// client is prepared to consume; `audio` is reserved for the
    /// future Opus pipeline (always `false` from clients today).
    StreamReady { video: bool, audio: bool },
    /// Client → host. Pause emission for the given display (e.g.
    /// window minimised). Host is free to stop encoding entirely for
    /// that display to save power.
    StreamPause { display: u8 },
    /// Client → host. Resume emission for the given display. Pairs
    /// with [`Self::StreamPause`]; host emits a fresh IDR before any
    /// subsequent P-frames so the client doesn't render a half-decoded
    /// stream.
    StreamResume { display: u8 },
}

// --- ClockSync ----------------------------------------------------------

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
