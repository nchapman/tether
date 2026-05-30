//! Pairing stream: first-contact device authentication.
//!
//! These messages ride a dedicated bidirectional QUIC stream that opens
//! *before* the control handshake (and before any session/input stream), so
//! the host can authenticate — or reject — a client before it can do anything
//! else. The crypto itself (symmetric SPAKE2 + channel-bound key confirmation)
//! lives in `tether-pairing`; these are just the envelopes that carry its
//! opaque byte payloads across the wire.
//!
//! # Flow
//!
//! ```text
//! client                                  host
//!   │  PairingClientMsg::Pair{spake2}  ──▶ │   (or ::Resume to reconnect)
//!   │ ◀── PairingServerMsg::PairChallenge  │   (or ::Authorized / ::Rejected)
//!   │     PairingConfirm{mac=C2H}      ──▶ │   host verifies, persists
//!   │ ◀── PairingResult::Confirmed{H2C}    │   (or ::Rejected)
//!   │     client verifies H2C, persists    │
//! ```
//!
//! A reconnect (`Resume`) short-circuits: the host already authenticated the
//! client by mutual-TLS cert fingerprint, so it just checks its allowlist and
//! answers `Authorized` or `Rejected` with no SPAKE2.
//!
//! # Versioning
//!
//! The enum-shaped messages add **variants** for new revisions, never appended
//! struct fields — same policy as [`crate::control`]. Bincode's varint
//! discriminator makes an unknown future variant fail decode cleanly on an
//! older peer. [`PairingConfirm`] is a plain struct (it carries only an opaque
//! MAC): bincode encodes it as just its field bytes with no discriminator, so
//! it is **not** wire-extensible — a future confirmation that needs more than a
//! MAC must be a new message type, not a field added here.

use serde::{Deserialize, Serialize};

/// The client's opening message on the pairing stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairingClientMsg {
    /// First contact. The user typed a PIN on the client; `spake2` is the
    /// client's symmetric-SPAKE2 message seeded with it. The host is expected
    /// to be in an open pairing window.
    Pair { spake2: Vec<u8> },
    /// Reconnect without a PIN. The client believes it is already paired (it
    /// has the host's cert in known-hosts). The host authorizes by checking
    /// the client's mutual-TLS cert fingerprint against its allowlist.
    Resume,
}

/// The host's reply to [`PairingClientMsg`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairingServerMsg {
    /// Host is in a pairing window: its SPAKE2 message, completing key
    /// agreement. The [`PairingConfirm`] / [`PairingResult`] exchange follows.
    PairChallenge { spake2: Vec<u8> },
    /// Host recognizes the client's cert (allowlist hit) — proceed straight to
    /// a session, no PIN.
    Authorized,
    /// Host refuses: not paired and no pairing window is open (or the window's
    /// PIN was already consumed). `reason` is for the client's UI; it is
    /// deliberately uniform and does not reveal *why* on the wire.
    Rejected { reason: String },
}

/// The client's key-confirmation MAC (the `C2H` direction), sent after a
/// [`PairingServerMsg::PairChallenge`]. Opaque to the wire layer; verified by
/// the host via `tether-pairing`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairingConfirm {
    pub mac: Vec<u8>,
}

/// The host's final word after verifying the client's [`PairingConfirm`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairingResult {
    /// Pairing succeeded: `mac` is the host's `H2C` confirmation for the client
    /// to verify before it persists the host's identity.
    Confirmed { mac: Vec<u8> },
    /// Pairing failed (confirmation mismatch). Uniform on the wire — the client
    /// can't tell a wrong PIN from a detected MITM. `reason` is UI text only.
    Rejected { reason: String },
}
