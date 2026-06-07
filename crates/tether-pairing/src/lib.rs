//! Device-pairing cryptography and trust-store persistence for Tether.
//!
//! Tether is remote-control software, so the host must authenticate every
//! client before it can inject input. Pairing turns a short human-transcribed
//! PIN into two persisted long-term identities using a balanced PAKE
//! (symmetric **SPAKE2**, the magic-wormhole short-code primitive), with the
//! key-confirmation step bound to the TLS channel so a relay/MITM cannot sit
//! in the middle.
//!
//! ## What lives here vs. the transport
//!
//! The pairing protocol logic is deliberately network-I/O-free, mirroring
//! `tether-protocol`: it knows how to run the SPAKE2 exchange and build/verify
//! confirmation MACs given the TLS exporter + peer fingerprints as plain bytes,
//! but it never touches quinn or the wire. `tether-transport` owns the streams
//! and feeds those bytes in. This crate also owns the small JSON trust stores so
//! host, client, and shell share one persistence format.
//!
//! ## The exchange
//!
//! 1. Both sides [`PairingStart::new`] with the PIN, exchange the one SPAKE2
//!    message each carries, and call [`PairingStart::into_keyed`] to derive a
//!    shared [`KeyedPairing`].
//! 2. Each side [`KeyedPairing::make_confirmation`] for its own direction and
//!    [`KeyedPairing::verify_confirmation`] the peer's. The confirmation is
//!    **direction-tagged** (`H2C` / `C2H`) and keyed by an HKDF-derived
//!    subkey, not the raw SPAKE2 output. Symmetric SPAKE2 is reflection-prone
//!    (the key is message-order-independent), and the two defenses cover
//!    distinct attackers:
//!      - The **direction tag** stops a *passive relay* that forwards SPAKE2
//!        bytes unmodified but can't terminate TLS: it can't replay one party's
//!        confirmation as the other direction's reply, because the tag is the
//!        first thing fed into the MAC.
//!      - The **TLS exporter** (channel binding) in the [`Transcript`] stops a
//!        *TLS-terminating MITM*: two TLS legs yield two different exporters, so
//!        the confirmation MACs can't agree. This is the load-bearing defense
//!        once an attacker can terminate TLS (a reflection attacker becomes
//!        exactly that case).
//!
//!    The transcript also folds in both cert fingerprints (kills
//!    unknown-key-share) and the protocol version (downgrade defense).
//!
//! The SPAKE2 key itself already commits to the password, both identities, and
//! both wire messages (verified against the crate's `finish()` transcript), so
//! the confirmation layer adds exactly the bindings SPAKE2 omits.
//!
//! ## Swapping the PAKE primitive
//!
//! The PAKE is confined to two functions ([`PairingStart::new`] and
//! [`PairingStart::into_keyed`]); everything security-load-bearing against a
//! relay/MITM — the TLS-exporter channel binding, the direction-tagged
//! confirmation MAC, the [`Transcript`] — is PAKE-agnostic and unaffected by the
//! choice. **CPace** is the CFRG-preferred balanced PAKE and the natural
//! successor (no fixed M/N generators, cleaner security argument); we stay on
//! SPAKE2 for now because RustCrypto's `spake2` is the more mature/maintained
//! Rust option and the wire envelopes already carry the PAKE message as opaque
//! bytes. Revisit before GA: if an audited, maintained CPace crate exists, the
//! swap is a contained change to those two functions and the dependency, with no
//! wire-format or re-pairing impact.

mod store;

pub use store::{
    parse_tagged_fingerprint, tag_fingerprint, HostEntry, KnownHosts, PairedStore, PeerEntry,
};

use std::path::PathBuf;

/// The directory Tether keeps its persistent identity and trust files in:
/// the cert/key pair, the client's `known_hosts.json`, and the host's
/// `paired_clients.json`. `$TETHER_CERT_DIR` overrides it (for tests or for
/// sharing one identity between instances); otherwise it's `$HOME/.tether`
/// (`%USERPROFILE%\.tether` on Windows).
///
/// Shared by `tether-host`, `tether-client`, and the Tauri shell so all three
/// agree on where these files live — the shell reads/writes the same
/// `known_hosts.json` the client engine uses.
pub fn config_dir() -> std::io::Result<PathBuf> {
    if let Some(dir) = std::env::var_os("TETHER_CERT_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "neither $TETHER_CERT_DIR nor $HOME/$USERPROFILE is set; \
                 can't choose a config directory",
            )
        })?;
    Ok(PathBuf::from(home).join(".tether"))
}

/// The client's known-hosts file path under [`config_dir`].
pub fn known_hosts_path() -> std::io::Result<PathBuf> {
    Ok(config_dir()?.join("known_hosts.json"))
}

use hkdf::Hkdf;
use hmac::digest::KeyInit;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use spake2::{Ed25519Group, Identity, Password, Spake2};
use subtle::ConstantTimeEq;

/// SHA-256 of a peer's DER certificate. Structurally identical to
/// `tether_transport::CertFingerprint`; kept local so this crate stays free of
/// a transport dependency.
pub type Fingerprint = [u8; 32];

/// Fixed SPAKE2 identity string for both symmetric parties. Domain-separates
/// Tether pairing from any other SPAKE2 deployment that might reuse the PIN.
pub const APP_IDENTITY: &[u8] = b"tether-pairing-v1";

/// HKDF `info` separating the confirmation subkey from the raw SPAKE2 output.
pub const CONFIRM_INFO: &[u8] = b"tether pairing v1 confirm";

/// `label` passed to the TLS exporter on both ends. Must be byte-identical on
/// host and client or the channel binding (and thus pairing) fails.
pub const EXPORTER_LABEL: &[u8] = b"tether pairing exporter v1";

/// `context` passed to the TLS exporter on both ends. Empty is a valid RFC 5705
/// context; the label already domain-separates this export. Shared here so the
/// host and client can't drift (a mismatch silently breaks the channel binding,
/// which would read as a wrong-PIN failure).
pub const EXPORTER_CONTEXT: &[u8] = b"";

/// Length of the exporter value bound into the transcript. Fixed so the
/// confirmation transcript's field layout is unambiguous.
pub const EXPORTER_LEN: usize = 32;

/// Length of a confirmation MAC (HMAC-SHA256).
pub const CONFIRM_LEN: usize = 32;

/// Number of decimal digits in a pairing PIN. Eight digits is the
/// magic-wormhole-class short code: short enough to read aloud or type, with
/// the actual security resting on burn-on-attempt single-flight (the online
/// guessing limit), not the PIN's entropy.
pub const PIN_DIGITS: usize = 8;

/// Generate a fresh random pairing PIN: [`PIN_DIGITS`] decimal digits,
/// zero-padded, drawn from the OS CSPRNG. Uses rejection sampling so every
/// value in `0..10^PIN_DIGITS` is equally likely (a plain modulo would bias
/// the low digits). Panics only if the OS RNG is unavailable — an
/// unrecoverable platform fault, not a condition a caller can sensibly handle.
pub fn generate_pin() -> String {
    // 8 digits → [0, 100_000_000). Reject u32 draws at or above the largest
    // multiple of the range so the modulo is unbiased. 100_000_000 * 42 =
    // 4_200_000_000 ≤ u32::MAX (4_294_967_295), so the reject band is tiny
    // (~2.2%) and the loop almost always takes one iteration.
    const RANGE: u32 = 100_000_000; // 10^8
    const REJECT_AT: u32 = RANGE * (u32::MAX / RANGE); // 4_200_000_000
                                                       // Make the rejection-sampling invariants unmistakable: the threshold must
                                                       // be an exact multiple of RANGE (so `draw % RANGE` is unbiased over the
                                                       // accepted band) and must not exceed u32::MAX.
    const _: () = assert!(
        REJECT_AT % RANGE == 0,
        "REJECT_AT must be a multiple of RANGE"
    );
    // The largest multiple of RANGE that is ≤ u32::MAX, so the rejected band
    // (REJECT_AT..=u32::MAX) is narrower than one RANGE — the loop rejects at
    // most ~RANGE/2^32 of draws.
    const _: () = assert!(
        REJECT_AT > u32::MAX - RANGE,
        "REJECT_AT must be the largest RANGE multiple ≤ u32::MAX"
    );
    loop {
        let mut bytes = [0u8; 4];
        getrandom::fill(&mut bytes).expect("OS CSPRNG unavailable");
        let draw = u32::from_le_bytes(bytes);
        if draw < REJECT_AT {
            return format!("{:0width$}", draw % RANGE, width = PIN_DIGITS);
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PairingError {
    /// SPAKE2 rejected the peer's message (wrong length, bad side byte, or a
    /// point that fails curve validation). A wrong *PIN* does **not** land
    /// here — it yields a different key and is caught at confirmation.
    #[error("SPAKE2 key agreement failed")]
    Spake2Failed,
}

/// Which side of the exchange a confirmation MAC authenticates. The tag is
/// mixed into the MAC so a *passive relay* can't replay one party's
/// confirmation as the other direction's reply. (A TLS-terminating MITM is
/// caught by the exporter binding in the [`Transcript`], not by this tag.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    HostToClient,
    ClientToHost,
}

impl Direction {
    fn tag(self) -> &'static [u8; 3] {
        match self {
            Direction::HostToClient => b"H2C",
            Direction::ClientToHost => b"C2H",
        }
    }
}

/// Inputs bound into the confirmation MAC. All caller-supplied; this crate
/// does no I/O. `exporter` is the TLS channel binding (fixed [`EXPORTER_LEN`]
/// bytes, identical on both ends — the fixed-width type makes a wrong-length
/// exporter a compile error rather than a silent transcript mismatch); the
/// fingerprints are the two peers' cert fingerprints; `protocol_version` is
/// `tether_protocol::PROTOCOL_VERSION`.
pub struct Transcript<'a> {
    pub exporter: &'a [u8; EXPORTER_LEN],
    pub fp_host: &'a Fingerprint,
    pub fp_client: &'a Fingerprint,
    pub protocol_version: &'a str,
}

/// First pairing state: holds the in-progress SPAKE2 exchange and the one
/// outbound message the caller must send to the peer.
pub struct PairingStart {
    state: Spake2<Ed25519Group>,
    outbound: Vec<u8>,
}

impl PairingStart {
    /// Begin a symmetric SPAKE2 exchange seeded with `pin`. The returned value
    /// carries the outbound message; send [`outbound_msg`](Self::outbound_msg)
    /// to the peer, then feed the peer's message to
    /// [`into_keyed`](Self::into_keyed).
    pub fn new(pin: &[u8]) -> Self {
        let (state, outbound) = Spake2::<Ed25519Group>::start_symmetric(
            &Password::new(pin),
            &Identity::new(APP_IDENTITY),
        );
        Self { state, outbound }
    }

    /// The SPAKE2 message to transmit to the peer.
    pub fn outbound_msg(&self) -> &[u8] {
        &self.outbound
    }

    /// Consume the peer's SPAKE2 message and derive the shared confirmation
    /// key. Errors only on a malformed peer message (not on a wrong PIN — that
    /// surfaces as a confirmation mismatch).
    pub fn into_keyed(self, peer_msg: &[u8]) -> Result<KeyedPairing, PairingError> {
        let shared = self
            .state
            .finish(peer_msg)
            .map_err(|_| PairingError::Spake2Failed)?;
        Ok(KeyedPairing::from_shared(&shared))
    }
}

/// Keyed pairing state: the SPAKE2 exchange is complete and a confirmation
/// subkey is derived. Use it to produce this side's MAC and verify the peer's.
pub struct KeyedPairing {
    confirm_key: [u8; 32],
}

impl KeyedPairing {
    fn from_shared(shared: &[u8]) -> Self {
        // Derive a dedicated confirmation key rather than MAC under the raw
        // SPAKE2 output: keeps the MAC key domain-separated from any other use
        // of the shared secret. salt=None is fine — `shared` is already
        // high-entropy; `info` does the separation.
        let hk = Hkdf::<Sha256>::new(None, shared);
        let mut confirm_key = [0u8; 32];
        hk.expand(CONFIRM_INFO, &mut confirm_key)
            .expect("32 bytes is a valid HKDF-SHA256 output length");
        Self { confirm_key }
    }

    /// Compute this side's confirmation MAC over the bound transcript.
    pub fn make_confirmation(&self, dir: Direction, t: &Transcript<'_>) -> [u8; CONFIRM_LEN] {
        // Field layout: the leading fields are all fixed-width — tag(3) ‖
        // exporter(32) ‖ fp_host(32) ‖ fp_client(32) — and the trailing
        // variable-length version is length-prefixed, so the boundaries are
        // unambiguous and no two distinct transcripts can collide regardless
        // of what fields are added later.
        let mut mac = <Hmac<Sha256> as KeyInit>::new_from_slice(&self.confirm_key)
            .expect("HMAC accepts a key of any length");
        mac.update(dir.tag());
        mac.update(t.exporter);
        mac.update(t.fp_host);
        mac.update(t.fp_client);
        let ver = t.protocol_version.as_bytes();
        let ver_len = u32::try_from(ver.len()).expect("protocol version length fits in u32");
        mac.update(&ver_len.to_be_bytes());
        mac.update(ver);
        mac.finalize().into_bytes().into()
    }

    /// Constant-time-verify the peer's confirmation MAC for `dir`. Returns
    /// `false` on any mismatch (wrong PIN, channel-binding mismatch, version
    /// skew, or a reflected message). Callers must report a *uniform* failure
    /// to the network peer — don't leak which check failed.
    pub fn verify_confirmation(&self, dir: Direction, t: &Transcript<'_>, peer_mac: &[u8]) -> bool {
        let expected = self.make_confirmation(dir, t);
        // Length isn't secret, so an early length check leaks nothing; the
        // value comparison is constant-time.
        peer_mac.len() == expected.len() && expected.ct_eq(peer_mac).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VER: &str = "tether/1";

    fn sample_transcript<'a>(
        exporter: &'a [u8; EXPORTER_LEN],
        fp_host: &'a Fingerprint,
        fp_client: &'a Fingerprint,
    ) -> Transcript<'a> {
        Transcript {
            exporter,
            fp_host,
            fp_client,
            protocol_version: VER,
        }
    }

    /// Run the SPAKE2 exchange for two parties sharing `pin`/`peer_pin` and
    /// return both keyed states.
    fn exchange(pin: &[u8], peer_pin: &[u8]) -> (KeyedPairing, KeyedPairing) {
        let a = PairingStart::new(pin);
        let b = PairingStart::new(peer_pin);
        let a_msg = a.outbound_msg().to_vec();
        let b_msg = b.outbound_msg().to_vec();
        (
            a.into_keyed(&b_msg).expect("a finishes"),
            b.into_keyed(&a_msg).expect("b finishes"),
        )
    }

    #[test]
    fn matching_pin_confirmations_verify_both_directions() {
        let (host, client) = exchange(b"48271930", b"48271930");
        let exporter = [7u8; EXPORTER_LEN];
        let fp_host = [1u8; 32];
        let fp_client = [2u8; 32];
        let t = sample_transcript(&exporter, &fp_host, &fp_client);

        let mh = host.make_confirmation(Direction::HostToClient, &t);
        let mc = client.make_confirmation(Direction::ClientToHost, &t);

        assert!(client.verify_confirmation(Direction::HostToClient, &t, &mh));
        assert!(host.verify_confirmation(Direction::ClientToHost, &t, &mc));
    }

    #[test]
    fn wrong_pin_fails_confirmation() {
        let (host, client) = exchange(b"48271930", b"00000000");
        let exporter = [7u8; EXPORTER_LEN];
        let fp_host = [1u8; 32];
        let fp_client = [2u8; 32];
        let t = sample_transcript(&exporter, &fp_host, &fp_client);

        let mh = host.make_confirmation(Direction::HostToClient, &t);
        assert!(!client.verify_confirmation(Direction::HostToClient, &t, &mh));
    }

    #[test]
    fn reflected_message_fails_wrong_direction_tag() {
        // The anti-reflection property: a host-direction MAC must never verify
        // as a client-direction reply, even though both sides hold the key.
        let (host, client) = exchange(b"48271930", b"48271930");
        let exporter = [7u8; EXPORTER_LEN];
        let fp_host = [1u8; 32];
        let fp_client = [2u8; 32];
        let t = sample_transcript(&exporter, &fp_host, &fp_client);

        let mh = host.make_confirmation(Direction::HostToClient, &t);
        // Reflecting Mh back as if it were the C2H reply must be rejected.
        assert!(!client.verify_confirmation(Direction::ClientToHost, &t, &mh));
    }

    #[test]
    fn each_transcript_field_is_bound() {
        let (host, client) = exchange(b"48271930", b"48271930");
        let exporter = [7u8; EXPORTER_LEN];
        let fp_host = [1u8; 32];
        let fp_client = [2u8; 32];
        let base = sample_transcript(&exporter, &fp_host, &fp_client);
        let mh = host.make_confirmation(Direction::HostToClient, &base);

        // Flip the exporter (channel-binding / MITM defense).
        let other_exporter = [9u8; EXPORTER_LEN];
        let t_exp = sample_transcript(&other_exporter, &fp_host, &fp_client);
        assert!(!client.verify_confirmation(Direction::HostToClient, &t_exp, &mh));

        // Flip the host fingerprint (unknown-key-share defense).
        let other_fp = [0xAAu8; 32];
        let t_fph = sample_transcript(&exporter, &other_fp, &fp_client);
        assert!(!client.verify_confirmation(Direction::HostToClient, &t_fph, &mh));

        // Flip the client fingerprint.
        let t_fpc = sample_transcript(&exporter, &fp_host, &other_fp);
        assert!(!client.verify_confirmation(Direction::HostToClient, &t_fpc, &mh));

        // Flip the protocol version (downgrade defense).
        let t_ver = Transcript {
            exporter: &exporter,
            fp_host: &fp_host,
            fp_client: &fp_client,
            protocol_version: "tether/2",
        };
        assert!(!client.verify_confirmation(Direction::HostToClient, &t_ver, &mh));
    }

    #[test]
    fn generated_pin_is_eight_digits() {
        // Run many draws: every PIN must be exactly PIN_DIGITS ASCII digits and
        // parse as a value inside the range. This also exercises the rejection
        // loop across many iterations.
        for _ in 0..1000 {
            let pin = generate_pin();
            assert_eq!(pin.len(), PIN_DIGITS);
            assert!(
                pin.chars().all(|c| c.is_ascii_digit()),
                "pin {pin} not all digits"
            );
            let value: u32 = pin.parse().expect("pin parses as a number");
            assert!(value < 100_000_000);
        }
    }

    #[test]
    fn corrupt_peer_message_errors() {
        let a = PairingStart::new(b"48271930");
        // Garbage that isn't a valid SPAKE2 message.
        let err = a.into_keyed(&[0xFF; 8]);
        assert!(matches!(err, Err(PairingError::Spake2Failed)));
    }

    #[test]
    fn wrong_length_mac_rejected_without_panic() {
        // Only the verifier matters here; the peer's real MAC is irrelevant.
        let (_, client) = exchange(b"48271930", b"48271930");
        let exporter = [7u8; EXPORTER_LEN];
        let fp_host = [1u8; 32];
        let fp_client = [2u8; 32];
        let t = sample_transcript(&exporter, &fp_host, &fp_client);
        // A truncated / oversized MAC must be rejected, not panic in ct_eq.
        assert!(!client.verify_confirmation(Direction::HostToClient, &t, &[0u8; 8]));
        assert!(!client.verify_confirmation(Direction::HostToClient, &t, &[0u8; 64]));
    }
}
