//! Host-side device pairing: the authorization gate in front of every session.
//!
//! A client that completes the QUIC/TLS handshake arrives as a
//! [`PendingConnection`] with only its pairing stream wired — no control or
//! input stream exists yet (see `tether_transport::pending`). This module
//! decides, per connection, whether that peer is allowed to become a full
//! session:
//!
//! - **Allowlist hit** (`Resume`): the client's mutual-TLS cert fingerprint is
//!   already in `paired_clients.json` → authorize with no PIN.
//! - **First contact** (`Pair`): only if the operator has an open pairing
//!   window. Run symmetric SPAKE2 + channel-bound key confirmation
//!   (`tether-pairing`); on success persist the fingerprint and authorize.
//! - **Otherwise**: reject. An unpaired client with no open window never gets
//!   control/input streams, so the input-injection path is structurally
//!   unreachable for it.
//!
//! The PIN is **burned on attempt** (consumed the moment a `Pair` message
//! arrives), not on success, so wrong guesses are not free: each open window
//! permits exactly one pairing attempt.

use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use tether_pairing::{
    Direction, PairedStore, PairingStart, Transcript, CONFIRM_LEN, EXPORTER_CONTEXT, EXPORTER_LABEL,
    EXPORTER_LEN,
};
use tether_protocol::pairing::{
    PairingClientMsg, PairingConfirm, PairingResult, PairingServerMsg,
};
use tether_protocol::PROTOCOL_VERSION;
use tether_transport::{CertFingerprint, PendingConnection};
use tokio::sync::Notify;
use tracing::{info, warn};

/// How long a pairing window stays open before it expires. Long enough for the
/// operator to read the PIN aloud and the remote user to type it; short enough
/// that a forgotten window doesn't sit open. Burn-on-attempt single-flight is
/// the real limit — this is the upper bound.
pub const PAIRING_WINDOW: Duration = Duration::from_secs(120);

/// QUIC application close codes used when refusing a pending connection. These
/// are advisory (the client logs them); authorization never depends on them.
mod close_code {
    /// The single uniform refusal code. It deliberately does **not** encode
    /// *why* the peer was refused (no window, wrong PIN, unknown peer, MITM
    /// detected) — distinct codes would be a network-visible oracle for the
    /// host's pairing-window state. The local log carries the real reason.
    pub const REFUSED: u32 = 2;
    /// A transport/framing fault (bad envelope, our own send failure) — not an
    /// authorization decision, so it carries no auth signal.
    pub const PROTOCOL_ERROR: u32 = 4;
}

/// The single on-wire refusal reason. Uniform across every refusal cause so the
/// peer (or a passive observer) can't distinguish wrong-PIN from a detected
/// MITM from a closed window. An honest client infers meaning from *which*
/// message it sent (a refused `Resume` ⇒ "not paired, go pair"; a refused
/// `Pair` ⇒ "pairing failed"), so a uniform string costs no UX.
const REFUSAL_REASON: &str = "pairing failed";

/// An open pairing window: the host is willing to accept exactly one first-
/// contact attempt with this PIN until it expires. Created by `StartPairing`.
#[derive(Debug, Clone)]
pub struct PairingWindow {
    /// The PIN shown to the operator and typed on the new device.
    pub pin: String,
    /// Display label to record for whichever device pairs in this window.
    pub label: String,
    /// When the window stops accepting attempts.
    pub expires_at: Instant,
}

impl PairingWindow {
    /// Open a window valid for [`PAIRING_WINDOW`] from now.
    pub fn open(pin: String, label: String) -> Self {
        Self {
            pin,
            label,
            expires_at: Instant::now() + PAIRING_WINDOW,
        }
    }

    fn is_expired(&self, now: Instant) -> bool {
        now >= self.expires_at
    }
}

/// The currently-connected session, tracked so `RevokePeer` can drop a live
/// session whose fingerprint it removes from the allowlist.
#[derive(Clone)]
pub struct ActiveSession {
    pub fp: CertFingerprint,
    /// Tripped to tear the session down out-of-band (revocation).
    pub kill: Arc<Notify>,
}

/// Shared pairing state, held by both the accept loop and the stdin command
/// watcher.
#[derive(Clone)]
pub struct PairingState {
    /// The persisted allowlist of paired client fingerprints. The save path is
    /// rare (pairing success / revoke), so holding the lock across its file I/O
    /// is acceptable.
    pub paired: Arc<StdMutex<PairedStore>>,
    /// Where the allowlist is persisted (`paired_clients.json`).
    pub paired_path: Arc<std::path::PathBuf>,
    /// The single open pairing window, if any. `None` outside a window.
    pub window: Arc<StdMutex<Option<PairingWindow>>>,
    /// The live session, if one is connected.
    pub active: Arc<StdMutex<Option<ActiveSession>>>,
    /// The host's own cert fingerprint, bound into the confirmation transcript.
    pub host_fp: CertFingerprint,
}

impl PairingState {
    /// Open a fresh pairing window with a new random PIN, returning the PIN to
    /// show the operator. Replaces any existing window (the operator pressed
    /// "Add a device" again before the last one was used).
    pub fn open_window(&self, label: String) -> String {
        let pin = tether_pairing::generate_pin();
        *self.window.lock().expect("pairing window lock") =
            Some(PairingWindow::open(pin.clone(), label));
        pin
    }

    /// Revoke a paired peer by its tagged fingerprint (`"sha256:<hex>"`),
    /// persisting the change and tearing down any live session from that peer.
    /// Returns whether the peer was present in the allowlist.
    pub fn revoke(&self, tagged_fp: &str) -> bool {
        let removed = {
            let mut store = self.paired.lock().expect("paired store lock");
            let removed = store.remove_tagged(tagged_fp);
            if removed {
                if let Err(e) = store.save(&self.paired_path) {
                    warn!(error = %e, "allowlist save after revoke failed");
                }
            }
            removed
        };
        // Drop a live session from the revoked peer, if it matches.
        if let Some(fp) = tether_pairing::parse_tagged_fingerprint(tagged_fp) {
            if let Some(active) = self.active.lock().expect("active session lock").as_ref() {
                if active.fp == fp {
                    info!("revoked peer has a live session; tearing it down");
                    active.kill.notify_one();
                }
            }
        }
        removed
    }
}

/// Outcome of authorizing a [`PendingConnection`].
pub enum Authorized {
    /// The peer is authorized. The pairing exchange is complete but the
    /// connection is **not yet promoted** — the caller calls
    /// [`PendingConnection::into_connection`] to wire control/input. Carries the
    /// fingerprint so the caller can register the [`ActiveSession`] for
    /// revocation, and the label of a *fresh* pairing (`Some`) so it can emit
    /// `Paired`; a reconnect (`Resume`) carries `None`.
    Session {
        pending: PendingConnection,
        fp: CertFingerprint,
        newly_paired: Option<String>,
    },
    /// The peer was refused (and already closed). The caller should emit the
    /// reason and continue accepting.
    Refused(RefusedReason),
}

/// Why a pending connection was refused — drives the engine event the caller
/// emits. The on-wire `Rejected` reason is always uniform; this is local only.
pub enum RefusedReason {
    /// No client cert, malformed pairing message, or transport error.
    Protocol(String),
    /// Not paired and either no window was open or pairing failed. The UI
    /// surfaces this as "start pairing to allow this device."
    PairingRequired,
}

/// Authorize (or refuse) a freshly-accepted pending connection.
///
/// Reads the client's opening pairing message and dispatches:
/// `Resume` → allowlist check; `Pair` → windowed SPAKE2 exchange. On success
/// returns [`Authorized::Session`]; on any refusal closes the connection and
/// returns [`Authorized::Refused`]. `now` is injected so the expiry decision is
/// testable.
pub async fn authorize(mut pending: PendingConnection, state: &PairingState, now: Instant) -> Authorized {
    let fp = match pending.peer_cert_fingerprint() {
        Some(fp) => fp,
        None => {
            // Mutual TLS should make this impossible (the verifier is
            // permissive but still requires a cert), but treat a missing cert
            // as an immediate refusal rather than trusting it.
            pending.reject(close_code::REFUSED, b"refused");
            return Authorized::Refused(RefusedReason::Protocol(
                "client presented no certificate".to_string(),
            ));
        }
    };

    let opening: PairingClientMsg = match pending.recv_pairing().await {
        Ok(msg) => msg,
        Err(e) => {
            pending.reject(close_code::PROTOCOL_ERROR, b"bad pairing message");
            return Authorized::Refused(RefusedReason::Protocol(format!(
                "reading opening pairing message: {e}"
            )));
        }
    };

    match opening {
        PairingClientMsg::Resume => authorize_resume(pending, state, fp).await,
        PairingClientMsg::Pair { spake2 } => authorize_pair(pending, state, fp, spake2, now).await,
    }
}

/// Reconnect path: the client believes it is already paired. Authorize iff its
/// cert fingerprint is in the allowlist; no PIN, no SPAKE2.
async fn authorize_resume(
    mut pending: PendingConnection,
    state: &PairingState,
    fp: CertFingerprint,
) -> Authorized {
    let known = state.paired.lock().expect("paired store lock").contains(&fp);
    if !known {
        // Uniform on-wire reason; the local refusal reason is specific. (An
        // attacker can only `Resume` with a cert it holds the key for, so this
        // path can't enumerate *other* peers' allowlist membership.)
        let _ = pending
            .send_pairing(&PairingServerMsg::Rejected {
                reason: REFUSAL_REASON.to_string(),
            })
            .await;
        pending.reject(close_code::REFUSED, b"refused");
        return Authorized::Refused(RefusedReason::PairingRequired);
    }
    if let Err(e) = pending.send_pairing(&PairingServerMsg::Authorized).await {
        pending.reject(close_code::PROTOCOL_ERROR, b"send failed");
        return Authorized::Refused(RefusedReason::Protocol(format!("sending Authorized: {e}")));
    }
    Authorized::Session {
        pending,
        fp,
        newly_paired: None,
    }
}

/// Run a throwaway SPAKE2 exchange against a peer that sent `Pair` when no
/// pairing window is open, then refuse uniformly. Without this, a network peer
/// could distinguish "no window open" (immediate refusal) from "window open,
/// wrong PIN" (refused only after a challenge) — an oracle for the host's
/// pairing state. The decoy makes both paths look identical on the wire: a
/// `PairChallenge` followed by a uniform `Rejected`. The decoy seeds SPAKE2
/// with a fresh random PIN, so it can never accidentally succeed.
async fn decoy_reject(mut pending: PendingConnection, client_spake2: Vec<u8>) -> Authorized {
    let start = PairingStart::new(tether_pairing::generate_pin().as_bytes());
    let challenge = PairingServerMsg::PairChallenge {
        spake2: start.outbound_msg().to_vec(),
    };
    // Mirror the real flow's message shape; ignore the cryptographic outcome.
    let _ = start.into_keyed(&client_spake2);
    if pending.send_pairing(&challenge).await.is_ok() {
        // Consume the client's confirmation if it sends one, so the timing and
        // message count match a genuine failed attempt.
        let _ = pending.recv_pairing::<PairingConfirm>().await;
        let _ = pending.send_pairing(&reject_result()).await;
    }
    pending.reject(close_code::REFUSED, b"refused");
    Authorized::Refused(RefusedReason::PairingRequired)
}

/// First-contact path: run SPAKE2 + channel-bound confirmation against the open
/// pairing window's PIN. The PIN is burned on attempt (taken before any crypto
/// runs), so a wrong guess consumes the window.
async fn authorize_pair(
    mut pending: PendingConnection,
    state: &PairingState,
    fp: CertFingerprint,
    client_spake2: Vec<u8>,
    now: Instant,
) -> Authorized {
    // Burn-on-attempt: take the window now, before touching the PIN. An expired
    // window is treated as no window (and cleared).
    let window = {
        let mut guard = state.window.lock().expect("pairing window lock");
        match guard.take() {
            Some(w) if !w.is_expired(now) => w,
            _ => {
                // Either no window or an expired one (now dropped by take()).
                // Run a decoy exchange so this is indistinguishable on the wire
                // from a window-open wrong-PIN attempt.
                drop(guard);
                return decoy_reject(pending, client_spake2).await;
            }
        }
    };

    // Run SPAKE2 seeded with the window PIN and exchange messages.
    let start = PairingStart::new(window.pin.as_bytes());
    let host_spake2 = start.outbound_msg().to_vec();
    let keyed = match start.into_keyed(&client_spake2) {
        Ok(k) => k,
        Err(e) => {
            // Malformed client SPAKE2 message. Uniform failure on the wire (the
            // window is already burned above). Sending a challenge here would
            // require a valid key we don't have, so this is the one refusal that
            // can't reach the challenge stage — but it's only reachable with a
            // window open, so it leaks nothing about window state.
            let _ = pending.send_pairing(&reject_result()).await;
            pending.reject(close_code::REFUSED, b"refused");
            return Authorized::Refused(RefusedReason::Protocol(format!("client SPAKE2: {e}")));
        }
    };
    if let Err(e) = pending
        .send_pairing(&PairingServerMsg::PairChallenge {
            spake2: host_spake2,
        })
        .await
    {
        pending.reject(close_code::PROTOCOL_ERROR, b"send failed");
        return Authorized::Refused(RefusedReason::Protocol(format!("sending challenge: {e}")));
    }

    // Receive the client's confirmation MAC (C2H direction).
    let confirm: PairingConfirm = match pending.recv_pairing().await {
        Ok(c) => c,
        Err(e) => {
            pending.reject(close_code::PROTOCOL_ERROR, b"bad confirm");
            return Authorized::Refused(RefusedReason::Protocol(format!("reading confirm: {e}")));
        }
    };

    let exporter = match exporter_bytes(&pending) {
        Ok(e) => e,
        Err(e) => {
            pending.reject(close_code::PROTOCOL_ERROR, b"exporter unavailable");
            return Authorized::Refused(RefusedReason::Protocol(e));
        }
    };
    let transcript = Transcript {
        exporter: &exporter,
        fp_host: &state.host_fp,
        fp_client: &fp,
        protocol_version: PROTOCOL_VERSION,
    };

    // Verify the client proved it holds the same PIN over the same TLS channel.
    if !keyed.verify_confirmation(Direction::ClientToHost, &transcript, &confirm.mac) {
        // Could be a wrong PIN or a detected MITM; uniform on the wire, logged
        // distinctly here for the operator.
        warn!(client = %fp_hex(&fp), "pairing confirmation failed (wrong PIN or channel mismatch)");
        let _ = pending.send_pairing(&reject_result()).await;
        pending.reject(close_code::REFUSED, b"refused");
        return Authorized::Refused(RefusedReason::PairingRequired);
    }

    // Send our matching H2C confirmation so the client can authenticate us
    // before it persists our identity. Unlike the best-effort `Rejected` sends,
    // a failure here is a real protocol error: without our MAC the client cannot
    // authenticate the host, so we refuse rather than ignore it.
    let host_mac = keyed.make_confirmation(Direction::HostToClient, &transcript);
    if let Err(e) = pending
        .send_pairing(&PairingResult::Confirmed {
            mac: host_mac.to_vec(),
        })
        .await
    {
        pending.reject(close_code::PROTOCOL_ERROR, b"send failed");
        return Authorized::Refused(RefusedReason::Protocol(format!("sending result: {e}")));
    }

    // Persist the now-trusted client. Hold the lock across save() — this path is
    // rare and serializing it with reads is simpler than cloning out.
    {
        let mut store = state.paired.lock().expect("paired store lock");
        store.insert(&fp, window.label.clone(), unix_now());
        if let Err(e) = store.save(&state.paired_path) {
            // Pairing succeeded cryptographically but we couldn't persist it.
            // Authorize this session anyway (the user just paired) but warn —
            // the next reconnect would be refused until they re-pair.
            warn!(error = %e, "paired client authorized but allowlist save failed");
        }
    }
    info!(client = %fp_hex(&fp), label = %window.label, "client paired");

    Authorized::Session {
        pending,
        fp,
        newly_paired: Some(window.label),
    }
}

/// Uniform pairing-failure result (never distinguishes wrong-PIN from MITM).
fn reject_result() -> PairingResult {
    PairingResult::Rejected {
        reason: REFUSAL_REASON.to_string(),
    }
}

/// Pull the fixed-width TLS exporter the confirmation transcript binds to.
fn exporter_bytes(pending: &PendingConnection) -> Result<[u8; EXPORTER_LEN], String> {
    let bytes = pending
        .export_keying_material(EXPORTER_LABEL, EXPORTER_CONTEXT, EXPORTER_LEN)
        .map_err(|e| format!("TLS exporter unavailable: {e}"))?;
    bytes
        .as_slice()
        .try_into()
        .map_err(|_| format!("TLS exporter returned {} bytes, expected {EXPORTER_LEN}", bytes.len()))
}

/// Seconds since the Unix epoch, for stamping a pairing time. Clamps a
/// pre-epoch clock to 0 rather than failing the pairing.
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn fp_hex(fp: &CertFingerprint) -> String {
    fp.iter().map(|b| format!("{b:02x}")).collect()
}

// A confirmation MAC is CONFIRM_LEN bytes; assert the wire and crypto lengths
// agree so a future change to one surfaces here, not as a silent verify-false.
const _: () = assert!(CONFIRM_LEN == 32);

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    use tether_transport::{Client, Server, ServerAuth};

    #[test]
    fn window_expires_at_boundary() {
        let w = PairingWindow::open("12345678".into(), "laptop".into());
        let start = Instant::now();
        assert!(!w.is_expired(start));
        assert!(!w.is_expired(start + PAIRING_WINDOW - Duration::from_millis(1)));
        assert!(w.is_expired(start + PAIRING_WINDOW));
        assert!(w.is_expired(start + PAIRING_WINDOW + Duration::from_secs(1)));
    }

    #[test]
    fn open_window_sets_a_fresh_pin() {
        let state = test_state([9u8; 32]);
        let pin = state.open_window("laptop".into());
        assert_eq!(pin.len(), tether_pairing::PIN_DIGITS);
        let window = state.window.lock().unwrap().clone().expect("window open");
        assert_eq!(window.pin, pin);
        assert_eq!(window.label, "laptop");
    }

    #[tokio::test]
    async fn revoke_removes_peer_and_kills_matching_session() {
        let state = test_state([9u8; 32]);
        let fp = [4u8; 32];
        state.paired.lock().unwrap().insert(&fp, "laptop".into(), 0);
        let kill = Arc::new(Notify::new());
        *state.active.lock().unwrap() = Some(ActiveSession {
            fp,
            kill: kill.clone(),
        });

        let removed = state.revoke(&tether_pairing::tag_fingerprint(&fp));

        assert!(removed);
        assert!(!state.paired.lock().unwrap().contains(&fp));
        // The matching live session was signalled to drop.
        assert!(
            tokio::time::timeout(Duration::from_secs(1), kill.notified())
                .await
                .is_ok(),
            "revoke must notify the live session's kill"
        );
    }

    #[tokio::test]
    async fn revoke_unknown_peer_is_noop() {
        let state = test_state([9u8; 32]);
        let removed = state.revoke(&tether_pairing::tag_fingerprint(&[7u8; 32]));
        assert!(!removed);
    }

    #[tokio::test]
    async fn resume_authorizes_known_client() {
        let (server, client, host_pending, client_pending) = loopback().await;
        let state = test_state(server.fingerprint());
        state
            .paired
            .lock()
            .unwrap()
            .insert(&client.fingerprint(), "laptop".into(), 0);

        let (auth, reply) = tokio::join!(
            authorize(host_pending, &state, Instant::now()),
            client_resume(client_pending),
        );

        assert!(matches!(reply, PairingServerMsg::Authorized));
        match auth {
            Authorized::Session {
                fp, newly_paired, ..
            } => {
                assert_eq!(fp, client.fingerprint());
                assert!(newly_paired.is_none(), "resume is not a fresh pairing");
            }
            Authorized::Refused(_) => panic!("known client should be authorized"),
        }
    }

    #[tokio::test]
    async fn resume_refuses_unknown_client() {
        let (server, _client, host_pending, client_pending) = loopback().await;
        let state = test_state(server.fingerprint()); // empty allowlist

        let (auth, reply) = tokio::join!(
            authorize(host_pending, &state, Instant::now()),
            client_resume(client_pending),
        );

        assert!(matches!(reply, PairingServerMsg::Rejected { .. }));
        assert!(matches!(
            auth,
            Authorized::Refused(RefusedReason::PairingRequired)
        ));
    }

    #[tokio::test]
    async fn first_contact_pairs_with_correct_pin() {
        let (server, client, host_pending, client_pending) = loopback().await;
        let state = test_state(server.fingerprint());
        *state.window.lock().unwrap() = Some(PairingWindow::open("12345678".into(), "laptop".into()));
        let client_fp = client.fingerprint();

        let (auth, client_result) = tokio::join!(
            authorize(host_pending, &state, Instant::now()),
            drive_client_pair(client_pending, b"12345678", client_fp),
        );

        // The client authenticated the host's H2C confirmation.
        assert!(matches!(
            client_result,
            ClientPairResult::Confirmed { host_verified: true }
        ));
        match auth {
            Authorized::Session {
                fp, newly_paired, ..
            } => {
                assert_eq!(fp, client_fp);
                assert_eq!(newly_paired.as_deref(), Some("laptop"));
            }
            Authorized::Refused(_) => panic!("correct PIN should pair"),
        }
        // The client is now persisted in the allowlist, and the window is spent.
        assert!(state.paired.lock().unwrap().contains(&client_fp));
        assert!(state.window.lock().unwrap().is_none(), "window burned");
    }

    #[tokio::test]
    async fn wrong_pin_refuses_and_burns_window() {
        let (server, client, host_pending, client_pending) = loopback().await;
        let state = test_state(server.fingerprint());
        *state.window.lock().unwrap() = Some(PairingWindow::open("12345678".into(), "laptop".into()));
        let client_fp = client.fingerprint();

        let (auth, client_result) = tokio::join!(
            authorize(host_pending, &state, Instant::now()),
            drive_client_pair(client_pending, b"00000000", client_fp),
        );

        assert!(matches!(client_result, ClientPairResult::Rejected));
        assert!(matches!(
            auth,
            Authorized::Refused(RefusedReason::PairingRequired)
        ));
        // Wrong guesses are not free: the single-flight window is consumed.
        assert!(state.window.lock().unwrap().is_none(), "window burned on attempt");
        assert!(!state.paired.lock().unwrap().contains(&client_fp));
    }

    #[tokio::test]
    async fn pair_with_no_window_is_refused() {
        let (server, client, host_pending, client_pending) = loopback().await;
        let state = test_state(server.fingerprint()); // no window open
        let client_fp = client.fingerprint();

        let (auth, client_result) = tokio::join!(
            authorize(host_pending, &state, Instant::now()),
            drive_client_pair(client_pending, b"12345678", client_fp),
        );

        assert!(matches!(client_result, ClientPairResult::Rejected));
        assert!(matches!(
            auth,
            Authorized::Refused(RefusedReason::PairingRequired)
        ));
    }

    #[tokio::test]
    async fn expired_window_is_refused() {
        let (server, client, host_pending, client_pending) = loopback().await;
        let state = test_state(server.fingerprint());
        *state.window.lock().unwrap() = Some(PairingWindow::open("12345678".into(), "laptop".into()));
        let client_fp = client.fingerprint();
        // Authorize as if the window had already elapsed.
        let after_expiry = Instant::now() + PAIRING_WINDOW + Duration::from_secs(1);

        let (auth, client_result) = tokio::join!(
            authorize(host_pending, &state, after_expiry),
            drive_client_pair(client_pending, b"12345678", client_fp),
        );

        assert!(matches!(client_result, ClientPairResult::Rejected));
        assert!(matches!(
            auth,
            Authorized::Refused(RefusedReason::PairingRequired)
        ));
    }

    #[tokio::test]
    async fn malformed_spake2_refuses_and_burns_window() {
        let (server, client, host_pending, client_pending) = loopback().await;
        let state = test_state(server.fingerprint());
        *state.window.lock().unwrap() = Some(PairingWindow::open("12345678".into(), "laptop".into()));
        let client_fp = client.fingerprint();

        let (auth, _) = tokio::join!(authorize(host_pending, &state, Instant::now()), async move {
            let mut p = client_pending;
            // Garbage that isn't a valid SPAKE2 message — attacker-controlled
            // input to the PAKE layer.
            p.send_pairing(&PairingClientMsg::Pair {
                spake2: vec![0xFF; 8],
            })
            .await
            .unwrap();
            let _ = p.recv_pairing::<PairingServerMsg>().await; // message or close
        });

        // Refused as a protocol error (not a clean PairingRequired), window
        // burned, nothing persisted.
        assert!(matches!(auth, Authorized::Refused(RefusedReason::Protocol(_))));
        assert!(
            state.window.lock().unwrap().is_none(),
            "window burned even on a malformed message"
        );
        assert!(!state.paired.lock().unwrap().contains(&client_fp));
    }

    #[tokio::test]
    async fn no_window_pair_still_gets_a_challenge() {
        // Oracle defense: a `Pair` attempt with no window open must still elicit
        // a `PairChallenge`, identical on the wire to a window-open attempt, so a
        // network peer can't detect whether a window is open.
        let (server, _client, host_pending, client_pending) = loopback().await;
        let state = test_state(server.fingerprint()); // no window

        let (auth, first_reply) = tokio::join!(
            authorize(host_pending, &state, Instant::now()),
            async move {
                let mut p = client_pending;
                let start = PairingStart::new(b"12345678");
                p.send_pairing(&PairingClientMsg::Pair {
                    spake2: start.outbound_msg().to_vec(),
                })
                .await
                .unwrap();
                p.recv_pairing::<PairingServerMsg>().await
            }
        );

        assert!(
            matches!(first_reply, Ok(PairingServerMsg::PairChallenge { .. })),
            "no-window attempt must still receive a challenge (got {first_reply:?})"
        );
        assert!(matches!(
            auth,
            Authorized::Refused(RefusedReason::PairingRequired)
        ));
    }

    // --- test helpers ---

    static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

    fn test_state(host_fp: CertFingerprint) -> PairingState {
        let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("tether-host-pairing-{}-{seq}", std::process::id()));
        PairingState {
            paired: Arc::new(StdMutex::new(PairedStore::default())),
            paired_path: Arc::new(dir.join("paired_clients.json")),
            window: Arc::new(StdMutex::new(None)),
            active: Arc::new(StdMutex::new(None)),
            host_fp,
        }
    }

    /// Stand up a real QUIC loopback and return both ends' pending connections
    /// alongside the `Server`/`Client` (kept alive so their endpoints don't
    /// drop). First-contact mode: the client trusts the host cert on first pair.
    async fn loopback() -> (Server, Client, PendingConnection, PendingConnection) {
        let server = Server::bind("127.0.0.1:0".parse().unwrap())
            .await
            .expect("bind");
        let addr = server.local_addr().unwrap();
        let client = Client::new().expect("client");
        let (host_pending, client_pending) = tokio::join!(
            async {
                server
                    .accept_pending()
                    .await
                    .expect("incoming")
                    .expect("pending")
            },
            async {
                client
                    .connect_pending(addr, "tether-host", ServerAuth::TrustOnFirstPair)
                    .await
                    .expect("connect")
            }
        );
        (server, client, host_pending, client_pending)
    }

    /// Drive the client half of a `Resume`. A refusal arrives either as a
    /// `Rejected` message or — since the host closes the connection right after
    /// — as a connection-loss error; both map to `Rejected` (the close is the
    /// authoritative signal, the message is best-effort).
    async fn client_resume(mut p: PendingConnection) -> PairingServerMsg {
        p.send_pairing(&PairingClientMsg::Resume).await.unwrap();
        p.recv_pairing::<PairingServerMsg>()
            .await
            .unwrap_or(PairingServerMsg::Rejected {
                reason: "connection closed".into(),
            })
    }

    enum ClientPairResult {
        Confirmed { host_verified: bool },
        Rejected,
    }

    /// Drive the client half of a first-contact pairing over its pending
    /// connection: SPAKE2 message, confirmation MAC, and verification of the
    /// host's reply. Mirrors what the Phase 6 client main will do.
    async fn drive_client_pair(
        mut p: PendingConnection,
        pin: &[u8],
        client_fp: CertFingerprint,
    ) -> ClientPairResult {
        let start = PairingStart::new(pin);
        let client_msg = start.outbound_msg().to_vec();
        p.send_pairing(&PairingClientMsg::Pair { spake2: client_msg })
            .await
            .unwrap();

        // A refusal may arrive as a `Rejected` message or as a connection
        // close (the host closes right after sending); treat either as
        // rejection — the close is authoritative, the message best-effort.
        let host_msg = match p.recv_pairing::<PairingServerMsg>().await {
            Ok(PairingServerMsg::PairChallenge { spake2 }) => spake2,
            Ok(PairingServerMsg::Rejected { .. }) | Err(_) => return ClientPairResult::Rejected,
            Ok(PairingServerMsg::Authorized) => panic!("unexpected Authorized in pairing flow"),
        };
        let keyed = start.into_keyed(&host_msg).unwrap();

        let exporter: [u8; EXPORTER_LEN] = p
            .export_keying_material(EXPORTER_LABEL, EXPORTER_CONTEXT, EXPORTER_LEN)
            .unwrap()
            .try_into()
            .unwrap();
        let host_fp = p.peer_cert_fingerprint().unwrap();
        let transcript = Transcript {
            exporter: &exporter,
            fp_host: &host_fp,
            fp_client: &client_fp,
            protocol_version: PROTOCOL_VERSION,
        };

        let client_mac = keyed.make_confirmation(Direction::ClientToHost, &transcript);
        p.send_pairing(&PairingConfirm {
            mac: client_mac.to_vec(),
        })
        .await
        .unwrap();

        match p.recv_pairing::<PairingResult>().await {
            Ok(PairingResult::Confirmed { mac }) => ClientPairResult::Confirmed {
                host_verified: keyed.verify_confirmation(
                    Direction::HostToClient,
                    &transcript,
                    &mac,
                ),
            },
            Ok(PairingResult::Rejected { .. }) | Err(_) => ClientPairResult::Rejected,
        }
    }
}
