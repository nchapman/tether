//! Client side of device pairing: prove the PIN (first contact) or resume a
//! pinned host, then promote the connection to a session.
//!
//! The host owns the authorization policy (`tether-host`'s `pairing` module);
//! this is just the client half of the wire exchange in
//! [`tether_protocol::pairing`], with the crypto in `tether-pairing`. The two
//! values the confirmation MAC binds to — the host's cert fingerprint and the
//! TLS exporter — both come from the [`PendingConnection`].
//!
//! Security note: on first contact the client verifies the host's `H2C`
//! confirmation **before** returning success. A failure there means the host
//! couldn't prove it holds the same PIN over the same TLS channel (a wrong PIN
//! or a relay/MITM), so we refuse rather than pin a possibly-attacker identity.

use anyhow::{anyhow, bail, Context};

use tether_pairing::{
    Direction, PairingStart, Transcript, EXPORTER_CONTEXT, EXPORTER_LABEL, EXPORTER_LEN,
};
use tether_protocol::pairing::{PairingClientMsg, PairingConfirm, PairingResult, PairingServerMsg};
use tether_protocol::PROTOCOL_VERSION;
use tether_transport::{CertFingerprint, Connection, PendingConnection};

/// How the client authenticates the host on this connection.
pub enum HostAuth {
    /// First contact: the user supplied a PIN. The TLS layer trusts the host
    /// cert on first pair; SPAKE2 + the confirmation MAC authenticate it. On
    /// success the caller persists the learned host fingerprint.
    FirstContact { pin: String },
    /// Reconnect: the host cert is already pinned (`ServerAuth::Pinned`), so the
    /// client just resumes; the host re-checks its allowlist by cert.
    Resume,
}

/// Drive the client side of pairing/resume over `pending`, then promote it to a
/// full [`Connection`]. Returns the connection and the host's cert fingerprint
/// (so a first-contact caller can record it in known-hosts). Any refusal — a
/// `Rejected` reply, a dropped connection, a wrong PIN, or a failed host
/// confirmation — surfaces as an error; no session is produced.
pub async fn establish(
    mut pending: PendingConnection,
    mode: &HostAuth,
    client_fp: CertFingerprint,
) -> anyhow::Result<(Connection, CertFingerprint)> {
    let host_fp = pending
        .peer_cert_fingerprint()
        .context("host presented no certificate")?;

    match mode {
        HostAuth::Resume => {
            pending.send_pairing(&PairingClientMsg::Resume).await?;
            // A refusal arrives as `Rejected` or — since the host closes right
            // after — as a connection-loss error; both mean "re-pair needed".
            match pending.recv_pairing::<PairingServerMsg>().await {
                Ok(PairingServerMsg::Authorized) => {}
                Ok(_) => bail!("host refused resume; pairing required"),
                Err(e) => bail!("host refused resume ({e}); pairing required"),
            }
        }
        HostAuth::FirstContact { pin } => {
            run_first_contact(&mut pending, pin, &host_fp, &client_fp).await?;
        }
    }

    let conn = pending
        .into_connection()
        .await
        .context("promoting paired connection to a session")?;
    Ok((conn, host_fp))
}

/// The SPAKE2 + channel-bound confirmation exchange. Returns `Ok` only once the
/// host's `H2C` confirmation verifies.
async fn run_first_contact(
    pending: &mut PendingConnection,
    pin: &str,
    host_fp: &CertFingerprint,
    client_fp: &CertFingerprint,
) -> anyhow::Result<()> {
    let start = PairingStart::new(pin.as_bytes());
    pending
        .send_pairing(&PairingClientMsg::Pair {
            spake2: start.outbound_msg().to_vec(),
        })
        .await?;

    // Uniform failure: a refusal is a `Rejected`/`Authorized`-shaped reply or a
    // closed connection — all "pairing failed" to us.
    let host_spake2 = match pending.recv_pairing::<PairingServerMsg>().await {
        Ok(PairingServerMsg::PairChallenge { spake2 }) => spake2,
        Ok(_) => bail!("pairing failed"),
        Err(_) => bail!("pairing failed (connection closed)"),
    };
    let keyed = start
        .into_keyed(&host_spake2)
        .map_err(|_| anyhow!("pairing failed (bad host message)"))?;

    let exporter: [u8; EXPORTER_LEN] = pending
        .export_keying_material(EXPORTER_LABEL, EXPORTER_CONTEXT, EXPORTER_LEN)?
        .try_into()
        .map_err(|_| anyhow!("TLS exporter returned an unexpected length"))?;
    let transcript = Transcript {
        exporter: &exporter,
        fp_host: host_fp,
        fp_client: client_fp,
        protocol_version: PROTOCOL_VERSION,
    };

    // Prove our PIN to the host (C2H).
    let client_mac = keyed.make_confirmation(Direction::ClientToHost, &transcript);
    pending
        .send_pairing(&PairingConfirm {
            mac: client_mac.to_vec(),
        })
        .await?;

    // Verify the host proved the same PIN over the same channel (H2C) before we
    // trust it. A mismatch is a wrong PIN or a detected MITM.
    match pending.recv_pairing::<PairingResult>().await {
        Ok(PairingResult::Confirmed { mac }) => {
            if !keyed.verify_confirmation(Direction::HostToClient, &transcript, &mac) {
                bail!("host confirmation failed — wrong PIN or a man-in-the-middle; not pairing");
            }
            Ok(())
        }
        Ok(PairingResult::Rejected { .. }) => bail!("pairing failed"),
        Err(_) => bail!("pairing failed (connection closed)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tether_transport::{Client, Server, ServerAuth};

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

    /// Scripted host: accept a `Resume` and reply `Authorized` (then promote) or
    /// `Rejected`.
    async fn host_resume(mut p: PendingConnection, authorize: bool) {
        let _: PairingClientMsg = p.recv_pairing().await.unwrap();
        if authorize {
            p.send_pairing(&PairingServerMsg::Authorized).await.unwrap();
            // Match the client's into_connection() so its promotion completes.
            p.into_connection().await.unwrap();
        } else {
            let _ = p
                .send_pairing(&PairingServerMsg::Rejected {
                    reason: "pairing failed".into(),
                })
                .await;
            p.reject(2, b"refused");
        }
    }

    /// Scripted host pairing side. `good_confirmation = false` sends a bogus
    /// `H2C` MAC to exercise the client's MITM check.
    async fn host_pair(
        mut p: PendingConnection,
        pin: &[u8],
        host_fp: CertFingerprint,
        client_fp: CertFingerprint,
        good_confirmation: bool,
    ) {
        let client_spake2 = match p.recv_pairing::<PairingClientMsg>().await.unwrap() {
            PairingClientMsg::Pair { spake2 } => spake2,
            PairingClientMsg::Resume => panic!("expected Pair"),
        };
        let start = PairingStart::new(pin);
        p.send_pairing(&PairingServerMsg::PairChallenge {
            spake2: start.outbound_msg().to_vec(),
        })
        .await
        .unwrap();
        let keyed = start.into_keyed(&client_spake2).unwrap();
        let confirm: PairingConfirm = p.recv_pairing().await.unwrap();

        let exporter: [u8; EXPORTER_LEN] = p
            .export_keying_material(EXPORTER_LABEL, EXPORTER_CONTEXT, EXPORTER_LEN)
            .unwrap()
            .try_into()
            .unwrap();
        let transcript = Transcript {
            exporter: &exporter,
            fp_host: &host_fp,
            fp_client: &client_fp,
            protocol_version: PROTOCOL_VERSION,
        };
        // (The host would verify C2H here; the client tests don't need to.)
        let _ = keyed.verify_confirmation(Direction::ClientToHost, &transcript, &confirm.mac);

        let mac = if good_confirmation {
            keyed
                .make_confirmation(Direction::HostToClient, &transcript)
                .to_vec()
        } else {
            vec![0u8; EXPORTER_LEN] // wrong MAC → client must reject
        };
        p.send_pairing(&PairingResult::Confirmed { mac })
            .await
            .unwrap();
        if good_confirmation {
            // Client will promote; match it so its into_connection() completes.
            p.into_connection().await.unwrap();
        }
    }

    #[tokio::test]
    async fn resume_authorized_promotes_and_returns_host_fp() {
        let (server, client, host_pending, client_pending) = loopback().await;
        let (host_done, client_res) = tokio::join!(
            host_resume(host_pending, true),
            establish(client_pending, &HostAuth::Resume, client.fingerprint()),
        );
        let _ = host_done;
        let (_conn, host_fp) = client_res.expect("resume should succeed");
        assert_eq!(host_fp, server.fingerprint());
    }

    #[tokio::test]
    async fn resume_rejected_errors() {
        let (_server, client, host_pending, client_pending) = loopback().await;
        let (_h, client_res) = tokio::join!(
            host_resume(host_pending, false),
            establish(client_pending, &HostAuth::Resume, client.fingerprint()),
        );
        assert!(client_res.is_err(), "rejected resume must error");
    }

    #[tokio::test]
    async fn first_contact_matching_pin_pairs_and_returns_host_fp() {
        let (server, client, host_pending, client_pending) = loopback().await;
        let client_fp = client.fingerprint();
        let host_fp = server.fingerprint();
        let mode = HostAuth::FirstContact {
            pin: "12345678".into(),
        };
        let (_h, client_res) = tokio::join!(
            host_pair(host_pending, b"12345678", host_fp, client_fp, true),
            establish(client_pending, &mode, client_fp),
        );
        let (_conn, learned_fp) = client_res.expect("matching PIN should pair");
        assert_eq!(
            learned_fp, host_fp,
            "client learns the host fingerprint to pin"
        );
    }

    #[tokio::test]
    async fn first_contact_bad_host_confirmation_is_rejected() {
        // The client must refuse if the host can't prove the PIN/channel — this
        // is the anti-MITM check on the client side.
        let (server, client, host_pending, client_pending) = loopback().await;
        let client_fp = client.fingerprint();
        let host_fp = server.fingerprint();
        let mode = HostAuth::FirstContact {
            pin: "12345678".into(),
        };
        let (_h, client_res) = tokio::join!(
            host_pair(host_pending, b"12345678", host_fp, client_fp, false),
            establish(client_pending, &mode, client_fp),
        );
        assert!(
            client_res.is_err(),
            "a bad host confirmation must be rejected, not pinned"
        );
    }
}
