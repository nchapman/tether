//! Transport-level pairing tests: a full first-contact SPAKE2 exchange driven
//! over a `PendingConnection`'s pairing stream, and the reject path. These
//! validate the plumbing (typed pairing messages flow on the pairing stream,
//! `into_connection` yields a working session afterward, `reject` prevents a
//! session). The host/client *policy* (allowlist, PIN window) lands in the
//! binaries; here both sides cooperate so we exercise the wire mechanics.

use std::net::Ipv4Addr;

use tether_pairing::{
    Direction, Fingerprint, PairingStart, Transcript, EXPORTER_CONTEXT, EXPORTER_LABEL,
    EXPORTER_LEN,
};
use tether_protocol::control::ControlMessage;
use tether_protocol::pairing::{PairingClientMsg, PairingConfirm, PairingResult, PairingServerMsg};
use tether_protocol::PROTOCOL_VERSION;
use tether_transport::{Client, Server, ServerAuth};

fn transcript<'a>(
    exporter: &'a [u8; EXPORTER_LEN],
    fp_host: &'a Fingerprint,
    fp_client: &'a Fingerprint,
) -> Transcript<'a> {
    Transcript {
        exporter,
        fp_host,
        fp_client,
        protocol_version: PROTOCOL_VERSION,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn first_contact_pairing_then_session() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_env_filter("tether_transport=trace,quinn=warn")
        .try_init();

    const PIN: &[u8] = b"48271930";

    let server = Server::bind((Ipv4Addr::LOCALHOST, 0).into()).await?;
    let server_addr = server.local_addr()?;
    let host_fp = server.fingerprint();

    // ---- Host side ----
    let server_task = tokio::spawn(async move {
        let mut pending = server.accept_pending().await.expect("server closed")?;
        let client_fp = pending
            .peer_cert_fingerprint()
            .expect("client presented a cert");

        // Expect the client's opening Pair message.
        let client_spake2 = match pending.recv_pairing::<PairingClientMsg>().await? {
            PairingClientMsg::Pair { spake2 } => spake2,
            other => panic!("expected Pair, got {other:?}"),
        };

        // Run our half of SPAKE2 and reply with the challenge.
        let host_start = PairingStart::new(PIN);
        let host_msg = host_start.outbound_msg().to_vec();
        let host_keyed = host_start.into_keyed(&client_spake2).expect("host keyed");
        pending
            .send_pairing(&PairingServerMsg::PairChallenge { spake2: host_msg })
            .await?;

        // Verify the client's confirmation, then send ours and authorize.
        let exporter = pending.export_keying_material(EXPORTER_LABEL, EXPORTER_CONTEXT, EXPORTER_LEN)?;
        let exporter: [u8; EXPORTER_LEN] = exporter.try_into().expect("exporter len");
        let t = transcript(&exporter, &host_fp, &client_fp);

        let client_mac = match pending.recv_pairing::<PairingConfirm>().await? {
            PairingConfirm { mac } => mac,
        };
        assert!(
            host_keyed.verify_confirmation(Direction::ClientToHost, &t, &client_mac),
            "client confirmation must verify"
        );
        let host_mac = host_keyed.make_confirmation(Direction::HostToClient, &t);
        pending
            .send_pairing(&PairingResult::Confirmed {
                mac: host_mac.to_vec(),
            })
            .await?;

        // Authorized — promote to a session and exchange a control message.
        let conn = pending.into_connection().await?;
        let msg = conn.recv_control().await?;
        assert!(matches!(msg, ControlMessage::ForceIdr));
        anyhow::Ok(client_fp)
    });

    // ---- Client side ----
    let client = Client::new()?;
    let client_fp = client.fingerprint();
    let mut pending = client
        .connect_pending(server_addr, "tether-host", ServerAuth::TrustOnFirstPair)
        .await?;

    // The host's fingerprint, learned from the cert it presented (the client
    // doesn't know it ahead of first contact).
    let observed_host_fp = pending
        .peer_cert_fingerprint()
        .expect("host presented a cert");

    let client_start = PairingStart::new(PIN);
    let client_msg = client_start.outbound_msg().to_vec();
    pending
        .send_pairing(&PairingClientMsg::Pair { spake2: client_msg })
        .await?;

    let host_spake2 = match pending.recv_pairing::<PairingServerMsg>().await? {
        PairingServerMsg::PairChallenge { spake2 } => spake2,
        other => panic!("expected PairChallenge, got {other:?}"),
    };
    let client_keyed = client_start.into_keyed(&host_spake2).expect("client keyed");

    let exporter = pending.export_keying_material(EXPORTER_LABEL, EXPORTER_CONTEXT, EXPORTER_LEN)?;
    let exporter: [u8; EXPORTER_LEN] = exporter.try_into().expect("exporter len");
    let t = transcript(&exporter, &observed_host_fp, &client_fp);

    let client_mac = client_keyed.make_confirmation(Direction::ClientToHost, &t);
    pending
        .send_pairing(&PairingConfirm {
            mac: client_mac.to_vec(),
        })
        .await?;

    let host_mac = match pending.recv_pairing::<PairingResult>().await? {
        PairingResult::Confirmed { mac } => mac,
        PairingResult::Rejected { reason } => panic!("unexpected rejection: {reason}"),
    };
    assert!(
        client_keyed.verify_confirmation(Direction::HostToClient, &t, &host_mac),
        "host confirmation must verify (proves the host knew the PIN)"
    );

    // Both sides confirmed — open the session and prove input/control flow.
    let conn = pending.into_connection().await?;
    conn.send_control(&ControlMessage::ForceIdr).await?;

    let observed_client_fp = server_task.await??;
    assert_eq!(
        observed_client_fp, client_fp,
        "host saw the client's real fingerprint"
    );
    assert_eq!(observed_host_fp, host_fp, "client saw the host's real fingerprint");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reject_prevents_session() -> anyhow::Result<()> {
    // An unauthorized peer that the host `reject`s must not be able to reach a
    // session: `into_connection` fails because the control/input streams are
    // never accepted.
    let server = Server::bind((Ipv4Addr::LOCALHOST, 0).into()).await?;
    let server_addr = server.local_addr()?;

    let server_task = tokio::spawn(async move {
        let pending = server.accept_pending().await.expect("server closed")?;
        // Host decides this peer is not authorized (no pairing window, not in
        // the allowlist) and refuses without wiring any session streams.
        pending.reject(1, b"not paired");
        anyhow::Ok(())
    });

    let client = Client::new()?;
    let pending = client
        .connect_pending(server_addr, "tether-host", ServerAuth::TrustOnFirstPair)
        .await?;

    // The host rejected. Promoting may *locally* succeed (opening a QUIC
    // stream is buffered and can return before the peer's CONNECTION_CLOSE
    // propagates), but no *working* session can exist: either `into_connection`
    // errors, or the first real control I/O does. The invariant under test is
    // "an unauthorized peer reaches no usable session", not which call surfaces
    // the close.
    match pending.into_connection().await {
        Err(_) => {} // rejected before the streams were wired — fine
        Ok(conn) => {
            let r = conn.recv_control().await;
            assert!(
                r.is_err(),
                "control I/O must fail after the host rejects, got {r:?}"
            );
        }
    }

    server_task.await??;
    Ok(())
}
