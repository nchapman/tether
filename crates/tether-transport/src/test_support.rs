//! In-memory `tokio::io::duplex` implementations of the channel traits.
//!
//! `cargo build` only exposes this module when the `test-support`
//! feature is enabled. Downstream crates (`tether-session`,
//! `tether-host`, `tether-client`) enable it under
//! `[dev-dependencies]` so their integration tests can substitute an
//! in-memory pair for the real QUIC-backed [`crate::Connection`]
//! without bringing up a UDP socket.
//!
//! The feature is strictly additive: it only adds items inside this
//! module; it never modifies any other public API. That contract
//! preserves feature-unification semantics — a release build that
//! happens to enable `test-support` transitively compiles identically
//! to one that doesn't (modulo the existence of this module).
//!
//! ## What's modeled, what isn't
//!
//! Only the [`crate::ControlChannel`] surface is implemented today —
//! that's what end-to-end session loopback tests need. The wire
//! framing matches the real `Connection`: 4-byte LE length prefix +
//! body bytes, bincode payload. The handshake clock-sync stamps are
//! applied at the same points as in production, so latency math
//! tests against this fake exercise the *real* estimator code path.
//!
//! [`crate::InputChannel`] and [`crate::VideoChannel`] fakes are
//! deliberately not provided yet. Add them when a test needs them;
//! the simplest path is the same `tokio::io::duplex` framing for
//! `InputChannel`, and an mpsc-backed unreliable + a separate mpsc
//! per uni stream for `VideoChannel`. The drop-on-overflow semantics
//! of `send_datagram` need a bounded mpsc — match the real
//! transport, don't `.await` block.

use std::sync::Arc;

use async_trait::async_trait;
use tether_protocol::control::{ClientHello, ClockSync, ControlMessage, ServerHello};
use tether_protocol::MonoNanos;
use tokio::io::{AsyncReadExt, AsyncWriteExt, DuplexStream, ReadHalf, WriteHalf};
use tokio::sync::Mutex;

use crate::channel::ControlChannel;
use crate::{Result, TransportError, MAX_FRAMED_MESSAGE};

/// Default per-direction in-memory buffer for the duplex pair.
/// 1 MiB is much larger than any control message we send (control
/// messages are well under 64 KiB by design — see
/// [`MAX_FRAMED_MESSAGE`]) so a single-threaded test never blocks on
/// the channel's internal buffer.
const DEFAULT_DUPLEX_BUFFER: usize = 1024 * 1024;

/// One end of an in-memory control channel pair.
///
/// Wraps a `tokio::io::DuplexStream` split into independent read and
/// write halves. The framing is the same length-prefixed bincode that
/// the real `Connection` uses on the QUIC control stream, so payload
/// bugs in the real code (encoding, decoding, length checks) are
/// reproducible against the fake.
pub struct DuplexControlChannel {
    send: Mutex<WriteHalf<DuplexStream>>,
    recv: Mutex<ReadHalf<DuplexStream>>,
}

/// Build a connected pair of [`DuplexControlChannel`]s. Returns
/// `(host_side, client_side)` by convention — call the first one's
/// [`ControlChannel::recv_client_hello`] / [`ControlChannel::send_server_hello`]
/// and the second's [`ControlChannel::client_handshake`].
///
/// Mechanically: two `tokio::io::duplex` pipes, cross-wired so each
/// side's send half is the other side's recv half.
#[must_use]
pub fn duplex_pair() -> (Arc<DuplexControlChannel>, Arc<DuplexControlChannel>) {
    duplex_pair_with_buffer(DEFAULT_DUPLEX_BUFFER)
}

/// Variant of [`duplex_pair`] that lets the caller pick the per-direction
/// buffer size. Use this only when a test wants to force the buffer to
/// stall (e.g., to test back-pressure).
#[must_use]
pub fn duplex_pair_with_buffer(
    buffer: usize,
) -> (Arc<DuplexControlChannel>, Arc<DuplexControlChannel>) {
    let (a, b) = tokio::io::duplex(buffer);
    let (a_recv, a_send) = tokio::io::split(a);
    let (b_recv, b_send) = tokio::io::split(b);
    let host = Arc::new(DuplexControlChannel {
        send: Mutex::new(a_send),
        recv: Mutex::new(a_recv),
    });
    let client = Arc::new(DuplexControlChannel {
        send: Mutex::new(b_send),
        recv: Mutex::new(b_recv),
    });
    (host, client)
}

impl DuplexControlChannel {
    async fn send_framed_raw<T: serde::Serialize>(&self, msg: &T) -> Result<()> {
        let bytes = tether_protocol::encode(msg)?;
        if bytes.len() > MAX_FRAMED_MESSAGE {
            return Err(TransportError::FrameTooLarge {
                size: bytes.len(),
                max: MAX_FRAMED_MESSAGE,
            });
        }
        let len = u32::try_from(bytes.len()).map_err(|_| TransportError::FrameTooLarge {
            size: bytes.len(),
            max: MAX_FRAMED_MESSAGE,
        })?;
        // Note: no `flush()` here. The real `Connection::write_framed`
        // doesn't flush either — QUIC streams send as soon as bytes
        // are written. Adding `flush` here would make the fake able
        // to surface back-pressure errors the real transport never
        // produces, and a future test that shrinks the duplex buffer
        // to deliberately force back-pressure would see fake-only
        // failures.
        let mut s = self.send.lock().await;
        s.write_all(&len.to_le_bytes()).await?;
        s.write_all(&bytes).await?;
        Ok(())
    }

    async fn recv_framed_raw<T: for<'de> serde::Deserialize<'de>>(&self) -> Result<T> {
        let mut r = self.recv.lock().await;
        let mut len_buf = [0u8; 4];
        r.read_exact(&mut len_buf)
            .await
            .map_err(|_| TransportError::StreamClosed)?;
        let len = u32::from_le_bytes(len_buf) as usize;
        if len > MAX_FRAMED_MESSAGE {
            return Err(TransportError::FrameTooLarge {
                size: len,
                max: MAX_FRAMED_MESSAGE,
            });
        }
        let mut buf = vec![0u8; len];
        r.read_exact(&mut buf)
            .await
            .map_err(|_| TransportError::StreamClosed)?;
        Ok(tether_protocol::decode(&buf)?)
    }
}

#[async_trait]
impl ControlChannel for DuplexControlChannel {
    async fn send_control(&self, msg: &ControlMessage) -> Result<()> {
        self.send_framed_raw(msg).await
    }

    async fn recv_control(&self) -> Result<ControlMessage> {
        self.recv_framed_raw().await
    }

    async fn client_handshake(&self, mut hello: ClientHello) -> Result<(ServerHello, ClockSync)> {
        // Mirror Connection::client_handshake exactly — the clock-sync
        // estimator the test exercises is the same `ClockSync::from_probe`,
        // so the stamps need to land at the same points.
        let t0 = MonoNanos::now();
        match &mut hello {
            ClientHello::V1(body) => body.clock_probe_t0 = t0,
        }
        self.send_framed_raw(&hello).await?;
        let server: ServerHello = self.recv_framed_raw().await?;
        let t3 = MonoNanos::now();
        let (t1_recv, t2_send) = match &server {
            ServerHello::V1(body) => (body.t1_server_recv, body.t2_server_send),
        };
        let sync = ClockSync::from_probe(t0, t1_recv, t2_send, t3);
        Ok((server, sync))
    }

    async fn recv_client_hello(&self) -> Result<(ClientHello, MonoNanos)> {
        let hello: ClientHello = self.recv_framed_raw().await?;
        let t1 = MonoNanos::now();
        Ok((hello, t1))
    }

    async fn send_server_hello(
        &self,
        mut server: ServerHello,
        client_t0: MonoNanos,
        t1_server_recv: MonoNanos,
    ) -> Result<()> {
        // Same stamp-just-before-write pattern as Connection. Tests
        // that assert RTT/offset bounds depend on this stamp being
        // late.
        match &mut server {
            ServerHello::V1(body) => {
                body.clock_probe_t0_echo = client_t0;
                body.t1_server_recv = t1_server_recv;
                body.t2_server_send = MonoNanos::now();
            }
        }
        self.send_framed_raw(&server).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tether_protocol::control::{
        ChromaSubsampling, ClientHelloV1, CodecKind, ControlMessage, GoodbyeCode, ServerHelloV1,
        VideoColorSpec,
    };
    use tokio::time::Duration;

    fn empty_client_hello() -> ClientHello {
        ClientHello::V1(ClientHelloV1 {
            client_name: "test".into(),
            preferred_codecs: vec![CodecKind::H264],
            max_resolution: None,
            clock_probe_t0: MonoNanos::ZERO,
            extensions: Default::default(),
            resume_token: None,
        })
    }

    fn empty_server_hello() -> ServerHello {
        ServerHello::V1(ServerHelloV1 {
            server_name: "test".into(),
            chosen_codec: CodecKind::H264,
            chosen_chroma: ChromaSubsampling::Yuv420,
            color_space: VideoColorSpec::sdr_desktop(),
            resolution: (0, 0),
            clock_probe_t0_echo: MonoNanos::ZERO,
            t1_server_recv: MonoNanos::ZERO,
            t2_server_send: MonoNanos::ZERO,
            extensions: Default::default(),
            resume_token: None,
        })
    }

    #[tokio::test]
    async fn handshake_round_trips_through_duplex() {
        let (host, client) = duplex_pair();

        let host_fut = async move {
            let (_hello, t1) = host.recv_client_hello().await.unwrap();
            host.send_server_hello(empty_server_hello(), MonoNanos::ZERO, t1)
                .await
                .unwrap();
        };
        let client_fut = async move {
            let (_server, sync) = client.client_handshake(empty_client_hello()).await.unwrap();
            // Loopback RTT should be tiny — the in-memory pipe takes
            // microseconds. Pin a generous upper bound so we don't
            // get flaky CI on a slow runner.
            assert!(sync.rtt_nanos < 100_000_000, "rtt was {} ns", sync.rtt_nanos);
        };
        tokio::join!(host_fut, client_fut);
    }

    #[tokio::test]
    async fn control_messages_round_trip_after_handshake() {
        let (host, client) = duplex_pair();
        let client_for_send = client.clone();
        tokio::spawn(async move {
            client_for_send
                .send_control(&ControlMessage::Goodbye {
                    reason: "test".into(),
                    code: GoodbyeCode::Clean,
                })
                .await
                .unwrap();
        });
        let msg = tokio::time::timeout(Duration::from_secs(1), host.recv_control())
            .await
            .expect("timeout")
            .unwrap();
        assert!(matches!(msg, ControlMessage::Goodbye { .. }));
    }

    #[tokio::test]
    async fn dropped_peer_surfaces_stream_closed() {
        let (host, client) = duplex_pair();
        drop(client);
        let err = host.recv_control().await.unwrap_err();
        assert!(matches!(err, TransportError::StreamClosed));
    }
}
