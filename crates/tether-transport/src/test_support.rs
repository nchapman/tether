//! In-memory implementations of the channel traits for tests.
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
//! ## Wire-faithfulness contract
//!
//! Every fake calls the **production** serializers
//! (`tether_protocol::encode` / `decode`) — never a hand-rolled stand-in.
//! Tests can therefore reproduce real wire-format bugs against the
//! fake. Where a test wants to inject malformed input, it does so
//! *before* the fake's serialization layer (raw bytes on the duplex
//! stream / channel), never by reimplementing partial framing.
//!
//! ## Channel-shape choices
//!
//! - [`ControlChannel`] and [`InputChannel`] are stream-shaped on the
//!   wire (length-prefix + bincode over a QUIC reliable stream); they
//!   ride `tokio::io::duplex` here with the same framing.
//! - [`VideoChannel`] datagrams are message-framed on the wire (one
//!   `Datagram` per QUIC datagram); they ride
//!   `tokio::sync::mpsc::channel<Datagram>` here — adding length-prefix
//!   over a duplex stream would itself be a shadow serializer.
//! - [`VideoChannel`] keyframes (IDR uni-streams) are reliable and
//!   effectively unbounded; they ride
//!   `tokio::sync::mpsc::unbounded_channel<VideoPacket>` here.
//!   Real QUIC reliable streams stall the sender's flow control rather
//!   than dropping, which `unbounded_channel` matches.
//!
//! The drop-on-overflow semantics of datagrams are preserved: when the
//! bounded mpsc is full, `send_datagram` silently drops the frame
//! (matching production's UDP-queue overflow behavior, which quinn
//! does not surface through a `Result`) and bumps the fake's
//! `datagrams_dropped` counter for test assertions.

#![deny(unused_imports)]

use std::sync::Arc;

use async_trait::async_trait;
use tether_protocol::control::{ClientHello, ClockSync, ControlMessage, ServerHello};
use tether_protocol::input::InputEvent;
use tether_protocol::video::VideoPacket;
use tether_protocol::MonoNanos;
use tokio::io::{AsyncReadExt, AsyncWriteExt, DuplexStream, ReadHalf, WriteHalf};
use tokio::sync::{mpsc, Mutex};

use crate::channel::{ControlChannel, InputChannel, VideoChannel};
use crate::{Datagram, Result, TransportError, MAX_FRAMED_MESSAGE};

/// Default per-direction in-memory buffer for the duplex pair.
/// 1 MiB is much larger than any control message we send (control
/// messages are well under 64 KiB by design — see
/// [`MAX_FRAMED_MESSAGE`]) so a single-threaded test never blocks on
/// the channel's internal buffer.
const DEFAULT_DUPLEX_BUFFER: usize = 1024 * 1024;

/// Default datagram-channel depth for [`video_duplex_pair`]. 256
/// matches roughly a quarter-second of fragments at 60 fps × 4 KB
/// frames — generous enough to not surface fake-only backpressure
/// under a typical test, small enough that a lossy/slow consumer test
/// can fill it deliberately.
const DEFAULT_VIDEO_DATAGRAM_DEPTH: usize = 256;

// ---------------------------------------------------------------------
// Stream-shaped framing helpers shared by ControlChannel + InputChannel
// ---------------------------------------------------------------------

async fn write_framed<W: AsyncWriteExt + Unpin, T: serde::Serialize>(
    w: &mut W,
    msg: &T,
) -> Result<()> {
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
    w.write_all(&len.to_le_bytes()).await?;
    w.write_all(&bytes).await?;
    Ok(())
}

async fn read_framed<R: AsyncReadExt + Unpin, T: for<'de> serde::Deserialize<'de>>(
    r: &mut R,
) -> Result<T> {
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

// ---------------------------------------------------------------------
// ControlChannel: duplex stream with length-prefixed bincode
// ---------------------------------------------------------------------

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
        // Note: no `flush()` here. The real `Connection::write_framed`
        // doesn't flush either — QUIC streams send as soon as bytes
        // are written. Adding `flush` here would make the fake able
        // to surface back-pressure errors the real transport never
        // produces, and a future test that shrinks the duplex buffer
        // to deliberately force back-pressure would see fake-only
        // failures.
        let mut s = self.send.lock().await;
        write_framed(&mut *s, msg).await
    }

    async fn recv_framed_raw<T: for<'de> serde::Deserialize<'de>>(&self) -> Result<T> {
        let mut r = self.recv.lock().await;
        read_framed(&mut *r).await
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

// ---------------------------------------------------------------------
// InputChannel: stream-shaped, mirrors ControlChannel framing
// ---------------------------------------------------------------------

/// One end of an in-memory input channel pair. The wire shape is the
/// same length-prefix + bincode that the real `Connection`'s input
/// stream uses, so a test that serializes 100 events through the fake
/// will produce identical bytes to the production path.
///
/// Like the real `Connection::InputStream`, each instance is logically
/// uni-directional: the host side reads, the client side writes.
/// `DuplexInputChannel` is symmetric on the trait, but tests should
/// pair `client.send_input` with `host.recv_input`. Calling the wrong
/// direction will simply deadlock on the duplex buffer (no "wrong
/// role" enforcement here — that's a production-only concern tied to
/// `InputStream::Send`/`Recv`).
pub struct DuplexInputChannel {
    send: Mutex<WriteHalf<DuplexStream>>,
    recv: Mutex<ReadHalf<DuplexStream>>,
}

/// Build a connected pair of [`DuplexInputChannel`]s. Returns
/// `(host_side, client_side)` by convention.
#[must_use]
pub fn input_duplex_pair() -> (Arc<DuplexInputChannel>, Arc<DuplexInputChannel>) {
    let (a, b) = tokio::io::duplex(DEFAULT_DUPLEX_BUFFER);
    let (a_recv, a_send) = tokio::io::split(a);
    let (b_recv, b_send) = tokio::io::split(b);
    let host = Arc::new(DuplexInputChannel {
        send: Mutex::new(a_send),
        recv: Mutex::new(a_recv),
    });
    let client = Arc::new(DuplexInputChannel {
        send: Mutex::new(b_send),
        recv: Mutex::new(b_recv),
    });
    (host, client)
}

#[async_trait]
impl InputChannel for DuplexInputChannel {
    async fn send_input(&self, evt: &InputEvent) -> Result<()> {
        let mut s = self.send.lock().await;
        write_framed(&mut *s, evt).await
    }

    async fn recv_input(&self) -> Result<InputEvent> {
        let mut r = self.recv.lock().await;
        read_framed(&mut *r).await
    }
}

// ---------------------------------------------------------------------
// VideoChannel: mpsc<Datagram> for unreliable, mpsc<VideoPacket> for IDR
// ---------------------------------------------------------------------

/// One end of an in-memory video channel pair.
///
/// Two separate transport directions modeled by two pairs of mpsc
/// channels:
/// - **Datagrams**: bounded `mpsc::channel<Datagram>`; `send_datagram`
///   silently drops on overflow (matches production: quinn's UDP queue
///   can overflow without surfacing through `Result`) and bumps the
///   `datagrams_dropped` counter for tests.
/// - **IDR uni-streams**: unbounded `mpsc::unbounded_channel<VideoPacket>`;
///   matches production's QUIC reliable-stream flow-control behavior
///   (stall the sender rather than drop).
///
/// Wire serialization uses the production `tether_protocol::encode` /
/// `decode` round-trip via `Datagram::Video(packet)` so payload-format
/// bugs in production are reproducible against the fake.
pub struct DuplexVideoChannel {
    dgram_tx: mpsc::Sender<Datagram>,
    dgram_rx: Mutex<mpsc::Receiver<Datagram>>,
    keyframe_tx: mpsc::UnboundedSender<VideoPacket>,
    keyframe_rx: Mutex<mpsc::UnboundedReceiver<VideoPacket>>,
    datagrams_dropped: std::sync::atomic::AtomicU64,
}

impl DuplexVideoChannel {
    /// Number of datagrams `send_datagram` has silently dropped due to
    /// the bounded channel being full. Cumulative since construction.
    /// Tests against backpressure / lossy-channel scenarios consult
    /// this rather than relying on a `Result::Err` (which production
    /// also never surfaces for queue overflow).
    pub fn datagrams_dropped(&self) -> u64 {
        self.datagrams_dropped
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Build a connected pair of [`DuplexVideoChannel`]s. Returns
/// `(host_side, client_side)`.
///
/// Datagram channel depth defaults to [`DEFAULT_VIDEO_DATAGRAM_DEPTH`].
/// For backpressure-specific tests, use [`video_duplex_pair_with_depth`].
#[must_use]
pub fn video_duplex_pair() -> (Arc<DuplexVideoChannel>, Arc<DuplexVideoChannel>) {
    video_duplex_pair_with_depth(DEFAULT_VIDEO_DATAGRAM_DEPTH)
}

#[must_use]
pub fn video_duplex_pair_with_depth(
    depth: usize,
) -> (Arc<DuplexVideoChannel>, Arc<DuplexVideoChannel>) {
    // Host → client direction.
    let (h2c_dgram_tx, h2c_dgram_rx) = mpsc::channel::<Datagram>(depth);
    let (h2c_kf_tx, h2c_kf_rx) = mpsc::unbounded_channel::<VideoPacket>();
    // Client → host direction (cursor datagrams flow this way today;
    // keyframes are host → client only, but the unbounded mpsc costs
    // nothing to keep symmetric and avoids surprising callers).
    let (c2h_dgram_tx, c2h_dgram_rx) = mpsc::channel::<Datagram>(depth);
    let (c2h_kf_tx, c2h_kf_rx) = mpsc::unbounded_channel::<VideoPacket>();

    let host = Arc::new(DuplexVideoChannel {
        dgram_tx: h2c_dgram_tx,
        dgram_rx: Mutex::new(c2h_dgram_rx),
        keyframe_tx: h2c_kf_tx,
        keyframe_rx: Mutex::new(c2h_kf_rx),
        datagrams_dropped: Default::default(),
    });
    let client = Arc::new(DuplexVideoChannel {
        dgram_tx: c2h_dgram_tx,
        dgram_rx: Mutex::new(h2c_dgram_rx),
        keyframe_tx: c2h_kf_tx,
        keyframe_rx: Mutex::new(h2c_kf_rx),
        datagrams_dropped: Default::default(),
    });
    (host, client)
}

#[async_trait]
impl VideoChannel for DuplexVideoChannel {
    fn send_datagram(&self, d: &Datagram) -> Result<()> {
        // Production round-trip: encode→decode so payload bugs surface
        // the same way they would on the wire. The decode is paid by
        // the receiver in recv_datagram; here we only validate that
        // the value *can* serialize at all (matches production: the
        // real send_datagram also calls encode before quinn ingests).
        let bytes = tether_protocol::encode(d)?;
        let decoded: Datagram = tether_protocol::decode(&bytes)?;
        match self.dgram_tx.try_send(decoded) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                // Silently drop — matches production UDP-queue
                // overflow. Test observers consult `datagrams_dropped`.
                self.datagrams_dropped
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(_)) => Err(TransportError::StreamClosed),
        }
    }

    async fn recv_datagram(&self) -> Result<Datagram> {
        let mut rx = self.dgram_rx.lock().await;
        rx.recv().await.ok_or(TransportError::StreamClosed)
    }

    async fn send_video_keyframe(&self, packet: &VideoPacket) -> Result<()> {
        // Round-trip through the production serializer the same way
        // datagrams do.
        let bytes = tether_protocol::encode(packet)?;
        let decoded: VideoPacket = tether_protocol::decode(&bytes)?;
        self.keyframe_tx
            .send(decoded)
            .map_err(|_| TransportError::StreamClosed)
    }

    async fn accept_video_keyframe(&self) -> Result<VideoPacket> {
        let mut rx = self.keyframe_rx.lock().await;
        rx.recv().await.ok_or(TransportError::StreamClosed)
    }
}

// ---------------------------------------------------------------------
// LossyChannel: deterministic loss + reorder wrapper for VideoChannel
// ---------------------------------------------------------------------

/// Configuration for [`LossyChannel`].
#[derive(Clone, Debug)]
pub struct LossyConfig {
    /// Probability that any given `send_datagram` is dropped before
    /// reaching the wrapped channel. Range `[0.0, 1.0]`.
    pub drop_probability: f64,
    /// Number of in-flight datagrams the channel holds back before
    /// releasing the oldest. `0` disables reordering. A window of `N`
    /// means up to `N+1` datagrams' worth of arrival-order scrambling.
    pub reorder_window: usize,
    /// Seed for the deterministic RNG. Tests should hard-code a value
    /// so failures are reproducible.
    pub seed: u64,
}

impl Default for LossyConfig {
    fn default() -> Self {
        Self {
            drop_probability: 0.0,
            reorder_window: 0,
            seed: 0,
        }
    }
}

/// Why a particular send was discarded — surfaced through
/// [`LossyChannel::drop_log`] for failure-mode debuggability.
#[derive(Clone, Debug)]
pub enum DropReason {
    /// `drop_probability` rolled true and the datagram was discarded
    /// before reaching the inner channel.
    RandomLoss,
}

/// One datagram dropped by [`LossyChannel`].
#[derive(Clone, Debug)]
pub struct DropEvent {
    /// Sequence number assigned by the wrapper (monotonic from 0 on
    /// each side). Distinct from the protocol's `frame_seq` —
    /// `LossyChannel` doesn't peek inside the datagram body.
    pub send_seq: u64,
    pub reason: DropReason,
}

/// Wrap any [`VideoChannel`] in a configurable loss + reorder layer.
///
/// Only the **datagram path** is affected: `send_datagram` /
/// `recv_datagram` apply the configured loss and reorder. The IDR
/// uni-stream path (`send_video_keyframe` / `accept_video_keyframe`)
/// is reliable in production and stays untouched here — matches QUIC
/// reliable-stream semantics.
///
/// Drops are recorded in [`Self::drop_log`] as full [`DropEvent`]s so
/// a failing test can name the exact sequence numbers it dropped,
/// rather than just a count.
pub struct LossyChannel<V: VideoChannel + ?Sized> {
    inner: Arc<V>,
    // Sync `std::sync::Mutex` for state the sync `send_datagram` trait
    // method touches — RNG roll + drop-log push happen atomically
    // under one lock, so a panic between them can't desync the two.
    // tokio's `Mutex` would force `try_lock`+panic-on-contention in
    // the sync method, which is a footgun if any future caller
    // shares the channel between tasks.
    sync_state: std::sync::Mutex<SyncState>,
    config: LossyConfig,
    send_seq: std::sync::atomic::AtomicU64,
    // tokio Mutex on the reorder buffer because `recv_datagram` is
    // async and may need to `.await` an inner recv while holding the
    // buffer to keep ordering coherent.
    reorder_buf: Mutex<std::collections::VecDeque<Datagram>>,
}

struct SyncState {
    rng: rand::rngs::SmallRng,
    drop_log: Vec<DropEvent>,
}

impl<V: VideoChannel + ?Sized> LossyChannel<V> {
    pub fn new(inner: Arc<V>, config: LossyConfig) -> Self {
        use rand::SeedableRng;
        Self {
            inner,
            sync_state: std::sync::Mutex::new(SyncState {
                rng: rand::rngs::SmallRng::seed_from_u64(config.seed),
                drop_log: Vec::new(),
            }),
            config,
            send_seq: Default::default(),
            reorder_buf: Default::default(),
        }
    }

    /// Snapshot of every drop the wrapper has performed so far. Cheap
    /// because `DropEvent: Clone` and the vec is small in practice.
    pub fn drop_log(&self) -> Vec<DropEvent> {
        self.sync_state.lock().expect("LossyChannel mutex poisoned").drop_log.clone()
    }
}

#[async_trait]
impl<V: VideoChannel + ?Sized> VideoChannel for LossyChannel<V> {
    fn send_datagram(&self, d: &Datagram) -> Result<()> {
        // Atomic roll + log under one std::sync lock — no async needed
        // because the trait method is sync, and a single lock means a
        // panic between roll and log can't leave the two desynced.
        let seq = self
            .send_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let drop_now = {
            use rand::Rng;
            let mut s = self.sync_state.lock().expect("LossyChannel mutex poisoned");
            let roll = self.config.drop_probability > 0.0
                && s.rng.random::<f64>() < self.config.drop_probability;
            if roll {
                s.drop_log.push(DropEvent {
                    send_seq: seq,
                    reason: DropReason::RandomLoss,
                });
            }
            roll
        };
        if drop_now {
            return Ok(());
        }
        self.inner.send_datagram(d)
    }

    async fn recv_datagram(&self) -> Result<Datagram> {
        if self.config.reorder_window == 0 {
            return self.inner.recv_datagram().await;
        }
        // Fill the reorder buffer to capacity, then yield a random
        // element from it. Random selection over a bounded window is
        // a standard model for jitter-induced reordering (simnet /
        // dummynet call this "reorder with window N").
        let mut buf = self.reorder_buf.lock().await;
        while buf.len() <= self.config.reorder_window {
            let next = self.inner.recv_datagram().await?;
            buf.push_back(next);
        }
        let idx = {
            use rand::Rng;
            let mut s = self.sync_state.lock().expect("LossyChannel mutex poisoned");
            s.rng.random_range(0..buf.len())
        };
        buf.remove(idx).ok_or_else(|| {
            // Unreachable: idx is bounded by buf.len() above, and the
            // lock has been held continuously. Surface as StreamClosed
            // rather than panic if some future refactor breaks the
            // invariant.
            TransportError::StreamClosed
        })
    }

    async fn send_video_keyframe(&self, packet: &VideoPacket) -> Result<()> {
        // Reliable stream — pass through unchanged.
        self.inner.send_video_keyframe(packet).await
    }

    async fn accept_video_keyframe(&self) -> Result<VideoPacket> {
        self.inner.accept_video_keyframe().await
    }
}

// ---------------------------------------------------------------------
// Centralized minimal-valid wire constructors
// ---------------------------------------------------------------------

/// Minimal-valid `ClientHello` for tests that don't care about its
/// contents. Centralized here so when `ClientHelloV1` gains a field,
/// exactly one site updates and every fake-using test recompiles.
#[must_use]
pub fn empty_client_hello() -> ClientHello {
    use tether_protocol::control::{ClientHelloV1, CodecKind};
    ClientHello::V1(ClientHelloV1 {
        client_name: "test".into(),
        preferred_codecs: vec![CodecKind::H264],
        max_resolution: None,
        viewport: None,
        clock_probe_t0: MonoNanos::ZERO,
        extensions: Default::default(),
        resume_token: None,
    })
}

/// Minimal-valid `ServerHello` for tests. See [`empty_client_hello`].
#[must_use]
pub fn empty_server_hello() -> ServerHello {
    use tether_protocol::control::{
        ChromaSubsampling, CodecKind, ServerHelloV1, VideoColorSpec,
    };
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

#[cfg(test)]
mod tests {
    use super::*;
    use tether_protocol::control::GoodbyeCode;
    use tether_protocol::input::{HidUsage, InputEvent, InputEventKind, Modifiers};
    use tether_protocol::video::{FrameFragmenter, VideoFrameMeta};
    use tokio::time::Duration;

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

    fn make_input_event(event_id: u64) -> InputEvent {
        InputEvent {
            event_id,
            t_client: MonoNanos::ZERO,
            device_id: 0,
            kind: InputEventKind::KeyDown {
                key: HidUsage(0x07_0004),
                modifiers: Modifiers::default(),
            },
        }
    }

    #[tokio::test]
    async fn input_events_round_trip_preserving_order() {
        let (host, client) = input_duplex_pair();
        let client_for_send = client.clone();
        tokio::spawn(async move {
            for i in 0..5u64 {
                client_for_send.send_input(&make_input_event(i)).await.unwrap();
            }
        });
        for expected in 0..5u64 {
            let evt = tokio::time::timeout(Duration::from_secs(1), host.recv_input())
                .await
                .expect("timeout")
                .unwrap();
            assert_eq!(evt.event_id, expected, "input events arrive in order");
            assert!(matches!(evt.kind, InputEventKind::KeyDown { .. }));
        }
    }

    fn make_video_meta() -> VideoFrameMeta {
        VideoFrameMeta {
            dimensions: (16, 16),
            keyframe: false,
            timing: tether_protocol::video::HostFrameTiming::default(),
            input_echo: Default::default(),
        }
    }

    #[tokio::test]
    async fn video_datagrams_round_trip_in_order() {
        let (host, client) = video_duplex_pair();
        let mut fragmenter = FrameFragmenter::new(0);
        let packets = fragmenter.fragment(make_video_meta(), b"hello".to_vec().into());
        let dgram = Datagram::Video(packets.into_iter().next().unwrap());
        host.send_datagram(&dgram).unwrap();
        let received = tokio::time::timeout(Duration::from_secs(1), client.recv_datagram())
            .await
            .expect("timeout")
            .unwrap();
        assert!(matches!(received, Datagram::Video(_)));
        assert_eq!(host.datagrams_dropped(), 0);
    }

    #[tokio::test]
    async fn video_datagram_full_channel_drops_silently() {
        let (host, _client) = video_duplex_pair_with_depth(1);
        let mut fragmenter = FrameFragmenter::new(0);
        let pkt = fragmenter
            .fragment(make_video_meta(), b"x".to_vec().into())
            .into_iter()
            .next()
            .unwrap();
        let dgram = Datagram::Video(pkt);
        // First send fills the slot.
        host.send_datagram(&dgram).unwrap();
        // Second send is silently dropped; production semantics
        // matched (queue overflow doesn't surface through Result).
        host.send_datagram(&dgram).unwrap();
        host.send_datagram(&dgram).unwrap();
        assert_eq!(host.datagrams_dropped(), 2);
    }

    #[tokio::test]
    async fn video_keyframe_uni_stream_round_trips() {
        let (host, client) = video_duplex_pair();
        let mut fragmenter = FrameFragmenter::new(0);
        let pkt = fragmenter.single_packet(
            VideoFrameMeta {
                dimensions: (16, 16),
                keyframe: true,
                timing: tether_protocol::video::HostFrameTiming::default(),
            input_echo: Default::default(),
            },
            b"idr-body".to_vec().into(),
        );
        host.send_video_keyframe(&pkt).await.unwrap();
        let received = tokio::time::timeout(Duration::from_secs(1), client.accept_video_keyframe())
            .await
            .expect("timeout")
            .unwrap();
        match received {
            VideoPacket::First { payload, .. } => assert_eq!(payload.as_ref(), b"idr-body"),
            other => panic!("expected First, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn lossy_channel_drops_datagrams_at_configured_rate() {
        let (host, _client) = video_duplex_pair();
        let lossy = LossyChannel::new(
            host.clone(),
            LossyConfig {
                drop_probability: 1.0, // guaranteed drop
                reorder_window: 0,
                seed: 42,
            },
        );
        let mut fragmenter = FrameFragmenter::new(0);
        for _ in 0..10 {
            let pkt = fragmenter
                .fragment(make_video_meta(), b"x".to_vec().into())
                .into_iter()
                .next()
                .unwrap();
            lossy.send_datagram(&Datagram::Video(pkt)).unwrap();
        }
        // Every send rolled true → all dropped.
        assert_eq!(host.datagrams_dropped(), 0, "inner channel sees no traffic");
        let log = lossy.drop_log();
        assert_eq!(log.len(), 10);
        assert!(matches!(log[0].reason, DropReason::RandomLoss));
        assert_eq!(log[0].send_seq, 0);
        assert_eq!(log[9].send_seq, 9);
    }

    #[tokio::test]
    async fn lossy_channel_zero_drop_passes_through() {
        let (host, client) = video_duplex_pair();
        let lossy = LossyChannel::new(host.clone(), LossyConfig::default());
        let mut fragmenter = FrameFragmenter::new(0);
        let pkt = fragmenter
            .fragment(make_video_meta(), b"hello".to_vec().into())
            .into_iter()
            .next()
            .unwrap();
        lossy.send_datagram(&Datagram::Video(pkt)).unwrap();
        let received = tokio::time::timeout(Duration::from_secs(1), client.recv_datagram())
            .await
            .expect("timeout")
            .unwrap();
        assert!(matches!(received, Datagram::Video(_)));
        assert!(lossy.drop_log().is_empty());
    }
}
