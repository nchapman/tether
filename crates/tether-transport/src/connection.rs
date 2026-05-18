use std::net::SocketAddr;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use tether_protocol::{
    control::ControlMessage, cursor::CursorPacket, input::InputEvent, video::VideoPacket,
    MAX_DATAGRAM_PAYLOAD,
};

use crate::{Result, TransportError, MAX_FRAMED_MESSAGE};

/// Length of the per-stream preamble written by the client and consumed by
/// the server immediately after `open_*` / `accept_*`. The bytes themselves
/// are unused — the preamble exists only to force a STREAM frame onto the
/// wire so the peer's `accept_*` future completes.
pub(crate) const STREAM_PREAMBLE_LEN: usize = 4;
pub(crate) const STREAM_PREAMBLE: &[u8; STREAM_PREAMBLE_LEN] = &[0u8; STREAM_PREAMBLE_LEN];

/// What rides on a QUIC datagram. We multiplex video and cursor packets
/// onto the same unreliable channel — both prefer drop-on-loss over
/// retransmit. The enum discriminant adds one byte per datagram; the
/// receiver demuxes via `match`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Datagram {
    Video(VideoPacket),
    Cursor(CursorPacket),
}

/// An established QUIC connection between host and client.
///
/// Use [`Connection::send_input`] / [`Connection::recv_input`] only on the
/// side of the connection where the input stream is opened in that
/// direction — the other side returns
/// [`TransportError::InputStreamWrongRole`]. All other methods work on
/// both sides.
pub struct Connection {
    conn: quinn::Connection,
    control_send: Mutex<quinn::SendStream>,
    control_recv: Mutex<quinn::RecvStream>,
    input: Mutex<InputStream>,
}

enum InputStream {
    /// Client side: we write input events.
    Send(quinn::SendStream),
    /// Host side: we read input events.
    Recv(quinn::RecvStream),
}

impl Connection {
    pub(crate) fn new_client(
        conn: quinn::Connection,
        control_send: quinn::SendStream,
        control_recv: quinn::RecvStream,
        input_send: quinn::SendStream,
    ) -> Self {
        Self {
            conn,
            control_send: Mutex::new(control_send),
            control_recv: Mutex::new(control_recv),
            input: Mutex::new(InputStream::Send(input_send)),
        }
    }

    pub(crate) fn new_host(
        conn: quinn::Connection,
        control_send: quinn::SendStream,
        control_recv: quinn::RecvStream,
        input_recv: quinn::RecvStream,
    ) -> Self {
        Self {
            conn,
            control_send: Mutex::new(control_send),
            control_recv: Mutex::new(control_recv),
            input: Mutex::new(InputStream::Recv(input_recv)),
        }
    }

    pub fn rtt(&self) -> Duration {
        self.conn.rtt()
    }

    pub fn max_datagram_size(&self) -> Option<usize> {
        self.conn.max_datagram_size()
    }

    pub fn remote_address(&self) -> SocketAddr {
        self.conn.remote_address()
    }

    /// Send a datagram. Sync: quinn buffers datagrams without yielding,
    /// returning [`quinn::SendDatagramError`] if the queue is full or the
    /// peer doesn't support datagrams. Callers should react to the error
    /// (typically: drop the frame and increment a counter).
    pub fn send_datagram(&self, d: &Datagram) -> Result<()> {
        let bytes = tether_protocol::encode(d)?;
        if bytes.len() > MAX_DATAGRAM_PAYLOAD {
            return Err(TransportError::FrameTooLarge {
                size: bytes.len(),
                max: MAX_DATAGRAM_PAYLOAD,
            });
        }
        self.conn.send_datagram(bytes.into())?;
        Ok(())
    }

    pub async fn recv_datagram(&self) -> Result<Datagram> {
        let bytes = self.conn.read_datagram().await?;
        Ok(tether_protocol::decode(&bytes)?)
    }

    pub async fn send_control(&self, msg: &ControlMessage) -> Result<()> {
        let bytes = tether_protocol::encode(msg)?;
        let mut s = self.control_send.lock().await;
        write_framed(&mut s, &bytes).await
    }

    pub async fn recv_control(&self) -> Result<ControlMessage> {
        let mut r = self.control_recv.lock().await;
        let bytes = read_framed(&mut r).await?;
        Ok(tether_protocol::decode(&bytes)?)
    }

    pub async fn send_input(&self, evt: &InputEvent) -> Result<()> {
        let bytes = tether_protocol::encode(evt)?;
        let mut g = self.input.lock().await;
        let InputStream::Send(send) = &mut *g else {
            return Err(TransportError::InputStreamWrongRole);
        };
        write_framed(send, &bytes).await
    }

    pub async fn recv_input(&self) -> Result<InputEvent> {
        let mut g = self.input.lock().await;
        let InputStream::Recv(recv) = &mut *g else {
            return Err(TransportError::InputStreamWrongRole);
        };
        let bytes = read_framed(recv).await?;
        Ok(tether_protocol::decode(&bytes)?)
    }

    /// Gracefully close the underlying QUIC connection. Both sides should
    /// have sent a `ControlMessage::Goodbye` first.
    pub fn close(&self, code: u32, reason: &[u8]) {
        self.conn.close(code.into(), reason);
    }
}

impl std::fmt::Debug for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Connection")
            .field("remote", &self.remote_address())
            .field("rtt", &self.rtt())
            .finish_non_exhaustive()
    }
}

async fn write_framed(send: &mut quinn::SendStream, body: &[u8]) -> Result<()> {
    let len = u32::try_from(body.len()).map_err(|_| TransportError::FrameTooLarge {
        size: body.len(),
        max: u32::MAX as usize,
    })?;
    send.write_all(&len.to_le_bytes()).await?;
    send.write_all(body).await?;
    Ok(())
}

async fn read_framed(recv: &mut quinn::RecvStream) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    read_exact(recv, &mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_FRAMED_MESSAGE {
        return Err(TransportError::FrameTooLarge {
            size: len,
            max: MAX_FRAMED_MESSAGE,
        });
    }
    let mut buf = vec![0u8; len];
    read_exact(recv, &mut buf).await?;
    Ok(buf)
}

async fn read_exact(recv: &mut quinn::RecvStream, buf: &mut [u8]) -> Result<()> {
    recv.read_exact(buf)
        .await
        .map_err(TransportError::from_read_exact)
}
