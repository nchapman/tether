//! Video datagram format.

use crate::MonoNanos;
use serde::{Deserialize, Serialize};

/// Timing fields populated by the host as a frame moves through its pipeline.
/// All times in **host-local** [`MonoNanos`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostFrameTiming {
    pub t_capture_kernel: MonoNanos,
    pub t_capture_userspace: MonoNanos,
    pub t_encode_submit: MonoNanos,
    pub t_encode_done: MonoNanos,
    pub t_send: MonoNanos,
}

/// Echo of input events the host injected since the previous frame. Lets the
/// client compute true motion-to-photon latency (not just network RTT).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputEchoBatch {
    pub event_ids: Vec<u64>,
}

/// Per-frame metadata. Sent in fragment 0 of a video frame, not repeated
/// across continuation packets.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoFrameMeta {
    pub timing: HostFrameTiming,
    pub keyframe: bool,
    pub input_echo: InputEchoBatch,
}

/// A single video datagram. Frames larger than the transport's max datagram
/// size are sliced into multiple packets sharing
/// `(display, stream_epoch, frame_seq)`.
///
/// `display` identifies which host display the frame came from. v0 is
/// single-monitor and always uses `display = 0`, but the field is present
/// so multi-monitor support can land later as a pure additive change.
/// Each display gets its own encoder thread and therefore its own
/// `stream_epoch` + `frame_seq` counters (cribbed from RustDesk's
/// `video_threads: HashMap<usize, _>` pattern).
///
/// `stream_epoch` is `u16` (varint-encoded as 1 byte for typical values) so
/// a long-lived session that restarts the encoder thousands of times can't
/// collide with prior epochs. The host bumps `stream_epoch` whenever the
/// encoder is restarted (resize, codec switch, HW context loss). Clients
/// drop all packets from prior epochs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoPacket {
    First {
        display: u8,
        stream_epoch: u16,
        frame_seq: u32,
        fragment_count: u16,
        meta: VideoFrameMeta,
        payload: Vec<u8>,
    },
    Continuation {
        display: u8,
        stream_epoch: u16,
        frame_seq: u32,
        fragment_index: u16,
        payload: Vec<u8>,
    },
}
