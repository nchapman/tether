use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use tether_protocol::control::{
    AudioSessionStats, SessionSummary, VideoProfile, VideoSessionStats,
};
use tracing::info;

pub struct SessionSummaryState {
    started: Instant,
    role: &'static str,
    codec: String,
    chroma: String,
    bit_depth: u32,
    audio_active: AtomicBool,
    pub video: VideoSummaryCounters,
    pub audio: AudioSummaryCounters,
}

impl SessionSummaryState {
    pub fn new(role: &'static str, profile: VideoProfile, audio_active: bool) -> Self {
        Self {
            started: Instant::now(),
            role,
            codec: format!("{:?}", profile.codec),
            chroma: format!("{:?}", profile.chroma),
            bit_depth: u32::from(profile.bit_depth),
            audio_active: AtomicBool::new(audio_active),
            video: VideoSummaryCounters::default(),
            audio: AudioSummaryCounters::default(),
        }
    }

    pub fn set_audio_active(&self, active: bool) {
        self.audio_active.store(active, Ordering::Relaxed);
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn snapshot(&self) -> SessionSummary {
        SessionSummary {
            role: self.role.to_string(),
            duration_ms: self.started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
            codec: self.codec.clone(),
            chroma: self.chroma.clone(),
            bit_depth: self.bit_depth,
            video: self.video.snapshot(),
            audio: self
                .audio_active
                .load(Ordering::Relaxed)
                .then(|| self.audio.snapshot()),
        }
    }
}

#[derive(Default)]
pub struct VideoSummaryCounters {
    pub frames_sent: AtomicU64,
    pub frames_received: AtomicU64,
    pub keyframes: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub bytes_received: AtomicU64,
    pub incomplete_frames: AtomicU64,
    pub fragment_loss_events: AtomicU64,
    pub decode_errors: AtomicU64,
    pub render_drop_frames: AtomicU64,
    pub idr_requests: AtomicU64,
    pub decode_queue_drop_frames: AtomicU64,
    pub transient_send_drop_frames: AtomicU64,
    pub fec_recovered_frames: AtomicU64,
    pub fec_recovered_fragments: AtomicU64,
    pub datagrams_sent: AtomicU64,
    pub parity_datagrams_sent: AtomicU64,
    pub max_datagrams_per_frame: AtomicU64,
    pub max_frame_bytes: AtomicU64,
    pub max_keyframe_bytes: AtomicU64,
    pub forced_idr_misses: AtomicU64,
}

impl VideoSummaryCounters {
    fn snapshot(&self) -> VideoSessionStats {
        VideoSessionStats {
            frames_sent: self.frames_sent.load(Ordering::Relaxed),
            frames_received: self.frames_received.load(Ordering::Relaxed),
            keyframes: self.keyframes.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            incomplete_frames: self.incomplete_frames.load(Ordering::Relaxed),
            fragment_loss_events: self.fragment_loss_events.load(Ordering::Relaxed),
            decode_errors: self.decode_errors.load(Ordering::Relaxed),
            render_drop_frames: self.render_drop_frames.load(Ordering::Relaxed),
            idr_requests: self.idr_requests.load(Ordering::Relaxed),
            decode_queue_drop_frames: self.decode_queue_drop_frames.load(Ordering::Relaxed),
            transient_send_drop_frames: self.transient_send_drop_frames.load(Ordering::Relaxed),
            fec_recovered_frames: self.fec_recovered_frames.load(Ordering::Relaxed),
            fec_recovered_fragments: self.fec_recovered_fragments.load(Ordering::Relaxed),
            datagrams_sent: self.datagrams_sent.load(Ordering::Relaxed),
            parity_datagrams_sent: self.parity_datagrams_sent.load(Ordering::Relaxed),
            max_datagrams_per_frame: self.max_datagrams_per_frame.load(Ordering::Relaxed),
            max_frame_bytes: self.max_frame_bytes.load(Ordering::Relaxed),
            max_keyframe_bytes: self.max_keyframe_bytes.load(Ordering::Relaxed),
            forced_idr_misses: self.forced_idr_misses.load(Ordering::Relaxed),
        }
    }
}

#[derive(Default)]
pub struct AudioSummaryCounters {
    pub packets_sent: AtomicU64,
    pub packets_received: AtomicU64,
    pub capture_frames: AtomicU64,
    pub underruns: AtomicU64,
    pub dropped_samples: AtomicU64,
    pub recovered_frames: AtomicU64,
    pub concealed_frames: AtomicU64,
    pub dropout_frames: AtomicU64,
    pub dropouts: AtomicU64,
    pub stale_packets: AtomicU64,
    pub decode_errors: AtomicU64,
}

impl AudioSummaryCounters {
    fn snapshot(&self) -> AudioSessionStats {
        AudioSessionStats {
            packets_sent: self.packets_sent.load(Ordering::Relaxed),
            packets_received: self.packets_received.load(Ordering::Relaxed),
            capture_frames: self.capture_frames.load(Ordering::Relaxed),
            underruns: self.underruns.load(Ordering::Relaxed),
            dropped_samples: self.dropped_samples.load(Ordering::Relaxed),
            recovered_frames: self.recovered_frames.load(Ordering::Relaxed),
            concealed_frames: self.concealed_frames.load(Ordering::Relaxed),
            dropout_frames: self.dropout_frames.load(Ordering::Relaxed),
            dropouts: self.dropouts.load(Ordering::Relaxed),
            stale_packets: self.stale_packets.load(Ordering::Relaxed),
            decode_errors: self.decode_errors.load(Ordering::Relaxed),
        }
    }
}

pub fn log_peer_session_summary(peer: &str, summary: Option<&SessionSummary>) {
    let Some(summary) = summary else { return };
    let audio = summary.audio.as_ref();
    info!(
        event = "peer_session_summary",
        peer,
        role = %summary.role,
        duration_ms = summary.duration_ms,
        codec = %summary.codec,
        chroma = %summary.chroma,
        bit_depth = summary.bit_depth,
        video_frames_sent = summary.video.frames_sent,
        video_frames_received = summary.video.frames_received,
        video_keyframes = summary.video.keyframes,
        video_bytes_sent = summary.video.bytes_sent,
        video_bytes_received = summary.video.bytes_received,
        video_incomplete_frames = summary.video.incomplete_frames,
        video_fragment_loss_events = summary.video.fragment_loss_events,
        video_decode_errors = summary.video.decode_errors,
        video_render_drop_frames = summary.video.render_drop_frames,
        video_idr_requests = summary.video.idr_requests,
        video_decode_queue_drop_frames = summary.video.decode_queue_drop_frames,
        video_transient_send_drop_frames = summary.video.transient_send_drop_frames,
        video_fec_recovered_frames = summary.video.fec_recovered_frames,
        video_fec_recovered_fragments = summary.video.fec_recovered_fragments,
        video_datagrams_sent = summary.video.datagrams_sent,
        video_parity_datagrams_sent = summary.video.parity_datagrams_sent,
        video_max_datagrams_per_frame = summary.video.max_datagrams_per_frame,
        video_max_frame_bytes = summary.video.max_frame_bytes,
        video_max_keyframe_bytes = summary.video.max_keyframe_bytes,
        video_forced_idr_misses = summary.video.forced_idr_misses,
        audio_packets_sent = audio.map_or(0, |s| s.packets_sent),
        audio_packets_received = audio.map_or(0, |s| s.packets_received),
        audio_capture_frames = audio.map_or(0, |s| s.capture_frames),
        audio_underruns = audio.map_or(0, |s| s.underruns),
        audio_dropped_samples = audio.map_or(0, |s| s.dropped_samples),
        audio_recovered_frames = audio.map_or(0, |s| s.recovered_frames),
        audio_concealed_frames = audio.map_or(0, |s| s.concealed_frames),
        audio_dropout_frames = audio.map_or(0, |s| s.dropout_frames),
        audio_dropouts = audio.map_or(0, |s| s.dropouts),
        audio_stale_packets = audio.map_or(0, |s| s.stale_packets),
        audio_decode_errors = audio.map_or(0, |s| s.decode_errors),
        "peer final session stats"
    );
}
