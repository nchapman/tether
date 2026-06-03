//! Linux system-audio capture via PipeWire.
//!
//! Unlike the video path (which needs the screencast portal to get a DMA-BUF
//! node), capturing the system's audio *output* needs no portal: a PipeWire
//! capture stream tagged `stream.capture.sink = true` auto-connects to the
//! default sink's monitor, so we receive exactly "what's playing" as
//! interleaved float PCM. We request F32 at the codec's rate/channels and let
//! PipeWire's stream adapter resample/remix from whatever the sink runs at, so
//! the frames we emit always match the Opus encoder built for the same config.
//!
//! A dedicated thread runs the PipeWire main loop. A loop timer polls the stop
//! flag every 100 ms and quits the loop, so [`AudioCaptureHandle::stop`] (which
//! sets the flag then joins) never deadlocks against a blocked `run()`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::{bounded, Sender, TrySendError};
use pipewire as pw;
use pw::properties::properties;
use pw::spa;

use crate::capture::{AudioCaptureHandle, CaptureError};
use crate::{AudioFrame, OpusConfig};

/// ~80 ms of 10 ms frames. Capture should always be drained promptly; a short
/// queue keeps capture-side latency tight and drops the newest on overflow.
const CAPTURE_CHANNEL_DEPTH: usize = 8;

/// How often the loop timer checks the stop flag. Bounds teardown latency.
const STOP_POLL: Duration = Duration::from_millis(100);

/// Start a PipeWire monitor-capture stream on a dedicated thread.
pub fn start(cfg: OpusConfig) -> Result<AudioCaptureHandle, CaptureError> {
    let (tx, rx) = bounded::<AudioFrame>(CAPTURE_CHANNEL_DEPTH);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = Arc::clone(&stop);

    let thread = thread::Builder::new()
        .name("tether-audio-capture".into())
        .spawn(move || {
            if let Err(e) = run(cfg, tx, stop_thread) {
                tracing::error!(error = %e, "Linux audio capture ended with error");
            }
        })
        .map_err(|e| CaptureError::Backend(format!("spawn capture thread: {e}")))?;

    Ok(AudioCaptureHandle::from_parts(rx, stop, thread))
}

/// Per-stream state the listener callbacks read and mutate. PipeWire runs every
/// callback on the loop thread, so a plain `bool` (not an atomic) is enough for
/// the one-shot overflow log.
struct UserData {
    mainloop: pw::main_loop::MainLoopRc,
    tx: Sender<AudioFrame>,
    channels: u8,
    sample_rate: u32,
    format: spa::param::audio::AudioInfoRaw,
    /// One-shot rate-limit so a full channel logs once, not every callback.
    dropped: bool,
    logged_format: bool,
}

fn run(cfg: OpusConfig, tx: Sender<AudioFrame>, stop: Arc<AtomicBool>) -> Result<(), CaptureError> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)
        .map_err(|e| CaptureError::Backend(format!("create main loop: {e}")))?;
    let context = pw::context::ContextRc::new(&mainloop, None)
        .map_err(|e| CaptureError::Backend(format!("create context: {e}")))?;
    let core = context
        .connect(None)
        .map_err(|e| CaptureError::Backend(format!("connect to pipewire: {e}")))?;

    let user_data = UserData {
        mainloop: mainloop.clone(),
        tx,
        channels: cfg.channels,
        sample_rate: cfg.sample_rate,
        format: spa::param::audio::AudioInfoRaw::new(),
        dropped: false,
        logged_format: false,
    };

    // `stream.capture.sink = true` is what turns a plain capture stream into a
    // monitor of the default sink — i.e. system output rather than a mic.
    let stream = pw::stream::StreamBox::new(
        &core,
        "tether-audio-capture",
        properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            // "Production" is the recording/capture role; "Music" signals a
            // *playback* intent and would mislead WirePlumber's routing.
            *pw::keys::MEDIA_ROLE => "Production",
            *pw::keys::STREAM_CAPTURE_SINK => "true",
        },
    )
    .map_err(|e| CaptureError::Backend(format!("create stream: {e}")))?;

    let _listener = stream
        .add_local_listener_with_user_data(user_data)
        .state_changed(|_, user_data, old, new| {
            tracing::info!(?old, ?new, "pipewire audio stream state");
            // A terminal Error (missing/evicted sink, permission denied) never
            // recovers on its own; quit so the thread exits and the consumer
            // sees a disconnect instead of a silently dry channel forever.
            if let pw::stream::StreamState::Error(msg) = new {
                tracing::error!(error = %msg, "pipewire audio stream error; stopping capture");
                user_data.mainloop.quit();
            }
        })
        .param_changed(|_, user_data, id, param| {
            let Some(param) = param else { return };
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }
            let (media_type, media_subtype) = match spa::param::format_utils::parse_format(param) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(error = ?e, "failed to parse audio format pod");
                    return;
                }
            };
            if media_type != spa::param::format::MediaType::Audio
                || media_subtype != spa::param::format::MediaSubtype::Raw
            {
                return;
            }
            if let Err(e) = user_data.format.parse(param) {
                tracing::warn!(error = ?e, "failed to parse audio format");
                return;
            }
            // We request an exact format; the adapter converts to it, so the
            // negotiated rate/channels should equal what we asked for. Log the
            // first negotiation, and warn on *any* divergence — including a
            // later renegotiation — since the encoder is fixed at the requested
            // config and `process` interprets bytes with those same values, so a
            // divergence would corrupt frame framing.
            let (nr, nc) = (user_data.format.rate(), user_data.format.channels());
            let diverges = nr != user_data.sample_rate || nc != u32::from(user_data.channels);
            if !user_data.logged_format {
                user_data.logged_format = true;
                tracing::info!(
                    format = ?user_data.format.format(),
                    rate = nr,
                    channels = nc,
                    "pipewire negotiated audio format"
                );
            }
            if diverges {
                tracing::warn!(
                    requested_rate = user_data.sample_rate,
                    requested_channels = user_data.channels,
                    negotiated_rate = nr,
                    negotiated_channels = nc,
                    "pipewire audio format diverged from request; frames may be misframed"
                );
            }
        })
        .process(|stream, user_data| {
            // Process every queued buffer (don't keep-freshest like video):
            // audio is continuous, so dropping intermediate buffers would punch
            // gaps into the stream rather than just adding latency.
            while let Some(mut buffer) = stream.dequeue_buffer() {
                let datas = buffer.datas_mut();
                let Some(data) = datas.first_mut() else {
                    continue;
                };
                let chunk = data.chunk();
                let offset = chunk.offset() as usize;
                let size = chunk.size() as usize;
                if size == 0 {
                    continue;
                }
                let Some(bytes) = data.data() else {
                    continue;
                };
                let end = offset.saturating_add(size).min(bytes.len());
                if end <= offset {
                    continue;
                }
                let Some(frame) = frame_from_bytes(
                    &bytes[offset..end],
                    user_data.channels,
                    user_data.sample_rate,
                ) else {
                    continue;
                };
                match user_data.tx.try_send(frame) {
                    Ok(()) => user_data.dropped = false,
                    Err(TrySendError::Full(_)) => {
                        if !user_data.dropped {
                            user_data.dropped = true;
                            tracing::warn!(
                                "audio capture channel full; dropping frames (consumer behind)"
                            );
                        }
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        // Consumer gone (handle dropped). Quit so this thread
                        // exits and tears the stream down promptly.
                        tracing::info!("audio capture receiver dropped; quitting pipewire loop");
                        user_data.mainloop.quit();
                        return;
                    }
                }
            }
        })
        .register()
        .map_err(|e| CaptureError::Backend(format!("register stream listener: {e}")))?;

    // Poll the stop flag from inside the loop so `stop()` (flag + join) can break
    // a blocked `mainloop.run()` without a cross-thread quit.
    let stop_timer = Arc::clone(&stop);
    let mainloop_timer = mainloop.clone();
    // Kept alive until the end of `run`: dropping a `TimerSource` deregisters it
    // from the loop, which would silently disable the stop poll.
    let timer = mainloop.loop_().add_timer(move |_| {
        if stop_timer.load(Ordering::Relaxed) {
            mainloop_timer.quit();
        }
    });
    timer.update_timer(Some(STOP_POLL), Some(STOP_POLL));

    let pod_bytes = build_audio_format_pod(cfg)?;
    let pod = spa::pod::Pod::from_bytes(&pod_bytes)
        .ok_or_else(|| CaptureError::Backend("audio format pod from_bytes returned None".into()))?;
    let mut params = [pod];

    // No RT_PROCESS: our `process` callback allocates the frame Vec and sends on
    // a channel, so it isn't real-time-safe. Running it in the main-loop thread
    // (the default) is correct for a copy-out capture stream — same choice the
    // video capture path makes.
    stream
        .connect(
            spa::utils::Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            &mut params,
        )
        .map_err(|e| CaptureError::Backend(format!("connect stream: {e}")))?;

    tracing::info!(
        sample_rate = cfg.sample_rate,
        channels = cfg.channels,
        "Linux system audio capture started (PipeWire sink monitor)"
    );
    mainloop.run();
    Ok(())
}

/// Build the EnumFormat POD requesting interleaved F32 PCM at the codec's
/// rate/channels. `UNPOSITIONED` lets PipeWire pick the standard channel layout
/// (FL/FR for stereo) rather than us sending unknown positions.
fn build_audio_format_pod(cfg: OpusConfig) -> Result<Vec<u8>, CaptureError> {
    let mut info = spa::param::audio::AudioInfoRaw::new();
    info.set_format(spa::param::audio::AudioFormat::F32LE);
    info.set_rate(cfg.sample_rate);
    info.set_channels(u32::from(cfg.channels));
    info.set_flags(spa::param::audio::AudioInfoRawFlags::UNPOSITIONED);

    let obj = spa::pod::Object {
        type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: spa::param::ParamType::EnumFormat.as_raw(),
        properties: info.into(),
    };
    let bytes = spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(obj),
    )
    .map_err(|e| CaptureError::Backend(format!("serialize audio format pod: {e:?}")))?
    .0
    .into_inner();
    Ok(bytes)
}

/// Decode an interleaved little-endian f32 byte region into one [`AudioFrame`].
/// Truncates to a whole number of frames if the chunk ever ends mid-frame.
fn frame_from_bytes(region: &[u8], channels: u8, sample_rate: u32) -> Option<AudioFrame> {
    let bytes_per_frame = channels.max(1) as usize * 4;
    let usable = region.len() - (region.len() % bytes_per_frame);
    if usable == 0 {
        return None;
    }
    let samples: Vec<f32> = region[..usable]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Some(AudioFrame::new(sample_rate, channels, samples))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_from_bytes_interleaves_stereo() {
        // Two stereo frames: (L0,R0)=(1.0,-1.0), (L1,R1)=(0.5,0.25).
        let mut bytes = Vec::new();
        for v in [1.0f32, -1.0, 0.5, 0.25] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let frame = frame_from_bytes(&bytes, 2, 48_000).expect("frame");
        assert_eq!(frame.channels, 2);
        assert_eq!(frame.sample_rate, 48_000);
        assert_eq!(frame.samples, vec![1.0, -1.0, 0.5, 0.25]);
        assert_eq!(frame.frames(), 2);
    }

    #[test]
    fn frame_from_bytes_truncates_partial_trailing_frame() {
        // Three f32s for a stereo stream = one whole frame + a dangling sample.
        let mut bytes = Vec::new();
        for v in [1.0f32, 2.0, 3.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let frame = frame_from_bytes(&bytes, 2, 48_000).expect("frame");
        assert_eq!(frame.samples, vec![1.0, 2.0], "drops the half frame");
    }

    #[test]
    fn frame_from_bytes_empty_region_is_none() {
        assert!(frame_from_bytes(&[], 2, 48_000).is_none());
    }
}
