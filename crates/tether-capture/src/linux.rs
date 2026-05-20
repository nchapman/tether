//! Linux screen capture via the xdg-desktop-portal ScreenCast portal +
//! PipeWire.
//!
//! v0: CPU-side BGRA / BGRx readback only. DMA-BUF zero-copy into a VAAPI
//! surface lands later and will let the encoder consume the frame without
//! a memory copy.
//!
//! Calling [`start`] performs the portal handshake (which triggers a
//! permission dialog on the user's desktop) and then spawns a dedicated
//! thread running PipeWire's main loop. Frames are pushed into a bounded
//! crossbeam channel. The thread currently has no clean shutdown path
//! when the receiver is dropped; this is a documented v0 limitation.

use std::os::fd::OwnedFd;

use ashpd::desktop::{
    screencast::{CursorMode, Screencast, SelectSourcesOptions, SourceType},
    PersistMode,
};
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use enumflags2::BitFlags;
use pipewire as pw;
use pw::properties::properties;
use pw::spa;
use tether_protocol::MonoNanos;

use crate::{CaptureError, CapturedFrame, CpuFrame, PixelFormat, Result};

const CAPTURE_CHANNEL_DEPTH: usize = 2;

/// Run the portal handshake and spawn the PipeWire stream thread.
/// Returns a receiver that emits one [`CapturedFrame`] per produced frame.
///
/// **Side effect:** triggers an xdg-desktop-portal permission dialog on
/// the user's desktop session. The call blocks (asynchronously) until the
/// user grants or denies access.
pub async fn start() -> Result<Receiver<CapturedFrame>> {
    let (node_id, fd) = open_portal().await?;
    tracing::info!(node_id, "portal handshake complete; spawning pipewire thread");

    let (tx, rx) = bounded::<CapturedFrame>(CAPTURE_CHANNEL_DEPTH);
    std::thread::Builder::new()
        .name("tether-capture-pipewire".into())
        .spawn(move || {
            if let Err(e) = run_pipewire(node_id, fd, tx) {
                tracing::error!(error = %e, "pipewire capture thread failed");
            }
        })?;
    Ok(rx)
}

async fn open_portal() -> Result<(u32, OwnedFd)> {
    let proxy = Screencast::new().await?;
    let session = proxy.create_session(Default::default()).await?;
    proxy
        .select_sources(
            &session,
            SelectSourcesOptions::default()
                .set_cursor_mode(CursorMode::Embedded)
                .set_sources(BitFlags::from(SourceType::Monitor))
                .set_multiple(false)
                .set_restore_token(None)
                .set_persist_mode(PersistMode::DoNot),
        )
        .await?;
    let response = proxy
        .start(&session, None, Default::default())
        .await?
        .response()?;
    let stream = response
        .streams()
        .first()
        .ok_or_else(|| CaptureError::Portal("portal returned no streams".into()))?
        .to_owned();
    let node_id = stream.pipe_wire_node_id();
    let fd = proxy
        .open_pipe_wire_remote(&session, Default::default())
        .await?;
    Ok((node_id, fd))
}

struct UserData {
    format: spa::param::video::VideoInfoRaw,
    sender: Sender<CapturedFrame>,
    /// Set the first time we observe a buffer with no mapped CPU bytes
    /// (typically because the compositor negotiated DMA-BUF, which v0
    /// can't import). Used to log once at warn level and then stay quiet
    /// instead of spamming at frame rate.
    unmapped_buffer_warned: bool,
}

fn run_pipewire(node_id: u32, fd: OwnedFd, sender: Sender<CapturedFrame>) -> Result<()> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_fd(fd, None)?;

    let user_data = UserData {
        format: spa::param::video::VideoInfoRaw::new(),
        sender,
        unmapped_buffer_warned: false,
    };

    let stream = pw::stream::StreamBox::new(
        &core,
        "tether-capture",
        properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Screen",
        },
    )?;

    let _listener = stream
        .add_local_listener_with_user_data(user_data)
        .state_changed(|_, _, old, new| {
            tracing::info!(?old, ?new, "pipewire stream state");
        })
        .param_changed(|_, user_data, id, param| {
            let Some(param) = param else { return };
            if id != pw::spa::param::ParamType::Format.as_raw() {
                return;
            }
            let (media_type, media_subtype) =
                match pw::spa::param::format_utils::parse_format(param) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(error = ?e, "failed to parse format pod");
                        return;
                    }
                };
            if media_type != pw::spa::param::format::MediaType::Video
                || media_subtype != pw::spa::param::format::MediaSubtype::Raw
            {
                return;
            }
            if let Err(e) = user_data.format.parse(param) {
                tracing::warn!(error = ?e, "failed to parse video format");
                return;
            }
            let f = &user_data.format;
            tracing::info!(
                spa_format = ?f.format(),
                width = f.size().width,
                height = f.size().height,
                fps_num = f.framerate().num,
                fps_denom = f.framerate().denom,
                "pipewire negotiated format"
            );
        })
        .process(|stream, user_data| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            // Sample userspace clock as early as possible — anything later
            // (memcpy, allocation) would fold into the capture-latency
            // metric. t_capture_kernel stays equal to this until we read
            // it out of MetaHeader::pts. TODO(linux-capture): MetaHeader.
            let t = MonoNanos::now();
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let data = &mut datas[0];

            let Some(pixel_format) = spa_format_to_ours(user_data.format.format()) else {
                tracing::warn!(
                    spa_format = ?user_data.format.format(),
                    "unsupported pixel format; dropping frame"
                );
                return;
            };

            let width = user_data.format.size().width;
            let height = user_data.format.size().height;
            if width == 0 || height == 0 {
                return;
            }

            // Drop chunks the compositor flagged as corrupted rather than
            // forwarding garbage pixels downstream.
            let chunk_flags = data.chunk().flags();
            if chunk_flags.contains(pw::spa::buffer::ChunkFlags::CORRUPTED) {
                tracing::trace!(?chunk_flags, "pipewire chunk CORRUPTED; dropping");
                return;
            }

            let chunk = data.chunk();
            // stride is an i32 in libspa (negative strides are used for
            // bottom-up textures); we reject those for v0 since the rest
            // of the pipeline assumes top-down BGRA.
            let stride = match usize::try_from(chunk.stride()) {
                Ok(s) => s,
                Err(_) => {
                    tracing::warn!(
                        stride = chunk.stride(),
                        "negative stride from pipewire; v0 doesn't handle flipped frames"
                    );
                    return;
                }
            };
            let offset = chunk.offset() as usize;
            let chunk_size = chunk.size() as usize;
            let row_bytes = (width as usize) * 4;
            let needed = offset
                .saturating_add((height as usize).saturating_sub(1) * stride)
                .saturating_add(row_bytes);

            // `data.data()` returns None whenever the buffer isn't backed
            // by mapped CPU memory — most commonly DMA-BUF, which we
            // don't yet import. Without this branch we'd silently drop
            // every frame forever. Log once at warn, then trace, to avoid
            // filling the log at frame rate.
            let Some(bytes) = data.data() else {
                if !user_data.unmapped_buffer_warned {
                    user_data.unmapped_buffer_warned = true;
                    tracing::warn!(
                        data_type = ?data.type_(),
                        "pipewire buffer has no mapped CPU memory (likely DMA-BUF). \
                         v0 doesn't import DMA-BUF; all frames will be dropped \
                         until the compositor renegotiates. Restart capture or \
                         try a session that prefers SHM/MemFd."
                    );
                } else {
                    tracing::trace!(data_type = ?data.type_(), "skipping unmapped buffer");
                }
                return;
            };
            // Two independent constraints: bytes must be big enough for
            // the full plane, AND the chunk must mark the full plane as
            // valid data. The previous .min(...) made the check pass when
            // the chunk was *too small*, which could then panic in the
            // per-row indexing below.
            if bytes.len() < needed {
                tracing::warn!(
                    bytes_len = bytes.len(),
                    needed,
                    stride,
                    height,
                    "pipewire buffer smaller than declared frame; dropping"
                );
                return;
            }
            if chunk_size < needed.saturating_sub(offset) {
                tracing::warn!(
                    chunk_size,
                    needed_data = needed.saturating_sub(offset),
                    "pipewire chunk smaller than declared frame; dropping"
                );
                return;
            }

            // Pack rows tightly in case stride > width*4 (compositor padding).
            let mut packed = Vec::with_capacity(row_bytes * height as usize);
            if stride == row_bytes {
                let end = offset + row_bytes * height as usize;
                packed.extend_from_slice(&bytes[offset..end]);
            } else {
                for row in 0..height as usize {
                    let start = offset + row * stride;
                    packed.extend_from_slice(&bytes[start..start + row_bytes]);
                }
            }

            let frame = CapturedFrame::Cpu(CpuFrame {
                width,
                height,
                format: pixel_format,
                data: packed,
                t_capture_kernel: t,
                t_capture_userspace: t,
            });

            match user_data.sender.try_send(frame) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    tracing::trace!("capture consumer slow, dropping frame");
                }
                Err(TrySendError::Disconnected(_)) => {
                    // TODO(linux-capture): signal mainloop.quit() so the
                    // thread exits when the receiver drops. Requires
                    // cloning MainLoopRc into UserData. v0 lets the thread
                    // run until process exit.
                    tracing::debug!("capture receiver dropped; thread will exit on process exit");
                }
            }
        })
        .register()?;

    let params = build_format_pod()?;
    let mut params_slice = [spa::pod::Pod::from_bytes(&params)
        .ok_or_else(|| CaptureError::PipeWire("pod from_bytes returned None".into()))?];

    stream.connect(
        spa::utils::Direction::Input,
        Some(node_id),
        pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
        &mut params_slice,
    )?;

    tracing::info!("pipewire stream connected; entering main loop");
    mainloop.run();
    Ok(())
}

fn build_format_pod() -> Result<Vec<u8>> {
    // Offer BGRx / BGRA / RGBx / RGBA only — all 4-byte pixel formats so
    // the downstream code (raw send + future BGRA->NV12 conversion) has
    // a uniform shape. Size range is 1x1 to 7680x4320 (8K), framerate
    // range 0..=240 fps. PipeWire picks one matching what the compositor
    // can serve.
    let obj = pw::spa::pod::object!(
        pw::spa::utils::SpaTypes::ObjectParamFormat,
        pw::spa::param::ParamType::EnumFormat,
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::MediaType,
            Id,
            pw::spa::param::format::MediaType::Video
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::MediaSubtype,
            Id,
            pw::spa::param::format::MediaSubtype::Raw
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::VideoFormat,
            Choice,
            Enum,
            Id,
            pw::spa::param::video::VideoFormat::BGRx,
            pw::spa::param::video::VideoFormat::BGRx,
            pw::spa::param::video::VideoFormat::BGRA,
            pw::spa::param::video::VideoFormat::RGBx,
            pw::spa::param::video::VideoFormat::RGBA,
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::VideoSize,
            Choice,
            Range,
            Rectangle,
            pw::spa::utils::Rectangle {
                width: 1920,
                height: 1080
            },
            pw::spa::utils::Rectangle {
                width: 1,
                height: 1
            },
            pw::spa::utils::Rectangle {
                width: 7680,
                height: 4320
            }
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::VideoFramerate,
            Choice,
            Range,
            Fraction,
            pw::spa::utils::Fraction { num: 60, denom: 1 },
            pw::spa::utils::Fraction { num: 0, denom: 1 },
            pw::spa::utils::Fraction {
                num: 240,
                denom: 1
            }
        ),
    );
    let bytes: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(obj),
    )
    .map_err(|e| CaptureError::PipeWire(format!("pod serialize: {e:?}")))?
    .0
    .into_inner();
    Ok(bytes)
}

fn spa_format_to_ours(spa: pw::spa::param::video::VideoFormat) -> Option<PixelFormat> {
    use pw::spa::param::video::VideoFormat as V;
    Some(match spa {
        V::BGRA | V::BGRx => PixelFormat::Bgra8,
        V::RGBA | V::RGBx => PixelFormat::Rgba8,
        _ => return None,
    })
}
