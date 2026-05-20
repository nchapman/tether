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

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

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

use crate::{
    CaptureError, CapturedDmaBuf, CapturedFrame, CpuFrame, GpuCapturedFrame, GpuCapturedGuard,
    GpuCapturedSource, PixelFormat, Result,
};

const CAPTURE_CHANNEL_DEPTH: usize = 2;

/// Run the portal handshake and spawn the PipeWire stream thread.
/// Returns a receiver that emits one [`CapturedFrame`] per produced frame.
///
/// `dmabuf_modifiers` is the list of DRM format modifiers the downstream
/// GPU importer can consume — typically obtained from
/// `tether_gpuconvert::importable_dmabuf_modifiers(AR24)`. The set is
/// passed to PipeWire as a multi-modifier DMA-BUF format offer alongside
/// the SHM fallback pod. Empty list ⇒ disable DMA-BUF entirely, advertise
/// only SHM.
///
/// **Side effect:** triggers an xdg-desktop-portal permission dialog on
/// the user's desktop session. The call blocks (asynchronously) until the
/// user grants or denies access.
pub async fn start(dmabuf_modifiers: Vec<u64>) -> Result<Receiver<CapturedFrame>> {
    let (node_id, fd) = open_portal().await?;
    tracing::info!(
        node_id,
        dmabuf_modifiers = dmabuf_modifiers.len(),
        "portal handshake complete; spawning pipewire thread"
    );

    let (tx, rx) = bounded::<CapturedFrame>(CAPTURE_CHANNEL_DEPTH);
    std::thread::Builder::new()
        .name("tether-capture-pipewire".into())
        .spawn(move || {
            if let Err(e) = run_pipewire(node_id, fd, tx, dmabuf_modifiers) {
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
    /// `Some(modifier)` once `param_changed` has seen a fixated
    /// modifier — i.e. the compositor accepted our DMA-BUF offer.
    /// `None` means SHM was negotiated (or negotiation hasn't
    /// completed yet); the process callback uses this to know which
    /// CapturedFrame variant to emit.
    negotiated_modifier: Option<u64>,
}

fn run_pipewire(
    node_id: u32,
    fd: OwnedFd,
    sender: Sender<CapturedFrame>,
    dmabuf_modifiers: Vec<u64>,
) -> Result<()> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_fd(fd, None)?;

    let user_data = UserData {
        format: spa::param::video::VideoInfoRaw::new(),
        sender,
        negotiated_modifier: None,
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
        .param_changed(|stream, user_data, id, param| {
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
            // Detect whether the compositor accepted our DMA-BUF offer.
            // Per the SPA convention, a fixated `Format` pod carrying a
            // `VideoModifier` property means DMA-BUF was negotiated;
            // absence of the property means SHM. We can't rely on
            // `VideoInfoRaw::modifier()` alone (it's zero-initialised),
            // so we look at the pod itself.
            //
            // Raw spa-sys constant: pipewire-rs 0.10 exposes
            // `FormatProperties::VideoModifier`, whose `.as_raw()` is
            // the right Id key.
            let modifier_key = pw::spa::utils::Id(
                pw::spa::param::format::FormatProperties::VideoModifier.as_raw(),
            );
            let has_modifier = unsafe {
                // SAFETY: the negotiated Format pod is always an
                // Object pod per the libspa contract for SPA_PARAM_Format.
                // pipewire-rs doesn't have a safe accessor that goes
                // pod → object for this case in 0.10; we use the raw
                // helper.
                let ptr = param.as_raw_ptr() as *const libspa_sys::spa_pod;
                let prop = libspa_sys::spa_pod_find_prop(
                    ptr.cast(),
                    std::ptr::null(),
                    modifier_key.0,
                );
                !prop.is_null()
            };
            user_data.negotiated_modifier =
                has_modifier.then_some(user_data.format.modifier());
            let f = &user_data.format;
            tracing::info!(
                spa_format = ?f.format(),
                width = f.size().width,
                height = f.size().height,
                fps_num = f.framerate().num,
                fps_denom = f.framerate().denom,
                dmabuf = has_modifier,
                modifier = user_data.negotiated_modifier,
                "pipewire negotiated format"
            );

            // Tell PipeWire which buffer types we can sink. The bitmask
            // semantics come from spa/buffer/buffer.h:
            //   bit (1 << SPA_DATA_MemPtr) → CPU-mapped pointer (SHM)
            //   bit (1 << SPA_DATA_MemFd)  → CPU-mapped via memfd
            //   bit (1 << SPA_DATA_DmaBuf) → GPU DMA-BUF fd
            // We advertise MemPtr+MemFd always (SHM fallback path); add
            // DmaBuf only when modifier negotiation succeeded — sending
            // the DmaBuf bit without a fixated modifier would let the
            // compositor hand us an opaque-tiled buffer we can't import.
            let mut data_type_mask = (1u32
                << libspa_sys::SPA_DATA_MemPtr)
                | (1u32 << libspa_sys::SPA_DATA_MemFd);
            if has_modifier {
                data_type_mask |= 1u32 << libspa_sys::SPA_DATA_DmaBuf;
            }
            let buffers_param = match build_buffers_param(data_type_mask) {
                Ok(bytes) => bytes,
                Err(e) => {
                    tracing::warn!(error = %e, "failed to build ParamBuffers");
                    return;
                }
            };
            let Some(pod) = spa::pod::Pod::from_bytes(&buffers_param) else {
                tracing::warn!("ParamBuffers pod from_bytes returned None");
                return;
            };
            let mut params = [pod];
            if let Err(e) = stream.update_params(&mut params) {
                tracing::warn!(error = %e, "pw_stream_update_params (Buffers) failed");
            }
        })
        .process(|stream, user_data| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            // Sample userspace clock as early as possible — anything later
            // (memcpy, allocation) would fold into the capture-latency
            // metric. t_capture_kernel stays equal to this until we read
            // it out of MetaHeader::pts.
            let t = MonoNanos::now();
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let data = &mut datas[0];

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

            let frame = match data.type_() {
                pw::spa::buffer::DataType::DmaBuf => {
                    match build_dmabuf_frame(data, user_data, width, height, t) {
                        Ok(f) => f,
                        Err(e) => {
                            tracing::warn!(error = %e, "DMA-BUF frame build failed");
                            return;
                        }
                    }
                }
                pw::spa::buffer::DataType::MemPtr
                | pw::spa::buffer::DataType::MemFd => {
                    match build_cpu_frame(data, user_data, width, height, t) {
                        Ok(f) => f,
                        Err(e) => {
                            tracing::warn!(error = %e, "CPU frame build failed");
                            return;
                        }
                    }
                }
                other => {
                    tracing::warn!(data_type = ?other, "unsupported buffer type");
                    return;
                }
            };

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

    // Offer DMA-BUF and SHM as separate EnumFormat alternatives. The
    // compositor picks the first one it can satisfy; DMA-BUF comes
    // first so a capable compositor takes the zero-copy path, and SHM
    // is the runtime fallback. param_changed observes which got fixated
    // and updates ParamBuffers accordingly.
    let shm_pod_bytes = build_format_pod(&[])?;
    let shm_pod = spa::pod::Pod::from_bytes(&shm_pod_bytes)
        .ok_or_else(|| CaptureError::PipeWire("shm pod from_bytes returned None".into()))?;
    let dmabuf_pod_bytes = if dmabuf_modifiers.is_empty() {
        None
    } else {
        Some(build_format_pod(&dmabuf_modifiers)?)
    };
    let mut params_storage: Vec<&spa::pod::Pod> = Vec::with_capacity(2);
    if let Some(ref bytes) = dmabuf_pod_bytes {
        params_storage.push(spa::pod::Pod::from_bytes(bytes).ok_or_else(|| {
            CaptureError::PipeWire("dmabuf pod from_bytes returned None".into())
        })?);
    }
    params_storage.push(shm_pod);

    stream.connect(
        spa::utils::Direction::Input,
        Some(node_id),
        pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
        &mut params_storage[..],
    )?;

    tracing::info!("pipewire stream connected; entering main loop");
    mainloop.run();
    Ok(())
}

/// Build one Format pod. Non-empty `modifiers` produces the DMA-BUF
/// variant (BGRx/BGRA only, VideoModifier prop carrying every supported
/// DRM modifier the GPU importer can consume); empty produces the SHM
/// variant (all four 4-byte SPA formats, no modifier prop).
///
/// Size range 1x1 to 7680x4320 (8K), framerate 0..=240 fps. Built with
/// pipewire-rs's object!/property! macros; the modifier property needs a
/// manual `Property` because the macro doesn't expose property flags,
/// and the prop is required to carry both MANDATORY and DONT_FIXATE —
/// Mutter and KWin treat absence of DONT_FIXATE as the legacy single-
/// modifier path and silently fall through to SHM.
fn build_format_pod(modifiers: &[u64]) -> Result<Vec<u8>> {
    let want_dmabuf = !modifiers.is_empty();
    let mut properties = vec![
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
    ];
    if want_dmabuf {
        properties.push(pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::VideoFormat,
            Choice,
            Enum,
            Id,
            pw::spa::param::video::VideoFormat::BGRx,
            pw::spa::param::video::VideoFormat::BGRx,
            pw::spa::param::video::VideoFormat::BGRA,
        ));
    } else {
        properties.push(pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::VideoFormat,
            Choice,
            Enum,
            Id,
            pw::spa::param::video::VideoFormat::BGRx,
            pw::spa::param::video::VideoFormat::BGRx,
            pw::spa::param::video::VideoFormat::BGRA,
            pw::spa::param::video::VideoFormat::RGBx,
            pw::spa::param::video::VideoFormat::RGBA,
        ));
    }
    properties.push(pw::spa::pod::property!(
        pw::spa::param::format::FormatProperties::VideoSize,
        Choice,
        Range,
        Rectangle,
        pw::spa::utils::Rectangle { width: 1920, height: 1080 },
        pw::spa::utils::Rectangle { width: 1, height: 1 },
        pw::spa::utils::Rectangle { width: 7680, height: 4320 }
    ));
    properties.push(pw::spa::pod::property!(
        pw::spa::param::format::FormatProperties::VideoFramerate,
        Choice,
        Range,
        Fraction,
        pw::spa::utils::Fraction { num: 60, denom: 1 },
        pw::spa::utils::Fraction { num: 0, denom: 1 },
        pw::spa::utils::Fraction { num: 240, denom: 1 }
    ));
    if want_dmabuf {
        // VideoModifier as a multi-element Long enum lets the compositor
        // pick whichever of our advertised modifiers it can satisfy.
        // SPA's Property type doesn't have a Long wrapper in
        // libspa::utils (i64 is the value type directly), so the macro
        // can't build this — hand-built `Property` it is.
        //
        // The `default` slot is what fixated negotiation falls back to
        // when neither side states a preference; we use the first
        // entry. Caller is expected to pass LINEAR first if it's in the
        // set so plain compositors don't gravitate to a tiled modifier
        // unnecessarily.
        let alternatives: Vec<i64> = modifiers
            .iter()
            .map(|m| i64::from_ne_bytes(m.to_ne_bytes()))
            .collect();
        let default = alternatives[0];
        properties.push(pw::spa::pod::Property {
            key: pw::spa::param::format::FormatProperties::VideoModifier.as_raw(),
            flags: pw::spa::pod::PropertyFlags::MANDATORY
                | pw::spa::pod::PropertyFlags::DONT_FIXATE,
            value: pw::spa::pod::Value::Choice(pw::spa::pod::ChoiceValue::Long(
                pw::spa::utils::Choice::<i64>(
                    pw::spa::utils::ChoiceFlags::empty(),
                    pw::spa::utils::ChoiceEnum::<i64>::Enum {
                        default,
                        alternatives,
                    },
                ),
            )),
        });
    }
    let obj = pw::spa::pod::Object {
        type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: pw::spa::param::ParamType::EnumFormat.as_raw(),
        properties,
    };
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

/// SPA-format → DRM fourcc mapping for DMA-BUF capture. The downstream
/// importer (wgpu's `texture_from_dmabuf_fd` + tether-gpuconvert) only
/// supports `Bgra8Unorm`, so we map the BGR-family SPA formats to the
/// matching DRM fourcc. RGB-family formats are rejected at the DMA-BUF
/// path; if a compositor only emits those, capture falls back to SHM
/// via the existing CPU path.
///
/// Fourcc values are little-endian 4-char codes from `<drm/drm_fourcc.h>`:
///   AR24 = DRM_FORMAT_ARGB8888  (32-bit ARGB, alpha in high byte) — BGRA in memory
///   XR24 = DRM_FORMAT_XRGB8888  (32-bit RGB, X in high byte)       — BGRx in memory
fn spa_format_to_drm_fourcc(spa: pw::spa::param::video::VideoFormat) -> Option<u32> {
    use pw::spa::param::video::VideoFormat as V;
    Some(match spa {
        V::BGRA => u32::from_le_bytes(*b"AR24"),
        V::BGRx => u32::from_le_bytes(*b"XR24"),
        _ => return None,
    })
}

fn build_cpu_frame(
    data: &mut pw::spa::buffer::Data,
    user_data: &UserData,
    width: u32,
    height: u32,
    t: MonoNanos,
) -> std::result::Result<CapturedFrame, String> {
    let Some(pixel_format) = spa_format_to_ours(user_data.format.format()) else {
        return Err(format!(
            "unsupported SPA pixel format on SHM path: {:?}",
            user_data.format.format()
        ));
    };

    let chunk = data.chunk();
    // stride is i32; reject negative (bottom-up) since the rest of the
    // pipeline assumes top-down.
    let stride = usize::try_from(chunk.stride())
        .map_err(|_| format!("negative stride {} (flipped frame)", chunk.stride()))?;
    let offset = chunk.offset() as usize;
    let chunk_size = chunk.size() as usize;
    let row_bytes = (width as usize) * 4;
    let needed = offset
        .saturating_add((height as usize).saturating_sub(1) * stride)
        .saturating_add(row_bytes);

    let Some(bytes) = data.data() else {
        // SHM path expects mapped CPU memory. If MAP_BUFFERS is set on
        // the stream connect (it is), the only way data() returns None
        // is a libpipewire-internal mapping failure — extremely rare.
        return Err("SHM buffer has no mapped memory".into());
    };
    if bytes.len() < needed {
        return Err(format!(
            "buffer too small: {} bytes, need {}",
            bytes.len(),
            needed
        ));
    }
    if chunk_size < needed.saturating_sub(offset) {
        return Err(format!(
            "chunk too small: {} bytes, need {}",
            chunk_size,
            needed.saturating_sub(offset)
        ));
    }

    // Pack rows tightly (compositor may pad stride > width*4).
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

    Ok(CapturedFrame::Cpu(CpuFrame {
        width,
        height,
        format: pixel_format,
        data: packed,
        t_capture_kernel: t,
        t_capture_userspace: t,
    }))
}

fn build_dmabuf_frame(
    data: &mut pw::spa::buffer::Data,
    user_data: &UserData,
    width: u32,
    height: u32,
    t: MonoNanos,
) -> std::result::Result<CapturedFrame, String> {
    let Some(modifier) = user_data.negotiated_modifier else {
        return Err("DMA-BUF buffer arrived but no modifier was negotiated".into());
    };
    let Some(fourcc) = spa_format_to_drm_fourcc(user_data.format.format()) else {
        return Err(format!(
            "unsupported SPA pixel format on DMA-BUF path: {:?}",
            user_data.format.format()
        ));
    };

    let chunk = data.chunk();
    let stride = u64::try_from(chunk.stride())
        .map_err(|_| format!("negative DMA-BUF stride {}", chunk.stride()))?;
    let offset = u64::from(chunk.offset());

    let raw_fd = data.fd();
    if raw_fd < 0 {
        return Err(format!("invalid DMA-BUF fd {raw_fd}"));
    }
    // Dup the fd so we own a reference independent of PipeWire's
    // buffer slot. The Buffer drops at the end of the callback,
    // queuing the slot back to PipeWire — at which point the
    // compositor may reuse the slot's *memory* for the next frame.
    // Dup'ing keeps the dma-buf object refcounted alive for the
    // downstream importer (libva, wgpu).
    //
    // Trade-off: the *memory contents* can still be overwritten if
    // PipeWire cycles back to the same slot before our consumer
    // reads. With the default 4-buffer pool, that's ~60 ms of
    // headroom at 60 fps — comfortably wider than encode latency.
    // The tight fix (delay queue-back until consumer drops) needs
    // raw pw_stream_queue_buffer calls; deferred until we measure
    // tearing in practice.
    // SAFETY: raw_fd was just returned by libspa for a DMA-BUF
    // SPA_DATA_DmaBuf data block; libc::dup is sound on any open fd.
    let dup_fd = unsafe {
        let f = libc::dup(raw_fd.as_raw_fd());
        if f < 0 {
            return Err(format!(
                "dup(DMA-BUF fd) failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        OwnedFd::from_raw_fd(f)
    };

    Ok(CapturedFrame::Gpu(GpuCapturedFrame {
        width,
        height,
        source: GpuCapturedSource::DmaBuf(CapturedDmaBuf {
            fourcc,
            fd: dup_fd,
            stride,
            offset,
            modifier,
        }),
        t_capture_kernel: t,
        t_capture_userspace: t,
        // No live PipeWire ref in the guard — see the dup() comment
        // above. A future refactor that holds the Buffer past callback
        // exit can stash it here.
        release_guard: GpuCapturedGuard::new(()),
    }))
}

/// Build a `SPA_TYPE_OBJECT_ParamBuffers` pod announcing which buffer
/// types we can consume. Called from `param_changed` once we know
/// whether DMA-BUF was negotiated.
fn build_buffers_param(data_type_mask: u32) -> Result<Vec<u8>> {
    // spa_sys::SPA_PARAM_BUFFERS_dataType is the property key. Wrapped
    // in libspa::utils::Id for the property! macro.
    let obj = pw::spa::pod::Object {
        type_: pw::spa::utils::SpaTypes::ObjectParamBuffers.as_raw(),
        id: pw::spa::param::ParamType::Buffers.as_raw(),
        properties: vec![pw::spa::pod::Property {
            key: libspa_sys::SPA_PARAM_BUFFERS_dataType,
            flags: pw::spa::pod::PropertyFlags::empty(),
            // SPA represents the buffer-type bitmask as an Int (i32)
            // per the historical struct layout in spa/param/buffers.h.
            value: pw::spa::pod::Value::Int(
                i32::try_from(data_type_mask).expect("buffer-type bits fit in i32"),
            ),
        }],
    };
    let bytes: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(obj),
    )
    .map_err(|e| CaptureError::PipeWire(format!("buffers pod serialize: {e:?}")))?
    .0
    .into_inner();
    Ok(bytes)
}
