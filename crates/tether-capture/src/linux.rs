//! Linux screen capture via the xdg-desktop-portal ScreenCast portal +
//! PipeWire.
//!
//! Frames arrive as either DMA-BUF (zero-copy into wgpu / VAAPI via the
//! [`crate::GpuCapturedSource::DmaBuf`] path) or CPU-side BGRA / BGRx
//! when the compositor falls back to SHM.
//!
//! Calling [`start`] performs the portal handshake (which triggers a
//! permission dialog on the user's desktop, unless a restore token
//! suppresses it) and then spawns a dedicated thread running PipeWire's
//! main loop. Frames are pushed into a bounded crossbeam channel. When
//! the frame receiver is dropped (the client disconnected and the host's
//! `handle_client` returned), the `process` callback quits the mainloop
//! via a `MainLoopRc` clone in `UserData`; the thread then exits and
//! drops the PipeWire stream + core + portal fd, ending the capture so
//! the compositor's "screen is being shared" indicator clears and no
//! session leaks across reconnects. The quit fires on the next buffer
//! after disconnect, so a fully idle stream lingers until its next
//! frame — fine in practice (capture is rarely perfectly static).

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use ashpd::desktop::{
    screencast::{CursorMode, Screencast, SelectSourcesOptions, SourceType},
    PersistMode, Session,
};
use std::sync::atomic::AtomicU32;
use std::sync::Arc;

use crossbeam_channel::{bounded, unbounded, Receiver as XReceiver, Sender, TrySendError};
use enumflags2::BitFlags;
use pipewire as pw;
use pw::properties::properties;
use pw::spa;
use std::sync::Mutex;
use tether_protocol::cursor::CursorPixelFormat;
use tether_protocol::MonoNanos;

use crate::cursor::{CursorEvent, CursorPosition, CursorShapeEvent, CursorSource};
use crate::{
    damage::NativeDamage, CaptureError, CaptureHandle, CapturedDmaBuf, CapturedFrame, CpuFrame,
    GpuCapturedFrame, GpuCapturedGuard, GpuCapturedSource, PixelFormat, Result,
};

/// Initial target FPS the Linux backend reports. Real cadence is
/// dictated by the PipeWire-negotiated framerate; the atomic is
/// surfaced so the ABR controller can request a renegotiation in the
/// future. Picked at 60 to match the typical compositor default; ABR
/// writes are accepted but currently log-only (see `CaptureHandle`
/// docs).
const LINUX_INITIAL_TARGET_FPS: u32 = 60;

// Single-slot hand-off: combined with the drain loop in the `process`
// callback, this keeps the freshest frame at every stage. A deeper
// channel would just bank latency for a consumer that's already behind.
const CAPTURE_CHANNEL_DEPTH: usize = 1;

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
pub async fn start(dmabuf_modifiers: Vec<u64>) -> Result<CaptureHandle> {
    let (node_id, fd, session) = open_portal().await?;
    tracing::info!(
        node_id,
        dmabuf_modifiers = dmabuf_modifiers.len(),
        "portal handshake complete; spawning pipewire thread"
    );

    // Hold the portal ScreenCast session alive and close it explicitly
    // when capture ends. Quitting the PipeWire mainloop stops the stream
    // but does NOT clear the compositor's "screen is being shared"
    // indicator — that's tied to the portal *session*, which lingers
    // until `Session::close` (or process exit, since the zbus connection
    // is shared process-wide). The PipeWire thread fires `close_tx` once
    // its mainloop exits; this task then closes the session on the tokio
    // runtime where the session's connection lives (closing from the
    // sync capture thread would have no reactor).
    let (close_tx, close_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        // `Err` means the sender dropped without signalling (capture
        // thread panicked before sending) — close anyway for cleanup.
        let _ = close_rx.await;
        match session.close().await {
            Ok(()) => tracing::info!("portal screencast session closed"),
            Err(e) => tracing::warn!(error = %e, "failed to close portal screencast session"),
        }
    });

    let (tx, rx) = bounded::<CapturedFrame>(CAPTURE_CHANNEL_DEPTH);
    // Cursor channels: shape changes are rare → unbounded crossbeam (a
    // misbehaving compositor can't blow memory because we dedup by id
    // at the parser); position is latest-wins via a shared snapshot.
    let (shape_tx, shape_rx) = unbounded::<CursorEvent>();
    let position_state = Arc::new(Mutex::new(None::<CursorPosition>));
    let position_state_for_cb = Arc::clone(&position_state);
    let target_fps = Arc::new(AtomicU32::new(LINUX_INITIAL_TARGET_FPS));
    // PipeWire-side FPS renegotiation is not yet wired; stash the
    // atomic on the handle so callers (the ABR controller) can
    // observe / set it, and document the no-op in `CaptureHandle`'s
    // crate-level doc. Threading the atomic into `run_pipewire`
    // proper lands when the format-renegotiation code does.
    std::thread::Builder::new()
        .name("tether-capture-pipewire".into())
        .spawn(move || {
            if let Err(e) = run_pipewire(
                node_id,
                fd,
                tx,
                dmabuf_modifiers,
                shape_tx,
                position_state_for_cb,
            ) {
                tracing::error!(error = %e, "pipewire capture thread failed");
            }
            // The mainloop has exited (receiver dropped → `quit()`, or an
            // early error above). Signal the closer task to tear down the
            // portal session. Send error = receiver already gone; nothing
            // to do.
            let _ = close_tx.send(());
        })?;
    let cursor_source: Box<dyn CursorSource> = Box::new(PipeWireCursorSource {
        shape_rx,
        position_state,
    });
    Ok(CaptureHandle::from_parts(rx, target_fps).with_cursor_source(cursor_source))
}

/// Receiver-side wiring for the PipeWire cursor extractor. The
/// producer side is the `param_changed`/`process` callback in
/// [`run_pipewire`]: it parses `SPA_META_Cursor` (plus the chained
/// `MetaBitmap` when present), dedup-hashes the pixel bytes for a
/// stable sprite id, and pushes events here.
///
/// Shape changes ride an unbounded crossbeam channel — real
/// compositors emit a handful of distinct sprites per session, and
/// the parser drops duplicates before sending. Position rides a
/// shared `Option<CursorPosition>` snapshot: PipeWire fires the
/// `process` callback at frame cadence (60-120 Hz), and the host
/// pump reads latest-wins.
pub struct PipeWireCursorSource {
    shape_rx: XReceiver<CursorEvent>,
    position_state: Arc<Mutex<Option<CursorPosition>>>,
}

#[cfg(test)]
impl PipeWireCursorSource {
    /// Test-only constructor that hands the producer side back to
    /// the caller. Lets a unit test exercise `next_event` /
    /// `poll_position` without spinning up a real PipeWire stream.
    fn for_test() -> (
        Self,
        Sender<CursorEvent>,
        Arc<Mutex<Option<CursorPosition>>>,
    ) {
        let (shape_tx, shape_rx) = unbounded::<CursorEvent>();
        let position_state = Arc::new(Mutex::new(None::<CursorPosition>));
        let source = Self {
            shape_rx,
            position_state: Arc::clone(&position_state),
        };
        (source, shape_tx, position_state)
    }
}

impl CursorSource for PipeWireCursorSource {
    fn next_event(&mut self) -> CursorEvent {
        self.shape_rx.try_recv().unwrap_or(CursorEvent::Idle)
    }

    fn poll_position(&mut self) -> Option<CursorPosition> {
        // Lock-poison is fatal: nothing in the producer thread
        // panics with the guard held in normal operation, and the
        // host has no useful "no cursor" fallback if the snapshot is
        // unreadable. Log and treat as transient None.
        self.position_state.lock().ok().and_then(|guard| *guard)
    }
}

/// Directory where we persist the ScreenCast restore token. Mirrors
/// the client's `$HOME/.tether` convention (see `tether-client`'s
/// `client_config_dir`); `TETHER_STATE_DIR` overrides it for tests and
/// sandboxed deployments.
fn state_dir() -> Option<std::path::PathBuf> {
    if let Some(dir) = std::env::var_os("TETHER_STATE_DIR") {
        return Some(std::path::PathBuf::from(dir));
    }
    std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".tether"))
}

fn restore_token_path() -> Option<std::path::PathBuf> {
    state_dir().map(|dir| dir.join("screencast-restore-token"))
}

/// Read the saved ScreenCast restore token, if any. Any error (no
/// state dir, missing file, unreadable) is treated as "no token" — the
/// portal then prompts and issues a fresh one.
fn load_restore_token() -> Option<String> {
    load_restore_token_from(&restore_token_path()?)
}

/// Persist the ScreenCast restore token for the next session. Best
/// effort: a failure just means we re-prompt next launch, so we log and
/// move on rather than failing the capture handshake.
fn save_restore_token(token: &str) {
    let Some(path) = restore_token_path() else {
        tracing::warn!("no state dir (HOME unset); cannot persist screencast restore token");
        return;
    };
    match save_restore_token_to(&path, token) {
        Ok(()) => tracing::info!(?path, "persisted screencast restore token"),
        Err(e) => tracing::warn!(error = %e, ?path, "failed to persist screencast restore token"),
    }
}

fn load_restore_token_from(path: &std::path::Path) -> Option<String> {
    let token = std::fs::read_to_string(path).ok()?;
    let token = token.trim();
    if token.is_empty() {
        None
    } else {
        Some(token.to_owned())
    }
}

fn save_restore_token_to(path: &std::path::Path, token: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // SECURITY: the restore token is a portal capability credential — a
    // co-user who reads it could potentially replay our share grant.
    // Owner-only, matching tether-pairing/tether-transport key files.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    // `mode(0o600)` above only applies when *creating* the file; if it
    // already existed with broader permissions, tighten it now so a
    // rotated token can't inherit a world-readable mode.
    file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    file.write_all(token.as_bytes())
}

/// Pick a portal cursor mode from the set the portal advertises.
///
/// The portal validates the requested cursor mode against its
/// `AvailableCursorModes` and rejects anything outside the set with
/// `InvalidArgument` ("Unavailable cursor mode N"), which fails the
/// whole session. So we only ever request a mode that is actually
/// advertised. Some backends report an *empty* set despite a recent
/// interface version (observed: xdg-desktop-portal-gnome interface v5
/// reporting 0 on GNOME/Wayland) — there we return `None` so the caller
/// omits the option and lets the portal fall back to its own default,
/// rather than forcing a mode it will reject.
///
/// Preference order: Metadata (cursor delivered as PipeWire stream meta,
/// so we can render it client-side) over Embedded (burned into the video
/// stream) over Hidden.
fn select_cursor_mode(available: BitFlags<CursorMode>) -> Option<CursorMode> {
    if available.contains(CursorMode::Metadata) {
        Some(CursorMode::Metadata)
    } else if available.contains(CursorMode::Embedded) {
        Some(CursorMode::Embedded)
    } else if available.contains(CursorMode::Hidden) {
        Some(CursorMode::Hidden)
    } else {
        None
    }
}

async fn open_portal() -> Result<(u32, OwnedFd, Session<Screencast>)> {
    let proxy = Screencast::new().await?;
    // Query supported cursor modes up front. GNOME 45+ and KDE
    // Plasma 6 both advertise Metadata, but legacy portals fall
    // back to Embedded only. If we request Metadata against a
    // portal that doesn't advertise it, the portal silently
    // accepts the call and uses Embedded — and we end up with a
    // burned-in cursor *and* no metadata, the worst of both
    // worlds.
    let available = proxy
        .available_cursor_modes()
        .await
        .map_err(|e| CaptureError::Portal(format!("AvailableCursorModes query: {e}")))?;
    let chosen_cursor_mode = select_cursor_mode(available);
    tracing::info!(
        ?available,
        ?chosen_cursor_mode,
        "portal cursor mode negotiation"
    );
    let session = proxy.create_session(Default::default()).await?;
    // Replay a previously-granted restore token so the compositor can
    // skip (or quietly confirm) the share dialog. `ExplicitlyRevoked`
    // asks the portal to remember the grant until the user revokes it
    // in their desktop's privacy settings — the behaviour unattended
    // remote access needs. Mirrors Sunshine / rustdesk server mode.
    let saved_token = load_restore_token();
    proxy
        .select_sources(
            &session,
            SelectSourcesOptions::default()
                .set_cursor_mode(chosen_cursor_mode)
                .set_sources(BitFlags::from(SourceType::Monitor))
                .set_multiple(false)
                .set_restore_token(saved_token.as_deref())
                .set_persist_mode(PersistMode::ExplicitlyRevoked),
        )
        .await?;
    let response = proxy
        .start(&session, None, Default::default())
        .await?
        .response()?;
    // The portal may hand back a fresh token on every Start (token
    // rotation), so persist whatever it returned, replacing the old
    // one. A missing token means the portal declined to persist (older
    // backend / user chose not to remember) — leave any prior token in
    // place rather than clobbering it. Save failures are non-fatal: the
    // session still works, we just re-prompt next launch.
    if let Some(token) = response.restore_token() {
        if Some(token) != saved_token.as_deref() {
            save_restore_token(token);
        }
    }
    let stream = response
        .streams()
        .first()
        .ok_or_else(|| CaptureError::Portal("portal returned no streams".into()))?
        .to_owned();
    let node_id = stream.pipe_wire_node_id();
    let fd = proxy
        .open_pipe_wire_remote(&session, Default::default())
        .await?;
    // Return the session (it owns its own zbus proxy/connection) so the
    // caller can close it on teardown; dropping it here would leak it on
    // the portal until process exit. The `proxy` can drop — `close` goes
    // through the session's own proxy.
    Ok((node_id, fd, session))
}

struct UserData {
    /// Clone of the capture thread's mainloop, so the `process`
    /// callback can `quit()` it when the frame receiver is gone (client
    /// disconnected). Without this the thread spins forever, leaking a
    /// PipeWire stream + portal screencast session per connection and
    /// keeping the compositor's "screen is being shared" indicator up.
    mainloop: pw::main_loop::MainLoopRc,
    format: spa::param::video::VideoInfoRaw,
    sender: Sender<CapturedFrame>,
    /// `Some(modifier)` once `param_changed` has seen a fixated
    /// modifier — i.e. the compositor accepted our DMA-BUF offer.
    /// `None` means SHM was negotiated (or negotiation hasn't
    /// completed yet); the process callback uses this to know which
    /// CapturedFrame variant to emit.
    negotiated_modifier: Option<u64>,
    /// Producer side of the cursor shape channel. Send-only here;
    /// receiver lives in [`PipeWireCursorSource`].
    cursor_shape_tx: Sender<CursorEvent>,
    /// Latest cursor position snapshot. The host pump reads this
    /// at frame cadence; producer writes on every process callback
    /// that sees a `SPA_META_Cursor`.
    cursor_position: Arc<Mutex<Option<CursorPosition>>>,
    /// Last sprite id forwarded as a `Shape` event. Used to dedup so
    /// a static cursor doesn't blow the unbounded shape channel.
    /// `0` is reserved by SPA as "no new cursor data," so we use
    /// `None` to mean "nothing emitted yet."
    last_shape_id: Option<u64>,
    /// Once-only logging guards. Tells us *which* leg of cursor-meta
    /// processing fired on the very first frame so a silent failure
    /// (compositor accepted CursorMode::Metadata but isn't actually
    /// attaching the meta) doesn't masquerade as "everything fine,
    /// just no movement."
    logged_first_cursor_meta: bool,
    logged_first_cursor_meta_absent: bool,
}

fn run_pipewire(
    node_id: u32,
    fd: OwnedFd,
    sender: Sender<CapturedFrame>,
    dmabuf_modifiers: Vec<u64>,
    cursor_shape_tx: Sender<CursorEvent>,
    cursor_position: Arc<Mutex<Option<CursorPosition>>>,
) -> Result<()> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_fd(fd, None)?;

    let user_data = UserData {
        mainloop: mainloop.clone(),
        format: spa::param::video::VideoInfoRaw::new(),
        sender,
        negotiated_modifier: None,
        cursor_shape_tx,
        cursor_position,
        last_shape_id: None,
        logged_first_cursor_meta: false,
        logged_first_cursor_meta_absent: false,
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
                let prop =
                    libspa_sys::spa_pod_find_prop(ptr.cast(), std::ptr::null(), modifier_key.0);
                !prop.is_null()
            };
            user_data.negotiated_modifier = has_modifier.then_some(user_data.format.modifier());
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
            let mut data_type_mask =
                (1u32 << libspa_sys::SPA_DATA_MemPtr) | (1u32 << libspa_sys::SPA_DATA_MemFd);
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
            // Resend the meta subscriptions inside `param_changed`,
            // not just at stream.connect. OBS Studio established this
            // pattern empirically: a number of compositors accept
            // CursorMode::Metadata in the portal call but only attach
            // SPA_META_Cursor to buffers if the meta is requested
            // *after* format negotiation completes. Sending the pods
            // at both points costs nothing extra (PipeWire dedups
            // by ParamType + meta::type), so we keep the connect-time
            // subscription as a defence-in-depth.
            let video_damage_param = match build_video_damage_meta_pod() {
                Ok(bytes) => bytes,
                Err(e) => {
                    tracing::warn!(error = %e, "failed to build VideoDamage meta pod");
                    return;
                }
            };
            let cursor_meta_param = match build_cursor_meta_pod() {
                Ok(bytes) => bytes,
                Err(e) => {
                    tracing::warn!(error = %e, "failed to build cursor meta pod");
                    return;
                }
            };
            let Some(buffers_pod) = spa::pod::Pod::from_bytes(&buffers_param) else {
                tracing::warn!("ParamBuffers pod from_bytes returned None");
                return;
            };
            let Some(damage_pod) = spa::pod::Pod::from_bytes(&video_damage_param) else {
                tracing::warn!("VideoDamage meta pod from_bytes returned None");
                return;
            };
            let Some(cursor_pod) = spa::pod::Pod::from_bytes(&cursor_meta_param) else {
                tracing::warn!("Cursor meta pod from_bytes returned None");
                return;
            };
            let mut params = [buffers_pod, damage_pod, cursor_pod];
            if let Err(e) = stream.update_params(&mut params) {
                tracing::warn!(error = %e, "pw_stream_update_params failed");
            }
        })
        .process(|stream, user_data| {
            // Drain the pending buffer queue and keep only the freshest.
            // Older buffers are dropped immediately; pipewire-rs' Buffer
            // RAII guard requeues them to PipeWire so the compositor can
            // refill. Without this, a slow consumer would have us keep
            // processing stale frames the compositor produced while we
            // were behind, racking up latency.
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            while let Some(newer) = stream.dequeue_buffer() {
                buffer = newer;
            }
            // Sample userspace clock as early as possible — anything later
            // (memcpy, allocation) would fold into the capture-latency
            // metric. t_capture_kernel stays equal to this until we read
            // it out of MetaHeader::pts.
            let t = MonoNanos::now();
            // Read SPA_META_VideoDamage + SPA_META_Cursor before
            // borrowing the data slice: `find_meta` takes &self while
            // `datas_mut` takes &mut self, and Rust's borrow checker
            // won't let them overlap.
            let native_damage = read_video_damage(&buffer);
            process_cursor_meta(&buffer, user_data);
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
                    match build_dmabuf_frame(data, user_data, width, height, t, native_damage) {
                        Ok(f) => f,
                        Err(e) => {
                            tracing::warn!(error = %e, "DMA-BUF frame build failed");
                            return;
                        }
                    }
                }
                pw::spa::buffer::DataType::MemPtr | pw::spa::buffer::DataType::MemFd => {
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
                    // Frame receiver dropped — the client disconnected and
                    // `handle_client` returned. Quit the mainloop so this
                    // thread exits and drops the PipeWire stream + core +
                    // portal fd, ending the capture (and clearing the
                    // compositor's share indicator) instead of leaking a
                    // session per connection.
                    tracing::info!("capture receiver dropped; quitting pipewire mainloop");
                    user_data.mainloop.quit();
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
    // Ask PipeWire to attach SPA_META_VideoDamage to each buffer. A
    // server that doesn't advertise the meta simply ignores the
    // request; the buffer callback's `find_meta` returns None in that
    // case and the consumer falls back to the hash classifier.
    let meta_pod_bytes = build_video_damage_meta_pod()?;
    let meta_pod = spa::pod::Pod::from_bytes(&meta_pod_bytes)
        .ok_or_else(|| CaptureError::PipeWire("meta pod from_bytes returned None".into()))?;
    // Same trick for SPA_META_Cursor — the meta carries pointer
    // position every buffer, plus a chained bitmap when the sprite
    // changes. A compositor that doesn't honour CursorMode::Metadata
    // simply omits the meta and we render no overlay (no regression
    // vs the embedded-cursor era — just no separately-rendered
    // pointer until the user moves to a meta-capable session).
    let cursor_meta_pod_bytes = build_cursor_meta_pod()?;
    let cursor_meta_pod = spa::pod::Pod::from_bytes(&cursor_meta_pod_bytes)
        .ok_or_else(|| CaptureError::PipeWire("cursor meta pod from_bytes returned None".into()))?;
    let mut params_storage: Vec<&spa::pod::Pod> = Vec::with_capacity(4);
    if let Some(ref bytes) = dmabuf_pod_bytes {
        params_storage.push(
            spa::pod::Pod::from_bytes(bytes).ok_or_else(|| {
                CaptureError::PipeWire("dmabuf pod from_bytes returned None".into())
            })?,
        );
    }
    params_storage.push(shm_pod);
    params_storage.push(meta_pod);
    params_storage.push(cursor_meta_pod);

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
/// Size range 1x1 to 7680x4320 (8K). Framerate: SPA Range with
/// default=60, min=0, max=240 — the compositor typically mirrors the
/// display refresh (60 Hz → 60 fps, 144 Hz → 144 fps clamped at our
/// max). 60 fps is the host's encoder target so that's the default
/// hint; compositors that can drive higher get to. The encoder
/// time_base is decoupled from the actual delivered rate (VBR
/// bitrate, GOP measured in frames) so frame-rate variation does not
/// stall the pipeline. Built with
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
        // SPA_META_VideoDamage isn't useful on the SHM fallback path:
        // the producer typically only attaches it to DMA-BUF buffers,
        // and even when present it'd race the memcpy we just did
        // above. The hash classifier on the CPU path is fast enough.
        native_damage: None,
    }))
}

fn build_dmabuf_frame(
    data: &mut pw::spa::buffer::Data,
    user_data: &UserData,
    width: u32,
    height: u32,
    t: MonoNanos,
    native_damage: Option<NativeDamage>,
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
        native_damage,
    }))
}

/// Snapshot the `SPA_META_VideoDamage` attachment, if any.
///
/// PipeWire's contract: when the meta block is present, an empty
/// region list means "no damage this frame" and a populated list
/// means damage was reported. Producers that don't attach the meta
/// at all yield `None`, which the consumer treats as "no opinion"
/// and falls back to the hash classifier.
///
/// Split from [`native_damage_from_region_count`] so the policy
/// (empty ⇒ idle) is unit-testable without faking PW's FFI region
/// type.
fn read_video_damage(buffer: &pw::buffer::Buffer<'_>) -> Option<NativeDamage> {
    let meta = buffer.find_meta::<pw::spa::buffer::meta::MetaVideoDamage>()?;
    Some(native_damage_from_region_count(meta.iter().count()))
}

/// Policy: empty damage region list ⇒ frame is idle.
fn native_damage_from_region_count(n: usize) -> NativeDamage {
    NativeDamage { idle: n == 0 }
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

/// Build the `SPA_TYPE_OBJECT_ParamMeta` pod that subscribes to the
/// `SPA_META_Cursor` block. The range is `CURSOR_META_SIZE(w, h) =
/// sizeof(spa_meta_cursor) + sizeof(spa_meta_bitmap) + w*h*4`
/// instantiated at OBS's published values (`min = 1×1`,
/// `default = 64×64`, `max = 1024×1024`).
///
/// Why match OBS exactly: a previous min of "header only, no bitmap"
/// told Mutter the consumer was happy without sprite pixels, and the
/// compositor silently dropped the entire meta attachment in
/// response. The OBS values are the only ones we've verified actually
/// produce a non-None `find_meta::<MetaCursor>()` on GNOME 45+ /
/// KDE Plasma 6.
fn build_cursor_meta_pod() -> Result<Vec<u8>> {
    let header = i32::try_from(
        std::mem::size_of::<libspa_sys::spa_meta_cursor>()
            + std::mem::size_of::<libspa_sys::spa_meta_bitmap>(),
    )
    .expect("cursor meta header size fits in i32");
    let cursor_size = |w: i32, h: i32| header + w * h * 4;
    let min_size = cursor_size(1, 1);
    let default_size = cursor_size(64, 64);
    let max_size = cursor_size(1024, 1024);
    let obj = pw::spa::pod::Object {
        type_: pw::spa::utils::SpaTypes::ObjectParamMeta.as_raw(),
        id: pw::spa::param::ParamType::Meta.as_raw(),
        properties: vec![
            pw::spa::pod::Property {
                key: libspa_sys::SPA_PARAM_META_type,
                flags: pw::spa::pod::PropertyFlags::empty(),
                value: pw::spa::pod::Value::Id(pw::spa::utils::Id(libspa_sys::SPA_META_Cursor)),
            },
            pw::spa::pod::Property {
                key: libspa_sys::SPA_PARAM_META_size,
                flags: pw::spa::pod::PropertyFlags::empty(),
                value: pw::spa::pod::Value::Choice(pw::spa::pod::ChoiceValue::Int(
                    pw::spa::utils::Choice::<i32>(
                        pw::spa::utils::ChoiceFlags::empty(),
                        pw::spa::utils::ChoiceEnum::<i32>::Range {
                            default: default_size,
                            min: min_size,
                            max: max_size,
                        },
                    ),
                )),
            },
        ],
    };
    let bytes: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(obj),
    )
    .map_err(|e| CaptureError::PipeWire(format!("cursor meta pod serialize: {e:?}")))?
    .0
    .into_inner();
    Ok(bytes)
}

/// Parse `SPA_META_Cursor` out of the buffer and forward any updates
/// to the cursor channel.
///
/// Position is written every frame the meta is present (compositors
/// always include a cursor meta when in `CursorMode::Metadata`, even
/// when only the pointer position changed). Shape is forwarded only
/// when the parser sees a *new* sprite — id `0` is SPA's "no new
/// cursor data" sentinel, and the dedup state in `UserData` skips
/// re-forwarding an id we've already sent.
fn process_cursor_meta(buffer: &pw::buffer::Buffer<'_>, user_data: &mut UserData) {
    let Some(meta) = buffer.find_meta::<pw::spa::buffer::meta::MetaCursor>() else {
        if !user_data.logged_first_cursor_meta_absent {
            tracing::warn!(
                "first frame: no SPA_META_Cursor on buffer (compositor accepted CursorMode::Metadata but is not attaching the meta)"
            );
            user_data.logged_first_cursor_meta_absent = true;
        }
        // CursorMode::Metadata advertised but compositor isn't
        // attaching the meta — flip visibility off so the client
        // doesn't show a stale overlay forever. Initialise the
        // snapshot if this is the first frame so the host pump
        // still emits a `visible: false` datagram and the client
        // suppresses any stale placeholder.
        if let Ok(mut guard) = user_data.cursor_position.lock() {
            match guard.as_mut() {
                Some(pos) => pos.visible = false,
                None => {
                    *guard = Some(CursorPosition {
                        x: 0,
                        y: 0,
                        visible: false,
                    });
                }
            }
        }
        return;
    };
    if !user_data.logged_first_cursor_meta {
        tracing::info!(
            id = meta.id(),
            valid = meta.is_valid(),
            pos = ?meta.position(),
            has_bitmap = meta.bitmap().is_some(),
            "first frame: SPA_META_Cursor present"
        );
        user_data.logged_first_cursor_meta = true;
    }
    if !meta.is_valid() {
        return;
    }
    let position = meta.position();
    if let Ok(mut guard) = user_data.cursor_position.lock() {
        *guard = Some(CursorPosition {
            x: position.x,
            y: position.y,
            visible: true,
        });
    }
    // Bitmap is only present on sprite changes. The id alone isn't
    // a reliable dedup key — compositors recycle ids across sessions
    // — so we hash the pixel bytes and use that as our stable sprite
    // id on the wire.
    let Some(bitmap) = meta.bitmap() else {
        return;
    };
    if !bitmap.is_valid() {
        return;
    }
    let Some(shape) = build_cursor_shape_from_bitmap(bitmap, meta.hotspot()) else {
        return;
    };
    if user_data.last_shape_id == Some(shape.id) {
        return;
    }
    tracing::debug!(
        id = shape.id,
        w = shape.width,
        h = shape.height,
        hotspot = ?shape.hotspot,
        prev = ?user_data.last_shape_id,
        "pipewire cursor sprite change"
    );
    user_data.last_shape_id = Some(shape.id);
    // try_send because the receiver might be lagging (host pump
    // hasn't drained yet); dropping a stale shape forward is the
    // right call — the next change will resend.
    let _ = user_data
        .cursor_shape_tx
        .try_send(CursorEvent::Shape(shape));
}

/// Convert a `MetaBitmap` payload into a wire-ready
/// [`CursorShapeEvent`]. Returns `None` on unsupported formats or
/// malformed dimensions. The pure swizzle/hash work lives in
/// [`build_cursor_shape`] so it stays testable without faking the
/// libspa `MetaBitmap` memory layout.
fn build_cursor_shape_from_bitmap(
    bitmap: &pw::spa::buffer::meta::MetaBitmap,
    hotspot: pw::spa::utils::Point,
) -> Option<CursorShapeEvent> {
    let size = bitmap.size();
    let width = u16::try_from(size.width).ok()?;
    let height = u16::try_from(size.height).ok()?;
    let stride = usize::try_from(bitmap.stride().unsigned_abs()).ok()?;
    let raw = bitmap.bitmap_data()?;
    build_cursor_shape(
        bitmap.format(),
        width,
        height,
        stride,
        raw,
        (hotspot.x, hotspot.y),
    )
}

/// Pure version: given the wire-format pixel bytes and dimensions,
/// produce the RGBA-normalised [`CursorShapeEvent`]. Pixel byte
/// layout for the supported `SPA_VIDEO_FORMAT` values (memory order,
/// low → high):
///   - `BGRA` / `BGRx` → `[B, G, R, A]`  swizzle to RGBA
///   - `RGBA` / `RGBx` → `[R, G, B, A]`  passthrough
///   - `ARGB`          → `[A, R, G, B]`  swizzle to RGBA
fn build_cursor_shape(
    format: pw::spa::param::video::VideoFormat,
    width: u16,
    height: u16,
    stride: usize,
    raw: &[u8],
    hotspot: (i32, i32),
) -> Option<CursorShapeEvent> {
    use pw::spa::param::video::VideoFormat as V;
    if width == 0 || height == 0 {
        return None;
    }
    let row_bytes = (width as usize).checked_mul(4)?;
    if stride < row_bytes {
        return None;
    }
    if raw.len() < stride.checked_mul(height as usize)? {
        return None;
    }
    let mut rgba: Vec<u8> = Vec::with_capacity(row_bytes * height as usize);
    for row in 0..height as usize {
        let start = row * stride;
        let line = &raw[start..start + row_bytes];
        for px in line.chunks_exact(4) {
            let (r, g, b, a) = match format {
                V::BGRA | V::BGRx => (px[2], px[1], px[0], px[3]),
                V::RGBA | V::RGBx => (px[0], px[1], px[2], px[3]),
                V::ARGB => (px[1], px[2], px[3], px[0]),
                _ => return None,
            };
            rgba.extend_from_slice(&[r, g, b, a]);
        }
    }
    // FNV-1a 64-bit. Build-stable and platform-stable so the same
    // sprite produces the same id across reconnects (and across
    // host binaries) — required by both the host's `seen_ids` dedup
    // and the client's LRU shape cache. `DefaultHasher`'s output is
    // explicitly not stable, so we hand-roll the standard one.
    let id = {
        let mut h: u64 = 0xcbf29ce484222325;
        for byte in width
            .to_le_bytes()
            .iter()
            .chain(height.to_le_bytes().iter())
            .chain(rgba.iter())
        {
            h ^= u64::from(*byte);
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    };
    Some(CursorShapeEvent {
        id,
        width,
        height,
        hotspot: (
            u16::try_from(hotspot.0.max(0)).unwrap_or(0),
            u16::try_from(hotspot.1.max(0)).unwrap_or(0),
        ),
        format: CursorPixelFormat::Rgba8,
        pixels: rgba,
    })
}

/// Build the `SPA_TYPE_OBJECT_ParamMeta` pod that subscribes to the
/// `SPA_META_VideoDamage` block.
///
/// The size we declare is the maximum amount of damage metadata we're
/// willing to receive: `SPA_META_VideoDamage` carries an array of
/// `spa_meta_region`s plus a trailing sentinel, and `find_meta` reads
/// whatever the producer wrote within that budget. We cap at 16
/// regions — well above the per-frame rectangle count any real
/// compositor emits — and the producer either trims or skips
/// per-rectangle data accordingly. Our consumer only inspects whether
/// the list is empty (idle) or non-empty (damaged), so the exact cap
/// doesn't affect correctness.
fn build_video_damage_meta_pod() -> Result<Vec<u8>> {
    let region_size = i32::try_from(std::mem::size_of::<libspa_sys::spa_meta_region>())
        .expect("spa_meta_region size fits in i32");
    // 16 regions plus the SPA_META trailer (one zero-region sentinel).
    let max_regions = 16i32;
    let meta_size = region_size
        .checked_mul(max_regions + 1)
        .expect("damage meta size fits in i32");
    let obj = pw::spa::pod::Object {
        type_: pw::spa::utils::SpaTypes::ObjectParamMeta.as_raw(),
        id: pw::spa::param::ParamType::Meta.as_raw(),
        properties: vec![
            pw::spa::pod::Property {
                key: libspa_sys::SPA_PARAM_META_type,
                flags: pw::spa::pod::PropertyFlags::empty(),
                value: pw::spa::pod::Value::Id(pw::spa::utils::Id(
                    libspa_sys::SPA_META_VideoDamage,
                )),
            },
            pw::spa::pod::Property {
                key: libspa_sys::SPA_PARAM_META_size,
                flags: pw::spa::pod::PropertyFlags::empty(),
                // CHOICE_RANGE rather than fixed Int — Sunshine and
                // pipewire-utils both use the range form here. xdg-
                // desktop-portal-gnome (and any consumer mirroring
                // the pod back for negotiation) prefers the range
                // encoding; a fixed Int silently degrades to
                // hash-only on stricter producers.
                value: pw::spa::pod::Value::Choice(pw::spa::pod::ChoiceValue::Int(
                    pw::spa::utils::Choice::<i32>(
                        pw::spa::utils::ChoiceFlags::empty(),
                        pw::spa::utils::ChoiceEnum::<i32>::Range {
                            default: meta_size,
                            min: region_size,
                            max: meta_size,
                        },
                    ),
                )),
            },
        ],
    };
    let bytes: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(obj),
    )
    .map_err(|e| CaptureError::PipeWire(format!("meta pod serialize: {e:?}")))?
    .0
    .into_inner();
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pw::spa::pod::{ChoiceValue, Value};

    #[test]
    fn region_count_zero_is_idle() {
        assert!(native_damage_from_region_count(0).idle);
    }

    #[test]
    fn cursor_mode_prefers_metadata() {
        let available = CursorMode::Metadata | CursorMode::Embedded | CursorMode::Hidden;
        assert_eq!(select_cursor_mode(available), Some(CursorMode::Metadata));
    }

    #[test]
    fn cursor_mode_falls_back_to_embedded_without_metadata() {
        // wlroots-style portals advertise Embedded + Hidden but not Metadata.
        let available = CursorMode::Embedded | CursorMode::Hidden;
        assert_eq!(select_cursor_mode(available), Some(CursorMode::Embedded));
    }

    #[test]
    fn cursor_mode_falls_back_to_hidden_when_only_option() {
        assert_eq!(
            select_cursor_mode(CursorMode::Hidden.into()),
            Some(CursorMode::Hidden)
        );
    }

    #[test]
    fn cursor_mode_empty_set_requests_nothing() {
        // Regression: xdg-desktop-portal-gnome (interface v5) reports an
        // empty AvailableCursorModes and rejects any explicit mode with
        // "Unavailable cursor mode N", which previously killed the whole
        // session. With no advertised mode we must omit the option.
        assert_eq!(select_cursor_mode(BitFlags::empty()), None);
    }

    #[test]
    fn restore_token_round_trips_through_disk() {
        let path = std::env::temp_dir().join(format!(
            "tether-restore-token-roundtrip-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        save_restore_token_to(&path, "tok-abc-123").expect("save succeeds");
        assert_eq!(
            load_restore_token_from(&path).as_deref(),
            Some("tok-abc-123")
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn restore_token_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let path =
            std::env::temp_dir().join(format!("tether-restore-token-perms-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        save_restore_token_to(&path, "secret-token").expect("save succeeds");
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "restore token must not be group/world readable"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn restore_token_tightens_preexisting_broad_perms() {
        // A token file left world-readable (older build, manual edit) must
        // be tightened on the next save — `OpenOptions::mode` only applies
        // on create, so the explicit chmod is what closes the exposure.
        use std::os::unix::fs::PermissionsExt;
        let path =
            std::env::temp_dir().join(format!("tether-restore-token-broad-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, "old").expect("seed file");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))
            .expect("seed broad perms");
        save_restore_token_to(&path, "new").expect("save succeeds");
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "save must tighten a pre-existing broad mode");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn restore_token_resave_with_same_value_round_trips() {
        // The open_portal dedup only writes when the token changed; this
        // guards that an idempotent re-save of the same value is a no-op
        // for the reader (content + perms stay correct).
        let path = std::env::temp_dir().join(format!(
            "tether-restore-token-resave-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        save_restore_token_to(&path, "tok").expect("first save");
        save_restore_token_to(&path, "tok").expect("second save");
        assert_eq!(load_restore_token_from(&path).as_deref(), Some("tok"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn restore_token_missing_file_is_none() {
        let path = std::env::temp_dir().join("tether-restore-token-absent-xyz");
        let _ = std::fs::remove_file(&path);
        assert_eq!(load_restore_token_from(&path), None);
    }

    #[test]
    fn restore_token_blank_file_is_none() {
        let path =
            std::env::temp_dir().join(format!("tether-restore-token-blank-{}", std::process::id()));
        save_restore_token_to(&path, "   \n").expect("save succeeds");
        // A whitespace-only file must read back as "no token" so we
        // don't replay an empty token into the portal.
        assert_eq!(load_restore_token_from(&path), None);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn save_creates_missing_parent_dirs() {
        let dir =
            std::env::temp_dir().join(format!("tether-restore-token-mkdir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("screencast-restore-token");
        save_restore_token_to(&path, "deep-token").expect("save creates parents");
        assert_eq!(
            load_restore_token_from(&path).as_deref(),
            Some("deep-token")
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn region_count_nonzero_is_changed() {
        for n in [1usize, 2, 7, 16, 1024] {
            assert!(
                !native_damage_from_region_count(n).idle,
                "{n} regions should not be idle"
            );
        }
    }

    /// Snapshot the structure of the `SPA_PARAM_Meta` pod we send for
    /// `SPA_META_VideoDamage`. Catches FFI-shape regressions of the
    /// "fixed Int vs CHOICE_RANGE_Int" kind that no manual session
    /// would surface — a stricter compositor would silently degrade
    /// to hash-only with no log warning.
    #[test]
    fn video_damage_meta_pod_has_choice_range_size() {
        let bytes = build_video_damage_meta_pod().expect("pod builds");
        let (_, value) = pw::spa::pod::deserialize::PodDeserializer::deserialize_any_from(&bytes)
            .expect("pod parses");
        let Value::Object(obj) = value else {
            panic!("expected Object, got {value:?}");
        };
        assert_eq!(
            obj.type_,
            pw::spa::utils::SpaTypes::ObjectParamMeta.as_raw()
        );
        assert_eq!(obj.id, pw::spa::param::ParamType::Meta.as_raw());

        let type_prop = obj
            .properties
            .iter()
            .find(|p| p.key == libspa_sys::SPA_PARAM_META_type)
            .expect("META_type property present");
        match &type_prop.value {
            Value::Id(pw::spa::utils::Id(raw)) => {
                assert_eq!(*raw, libspa_sys::SPA_META_VideoDamage)
            }
            other => panic!("META_type should be Id, got {other:?}"),
        }

        let size_prop = obj
            .properties
            .iter()
            .find(|p| p.key == libspa_sys::SPA_PARAM_META_size)
            .expect("META_size property present");
        let Value::Choice(ChoiceValue::Int(choice)) = &size_prop.value else {
            panic!("META_size must be CHOICE_Int, got {:?}", size_prop.value);
        };
        let pw::spa::utils::ChoiceEnum::Range { default, min, max } = &choice.1 else {
            panic!("META_size must use Range, got {:?}", choice.1);
        };
        let region_size = i32::try_from(std::mem::size_of::<libspa_sys::spa_meta_region>())
            .expect("region size fits in i32");
        assert_eq!(*min, region_size, "min must be one region");
        assert!(
            *max >= *default && *default >= *min,
            "size range must be ordered: min={min} default={default} max={max}"
        );
        assert!(
            *max % region_size == 0,
            "max must be an integer multiple of region size (got {max} vs {region_size})"
        );
    }

    #[test]
    fn cursor_meta_pod_targets_spa_meta_cursor() {
        let bytes = build_cursor_meta_pod().expect("cursor pod builds");
        let (_, value) = pw::spa::pod::deserialize::PodDeserializer::deserialize_any_from(&bytes)
            .expect("cursor pod parses");
        let Value::Object(obj) = value else {
            panic!("expected Object, got {value:?}");
        };
        let type_prop = obj
            .properties
            .iter()
            .find(|p| p.key == libspa_sys::SPA_PARAM_META_type)
            .expect("META_type present");
        let Value::Id(pw::spa::utils::Id(raw)) = &type_prop.value else {
            panic!("META_type should be Id, got {:?}", type_prop.value);
        };
        assert_eq!(*raw, libspa_sys::SPA_META_Cursor);
    }

    /// Lock the cursor meta size range to OBS's published values
    /// (`CURSOR_META_SIZE(1×1)` .. `CURSOR_META_SIZE(1024×1024)`,
    /// default `64×64`). An earlier draft used `min = header only,
    /// no bitmap` which Mutter took as a hint to skip the meta
    /// attachment entirely — `find_meta::<MetaCursor>()` returned
    /// `None` on every buffer with no log surface. Failure mode is
    /// silent, so we encode the expectation.
    #[test]
    fn cursor_meta_pod_size_range_matches_obs() {
        let bytes = build_cursor_meta_pod().expect("cursor pod builds");
        let (_, value) = pw::spa::pod::deserialize::PodDeserializer::deserialize_any_from(&bytes)
            .expect("cursor pod parses");
        let Value::Object(obj) = value else {
            panic!("expected Object, got {value:?}");
        };
        let size_prop = obj
            .properties
            .iter()
            .find(|p| p.key == libspa_sys::SPA_PARAM_META_size)
            .expect("META_size present");
        let Value::Choice(ChoiceValue::Int(choice)) = &size_prop.value else {
            panic!("META_size must be CHOICE_Int, got {:?}", size_prop.value);
        };
        let pw::spa::utils::ChoiceEnum::Range { default, min, max } = &choice.1 else {
            panic!("META_size must use Range, got {:?}", choice.1);
        };
        let header = i32::try_from(
            std::mem::size_of::<libspa_sys::spa_meta_cursor>()
                + std::mem::size_of::<libspa_sys::spa_meta_bitmap>(),
        )
        .expect("header size fits in i32");
        assert_eq!(
            *min,
            header + 4,
            "min must reserve a 1×1 bitmap (matching OBS CURSOR_META_SIZE)"
        );
        assert_eq!(
            *default,
            header + 64 * 64 * 4,
            "default must reserve a 64×64 bitmap (matching OBS CURSOR_META_SIZE)"
        );
        assert_eq!(
            *max,
            header + 1024 * 1024 * 4,
            "max must reserve a 1024×1024 bitmap (matching OBS CURSOR_META_SIZE)"
        );
    }

    /// 1×1 sprite with a known input pixel through each supported
    /// format. The expected output is RGBA `[R, G, B, A]`.
    #[test]
    fn build_cursor_shape_swizzles_per_format() {
        use pw::spa::param::video::VideoFormat as V;
        // input is "red, green, blue, alpha = 0xFF" in each variant's
        // byte order. Resulting RGBA should be [R=0x10, G=0x20, B=0x30, A=0xFF].
        let cases: &[(V, [u8; 4])] = &[
            (V::BGRA, [0x30, 0x20, 0x10, 0xFF]),
            (V::BGRx, [0x30, 0x20, 0x10, 0xFF]),
            (V::RGBA, [0x10, 0x20, 0x30, 0xFF]),
            (V::RGBx, [0x10, 0x20, 0x30, 0xFF]),
            (V::ARGB, [0xFF, 0x10, 0x20, 0x30]),
        ];
        for (fmt, raw) in cases {
            let shape = build_cursor_shape(*fmt, 1, 1, 4, raw, (0, 0))
                .unwrap_or_else(|| panic!("build_cursor_shape returned None for {fmt:?}"));
            assert_eq!(
                shape.pixels,
                vec![0x10, 0x20, 0x30, 0xFF],
                "format {fmt:?} did not produce expected RGBA"
            );
        }
    }

    /// Stride padding past the row should be skipped — only the
    /// first `width * 4` bytes per row are visible pixels.
    #[test]
    fn build_cursor_shape_respects_stride_padding() {
        // 2×1 BGRA, stride = 16 (8 bytes padding).
        let mut raw = vec![0u8; 16];
        raw[0..4].copy_from_slice(&[0x30, 0x20, 0x10, 0xFF]); // BGRA → R=10
        raw[4..8].copy_from_slice(&[0x60, 0x50, 0x40, 0xFF]); // BGRA → R=40
                                                              // Padding (bytes 8..16) is junk that must not appear in output.
        for b in raw.iter_mut().take(16).skip(8) {
            *b = 0xAB;
        }
        let shape = build_cursor_shape(
            pw::spa::param::video::VideoFormat::BGRA,
            2,
            1,
            16,
            &raw,
            (0, 0),
        )
        .expect("stride-padded build succeeds");
        assert_eq!(
            shape.pixels,
            vec![0x10, 0x20, 0x30, 0xFF, 0x40, 0x50, 0x60, 0xFF]
        );
    }

    /// Identical pixels → identical id; one-byte change → different id.
    /// The wire-side dedup relies on this stability.
    #[test]
    fn cursor_shape_id_is_pixel_stable() {
        let raw_a = [0x30, 0x20, 0x10, 0xFF];
        let raw_b = [0x30, 0x20, 0x11, 0xFF];
        let a = build_cursor_shape(
            pw::spa::param::video::VideoFormat::BGRA,
            1,
            1,
            4,
            &raw_a,
            (0, 0),
        )
        .unwrap();
        let a2 = build_cursor_shape(
            pw::spa::param::video::VideoFormat::BGRA,
            1,
            1,
            4,
            &raw_a,
            (1, 2), // hotspot must not influence id
        )
        .unwrap();
        let b = build_cursor_shape(
            pw::spa::param::video::VideoFormat::BGRA,
            1,
            1,
            4,
            &raw_b,
            (0, 0),
        )
        .unwrap();
        assert_eq!(a.id, a2.id, "id must depend on pixels + dims only");
        assert_ne!(a.id, b.id, "different pixel bytes must yield different id");
    }

    /// Unsupported source format yields `None`, not garbage RGBA.
    /// Catches the case where a compositor advertises a cursor in an
    /// exotic format we haven't taught the swizzle about.
    #[test]
    fn build_cursor_shape_rejects_unsupported_format() {
        let raw = [0u8; 4];
        let shape = build_cursor_shape(
            pw::spa::param::video::VideoFormat::I420,
            1,
            1,
            4,
            &raw,
            (0, 0),
        );
        assert!(shape.is_none());
    }

    /// Empty shape channel + uninitialised position snapshot ⇒ the
    /// source idles cleanly. This is the state the host pump sees on
    /// the very first tick before the PipeWire callback has run.
    #[test]
    fn pipewire_cursor_source_initial_state_is_idle() {
        let (mut source, _shape_tx, _position) = PipeWireCursorSource::for_test();
        assert_eq!(source.next_event(), CursorEvent::Idle);
        assert_eq!(source.poll_position(), None);
    }

    /// Producer-side shape sends arrive on `next_event` in FIFO
    /// order; once drained, the source returns to `Idle`.
    #[test]
    fn pipewire_cursor_source_drains_shape_events_in_order() {
        let (mut source, shape_tx, _position) = PipeWireCursorSource::for_test();
        let mk = |id: u64| CursorShapeEvent {
            id,
            width: 16,
            height: 16,
            hotspot: (0, 0),
            format: CursorPixelFormat::Rgba8,
            pixels: vec![0; 16 * 16 * 4],
        };
        shape_tx.send(CursorEvent::Shape(mk(1))).expect("send 1");
        shape_tx.send(CursorEvent::Shape(mk(2))).expect("send 2");
        match source.next_event() {
            CursorEvent::Shape(s) => assert_eq!(s.id, 1),
            other => panic!("expected Shape(1), got {other:?}"),
        }
        match source.next_event() {
            CursorEvent::Shape(s) => assert_eq!(s.id, 2),
            other => panic!("expected Shape(2), got {other:?}"),
        }
        assert_eq!(source.next_event(), CursorEvent::Idle);
    }

    /// `poll_position` returns the *latest* snapshot — not a queue.
    /// Multiple rapid producer writes between two consumer polls
    /// collapse to one; that's the latest-wins property the wire
    /// protocol depends on (one position datagram per pump tick).
    #[test]
    fn pipewire_cursor_source_position_is_latest_wins() {
        let (mut source, _shape_tx, position) = PipeWireCursorSource::for_test();
        *position.lock().unwrap() = Some(CursorPosition {
            x: 100,
            y: 200,
            visible: true,
        });
        *position.lock().unwrap() = Some(CursorPosition {
            x: 300,
            y: 400,
            visible: true,
        });
        assert_eq!(
            source.poll_position(),
            Some(CursorPosition {
                x: 300,
                y: 400,
                visible: true
            })
        );
        // Second poll without an intervening producer write returns
        // the same value, not Idle/None. Important: the pump
        // debounces by equality, not by "did we just poll."
        assert_eq!(
            source.poll_position(),
            Some(CursorPosition {
                x: 300,
                y: 400,
                visible: true
            })
        );
    }

    /// Visibility flag is part of the snapshot; producer writing
    /// `visible: false` must round-trip exactly so the host pump
    /// emits a hide-cursor datagram instead of holding the last
    /// visible position.
    #[test]
    fn pipewire_cursor_source_propagates_visibility_false() {
        let (mut source, _shape_tx, position) = PipeWireCursorSource::for_test();
        *position.lock().unwrap() = Some(CursorPosition {
            x: 0,
            y: 0,
            visible: false,
        });
        assert_eq!(
            source.poll_position(),
            Some(CursorPosition {
                x: 0,
                y: 0,
                visible: false
            })
        );
    }
}
