# Tether — Architecture

Tether is a low-latency open-source remote desktop in Rust. The first
working end-to-end target is **Linux ↔ Linux on a LAN over QUIC**,
with hardware H.264 or HEVC (negotiated per session) at 60 fps default
and a zero-copy capture→encode→decode→render path. Profile negotiation
is **probe-driven** — every layer (capture, encode, decode, renderer
import) advertises only what a real attempt against the live driver
has confirmed it can deliver. The shipped preference list ranks
4:4:4 over 4:2:0 (text fidelity beats subsampled chroma on desktop
content), and 10-bit over 8-bit at each chroma rung (precision
beats banding). See `docs/CODEC_CAPABILITIES.md` for the per-layer
hard-limit vs. probed framing and the M-series VT capability matrix.
**macOS host** (ScreenCaptureKit capture, VideoToolbox encoder,
CGEvent input injection) and **macOS client** (VideoToolbox decode
+ Metal IOSurface→wgpu render + winit input capture) are both
wired end-to-end. The macOS client covers HEVC Main, Main10, and
Main 4:4:4 (8 and 10-bit) decode — the renderer's biplanar 8 /
biplanar 16 / packed XYUV layouts cover the IOSurface and dma-buf
shapes each profile produces, verified by the four
`iosurface_zero_copy_roundtrip_*` tests in
`tether-render/src/iosurface_test.rs`. **Windows host** (DXGI Desktop
Duplication capture → D3D11 Video Processor BGRA→NV12 → vendor-selected
hardware encode) and a **Windows client** decode→render path are wired
end-to-end and verified in a live loopback session. Encode picks the
backend from the DXGI adapter's PCI vendor — Intel→QSV, AMD→AMF,
NVIDIA→NVENC — with Media Foundation as the vendor-agnostic fallback;
4:2:0 only (the Video Processor has no 4:4:4 output path, so Windows
never advertises 4:4:4). **System-output audio** is wired end-to-end
on all three platforms — capture (Linux PipeWire sink monitor, macOS
ScreenCaptureKit, Windows WASAPI loopback) → Opus → unreliable
datagrams → cpal playback (see `tether-audio` and the audio packet
below). See `docs/CODEC_CAPABILITIES.md` for the Windows capture/encode
layers and their per-backend limits.

This document walks the system top-down: what the workspace contains,
how a single frame flows from compositor pixels to the remote display,
which traits and types make the cross-platform story additive, and what
is deliberately out of scope today.

---

## Workspace layout

```
tether/
├── apps/
│   ├── tether-host          # the computer being viewed/controlled
│   └── tether-client        # the computer doing the viewing
└── crates/
    ├── tether-protocol      # wire format, no I/O. Pure types + framing.
    ├── tether-transport     # QUIC server + client (quinn-backed) + role traits (ControlChannel etc.) for loopback testing
    ├── tether-capture       # screen capture: PipeWire (Linux), test pattern
    ├── tether-codec         # Encoder + Decoder traits; H.264 SW + VAAPI
    ├── tether-gpuconvert    # host-side BGRA→NV12 compute + DMA-BUF export
    ├── tether-render        # client-side wgpu renderer (NV12 → window)
    ├── tether-input         # keyboard/mouse capture (client) + injection (host)
    ├── tether-session       # HostSession/ClientSession handshake + IDR coalescing + stats
    └── tether-vaapi         # hand-rolled libva FFI (vaExportSurfaceHandle etc.)
```

Two binaries, nine library crates. `tether-protocol` has no I/O at
all — it's the contract both sides speak. Every other crate is
single-purpose so a future platform backend lands as a sibling file in
the relevant crate (e.g. `tether-capture/src/macos.rs`).

---

## The frame hot path

End-to-end for one frame, host on a Linux Wayland session, client on
any Linux machine. The **macOS host** variant diverges only inside the
capture→encoder hop: at 1:1 (capture_dims == encode_dims)
ScreenCaptureKit hands NV12 IOSurfaces straight to VideoToolbox with
no gpuconvert step; when the client viewport asks for a smaller
encode, the `Nv12IOSurfaceBridge` runs the same Mitchell scaler
pipelines as Linux on the Y and UV planes into a pooled destination
IOSurface that's then fed to VideoToolbox. The rest of the pipeline
from `FrameFragmenter` onward is identical. See the dedicated
**macOS host (shipping today)** subsection further down.

```
┌─────────────────────────────────────────────────────────────────────┐
│ HOST                                                                │
│                                                                     │
│   compositor framebuffer                                            │
│         │                                                           │
│         ▼                                                           │
│   xdg-desktop-portal ScreenCast  ─── permission dialog, once        │
│         │                                                           │
│         ▼                                                           │
│   PipeWire stream (DMA-BUF, multi-modifier negotiation)             │
│         │     BGRx, e.g. modifier=I915_FORMAT_MOD_4_TILED           │
│         ▼                                                           │
│   tether-capture::linux                                             │
│     CapturedFrame::Gpu { DmaBuf { fd, modifier, stride, offset } }  │
│         │                                                           │
│         ▼                                                           │
│   send loop pre-encode gates (apps/tether-host)                     │
│     • native damage short-circuit: CapturedFrame::native_damage     │
│       (PipeWire SPA_META_VideoDamage on Linux; SCStreamFrameInfo    │
│       .status on macOS) skips encode on NativeDamage { idle: true } │
│       regardless of pixel hash                                      │
│     • damage classifier (HashDamage, CPU frames only — GPU frames   │
│       report Unknown and pass through; skip predicate is gated      │
│       additionally by IdrSignal::peek so forced IDRs always go)     │
│     • viewport rebuild check: encode dims = letterbox-fit of       │
│       capture inside the latest ControlMessage::SetClientViewport   │
│       directive, clamped to 16-pixel alignment. Linux GPU paths     │
│       run tether-scaler (Mitchell-Netravali in linear-light)        │
│       between PipeWire's BGRA dma-buf import and the chroma         │
│       bridge; CPU paths bilinear-resize before encode_bgra.         │
│       macOS GPU paths run the same Mitchell pipelines via the       │
│       Nv12IOSurfaceBridge (gpuconvert) — see Mac host scaler        │
│       below.                                                        │
│     • ABR tick: drains the latest ClientStats + quinn path stats    │
│       into tether_session::abr::AbrController; calls                │
│       set_bitrate_kbps when the controller crosses a hysteresis     │
│       boundary. Asymmetric: fast climb after a collapse, slow on    │
│       steady state. Bitrate-only (no runtime FPS gear — no          │
│       capture backend honours runtime FPS retunes today).           │
│         │                                                           │
│         ▼                                                           │
│   tether-gpuconvert::Nv12DmaBuf                                     │
│     • imports BGRA dma-buf as wgpu::Texture (Vulkan hal escape)     │
│     • compute pass writes BGRA→NV12 (BT.709 limited range) into     │
│       a *shared* VkDeviceMemory: R8 (Y) + Rg8 (UV) at distinct      │
│       offsets, exported as ONE dma-buf fd                           │
│     • underlying Vulkan image is allocated at                        │
│       align_up(width, 64 / bytes_per_luma_px) × align_up(height,    │
│       16) — libva-intel-driver mis-reads the dma-buf if luma row   │
│       pitch isn't 64-byte aligned; the compute dispatch covers     │
│       only the visible region and the encoder crops the padding   │
│       via declared frame width. See shared_nv12.rs for the         │
│       permanence rationale.                                         │
│         │                                                           │
│         ▼                                                           │
│   tether-codec::vaapi::VaapiEncoder                                 │
│     • av_hwframe_map(DRM_PRIME → VAAPI) on the single fd            │
│     • h264_vaapi or hevc_vaapi encode → Annex-B NAL units           │
│         │                                                           │
│         ▼                                                           │
│   tether-codec::drain_encoder                                       │
│     • prepend codec extradata (SPS/PPS/VPS) to every keyframe so    │
│       each IDR is self-decodable                                    │
│         │                                                           │
│         ▼                                                           │
│   tether-protocol::video::FrameFragmenter                           │
│     • P-frame:  fragment() → MTU-sized VideoPackets + Reed-Solomon  │
│                 parity (20% default; GF(2^8) caps primaries at      │
│                 FEC_MAX_PRIMARY_SHARDS=212 at 20%, oversize frames  │
│                 fall back to no-FEC); shard size = FEC_SHARD_SIZE   │
│                 = FIRST_PAYLOAD_BUDGET (1100 B)                     │
│     • Keyframe: single_packet() → one fragment_count=1 packet,      │
│                 reliable per-IDR uni stream path                    │
│         │       attach HostFrameTiming + stream_epoch + frame_seq   │
│         ▼                                                           │
│   tether-session::PacedSender                                       │
│     • begin_frame(now) + per-packet wait_for_slot(wire_size) spread │
│       fragments across the frame interval so a burst doesn't pin    │
│       the network queue and induce correlated loss. Bitrate is an   │
│       Arc<AtomicU64> shared with the ABR controller so retunes      │
│       are live without rebuilding the pacer.                        │
│         │                                                           │
│         ▼                                                           │
│   tether-transport::Server                                          │
│     • P-frames + Parity → QUIC datagrams (unreliable, low latency)  │
│     • IDRs → fresh QUIC uni stream per IDR (reliable, ~1 RTT        │
│       deterministic recovery on loss vs one GOP stochastic)         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼  network (LAN today; NAT TBD)
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│ CLIENT                                                              │
│                                                                     │
│   tether-transport::Connection                                      │
│     • tokio::select! races recv_datagram (P-frames, cursor) and     │
│       accept_video_keyframe (per-IDR uni stream); both produce      │
│       VideoPackets fed to the same reassembler                      │
│         │                                                           │
│         ▼                                                           │
│   FrameReassembler                                                  │
│     • validate_packet_sizing first: rejects fragment_count >        │
│       MAX_FRAGMENTS_PER_FRAME (1024) or total_body_len >            │
│       MAX_FRAME_BODY_BYTES before any allocation                    │
│     • drops cross-epoch fragments                                   │
│     • Reed-Solomon recovery from Parity packets when primaries      │
│       arrive incomplete                                             │
│     • wall-clock-evicts pending frames older than max_pending_age   │
│       (500ms default) on every fragment                             │
│         │   complete encoded frame                                  │
│         ▼  (bounded crossbeam channel, capacity 8)                  │
│   tether-decode::run::run_thread (owns the Decoder)                 │
│     • VAAPI submit + drain                                          │
│     • classified error recovery: Flush (cheap) → Rebuild            │
│       (REBUILD_BUDGET=10) → Idr (request from host); rate-limited   │
│       at IDR_RATE_LIMIT (500ms) so a decode-error storm can't pin   │
│       the encoder. NO_OUTPUT_WATCHDOG=1500ms triggers Idr on        │
│       silent decoder stalls.                                        │
│     • vaSyncSurface + vaExportSurfaceHandle → DRM_PRIME             │
│         │   Frame::Gpu(GpuFrame { DmaBuf { fd, stride, modifier } })│
│         ▼  (LatestFrame single-slot drop-oldest)                    │
│   tether-render::gpu                                                │
│     • imports VAAPI dma-buf as two wgpu textures (Y + UV planes)    │
│     • fragment shader does NV12→RGB matrix at present time          │
│     • FrameAgeTracker / decide_present: skip a frame past its       │
│       deadline (late-streak hysteresis + post-skip cooldown);       │
│       PresentPolicy::{LowLatency, Smooth} picks the wgpu present    │
│       mode. wgpu present to the winit window.                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

No CPU memcpy on the host between PipeWire and the encoder. No CPU
memcpy on the client between the decoder and the renderer. Measured on
an Intel Arc iGPU (Meteor Lake), the host's average encode time is
~7-8 ms per 2880×1920 frame; client glass-to-glass latency on loopback
is ~20-25 ms including the ~10 ms present scheduler wait.

**Threading model on the client.** Three concurrent owners on the
critical path: the recv tokio task (QUIC poll + reassemble + hand off
to decoder), a dedicated `std::thread` named `tether-decode` running
`tether_decode::run::run_thread` (owns the VAAPI decoder, classified
error recovery, and the auto-IDR rate-limit; the host-recovery seam
is an injected `request_idr: Arc<dyn Fn() + Send + Sync>` callback
plus a `warnings: Arc<dyn Fn() -> u64>` reader so the run-thread is
backend- and transport-agnostic and loopback-testable), and winit's
main thread running the wgpu render loop. Communication is via crossbeam channels
(recv→decode is bounded(8) so decoder backpressure shows up as
`decode_queue_drops` rather than starving the recv loop;
decode→stats is unbounded, drained non-blocking once per recv
iteration) and the `LatestFrame` single-slot mutex for the
decode→render handoff. A GPU-driver stall inside libavcodec/libva
(`vaSyncSurface`, `vaExportSurfaceHandle`) is contained to the decode
thread — the recv loop keeps polling QUIC and the input-send task
keeps responding.

**Threading model on the host.** Capture + encode + send live on a
dedicated `std::thread` (`tether-host-send`). The sync thread calls
into async transport via `tokio::runtime::Handle::block_on` for the
reliable-IDR write (which is the only blocking call in the loop); the
P-frame datagram path is sync inside quinn. Blocking on the IDR write
preserves ordering — by the time the next iteration of the loop runs,
the keyframe is queued in quinn's send buffer and quinn's FIFO send
order keeps it ahead of the following P-frames on the wire.

**60 fps budget audit (Intel Arc iGPU, Meteor Lake; 60 fps budget =
16.67 ms/frame; p99 over 60 sampled frames):**

| Codec | Resolution | encode_bgra p99 | encode_dmabuf p99 | decode p99 |
| ----- | ---------- | --------------- | ----------------- | ---------- |
| H.264 | 1920×1080  | 13.20 ms        | 5.00 ms           | 1.55 ms    |
| HEVC  | 1920×1080  | 16.15 ms        | 5.08 ms           | 1.45 ms    |
| H.264 | 3840×2160  | 50.40 ms ⚠      | 12.26 ms          | 4.15 ms    |
| HEVC  | 3840×2160  | 53.63 ms ⚠      | 14.55 ms          | 3.62 ms    |

`encode_bgra` is the CPU-upload path the synthetic test-pattern source
uses (BGRA→NV12 swscale + `av_hwframe_transfer_data` upload, then
encode). `encode_dmabuf` is the production zero-copy path (PipeWire
DMA-BUF → tether-gpuconvert NV12 DMA-BUF → encoder). `decode` is
VAAPI `submit` + drain.

Headlines:
- **Production path fits at both resolutions**. encode_dmabuf has
  ~4 ms headroom at 4K60 H.264, ~2 ms at 4K60 HEVC. Tight but real.
- **The BGRA-upload path is the bottleneck at 4K60**, not the
  encoder. The test pattern won't drive a 4K60 stream; real users
  on PipeWire DMA-BUF will. If we ever want a CPU-friendly capture
  fallback, the upload step needs to move off the host CPU (e.g.,
  a Vulkan compute shader doing the swizzle).
- **Decode is cheap everywhere** — under 5 ms p99 even at 4K60.
- **H.264 vs HEVC are within noise** on this hardware. Codec choice
  is about wire bitrate, not encoder cost.

If a workload starts pressing the encode_dmabuf budget at 4K60 (e.g.
a high-motion scene that pushes the rate-control into more expensive
coding modes), `async_depth=2` is the principled lever — one
additional frame of pipeline latency in exchange for parallel encode
slots. Don't reach for it until the measurement says we need it.

Benchmarks live in `crates/tether-codec/src/vaapi/bench.rs`. Run
locally with:

```text
cargo test -p tether-codec --lib bench -- --ignored --nocapture --test-threads=1
```

---

## Cross-platform additivity

Two type seams keep the per-platform GPU integration from leaking into
the cross-cutting code:

**Capture side — `CapturedFrame` enum** (`tether-capture/src/lib.rs`):

```rust
pub enum CapturedFrame { Cpu(CpuFrame), Gpu(GpuCapturedFrame) }

pub enum GpuCapturedSource {
    #[cfg(target_os = "linux")] DmaBuf(CapturedDmaBuf),
    #[cfg(target_os = "macos")] IOSurface(CapturedIOSurface),
    #[cfg(target_os = "windows")] D3D11Texture(CapturedD3D11Texture),
}
```

Each variant is cfg-gated *per platform* so the consumer's `match` is
exhaustive without a catch-all that would silently swallow future
variants. A `GpuCapturedGuard` (a sealed-trait box; type alias for
`tether_protocol::GpuResourceGuard`) lets the capture backend stash
whatever per-frame refcounted handles it needs to keep alive while
the importer reads — without leaking the backend's concrete types
through the public API. The same sealed type is re-exported as
`GpuFrameGuard` in `tether-codec` for the decode side; one
implementation, two named uses.

**Encoder side — `Encoder::encode_gpu(GpuEncoderFrame<'_>)`**
(`tether-codec/src/lib.rs`):

```rust
pub trait Encoder: Send {
    fn encode_bgra(&mut self, bgra: &[u8], pts: i64, kf: bool) -> Result<...>;
    fn encode_gpu(&mut self, _frame: GpuEncoderFrame<'_>, _pts: i64, _kf: bool)
        -> Result<...> { Err(CodecError::UnsupportedInputFormat) }
    // ... is_hardware, codec_kind, name ...
}

pub enum GpuEncoderFrame<'a> {
    #[cfg(target_os = "linux")] DmaBuf(&'a DmaBufFrame),
    #[cfg(target_os = "macos")] IOSurface(&'a IOSurfaceFrame),
    #[cfg(target_os = "windows")] D3D11Texture(&'a D3D11TextureFrame),
    #[doc(hidden)] _Phantom(PhantomData<&'a ()>),
}
```

The trait method is cross-platform; variants inside `GpuEncoderFrame`
are cfg-gated internally. The host's dispatch doesn't need a per-
platform `#[cfg]`. All three platform variants are now live; a fourth
backend is a variant plus a module, not a refactor.

**Encoder-backend dispatch** lives in `tether_codec::probe`: a
`#[cfg(target_os = "linux")]` `build_encoder` arm constructs
`VaapiEncoder`, a `#[cfg(target_os = "macos")]` arm constructs
`VideoToolboxEncoder`, and a `#[cfg(target_os = "windows")]`
`build_encoder_d3d11` arm constructs `D3D11Encoder` — all return
`Box<dyn Encoder>` so the host send loop is backend-agnostic. On
Windows the *vendor* selection happens one level down, in
`D3D11Encoder::new` → `backends_for_vendor(codec, vendor_id)`: it tries
the GPU's native encoder first (`hevc_qsv`/`hevc_amf`/`hevc_nvenc`) then
`hevc_mf`. An unknown vendor falls back to Media Foundation **only** —
speculatively constructing a foreign vendor's encoder faults inside that
vendor's runtime. A second Linux backend (the tracked NVENC follow-up)
lands as an inner `match` inside the Linux arm that prefers NVENC when
the probe accepts it and falls through to VAAPI otherwise — no signature
change at the call site.

**Decoder side** uses the same shape, mirrored: `Decoder::next_frame ->
Frame::{Cpu(DecodedFrame), Gpu(GpuFrame)}` where `GpuFrame.source` is a
`GpuFrameSource` enum with the same cfg gating. The decoder hands the
renderer a `GpuFrameGuard` (the shared `GpuResourceGuard` re-export) that
holds the source `AVFrame` ref alive until the renderer drops it.

**macOS host.** ScreenCaptureKit emits NV12
`CMSampleBuffer`s; `tether-capture::macos` unwraps each to its
`IOSurface` and forwards as `CapturedFrame::Gpu(GpuCapturedSource::IOSurface(...))`.
`tether-codec::videotoolbox::VideoToolboxEncoder` wraps the IOSurface
in a fresh `CVPixelBuffer` (`CVPixelBufferCreateWithIOSurface`) and
feeds it to `h264_videotoolbox` / `hevc_videotoolbox` via the AVFrame
`data[3]` slot — no NV12 conversion step is needed (SCK delivers
NV12 video range natively), so the macOS host has no analogue of
`tether-gpuconvert`, no `BridgeState`, and no chroma-aware dispatch
(VideoToolbox is 4:2:0 only — see the per-platform capability gate
in the negotiation section above). Input injection is via `enigo`'s
CGEvent backend (`inject::enigo_backend`, shared with the Windows
SendInput backend). The Linux host injects through `/dev/uinput`
(`inject::uinput`) instead — portal-free, so input needs no
per-session prompt; see the input section below.

`encode_iosurface_frame` in `apps/tether-host/src/main.rs` is the
macOS sibling of `encode_gpu_frame`: same `EncoderSlot` shape, same
post-encode `FrameFragmenter` path — just simpler in the middle.

**macOS client.** VideoToolbox decoder (`tether-codec::videotoolbox::decoder`)
constructs per the negotiated codec; output IOSurfaces are imported
into wgpu via `tether-render::metal::import_iosurface_textures`
(`MTLDevice::newTextureWithIOSurface`) and presented through the
same `LatestFrame` → wgpu surface path the Linux client uses. The
renderer's import is verified by `iosurface_zero_copy_roundtrip_*`
in `tether-render/src/iosurface_test.rs` for HEVC Main / Main10 /
Main 4:4:4 8-bit / Main 4:4:4 10-bit.

---

## Robustness (mid-session freezes are unacceptable)

Video is the load-bearing feature; even rare freezes are unacceptable.
The pipeline carries the following defenses, all measured and
documented in `docs/INVESTIGATION_lan_freeze.md`:

- **Non-blocking tracing writer.** `tracing_appender::non_blocking`
  on both apps. Required because the FFmpeg `av_log` callback runs on
  the decoder thread; a sync subscriber would block decode under an
  error storm.
- **Async `av_log` bridge.** The FFmpeg log callback bumps an atomic
  counter (load-bearing — the client polls this to detect "ffmpeg is
  unhappy" states that don't surface through the API) then `try_send`s
  a `LogRecord` onto a bounded crossbeam channel. A dedicated
  drainer thread emits to tracing. The decoder thread never blocks on
  log writes; on channel-full the line is dropped but the counter
  still advances.
- **`SO_RCVBUF = 16 MiB` on the client UDP socket.** Linux's ~208 KB
  default overflows in milliseconds under bursty keyframes; the
  kernel silently drops packets *before* quinn's datagram buffer
  ever sees them.
- **`FrameReassembler` wall-clock timeout.** Pending frames older
  than `max_pending_age` (500 ms default) are evicted on every
  fragment in addition to the existing frame_seq-distance eviction.
  Stops a quiet-stream encoder restart from leaking incomplete frames.
- **Reliable per-IDR uni streams.** Detailed in "Protocol shape"
  below.
- **Dedicated decode thread (`tether-decode`).** Owns the VAAPI
  decoder and the auto-IDR 500 ms rate-limit. A GPU-driver stall
  inside libavcodec/libva can't starve the recv loop's tokio task
  or the input-send task running on the same runtime.
- **`LatestFrame` single-slot drop-oldest channel between decoder
  and renderer.** A remote-desktop viewer wants the freshest decoded
  frame, not a queued backlog. Crossbeam bounded(N) drop-newest is
  exactly wrong for this hop — under render backpressure the user
  would stare at stale pixels while newer frames got rejected.
- **Same freshest-wins principle on the Windows capture→encode hop.**
  The DXGI capture thread uses a single-slot drop-oldest mailbox plus a
  texture-pool free-list with an ownership handshake: a pool slot is
  reused only once the frame's `release_guard` drops, so the capture
  thread never `CopyResource`s over a texture the encoder's Video
  Processor is still sampling (the cause of an earlier progressive-
  corruption regression). Because capture and encode share one D3D11
  immediate context, GPU commands execute in submission order and the
  channel's happens-before edge is sufficient — no GPU fence/keyed-mutex
  needed. Shutdown is detected via a consumer-liveness token, since the
  evict-clone the producer holds masks channel `Disconnected`.
- **DoS-relevant transport limits.** `MAX_VIDEO_STREAM_MESSAGE = 2 MiB`
  caps per-keyframe-stream allocation; `max_concurrent_uni_streams = 4`
  prevents a peer from opening thousands of streams and pinning
  receive-side buffers. On the datagram path,
  `validate_packet_sizing` is the *first* call in
  `FrameReassembler::handle`: any packet declaring
  `fragment_count > MAX_FRAGMENTS_PER_FRAME` (1024) or implying a
  body larger than `MAX_FRAME_BODY_BYTES` is dropped before the
  reassembler allocates anything keyed on that frame. The host's
  control recv loop applies the same defensive shape on the other
  direction (250 ms IDR-trigger floor — see invariant 3).

## Protocol shape

`tether-protocol` defines five logical channels, each carried by
its own QUIC primitive:

- **Control** (reliable, bidirectional) — length-prefixed bincode
  `ControlMessage`. Handshake (`ClientHello` / `ServerHello`, both
  tagged-enum envelopes), `Goodbye` (with machine-readable
  `GoodbyeCode`), `ForceIdr`, `ClockProbe*`, cursor shapes
  (`CursorShape` / `CursorUseShape`), display topology
  (`DisplayList` / `SetActiveDisplays`), stream lifecycle
  (`StreamReady` / `StreamPause` / `StreamResume`), receiver
  telemetry (`ClientStats`), and the open-ended `Extension { key,
  payload }` escape for future features that aren't worth a typed
  variant yet (clipboard, file transfer, gamepad rumble, auth, …).
- **Video** — `VideoPacket::First { display, stream_epoch, frame_seq,
  fragment_count, meta: VideoFrameMetaEnvelope, payload }` or
  `::Continuation { …, fragment_index, payload }`. Two transport
  paths share the same wire shape: **P-frames** ride unreliable QUIC
  datagrams (split MTU-sized via `FrameFragmenter::fragment`);
  **IDR keyframes** ride a fresh QUIC unidirectional stream per IDR
  via `Connection::send_video_keyframe`, carrying one
  fragment_count=1 packet built by `FrameFragmenter::single_packet`.
  Reliable streams turn IDR recovery from stochastic (wait for next
  encoder IDR) into deterministic 1-RTT QUIC retransmit on loss. The
  receiver's `FrameReassembler` doesn't care which path delivered
  the fragment — it keys on `(display, stream_epoch, frame_seq)`
  regardless. `stream_epoch` is `u32` so encoder restarts can't wrap.
  `VideoFrameMetaEnvelope` is a versioned wrap around
  `VideoFrameMeta` so future per-frame metadata (HDR ROI QP) lands
  as additive variants instead of struct-field appends.
- **Audio** (unreliable datagrams, host → client) — `AudioPacket::Opus
  { stream_epoch, frame_seq, t_capture, payload }`. The full Opus
  capture → encode → decode → playback pipeline is implemented in
  `tether-audio` (per-platform system-output capture, libopus codec
  with PLC, lock-free jitter ring + cap-and-drop playback policy),
  negotiated host-authoritatively via the `tether.audio` hello
  extension.
- **Cursor** (unreliable datagrams, high priority) — pointer position
  only (`HostCursorPacket::Position`, `ClientCursorPacket`). Sprite
  payloads ride the reliable control stream (too large for a
  1200-byte datagram).
- **Input** (reliable stream, client → host) — `InputEvent { event_id,
  t_client, device_id, kind }`. `device_id` is hardcoded `0` (primary
  keyboard/mouse) today; reserved for future gamepad / pen /
  multi-touch devices.

Forward-compat hooks every feature added later relies on:

- **Hello extension map** with reverse-DNS-style keys (`tether.audio`,
  `tether.pixel-format`, `tether.cap.*`). Receivers ignore unknown
  keys; capability keys (`tether.cap.*`) follow an echo-to-accept
  convention.
- **`ControlMessage::Extension { key, payload }`** as the escape for
  any new control message that doesn't fit the typed variants.
- **`VideoFrameMetaEnvelope`** so per-frame metadata grows by enum
  variant rather than struct field.

### Session orchestration and the channel-trait abstraction

The application-layer handshake (post-QUIC: extension parsing, profile
negotiation, `Goodbye(InternalError)` on no-match, initial `ForceIdr`)
lives in `tether-session::{HostSession, ClientSession}`, not inline in
the app binaries. `tether-transport` defines four role-shaped traits —
`ControlChannel`, `InputChannel`, `VideoChannel`, `ConnectionInfo` —
each mirroring how a specific consumer uses the connection; the
concrete `Connection` impls all four. `HostSession::accept` and
`ClientSession::connect` take `Arc<dyn ControlChannel>` so the
handshake is loopback-testable through
`tether_transport::test_support::DuplexControlChannel` (a
`tokio::io::duplex`-backed impl gated behind the `test-support`
feature, used by `crates/tether-session/tests/loopback.rs`). The
same feature ships `DuplexVideoChannel` and `DuplexInputChannel`
fakes (so a session test can drive the full
fragmenter→datagram→reassembler→decoder path without QUIC), a
`LossyChannel` wrapper that drops a configurable percentage of
packets (used to exercise the Reed-Solomon recovery path under
random loss, alongside the proptest at
`crates/tether-protocol/tests/fragmenter_property.rs`), and
`tether_decode::test_support::FakeDecoder` so the decode run-thread
can be driven with synthetic output without VAAPI/VideoToolbox in
the loop.

The handshake is split across two `ControlChannel` methods —
`recv_client_hello` returning `(ClientHello, t1_server_recv)`, then
`send_server_hello(server, client_t0, t1)` which stamps `t0_echo` /
`t1` / `t2_server_send` immediately before serializing the wire bytes.
Splitting at that seam keeps the clock-sync stamps inside the wire
layer (so a slow `HostSession` orchestration step still produces a
late `t2`) while profile-negotiation policy stays in the session
layer. The trait methods carry an unenforced ordering invariant (recv
once before send once); orchestration code routes through the
`HostHandshake` → `ClientHelloReceived` typestate in
`tether-transport::handshake`, which owns the `Arc<dyn ControlChannel>`
across the two transitions and consumes itself on each step — making
double-send and send-before-recv uncompilable rather than runtime
bugs. The channel is returned out of `send_server_hello` so the
caller can resume post-handshake control-message exchange. App binaries hold an `Arc<Connection>` for the recv tasks
(video / input / datagram are outside the `ControlChannel` surface)
and pass a coerced `Arc<dyn ControlChannel>` into the session call
only.

`tether-host`'s `main` runs a reconnect loop: per-session errors log
and continue; only a server close ends the process. The whole
per-connection graph (encoder thread, uinput injector, recv tasks)
lives inside `handle_client` and drops when it returns, so nothing
leaks across reconnects.

### Codec / chroma / depth negotiation

Video profile is negotiated host-authoritatively via two hello
extensions. The client advertises its decode capabilities under
`tether.cap.video.decode-profiles` as `Vec<VideoProfile { codec,
chroma, bit_depth }>` (computed by `tether_probe::client_decode_profiles()`).
The host intersects that set with its own buildable encode profiles
(from `tether_probe::host_encode_profiles()`, an `OnceLock`-cached
real-round-trip probe — capture → bridge → encoder → decoder per
profile) and picks the best mutual match against a fixed preference
list:

1. HEVC 4:4:4 8-bit (desktop-quality top rung — preserves text and UI
   chroma detail that 4:2:0 visibly smears).
2. HEVC 4:2:0 8-bit.
3. H.264 4:2:0 8-bit (universal floor; H.264 4:4:4 is absent because
   VAAPI has no encode profile for it).

The chosen profile is echoed in `tether.cap.video.encode-profile`;
`ServerHelloV1.chosen_codec` / `chosen_chroma` carry the same
information in legacy form so older clients can interoperate. Absent
client extension is treated as the universal floor.

Both the VAAPI encoder (`VaapiEncoder::new` takes `VideoProfile`,
switches `sw_format` + the AVCodecContext `profile` field for
`AV_PROFILE_HEVC_REXT` on 4:4:4 + BGRA→input swscale stage; color
primaries / transfer / colorspace / range tagged explicitly on the
context so the SPS VUI doesn't say "Unspecified") and the gpuconvert
bridge branch on `(chroma, bit_depth)` at construction:

- `(Yuv420, 8)` → `Nv12DmaBuf` (R8 + Rg8 biplanar, fourcc `NV12`).
- `(Yuv420, 10)` → `Bgra2P010DmaBuf` (R16 + Rg16 biplanar, fourcc
  `P010` with 10-bit data MSB-aligned in 16-bit cells).
- `(Yuv444, 8)` → `Yuv444DmaBuf` (single-plane Rgba8Unorm packed,
  fourcc `XYUV` per DRM_FORMAT_XYUV8888).
- `(Yuv444, 10)` → `Bgra2Xv30DmaBuf` (single-plane Rgb10a2Unorm
  packed, fourcc `XV30` per DRM_FORMAT_XV30 — biplanar P410 has no
  `vaapi_drm_format_map` entry, so packed is the only viable input).

The renderer dispatches on a derived `RenderLayout`
(`Biplanar8` / `Biplanar16` / `PackedXYUV`) rather than chroma
directly: Yuv420 is always biplanar (NV12 / P010, half-res UV);
Yuv444 is biplanar on macOS (NV24 / `'xf44'` / `'P410'` IOSurface from
VT, full-res UV through the same Y + UV shader at 8 or 16 bit) and on
Linux is packed XYUV at 8-bit, biplanar P410-style 16-bit at 10-bit
(driver-dependent — RADV has emitted both packed and biplanar across
Mesa versions; see `gpu/import.rs:53` for the failure mode if the
decoder picks packed). Mid-session chroma or bit-depth switch is not
supported — same rebuild path as a mid-session resolution change
(encoder + bridge + render pipeline all reset).

**Per-platform asymmetries.** `tether_probe::host_supported_profiles()`
does a real encode + decode round trip per profile against the live
driver (fixture IDRs ship in `crates/tether-probe/fixtures/probe/`;
the `ProfileProbe` trait is in `crates/tether-probe/src/profile_probe.rs`).
On Linux the round trip pulls in `tether-gpuconvert` to produce a
real dma-buf, which is the architectural reason this crate exists —
the codec layer can't depend on gpuconvert without a cycle. Empirical results on M-series Apple Silicon:
VideoToolbox has no HEVC Main 4:4:4 *encode* path (so macOS hosts
never advertise 4:4:4 — the intersection lands on HEVC 4:2:0 or H.264
4:2:0), but the *decode* path produces a `'444v'` NV24 IOSurface and
the renderer's biplanar import handles it. A Linux→Mac session can
therefore negotiate HEVC 4:4:4 even though Mac→anything cannot. On
Linux, VAAPI driver capabilities vary; the probe catches
driver-specific 4:4:4 gaps that a codec-keyed construction probe
would miss.

**Bitrate is chroma-aware.** `derive_bitrate_kbps` takes a
`VideoProfile` and applies a 1.4× multiplier for `Yuv444` on top of
the per-codec efficiency factor. 4:4:4 carries 3× the chroma samples
of 4:2:0; rate-control absorbs some of that but not all, so a chroma-
blind budget produces visibly blocky chroma in the same numbers that
were sized for subsampled video.

**Renderer accepts both YUV444 dma-buf shapes.** `vaExportSurfaceHandle`
with `SEPARATE_LAYERS` is a *hint* the libva spec lets drivers ignore.
Intel media-driver and current mesa return three R8 layers (one plane
each); older mesa and nvidia-vaapi-driver return one `YU24` layer
carrying three plane offsets. The import path accepts either. The
encoder-side `yuv444_dmabuf_to_codec_frame` produces the
one-layer/three-plane form, which matches VAAPI's PRIME_2 *importer*
expectation on Main444.

The `tether.pixel-format` extension echoes the on-wire pixel format
of the encoded stream (`Nv12` for 4:2:0 8-bit, `P010` for 4:2:0
10-bit, `Yuv444p` for HEVC Main 4:4:4 8-bit, `P410` for HEVC Main
4:4:4 10-bit) so client decoders that wire their import path before
the first SPS arrives can pick the right plane layout up front.

Four non-negotiable invariants tracked end-to-end:

1. **Clock sync.** Handshake measures RTT and computes a `MonoNanos`
   offset between host and client clocks. Every video fragment carries
   `HostFrameTiming { t_capture_kernel, t_capture_userspace,
   t_encode_submit, t_encode_done }` so the client can attribute
   end-to-end latency to each pipeline segment in its own clock.
2. **Stream epoch.** Resolution changes (display reconfig, future
   monitor handoff) bump the epoch and force a fresh IDR. The
   defragmenter discards any pre-epoch fragments still in flight,
   preventing fusion of a partial old frame with a new one.
3. **On-demand IDR.** Client can request a keyframe at any time via
   `ControlMessage::ForceIdr` or `ControlMessage::RequestRecovery
   { last_known_good_frame_id }`. The host swap-and-zeros an
   `AtomicBool` so multiple requests between encode calls coalesce
   to one; the control recv loop additionally enforces a 250 ms
   floor between any IDR-triggering messages so a client flood
   can't pin the encoder in perpetual-keyframe mode. GOP is long
   (~240 frames); IDRs are driven by request, not cadence — the GOP
   is the safety net, not the primary recovery mechanism. The
   client's decode run-thread also rate-limits IDR emission to one
   per `IDR_RATE_LIMIT` (500 ms). Phase-1 `RequestRecovery` collapses
   to IDR on the host side — an LTR-aware recovery path that uses
   `last_known_good_frame_id` to pick a reference instead of
   sending a fresh keyframe is parked until NVENC lands; FFmpeg's
   `h264_vaapi` / `hevc_vaapi` wrappers expose no LTR plumbing
   (see CODEC_CAPABILITIES.md).
4. **Self-decodable IDRs.** Every keyframe carries its own codec
   parameter sets (SPS/PPS for H.264, VPS+SPS+PPS for HEVC).
   Achieved by setting `AV_CODEC_FLAG_GLOBAL_HEADER` on the VAAPI
   encoder so libavcodec stashes parameter sets in `extradata` at
   `open()`, and prepending `extradata` to every keyframe packet
   inside `drain_encoder`. Without this, only the encoder's very
   first packet would carry parameter sets, and a client that loses
   that first IDR or rebuilds its decoder mid-session is stuck.

---

## Input and cursor

Host side, Linux:
- `tether-input::inject::uinput` creates three virtual devices via
  `/dev/uinput` (keyboard, relative pointer, absolute pointer — split
  so libinput doesn't see a device mixing REL and ABS). Portal-free, so
  unlike screen capture it needs **no per-session prompt**; the only
  gate is `/dev/uinput` access, granted once by a udev rule. The rule
  ships in the deb/rpm packages (zero prompts); otherwise the host
  installs it lazily on first connection through a GUI PolicyKit dialog
  and retries (`apps/tether-host/src/setup_input.rs::linux_injector`) —
  the same "permission prompt on first use" model as the screen-capture
  portal. `connect()` returns `InjectError::DeviceUnavailable` for the
  missing-node / not-writable cases so the host knows to prompt rather
  than fail; a declined grant degrades to no-op input, never a fatal
  error. `tether-host --setup-input` remains as an explicit
  pre-provision step (headless / scripted setup). We deliberately do
  not use the RemoteDesktop portal/libei: it re-prompts every session
  and its restore-token support is unreliable across compositors. This
  matches Sunshine and rustdesk's server mode. Caveat: uinput emits
  physical keycodes, so the client's `Text` events are reverse-mapped
  to a US-QWERTY layout — non-US host layouts and non-ASCII codepoints
  don't round-trip and are dropped.
- Cursor updates are a separate channel from input events so the client
  can render a remote cursor sprite over the decoded video without
  waiting for the next encoded frame.

Client side: native `winit` event loop captures keyboard/mouse, encodes
to a transport-agnostic HID-style `InputEvent`, sends to host.

**Relative mouse mode** is wired end-to-end for FPS / 3D apps that
need pointer-lock semantics rather than absolute coordinates.
`ControlMessage::SetCursorMode { mode: CursorMode::{Absolute,
Relative} }` switches the host's dispatch; client-side, winit's
`DeviceEvent::MouseMotion` feeds
`tether-render::relative_mouse::SubPixelAccum` (a DDA-style residue
accumulator that folds f64 sub-pixel deltas into `i16` emissions —
sub-pixel motion accumulates across events instead of being
rounded away). `InputEventKind::RelativeMouseMove { dx, dy,
modifiers }` rides the input stream. Ctrl+Alt+G toggles the mode
on the client; cursor grab uses winit `CursorGrabMode::Locked` and
falls back to `Confined`. Host-side, the injector clamps each
delta to ±1000 px before emitting it (REL_X/REL_Y on Linux uinput,
`Coordinate::Rel` on macOS/Windows enigo) so a malformed or hostile
event can't teleport the cursor.

**`CaptureHandle` seam** (`tether-capture/src/lib.rs`). Every
backend's `start()` returns `CaptureHandle { rx:
Receiver<CapturedFrame>, target_fps: Arc<AtomicU32> }`. The
test-pattern source reads the atomic each producer iteration; the
Linux PipeWire and macOS SCK backends accept the atomic for API
symmetry but do not yet renegotiate the underlying stream when it
changes (per-backend follow-up — SCK needs an
`SCStreamConfiguration.minimumFrameInterval` reapply, PipeWire
needs a re-`set_param`).

---

## What's deliberately out of scope (today)

Listed to set expectations; each is a real follow-up, not a "never":

- **Windows: remaining gaps.** The Windows host (DXGI capture, D3D11
  vendor-selected encode) and client (D3D11VA decode → native D3D11
  render) are wired and loopback-verified, with the decoder exporting
  GPU-resident shared-handle frames (no CPU download, no wgpu/Vulkan
  bridge) at 4:2:0 8-bit (NV12) and 10-bit (P010 / Main10). What's still
  open: there is no 4:4:4 encode path (the Video Processor only outputs
  4:2:0); cross-device decode→present sync currently relies on the
  handoff latency rather than a shared device, validated on shared iGPUs
  (discrete GPUs may need the single-device model — see the
  `import_shared_biplanar` note).
- **AV1.** H.264 and HEVC are supported; AV1 needs a different VAAPI
  decoder probe (no `vaapi_av1` encode entrypoint on most current
  Intel iGPUs) and a separate codec_id path. The probe stub returns
  `CodecNotFound` for AV1 today.
- **HDR (BT.2020 + PQ / HLG).** The 10-bit *bit-depth* path is in
  place at every layer (see `docs/CODEC_CAPABILITIES.md`):
  `PROFILE_PREFERENCE` advertises HEVC Main10 and HEVC Main 4:4:4
  10-bit; the host encode-side bridge produces P010 dma-buf for
  4:2:0 (R16 + Rg16 biplanar) and packed XV30 dma-buf for 4:4:4
  (Rgb10a2Unorm, via `Bgra2Xv30DmaBuf`); the renderer has an
  R16/Rg16 biplanar `RenderLayout::Biplanar16` for both Linux
  dma-buf and macOS IOSurface (`'P010'`/`'P410'`/`'xf44'`) paths, and the
  Windows native D3D11 renderer samples P010 via R16/R16G16 plane SRVs
  (Main10 4:2:0); the shader carries a `luma_scale` uniform (D3D11: a
  `RANGE_KIND_LIMITED_10` branch) that compensates 10-in-16 MSB-aligned
  sampler reads. What's *not* yet in place is HDR
  signalling proper (BT.2020 primaries, PQ / HLG transfer curves in
  the EOTF dispatch, HDR-capable surface format) — the renderer
  hard-pins BT.709 limited range regardless of `bit_depth`. 10-bit
  on the wire today buys precision (less banding on gradients)
  without HDR luminance range.
- **NAT traversal.** LAN direct only. QUIC's pluggable transport makes
  adding ICE later straightforward; today the user runs the client
  binary with a host IP.
- **Real cursor sprite capture.** `tether-capture::cursor::CursorSource`
  trait + `PlaceholderCursorSource` are in place; per-platform
  backends (Wayland `SPA_META_Cursor` parser, macOS `NSCursor` or
  SCK Sonoma+ cursor metadata) are not. The current host sends one
  16×16 checkerboard at session start and the cursor remains burned
  into the captured frame (`CursorMode::Embedded` on the portal).
  The seam is genuine — when a real backend lands, it drops into
  `cursor.rs` without touching `handle_client`.
- **Profile + rate-control probe (VAAPI).** Today we hard-code
  `profile=main` and `rc_mode=VBR`. Apollo carries an AMD-specific
  CBR/VBR probe and Sunshine probes the profile table via
  `vaQueryConfigProfiles`. Both are the principled extensions, both
  need a small libva FFI add (vaQueryConfigProfiles,
  vaGetConfigAttributes) and an AMD test box to validate against.
- **Multi-monitor.** Single primary monitor capture; the
  keyframe-sender path assumes one fragmenter, and `display = 0` is
  hard-coded throughout the host send loop.
- **Audio: remaining gaps.** System-output capture → Opus → playback is
  wired on all three platforms and negotiated via the `tether.audio`
  hello extension, but there is no user-facing control yet — audio is
  on whenever both peers support it (host `--no-audio` opts out). A
  per-session mute/volume toggle waits on the user-prefs work. No
  microphone / client → host audio path; system output only.
- **Periodic safety-net IDR.** Sunshine deliberately omits this on
  the NVENC path; we follow suit. Worth adding if we ever see a
  "client went silent without observing decode failure" stall mode
  (display sleep, decoder lockup) — cheap insurance, ~20 LOC.
- **FEC on keyframes specifically.** P-frame datagrams now carry
  Reed-Solomon parity (see hot path), but IDR keyframes still ride
  reliable per-IDR QUIC uni streams for deterministic 1-RTT recovery
  — no FEC overhead on the keyframe path. Sunshine/Apollo/Moonlight
  use FEC on keyframes because RTP-over-UDP has no reliable side-
  channel; we have QUIC streams and don't.
- **Reference frame invalidation (RFI).** Cheaper than a full IDR for
  recovering from limited reference loss, but adds wire complexity
  and the benefit at our resolutions is marginal. Skip.
- **`SO_PRIORITY` / `IP_TOS` QoS tagging.** Sunshine sets these on the
  Linux send socket. Linux-specific; worth its own change once we
  have benchmarks showing it helps on contested networks.
- **Wgpu device-loss recovery on the client.** A driver crash today
  exits the client process. A single rebuild attempt before fatal
  exit would be a UX win.
- **Periodic clock re-probe.** The handshake measures one offset; on a
  long-running session (thermal drift, laptop sleep/wake) the offset
  could drift. `ClockProbeRequest` exists but nothing initiates
  re-probes. For sub-minute sessions, fine.
- **Wraparound safety for `frame_seq` / `stream_epoch`.** Both are
  `u32` with `wrapping_add`; at 60 fps it'd take 828 days to wrap
  `frame_seq` and 2^32 encoder restarts for `stream_epoch`. Not
  reachable today; document if we ever support multi-day persistent
  sessions.
- **Explicit GPU sync (`sync_file` fd).** The host blocks on
  `device.poll(wait_indefinitely)` after the compute pass to ensure
  the Vulkan write retires before VAAPI reads. Works in practice on
  Mesa because the driver conservatively tracks the implicit dma-buf
  reservation-object fence; AMDVLK and some non-Mesa stacks need an
  explicit `VkSemaphore`-exported sync_file. Belt-and-suspenders fix.
- **Ring buffer of NV12 export targets.** A single Y+UV pair is reused
  across frames; the encoder's read of frame N must finish before the
  compute pass for frame N+1 begins. Holds at 50fps with 8ms encode;
  would matter at 120fps or under encoder stall.

---

## Known design constraints

A few choices that look strange in isolation but have load-bearing
reasons:

- **NV12 export is hard-pinned to `DRM_FORMAT_MOD_LINEAR`.** The
  encode-side dma-buf hands off to FFmpeg's
  `av_hwframe_map(DRM_PRIME → VAAPI)`, which requires both planes in
  one DRM object at distinct offsets. Tiled modifiers don't have a
  portable shared-allocation contract that VAAPI's DRM_PRIME importer
  honours; LINEAR is the only modifier that does. The PipeWire
  *capture* side advertises whatever modifiers the GPU importer can
  consume (queried via `vkGetPhysicalDeviceFormatProperties2`) and
  imports the compositor's tiled buffer natively.
- **Bridge construction is fatal-on-failure mid-session.** A startup
  probe (`tether_gpuconvert::importable_dmabuf_modifiers`) verifies
  the wgpu+Vulkan+features chain works before PipeWire advertises
  DMA-BUF. If runtime bridge init then fails, that's device-loss or
  OOM — exit the send loop loudly rather than silently drop every
  subsequent Gpu frame and freeze the client.
- **Hard requirement on hardware codecs.** SW H264 is gated behind
  `cfg(test)`. The probe returns `NoHardwareCodec` with actionable
  diagnostics ("run vainfo", "check kernel module") rather than
  silently falling back to a SW path that would tank latency. On
  macOS the VideoToolbox encoder additionally sets `allow_sw=0` and
  errors out if FFmpeg leaves that option unconsumed (older builds
  with stale option tables), so the GPU-only invariant doesn't decay
  silently across FFmpeg versions.
- **FFmpeg build requirements.** Linux host needs FFmpeg built with
  `--enable-vaapi` (Ubuntu's default is fine; check `ffmpeg -encoders | grep vaapi`).
  macOS host needs `--enable-videotoolbox` (Homebrew's `ffmpeg`
  formula enables it by default — verify with
  `ffmpeg -encoders | grep videotoolbox`). Custom or trimmed builds
  may omit either; the probe surfaces a `CodecNotFound` with the
  exact encoder name to look for.
- **macOS Swift runtime rpath.** The screencapturekit / apple-cf
  crates ship Swift shims that link `libswift_Concurrency.dylib` via
  `@rpath`. Their build scripts emit
  `cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift`, but cargo only
  propagates that to the artifact built by the same script. The
  workspace `.cargo/config.toml` bakes the OS Swift runtime location
  into every macOS link product as a workspace-wide rpath. Drop the
  override once the upstream crates emit
  `rustc-link-arg-bins-tests-examples` instead.
- **wgpu pinned to a trunk SHA, not a release.** `texture_from_dmabuf_fd`
  isn't in a published wgpu release yet. Workspace-level
  `[patch.crates-io]` keeps every wgpu sub-crate on the same SHA so a
  future workspace member pulling wgpu via crates.io doesn't get a
  second incompatible copy. Drop the pin when wgpu cuts a release with
  the API.

---

## Reference reading

Tracked locally under `~/Code/refs/` for cross-checking design
decisions:

- **RustDesk** — `EncoderApi` trait shape, hardware-probe cache,
  QoS hysteresis pattern.
- **Sunshine** — PipeWire DMA-BUF negotiation reference (modifier
  property flags, `eglQueryDmaBufModifiersEXT` for the multi-modifier
  list, ParamBuffers shape).
- **Moonlight (Android)** — zero-copy MediaCodec → Surface as the
  conceptual analogue for the VAAPI → wgpu import direction.

Where Tether's path diverges from these references it's intentional and
called out in the relevant module doc.
