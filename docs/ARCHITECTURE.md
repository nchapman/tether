# Tether — Architecture

Tether is a low-latency open-source remote desktop in Rust. The first
working end-to-end target is **Linux ↔ Linux on a LAN over QUIC**, with
hardware H.264 or HEVC (negotiated per session) at 60 fps default and
a zero-copy capture→encode→decode→render path. **macOS host (capture +
encode + input injection) compiles and the encoder round-trips on
Apple Silicon** via ScreenCaptureKit, VideoToolbox, and CGEvent;
end-to-end LAN streaming from a Mac to a Linux client is the next
demo milestone. The macOS client (VideoToolbox decode + Metal render
+ winit input capture) and the Windows backends (DXGI / Media
Foundation / D3D11) are additional modules per platform, not a
rewrite of the core path.

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
    ├── tether-transport     # QUIC server + client (quinn-backed)
    ├── tether-capture       # screen capture: PipeWire (Linux), test pattern
    ├── tether-codec         # Encoder + Decoder traits; H.264 SW + VAAPI
    ├── tether-gpuconvert    # host-side BGRA→NV12 compute + DMA-BUF export
    ├── tether-render        # client-side wgpu renderer (NV12 → window)
    ├── tether-input         # keyboard/mouse capture (client) + injection (host)
    ├── tether-session       # cross-platform session helpers (IDR coalescing, stats)
    └── tether-vaapi         # hand-rolled libva FFI (vaExportSurfaceHandle etc.)
```

Two binaries, nine library crates. `tether-protocol` has no I/O at
all — it's the contract both sides speak. Every other crate is
single-purpose so a future platform backend lands as a sibling file in
the relevant crate (e.g. `tether-capture/src/macos.rs`).

---

## The frame hot path

End-to-end for one frame, host on a Linux Wayland session, client on
any Linux machine:

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
│   tether-gpuconvert::Nv12DmaBuf                                     │
│     • imports BGRA dma-buf as wgpu::Texture (Vulkan hal escape)     │
│     • compute pass writes BGRA→NV12 (BT.709 limited range) into     │
│       a *shared* VkDeviceMemory: R8 (Y) + Rg8 (UV) at distinct      │
│       offsets, exported as ONE dma-buf fd                           │
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
│     • P-frame:  fragment() → MTU-sized VideoPackets, datagram path  │
│     • Keyframe: single_packet() → one fragment_count=1 packet,      │
│                 reliable per-IDR uni stream path                    │
│         │       attach HostFrameTiming + stream_epoch + frame_seq   │
│         ▼                                                           │
│   tether-transport::Server                                          │
│     • P-frames → QUIC datagrams (unreliable, low latency)           │
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
│     • drops cross-epoch fragments                                   │
│     • wall-clock-evicts pending frames older than max_pending_age   │
│       (500ms default) on every fragment                             │
│         │   complete encoded frame                                  │
│         ▼  (bounded crossbeam channel, capacity 8)                  │
│   tether-decode std::thread (owns the Decoder)                      │
│     • VAAPI submit + drain                                          │
│     • on decode error / libavcodec warn, fire rate-limited          │
│       (500ms) ForceIdr via stashed tokio::runtime::Handle           │
│     • vaSyncSurface + vaExportSurfaceHandle → DRM_PRIME             │
│         │   Frame::Gpu(GpuFrame { DmaBuf { fd, stride, modifier } })│
│         ▼  (LatestFrame single-slot drop-oldest)                    │
│   tether-render::gpu                                                │
│     • imports VAAPI dma-buf as two wgpu textures (Y + UV planes)    │
│     • fragment shader does NV12→RGB matrix at present time          │
│     • wgpu present to the winit window                              │
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
to decoder), a dedicated `std::thread` named `tether-decode` (owns
the VAAPI decoder + auto-IDR rate-limit), and winit's main thread
running the wgpu render loop. Communication is via crossbeam channels
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
    // future: #[cfg(target_os = "windows")] D3D11Texture(...),
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
    #[doc(hidden)] _Phantom(PhantomData<&'a ()>),
}
```

The trait method is cross-platform; variants inside `GpuEncoderFrame`
are cfg-gated internally. The host's dispatch doesn't need a per-
platform `#[cfg]`. The Windows D3D11 variant slots in the same way
when that backend arrives.

**Decoder side** uses the same shape, mirrored: `Decoder::next_frame ->
Frame::{Cpu(DecodedFrame), Gpu(GpuFrame)}` where `GpuFrame.source` is a
`GpuFrameSource` enum with the same cfg gating. The decoder hands the
renderer a `GpuFrameGuard` (the shared `GpuResourceGuard` re-export) that
holds the source `AVFrame` ref alive until the renderer drops it.

**macOS host (shipping today).** ScreenCaptureKit emits NV12
`CMSampleBuffer`s; `tether-capture::macos` unwraps each to its
`IOSurface` and forwards as `CapturedFrame::Gpu(GpuCapturedSource::IOSurface(...))`.
`tether-codec::videotoolbox::VideoToolboxEncoder` wraps the IOSurface
in a fresh `CVPixelBuffer` (`CVPixelBufferCreateWithIOSurface`) and
feeds it to `h264_videotoolbox` / `hevc_videotoolbox` via the AVFrame
`data[3]` slot — no NV12 conversion step is needed (SCK delivers
NV12 video range natively), so the macOS host has no analogue of
`tether-gpuconvert`. Input injection is via `enigo`'s CGEvent
backend, sharing the modifier-reconciliation and HID→Key code with
the Linux libei path through `inject::enigo_backend`. macOS client
(VideoToolbox decode + Metal IOSurface→wgpu render + winit input
capture) is a separate plan; for now a macOS host streams to a
Linux VAAPI client.

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
- **DoS-relevant transport limits.** `MAX_VIDEO_STREAM_MESSAGE = 2 MiB`
  caps per-keyframe-stream allocation; `max_concurrent_uni_streams = 4`
  prevents a peer from opening thousands of streams and pinning
  receive-side buffers.

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
  { stream_epoch, frame_seq, t_capture, payload }`. The wire shape
  ships in V1; the Opus capture/encode/decode pipeline is its own
  future workstream (no protocol bump needed when it lands).
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

### Codec / chroma / depth negotiation

Video profile is negotiated host-authoritatively via two hello
extensions. The client advertises its decode capabilities under
`tether.cap.video.decode-profiles` as `Vec<VideoProfile { codec,
chroma, bit_depth }>`. The host intersects that set with its own
buildable encode profiles (from a `OnceLock`-cached
`supported_encode_profiles()` probe that calls the real
`VaapiEncoder::new` at 128×128 per triple) and picks the best mutual
match against a fixed preference list:

1. HEVC 4:4:4 8-bit (desktop-quality top rung — preserves text and UI
   chroma detail that 4:2:0 visibly smears).
2. HEVC 4:2:0 8-bit.
3. H.264 4:2:0 8-bit (universal floor; H.264 4:4:4 is absent because
   VAAPI has no encode profile for it).

The chosen profile is echoed in `tether.cap.video.encode-profile`;
`ServerHelloV1.chosen_codec` / `chosen_chroma` carry the same
information in legacy form so older clients can interoperate. Absent
client extension is treated as the universal floor.

Both the encoder (`VaapiEncoder::new` takes `VideoProfile`, switches
`sw_format` + VAAPI `profile=` string + BGRA→input swscale stage),
the gpuconvert bridge (NV12 with two-plane R8+Rg8 export vs. YUV444
with three-plane R8 export), and the renderer (NV12 fragment shader
with `Y + UV` bind group vs. YUV444 sibling with `Y + U + V`) branch
on the negotiated chroma at construction. Mid-session chroma switch
is not supported — same rebuild path as a mid-session resolution
change (encoder + bridge + render pipeline all reset).

The `tether.pixel-format` extension echoes the on-wire pixel format
of the encoded stream (`Nv12` for 4:2:0, `Yuv444p` for HEVC Main444)
so client decoders that wire their import path before the first SPS
arrives can pick the right plane layout up front.

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
   `ControlMessage::ForceIdr`. The host swap-and-zeros an `AtomicBool`
   so multiple requests between encode calls coalesce to one. GOP is
   long (~240 frames) and IDRs are driven by request, not cadence —
   the GOP is the safety net, not the primary recovery mechanism.
   The client's decode thread also rate-limits ForceIdr emission to
   one per 500 ms so a decode-error storm can't generate a keyframe
   storm on the host.
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
- `tether-input::inject::linux` uses `enigo` with the `libei_tokio`
  feature. Portal-mediated, same permission model as screen capture.
  No X11 backend — the host has to be a Wayland session anyway since
  capture is PipeWire-anchored.
- Cursor updates are a separate channel from input events so the client
  can render a remote cursor sprite over the decoded video without
  waiting for the next encoded frame.

Client side: native `winit` event loop captures keyboard/mouse, encodes
to a transport-agnostic HID-style `InputEvent`, sends to host.

---

## What's deliberately out of scope (today)

Listed to set expectations; each is a real follow-up, not a "never":

- **macOS client and Windows backends.** macOS host ships today
  (see "Cross-platform additivity"); the macOS client (VideoToolbox
  decode + Metal IOSurface→wgpu render + winit input capture) and
  the Windows host/client (DXGI / Media Foundation / D3D11) are
  follow-up modules per platform.
- **AV1.** H.264 and HEVC are supported; AV1 needs a different VAAPI
  decoder probe (no `vaapi_av1` encode entrypoint on most current
  Intel iGPUs) and a separate codec_id path. The probe stub returns
  `CodecNotFound` for AV1 today.
- **HEVC Main10 / HDR.** HEVC Main 8-bit only. Main10 requires a
  10-bit capture path on the host (PipeWire format negotiation) and
  an HDR-aware renderer on the client (BT.2020 + PQ/HLG in the
  fragment shader). Both are real, neither exists yet.
- **NAT traversal.** LAN direct only. QUIC's pluggable transport makes
  adding ICE later straightforward; today the user runs the client
  binary with a host IP.
- **Adaptive bitrate.** Default scales with resolution + fps + codec
  via `derive_bitrate_kbps` in the host (1080p60 H.264 = 8 Mbps;
  HEVC × 0.7). The trait exposes `supports_changing_bitrate` /
  `set_bitrate_kbps` so a closed-loop QoS controller can drive this
  per-window — not yet wired up.
- **Profile + rate-control probe (VAAPI).** Today we hard-code
  `profile=main` and `rc_mode=VBR`. Apollo carries an AMD-specific
  CBR/VBR probe and Sunshine probes the profile table via
  `vaQueryConfigProfiles`. Both are the principled extensions, both
  need a small libva FFI add (vaQueryConfigProfiles,
  vaGetConfigAttributes) and an AMD test box to validate against.
- **Multi-monitor.** Single primary monitor capture; the
  keyframe-sender path assumes one fragmenter, and `display = 0` is
  hard-coded throughout the host send loop.
- **Audio.** Video + input only.
- **Periodic safety-net IDR.** Sunshine deliberately omits this on
  the NVENC path; we follow suit. Worth adding if we ever see a
  "client went silent without observing decode failure" stall mode
  (display sleep, decoder lockup) — cheap insurance, ~20 LOC.
- **FEC on keyframes.** Reliable per-IDR streams give us deterministic
  1-RTT recovery with no bandwidth overhead in the no-loss case.
  Sunshine/Apollo/Moonlight use FEC because RTP-over-UDP has no
  reliable side-channel; we have QUIC streams and don't need to
  reinvent FEC.
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
