# Tether — Architecture

Tether is a low-latency open-source remote desktop in Rust. The first
working end-to-end target is **Linux ↔ Linux on a LAN over QUIC**, with
H.264 hardware encode/decode and a zero-copy capture→encode→decode→
render path. The crate boundaries are shaped so adding macOS
(ScreenCaptureKit / VideoToolbox / IOSurface) and Windows (DXGI /
Media Foundation / D3D11) backends is a new module per platform, not a
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
│     • h264_vaapi encode → Annex-B NAL units                         │
│         │                                                           │
│         ▼                                                           │
│   tether-protocol::video::FrameFragmenter                           │
│     • split into MTU-sized fragments, attach HostFrameTiming +      │
│       stream_epoch + frame_seq + fragment_index                     │
│         │                                                           │
│         ▼                                                           │
│   tether-transport::Server (QUIC datagrams)                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼  network (LAN today; NAT TBD)
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│ CLIENT                                                              │
│                                                                     │
│   tether-transport::Connection (QUIC datagrams)                     │
│         │                                                           │
│         ▼                                                           │
│   FrameDefragmenter (drops cross-epoch fragments)                   │
│         │   complete H.264 frame                                    │
│         ▼                                                           │
│   tether-codec::vaapi::VaapiDecoder                                 │
│     • h264_vaapi decode into a VASurface from an owned surface pool │
│     • vaExportSurfaceHandle → DRM_PRIME (per-frame fresh export)    │
│         │   Frame::Gpu(GpuFrame { DmaBuf { fd, stride, modifier } })│
│         ▼                                                           │
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

---

## Cross-platform additivity

Two type seams keep the per-platform GPU integration from leaking into
the cross-cutting code:

**Capture side — `CapturedFrame` enum** (`tether-capture/src/lib.rs`):

```rust
pub enum CapturedFrame { Cpu(CpuFrame), Gpu(GpuCapturedFrame) }

pub enum GpuCapturedSource {
    #[cfg(target_os = "linux")] DmaBuf(CapturedDmaBuf),
    // future: #[cfg(target_os = "macos")] IOSurface(...),
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
    #[doc(hidden)] _Phantom(PhantomData<&'a ()>),
}
```

The trait method is cross-platform; variants inside `GpuEncoderFrame`
are cfg-gated internally. The host's dispatch doesn't need a per-
platform `#[cfg]`. When IOSurface/D3D11 backends arrive, they add a
variant to the enum and an `encode_gpu` impl on their `Encoder` — no
change to the trait shape or the call site.

**Decoder side** uses the same shape, mirrored: `Decoder::next_frame ->
Frame::{Cpu(DecodedFrame), Gpu(GpuFrame)}` where `GpuFrame.source` is a
`GpuFrameSource` enum with the same cfg gating. The decoder hands the
renderer a `GpuFrameGuard` (the shared `GpuResourceGuard` re-export) that
holds the source `AVFrame` ref alive until the renderer drops it.

The macOS migration path is roughly: add `tether-capture/src/macos.rs`
emitting `CapturedFrame::Gpu(GpuCapturedSource::IOSurface(...))`, add a
`VideoToolboxEncoder` impl in `tether-codec` whose `encode_gpu`
unwraps `GpuEncoderFrame::IOSurface`, and the host's run loop is
unchanged. Same for the decoder/renderer pair on the client.

---

## Protocol shape

`tether-protocol` defines three kinds of messages, each on its own
QUIC primitive:

- **Control** — a bidirectional QUIC stream, length-prefixed
  bincode-encoded `ControlMessage` enum. Carries the handshake
  (`ClientHello` / `ServerHello`, both tagged-enum envelopes for
  forward compatibility), `Goodbye` (with a machine-readable
  `GoodbyeCode`), `ForceIdr` requests, cursor updates, input
  batches, and display-dim notifications. Never used for video.
- **Video** — QUIC datagrams. Each datagram carries one
  `VideoPacket::First { display, stream_epoch, frame_seq,
  fragment_count, meta, payload }` or
  `VideoPacket::Continuation { display, stream_epoch, frame_seq,
  fragment_index, payload }`, where `meta` is `VideoFrameMeta {
  timing: HostFrameTiming, keyframe, input_echo, dimensions }`. The
  receiver defragments, drops any fragments whose epoch doesn't
  match the current one (resolution change → epoch bump), and
  forwards complete frames to the decoder.
- **Input/cursor** — separate from control because input events are
  high-rate and don't need re-transmission; sent as small unreliable
  datagrams keyed by a monotonic sequence number.

Three non-negotiable invariants tracked end-to-end:

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

- **macOS and Windows backends.** Crate boundaries are ready
  (see "Cross-platform additivity"); the modules don't exist yet.
- **AV1 / HEVC.** H.264 only. `tether-codec` is set up to accept new
  `Encoder`/`Decoder` impls without touching the trait.
- **NAT traversal.** LAN direct only. QUIC's pluggable transport makes
  adding ICE later straightforward; today the user runs the client
  binary with a host IP.
- **Adaptive bitrate.** Fixed 4 Mbps. The trait exposes
  `supports_changing_bitrate` / `set_bitrate_kbps` so a control loop
  can drive this when QoS data arrives — not wired up.
- **Multi-monitor.** Single primary monitor capture.
- **Audio.** Video + input only.
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
  silently falling back to a SW path that would tank latency.
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
