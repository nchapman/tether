# Tether — Codec & Capture Capability Matrix

This document is the reference for what video profiles, chroma formats,
and bit depths Tether can actually move end-to-end on each platform,
and — critically — *why* each limit exists. We make exactly two kinds
of claim here:

1. **Hard limits** — backed by a spec, header, driver source, or
   vendor documentation. These don't need probing because the answer
   cannot change without a new OS release or driver version. We cite
   the source.
2. **Empirical claims** — things we verified at runtime by actually
   trying. Anything not in category 1 belongs to category 2. We
   never guess: either we have a citation or we have a probe.

The motivating moment for this document was discovering that
VideoToolbox on Apple Silicon *does* hardware-decode HEVC Main 4:4:4
to a `'444v'` NV24 IOSurface, despite our renderer having rejected
4:4:4 input on macOS for months on the assumption that "VT is 4:2:0
only." That assumption came from misreading FFmpeg's wrapper, not
from a real Apple limitation. The fix wasn't a one-line filter; it
was a probe system that tries an actual encode + decode round trip
per profile. This doc exists so we don't repeat that mistake — and
so future contributors know which constraints are real walls and
which are just untested doors.

---

## Layers and where limits come from

A single frame on the host passes through four capability gates
before the wire, and three more on the client. Each layer can be
the binding constraint, and each has a different reason for its
limits:

```
HOST                                      CLIENT
─────────────────────────────────         ─────────────────────────────
1. Capture (SCK / PipeWire)               5. Decoder (VT / VAAPI)
   → what the OS lets us scrape              → what the GPU lets us decode
                                              and what fourcc it emits
2. GPU convert (wgpu compute)
   → what shader formats we support       6. Renderer import (wgpu HAL)
                                              → what texture formats can
3. Encoder (VAAPI / VT)                      come in via dma-buf or
   → what the silicon + driver               IOSurface
   + FFmpeg wrapper accept
                                          7. Sampler / shader
4. Wire (negotiated profile)                  → numeric range correctness
                                              (matters for 10-bit)
```

The interesting failures are at the seams: an encoder layer that
*advertises* a profile via the FFmpeg API but doesn't actually
accept the matching pixel format; a decoder that opens the codec
context but silently falls back to software; a kernel/driver
combination that exposes a `VAProfile` enum but rejects the
encoder configuration call. Spec-level capability and runtime
capability are not the same thing, and this is the central
reason we cannot get away with table-driven matchmaking — we need
real probes.

---

## How to read this doc

Each section below has the same shape:

- **What's hard-limited** — entries that we will never bother to
  probe. Each comes with a source citation and a one-sentence
  explanation of *why* the limit exists at that layer.
- **What's probed** — entries where the answer depends on
  hardware, driver version, OS release, or a combination, and we
  resolve it at runtime by trying the operation. Each comes with
  the location of the probe and a note on what failure looks like.
- **What's open** — entries we believe should work but have not
  yet wired up. These become future probes.

When you add a new encoder or decoder profile, the test is whether
you can place it in one of the first two buckets. If you can't,
you need to add a probe.

---

## Layer 1 — ScreenCaptureKit (macOS capture)

ScreenCaptureKit (SCK) is Apple's modern capture API and the only
way to get composited screen pixels on macOS without private
entitlements. We use the `screencapturekit` Rust crate v6.0.1
(`SCStreamConfiguration::pixelFormat`). Apple documents six
accepted pixel formats:

| Fourcc | Chroma     | Bit depth | Notes                              |
| ------ | ---------- | --------- | ---------------------------------- |
| `BGRA` | 4:4:4 RGB  | 8         | Default until macOS 26 on Apple Silicon |
| `l10r` | 4:4:4 RGB  | 10        | ARGB2101010 packed                 |
| `420v` | 4:2:0 YCbCr| 8         | Biplanar, video range — what we use today |
| `420f` | 4:2:0 YCbCr| 8         | Biplanar, full range               |
| `xf44` | 4:4:4 YCbCr| 10        | Biplanar full-range — Apple's documented 10-bit YCbCr capture format |
| `RGhA` | 4:4:4 RGB  | half-float| 64-bit HDR                         |

Sources:
[`screencapturekit-rs` crate source](https://github.com/doom-fish/screencapturekit-rs/blob/main/src/stream/configuration/pixel_format.rs);
[Apple WWDC24 "Capture HDR content with ScreenCaptureKit"](https://developer.apple.com/videos/play/wwdc2024/10088/);
[Xamarin macOS bindings reference](https://github.com/xamarin/xamarin-macios/wiki/ScreenCaptureKit-macOS-xcode16.0-b1).

### Why 8-bit 4:4:4 capture (`'444v'`, `'444f'`) is *not* on this list

The CoreVideo framework defines the fourcc constants
`kCVPixelFormatType_444YpCbCr8BiPlanarVideoRange` (`'444v'`) and
`kCVPixelFormatType_444YpCbCr8BiPlanarFullRange` (`'444f'`), and
VideoToolbox emits them when decoding HEVC Main 4:4:4. But
ScreenCaptureKit's `pixelFormat` property has its own accepted-set
that is a *subset* of CoreVideo's universe. Apple documents the
six formats above and does not document `'444v'`/`'444f'` as
acceptable inputs.

The Rust crate exposes an `Unknown(FourCharCode)` escape hatch, so
we *can* pass `'444v'` to SCK and see what happens — SCK might
accept it, might reject the configuration, or might accept it and
silently produce zero frames. We do not know without trying. This
is the most consequential probe in this layer: if SCK accepts
`Unknown('444v')`, then Mac → anything HEVC 4:4:4 8-bit is end-to-end
viable assuming the encoder side works. If it doesn't, the only
8-bit 4:4:4 capture path on macOS is to capture BGRA and run a
gpuconvert shader.

### What's hard-limited

- **H.264 4:4:4 capture format**: there is no H.264 4:4:4 SCK
  format because there is no biplanar H.264-friendly 4:4:4 fourcc
  in SCK's enum that maps to anything an H.264 encoder accepts.
  This is moot because we don't ship H.264 4:4:4 anyway (see
  Layer 3).
- **4:2:2 capture**: SCK exposes no 4:2:2 format at all, in any
  bit depth. The closest is `xf44` (4:4:4 10-bit) which the
  encoder could downsample. This is a real Apple-documented limit.

### What's probed

`tether-capture::macos::probe_capture_pixel_formats()` runs at
host startup, attempts `SCStream::start_capture` for each
candidate, and (for `PixelFormat::Unknown(FourCharCode)`
variants — the SCK escape hatch) waits briefly for the first
sample and verifies its delivered fourcc matches the requested
one. Acceptance signals are stored in `SckCaptureCapability` and
logged; the live stream still uses `420v` regardless until the
renderer wires up the higher-chroma / 10-bit IOSurface import
paths.

Empirical results on M4 Max (macOS 26, SCK v6.0.1):

| Format         | Accepted | Frame verification |
| -------------- | :------: | ------------------ |
| `BGRA`         | ✅       | not required (documented enum variant) |
| `420v` (NV12 video range)    | ✅ | not required (documented) |
| `420f` (NV12 full range)     | ✅ | not required (documented) |
| `'444v'` (NV24 video, Unknown escape) | ✅ | delivered samples carry `'444v'` |
| `'444f'` (NV24 full, Unknown escape)  | ✅ | delivered samples carry `'444f'` |
| `xf44` (10-bit 4:4:4)        | ✅ | not required (documented) |

Two consequential findings:

1. **SCK does deliver `'444v'` / `'444f'` on M4 Max** despite
   being undocumented in Apple's `SCStreamConfiguration.pixelFormat`
   list. The frame-arrival check rules out the "SCK accepts the
   FourCharCode but silently downgrades" failure mode for these
   two specifically. Combined with VT's confirmed Main 4:4:4
   encode acceptance (see the encode section above), macOS host
   → anything HEVC Main 4:4:4 8-bit is reachable end-to-end.
2. **`xf44` is reachable** — the documented 10-bit 4:4:4 capture
   format. The renderer-side 10-bit IOSurface import is the
   remaining gate.

---

## Layer 2 — PipeWire (Linux capture)

PipeWire's portal-based capture negotiates DMA-BUF formats
advertised by the compositor and importable by the encoder side.
We negotiate BGRA + linear modifier today and run a compute shader
to produce NV12 / NV24 / XYUV depending on the chosen encode profile.

### What's hard-limited

- **DMA-BUF tiling modifier**: capture-side BGRA can use any
  modifier the GPU importer accepts (we advertise the union of
  what wgpu/Vulkan can import). **Export side is pinned to
  `DRM_FORMAT_MOD_LINEAR`** because tiled modifiers don't have a
  portable shared-allocation contract that VAAPI's DRM_PRIME
  importer honours. See the comment in `tether-gpuconvert`.

### What's probed

- **Importer compatibility at startup**: `importable_dmabuf_modifiers()`
  walks the wgpu+Vulkan+features chain and returns the modifier
  set the encoder side can actually import. If this returns empty
  we exit at startup rather than fail mid-session.

---

## Layer 3 — VideoToolbox encode (macOS host)

VideoToolbox is Apple's hardware video API. We drive it through
FFmpeg's `hevc_videotoolbox` and `h264_videotoolbox` wrappers,
which means our capability surface is the **intersection** of:

1. What the Apple Silicon media engine implements,
2. What VideoToolbox exposes via its public API
   (`kVTProfileLevel_*` constants), and
3. What FFmpeg's wrapper bothers to plumb through as accepted
   `AVPixelFormat`s on the codec's `pix_fmts` list.

Any of those three layers can be the binding constraint, and they
do not always agree. This is why FFmpeg appearing to advertise no
`p010le` input for `hevc_videotoolbox` does not necessarily mean
the underlying Apple API can't do it — the wrapper might just not
have plumbed it. Conversely, the wrapper listing a profile like
`main42210` doesn't prove VT will actually accept matching input.

### What's hard-limited

- **H.264 4:4:4 encode**: VideoToolbox exposes no H.264 4:4:4
  profile constant, and the FFmpeg `h264_videotoolbox` profile
  list contains only `baseline / constrained_baseline / main /
  high / constrained_high / extended` (all 4:2:0 8-bit). There's
  no path to drive a 4:4:4 H.264 encode on Apple Silicon through
  any public API. Source: `ffmpeg -h encoder=h264_videotoolbox`.

### What's probed

`tether-codec/src/videotoolbox/probe.rs` attempts every profile
in `PROFILE_PREFERENCE` against the live VT wrapper — no
pre-filter. `VideoToolboxEncoder::new` maps `(chroma, bit_depth)`
to an FFmpeg pix_fmt via `vt_sw_format` and lets `encoder.open()`
be the authority on whether VT accepts that combination. The
probe layer takes any error path as `encode=false`.

Empirical results on M4 Max (Homebrew ffmpeg, VideoToolbox):

| Profile                  | encode | decode | output IOSurface |
| ------------------------ | :----: | :----: | ---------------- |
| HEVC 4:2:0 8-bit (Main)  | ✅     | ✅     | `'420v'`          |
| HEVC 4:2:0 10-bit (Main10)| ✅    | ✅     | needs probe      |
| HEVC 4:4:4 8-bit (Rext)  | ✅     | ✅     | `'444v'` (NV24)   |
| HEVC 4:4:4 10-bit (Rext) | ✅     | ✅     | needs probe      |
| H.264 4:2:0 8-bit        | ✅     | ✅     | `'420v'`          |

Two consequential findings vs. the original audit:

1. **HEVC Main444 encode is real on M4 Max.** The earlier "VT
   has no Main444 encode" claim was an artifact of our own probe
   short-circuit pre-filtering non-(Yuv420 8-bit) before reaching
   the encoder. Once the short-circuit was removed, the FFmpeg
   wrapper accepts NV24 / P410LE input. The probe currently
   verifies *encoder acceptance*, not *bitstream conformance* —
   it's possible VT silently downsamples to 4:2:0 internally;
   confirming that needs an encode→decode round-trip with an
   IOSurface fourcc assertion (planned follow-up).
2. **HEVC Main10 encode is real on M4 Max.** Same shape — FFmpeg
   accepts P010LE input. Same bitstream-conformance caveat applies.

### What's open

- **Per-generation results (M1 / M2 / M3 / M4).** Probe matrix
  above is M4 Max only. Pre-M3 silicon may be more limited for
  4:2:2 specifically.
- **Bitstream-conformance round-trip.** Encoder acceptance ≠
  correct chroma in the emitted bitstream. A real probe that
  encodes → decodes → asserts the output IOSurface fourcc
  matches the input chroma is the next-strongest signal short
  of decoding on a different platform.
- **HEVC Main422 8/10-bit encode.** Not exercised — would
  require adding `ChromaSubsampling::Yuv422` on the wire (see
  Commit 5 in the rollout plan). The 4:2:2 fixtures are checked
  in (`hevc_yuv422_*bit.idr`) ready for that change.

---

## Layer 3 (Linux) — VAAPI encode

VAAPI is the Linux equivalent surface. Capability is the
intersection of:

1. The libva `VAProfile` enum (the universe of what *can* exist),
2. What the user's driver (`intel-media-driver` / `i965` / Mesa
   radeonsi / `libva-vdpau-driver`) implements,
3. What FFmpeg's `vaapi_encode_*` wrappers map to,
4. What input pixel format / DRM fourcc the
   `vaapi_drm_format_map` recognises for `av_hwframe_map(DRM_PRIME →
   VAAPI)`.

### What's hard-limited

- **H.264 4:4:4 encode**: there is no `VAProfileH264*High444*`
  entry for *encode* in the libva spec, and no driver ships one.
  Tether's `vaapi::encoder::new` early-returns. Source:
  [libva `va.h`](https://github.com/intel/libva/blob/master/va/va.h).
- **HEVC 4:4:4 encode on AMD / NVIDIA**: no Mesa or
  libva-vdpau-driver build exposes `VAProfileHEVCMain444` for
  encode. Intel iHD Ice Lake+ is the only `VAProfileHEVCMain444`
  encode-capable driver in the wild. Source:
  [Intel Quick Sync support matrix article 000057555](https://www.intel.com/content/www/us/en/support/articles/000057555/graphics.html).
- **Planar YUV444P DMA-BUF import**: FFmpeg's
  `vaapi_drm_format_map` has no entry for planar 4:4:4. Only
  packed `DRM_FORMAT_XYUV8888` (FFmpeg's `VUYX`) survives the
  `av_hwframe_map` from DRM_PRIME into a VAAPI surface. This is
  why our 4:4:4 gpuconvert shader produces packed XYUV instead
  of biplanar. Source: FFmpeg `libavutil/hwcontext_vaapi.c`.
- **DRM_PRIME export of 4:4:4 8-bit is XYUV-only**: same map,
  used in reverse. The renderer-side import path assumes XYUV
  for 4:4:4 on Linux because that's what comes out.

### What's probed

The probe at `tether-codec/src/vaapi/probe.rs` constructs an
encoder, encodes one frame, and tries the matching decoder. This
catches three failure modes the spec doesn't:

- Drivers that expose a `VAProfile` but reject the encoder
  configuration call (commonly older Intel driver versions on
  marginal silicon).
- Drivers that accept the configuration but produce broken
  output (probe round-trips through a decoder).
- Format-map mismatches between encoder input and what
  gpuconvert can produce.

10-bit HEVC encode (Main10, Main422_10, Main444_10) is currently
not probed and not in `PROFILE_PREFERENCE`. **No documented
driver supports any 10-bit VAAPI encode profile across vendors**,
but that absence is an absence of documentation, not a proof —
needs an actual probe before we trust it.

---

## Layer 5 — VideoToolbox decode (macOS client)

The Apple Silicon HEVC decoder block is *much* more capable than
the encoder block — and much more capable than the FFmpeg wrapper
makes it look. The most reliable secondary source we found is the
[Jellyfin/StaZhu enable-chromium-hevc reference](https://github.com/StaZhu/enable-chromium-hevc-hardware-decoding),
which states: "Apple Silicon Mac supports HEVC Rext hardware
decoding of 8 ~ 10b 4:0:0, 4:2:0, 4:2:2, 4:4:4 contents, and
software decoding of 12b ... contents."

### What's hard-limited

- **HEVC 12-bit anything → software fallback**: the Apple Silicon
  hardware decoder block doesn't include 12-bit support; VT will
  open the session but emit decoded frames into a CPU-backed
  buffer rather than an IOSurface. Tether's `Frame::Gpu` vs
  `Frame::Cpu` discrimination treats this as "not supported for
  the hot path."
- **FFmpeg `hevc_videotoolbox` decoder is format-blind**:
  `avcodec_open2()` succeeds regardless of stream profile because
  the wrapper declares `AV_PIX_FMT_VIDEOTOOLBOX` globally and
  defers to `get_format()`. This means we **cannot trust
  `open()`** as a capability signal — the only honest answer
  comes from feeding a real fixture and seeing what comes back.
  This is *the* reason the probe round-trips a real IDR per
  profile rather than just opening codec contexts.

### What's probed

`tether-codec/src/videotoolbox/probe.rs::probe_decode` submits a
checked-in fixture, calls `signal_eof()` to force VT's wrapper to
drain (it buffers the first packet pending a second packet or
EOF — discovered empirically), and asserts a `Frame::Gpu` result.
We currently probe HEVC 4:2:0 8-bit and HEVC 4:4:4 8-bit. We
should add:

- **HEVC Main10 4:2:0 10-bit** — output fourcc likely `'x420'`
  (or biplanar 10-bit), to be confirmed by probe.
- **HEVC Main422 8/10-bit** — output fourcc unknown.
- **HEVC Main444 10-bit** — output fourcc almost certainly
  `'xf44'` or `'P410'`; the difference matters for the renderer
  (packed vs biplanar).

### What's hard-confirmed about platform asymmetry

M1 / M2 / M3 / M4 are all identical for HEVC decode capability
per Softron's documentation and Jellyfin's empirical results.
AV1 hardware decode is M3+ only — not relevant to us today.

---

## Layer 6 — Renderer import (wgpu HAL)

The renderer doesn't care what codec produced the surface; it
cares about plane count, per-plane format, and DRM/IOSurface
fourcc. Today we support two layouts (`RenderLayout::Biplanar` and
`RenderLayout::PackedXYUV`) at 8-bit.

### What's hard-limited

- **wgpu pin at the trunk SHA in `Cargo.toml`** does not expose
  `TextureFormat::P010` or `TextureFormat::P410` as variants.
  Those exist on newer wgpu trunk but behind feature flags and
  not at our pinned commit. **10-bit biplanar import on our
  current pin must go via `R16Unorm` + `Rg16Unorm` per plane.**
  Bumping the pin to acquire P010/P410 is an option; doing
  R16/Rg16 manually is the path with the fewest dependencies.
- **MSB-aligned 10-in-16 sampler scaling**: a 10-bit value max
  (1023) stored in bits [15:6] of a 16-bit word reads through an
  `R16Unorm` sampler as `65472 / 65535 ≈ 0.999`, **not 1.0**.
  This is a hardware behaviour of normalized samplers, not a
  bug. The shader must compensate (multiply by `65535.0 /
  65472.0`) or use an integer load and unpack manually. There
  is no escape — a sampler that "just gives 1.0" for max 10-bit
  values does not exist for this storage convention.

### What's probed

- **Importable dma-buf modifiers at startup** — already in place
  via `importable_dmabuf_modifiers()`. For 10-bit we'll need to
  extend this to verify `R16_UNORM` and `R16G16_UNORM` work over
  dma-buf with the modifiers we'd actually receive.
- **Metal IOSurface import of 16-bit biplanar** — needs a
  startup check that `MTLDevice::newTextureWithDescriptor:iosurface:plane:`
  accepts `R16Unorm` / `Rg16Unorm` on an actual `xf44` or `P410`
  IOSurface. Apple documents Metal supporting these texture
  formats since macOS 10.14, but binding them to an IOSurface
  plane has its own per-fourcc rules.

### What's open

- **Packed 10-bit 4:4:4 (`'xf44'` style)**: would need a wgpu
  format equivalent to `VK_FORMAT_A2R10G10B10_UNORM_PACK32`.
  wgpu's `Rgb10a2Unorm` is the candidate but its dma-buf import
  story is unverified at our pinned SHA.

---

## Shipped protocol matrix

`PROFILE_PREFERENCE` now contains five entries (best-first):

1. HEVC 4:4:4 10-bit — `VideoProfile::HEVC_10BIT_444`
2. HEVC 4:4:4 8-bit  — `VideoProfile::HEVC_8BIT_444`
3. HEVC 4:2:0 10-bit — `VideoProfile::HEVC_10BIT_420` (Main10)
4. HEVC 4:2:0 8-bit  — `VideoProfile::HEVC_8BIT_420` (Main)
5. H.264 4:2:0 8-bit — `VideoProfile::H264_8BIT_420` (universal floor)

`VideoProfile { codec, chroma, bit_depth }` has a `u8` `bit_depth`
field. The renderer's `RenderLayout::Biplanar16` variant + the
`luma_scale` shader uniform handle the 10-bit display side; the
encoder/decoder probes handle the host side. The negotiator picks
the first entry that appears in *both* the host's
`supported_encode_profiles()` and the client's advertised
`supported_decode_profiles()` — anything a given device's hardware
can't deliver gets filtered out by the probe layer automatically.

HEVC 4:2:2 8/10-bit is *not* yet in the matrix — it requires
`ChromaSubsampling::Yuv422` on the wire, which is a separate wire
change. The decode-probe fixtures for it are checked in
(`hevc_yuv422_*bit.idr`) ready for that change.

The principle from the introduction reapplies: every entry above
is conditional on a probe, not an assumption. The day Apple or
AMD ships a driver that flips one of the "no documented support"
cases into "works," the probe layer discovers it without code
changes here.

---

## Adding new profiles: the checklist

When adding a new profile to `PROFILE_PREFERENCE`:

1. **Does it require new layer-3 (encode) capability?** Add an
   encoder probe in `tether-codec/src/{vaapi,videotoolbox}/probe.rs`.
   Probe must do a real encode of a real frame, not just open
   the codec context.
2. **Does it require new layer-5 (decode) capability?** Add a
   matching decoder probe with a checked-in fixture in
   `crates/tether-codec/fixtures/probe/`. Probe must consume the
   fixture and assert `Frame::Gpu` (not `Frame::Cpu`) — the
   `Frame::Cpu` discriminator is what catches silent SW fallback.
3. **Does it require new layer-1 (capture) capability?** Add a
   capture-side probe in `tether-capture::{macos,linux}` that
   attempts the format and reports back. Note that SCK in
   particular silently does the wrong thing for some
   configurations — the probe must wait for actual frames.
4. **Does it require new layer-6 (renderer) capability?** Add a
   matching `RenderLayout` variant if needed, then a startup
   import probe similar to `importable_dmabuf_modifiers()`.
5. **Update `VideoProfile` wire round-trip tests** in
   `tether-protocol` to cover the new combo (bit_depth in
   particular — easy to forget).
6. **Update this document** with the new entry's source
   (hard-limit citation) or probe location (empirical).

If you find yourself adding a runtime-conditional or `cfg`
filter in `apps/tether-{host,client}` to exclude a profile that
the probe layer is reporting as supported, **stop**. That's the
shape of the bug we wrote this doc to prevent. The profile
either works end-to-end or the probe was wrong; fix one of those,
don't paper over it downstream.
