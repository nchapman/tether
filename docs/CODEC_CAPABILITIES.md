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
1. Capture (SCK / PipeWire / DXGI)        5. Decoder (VT / VAAPI / D3D11VA)
   → what the OS lets us scrape              → what the GPU lets us decode
                                              and what fourcc it emits
2. GPU convert (wgpu compute,
   ID3D11VideoProcessor on Windows)       6. Renderer import (wgpu HAL)
   → what shader formats we support          → what texture formats can
                                              come in via dma-buf or
3. Encoder (VAAPI / VT / D3D11)              IOSurface
   → what the silicon + driver
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
in `PROFILE_PREFERENCE` against the live VT wrapper. The encode
probe is an **end-to-end chroma-survival round-trip**: encoder
constructed at the requested profile, fed a high-frequency red /
green BGRA stripe pattern (full chroma swing at 1px pitch — any
silent 4:4:4 → 4:2:0 downsample collapses adjacent columns into
a uniform yellow bar), output packets round-tripped through a
fresh VT decoder, and the decoded IOSurface fourcc must land in
the expected family for the requested profile. Encoder
*acceptance* alone (the encoder opens, packets emit) is not
enough — VT has documented history of accepting input it then
silently transcodes (e.g. H.264 4:2:2 → 4:2:0), and our pre-
round-trip probe was over-claiming HEVC 4:4:4 encode for exactly
that reason. The fourcc check rejects silent downsample.

Empirical results on M4 Max (Homebrew ffmpeg, VideoToolbox):

| Profile                  | encode | decode | output IOSurface (encode round-trip) |
| ------------------------ | :----: | :----: | ----------------------------------- |
| HEVC 4:2:0 8-bit (Main)  | ✅     | ✅     | `'420v'`                            |
| HEVC 4:2:0 10-bit (Main10)| ✅    | ✅     | `'x420'`                            |
| HEVC 4:4:4 8-bit (Rext)  | ❌ (silent downsample) | ✅ | encoder produces `'420v'` |
| HEVC 4:4:4 10-bit (Rext) | ❌ (silent downsample) | ✅ | encoder produces `'x420'` |
| H.264 4:2:0 8-bit        | ✅     | ✅     | `'420v'`                            |

Per-row reading: an `encode=❌` here means the VT encoder accepts
the input (the FFmpeg wrapper plumbs it) and produces a bitstream,
but the resulting bitstream's chroma format doesn't match what we
asked for — the silent-downsample case. The named hardware test
`videotoolbox_round_trip_chroma_matrix` cross-checks the probe's
`encode` bit against the same round-trip the probe runs internally,
and a disagreement panics; that's the regression guard against the
probe ever silently slipping past a future downsample.

### What's open

- **Per-generation results (M1 / M2 / M3 / M4).** Probe matrix
  above is M4 Max only. Pre-M3 silicon may be more limited for
  4:2:2 specifically and may also accept-then-silently-downsample
  for the 4:4:4 cases the probe correctly rejects on M4 Max.
- **HEVC Main422 8/10-bit encode.** Not exercised — would
  require `ChromaSubsampling::Yuv422` on the wire. The 4:2:2
  fixtures are checked in (`hevc_yuv422_*bit.idr`) ready for
  that change.

The SPS second-signal probe noted as open in a prior revision
landed in commit `ce6aa94` — `crates/tether-codec/src/bitstream_sps.rs`
parses `chroma_format_idc` + `bit_depth_luma_minus8` directly from
the encoder's emitted SPS NAL and cross-checks against the
IOSurface fourcc family. An explicit disagreement between the two
signals fails the probe; an absent SPS signal (unmodeled profile)
falls back to fourcc-only. The probe layer now has two
independent gates against silent transforms.

### Live macOS 10-bit pipeline (M-series)

As of commit `513f4c7` the producer side is wired:

- SCK probes the eight pixel formats it might accept and caches
  the result for the process lifetime
  (`tether_capture::macos::probe_capture_pixel_formats`).
- `tether_capture::macos::sck_pixel_format_for_profile(profile)`
  maps each negotiated `VideoProfile` to the SCK fourcc the
  capture should configure, mirroring `vt_sw_format` on the
  encoder side so the IOSurface fourcc and the encoder's
  `sw_format` agree by construction.
- `VideoToolboxEncoder::submit_iosurface` accepts the matching
  fourcc families per `(chroma, bit_depth)` and refuses
  cross-bucket submissions, replacing the previous hard NV12-only
  guard. The encoder's `vt_sw_format` already supports `P010LE`
  (10-bit 4:2:0), `NV24` / `P410LE` (4:4:4 8 / 10-bit).
- The host's `capture_filtered_encode_profiles` (macOS arm)
  filters the encoder-probed profile list down to what the SCK
  probe says this Mac can deliver. On M4 Max + macOS 26 that
  intersects to HEVC 4:2:0 8-bit + 10-bit (Main10); the 4:4:4
  rows are filtered out by the VT encode probe's silent-downsample
  detection, not by the SCK side.

The Linux producer side is the remaining gap (see the next
section).

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

### Verified-negative VAAPI rate-control knobs (Intel iHD)

Three rate-control knobs that look like they should be live-tunable
through FFmpeg's `vaapi_encode_*` wrappers are verified-negative on
Intel iHD (Meteor Lake) with FFmpeg n8.1.1. All three share a
mechanism: FFmpeg's `vaapi_encode_init_rate_control` builds the
`VAEncMiscParameterRateControl` misc buffer **once** at encoder
`init()` from `AVCodecContext.bit_rate` / `qmin` / `qmax`; the
per-frame issue path doesn't re-read those fields, and there's no
`AVOption` or `AVFrame` side-data plumbing that would let a runtime
change reach the driver. The `VaapiEncoder::unused_avoptions()`
accessor was added so hardware tests can detect when an AVOption
fell into the leftover dict (i.e. the driver / wrapper didn't
consume it).

| Knob | Hardware test | Outcome on Intel iHD |
| ---- | ------------- | -------------------- |
| `intra_refresh` (CIR / pseudo-IDR-free recovery) | `vaapi_intra_refresh_round_trip` | SKIPs with diagnostic eprintln; AVOption falls into the unused dict, output is indistinguishable from non-CIR |
| `qmin` floor (per-codec H264=19, HEVC=23, AV1=32 scaffolded in `vaapi/encoder.rs`, mirroring Sunshine) | `vaapi_min_qp_floor_reduces_bitstream` | SKIPs; floor change doesn't shrink bitstream |
| Live `bit_rate` retune (mid-session ABR) | `vaapi_bitrate_retune_changes_bitstream_size` | SKIPs; bitstream size unchanged before/after retune |

Consequence for the runtime: `VaapiEncoder` deliberately leaves
`supports_changing_bitrate` at the trait default `false`. The
host's ABR controller (`tether_session::abr::AbrController`) is
disabled entirely on VAAPI hosts as a result — this is correct
behaviour, not a bug. ABR will re-enable on Linux when a backend
that actually honours live retune lands (NVENC — issue #16). The
per-codec min-QP scaffolding stays in the encoder so it can be
hooked up the moment a backend does plumb it; today it's
documentation-grade only on VAAPI.

LTR (long-term reference frames; protocol-side
`ControlMessage::RequestRecovery` carries the wire payload today)
is parked on the same upstream gap: FFmpeg n8.1.1 has zero LTR
plumbing in `h264_vaapi` / `hevc_vaapi` (no AVOption, no AVFrame
side-data field, no AVCodecContext field). Sunshine's VAAPI
encoder falls back to IDR for `invalidate_ref_frames`; RustDesk
hardcodes `support_changing_quality = false` on VAAPI for the same
reason. Tracked as issue #11 (LTR), with the "live bitrate retune
upstream-blocked" finding closing the related #15. NVENC (issue
#16) is the path that may unlock both — caveat that runtime LTR
mark / use may require direct NVENC SDK calls rather than just
AVOption + AVFrame side-data, to be verified against the installed
FFmpeg version when that backend lands.

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

### Linux 10-bit encode — empirical state (post-handoff)

The handoff plan (`Bgra2P010DmaBuf` bridge + encoder bit_depth gate
lift + storage-image probe + host capture filter) shipped across
five commits. Each layer of the chain works on its own; the
end-to-end path stops at the driver layer.

| Layer | Build cell | Run cell |
|-------|-----------|----------|
| `tether_gpuconvert::Bgra2P010DmaBuf` | ✅ | ✅ Y plane reads back BT.709-correct red Y=250 and Cb=409 / Cr=960 on 10-bit storage |
| `tether_gpuconvert::storable_dmabuf_modifiers(R16/GR32)` | ✅ | ✅ Mesa intel-Vulkan advertises STORAGE_IMAGE on both for LINEAR |
| `tether_codec::vaapi::VaapiEncoder::new(Main10)` (avcodec_open2) | ✅ | ✅ on Intel iHD + FFmpeg 8.1 |
| `tether_codec::vaapi::VaapiEncoder::submit_dmabuf` (P010 fourcc match) | ✅ | ✅ table accepts P010 + R16/GR32 layer shapes |
| `av_hwframe_map(DRM_PRIME → VAAPI)` on P010 dma-buf | ✅ | ❌ **driver gap** — Intel iHD + Mesa + FFmpeg 8.1 rejects with "DRM format not supported by VAAPI". Both single-layer P010+2-planes and two-layer R16+GR32+1-plane descriptor shapes fail. |

The driver layer's `vaapi_drm_format_map` table doesn't include an
entry that matches the P010 descriptor on this combination. This is
the empirical answer to the earlier "no documented driver supports
any 10-bit VAAPI encode profile" question — confirmed for the M-series
hardware available locally (Intel Arc on Meteor Lake). AMD radeonsi
and NVIDIA nvidia-vaapi-driver are untested but documentation
suggests the same gap.

**Host doesn't advertise Main10 to clients** on a driver that hits
this gap: `apps/tether-host/src/main.rs::warm_gpuconvert_capability_cache`
runs a real `submit_dmabuf` round-trip at startup (not just the
storage-image probe), and the cache populates `false` if the round-trip
fails. The codec-side probe alone is construction-only for 10-bit
(can't tell apart "driver lacks dma-buf map entry" from "driver
genuinely supports the path") so the host's warm-time check is the
authoritative gate.

**Path forward** if/when a driver gains P010 dma-buf support:
- No code changes needed in tether — the probe will populate `true`
  and Main10 will appear in the advertised profile set
- The renderer round-trip test
  (`crates/tether-render/src/dmabuf_test.rs::dmabuf_zero_copy_roundtrip_hevc_main10`)
  will go from SKIP to PASS on that driver
- Cross-platform path (macOS host → Linux client at Main10) already
  works today; it's only the Linux *encode* side that hits the gap

### 4:4:4 10-bit — wired on the encode side via packed XV30

VAAPI's `vaapi_drm_format_map` lists `AV_PIX_FMT_XV30LE` (packed
10:10:10:2) as the only 10-bit 4:4:4 entry — planar P410 has no map
row, mirroring the 8-bit YUV444P→XYUV story. The host now produces
XV30 via `tether-gpuconvert::Bgra2Xv30DmaBuf`: a BGRA→Rgb10a2Unorm
compute pass writes a single packed plane into a dma-buf exported as
`VK_FORMAT_A2B10G10R10_UNORM_PACK32` (DRM_FORMAT_XV30, R=Y / G=U /
B=V / A=X under the Vulkan PACK32 channel mapping; the 2-bit X is
unused per VAAPI). Pitch alignment is 128 bytes / 32 luma pixels and
height alignment is 32 rows — Intel HEVC 4:4:4 encode reads 32×32
CTUs and the analogous pitch boundary lands at 4× the NV12-luma
constraint. The probe-side gate (`tether-probe/src/host/vaapi.rs`)
runs a real `submit_dmabuf` round trip through the same bridge, so
`(Yuv444, 10)` only ends up in the advertised set when the driver
actually accepts it. Decoder output format for the 4:4:4 10-bit case
is driver-dependent (RADV has emitted both packed XV30 and biplanar
P410-style 16-bit across Mesa versions) — the renderer-import side
assumes biplanar 16-bit (`RenderLayout::Biplanar16`); if a driver
emits packed XV30, the import errors with "expected 2 layers, got 1"
and a new `RenderLayout::Packed1010102` variant is needed. See
`gpu/import.rs:53` for the comment that surfaces this on the
failure path, and `dmabuf_test.rs::roundtrip_hevc_main444_10bit_identity`
for the hardware test that gates it on RADV/NVK.

---

## Layer 2 (Windows) — DXGI Desktop Duplication (capture)

`IDXGIOutputDuplication` hands us the composited desktop as a single
`ID3D11Texture2D`. We `CopyResource` it into a pool of owned
`DXGI_FORMAT_B8G8R8A8_UNORM` textures (so `ReleaseFrame` can be called
promptly) and pass the pool texture to the encoder on the **same**
shared `ID3D11Device`. Capture is therefore always **BGRA 8-bit** —
the same 4:4:4 RGB starting point as macOS's `BGRA`.

### What's hard-limited

- **BGRA 8-bit only.** Desktop Duplication delivers SDR desktops as
  B8G8R8A8. (HDR duplication via `R16G16B16A16_FLOAT` exists but we
  don't capture it — there is no HDR encode path on Windows yet.)
  Chroma/bit-depth is decided entirely at the encode layer below.

### What's probed

- Nothing format-wise — BGRA is unconditional. The one thing read from
  the capture device is the DXGI adapter's **PCI vendor ID**
  (`GetDesc1().VendorId`), which selects the encoder backend (below).

### Capture→encode handoff (not a capability, but load-bearing)

The capture thread and the encode loop run on separate threads sharing
one D3D11 immediate context. The handoff is a single-slot **drop-oldest**
mailbox plus a texture-pool free-list with an ownership handshake: a pool
slot is reused only once the frame's `release_guard` (`SlotReturn`) drops,
so the capture thread never overwrites a texture the encoder's Video
Processor is still sampling. Because both sides share one immediate
context, GPU commands execute in submission order and the channel's
happens-before edge orders the `CopyResource` ahead of the blit — no GPU
fence/keyed-mutex needed (Apollo needs a keyed mutex only because it uses
separate devices per component). This closed an earlier progressive-
corruption regression; see `tether-capture/src/windows.rs`.

---

## Layer 3 (Windows) — D3D11 encode (host)

The host converts the captured BGRA texture to NV12 (8-bit) or P010
(10-bit) with the fixed-function `ID3D11VideoProcessor`, then feeds it to
an FFmpeg hardware encoder selected by GPU vendor in
`backends_for_vendor(codec, vendor_id)`:

| Vendor (PCI ID)   | HEVC chain              | H.264 chain             |
| ----------------- | ----------------------- | ----------------------- |
| Intel (`0x8086`)  | `hevc_qsv` → `hevc_mf`  | `h264_qsv` → `h264_mf`  |
| AMD (`0x1002`)    | `hevc_amf` → `hevc_mf`  | `h264_amf` → `h264_mf`  |
| NVIDIA (`0x10de`) | `hevc_nvenc` → `hevc_mf`| `h264_nvenc` → `h264_mf`|
| unknown           | `hevc_mf` **only**      | `h264_mf` **only**      |

Unknown-vendor is MF-only on purpose: speculatively constructing a
foreign vendor's encoder (e.g. `hevc_amf` on an Intel GPU) faults inside
that vendor's runtime (`STATUS_ACCESS_VIOLATION`), not a recoverable
error. Media Foundation is the vendor-agnostic encoder and works on any
D3D11 GPU. (Production always passes a real `vendor_id` from DXGI, so the
live path only ever falls back to MF.)

### What's hard-limited

- **No 4:4:4.** The Video Processor's BGRA→YUV blit only produces 4:2:0
  surfaces (NV12 / P010). `D3D11Encoder::new` rejects 4:4:4 at
  construction, and the Windows host probe excludes it from the
  advertised set, so negotiation never picks a profile we'd silently
  downsample. A real 4:4:4 path needs custom HLSL conversion shaders
  (AYUV / Y410, as in Apollo) — not wired. This mirrors the macOS-encode
  4:4:4 gap exactly.
- **QSV hw_frames pool size = 1.** The Intel D3D11 driver rejects
  multi-slice NV12 texture arrays (`DXGI_ERROR_INVALID_CALL`), so the QSV
  path allocates one surface and reuses it every frame (`av_hwframe_map`,
  Apollo's pattern; `async_depth=1` keeps it synchronous).
- **QSV `low_power` / `low_delay_brc` are off.** Both put the media
  driver into a mode that demands an explicit per-frame
  `mfxEncodeCtrl.FrameType`, which our `pict_type`-only path doesn't
  supply — every frame is rejected `Invalid FrameType:0`
  (`AVERROR_INVALIDDATA`). They gave no measured latency win either (the
  real cost is GPU contention, below). Verified by
  `d3d11_qsv_gpu_encode_decode_roundtrip`.
- **AV1: not wired** (`backends_for_vendor` returns empty).

### Low-latency option set (per backend)

B-frames are off everywhere (`max_b_frames=0`, no reorder delay). Beyond
that, grounded in `ffmpeg -h encoder=…` on the linked build:

- **QSV**: `forced_idr=1`, `async_depth=1` (default would be 4).
- **AMF**: `usage=ultralowlatency`, `quality=speed`, `latency=1`,
  **`async_depth=1`** (amfenc defaults this to **16** — "higher values
  increase output latency"; this was a real latency bug), `forced_idr=1`,
  `gops_per_idr=1`.
- **NVENC**: **`delay=0`** (default is `INT_MAX`), `zerolatency=1`,
  `tune=ull`, `rc=cbr`, `surfaces=1`, `forced-idr=1`.
- **MF**: `hw_encoding=1`.

### What's probed

Unlike VAAPI/VideoToolbox, the Windows host does **not** run a
destructive per-profile encode probe: AMF has a single-session limit and
doesn't reliably release sessions on drop, so probing N profiles would
exhaust it for the live encoder. Instead `probe_host()` reports every
4:2:0 profile (H.264, HEVC Main, HEVC Main10) as supported and excludes
4:4:4; the live encoder fail-fasts with a clear error if the hardware
genuinely can't encode the negotiated profile. The QSV/AMF/NVENC GPU
round-trip *is* covered by hardware tests
(`d3d11_{qsv,amf,nvenc}_gpu_encode_decode_roundtrip`), each gated on the
present GPU vendor so it asserts on matching hardware and SKIPs elsewhere.

### Latency note (loopback)

A measured single-iGPU loopback session (host encode + client
decode/render/present on the same GPU) shows the encode stage dominating
at ~85–128 ms `avg_encode_ms`, while capture handoff (~30–50 ms), wire
send (~0.1 ms), and client decode (~4–8 ms) are all minor. This is GPU-
queue contention, not an encoder-config problem (isolated QSV encode is
~5 ms): `receive_packet`'s blocking MFX `SyncOperation` waits behind the
client's GPU work on the shared chip. The decisive next step is a
two-machine test, where the encode stage is expected to collapse. A
native oneVPL / NvEncodeAPI path is the lever if it doesn't.

---

## Layer 5 (Windows) — D3D11VA decode (client)

`D3D11Decoder` decodes H.264 / HEVC via D3D11VA and exports each decoded
surface **GPU-resident**: it `CopySubresourceRegion`s both planes of the
decode pool slice into a single `MISC_SHARED` biplanar staging texture
and hands the renderer a shared NT handle (`Frame::Gpu`). The format
follows the decode surface — `DXGI_FORMAT_NV12` for 4:2:0 8-bit,
`DXGI_FORMAT_P010` for 4:2:0 10-bit (Main10) — and is carried in
`D3D11DecodedTexture::format` so the native D3D11 renderer
(`tether_render::d3d11`) opens the matching per-plane SRVs (R8 + R8G8 for
NV12, R16 + R16G16 for P010) on its own device. There is no wgpu/Vulkan
bridge. The `download_frame_cpu` path remains only as an 8-bit NV12
fallback for the null-decode-texture anomaly.

Decode matches the encode side: 4:2:0 (NV12 / P010), no 4:4:4. The client
decode probe (`tether-probe/src/host/d3d11.rs`) routes through this GPU
export path, so a Main10 fixture decodes to P010 and comes back as
`Frame::Gpu` — confirming the full chain per profile. The renderer's
`supports_10bit_render` probes D3D11 P010 texture support to gate the
10-bit decode advert (R16/R16G16 plane sampling is an FL11.0 baseline, so
P010 texture support is the only real variable). Both 8-bit and 10-bit
are exercised end-to-end by `d3d11_coord_fixture_decode_render_roundtrip_8bit`
/ `_10bit` in `tether-render/src/d3d11/mod.rs`.

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

AV1 on macOS is held out **in both directions** today. Hardware
decode exists on M3 and M4 generation silicon and FFmpeg ships a
working `videotoolbox_av1` hwaccel, but tether has no hardware
test exercising that path, so we don't advertise decode either.
On the encode side, FFmpeg 8.1 has no `av1_videotoolbox` encoder
(verified: `ffmpeg -encoders | grep videotoolbox` returns only
H.264 / HEVC / ProRes; no patch in ffmpeg-devel as of 2026-05).
Whether any currently-shipped Apple Silicon exposes hardware AV1
encode at all is independently unconfirmed in public sources.

Both `vt_av_codec_id(Av1)` and `vt_codec_cname(Av1)` return
`CodecNotFound`, so the probe surfaces AV1 as Unsupported at the
codec-construction stage on both encode and decode, and AV1
disappears from `host_encode_profiles()` and
`host_decode_profiles()` on this platform. To re-enable decode,
land a `videotoolbox_av1_decode_smoke` hardware test using the
existing `tether-probe` fixtures and flip the decoder.rs arm.

---

## Layer 6 — Renderer import (wgpu HAL)

The renderer doesn't care what codec produced the surface; it
cares about plane count, per-plane format, and DRM/IOSurface
fourcc. Today we support three layouts: `RenderLayout::Biplanar8`
(NV12 / NV24, 8-bit), `RenderLayout::Biplanar16` (P010 / P410 /
`'xf44'`, 10-bit MSB-aligned in 16-bit cells), and
`RenderLayout::PackedXYUV` (DRM_FORMAT_XYUV8888, 4:4:4 8-bit).
A packed 10-bit 4:4:4 variant (`Rgb10a2Unorm`) is not yet wired
on the import side — see `## Layer 5 — VAAPI encode` for why
the producer side ships ahead of the consumer side.

### What's hard-limited

- **wgpu has no high-level multi-plane `TextureFormat::P010` /
  `P410`**, so 10-bit biplanar import on Linux dma-buf and macOS
  IOSurface both go via separate `R16Unorm` Y + `Rg16Unorm` UV
  planes per-import; the same shader path handles both because the
  storage convention (10-bit data MSB-aligned in 16-bit cells) is
  identical between P010 and P410. `Rgb10a2Unorm` is exposed on
  our pinned wgpu and is used today as a storage *output* format
  in `Bgra2Xv30DmaBuf`; whether it can be imported via
  `texture_from_dmabuf_fd` for the renderer-side packed-XV30
  variant is the open hardware question.
- **MSB-aligned 10-in-16 sampler arithmetic**: a 10-bit value max
  (1023) stored in bits [15:6] of a 16-bit word reads through an
  `R16Unorm` sampler as `60160 / 65535 ≈ 0.918` (limited-range
  white, raw 10-bit value 940 × 64 = 60160). This is a hardware
  behaviour of normalized samplers, not a bug. The renderer's
  shader uses 10-bit-derived limited-range breakpoints directly
  (`(y - 4096/65535) * (65535/56064)`) rather than the 8-bit
  ones, so the same `y_lim` sample lands on the right `[0, 1]`
  normalised value at all luma levels — black, mid-grey, and
  white — without an intermediate `luma_scale` indirection. An
  earlier renderer version used an intermediate scale +
  8-bit-derived breakpoints, which produced a systematic ~1%
  mid-tone lift; the per-bit-depth dispatch eliminates it.

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

- **Packed 10-bit 4:4:4 on the renderer-import side.** The
  encode-side producer is wired (`Bgra2Xv30DmaBuf` writes
  `Rgb10a2Unorm` via `VK_FORMAT_A2B10G10R10_UNORM_PACK32` dma-buf
  export — DRM_FORMAT_XV30). The renderer-import side has not been
  verified on hardware: VAAPI's decoded surface format for HEVC
  Main 4:4:4 10-bit is driver-dependent. If a driver exports packed
  XV30 rather than biplanar 16-bit, the renderer needs a new
  `RenderLayout::Packed1010102` variant that samples
  `Rgb10a2Unorm` directly (analogous to `PackedXYUV` for the 8-bit
  case). The `dmabuf_test::roundtrip_hevc_main444_10bit_identity`
  cell on RADV/NVK is the gate that surfaces this.

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
shader's `range_kind` dispatch handle the 10-bit display side
(native 10-bit limited-range breakpoints, no intermediate
`luma_scale` indirection); the encoder/decoder probes handle the
host side. The negotiator picks the first entry that appears in
*both* the host's `tether_probe::host_encode_profiles()` and the
client's advertised `tether_probe::client_decode_profiles()` —
anything a given device's hardware can't deliver gets filtered out
by the probe layer automatically.

**One-call capability discovery.** As of the tether-probe rewrite,
`host_supported_profiles()` is the authoritative answer for "what
can this hardware deliver end-to-end." It round-trips every
preference-list entry through the full production chain (capture →
bridge → encoder → decoder) and tags each rejection with a
`PipelineStage` (`Capture` / `Construct` / `Submit` / `Decode`) so
operators reading the startup log see exactly which stage rejected
each profile. The Intel iHD Main10 case now reports
`Unsupported { stage: Submit, reason: "submit_dmabuf: ..." }` rather
than the layered "codec said yes but gpuconvert said no" of the
previous three-cache scaffolding.

**Live status by platform pair** (post-10-bit handoff):

| Host       | Client     | Expected pick (best mutual)      | Notes |
| ---------- | ---------- | -------------------------------- | ----- |
| Linux      | Linux      | HEVC 4:4:4 8-bit                 | 10-bit Linux encode wired through the host's warm-time probe but blocked at the driver layer on the hardware tested (Intel iHD + Mesa + FFmpeg 8.1 rejects P010 dma-buf via `av_hwframe_map`). Main10 advertises only on drivers that pass the live `submit_dmabuf` round-trip — see the "Linux 10-bit encode — empirical state" section above. |
| macOS M-series | Linux  | HEVC 4:2:0 10-bit (Main10)       | SCK delivers `'x420'` → VT encodes P010 → client decodes via VAAPI |
| macOS M-series | macOS  | HEVC 4:2:0 10-bit (Main10)       | Same encode side; client decodes back to `'x420'` IOSurface |
| Linux      | macOS      | HEVC 4:4:4 8-bit                 | VT client decodes 4:4:4 via NV24 even though encode side is 4:2:0-only |
| Windows    | Windows    | HEVC 4:2:0 (Main10 / Main)       | Vendor-selected encode (QSV/AMF/NVENC, MF fallback) → D3D11VA decode; loopback-verified. **4:2:0 only — 4:4:4 excluded** (no VP path). Encode profiles are advertised without a per-profile probe (AMF single-session limit), so a 10-bit pick relies on the live encoder, not a warm-time round-trip. |
| Windows    | any        | HEVC 4:2:0 or H.264              | Host advertises H.264 + HEVC Main + Main10; intersection with the client's decode set picks the best 4:2:0 rung. |
| any        | legacy 8-bit only | H.264 4:2:0 8-bit         | Universal floor; legacy client without decode-profiles extension |

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

When adding a new profile to `tether_probe::PROFILE_PREFERENCE`:

1. **Does it require new layer-3 (encode) capability?** Extend the
   encoder probe in
   `crates/tether-probe/src/host/{vaapi,videotoolbox}.rs`. Probe
   must do a real encode of a real frame, not just open the codec
   context. Tag stage-of-failure with the right `PipelineStage`.
2. **Does it require new layer-5 (decode) capability?** Add a
   matching decoder probe with a checked-in fixture in
   `crates/tether-probe/fixtures/probe/`. Probe must consume the
   fixture and assert `Frame::Gpu` (not `Frame::Cpu`) — the
   `Frame::Cpu` discriminator is what catches silent SW fallback.
3. **Does it require new layer-1 (capture) capability?** The Linux
   probe constructs the real gpuconvert bridge in
   `probe_10bit_submit` (extend the dispatch for new chromas); the
   macOS probe consults `sck_pixel_format_for_profile`. If a new
   capture path is needed, wire its capability into the relevant
   `host/*.rs` capture-stage check.
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
