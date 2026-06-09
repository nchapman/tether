# Viewport, HiDPI, and Display-Mode Policy

This note defines the rules Tether should follow for stream sizing,
presentation, and host display-mode changes. The current implementation already
has useful pieces: `DisplayDescriptor::scale_num/scale_den`,
`SetViewportHint`, host-side `encode_dims_for_viewport`, and renderer resize
events in physical pixels. The missing piece is a consistent vocabulary and a
policy that does not let an arbitrary client window size become the host's
effective desktop resolution.

## Reference Behaviors

- Sunshine/Moonlight split stream resolution from host display-mode control.
  Moonlight stores an explicit stream width/height, exposes fixed and custom
  resolution choices, and Sunshine can optionally change the host display to
  the client's requested stream resolution when that feature is enabled. Sunshine
  also exposes the requested client width/height/FPS to application hooks via
  `SUNSHINE_CLIENT_WIDTH`, `SUNSHINE_CLIENT_HEIGHT`, and
  `SUNSHINE_CLIENT_FPS`.
- Sunshine's automatic display-device configuration is explicit policy. Its
  `dd_resolution_option = auto` means "change resolution to the requested
  resolution from the client"; it is not a consequence of resizing a viewer
  window.
- Moonlight treats native resolution as a client display property, not as the
  current window size. Its desktop client has platform-specific native-mode
  detection; on macOS it uses CoreGraphics because SDL alone cannot identify the
  true native Retina mode, and on Wayland it deliberately avoids DPI-scaled
  sizes when finding native resolution.
- RustDesk separates view scaling from remote resolution. Its documented display
  defaults include `view-style = original|adaptive`; source-side resolution
  changes are separate messages/actions. Its UI scaling code keeps an "original"
  path that does not fit-to-window, and it carries remote scale information for
  cursor/input mapping.
- RustDesk also has a useful HiDPI lesson: do not derive logical dimensions from
  physical dimensions and a fractional scale when the compositor can report the
  logical size directly. Fractional scale systems can make `physical / scale`
  round to the wrong logical size.

## Coordinate Spaces

Every width/height in protocol, policy, and logs must name its coordinate space:

- `host_mode_px`: the host OS display mode in backing pixels.
- `host_capture_px`: the actual captured frame pixel grid after platform capture
  setup and any capture-time alignment/cropping policy.
- `host_logical`: platform input/display units when they differ from backing
  pixels, such as macOS points.
- `client_display_px`: the physical pixel size of the client monitor/output.
- `client_viewport_px`: the physical pixel size of the Tether render surface
  available for video.
- `client_logical`: client UI units, such as winit logical size or platform
  points. This is never used to size the stream.
- `stream_px`: encoded video dimensions visible to the decoder/renderer.
- `present_rect_px`: the rectangle in `client_viewport_px` where decoded video
  is drawn.

Unqualified `width`, `height`, `resolution`, `viewport`, and `scale` are banned
from new sizing policy APIs and logs unless the surrounding type name already
contains the coordinate space.

## HiDPI Rules

1. Stream sizing uses physical/backing pixels only. `SetViewportHint` must mean
   `client_viewport_px`, not logical window size.
2. `DisplayMode` dimensions are physical/backing pixels. `DisplayDescriptor`
   scale is metadata for UI and input mapping, not a way to reconstruct the
   display mode.
3. Prefer directly observed dimensions over derived dimensions. If a platform can
   report both physical and logical sizes, carry both. Do not compute
   `logical = physical / scale` as the authoritative value on fractional-scale
   desktops.
4. macOS capture/display enumeration must use backing-pixel APIs for
   `host_mode_px`/`host_capture_px`. Point-space APIs are only for input
   injection and cursor mapping.
5. Client-side "native" means the selected client output's physical pixel mode,
   optionally minus a platform safe area. It does not mean the current Tether
   window size.

## Default Presentation Policy

The default should be `FitNoUpscale`:

1. The host captures the selected display at `host_capture_px`.
2. The client reports `client_viewport_px` whenever the render surface changes.
3. The host computes the largest aspect-preserving rectangle that fits
   `host_capture_px` inside `client_viewport_px`.
4. If that fit scale is less than `1.0`, the host may downscale and encode at the
   aligned fit size. The client presents the decoded frame 1:1 in
   `client_viewport_px`, centered with letterbox/pillarbox margins.
5. If that fit scale is greater than or equal to `1.0`, the host encodes
   `host_capture_px`. The client presents the decoded frame 1:1, centered. A
   larger client viewport does not cause host-side upscaling and does not change
   the host display mode.
6. Alignment can shrink the stream a few pixels below the mathematical fit. Keep
   presentation 1:1 by default; do not blur the result back up just to remove a
   thin margin.
7. Bitrate and encoder rebuild decisions are based on `stream_px`, not raw
   `client_viewport_px`.

This directly fixes the uncomfortable case: if the client display is larger and
the host display fits, the host streams native pixels and the client shows them
1:1 instead of treating the oversized viewport as a new target resolution.

## View Modes

Tether should expose view behavior as a client preference, independent of host
display-mode behavior:

- `FitNoUpscale` default: downscale on the host when needed; otherwise 1:1.
- `Original`: always request `stream_px = host_capture_px`; if the viewport is
  smaller, the client presents the native stream 1:1 centered and clipped rather
  than resizing the host stream. Scroll/pan controls can refine this later
  without changing the stream-sizing contract.
- `Fit`: fit the video into the viewport and allow client-side upscale when the
  viewport is larger than `stream_px`. This is a presentation choice only.
- `Fill`: crop-preserving fill for users who prefer no letterbox. This is also a
  presentation choice only.

Only `FitNoUpscale` and `Original` exist in the first implementation, exposed
by the client as `--view-mode fit-no-upscale|original`. `FitNoUpscale` presents
1:1 when the stream fits and fits down only when needed. `Original` still sends
one startup viewport hint so the host can open the video gate, but uses a
no-cap viewport (`u32::MAX × u32::MAX`) and ignores subsequent window resizes
for stream sizing; presentation remains 1:1 and clips when the surface is
smaller. The key invariant is that none of these view modes changes the host OS
display mode.

## Host Display-Mode Matching

Changing the host's real display mode is a separate, explicit operation:

1. The client may request "match client display" only from a deliberate user
   action or a saved per-peer preference, not from normal window resize.
2. The desired mode is derived from `client_display_px` and refresh rate for the
   selected client output, not from `client_viewport_px`.
3. The request uses `SetDisplayMode` and requires `DisplayModeResult`. Failed,
   unsupported, or inexact matches leave capture at the current host mode.
4. By default, require an exact host `available_modes` match. A future virtual
   display backend can advertise synthetic exact modes; physical displays should
   not silently switch to a different aspect ratio.
5. If a mode is applied, `restore_on_disconnect` defaults to true and the host
   sends a fresh `DisplayList`.
6. Display-mode matching is disabled while multiple clients control the same
   physical display unless ownership is explicit.

This follows the Sunshine/Moonlight lesson: matching host resolution to the
client can be valuable, but it must be an explicit display-device policy. It
also follows the RustDesk lesson: viewing/scaling modes and remote resolution
changes must remain separate concepts.

## Input and Cursor Mapping

1. Client pointer events are first mapped from `client_viewport_px` through
   `present_rect_px` into normalized video coordinates.
2. Normalized video coordinates map to `host_capture_px`.
3. Platform injection then maps `host_capture_px` to `host_logical` when the OS
   consumes logical units, as macOS CGEvent does.
4. Relative mouse motion bypasses presentation and DPI scaling; it remains device
   delta input.
5. Cursor images and hotspots carry their source pixel scale. Presentation scale
   affects cursor drawing, but not the source cursor position.

## Protocol Implications

The existing `Viewport` type can remain as a wire-compatible physical-pixel
`client_viewport_px`, but docs and logs should say that explicitly. To support
host-mode matching cleanly, the client needs to report the selected client
display's physical mode separately from the render surface. That should be a
typed control or negotiated extension carrying at least:

- client display id for the session-local output;
- `client_display_px`;
- refresh rate in millihertz;
- optional safe-area rect in physical pixels;
- scale ratio for UI/input diagnostics only.

`SetDisplayMode` remains the only message that can change the host display mode.
`SetViewportHint` remains best-effort stream sizing input.

## Test Rules

The policy should be enforced with pure tests before platform work:

- Host smaller than client viewport: `FitNoUpscale` returns `stream_px =
  host_capture_px`.
- Host larger than client viewport: returns the largest aligned fit, preserving
  aspect ratio and never exceeding either source or viewport.
- Aspect mismatch: returns centered letterbox/pillarbox dimensions, no stretch.
- Fractional scale case: physical mode, logical size, and scale are carried as
  separate values; no policy test derives one as authoritative from the others.
- macOS Retina case: backing-pixel capture can still map input through
  point-space injection.
- View-mode tests prove presentation choices do not emit `SetDisplayMode`.
- Display-mode matching tests require explicit request/ack and reject inexact
  physical-display modes by default.

## References

- Sunshine configuration: `dd_resolution_option = auto` changes resolution to
  the requested resolution from the client; refresh-rate matching is similarly
  explicit.
  <https://docs.lizardbyte.dev/projects/sunshine/master/md_docs_2configuration.html>
- Sunshine source reference inspected at
  `6bc2701160ed181e2cd01969bf2c8cd9e77957d3`: `src/process.cpp` exports
  `SUNSHINE_CLIENT_WIDTH`/`HEIGHT`/`FPS`; `src/display_device.cpp` applies
  automatic requested-resolution policy only when the client has enabled the
  optimization path.
- Moonlight docs warn that stream resolution and host display resolution should
  match for HDR quality, which implies they are separate settings.
  <https://github.com/moonlight-stream/moonlight-docs/wiki/Setup-Guide>
- Moonlight source reference inspected at
  `02004bac3f61d630b6a5e388603a3dc4eec2b30b`:
  `app/settings/streamingpreferences.cpp`,
  `app/cli/commandlineparser.cpp`, and `app/streaming/streamutils.cpp` show
  explicit stream resolution settings, aspect-preserving presentation, and
  platform-specific native/HiDPI handling.
- RustDesk advanced settings document `view-style = original|adaptive` and
  separate display options.
  <https://rustdesk.com/docs/en/self-host/client-configuration/advanced-settings/>
- RustDesk source reference inspected at
  `6426269d41a73f678180a9ef733300a8f8a1b912`: `src/ui/remote.tis` keeps
  original/adaptive view scaling separate from input mapping; `src/flutter.rs`
  avoids deriving fractional-scale logical dimensions from physical dimensions.
