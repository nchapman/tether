# Tether Protocol v1

`PROTOCOL_VERSION` is `tether/1`. The current wire format is the first stable
protocol shape; no pre-v1 compatibility is preserved.

## Transports

- Reliable control: length-framed protobuf messages from the `tether.v1` schema.
- Reliable input: length-framed protobuf `InputEvent` messages.
- Unreliable media datagrams: compact serde/bincode envelopes for video, cursor,
  and audio packets. Video continues to use FEC fragmentation on datagrams.

Reliable frames keep the existing size limits and reject oversized frames before
decode. Media datagrams keep the existing bounded decode path.

## Handshake

The client sends `ClientHello`:

- `client_name`
- `decode_profiles`
- `initial_viewport`
- `input_capabilities`
- `requested_features`

`initial_viewport` is a wire field, but production clients do not rely on a
guessed size to start video. The real startup gate is a post-handshake
`SetViewportHint` carrying the renderer's measured physical-pixel viewport
followed by `StreamReady { video: true, ... }`; the host must not emit video
for a stream until both the video-ready flag and a valid viewport are present.

The host replies with `ServerHandshake`:

- `Accepted(ServerHello)` for a negotiated session.
- `Rejected(HandshakeFailure)` when the host cannot or will not start the
  session.

`ServerHello` includes:

- `server_name`
- `video: NegotiatedVideo`
- `audio: Option<AudioConfig>`
- `displays`
- `accepted_features`
- `video_streams` reserved for future active multi-display streaming

Video negotiation is host-authoritative. The client advertises decode profiles;
the host intersects that list with host encode capabilities and returns one
`NegotiatedVideo`. The client must reject a chosen profile it did not advertise.
If there is no mutual profile, the host sends a typed handshake rejection rather
than a fallback `ServerHello`.

Clock sync is not part of hello. Immediately after handshake, the client sends a
burst of `CLOCK_SYNC_PROBE_SAMPLES` `ClockProbeRequest` messages over the
control channel, computes `ClockSync` from the minimum-RTT matching response,
and repeats the same burst periodically during the session (currently every
30 seconds).

## IDs

Protocol IDs are typed:

- `DisplayId(u32)`
- `VideoStreamId(u32)`
- `RequestId(u64)`

Video datagrams route by `stream_id`. Stream metadata maps streams to host
displays/sources through `NegotiatedVideo` and future stream descriptors.

## Display Topology

`DisplayList` is authoritative. Each `DisplayDescriptor` includes:

- `id`
- `name`
- `position`
- `scale_num` / `scale_den` (nonzero rational logical-to-physical scale)
- `primary`
- `current_mode`
- `available_modes`
- `can_set_mode`

`DisplayMode` is `{ width, height, refresh_millihz }`; width and height are
physical/backing pixels.

The host sends the best topology it can observe in `ServerHandshake`.
Production hosts enumerate the local display system; explicit test-pattern
sessions advertise a synthetic primary display. If the capture backend later
reveals a more exact captured display/source mode or scale, the host sends a
fresh `DisplayList`. On Linux portal capture, the refreshed mode is the
PipeWire frame pixel grid and the refreshed scale is derived from portal
compositor-space bounds when available.

`SetViewportHint { stream_id, viewport }` is a best-effort encoder sizing hint
in physical pixels. It does not change host resolution and does not require an
acknowledgement. For the initial stream, it is also part of startup readiness:
production clients send the first viewport hint before `StreamReady`, and hosts
wait for both. The hint is the density-correct presentation target, not
necessarily the OS window surface size: `Fit` caps it at logical 100%, while
`Actual Size` reports logical 100% directly. The host's no-upscale fit rule
still resolves overlarge hints to native capture pixels.

`ClientDisplayMetrics` reports the client output hosting the render surface:
session-local display id, physical `DisplayMode`, logical-to-physical scale
ratio, and optional physical safe area. It is display-mode-matching and
diagnostic input only; it does not request a host display-mode change.

`SetDisplayMode` is the real host display-mode request:

- `request_id`
- `display_id`
- `mode`
- `restore_on_disconnect`

The host must reply with `DisplayModeResult`. Current platform backends return
`Unsupported` until OS-level display mode control is wired. If a mode is applied,
the host sends an updated `DisplayList`.

## Extension Lane

Optional features negotiate with:

- `FeatureAdvert { key, min_version, max_version, payload }`
- `FeatureAccept { key, version, payload }`
- `ExtensionMessage { key, version, request_id, reply_to, payload }`

Unknown or unnegotiated extension messages are protocol errors. First-party
behavior should move to typed control messages once committed.

`ExtensionMessage.payload` is capped at 16 KiB. The extension lane is not a
bulk-transfer lane; file transfer, large clipboard contents, and any other
multi-frame payload must use a dedicated reliable QUIC bulk stream when that
feature lands. Bulk data must not raise the control-frame limit or ride through
`ExtensionMessage`.

When an extension graduates to first-party typed control, peers must accept both
the negotiated extension form and the new typed form for one feature/protocol
version. A new client must not switch to a typed oneof message until the host's
feature acceptance says that form is supported.

## Media Datagrams

The proven media design is unchanged:

- All video frames, including IDRs, ride unreliable datagrams.
- `FrameFragmenter` splits frames into bounded shards and Reed-Solomon parity.
- `stream_epoch` invalidates stale fragments across encoder/display restarts.
- Keyframes are self-decodable: H.264/HEVC carry repeated parameter sets, and
  AV1 carries the sequence header in-band.
- Cursor and audio use latest-wins/drop-oldest semantics.

Media envelopes are intentionally compact and do not have protobuf-style
unknown-field skipping. `VideoFrameMetaEnvelope` stays `V1` unless a future
metadata version is negotiated in hello, either through `NegotiatedVideo` or an
accepted video feature. A host must not emit a future metadata envelope variant
to a client that only negotiated `V1`, because old decoders reject unknown
bincode enum variants before they can use the shard payload or FEC parity.

## Stream Readiness

`StreamReady { video, audio }` is the client's declaration that the receiving
side has initialized the corresponding pipeline:

- `video = true` means the decoder constructed successfully and the client has
  sent a valid viewport hint for the stream.
- `audio = true` means the Opus decoder, jitter ring, and output stream are
  initialized. If audio startup fails or times out, the client sends
  `audio = false` and the session proceeds video-only.

The host drops captured video/audio until the matching readiness bit is open.
For video, the host also forces an IDR when opening the gate so the client starts
from a self-decodable frame.

## Session Shutdown and Stats

Normal and fatal shutdown use `ControlMessage::Goodbye { reason, code,
final_stats }`. `code` is machine-readable (`Clean`, `ProtocolError`,
`UnsupportedVersion`, `InternalError`, or an unknown future value); `reason` is
diagnostic text.

`final_stats` carries the sender's final `SessionSummary`, including video and
optional audio counters. The video summary includes send/receive counts, FEC
recovery, decode errors, render drops, decoder queue drops, and stream-epoch
drop counters (`decode_stale_epoch_drop_frames`,
`decode_epoch_throttle_drop_frames`). Audio summaries include packet/frame
counts, recovery/concealment/drop counters, and `decode_queue_drop_packets`.

A peer that receives `Goodbye` may send one reciprocal stats-bearing `Goodbye`
if it has not already sent one. Implementations must guard against echo loops;
the second `Goodbye` is the final exchange, not a new shutdown request.

## Future-Feature Checks

The v1 shape is intended to cover RustDesk-like feature growth without another
wire reset:

- Multi-display: typed `DisplayId`, `VideoStreamId`, display lists, active
  display selection, and the reserved `ServerHello.video_streams` list for
  future per-stream descriptors.
- Host resolution changes: `SetDisplayMode` request/result plus display
  capability flags.
- Clipboard: feature negotiation first, then typed messages when the behavior is
  committed.
- File transfer / large clipboard: a dedicated reliable QUIC bulk channel, not
  control messages or extensions.
- Gamepad/touch: input capability adverts and typed future input/control
  messages.
- Virtual displays: display descriptors can report synthetic displays and
  mode-setting capabilities.
- Telemetry/privacy/permission states: typed control messages for committed
  behavior; extension lane for experiments.
