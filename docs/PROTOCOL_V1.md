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

Video negotiation is host-authoritative. The client advertises decode profiles;
the host intersects that list with host encode capabilities and returns one
`NegotiatedVideo`. The client must reject a chosen profile it did not advertise.
If there is no mutual profile, the host sends a typed handshake rejection rather
than a fallback `ServerHello`.

Clock sync is not part of hello. Immediately after handshake, the client sends a
`ClockProbeRequest` over the control channel and computes `ClockSync` from the
matching `ClockProbeResponse`.

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
- `scale_num` / `scale_den`
- `primary`
- `current_mode`
- `available_modes`
- `can_set_mode`

`DisplayMode` is `{ width, height, refresh_millihz }`.

The host sends the best topology it can observe in `ServerHandshake`.
Production hosts enumerate the local display system; test-pattern/headless
fallbacks advertise a synthetic primary display. If the capture backend later
reveals a more exact primary capture mode, the host sends a fresh
`DisplayList`.

`SetViewportHint { stream_id, viewport }` is a best-effort encoder sizing hint.
It does not change host resolution and does not require an acknowledgement.

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

## Media Datagrams

The proven media design is unchanged:

- All video frames, including IDRs, ride unreliable datagrams.
- `FrameFragmenter` splits frames into bounded shards and Reed-Solomon parity.
- `stream_epoch` invalidates stale fragments across encoder/display restarts.
- IDRs are self-decodable.
- Cursor and audio use latest-wins/drop-oldest semantics.

## Future-Feature Checks

The v1 shape is intended to cover RustDesk-like feature growth without another
wire reset:

- Multi-display: typed `DisplayId`, `VideoStreamId`, display lists, active
  display selection.
- Host resolution changes: `SetDisplayMode` request/result plus display
  capability flags.
- Clipboard/file transfer: feature negotiation first, then typed messages when
  the behavior is committed.
- Gamepad/touch: input capability adverts and typed future input/control
  messages.
- Virtual displays: display descriptors can report synthetic displays and
  mode-setting capabilities.
- Telemetry/privacy/permission states: typed control messages for committed
  behavior; extension lane for experiments.
