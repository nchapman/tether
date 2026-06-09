//! Control stream: handshake, clock probes, stream control, display topology,
//! and negotiated extension traffic.
//!
//! Reliable handshake/control/input messages are encoded as protobuf messages
//! under the `tether.v1` schema. New optional fields may be added with fresh
//! numeric tags; older peers skip fields they do not know. First-party behavior
//! that is part of the supported protocol should use typed fields or typed
//! [`ControlMessage`] variants, not opaque extension payloads.
//!
//! Media datagrams intentionally stay on the compact serde/bincode path: video
//! packets, cursor positions, and audio packets are hot-path bounded datagrams
//! where the current envelope is already explicit and loss-tolerant.
//!
//! # Extension lane
//!
//! Experimental or third-party features negotiate through
//! [`FeatureAdvert`] / [`FeatureAccept`] in hello. After a feature is accepted,
//! peers may send [`ControlMessage::Extension`] with the accepted key/version.
//! Unknown or unnegotiated extension messages are protocol errors at the session
//! layer; first-party committed features should graduate to typed messages.

use crate::{pb, CodecError, MonoNanos, ReliableMessage};
use prost::Message as _;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DisplayId(pub u32);

impl std::fmt::Display for DisplayId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VideoStreamId(pub u32);

impl std::fmt::Display for VideoStreamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl From<u8> for VideoStreamId {
    fn from(value: u8) -> Self {
        Self(u32::from(value))
    }
}

impl From<u32> for VideoStreamId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub u64);

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodecKind {
    H264,
    Hevc,
    Av1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChromaSubsampling {
    Yuv420,
    Yuv444,
}

/// Negotiation unit for video codec capabilities.
///
/// The client advertises the set it can decode in [`ClientHello`]; the host
/// intersects with its encode capabilities and returns the selected profile in
/// [`ServerHello::video`].
///
/// `bit_depth` is `u8` rather than an enum so the wire form stays
/// stable as new depths land — 8, 10, and (hypothetically) 12 all
/// round-trip identically. Decode-side callers should use
/// [`is_known_bit_depth`] to filter out values this build doesn't
/// implement, rather than trusting the wire blindly. A peer sending
/// `bit_depth = 9`, `255`, or `0` is either malformed or from the
/// future — both cases reduce to "drop this profile, don't try to
/// build a pipeline for it."
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoProfile {
    pub codec: CodecKind,
    pub chroma: ChromaSubsampling,
    pub bit_depth: u8,
}

/// The bit depths this build of tether knows how to construct an
/// encode/decode/render pipeline for. Adding a new depth means
/// extending the encoder dispatch (`derive_bitrate_kbps`,
/// `EncoderSlot` chroma/depth match arms), the decoder side (codec
/// probe), and the renderer (texture-format dispatch) at the same
/// time — anything less ships a half-wired profile.
pub const KNOWN_BIT_DEPTHS: &[u8] = &[8, 10];

/// `true` iff `depth` is one of [`KNOWN_BIT_DEPTHS`]. Decode-side
/// boundary check for `VideoProfile`s that arrived over the wire.
#[must_use]
pub fn is_known_bit_depth(depth: u8) -> bool {
    KNOWN_BIT_DEPTHS.contains(&depth)
}

impl VideoProfile {
    /// The universal floor: H.264 Main 4:2:0 8-bit. Every host backend
    /// we ship supports it, every client backend can decode it.
    pub const H264_8BIT_420: Self = Self {
        codec: CodecKind::H264,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 8,
    };

    /// HEVC Main 4:2:0 8-bit — the better-compression mid-rung. Most
    /// VAAPI / VideoToolbox hosts and clients support this today.
    pub const HEVC_8BIT_420: Self = Self {
        codec: CodecKind::Hevc,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 8,
    };

    /// HEVC Main444 8-bit — the desktop-quality top rung. Preserves
    /// full chroma resolution (no subsampling) so antialiased text
    /// edges and saturated UI accents stay sharp. Requires VAAPI
    /// HEVC Main444 (Intel Tiger Lake+ / AMD VCN3+) on the host and
    /// matching decode on the client.
    pub const HEVC_8BIT_444: Self = Self {
        codec: CodecKind::Hevc,
        chroma: ChromaSubsampling::Yuv444,
        bit_depth: 8,
    };

    /// HEVC Main10 (4:2:0 10-bit). The 10-bit equivalent of the
    /// Main rung — same chroma resolution, finer quantisation grid
    /// (10 bits per sample instead of 8). Visible benefit on
    /// gradient-heavy desktop content where 8-bit banding shows up.
    /// macOS hosts produce this via VT HEVC Main10 encode; Linux
    /// hosts produce it via VAAPI Main10 (driver support varies —
    /// the probe layer answers).
    pub const HEVC_10BIT_420: Self = Self {
        codec: CodecKind::Hevc,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 10,
    };

    /// HEVC Main 4:4:4 10-bit — full chroma resolution + 10-bit
    /// precision. The top rung for quality. macOS VT confirmed to
    /// accept this end-to-end (encoder + decoder); Linux VAAPI
    /// driver support is sparse and the probe layer surfaces the
    /// reality per device.
    pub const HEVC_10BIT_444: Self = Self {
        codec: CodecKind::Hevc,
        chroma: ChromaSubsampling::Yuv444,
        bit_depth: 10,
    };

    /// AV1 Main 4:2:0 8-bit. The next-generation codec rung: at the
    /// same quality as HEVC Main, AV1 spends roughly 30% less
    /// bitrate, which matters most on congested links. Encode is
    /// VAAPI-only in this build (Intel Arc / AMD RDNA 3); macOS
    /// hosts cannot produce AV1 because VideoToolbox is decode-only.
    /// Decode is wider: VAAPI (Intel Tiger Lake+, AMD RDNA 2+, NVIDIA
    /// Ampere+) and VideoToolbox (Apple Silicon M3+).
    pub const AV1_8BIT_420: Self = Self {
        codec: CodecKind::Av1,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 8,
    };

    /// AV1 Main 4:2:0 10-bit. Same encode/decode hardware envelope as
    /// [`AV1_8BIT_420`] with finer quantisation. AV1 4:4:4 is out of
    /// scope for this build (rare encoder support; HEVC 4:4:4 covers
    /// the desktop-text use case).
    pub const AV1_10BIT_420: Self = Self {
        codec: CodecKind::Av1,
        chroma: ChromaSubsampling::Yuv420,
        bit_depth: 10,
    };
}

// =============================================================
// Four-axis color spec. Carried first-class on `ServerHello::video`.
// =============================================================
//
// A video stream's color identity is four orthogonal things, each
// of which can change independently per codec / source / display.
// H.264 and HEVC carry these in the SPS VUI parameters; we expose
// them on the wire so the decoder + renderer can negotiate the
// right shader path without having to re-parse the bitstream.
//
// Range × Matrix × Transfer × Primaries combinations describe every
// common SDR and HDR pipeline; the variants here are the subset
// Tether commits to supporting in the foreseeable roadmap. New
// variants land as additive enum additions per the bincode
// versioning policy.

/// Pixel-value range of the encoded Y / Cb / Cr planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorRange {
    /// "TV" / "video" range. Y in 16..235, Cb/Cr in 16..240. The
    /// default for broadcast and what every Tether encoder produces
    /// today (`PixelFormat::YCbCr_420v` on SCK; explicit
    /// limited-range quantize in `tether-gpuconvert`).
    Limited,
    /// "PC" / "full" range. Y/Cb/Cr each span the full 0..255. Some
    /// desktop captures hand us bytes in this range; the decoder
    /// needs to know so it skips the limited-range expand step.
    Full,
}

/// Y'CbCr ↔ R'G'B' conversion matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorMatrix {
    /// BT.709 (Rec. ITU-R BT.709-6). The standard for HD video and
    /// what every encoder we ship configures today.
    Bt709,
    /// BT.2020 non-constant luminance. Required for HDR pipelines
    /// (PQ / HLG). Not yet wired on either end.
    Bt2020Ncl,
    /// Identity matrix — R'G'B' rides the Y/Cb/Cr channels directly.
    /// Useful for lossless screen capture; reserved.
    Identity,
}

/// Transfer function (a.k.a. EOTF / OETF pair) that maps between
/// gamma-encoded R'G'B' and linear light.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorTransfer {
    /// BT.709 OETF (Rec. ITU-R BT.709-6, eq. 1.2). What broadcast
    /// content uses. Tether currently treats source bytes as
    /// BT.709-encoded — the assumption that produces a ≤~5% mismatch
    /// on actually-sRGB sources like a desktop framebuffer.
    Bt709,
    /// sRGB OETF (IEC 61966-2-1). The right answer for desktop
    /// screen capture; matches what every Mac/Windows/Linux
    /// compositor framebuffer is encoded as.
    Srgb,
    /// Perceptual Quantizer (SMPTE ST 2084 / Rec. ITU-R BT.2100).
    /// HDR; reserved.
    Pq,
    /// Hybrid Log-Gamma (Rec. ITU-R BT.2100). HDR; reserved.
    Hlg,
    /// Linear light. Useful for offline / lossless pipelines;
    /// reserved.
    Linear,
}

/// Color primaries (the chromaticity coordinates of R, G, B).
///
/// Note: every variant here pairs with a corresponding
/// [`ColorMatrix`] variant — BT.709 primaries with `ColorMatrix::Bt709`,
/// BT.2020 primaries with `ColorMatrix::Bt2020Ncl`. Adding a new
/// primaries variant without its matching matrix variant produces
/// a silent wrong-color path (the decoder applies the wrong gamut
/// conversion), so the two enums grow together. Display P3 is the
/// next likely addition for Apple displays; it'll land alongside a
/// `ColorMatrix::DisplayP3` variant (or, equivalently, a Bt709 →
/// DisplayP3 gamut-mapping step in the renderer keyed off the
/// primaries field).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorPrimaries {
    /// BT.709 / sRGB primaries. Identical chromaticities; the two
    /// differ only in transfer function. Default for SDR.
    Bt709,
    /// BT.2020 / Rec. ITU-R BT.2100 primaries. Wider gamut; required
    /// for HDR.
    Bt2020,
}

/// The four-axis tuple. Carried first-class on
/// [`NegotiatedVideo::color_space`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoColorSpec {
    pub matrix: ColorMatrix,
    pub range: ColorRange,
    pub transfer: ColorTransfer,
    pub primaries: ColorPrimaries,
}

impl VideoColorSpec {
    /// The honest spec for a desktop screen capture on macOS, Linux
    /// Wayland, or Windows: sRGB transfer (compositor framebuffer
    /// reality) with BT.709 matrix/primaries and limited range. What
    /// `tether-host` advertises today. The decoder applies the sRGB
    /// EOTF to match.
    #[must_use]
    pub const fn sdr_desktop() -> Self {
        Self {
            matrix: ColorMatrix::Bt709,
            range: ColorRange::Limited,
            transfer: ColorTransfer::Srgb,
            primaries: ColorPrimaries::Bt709,
        }
    }

    /// BT.709 transfer + matrix + primaries, limited range. The
    /// classic broadcast spec; what a video file (`.mp4`, `.mkv`)
    /// containing BT.709 content would carry. No host backend
    /// advertises this today (everything goes through
    /// `sdr_desktop`), but the variant exists so a future
    /// file-playback or video-conference source can advertise its
    /// real transfer.
    #[must_use]
    pub const fn sdr_bt709() -> Self {
        Self {
            matrix: ColorMatrix::Bt709,
            range: ColorRange::Limited,
            transfer: ColorTransfer::Bt709,
            primaries: ColorPrimaries::Bt709,
        }
    }
}

impl Default for VideoColorSpec {
    /// `sdr_desktop()` — what every current host backend produces.
    fn default() -> Self {
        Self::sdr_desktop()
    }
}

/// Pixel/bit-depth format the host's video stream uses. The hardware
/// decoder pipeline (VAAPI, VideoToolbox, Media Foundation) needs this
/// up front — before parsing the SPS — to pick the right import path.
/// Advertised as part of [`NegotiatedVideo`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelFormat {
    /// 8-bit packed BGRA (capture-side default; never on the wire in
    /// encoded video, but listed for completeness — a future raw
    /// debug stream could use it).
    Bgra8,
    /// 8-bit 4:2:0 planar. The default for encoded video.
    Nv12,
    /// 10-bit 4:2:0 biplanar — the path Main10 / HDR rides. 10-bit
    /// samples MSB-aligned in 16-bit cells per FFmpeg's P010LE
    /// convention.
    P010,
    /// 8-bit 4:4:4 planar (Y/U/V each at full resolution). The path
    /// HEVC Main444 emits for 8-bit 4:4:4 streams.
    Yuv444p,
    /// 10-bit 4:4:4 biplanar — the path HEVC Main 4:4:4 10-bit rides
    /// on macOS (matching VT's `'P410'` / `'xf44'` IOSurfaces) and
    /// (if the renderer-side biplanar 16-bit import is what's wired)
    /// on Linux too. 10-bit MSB-aligned in 16-bit cells.
    P410,
}

/// One concrete physical/backing-pixel display mode the host can report or
/// apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisplayMode {
    pub width: u32,
    pub height: u32,
    pub refresh_millihz: u32,
}

impl DisplayMode {
    #[must_use]
    pub const fn new(width: u32, height: u32, refresh_millihz: u32) -> Self {
        Self {
            width,
            height,
            refresh_millihz,
        }
    }
}

/// Describes one host display. `DisplayList` is authoritative: clients replace
/// their local topology view with each message rather than merging.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisplayDescriptor {
    /// Host-assigned display id. Stable for the lifetime of a session.
    pub id: DisplayId,
    /// Human-readable name as the host's compositor reports it
    /// (`HDMI-A-1`, `DP-3`, `Built-in Retina Display`, etc.). May be
    /// empty if the host's capture backend doesn't expose names.
    pub name: String,
    /// Logical-to-physical scale as a rational `num / den`. `(1, 1)`
    /// for a 1× display, `(2, 1)` for a Retina-style 2×, `(3, 2)` for
    /// 150% Windows scaling. Rational so common factors round-trip
    /// without `f32` precision loss; the client computes the final
    /// scale factor as needed.
    pub scale_num: u16,
    pub scale_den: u16,
    pub primary: bool,
    /// Origin of this display in the virtual desktop, in physical
    /// pixels. Lets the client place multi-display windows correctly
    /// without re-deriving the topology.
    pub position: (i32, i32),
    pub current_mode: DisplayMode,
    pub available_modes: Vec<DisplayMode>,
    pub can_set_mode: bool,
}

/// NTP-style three-way clock probe. The receiver records `t3` locally on
/// arrival; the offset between the two monotonic clocks is then
/// `((t1 - t0) + (t2 - t3)) / 2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockProbe {
    pub t0_sender: MonoNanos,
    pub t1_receiver_recv: MonoNanos,
    pub t2_receiver_send: MonoNanos,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureAdvert {
    pub key: String,
    pub min_version: u32,
    pub max_version: u32,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureAccept {
    pub key: String,
    pub version: u32,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtensionMessage {
    pub key: String,
    pub version: u32,
    pub request_id: RequestId,
    pub reply_to: RequestId,
    pub payload: Vec<u8>,
}

/// Extension messages are control-plane escape hatches, not a bulk-transfer
/// lane. Large clipboard/file payloads must use an independent reliable QUIC
/// bulk stream so they cannot head-of-line block core control messages.
pub const MAX_EXTENSION_PAYLOAD_BYTES: usize = 16 * 1024;

/// Cursor sprites ride reliable control, but should stay within the existing
/// control-frame budget. Hosts already downsample large cursors before send;
/// this guards hostile or malformed peers on direct decode paths.
pub const MAX_CURSOR_SHAPE_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputCapabilities {
    pub keyboard: bool,
    pub mouse: bool,
    pub relative_mouse: bool,
    pub text: bool,
}

impl Default for InputCapabilities {
    fn default() -> Self {
        Self {
            keyboard: true,
            mouse: true,
            relative_mouse: true,
            text: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NegotiatedVideo {
    pub stream_id: VideoStreamId,
    pub display_id: DisplayId,
    pub profile: VideoProfile,
    pub pixel_format: PixelFormat,
    pub color_space: VideoColorSpec,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientHello {
    pub client_name: String,
    pub decode_profiles: Vec<VideoProfile>,
    pub initial_viewport: Option<Viewport>,
    pub input_capabilities: InputCapabilities,
    pub requested_features: Vec<FeatureAdvert>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerHello {
    pub server_name: String,
    pub video: NegotiatedVideo,
    pub audio: Option<crate::audio::AudioConfig>,
    pub displays: Vec<DisplayDescriptor>,
    pub accepted_features: Vec<FeatureAccept>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandshakeFailure {
    pub code: GoodbyeCode,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServerHandshake {
    Accepted(ServerHello),
    Rejected(HandshakeFailure),
}

// --- Goodbye ------------------------------------------------------------

/// Machine-readable reason for a [`ControlMessage::Goodbye`]. Lets the
/// peer distinguish "host shutting down cleanly" from "protocol
/// violation" without parsing the human-readable `reason` string.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoodbyeCode {
    /// Local side initiated a clean shutdown (user quit, system
    /// suspend, etc.). The peer should not retry.
    Clean,
    /// Local side observed a protocol violation in the peer's
    /// messages. Retrying with the same binary is unlikely to help.
    ProtocolError,
    /// Handshake's variant tag didn't decode, i.e. the peer is
    /// speaking a newer protocol revision than this build knows.
    UnsupportedVersion,
    /// Catch-all for internal errors not otherwise classified. The
    /// peer may retry; a transient failure here doesn't preclude
    /// reconnect.
    InternalError,
    /// Forward-compatible holder for a newer peer's machine-readable code.
    /// Receivers can log the numeric value without failing the whole control
    /// message.
    Unknown(i32),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSummary {
    pub role: String,
    pub duration_ms: u64,
    pub codec: String,
    pub chroma: String,
    pub bit_depth: u32,
    pub video: VideoSessionStats,
    pub audio: Option<AudioSessionStats>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoSessionStats {
    pub frames_sent: u64,
    pub frames_received: u64,
    pub keyframes: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub incomplete_frames: u64,
    pub fragment_loss_events: u64,
    pub decode_errors: u64,
    pub render_drop_frames: u64,
    pub idr_requests: u64,
    pub decode_queue_drop_frames: u64,
    pub transient_send_drop_frames: u64,
    pub fec_recovered_frames: u64,
    pub fec_recovered_fragments: u64,
    pub datagrams_sent: u64,
    pub parity_datagrams_sent: u64,
    pub max_datagrams_per_frame: u64,
    pub max_frame_bytes: u64,
    pub max_keyframe_bytes: u64,
    pub forced_idr_misses: u64,
    pub decode_stale_epoch_drop_frames: u64,
    pub decode_epoch_throttle_drop_frames: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioSessionStats {
    pub packets_sent: u64,
    pub packets_received: u64,
    pub capture_frames: u64,
    pub underruns: u64,
    pub dropped_samples: u64,
    pub recovered_frames: u64,
    pub concealed_frames: u64,
    pub dropout_frames: u64,
    pub dropouts: u64,
    pub stale_packets: u64,
    pub decode_errors: u64,
    pub decode_queue_drop_packets: u64,
}

/// Cursor input model. Carried by
/// [`ControlMessage::SetCursorMode`]; default is `Absolute`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CursorMode {
    /// Today's behavior: client sends normalized pointer
    /// coordinates via `ClientCursorPacket`; host does
    /// `enigo::move_mouse(_, _, Coordinate::Abs)`. Correct for
    /// productivity apps; broken for games that recenter the
    /// cursor every frame or read raw HID deltas.
    #[default]
    Absolute,
    /// Client grabs/hides the cursor and emits device-level
    /// deltas via `InputEventKind::RelativeMouseMove`. Host
    /// dispatches via `enigo::move_mouse(_, _, Coordinate::Rel)`
    /// which routes through the OS's native delta API (the only
    /// path raw-input games can observe).
    Relative,
    /// Forward-compatible holder for a newer cursor mode. Current apps treat
    /// this as advisory and do not switch local cursor behavior on it.
    Unknown(i32),
}

/// Messages exchanged on the reliable control stream after handshake.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlMessage {
    /// Client requests an immediate IDR (e.g., after detected packet loss).
    ForceIdr,
    /// Client → host. Loss-driven recovery request: the host emits a
    /// fresh IDR.
    ///
    /// This message is a latency win over waiting for the next
    /// decoder-driven `ForceIdr`: the client signals as soon as the
    /// reassembler observes a stale-dropped fragment, before the
    /// decoder even gets a chance to fail.
    ///
    /// `last_reassembled_frame_id` carries the newest frame the *client's
    /// reassembler* completed when it fired this request. It is a
    /// diagnostic hint only — the host logs it and always responds with a
    /// full IDR. **It is NOT decode-verified:** a frame can reassemble (FEC
    /// rebuilt every shard) yet still be a P-frame referencing an earlier
    /// frame the decoder concealed or skipped, so "reassembled" is strictly
    /// weaker than "decoded cleanly". Do not use this value to select an
    /// encoder reference (e.g. a Long-Term Reference re-prediction): that
    /// requires a decode-success signal fed back from the decode thread,
    /// which we deliberately do not have. Reference-frame invalidation is
    /// intentionally not implemented — it is NVENC-/Windows-mostly and
    /// structurally unavailable on the VAAPI path (FFmpeg's `vaapi_encode.c`
    /// builds the picture/rate-control params once at init), while FEC + a
    /// bounded GOP already cover its failure mode.
    RequestRecovery {
        last_reassembled_frame_id: u32,
    },
    /// Periodic clock-sync re-probe (either side may initiate).
    ClockProbeRequest {
        t0_sender: MonoNanos,
    },
    ClockProbeResponse(ClockProbe),
    Goodbye {
        reason: String,
        code: GoodbyeCode,
        final_stats: Option<Box<SessionSummary>>,
    },
    /// Escape hatch for features that don't yet warrant a typed variant.
    /// Payload is opaque to the core protocol; feature specs define their own
    /// small, versioned payload bodies. Unknown keys are logged + dropped. See
    /// the module-level doc.
    Extension(ExtensionMessage),
    /// Host → client. Cursor sprite (RGBA pixels). Routed on the
    /// reliable control stream rather than the cursor datagram channel
    /// because a 64×64 RGBA sprite exceeds the 1200-byte datagram
    /// budget and reassembly would be more complex than the win.
    /// Clients cache shapes by `id` and switch via [`Self::CursorUseShape`].
    CursorShape {
        id: u64,
        hotspot: (u16, u16),
        width: u16,
        height: u16,
        format: crate::cursor::CursorPixelFormat,
        pixels: Vec<u8>,
    },
    /// Host → client. Activate a previously-sent [`Self::CursorShape`]
    /// by id. Separate from `CursorShape` so the host can switch
    /// between cached cursors without re-sending the pixels.
    CursorUseShape {
        id: u64,
    },
    /// Host → client. Full display topology. Sent post-handshake and
    /// again on hotplug (display added/removed) — the client treats
    /// each message as the authoritative replacement. Single-monitor
    /// hosts send a one-element list; the field shape supports
    /// multi-monitor without a V2.
    DisplayList {
        displays: Vec<DisplayDescriptor>,
    },
    /// Client → host. Subscribe to a subset of host displays. Empty
    /// vec means "the primary" (today, the only one). Sent on the
    /// client's user-driven display switch; host stops emitting video
    /// for displays not in the set.
    SetActiveDisplays {
        displays: Vec<DisplayId>,
    },
    /// Client → host. Sent once after the client has finished building
    /// its decoders and audio sink, so the host doesn't start blasting
    /// media before the receive side is ready. Booleans indicate which
    /// negotiated streams the client is prepared to consume.
    StreamReady {
        video: bool,
        audio: bool,
    },
    /// Client → host. Pause emission for the given display (e.g.
    /// window minimised). Host is free to stop encoding entirely for
    /// that display to save power.
    StreamPause {
        stream_id: VideoStreamId,
    },
    /// Client → host. Resume emission for the given display. Pairs
    /// with [`Self::StreamPause`]; host emits a fresh IDR before any
    /// subsequent P-frames so the client doesn't render a half-decoded
    /// stream.
    StreamResume {
        stream_id: VideoStreamId,
    },
    /// Client → host. Periodic receive-side telemetry (1 Hz typical).
    /// Feeds adaptive-bitrate / FEC / codec-downshift policy on the host.
    /// Count fields are per the elapsed `window_ms`; `rtt_us` is the
    /// client's current QUIC RTT estimate. FEC recovery counts are successful
    /// repairs only; failed incomplete frames are reported separately.
    ClientStats {
        window_ms: u32,
        frames_received: u32,
        incomplete_frames: u32,
        fragment_loss_events: u32,
        rtt_us: u32,
        fec_recovered_frames: u32,
        fec_recovered_fragments: u32,
    },
    /// Either side → other. Switch the cursor input model. Toggles
    /// mid-session without renegotiating; the receiver echoes the
    /// same message back as ack. While [`CursorMode::Relative`] is
    /// active the client stops emitting `ClientCursorPacket` and
    /// instead emits `InputEventKind::RelativeMouseMove` on the
    /// reliable input stream — the only path that survives
    /// recenter-loop games and raw-input HID readers.
    SetCursorMode {
        mode: CursorMode,
    },
    /// Client → host. Best-effort request to retarget one encoded stream at
    /// the client's current viewport dimensions. This does not change the
    /// host display mode and does not require an ack.
    SetViewportHint {
        stream_id: VideoStreamId,
        viewport: Viewport,
    },
    /// Client → host. Request a real host display mode change. The host must
    /// reply with [`Self::DisplayModeResult`].
    SetDisplayMode {
        request_id: RequestId,
        display_id: DisplayId,
        mode: DisplayMode,
        restore_on_disconnect: bool,
    },
    /// Host → client. Result of a [`Self::SetDisplayMode`] request.
    DisplayModeResult {
        request_id: RequestId,
        display_id: DisplayId,
        status: DisplayModeStatus,
        actual_mode: Option<DisplayMode>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisplayModeStatus {
    Applied,
    Unsupported,
    InvalidMode,
    PermissionDenied,
    Failed,
    Unknown(i32),
}

/// Physical-pixel dimensions of the client's rendering surface. Used to size
/// the encoder so we don't ship more pixels than the client will display.
/// Width and height are positive integers — `0` in either is treated by the
/// host as "ignore this viewport, use native."
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

impl Viewport {
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// `true` if both dimensions are non-zero. Hosts must check this
    /// before using a viewport — a peer that sends `(0, 0)` is
    /// explicitly opting out of viewport-driven sizing for the
    /// duration of the connection.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.width > 0 && self.height > 0
    }
}

// --- ClockSync ----------------------------------------------------------

/// Number of timestamp probes the client sends as a burst after handshake.
///
/// Clock offset estimation is only as good as the least-queued sample. A
/// single probe taken while the host/client are still spawning render and GPU
/// resources can overestimate frame age by an RTT-sized queueing spike, which
/// feeds directly into latency telemetry and present-policy frame skipping.
/// A small min-RTT burst follows standard NTP/PTP practice without adding N
/// full round trips to startup.
pub const CLOCK_SYNC_PROBE_SAMPLES: usize = 8;

/// Result of a successful three-way handshake probe. Computed from
/// `t0_client_send`, `t1_server_recv`, `t2_server_send`, `t3_client_recv`
/// using the standard NTP formula: `offset = ((t1 - t0) + (t2 - t3)) / 2`,
/// `rtt = (t3 - t0) - (t2 - t1)`.
///
/// Convention: `offset_nanos` is signed and reads as "remote clock minus
/// local clock". To translate a host (remote) timestamp into the client's
/// (local) clock: `client_equiv = host_time - offset_nanos`. Positive
/// offset means the remote was ahead at sample time. Cheap to clone;
/// passes around the connection's owned latency view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockSync {
    pub offset_nanos: i64,
    pub rtt_nanos: u64,
    /// Local-clock time at which this sync was sampled. Useful for
    /// detecting drift in long-running connections (re-probe on a timer
    /// and warn if the offset has moved much).
    pub sampled_at_local: MonoNanos,
}

impl ClockSync {
    /// Build a `ClockSync` from the four probe timestamps. `t0` and `t3`
    /// are in the local clock, `t1` and `t2` are in the remote clock.
    /// `rtt` saturates at zero in the (impossible-without-clock-jump)
    /// case where the remote claims it spent more time processing than
    /// the local observed in total.
    #[must_use]
    pub fn from_probe(
        t0_local_send: MonoNanos,
        t1_remote_recv: MonoNanos,
        t2_remote_send: MonoNanos,
        t3_local_recv: MonoNanos,
    ) -> Self {
        let t0 = i128::from(t0_local_send.0);
        let t1 = i128::from(t1_remote_recv.0);
        let t2 = i128::from(t2_remote_send.0);
        let t3 = i128::from(t3_local_recv.0);
        let offset = ((t1 - t0) + (t2 - t3)) / 2;
        let total = t3 - t0;
        let remote_processing = t2 - t1;
        let rtt = (total - remote_processing).max(0);
        Self {
            offset_nanos: i64::try_from(offset.clamp(i128::from(i64::MIN), i128::from(i64::MAX)))
                .expect("clamped to i64 range"),
            rtt_nanos: u64::try_from(rtt.clamp(0, i128::from(u64::MAX)))
                .expect("clamped to u64 range"),
            sampled_at_local: t3_local_recv,
        }
    }

    /// Pick the least-queued probe sample. Min-RTT is the best available proxy
    /// for path symmetry, so its offset is the least likely to include startup
    /// queueing from process spawn, GPU probing, or a busy control loop.
    #[must_use]
    pub fn best_sample(samples: impl IntoIterator<Item = Self>) -> Option<Self> {
        samples.into_iter().min_by_key(|sync| sync.rtt_nanos)
    }

    /// Translate a timestamp from the remote clock into the local clock.
    /// Saturates at zero rather than overflowing if the offset would
    /// drive the result negative (only possible early in a connection
    /// before both clocks have advanced past zero).
    #[must_use]
    pub fn remote_to_local(&self, remote: MonoNanos) -> MonoNanos {
        let v = i128::from(remote.0) - i128::from(self.offset_nanos);
        MonoNanos(u64::try_from(v.clamp(0, i128::from(u64::MAX))).expect("clamped to u64 range"))
    }
}

fn mono_to_pb(value: MonoNanos) -> pb::MonoNanos {
    pb::MonoNanos { value: value.0 }
}

fn mono_from_pb(value: Option<pb::MonoNanos>) -> Result<MonoNanos, CodecError> {
    Ok(MonoNanos(
        value.ok_or(CodecError::Wire("missing MonoNanos"))?.value,
    ))
}

fn codec_to_pb(value: CodecKind) -> i32 {
    match value {
        CodecKind::H264 => 1,
        CodecKind::Hevc => 2,
        CodecKind::Av1 => 3,
    }
}

fn codec_from_pb(value: i32) -> Result<CodecKind, CodecError> {
    match value {
        1 => Ok(CodecKind::H264),
        2 => Ok(CodecKind::Hevc),
        3 => Ok(CodecKind::Av1),
        _ => Err(CodecError::Wire("unknown CodecKind")),
    }
}

fn chroma_to_pb(value: ChromaSubsampling) -> i32 {
    match value {
        ChromaSubsampling::Yuv420 => 1,
        ChromaSubsampling::Yuv444 => 2,
    }
}

fn chroma_from_pb(value: i32) -> Result<ChromaSubsampling, CodecError> {
    match value {
        1 => Ok(ChromaSubsampling::Yuv420),
        2 => Ok(ChromaSubsampling::Yuv444),
        _ => Err(CodecError::Wire("unknown ChromaSubsampling")),
    }
}

fn profile_to_pb(value: VideoProfile) -> pb::VideoProfile {
    pb::VideoProfile {
        codec: codec_to_pb(value.codec),
        chroma: chroma_to_pb(value.chroma),
        bit_depth: u32::from(value.bit_depth),
    }
}

fn profile_from_pb(value: pb::VideoProfile) -> Result<VideoProfile, CodecError> {
    Ok(VideoProfile {
        codec: codec_from_pb(value.codec)?,
        chroma: chroma_from_pb(value.chroma)?,
        bit_depth: u8::try_from(value.bit_depth).map_err(|_| CodecError::Wire("bit depth > u8"))?,
    })
}

fn advertised_profile_from_pb(value: pb::VideoProfile) -> Option<VideoProfile> {
    let codec = codec_from_pb(value.codec).ok()?;
    let chroma = chroma_from_pb(value.chroma).ok()?;
    let bit_depth = u8::try_from(value.bit_depth).ok()?;
    Some(VideoProfile {
        codec,
        chroma,
        bit_depth,
    })
}

fn color_to_pb(value: VideoColorSpec) -> pb::VideoColorSpec {
    pb::VideoColorSpec {
        matrix: match value.matrix {
            ColorMatrix::Bt709 => 1,
            ColorMatrix::Bt2020Ncl => 2,
            ColorMatrix::Identity => 3,
        },
        range: match value.range {
            ColorRange::Limited => 1,
            ColorRange::Full => 2,
        },
        transfer: match value.transfer {
            ColorTransfer::Bt709 => 1,
            ColorTransfer::Srgb => 2,
            ColorTransfer::Pq => 3,
            ColorTransfer::Hlg => 4,
            ColorTransfer::Linear => 5,
        },
        primaries: match value.primaries {
            ColorPrimaries::Bt709 => 1,
            ColorPrimaries::Bt2020 => 2,
        },
    }
}

fn color_from_pb(value: Option<pb::VideoColorSpec>) -> Result<VideoColorSpec, CodecError> {
    let value = value.ok_or(CodecError::Wire("missing VideoColorSpec"))?;
    Ok(VideoColorSpec {
        matrix: match value.matrix {
            1 => ColorMatrix::Bt709,
            2 => ColorMatrix::Bt2020Ncl,
            3 => ColorMatrix::Identity,
            _ => return Err(CodecError::Wire("unknown ColorMatrix")),
        },
        range: match value.range {
            1 => ColorRange::Limited,
            2 => ColorRange::Full,
            _ => return Err(CodecError::Wire("unknown ColorRange")),
        },
        transfer: match value.transfer {
            1 => ColorTransfer::Bt709,
            2 => ColorTransfer::Srgb,
            3 => ColorTransfer::Pq,
            4 => ColorTransfer::Hlg,
            5 => ColorTransfer::Linear,
            _ => return Err(CodecError::Wire("unknown ColorTransfer")),
        },
        primaries: match value.primaries {
            1 => ColorPrimaries::Bt709,
            2 => ColorPrimaries::Bt2020,
            _ => return Err(CodecError::Wire("unknown ColorPrimaries")),
        },
    })
}

fn pixel_to_pb(value: PixelFormat) -> i32 {
    match value {
        PixelFormat::Bgra8 => 1,
        PixelFormat::Nv12 => 2,
        PixelFormat::P010 => 3,
        PixelFormat::Yuv444p => 4,
        PixelFormat::P410 => 5,
    }
}

fn pixel_from_pb(value: i32) -> Result<PixelFormat, CodecError> {
    match value {
        1 => Ok(PixelFormat::Bgra8),
        2 => Ok(PixelFormat::Nv12),
        3 => Ok(PixelFormat::P010),
        4 => Ok(PixelFormat::Yuv444p),
        5 => Ok(PixelFormat::P410),
        _ => Err(CodecError::Wire("unknown PixelFormat")),
    }
}

fn viewport_to_pb(value: Viewport) -> pb::Viewport {
    pb::Viewport {
        width: value.width,
        height: value.height,
    }
}

fn viewport_from_pb(value: pb::Viewport) -> Viewport {
    Viewport::new(value.width, value.height)
}

fn display_mode_to_pb(value: DisplayMode) -> pb::DisplayMode {
    pb::DisplayMode {
        width: value.width,
        height: value.height,
        refresh_millihz: value.refresh_millihz,
    }
}

fn display_mode_from_pb(value: Option<pb::DisplayMode>) -> Result<DisplayMode, CodecError> {
    let value = value.ok_or(CodecError::Wire("missing DisplayMode"))?;
    Ok(DisplayMode {
        width: value.width,
        height: value.height,
        refresh_millihz: value.refresh_millihz,
    })
}

fn display_to_pb(value: DisplayDescriptor) -> pb::DisplayDescriptor {
    pb::DisplayDescriptor {
        id: value.id.0,
        name: value.name,
        position_x: value.position.0,
        position_y: value.position.1,
        scale_num: u32::from(value.scale_num),
        scale_den: u32::from(value.scale_den),
        primary: value.primary,
        current_mode: Some(display_mode_to_pb(value.current_mode)),
        available_modes: value
            .available_modes
            .into_iter()
            .map(display_mode_to_pb)
            .collect(),
        can_set_mode: value.can_set_mode,
    }
}

fn display_from_pb(value: pb::DisplayDescriptor) -> Result<DisplayDescriptor, CodecError> {
    Ok(DisplayDescriptor {
        id: DisplayId(value.id),
        name: value.name,
        scale_num: u16::try_from(value.scale_num)
            .map_err(|_| CodecError::Wire("display scale numerator > u16"))?,
        scale_den: u16::try_from(value.scale_den)
            .map_err(|_| CodecError::Wire("display scale denominator > u16"))?,
        primary: value.primary,
        position: (value.position_x, value.position_y),
        current_mode: display_mode_from_pb(value.current_mode)?,
        available_modes: value
            .available_modes
            .into_iter()
            .map(|m| display_mode_from_pb(Some(m)))
            .collect::<Result<_, _>>()?,
        can_set_mode: value.can_set_mode,
    })
}

fn feature_advert_to_pb(value: FeatureAdvert) -> pb::FeatureAdvert {
    pb::FeatureAdvert {
        key: value.key,
        min_version: value.min_version,
        max_version: value.max_version,
        payload: value.payload,
    }
}

fn feature_advert_from_pb(value: pb::FeatureAdvert) -> FeatureAdvert {
    FeatureAdvert {
        key: value.key,
        min_version: value.min_version,
        max_version: value.max_version,
        payload: value.payload,
    }
}

fn feature_accept_to_pb(value: FeatureAccept) -> pb::FeatureAccept {
    pb::FeatureAccept {
        key: value.key,
        version: value.version,
        payload: value.payload,
    }
}

fn feature_accept_from_pb(value: pb::FeatureAccept) -> FeatureAccept {
    FeatureAccept {
        key: value.key,
        version: value.version,
        payload: value.payload,
    }
}

fn extension_to_pb(value: ExtensionMessage) -> pb::ExtensionMessage {
    pb::ExtensionMessage {
        key: value.key,
        version: value.version,
        request_id: value.request_id.0,
        reply_to: value.reply_to.0,
        payload: value.payload,
    }
}

fn extension_from_pb(value: pb::ExtensionMessage) -> ExtensionMessage {
    debug_assert!(
        value.payload.len() <= MAX_EXTENSION_PAYLOAD_BYTES,
        "extension payload cap should be checked before conversion"
    );
    ExtensionMessage {
        key: value.key,
        version: value.version,
        request_id: RequestId(value.request_id),
        reply_to: RequestId(value.reply_to),
        payload: value.payload,
    }
}

fn input_caps_to_pb(value: InputCapabilities) -> pb::InputCapabilities {
    pb::InputCapabilities {
        keyboard: value.keyboard,
        mouse: value.mouse,
        relative_mouse: value.relative_mouse,
        text: value.text,
    }
}

fn input_caps_from_pb(value: Option<pb::InputCapabilities>) -> InputCapabilities {
    value.map_or_else(InputCapabilities::default, |v| InputCapabilities {
        keyboard: v.keyboard,
        mouse: v.mouse,
        relative_mouse: v.relative_mouse,
        text: v.text,
    })
}

fn audio_to_pb(value: crate::audio::AudioConfig) -> pb::AudioConfig {
    pb::AudioConfig {
        sample_rate_hz: value.sample_rate_hz,
        channels: u32::from(value.channels),
        streams: u32::from(value.streams),
        coupled_streams: u32::from(value.coupled_streams),
        channel_mapping: value.channel_mapping,
    }
}

fn audio_from_pb(value: pb::AudioConfig) -> Result<crate::audio::AudioConfig, CodecError> {
    Ok(crate::audio::AudioConfig {
        sample_rate_hz: value.sample_rate_hz,
        channels: u8::try_from(value.channels).map_err(|_| CodecError::Wire("channels > u8"))?,
        streams: u8::try_from(value.streams).map_err(|_| CodecError::Wire("streams > u8"))?,
        coupled_streams: u8::try_from(value.coupled_streams)
            .map_err(|_| CodecError::Wire("coupled_streams > u8"))?,
        channel_mapping: value.channel_mapping,
    })
}

fn video_to_pb(value: NegotiatedVideo) -> pb::NegotiatedVideo {
    pb::NegotiatedVideo {
        stream_id: value.stream_id.0,
        display_id: value.display_id.0,
        profile: Some(profile_to_pb(value.profile)),
        pixel_format: pixel_to_pb(value.pixel_format),
        color_space: Some(color_to_pb(value.color_space)),
    }
}

fn video_from_pb(value: Option<pb::NegotiatedVideo>) -> Result<NegotiatedVideo, CodecError> {
    let value = value.ok_or(CodecError::Wire("missing NegotiatedVideo"))?;
    Ok(NegotiatedVideo {
        stream_id: VideoStreamId(value.stream_id),
        display_id: DisplayId(value.display_id),
        profile: profile_from_pb(
            value
                .profile
                .ok_or(CodecError::Wire("missing VideoProfile"))?,
        )?,
        pixel_format: pixel_from_pb(value.pixel_format)?,
        color_space: color_from_pb(value.color_space)?,
    })
}

fn handshake_failure_to_pb(value: HandshakeFailure) -> pb::HandshakeFailure {
    pb::HandshakeFailure {
        code: goodbye_to_pb(value.code),
        reason: value.reason,
    }
}

fn handshake_failure_from_pb(value: pb::HandshakeFailure) -> Result<HandshakeFailure, CodecError> {
    Ok(HandshakeFailure {
        code: goodbye_from_pb(value.code),
        reason: value.reason,
    })
}

fn status_to_pb(value: DisplayModeStatus) -> i32 {
    match value {
        DisplayModeStatus::Applied => 1,
        DisplayModeStatus::Unsupported => 2,
        DisplayModeStatus::InvalidMode => 3,
        DisplayModeStatus::PermissionDenied => 4,
        DisplayModeStatus::Failed => 5,
        DisplayModeStatus::Unknown(value) => value,
    }
}

fn status_from_pb(value: i32) -> DisplayModeStatus {
    match value {
        1 => DisplayModeStatus::Applied,
        2 => DisplayModeStatus::Unsupported,
        3 => DisplayModeStatus::InvalidMode,
        4 => DisplayModeStatus::PermissionDenied,
        5 => DisplayModeStatus::Failed,
        _ => DisplayModeStatus::Unknown(value),
    }
}

fn goodbye_to_pb(value: GoodbyeCode) -> i32 {
    match value {
        GoodbyeCode::Clean => 1,
        GoodbyeCode::ProtocolError => 2,
        GoodbyeCode::UnsupportedVersion => 3,
        GoodbyeCode::InternalError => 4,
        GoodbyeCode::Unknown(value) => value,
    }
}

fn goodbye_from_pb(value: i32) -> GoodbyeCode {
    match value {
        1 => GoodbyeCode::Clean,
        2 => GoodbyeCode::ProtocolError,
        3 => GoodbyeCode::UnsupportedVersion,
        4 => GoodbyeCode::InternalError,
        _ => GoodbyeCode::Unknown(value),
    }
}

fn session_summary_to_pb(value: SessionSummary) -> pb::SessionSummary {
    pb::SessionSummary {
        role: value.role,
        duration_ms: value.duration_ms,
        codec: value.codec,
        chroma: value.chroma,
        bit_depth: value.bit_depth,
        video: Some(video_session_stats_to_pb(value.video)),
        audio: value.audio.map(audio_session_stats_to_pb),
    }
}

fn session_summary_from_pb(value: pb::SessionSummary) -> Result<SessionSummary, CodecError> {
    Ok(SessionSummary {
        role: value.role,
        duration_ms: value.duration_ms,
        codec: value.codec,
        chroma: value.chroma,
        bit_depth: value.bit_depth,
        video: value
            .video
            .map(video_session_stats_from_pb)
            .unwrap_or_default(),
        audio: value.audio.map(audio_session_stats_from_pb),
    })
}

fn video_session_stats_to_pb(value: VideoSessionStats) -> pb::VideoSessionStats {
    pb::VideoSessionStats {
        frames_sent: value.frames_sent,
        frames_received: value.frames_received,
        keyframes: value.keyframes,
        bytes_sent: value.bytes_sent,
        bytes_received: value.bytes_received,
        incomplete_frames: value.incomplete_frames,
        fragment_loss_events: value.fragment_loss_events,
        decode_errors: value.decode_errors,
        render_drop_frames: value.render_drop_frames,
        idr_requests: value.idr_requests,
        decode_queue_drop_frames: value.decode_queue_drop_frames,
        transient_send_drop_frames: value.transient_send_drop_frames,
        fec_recovered_frames: value.fec_recovered_frames,
        fec_recovered_fragments: value.fec_recovered_fragments,
        datagrams_sent: value.datagrams_sent,
        parity_datagrams_sent: value.parity_datagrams_sent,
        max_datagrams_per_frame: value.max_datagrams_per_frame,
        max_frame_bytes: value.max_frame_bytes,
        max_keyframe_bytes: value.max_keyframe_bytes,
        forced_idr_misses: value.forced_idr_misses,
        decode_stale_epoch_drop_frames: value.decode_stale_epoch_drop_frames,
        decode_epoch_throttle_drop_frames: value.decode_epoch_throttle_drop_frames,
    }
}

fn video_session_stats_from_pb(value: pb::VideoSessionStats) -> VideoSessionStats {
    VideoSessionStats {
        frames_sent: value.frames_sent,
        frames_received: value.frames_received,
        keyframes: value.keyframes,
        bytes_sent: value.bytes_sent,
        bytes_received: value.bytes_received,
        incomplete_frames: value.incomplete_frames,
        fragment_loss_events: value.fragment_loss_events,
        decode_errors: value.decode_errors,
        render_drop_frames: value.render_drop_frames,
        idr_requests: value.idr_requests,
        decode_queue_drop_frames: value.decode_queue_drop_frames,
        transient_send_drop_frames: value.transient_send_drop_frames,
        fec_recovered_frames: value.fec_recovered_frames,
        fec_recovered_fragments: value.fec_recovered_fragments,
        datagrams_sent: value.datagrams_sent,
        parity_datagrams_sent: value.parity_datagrams_sent,
        max_datagrams_per_frame: value.max_datagrams_per_frame,
        max_frame_bytes: value.max_frame_bytes,
        max_keyframe_bytes: value.max_keyframe_bytes,
        forced_idr_misses: value.forced_idr_misses,
        decode_stale_epoch_drop_frames: value.decode_stale_epoch_drop_frames,
        decode_epoch_throttle_drop_frames: value.decode_epoch_throttle_drop_frames,
    }
}

fn audio_session_stats_to_pb(value: AudioSessionStats) -> pb::AudioSessionStats {
    pb::AudioSessionStats {
        packets_sent: value.packets_sent,
        packets_received: value.packets_received,
        capture_frames: value.capture_frames,
        underruns: value.underruns,
        dropped_samples: value.dropped_samples,
        recovered_frames: value.recovered_frames,
        concealed_frames: value.concealed_frames,
        dropout_frames: value.dropout_frames,
        dropouts: value.dropouts,
        stale_packets: value.stale_packets,
        decode_errors: value.decode_errors,
        decode_queue_drop_packets: value.decode_queue_drop_packets,
    }
}

fn audio_session_stats_from_pb(value: pb::AudioSessionStats) -> AudioSessionStats {
    AudioSessionStats {
        packets_sent: value.packets_sent,
        packets_received: value.packets_received,
        capture_frames: value.capture_frames,
        underruns: value.underruns,
        dropped_samples: value.dropped_samples,
        recovered_frames: value.recovered_frames,
        concealed_frames: value.concealed_frames,
        dropout_frames: value.dropout_frames,
        dropouts: value.dropouts,
        stale_packets: value.stale_packets,
        decode_errors: value.decode_errors,
        decode_queue_drop_packets: value.decode_queue_drop_packets,
    }
}

fn cursor_mode_to_pb(value: CursorMode) -> i32 {
    match value {
        CursorMode::Absolute => 1,
        CursorMode::Relative => 2,
        CursorMode::Unknown(value) => value,
    }
}

fn cursor_mode_from_pb(value: i32) -> CursorMode {
    match value {
        1 => CursorMode::Absolute,
        2 => CursorMode::Relative,
        _ => CursorMode::Unknown(value),
    }
}

fn cursor_pixel_to_pb(value: crate::cursor::CursorPixelFormat) -> i32 {
    match value {
        crate::cursor::CursorPixelFormat::Rgba8 => 1,
    }
}

fn cursor_pixel_from_pb(value: i32) -> Result<crate::cursor::CursorPixelFormat, CodecError> {
    match value {
        1 => Ok(crate::cursor::CursorPixelFormat::Rgba8),
        _ => Err(CodecError::Wire("unknown CursorPixelFormat")),
    }
}

impl ReliableMessage for ClientHello {
    fn encode_reliable(&self) -> Vec<u8> {
        pb::ClientHello {
            client_name: self.client_name.clone(),
            decode_profiles: self
                .decode_profiles
                .iter()
                .copied()
                .map(profile_to_pb)
                .collect(),
            initial_viewport: self.initial_viewport.map(viewport_to_pb),
            input_capabilities: Some(input_caps_to_pb(self.input_capabilities)),
            requested_features: self
                .requested_features
                .iter()
                .cloned()
                .map(feature_advert_to_pb)
                .collect(),
        }
        .encode_to_vec()
    }

    fn decode_reliable(bytes: &[u8]) -> Result<Self, CodecError> {
        let hello = pb::ClientHello::decode(bytes)?;
        Ok(Self {
            client_name: hello.client_name,
            decode_profiles: hello
                .decode_profiles
                .into_iter()
                .filter_map(advertised_profile_from_pb)
                .collect(),
            initial_viewport: hello.initial_viewport.map(viewport_from_pb),
            input_capabilities: input_caps_from_pb(hello.input_capabilities),
            requested_features: hello
                .requested_features
                .into_iter()
                .map(feature_advert_from_pb)
                .collect(),
        })
    }
}

impl ReliableMessage for ServerHello {
    fn encode_reliable(&self) -> Vec<u8> {
        pb::ServerHello {
            server_name: self.server_name.clone(),
            video: Some(video_to_pb(self.video.clone())),
            audio: self.audio.clone().map(audio_to_pb),
            displays: self.displays.iter().cloned().map(display_to_pb).collect(),
            accepted_features: self
                .accepted_features
                .iter()
                .cloned()
                .map(feature_accept_to_pb)
                .collect(),
            video_streams: Vec::new(),
        }
        .encode_to_vec()
    }

    fn decode_reliable(bytes: &[u8]) -> Result<Self, CodecError> {
        let hello = pb::ServerHello::decode(bytes)?;
        Ok(Self {
            server_name: hello.server_name,
            video: video_from_pb(hello.video)?,
            audio: hello.audio.map(audio_from_pb).transpose()?,
            displays: hello
                .displays
                .into_iter()
                .map(display_from_pb)
                .collect::<Result<_, _>>()?,
            accepted_features: hello
                .accepted_features
                .into_iter()
                .map(feature_accept_from_pb)
                .collect(),
        })
    }
}

impl ReliableMessage for ServerHandshake {
    fn encode_reliable(&self) -> Vec<u8> {
        use pb::server_handshake::Kind;
        let kind = match self.clone() {
            ServerHandshake::Accepted(server) => {
                let pb = pb::ServerHello {
                    server_name: server.server_name,
                    video: Some(video_to_pb(server.video)),
                    audio: server.audio.map(audio_to_pb),
                    displays: server.displays.into_iter().map(display_to_pb).collect(),
                    accepted_features: server
                        .accepted_features
                        .into_iter()
                        .map(feature_accept_to_pb)
                        .collect(),
                    video_streams: Vec::new(),
                };
                Kind::Accepted(pb)
            }
            ServerHandshake::Rejected(failure) => Kind::Rejected(handshake_failure_to_pb(failure)),
        };
        pb::ServerHandshake { kind: Some(kind) }.encode_to_vec()
    }

    fn decode_reliable(bytes: &[u8]) -> Result<Self, CodecError> {
        use pb::server_handshake::Kind;
        let handshake = pb::ServerHandshake::decode(bytes)?;
        match handshake
            .kind
            .ok_or(CodecError::Wire("missing ServerHandshake kind"))?
        {
            Kind::Accepted(server) => Ok(ServerHandshake::Accepted(ServerHello {
                server_name: server.server_name,
                video: video_from_pb(server.video)?,
                audio: server.audio.map(audio_from_pb).transpose()?,
                displays: server
                    .displays
                    .into_iter()
                    .map(display_from_pb)
                    .collect::<Result<_, _>>()?,
                accepted_features: server
                    .accepted_features
                    .into_iter()
                    .map(feature_accept_from_pb)
                    .collect(),
            })),
            Kind::Rejected(failure) => Ok(ServerHandshake::Rejected(handshake_failure_from_pb(
                failure,
            )?)),
        }
    }
}

impl ReliableMessage for ControlMessage {
    fn encode_reliable(&self) -> Vec<u8> {
        use pb::control_message::Kind;
        let kind = match self.clone() {
            ControlMessage::ForceIdr => Kind::ForceIdr(pb::Empty {}),
            ControlMessage::RequestRecovery {
                last_reassembled_frame_id,
            } => Kind::RequestRecovery(pb::RequestRecovery {
                last_reassembled_frame_id,
            }),
            ControlMessage::ClockProbeRequest { t0_sender } => {
                Kind::ClockProbeRequest(pb::ClockProbeRequest {
                    t0_sender: Some(mono_to_pb(t0_sender)),
                })
            }
            ControlMessage::ClockProbeResponse(probe) => Kind::ClockProbeResponse(pb::ClockProbe {
                t0_sender: Some(mono_to_pb(probe.t0_sender)),
                t1_receiver_recv: Some(mono_to_pb(probe.t1_receiver_recv)),
                t2_receiver_send: Some(mono_to_pb(probe.t2_receiver_send)),
            }),
            ControlMessage::Goodbye {
                reason,
                code,
                final_stats,
            } => Kind::Goodbye(pb::Goodbye {
                reason,
                code: goodbye_to_pb(code),
                final_stats: final_stats.map(|summary| Box::new(session_summary_to_pb(*summary))),
            }),
            ControlMessage::Extension(msg) => Kind::Extension(extension_to_pb(msg)),
            ControlMessage::CursorShape {
                id,
                hotspot,
                width,
                height,
                format,
                pixels,
            } => Kind::CursorShape(pb::CursorShape {
                id,
                hotspot_x: u32::from(hotspot.0),
                hotspot_y: u32::from(hotspot.1),
                width: u32::from(width),
                height: u32::from(height),
                format: cursor_pixel_to_pb(format),
                pixels,
            }),
            ControlMessage::CursorUseShape { id } => {
                Kind::CursorUseShape(pb::CursorUseShape { id })
            }
            ControlMessage::DisplayList { displays } => Kind::DisplayList(pb::DisplayList {
                displays: displays.into_iter().map(display_to_pb).collect(),
            }),
            ControlMessage::SetActiveDisplays { displays } => {
                Kind::SetActiveDisplays(pb::SetActiveDisplays {
                    displays: displays.into_iter().map(|d| d.0).collect(),
                })
            }
            ControlMessage::StreamReady { video, audio } => {
                Kind::StreamReady(pb::StreamReady { video, audio })
            }
            ControlMessage::StreamPause { stream_id } => Kind::StreamPause(pb::StreamPause {
                stream_id: stream_id.0,
            }),
            ControlMessage::StreamResume { stream_id } => Kind::StreamResume(pb::StreamResume {
                stream_id: stream_id.0,
            }),
            ControlMessage::ClientStats {
                window_ms,
                frames_received,
                incomplete_frames,
                fragment_loss_events,
                rtt_us,
                fec_recovered_frames,
                fec_recovered_fragments,
            } => Kind::ClientStats(pb::ClientStats {
                window_ms,
                frames_received,
                incomplete_frames,
                fragment_loss_events,
                rtt_us,
                fec_recovered_frames,
                fec_recovered_fragments,
            }),
            ControlMessage::SetCursorMode { mode } => Kind::SetCursorMode(pb::SetCursorMode {
                mode: cursor_mode_to_pb(mode),
            }),
            ControlMessage::SetViewportHint {
                stream_id,
                viewport,
            } => Kind::SetViewportHint(pb::SetViewportHint {
                stream_id: stream_id.0,
                viewport: Some(viewport_to_pb(viewport)),
            }),
            ControlMessage::SetDisplayMode {
                request_id,
                display_id,
                mode,
                restore_on_disconnect,
            } => Kind::SetDisplayMode(pb::SetDisplayMode {
                request_id: request_id.0,
                display_id: display_id.0,
                mode: Some(display_mode_to_pb(mode)),
                restore_on_disconnect,
            }),
            ControlMessage::DisplayModeResult {
                request_id,
                display_id,
                status,
                actual_mode,
            } => Kind::DisplayModeResult(pb::DisplayModeResult {
                request_id: request_id.0,
                display_id: display_id.0,
                status: status_to_pb(status),
                actual_mode: actual_mode.map(display_mode_to_pb),
            }),
        };
        pb::ControlMessage { kind: Some(kind) }.encode_to_vec()
    }

    fn decode_reliable(bytes: &[u8]) -> Result<Self, CodecError> {
        use pb::control_message::Kind;
        let msg = pb::ControlMessage::decode(bytes)?;
        let kind = msg
            .kind
            .ok_or(CodecError::Wire("missing ControlMessage kind"))?;
        match kind {
            Kind::ForceIdr(_) => Ok(ControlMessage::ForceIdr),
            Kind::RequestRecovery(v) => Ok(ControlMessage::RequestRecovery {
                last_reassembled_frame_id: v.last_reassembled_frame_id,
            }),
            Kind::ClockProbeRequest(v) => Ok(ControlMessage::ClockProbeRequest {
                t0_sender: mono_from_pb(v.t0_sender)?,
            }),
            Kind::ClockProbeResponse(v) => Ok(ControlMessage::ClockProbeResponse(ClockProbe {
                t0_sender: mono_from_pb(v.t0_sender)?,
                t1_receiver_recv: mono_from_pb(v.t1_receiver_recv)?,
                t2_receiver_send: mono_from_pb(v.t2_receiver_send)?,
            })),
            Kind::Goodbye(v) => Ok(ControlMessage::Goodbye {
                reason: v.reason,
                code: goodbye_from_pb(v.code),
                final_stats: v
                    .final_stats
                    .map(|summary| session_summary_from_pb(*summary).map(Box::new))
                    .transpose()?,
            }),
            Kind::Extension(v) => {
                if v.payload.len() > MAX_EXTENSION_PAYLOAD_BYTES {
                    return Err(CodecError::Wire("extension payload too large"));
                }
                Ok(ControlMessage::Extension(extension_from_pb(v)))
            }
            Kind::CursorShape(v) => {
                if v.pixels.len() > MAX_CURSOR_SHAPE_BYTES {
                    return Err(CodecError::Wire("cursor shape payload too large"));
                }
                let width =
                    u16::try_from(v.width).map_err(|_| CodecError::Wire("cursor width > u16"))?;
                let height =
                    u16::try_from(v.height).map_err(|_| CodecError::Wire("cursor height > u16"))?;
                let expected = usize::from(width)
                    .saturating_mul(usize::from(height))
                    .saturating_mul(4);
                if v.pixels.len() != expected {
                    return Err(CodecError::Wire("cursor shape payload length mismatch"));
                }
                Ok(ControlMessage::CursorShape {
                    id: v.id,
                    hotspot: (
                        u16::try_from(v.hotspot_x)
                            .map_err(|_| CodecError::Wire("hotspot_x > u16"))?,
                        u16::try_from(v.hotspot_y)
                            .map_err(|_| CodecError::Wire("hotspot_y > u16"))?,
                    ),
                    width,
                    height,
                    format: cursor_pixel_from_pb(v.format)?,
                    pixels: v.pixels,
                })
            }
            Kind::CursorUseShape(v) => Ok(ControlMessage::CursorUseShape { id: v.id }),
            Kind::DisplayList(v) => Ok(ControlMessage::DisplayList {
                displays: v
                    .displays
                    .into_iter()
                    .map(display_from_pb)
                    .collect::<Result<_, _>>()?,
            }),
            Kind::SetActiveDisplays(v) => Ok(ControlMessage::SetActiveDisplays {
                displays: v.displays.into_iter().map(DisplayId).collect(),
            }),
            Kind::StreamReady(v) => Ok(ControlMessage::StreamReady {
                video: v.video,
                audio: v.audio,
            }),
            Kind::StreamPause(v) => Ok(ControlMessage::StreamPause {
                stream_id: VideoStreamId(v.stream_id),
            }),
            Kind::StreamResume(v) => Ok(ControlMessage::StreamResume {
                stream_id: VideoStreamId(v.stream_id),
            }),
            Kind::ClientStats(v) => Ok(ControlMessage::ClientStats {
                window_ms: v.window_ms,
                frames_received: v.frames_received,
                incomplete_frames: v.incomplete_frames,
                fragment_loss_events: v.fragment_loss_events,
                rtt_us: v.rtt_us,
                fec_recovered_frames: v.fec_recovered_frames,
                fec_recovered_fragments: v.fec_recovered_fragments,
            }),
            Kind::SetCursorMode(v) => Ok(ControlMessage::SetCursorMode {
                mode: cursor_mode_from_pb(v.mode),
            }),
            Kind::SetViewportHint(v) => Ok(ControlMessage::SetViewportHint {
                stream_id: VideoStreamId(v.stream_id),
                viewport: viewport_from_pb(
                    v.viewport
                        .ok_or(CodecError::Wire("missing viewport hint"))?,
                ),
            }),
            Kind::SetDisplayMode(v) => Ok(ControlMessage::SetDisplayMode {
                request_id: RequestId(v.request_id),
                display_id: DisplayId(v.display_id),
                mode: display_mode_from_pb(v.mode)?,
                restore_on_disconnect: v.restore_on_disconnect,
            }),
            Kind::DisplayModeResult(v) => Ok(ControlMessage::DisplayModeResult {
                request_id: RequestId(v.request_id),
                display_id: DisplayId(v.display_id),
                status: status_from_pb(v.status),
                actual_mode: v
                    .actual_mode
                    .map(|m| display_mode_from_pb(Some(m)))
                    .transpose()?,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_sync_recovers_offset_and_rtt() {
        // Client send at t0=100; remote (host) clock is +400ns ahead;
        // one-way delay is 5ns; remote spends 10ns processing.
        let t0 = MonoNanos(100);
        let t1 = MonoNanos(505); // 100 + 5 (one-way) + 400 (offset)
        let t2 = MonoNanos(515); // t1 + 10 (processing)
        let t3 = MonoNanos(120); // t0 + 5 + 10 + 5
        let sync = ClockSync::from_probe(t0, t1, t2, t3);
        assert_eq!(sync.offset_nanos, 400);
        assert_eq!(sync.rtt_nanos, 10);
    }

    #[test]
    fn clock_sync_best_sample_keeps_min_rtt_offset() {
        let noisy = ClockSync {
            offset_nanos: 145_000_000,
            rtt_nanos: 290_000_000,
            sampled_at_local: MonoNanos(10),
        };
        let clean = ClockSync {
            offset_nanos: 4_000,
            rtt_nanos: 5_000_000,
            sampled_at_local: MonoNanos(20),
        };

        assert_eq!(ClockSync::best_sample([noisy, clean]), Some(clean));
    }

    #[test]
    fn remote_to_local_subtracts_offset() {
        let sync = ClockSync {
            offset_nanos: 400,
            rtt_nanos: 10,
            sampled_at_local: MonoNanos(120),
        };
        assert_eq!(sync.remote_to_local(MonoNanos(1000)), MonoNanos(600));
    }

    #[test]
    fn remote_to_local_saturates_on_underflow() {
        let sync = ClockSync {
            offset_nanos: 1000,
            rtt_nanos: 10,
            sampled_at_local: MonoNanos(0),
        };
        assert_eq!(sync.remote_to_local(MonoNanos(100)), MonoNanos(0));
    }

    #[test]
    fn clock_sync_zero_rtt_is_legal() {
        // t0 == t3, t1 == t2: the impossible "zero round trip" case
        // that can arise on the very first probe over a loopback or
        // shared-memory transport. Math should produce rtt=0, not
        // saturate to something weird.
        let t = MonoNanos(1_000);
        let sync = ClockSync::from_probe(t, t, t, t);
        assert_eq!(sync.rtt_nanos, 0);
        assert_eq!(sync.offset_nanos, 0);
    }

    #[test]
    fn clock_sync_rtt_saturates_when_remote_claims_negative_processing() {
        // Pathological case: remote reports t2 < t1 (clock jumped
        // backwards mid-probe). Total - processing would be > total,
        // which would naively inflate rtt. Code uses `.max(0)` on rtt
        // — this test pins that.
        let t0 = MonoNanos(100);
        let t1 = MonoNanos(500);
        let t2 = MonoNanos(400); // t2 < t1: remote clock jumped backwards
        let t3 = MonoNanos(120);
        let sync = ClockSync::from_probe(t0, t1, t2, t3);
        // (t3-t0) - (t2-t1) = 20 - (-100) = 120; non-negative, so no
        // saturation; the test guards against a future regression
        // that would let this go negative.
        assert!(sync.rtt_nanos <= u64::from(u32::MAX));
    }

    #[test]
    fn clock_sync_handles_huge_offset() {
        // Offset close to i64::MAX. Should clamp inside from_probe
        // rather than panic on the i64::try_from. The values are
        // chosen so (t1 - t0) is large and positive; (t2 - t3) is
        // similarly large.
        let t0 = MonoNanos(0);
        let t1 = MonoNanos(u64::MAX / 2);
        let t2 = MonoNanos((u64::MAX / 2).saturating_add(10));
        let t3 = MonoNanos(20);
        let sync = ClockSync::from_probe(t0, t1, t2, t3);
        // No panic; offset clamped into i64 range.
        assert!(sync.offset_nanos > 0);
    }
}
