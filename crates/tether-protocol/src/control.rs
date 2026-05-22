//! Control stream: handshake, clock sync, IDR requests, shutdown.
//!
//! # Versioning policy
//!
//! Both handshake messages ([`ClientHello`], [`ServerHello`]) are
//! `enum`-shaped wire envelopes whose variants tag a body struct
//! ([`ClientHelloV1`], [`ServerHelloV1`]). New protocol revisions land
//! as additional variants (`V2(ClientHelloV2)` etc.). Bincode encodes
//! enum variants with a varint discriminator, so a future V2 sent to
//! a V1-only receiver fails decode cleanly (`DecodeError::UnexpectedVariant`)
//! rather than silently misinterpreting the bytes. The transport
//! surfaces that as a `Result::Err` to the caller, which then closes
//! the connection — there's no in-band Goodbye round-trip on a
//! decode failure (we'd have to know what the peer can parse, which
//! is exactly what failed).
//!
//! Inside a body struct, **only new variants are wire-additive — never
//! appended fields**. Bincode is strictly positional within a struct
//! with no length prefix, so a V1 encoder that grew a new field at the
//! end would either leave trailing bytes attributed to the *next*
//! framed message or hit EOF in an older decoder. Adding anything to
//! `ClientHelloV1` after this point requires a `ClientHelloV2`.
//! The `#[serde(default)]` attributes on `extensions` and `resume_token`
//! are no-ops under bincode (positional, never sees a missing field) but
//! stay correct under serde formats with field-name tagging, in case
//! we ever export to JSON for telemetry.
//!
//! Forward-compatible *opt-in features* go through [`ClientHelloV1::extensions`]
//! / [`ServerHelloV1::extensions`] instead. The map shape lets either
//! side advertise a feature key (and its value payload) without a wire
//! revision; unknown keys are ignored. **Key naming convention:**
//! reverse-DNS-style `vendor.feature` (`tether.adaptive-bitrate`,
//! `tether.av1-preferred`) so first-party and future third-party
//! extensions don't collide.
//!
//! [`ControlMessage`] is closed: new variants on it are NOT
//! wire-additive (the discriminator collides with future codepoints
//! the peer might already understand differently). Adding a new
//! `ControlMessage` variant requires landing a `ClientHelloV2`
//! alongside it.
//!
//! # Reserved extension keys
//!
//! These keys have meanings the protocol has committed to but the
//! payload formats aren't yet stabilised. First-party features should
//! not collide; third-party builds that want to use them should pick
//! a different reverse-DNS prefix.
//!
//! - `tether.audio` — host audio config; payload is bincode-encoded
//!   [`crate::audio::AudioConfig`]. Advertised on `ServerHelloV1`.
//! - `tether.pixel-format` — host video pixel format; payload is
//!   bincode-encoded [`PixelFormat`]. Advertised on `ServerHelloV1`.
//! - `tether.gamepad-rumble` — host → client rumble command; rides
//!   [`ControlMessage::Extension`] until it earns a typed variant
//!   in a future hello revision. Payload shape: TBD pending the
//!   gamepad pipeline.
//! - `tether.cap.*` — capability advertisement. See the next section.
//! - `tether.cap.video.decode-profiles` — client → host. Bincode
//!   `Vec<VideoProfile>`; the full set of video profiles the client can
//!   decode. Absence is interpreted as legacy `[{H264, Yuv420, 8}]`.
//! - `tether.cap.video.encode-profile` — host → client. Bincode
//!   [`VideoProfile`]; the single profile the host picked from the
//!   intersection of its encode capabilities and the client's decode
//!   capabilities. Echoed in the [`ServerHelloV1`]; the inline
//!   [`ServerHelloV1::chosen_codec`] / [`ServerHelloV1::chosen_chroma`]
//!   fields carry the same information for legacy clients.
//!
//! # Capability advertisement (`tether.cap.*`)
//!
//! Any hello-extension key beginning with `tether.cap.` is a capability
//! advertisement. The sender is offering the feature; the receiver, if
//! it accepts the offer, MUST echo the key (with the negotiated payload
//! it agrees to use) back in its own hello `extensions`. Receivers that
//! recognise a `tether.cap.*` key but do not accept it MUST omit it
//! from the echo; receivers that do not recognise it MUST omit it from
//! the echo (the standard "unknown key, ignore" rule). Receivers MUST
//! NOT silently *consume* a capability key without acknowledging —
//! either echo or drop.
//!
//! That gives every future negotiated feature (FEC tuning, relay path,
//! adaptive-bitrate handshake, bandwidth probe parameters,
//! multi-stream session) a single shared idiom for "did the peer
//! accept this?" — no per-feature ack message, no per-feature timeout
//! dance. The presence of the key in the *other* side's hello
//! `extensions` is the ack; absence is rejection.
//!
//! Example flow for a hypothetical FEC negotiation:
//! - Client → `tether.cap.fec = encode(FecCapability { reed_solomon: true, .. })`.
//! - Host accepts: echoes `tether.cap.fec = encode(FecCapability { reed_solomon: true, .. })`
//!   with the parameters it agrees to (likely a subset / tightened
//!   variant of what the client offered).
//! - Host declines: omits the key. Client sees no echo and falls back.
//!
//! # The `Extension` escape
//!
//! [`ControlMessage::Extension`] exists so that future features which
//! need a new control message *do not* force a `ClientHelloV2`. A
//! sender publishes a reverse-DNS-keyed payload; receivers that
//! don't recognise the key log and drop it without erroring. Typical
//! lifecycle of a new control feature:
//!
//! 1. Ship the feature as `ControlMessage::Extension { key:
//!    "tether.feature-x", payload: <bincode-encoded body> }`.
//! 2. Soak it until the shape stabilises across deployments.
//! 3. When promoting to a typed variant becomes worthwhile (compile-
//!    time field checking, no String overhead per message), land it
//!    as part of the next `ClientHelloV{N+1}` bump.
//!
//! Receivers MUST log unknown extension keys at `debug` so an
//! operator can see "peer is speaking a feature this build doesn't
//! support yet."

use std::collections::BTreeMap;

use crate::MonoNanos;
use serde::{Deserialize, Serialize};

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
/// One end advertises the set it can decode; the other intersects with
/// the set it can encode and picks the best mutual profile against a
/// fixed preference order. Carried in hello extensions
/// ([`CLIENT_DECODE_PROFILES_EXTENSION_KEY`] / [`SERVER_ENCODE_PROFILE_EXTENSION_KEY`])
/// rather than inline fields so adding a new axis (10-bit, future
/// HDR-specific profile) doesn't require a [`ClientHelloV1`] bump.
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
    /// we ship supports it, every client backend can decode it. This
    /// is what a legacy peer (no `tether.cap.video.*` extension) is
    /// assumed to support.
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
}

/// Hello extension key. Client → host. Payload: bincode
/// `Vec<VideoProfile>`. Absence is legacy `[VideoProfile::H264_8BIT_420]`.
pub const CLIENT_DECODE_PROFILES_EXTENSION_KEY: &str = "tether.cap.video.decode-profiles";

/// Hello extension key. Host → client. Payload: bincode [`VideoProfile`].
/// The single profile the host picked. Absence means the host built
/// against an older protocol revision — the inline [`ServerHelloV1::chosen_codec`]
/// and [`ServerHelloV1::chosen_chroma`] fields are then the source of truth.
pub const SERVER_ENCODE_PROFILE_EXTENSION_KEY: &str = "tether.cap.video.encode-profile";

// =============================================================
// Four-axis color spec. Carried first-class on `ServerHelloV1`.
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
/// [`ServerHelloV1::color_space`].
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
/// Advertised via [`ServerHelloV1::extensions`] under
/// [`PIXEL_FORMAT_EXTENSION_KEY`]; absence implies `Nv12`.
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
    /// HEVC Main444 emits; selected when [`ServerHelloV1::chosen_chroma`]
    /// is [`ChromaSubsampling::Yuv444`].
    Yuv444p,
    /// 10-bit 4:4:4 biplanar — the path HEVC Main 4:4:4 10-bit rides
    /// on macOS (matching VT's `'P410'` / `'xf44'` IOSurfaces) and
    /// (if the renderer-side biplanar 16-bit import is what's wired)
    /// on Linux too. 10-bit MSB-aligned in 16-bit cells.
    P410,
}

/// Hello extension key for [`PixelFormat`] advertisement. Reverse-DNS
/// per the [`ClientHelloV1::extensions`] convention.
pub const PIXEL_FORMAT_EXTENSION_KEY: &str = "tether.pixel-format";

/// Describes one host display, carried in
/// [`ControlMessage::DisplayList`]. Today the host always sends a
/// one-element list (single-monitor); the field shape is here so adding
/// multi-monitor support later is purely additive — no new
/// `ControlMessage` variant, no V2 hello bump. `refresh_mhz` and the
/// `scale_*` pair also cover what would otherwise need their own
/// extension keys (display geometry / HiDPI hints).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisplayDescriptor {
    /// Host-assigned display id. Stable for the lifetime of a session;
    /// matches the `display: u8` on `VideoPacket` / cursor packets.
    pub id: u8,
    /// Human-readable name as the host's compositor reports it
    /// (`HDMI-A-1`, `DP-3`, `Built-in Retina Display`, etc.). May be
    /// empty if the host's capture backend doesn't expose names.
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Refresh rate in millihertz so 60 Hz = 60_000 and 59.94 Hz =
    /// 59_940 round-trip without floating point.
    pub refresh_mhz: u32,
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

// --- Versioned handshake envelopes --------------------------------------

/// Wire envelope for the client's opening message.
///
/// New protocol revisions add additional variants alongside `V1`. See
/// the module-level docs for the full versioning policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClientHello {
    V1(ClientHelloV1),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientHelloV1 {
    pub client_name: String,
    /// Codecs the client can decode, ordered by preference.
    pub preferred_codecs: Vec<CodecKind>,
    /// Client's maximum displayable resolution (host decides actual).
    pub max_resolution: Option<(u32, u32)>,
    /// First leg of the handshake clock probe — client's monotonic time at
    /// the moment of send.
    pub clock_probe_t0: MonoNanos,
    /// Opt-in extension envelope for forward-compatible feature flags
    /// (AV1 advertise, adaptive bitrate hint, future channels). Both
    /// sides ignore keys they don't recognise — adding a feature here
    /// does not require a protocol revision.
    #[serde(default)]
    pub extensions: BTreeMap<String, Vec<u8>>,
    /// If the client is attempting to resume a previous session, the
    /// token the host issued in that session's [`ServerHelloV1::resume_token`].
    /// `None` on a fresh session. The host may accept (preserving
    /// `stream_epoch` so the renderer can continue without a black
    /// frame) or reject (issue a fresh epoch); semantics are TBD. The
    /// protocol shape is here now so adding the implementation later
    /// doesn't require a wire bump.
    #[serde(default)]
    pub resume_token: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServerHello {
    V1(ServerHelloV1),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerHelloV1 {
    pub server_name: String,
    pub chosen_codec: CodecKind,
    pub chosen_chroma: ChromaSubsampling,
    /// Color identity of the encoded stream — matrix, range,
    /// transfer, primaries. Negotiated end-to-end so the renderer
    /// can dispatch the right shader path (EOTF + matrix) without
    /// guessing.
    pub color_space: VideoColorSpec,
    pub resolution: (u32, u32),
    /// Echo of the client's `clock_probe_t0` so the client can match
    /// the response to the request it sent.
    pub clock_probe_t0_echo: MonoNanos,
    /// Server-monotonic time when `ClientHello` was received.
    pub t1_server_recv: MonoNanos,
    /// Server-monotonic time when this message is sent.
    pub t2_server_send: MonoNanos,
    /// Opt-in extension envelope — same purpose as
    /// [`ClientHelloV1::extensions`].
    #[serde(default)]
    pub extensions: BTreeMap<String, Vec<u8>>,
    /// Opaque token the client can stash and present in a future
    /// [`ClientHelloV1::resume_token`] to attempt session resume.
    /// `None` if the host doesn't (yet) implement resume.
    #[serde(default)]
    pub resume_token: Option<Vec<u8>>,
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
}

/// Messages exchanged on the reliable control stream after handshake.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlMessage {
    /// Client requests an immediate IDR (e.g., after detected packet loss).
    ForceIdr,
    /// Periodic clock-sync re-probe (either side may initiate).
    ClockProbeRequest { t0_sender: MonoNanos },
    ClockProbeResponse(ClockProbe),
    Goodbye {
        reason: String,
        code: GoodbyeCode,
    },
    /// Escape hatch for features that don't yet warrant a typed
    /// variant. Payload is opaque to the protocol — the convention
    /// is reverse-DNS-style keys (`tether.clipboard`,
    /// `tether.gamepad-rumble`) with a bincode-encoded body. Unknown
    /// keys are logged + dropped. See the module-level doc.
    Extension {
        key: String,
        payload: Vec<u8>,
    },
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
    CursorUseShape { id: u64 },
    /// Host → client. Full display topology. Sent post-handshake and
    /// again on hotplug (display added/removed) — the client treats
    /// each message as the authoritative replacement. Single-monitor
    /// hosts send a one-element list; the field shape supports
    /// multi-monitor without a V2.
    DisplayList { displays: Vec<DisplayDescriptor> },
    /// Client → host. Subscribe to a subset of host displays. Empty
    /// vec means "the primary" (today, the only one). Sent on the
    /// client's user-driven display switch; host stops emitting video
    /// for displays not in the set.
    SetActiveDisplays { displays: Vec<u8> },
    /// Client → host. Sent once after the client has finished building
    /// its decoders, so the host doesn't start blasting video before
    /// the receive side is ready. Booleans indicate which streams the
    /// client is prepared to consume; `audio` is reserved for the
    /// future Opus pipeline (always `false` from clients today).
    StreamReady { video: bool, audio: bool },
    /// Client → host. Pause emission for the given display (e.g.
    /// window minimised). Host is free to stop encoding entirely for
    /// that display to save power.
    StreamPause { display: u8 },
    /// Client → host. Resume emission for the given display. Pairs
    /// with [`Self::StreamPause`]; host emits a fresh IDR before any
    /// subsequent P-frames so the client doesn't render a half-decoded
    /// stream.
    StreamResume { display: u8 },
    /// Client → host. Periodic receive-side telemetry (1 Hz typical).
    /// Feeds future adaptive-bitrate / FEC / codec-downshift policy on
    /// the host. Counters are per the elapsed `interval_ms` window;
    /// `rtt_ewma_us` is the EWMA over the connection's lifetime so far.
    ClientStats {
        interval_ms: u32,
        frames_received: u32,
        frames_dropped: u32,
        fragments_lost: u32,
        rtt_ewma_us: u32,
    },
}

// --- ClockSync ----------------------------------------------------------

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
            offset_nanos: i64::try_from(offset.clamp(
                i128::from(i64::MIN),
                i128::from(i64::MAX),
            ))
            .expect("clamped to i64 range"),
            rtt_nanos: u64::try_from(rtt.clamp(0, i128::from(u64::MAX)))
                .expect("clamped to u64 range"),
            sampled_at_local: t3_local_recv,
        }
    }

    /// Translate a timestamp from the remote clock into the local clock.
    /// Saturates at zero rather than overflowing if the offset would
    /// drive the result negative (only possible early in a connection
    /// before both clocks have advanced past zero).
    #[must_use]
    pub fn remote_to_local(&self, remote: MonoNanos) -> MonoNanos {
        let v = i128::from(remote.0) - i128::from(self.offset_nanos);
        MonoNanos(
            u64::try_from(v.clamp(0, i128::from(u64::MAX)))
                .expect("clamped to u64 range"),
        )
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
