//! Shell ↔ engine IPC for the Tether multi-process UI.
//!
//! The Tauri shell (control plane) spawns the `tether-host` /
//! `tether-client` engine binaries as child processes and talks to them
//! over **stdio JSON-lines**: one JSON object per line. The engine's
//! **stdout carries IPC events only** (tracing logs are routed to stderr
//! in IPC mode, see each binary's `init_tracing`); stdin carries
//! commands. Stdout is block-buffered when piped, so [`Reporter`] flushes
//! after every line — otherwise the shell wouldn't see events until the
//! buffer filled.
//!
//! These types are the wire contract between three crates (shell, host,
//! client); they live here so the three can't drift. Following the repo
//! convention (see `tether-protocol`), every message has a round-trip
//! test below.
//!
//! Scope is deliberately the connection lifecycle + pairing the shell
//! reflects in its UI + tray. Variants land only when a binary actually
//! emits them — no speculative variants (e.g. a `Stats` event) ahead of the
//! feature that drives them.

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

/// Engine → shell, written one-per-line to the engine's stdout.
///
/// `#[serde(tag = "event")]` gives each line a flat
/// `{"event":"listening", ...}` shape that's trivial to switch on in the
/// shell's TypeScript without a discriminator wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum EngineEvent {
    /// Host bound its socket and is ready for clients. `fingerprint` is
    /// the hex SHA-256 the user pairs with out of band.
    Listening { addr: String, fingerprint: String },
    /// Host accepted a client connection.
    PeerConnected { peer: String },
    /// The host's current client session ended (clean or error); the
    /// host loops back to accepting.
    PeerDisconnected { reason: String },
    /// Client is dialing the host (pre-handshake).
    Connecting { host: String },
    /// Client finished the handshake; `profile` is the negotiated video
    /// profile, surfaced so the UI can show what's actually streaming.
    Connected { host: String, profile: String },
    /// Client session ended.
    Disconnected { reason: String },
    /// A fatal error the UI should surface. The engine exits shortly
    /// after emitting this.
    Error { message: String },
    /// Host opened a pairing window: show this PIN for the user to type on
    /// the new device. `expires_in_secs` drives the UI countdown; the PIN is
    /// single-use and the window closes when it elapses.
    PairingPin { pin: String, expires_in_secs: u64 },
    /// A device completed pairing and was added to the allowlist. `label` is
    /// the name recorded for it; `peer` is its address.
    Paired { peer: String, label: String },
    /// An unpaired client tried to connect while no pairing window was open;
    /// it was refused. The UI prompts the user to start pairing to allow it.
    PairingRequired { peer: String },
    /// The host's current paired-clients allowlist, sent in reply to
    /// [`ShellCommand::ListPeers`] and also pushed after a pairing or
    /// revocation so the UI's device list stays current without polling.
    PeerList { peers: Vec<PairedPeer> },
}

/// One entry in [`EngineEvent::PeerList`]: a paired client the host will admit.
/// `fingerprint` is the algorithm-tagged (`"sha256:<hex>"`) identity the UI
/// passes back to [`ShellCommand::RevokePeer`]; `label` is its display name and
/// `paired_at_unix` is when it was paired (Unix seconds).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairedPeer {
    pub fingerprint: String,
    pub label: String,
    pub paired_at_unix: u64,
}

/// Shell → engine, written one-per-line to the engine's stdin.
///
/// Stdin EOF (the shell process dying) is an implicit `Stop` — engines
/// treat a closed stdin the same as an explicit stop so a crashed shell
/// never orphans an engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum ShellCommand {
    /// Tear down the session and exit gracefully.
    Stop,
    /// Open a pairing window (host only): generate a PIN and accept one
    /// pairing attempt. `label` is the name to record for the device that
    /// pairs in this window. The host replies with [`EngineEvent::PairingPin`].
    StartPairing { label: String },
    /// Remove a paired device from the allowlist (host only), keyed by its
    /// algorithm-tagged fingerprint (`"sha256:<hex>"`). Any live session from
    /// that device is also dropped.
    RevokePeer { fingerprint: String },
    /// Ask the host (host only) to report its paired-clients allowlist; it
    /// replies with [`EngineEvent::PeerList`].
    ListPeers,
}

const COMMAND_VARIANTS: &[&str] = &["stop", "start_pairing", "revoke_peer", "list_peers"];

const CMD_ONLY_FIELDS: &[&str] = &["cmd"];
const START_PAIRING_FIELDS: &[&str] = &["cmd", "label"];
const REVOKE_PEER_FIELDS: &[&str] = &["cmd", "fingerprint"];

impl<'de> Deserialize<'de> for ShellCommand {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Cmd,
            Label,
            Fingerprint,
        }

        struct ShellCommandVisitor;

        impl<'de> Visitor<'de> for ShellCommandVisitor {
            type Value = ShellCommand;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a shell command object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut cmd = None;
                let mut label = None;
                let mut fingerprint = None;

                while let Some(field) = map.next_key()? {
                    match field {
                        Field::Cmd => {
                            if cmd.replace(map.next_value::<String>()?).is_some() {
                                return Err(de::Error::duplicate_field("cmd"));
                            }
                        }
                        Field::Label => {
                            if label.replace(map.next_value::<String>()?).is_some() {
                                return Err(de::Error::duplicate_field("label"));
                            }
                        }
                        Field::Fingerprint => {
                            if fingerprint.replace(map.next_value::<String>()?).is_some() {
                                return Err(de::Error::duplicate_field("fingerprint"));
                            }
                        }
                    }
                }

                let cmd = cmd.ok_or_else(|| de::Error::missing_field("cmd"))?;
                match cmd.as_str() {
                    "stop" => {
                        if label.is_some() {
                            return Err(de::Error::unknown_field("label", CMD_ONLY_FIELDS));
                        }
                        if fingerprint.is_some() {
                            return Err(de::Error::unknown_field("fingerprint", CMD_ONLY_FIELDS));
                        }
                        Ok(ShellCommand::Stop)
                    }
                    "start_pairing" => {
                        if fingerprint.is_some() {
                            return Err(de::Error::unknown_field(
                                "fingerprint",
                                START_PAIRING_FIELDS,
                            ));
                        }
                        let label = label.ok_or_else(|| de::Error::missing_field("label"))?;
                        Ok(ShellCommand::StartPairing { label })
                    }
                    "revoke_peer" => {
                        if label.is_some() {
                            return Err(de::Error::unknown_field("label", REVOKE_PEER_FIELDS));
                        }
                        let fingerprint =
                            fingerprint.ok_or_else(|| de::Error::missing_field("fingerprint"))?;
                        Ok(ShellCommand::RevokePeer { fingerprint })
                    }
                    "list_peers" => {
                        if label.is_some() {
                            return Err(de::Error::unknown_field("label", CMD_ONLY_FIELDS));
                        }
                        if fingerprint.is_some() {
                            return Err(de::Error::unknown_field("fingerprint", CMD_ONLY_FIELDS));
                        }
                        Ok(ShellCommand::ListPeers)
                    }
                    other => Err(de::Error::unknown_variant(other, COMMAND_VARIANTS)),
                }
            }
        }

        deserializer.deserialize_map(ShellCommandVisitor)
    }
}

impl ShellCommand {
    /// Parse one JSON-line command from the shell. Returns the textual
    /// parse error so the caller can log a malformed line and keep
    /// reading without taking a `serde_json` dependency of its own.
    ///
    /// # Errors
    ///
    /// Returns an error string when the line is not a valid JSON
    /// [`ShellCommand`], including unknown command names, missing fields,
    /// or extra fields.
    pub fn from_line(line: &str) -> Result<Self, String> {
        serde_json::from_str(line).map_err(|e| e.to_string())
    }
}

/// How a binary reports lifecycle: human-friendly stdout (default CLI
/// behavior) or machine-readable JSON-lines (when launched by the shell
/// with `--ipc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reporter {
    /// Friendly text on stdout, the way the standalone CLIs have always
    /// printed.
    Human,
    /// One [`EngineEvent`] JSON object per line on stdout, flushed
    /// immediately.
    Json,
}

impl Reporter {
    /// `--ipc` on the command line selects [`Reporter::Json`].
    #[must_use]
    pub fn from_ipc_flag(ipc: bool) -> Self {
        if ipc {
            Self::Json
        } else {
            Self::Human
        }
    }

    /// True when the binary is shell-driven (JSON-lines on stdout). The
    /// mains use this to suppress human-only chrome (usage hints, the
    /// cert-dir note) that has no [`EngineEvent`] equivalent.
    #[must_use]
    pub fn is_json(self) -> bool {
        matches!(self, Self::Json)
    }

    /// Report one lifecycle event. In `Json` mode this is a flushed
    /// JSON line; in `Human` mode it's the friendly text the CLIs
    /// historically printed.
    pub fn emit(self, event: &EngineEvent) {
        use std::io::Write;
        match self {
            Self::Json => {
                // serde_json on these flat structs cannot fail, but if
                // it ever did, dropping the line is better than panicking
                // on the host's status path.
                if let Ok(line) = serde_json::to_string(event) {
                    let mut out = std::io::stdout().lock();
                    let _ = writeln!(out, "{line}");
                    let _ = out.flush();
                }
            }
            Self::Human => {
                for line in event.human_lines() {
                    println!("{line}");
                }
            }
        }
    }
}

impl EngineEvent {
    /// Friendly multi-line rendering for [`Reporter::Human`]. Mirrors the
    /// text the standalone CLIs printed before IPC mode existed so the
    /// bare-binary experience is unchanged.
    fn human_lines(&self) -> Vec<String> {
        match self {
            EngineEvent::Listening { addr, fingerprint } => vec![
                format!("tether-host listening on {addr}"),
                format!("cert fingerprint: {fingerprint}"),
                format!("client cmd:      tether-client {addr} {fingerprint}"),
            ],
            EngineEvent::PeerConnected { peer } => vec![format!("client connected: {peer}")],
            EngineEvent::PeerDisconnected { reason } => {
                vec![format!("client disconnected: {reason}")]
            }
            EngineEvent::Connecting { host } => vec![format!("connecting to {host}…")],
            EngineEvent::Connected { host, profile } => {
                vec![format!("connected to {host} ({profile})")]
            }
            EngineEvent::Disconnected { reason } => vec![format!("disconnected: {reason}")],
            EngineEvent::Error { message } => vec![format!("error: {message}")],
            EngineEvent::PairingPin {
                pin,
                expires_in_secs,
            } => vec![format!(
                "pairing PIN: {pin} (expires in {expires_in_secs}s)"
            )],
            EngineEvent::Paired { peer, label } => {
                vec![format!("paired with {label} ({peer})")]
            }
            EngineEvent::PairingRequired { peer } => vec![format!(
                "unpaired client {peer} refused; start pairing to allow it"
            )],
            EngineEvent::PeerList { peers } => {
                let mut lines = vec![format!("paired devices ({}):", peers.len())];
                lines.extend(
                    peers
                        .iter()
                        .map(|p| format!("  {} ({})", p.label, p.fingerprint)),
                );
                lines
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every wire type round-trips through JSON — the shell parses
    /// exactly what the engine serializes.
    fn assert_event_roundtrip(event: &EngineEvent) {
        let line = serde_json::to_string(event).expect("serialize");
        assert!(!line.contains('\n'), "events must serialize to one line");
        let back: EngineEvent = serde_json::from_str(&line).expect("deserialize");
        assert_eq!(event, &back);
    }

    #[test]
    fn engine_events_roundtrip() {
        assert_event_roundtrip(&EngineEvent::Listening {
            addr: "127.0.0.1:7654".into(),
            fingerprint: "ab12".into(),
        });
        assert_event_roundtrip(&EngineEvent::PeerConnected {
            peer: "127.0.0.1:5000".into(),
        });
        assert_event_roundtrip(&EngineEvent::PeerDisconnected {
            reason: "clean".into(),
        });
        assert_event_roundtrip(&EngineEvent::Connecting {
            host: "127.0.0.1:7654".into(),
        });
        assert_event_roundtrip(&EngineEvent::Connected {
            host: "127.0.0.1:7654".into(),
            profile: "HEVC Main".into(),
        });
        assert_event_roundtrip(&EngineEvent::Disconnected {
            reason: "host closed".into(),
        });
        assert_event_roundtrip(&EngineEvent::Error {
            message: "no decoder".into(),
        });
        assert_event_roundtrip(&EngineEvent::PairingPin {
            pin: "48271930".into(),
            expires_in_secs: 120,
        });
        assert_event_roundtrip(&EngineEvent::Paired {
            peer: "127.0.0.1:5000".into(),
            label: "Nick's laptop".into(),
        });
        assert_event_roundtrip(&EngineEvent::PairingRequired {
            peer: "127.0.0.1:5000".into(),
        });
        assert_event_roundtrip(&EngineEvent::PeerList {
            peers: vec![PairedPeer {
                fingerprint: "sha256:abcd".into(),
                label: "Nick's laptop".into(),
                paired_at_unix: 1_700_000_000,
            }],
        });
        // Empty list must round-trip too (no devices paired yet).
        assert_event_roundtrip(&EngineEvent::PeerList { peers: vec![] });
    }

    #[test]
    fn shell_command_roundtrips() {
        for cmd in [
            ShellCommand::Stop,
            ShellCommand::StartPairing {
                label: "Nick's laptop".into(),
            },
            ShellCommand::RevokePeer {
                fingerprint: "sha256:abcd".into(),
            },
            ShellCommand::ListPeers,
        ] {
            let line = serde_json::to_string(&cmd).expect("serialize");
            let back: ShellCommand = serde_json::from_str(&line).expect("deserialize");
            assert_eq!(cmd, back);
        }
    }

    /// Pin the flat command tag the supervisor writes to engine stdin.
    #[test]
    fn start_pairing_has_flat_cmd_tag() {
        let line = serde_json::to_string(&ShellCommand::StartPairing {
            label: "laptop".into(),
        })
        .unwrap();
        assert_eq!(line, r#"{"cmd":"start_pairing","label":"laptop"}"#);
        let revoke = serde_json::to_string(&ShellCommand::RevokePeer {
            fingerprint: "sha256:abcd".into(),
        })
        .unwrap();
        assert_eq!(
            revoke,
            r#"{"cmd":"revoke_peer","fingerprint":"sha256:abcd"}"#
        );
        let list = serde_json::to_string(&ShellCommand::ListPeers).unwrap();
        assert_eq!(list, r#"{"cmd":"list_peers"}"#);
    }

    /// Pin the flat `event` tag + nested peer shape the shell's TS reads.
    #[test]
    fn peer_list_has_flat_event_tag() {
        let line = serde_json::to_string(&EngineEvent::PeerList {
            peers: vec![PairedPeer {
                fingerprint: "sha256:ab".into(),
                label: "x".into(),
                paired_at_unix: 7,
            }],
        })
        .unwrap();
        assert_eq!(
            line,
            r#"{"event":"peer_list","peers":[{"fingerprint":"sha256:ab","label":"x","paired_at_unix":7}]}"#
        );
    }

    #[test]
    fn from_line_parses_pairing_commands() {
        assert_eq!(
            ShellCommand::from_line(r#"{"cmd":"start_pairing","label":"x"}"#).unwrap(),
            ShellCommand::StartPairing { label: "x".into() }
        );
        assert_eq!(
            ShellCommand::from_line(r#"{"cmd":"revoke_peer","fingerprint":"sha256:ab"}"#).unwrap(),
            ShellCommand::RevokePeer {
                fingerprint: "sha256:ab".into()
            }
        );
    }

    /// The flat `event` tag is the shape the shell's TS switches on; pin
    /// it so a rename can't silently break the frontend.
    #[test]
    fn listening_has_flat_event_tag() {
        let line = serde_json::to_string(&EngineEvent::Listening {
            addr: "a".into(),
            fingerprint: "f".into(),
        })
        .unwrap();
        assert_eq!(
            line,
            r#"{"event":"listening","addr":"a","fingerprint":"f"}"#
        );
    }

    /// An unknown command line is a clean parse error, not a panic — the
    /// engine's stdin reader logs and ignores it.
    #[test]
    fn unknown_command_fails_cleanly() {
        assert!(ShellCommand::from_line(r#"{"cmd":"explode"}"#).is_err());
    }

    /// Extra fields are rejected so a malformed supervisor command cannot
    /// accidentally execute as a valid command with ignored baggage.
    #[test]
    fn command_with_unknown_field_fails_cleanly() {
        let err = ShellCommand::from_line(r#"{"cmd":"stop","fingerprint":"sha256:ab"}"#)
            .expect_err("extra fields must be rejected");
        assert!(err.contains("unknown field"), "{err}");
    }

    #[test]
    fn command_missing_required_field_fails_cleanly() {
        let err = ShellCommand::from_line(r#"{"cmd":"start_pairing"}"#)
            .expect_err("missing label must be rejected");
        assert!(err.contains("missing field"), "{err}");
    }

    #[test]
    fn from_line_parses_stop() {
        assert_eq!(
            ShellCommand::from_line(r#"{"cmd":"stop"}"#).unwrap(),
            ShellCommand::Stop
        );
    }
}
