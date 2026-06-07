// Typed wrappers over the Tauri command/event boundary, plus the shapes the
// shell shares with the Rust side. Keeping the `invoke` names and payloads in
// one place means the rest of the UI never stringly-types a command.

import { invoke } from "@tauri-apps/api/core";

/// One saved computer, mirroring `known_hosts::SavedHost` on the Rust side.
export type SavedHost = {
  addr: string;
  label: string;
  paired_at_unix: number;
  last_connected_unix: number | null;
};

/// One paired client the host will admit, mirroring `tether_ipc::PairedPeer`.
export type PairedPeer = {
  fingerprint: string;
  label: string;
  paired_at_unix: number;
};

/// Engine → shell lifecycle events, mirroring `tether_ipc::EngineEvent`
/// (flattened) plus the `role` the supervisor tags each line with.
export type EngineEvent =
  | { role: "host" | "client"; event: "listening"; addr: string; fingerprint: string }
  | { role: "host" | "client"; event: "peer_connected"; peer: string }
  | { role: "host" | "client"; event: "peer_disconnected"; reason: string }
  | { role: "host" | "client"; event: "connecting"; host: string }
  | { role: "host" | "client"; event: "connected"; host: string; profile: string }
  | { role: "host" | "client"; event: "disconnected"; reason: string }
  | { role: "host" | "client"; event: "error"; message: string }
  | { role: "host" | "client"; event: "pairing_pin"; pin: string; expires_in_secs: number }
  | { role: "host" | "client"; event: "paired"; peer: string; label: string }
  | { role: "host" | "client"; event: "pairing_required"; peer: string }
  | { role: "host" | "client"; event: "peer_list"; peers: PairedPeer[] };

export type EngineExited = { role: "host" | "client" };

/// Persisted shell preferences, mirroring `prefs::ShellPrefs` on the Rust side.
export type ShellPrefs = {
  /// Whether hosting was last left on; the shell re-applies this on launch.
  sharing_enabled: boolean;
};

// --- Client / address book ---------------------------------------------------

export const listKnownHosts = () => invoke<SavedHost[]>("list_known_hosts");

export const renameKnownHost = (addr: string, label: string) =>
  invoke<void>("rename_known_host", { addr, label });

export const forgetKnownHost = (addr: string) =>
  invoke<void>("forget_known_host", { addr });

/// Connect to a host. A `pin` means first-contact pairing; without one the
/// client reconnects via the host fingerprint it pinned last time.
export const connectClient = (args: {
  addr: string;
  pin?: string | null;
  label?: string | null;
}) => invoke<void>("connect_client", args);

export const disconnectClient = () => invoke<void>("stop_engine", { role: "client" });

// --- Host / sharing ----------------------------------------------------------

export const startHost = () => invoke<void>("start_host");

export const stopHost = () => invoke<void>("stop_engine", { role: "host" });

export const startPairing = (label: string) =>
  invoke<void>("start_pairing", { label });

export const revokePeer = (fingerprint: string) =>
  invoke<void>("revoke_peer", { fingerprint });

export const listPeers = () => invoke<void>("list_peers");

// --- Preferences -------------------------------------------------------------

/// Read the persisted shell prefs (sharing posture). Used on mount to decide
/// whether to auto-start hosting.
export const getPrefs = () => invoke<ShellPrefs>("get_prefs");

// --- Window chrome -----------------------------------------------------------

export const hideWindow = () => invoke<void>("hide_window");
export const showWindow = () => invoke<void>("show_window");

// --- Updates -----------------------------------------------------------------

export const checkForUpdates = () => invoke<string | null>("check_for_updates");
export const installUpdate = () => invoke<void>("install_update");

// --- Helpers -----------------------------------------------------------------

/// A human-readable error plus whether the fix is to pair again (the host no
/// longer trusts this computer), which the UI surfaces as a "Pair again" action.
export type FriendlyError = { message: string; pairAgain: boolean };

/// Translate a raw engine error string into human guidance. Engine errors are
/// freeform (`Error{message}`), so we match on the stable phrases the
/// host/client emit and fall back to the raw text. Never show the user a bare
/// `connect failed: …` dump — that's the biggest "this is a dev tool" tell.
export function friendlyError(raw: string, hostName: string): FriendlyError {
  const m = raw.toLowerCase();
  if (m.includes("pairing required") || m.includes("refused resume")) {
    return {
      message: `${hostName} no longer recognizes this computer. You'll need to pair again with a new PIN.`,
      pairAgain: true,
    };
  }
  if (m.includes("pairing failed")) {
    return {
      message:
        "That PIN didn't match — it may be wrong or expired. Check the PIN on the other computer and try again.",
      pairAgain: false,
    };
  }
  if (m.includes("connect failed") || m.includes("timed out") || m.includes("timeout") || m.includes("refused")) {
    return {
      message: `Couldn't reach ${hostName}. Make sure it's turned on and sharing is enabled there.`,
      pairAgain: false,
    };
  }
  return { message: raw, pairAgain: false };
}

/// "2h ago" / "yesterday" / "Mar 28" style recency for a saved host. Uses
/// `last_connected_unix` when present, else `paired_at_unix` (prefixed
/// "paired"). `nowMs` is injectable for testing.
export function recencyLabel(host: SavedHost, nowMs = Date.now()): string {
  const usedConnect = host.last_connected_unix != null;
  const unix = host.last_connected_unix ?? host.paired_at_unix;
  const rel = relativeTime(unix * 1000, nowMs);
  return usedConnect ? rel : `paired ${rel}`;
}

function relativeTime(thenMs: number, nowMs: number): string {
  const sec = Math.max(0, Math.round((nowMs - thenMs) / 1000));
  if (sec < 45) return "just now";
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.round(hr / 24);
  if (day === 1) return "yesterday";
  if (day < 7) return `${day}d ago`;
  // Older than a week: a calendar date reads better than "37d ago".
  return new Date(thenMs).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}
