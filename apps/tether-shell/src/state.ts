// UI-side session state derived from engine events. Kept separate from ipc.ts
// (which is the wire shapes) so components can import these types without a
// dependency on App.

import type { PairedPeer } from "./ipc";

/// How the client connection initiated, so we route a failure to the right
/// surface: a saved-row click shows an inline error under that row; an Add
/// sheet attempt shows the error in the sheet.
export type ConnectVia = "saved" | "add";

/// An inline error shown under a saved row. `pairAgain` marks the case where
/// the host no longer recognizes this computer, so the row offers a "Pair
/// again" action (re-opens the Add sheet pre-filled with that address).
export type RowError = { message: string; pairAgain?: boolean };

export type ClientState =
  | { kind: "idle" }
  | { kind: "connecting"; addr: string; via: ConnectVia }
  | { kind: "connected"; addr: string; profile: string }
  | { kind: "error"; addr: string; via: ConnectVia; message: string };

/// Host (sharing) state mirrored from the host engine's events.
export type HostState = {
  running: boolean;
  addr: string;
  fingerprint: string;
  /// The connected client's address, or null while waiting.
  peer: string | null;
  /// Active pairing PIN and its live countdown, or null when not pairing.
  pin: string | null;
  secondsLeft: number;
  pendingPeer: string | null;
  peers: PairedPeer[];
  error: string | null;
};

export const initialHostState: HostState = {
  running: false,
  addr: "",
  fingerprint: "",
  peer: null,
  pin: null,
  secondsLeft: 0,
  pendingPeer: null,
  peers: [],
  error: null,
};
