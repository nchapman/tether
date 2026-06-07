import { useRef, useState } from "react";

import {
  type EngineEvent,
  type SavedHost,
  connectClient,
  disconnectClient,
  forgetKnownHost,
  renameKnownHost,
  hideWindow,
  showWindow,
  friendlyError,
} from "../ipc";
import {
  type ClientState,
  type Confirm,
  type ConnectVia,
  type RowError,
  type SheetName,
} from "../state";

/// What the client session needs from the App shell. Everything here is a
/// stable React setter or a `useRef` (whose object identity is stable across
/// renders) or a function that only reads through such refs — so the handlers
/// below stay correct even when captured once by a long-lived Tauri listener.
export type ClientSessionDeps = {
  windowHidden: React.MutableRefObject<boolean>;
  restoreWindow: () => void;
  refreshHosts: () => void;
  labelFor: (addr: string) => string;
  setSheet: React.Dispatch<React.SetStateAction<SheetName>>;
  setConfirm: (confirm: Confirm | null) => void;
};

/// Owns the client (outbound connection) state machine: connecting →
/// connected / errored, the per-row inline errors, and the actions that drive
/// it. Mirrors the live state through `clientRef` so the once-registered engine
/// listeners read current values (see [`ClientSessionDeps`]).
export function useClientSession({
  windowHidden,
  restoreWindow,
  refreshHosts,
  labelFor,
  setSheet,
  setConfirm,
}: ClientSessionDeps) {
  const [client, setClient] = useState<ClientState>({ kind: "idle" });
  const [rowErrors, setRowErrors] = useState<Record<string, RowError>>({});
  const clientRef = useRef(client);
  clientRef.current = client;

  /// Fold an `engine-status` event for the client role into state.
  function handleEvent(p: EngineEvent) {
    const prev = clientRef.current;
    switch (p.event) {
      case "connecting":
        // We set "connecting" optimistically at initiate-time (so we know the
        // addr + via); only adopt the event if we somehow missed that.
        if (prev.kind === "idle") {
          setClient({ kind: "connecting", addr: p.host, via: "saved" });
        }
        break;
      case "connected": {
        const addr =
          prev.kind === "connecting" || prev.kind === "connected"
            ? prev.addr
            : p.host;
        setClient({ kind: "connected", addr, profile: p.profile });
        setRowErrors((e) => {
          const n = { ...e };
          delete n[addr];
          return n;
        });
        // New first-contact entry + refreshed recency both come from disk.
        refreshHosts();
        setSheet("none");
        // Get the chrome out of the way; the engine's video window is what the
        // user wants in front now.
        hideWindow();
        windowHidden.current = true;
        break;
      }
      case "disconnected":
        restoreWindow();
        setClient({ kind: "idle" });
        break;
      case "error": {
        // Force-show the window so the failure is visible even for a
        // tray-initiated connect (which never hid it through us).
        showWindow();
        windowHidden.current = false;
        const addr = prev.kind !== "idle" ? prev.addr : "";
        const via: ConnectVia =
          prev.kind === "connecting" || prev.kind === "error" ? prev.via : "saved";
        const fe = friendlyError(p.message, labelFor(addr));
        if (via === "add") {
          setClient({ kind: "error", addr, via, message: fe.message });
        } else {
          setRowErrors((e) => ({ ...e, [addr]: { message: fe.message, pairAgain: fe.pairAgain } }));
          setClient({ kind: "idle" });
        }
        break;
      }
    }
  }

  /// The engine exited (clean or crash); bring the chrome back if we hid it.
  /// Don't clobber an error the Add sheet is showing; otherwise stop a stuck
  /// spinner and re-home to the address book.
  function handleExit() {
    const prev = clientRef.current;
    restoreWindow();
    if (prev.kind !== "error") setClient({ kind: "idle" });
  }

  async function beginConnect(
    addr: string,
    via: ConnectVia,
    pin: string | null = null,
    label: string | null = null,
  ) {
    setRowErrors((e) => {
      const n = { ...e };
      delete n[addr];
      return n;
    });
    setClient({ kind: "connecting", addr, via });
    try {
      await connectClient({ addr, pin, label });
    } catch (e) {
      const fe = friendlyError(String(e), labelFor(addr));
      if (via === "add") setClient({ kind: "error", addr, via, message: fe.message });
      else {
        setRowErrors((er) => ({ ...er, [addr]: { message: fe.message, pairAgain: fe.pairAgain } }));
        setClient({ kind: "idle" });
      }
    }
  }

  /// Connect to a saved row. If a different session is live, confirm the switch
  /// first so we never silently drop a connection.
  function onConnect(addr: string) {
    const c = clientRef.current;
    if ((c.kind === "connected" || c.kind === "connecting") && c.addr !== addr) {
      setConfirm({
        title: "Switch computer?",
        message: `Disconnect from ${labelFor(c.addr)} and connect to ${labelFor(addr)}?`,
        confirmLabel: "Switch",
        onConfirm: () => {
          setConfirm(null);
          beginConnect(addr, "saved");
        },
      });
    } else {
      beginConnect(addr, "saved");
    }
  }

  /// Explicit Disconnect. Like stopping the host, the engine's explicit stop
  /// emits no `engine-exited`, so reset optimistically. The window is already
  /// visible (the user clicked Disconnect in it), so there's nothing to restore.
  function onDisconnect() {
    disconnectClient().catch((e) => console.warn(e));
    windowHidden.current = false;
    setClient({ kind: "idle" });
  }

  function onForget(target: SavedHost) {
    setConfirm({
      title: `Forget ${target.label}?`,
      message: "You'll need its pairing PIN to connect again.",
      confirmLabel: "Forget",
      danger: true,
      onConfirm: () => {
        setConfirm(null);
        forgetKnownHost(target.addr).then(refreshHosts).catch((e) => console.warn(e));
      },
    });
  }

  function onRename(addr: string, label: string) {
    renameKnownHost(addr, label).then(refreshHosts).catch((e) => console.warn(e));
  }

  /// Clear a stale Add-sheet error from a prior attempt (called when reopening
  /// the Add sheet). Saved-row errors live in `rowErrors`, not here.
  function clearAddError() {
    if (clientRef.current.kind === "error") setClient({ kind: "idle" });
  }

  // The Add sheet needs a coarse status + the error message for its `via:"add"`
  // attempts (saved-row attempts surface inline, not in the sheet).
  const addStatus: "idle" | "connecting" | "error" =
    client.kind === "connecting" && client.via === "add"
      ? "connecting"
      : client.kind === "error" && client.via === "add"
        ? "error"
        : "idle";
  const addError = client.kind === "error" && client.via === "add" ? client.message : undefined;

  return {
    client,
    rowErrors,
    handleEvent,
    handleExit,
    beginConnect,
    onConnect,
    onDisconnect,
    onForget,
    onRename,
    clearAddError,
    addStatus,
    addError,
  };
}
