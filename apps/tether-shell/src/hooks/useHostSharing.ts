import { useEffect, useLayoutEffect, useRef, useState } from "react";

import {
  type EngineEvent,
  type PairedPeer,
  startHost,
  stopHost,
  startPairing,
  revokePeer,
  listPeers,
} from "../ipc";
import { type Confirm, type HostState, initialHostState } from "../state";

export type HostSharingDeps = {
  setConfirm: (confirm: Confirm | null) => void;
};

/// Owns the host (sharing) state machine mirrored from the host engine's
/// events, the pairing-PIN countdown, and the sharing actions. `hostRef` mirrors
/// live state for the once-registered engine listeners and for App's
/// open-sharing handler.
export function useHostSharing({ setConfirm }: HostSharingDeps) {
  const [host, setHost] = useState<HostState>(initialHostState);
  const hostRef = useRef(host);
  useLayoutEffect(() => {
    hostRef.current = host;
  }, [host]);

  // Tick the pairing-PIN countdown; clear when it elapses.
  useEffect(() => {
    if (host.pin === null || host.secondsLeft <= 0) return;
    const t = setTimeout(() => {
      setHost((h) => {
        const secondsLeft = Math.max(0, h.secondsLeft - 1);
        return { ...h, secondsLeft, pin: secondsLeft === 0 ? null : h.pin };
      });
    }, 1000);
    return () => clearTimeout(t);
  }, [host.pin, host.secondsLeft]);

  /// Fold an `engine-status` event for the host role into state.
  function handleEvent(p: EngineEvent) {
    switch (p.event) {
      case "listening":
        setHost((h) => ({ ...h, running: true, addr: p.addr, error: null }));
        void listPeers().catch((e: unknown) => console.warn("list_peers failed", e));
        break;
      case "peer_connected":
        setHost((h) => ({ ...h, peer: p.peer }));
        break;
      case "peer_disconnected":
        setHost((h) => ({ ...h, peer: null }));
        break;
      case "pairing_pin":
        setHost((h) => ({ ...h, pin: p.pin, secondsLeft: p.expires_in_secs }));
        break;
      case "paired":
        setHost((h) => ({ ...h, pin: null, pendingPeer: null }));
        break;
      case "pairing_required":
        setHost((h) => ({ ...h, pendingPeer: p.peer }));
        break;
      case "peer_list":
        setHost((h) => ({ ...h, peers: p.peers }));
        break;
      case "error":
        setHost((h) => ({ ...h, running: false, error: p.message }));
        break;
    }
  }

  /// The host engine exited (crash or self-exit); reset to off.
  function handleExit() {
    setHost(initialHostState);
  }

  /// Start sharing. Surfaces a spawn failure (e.g. engine binary missing) into
  /// `host.error` so the Sharing sheet explains why it didn't come up, rather
  /// than a silently-off toggle. Shared by the sheet toggle and the launch-time
  /// posture auto-restore.
  function startSharing() {
    void startHost().catch((e: unknown) => {
      console.warn("start host failed", e);
      setHost((h) => ({ ...h, error: String(e) }));
    });
  }

  /// The supervisor's *explicit* stop emits no `engine-exited` (it removes the
  /// handle before the stdout reader hits EOF), so reset optimistically here —
  /// otherwise the toggle would never flip back to off. stop_engine reliably
  /// tears the engine down, so showing "off" immediately is correct.
  function stopSharing() {
    void stopHost().catch((e: unknown) => console.warn(e));
    setHost(initialHostState);
  }

  function onStopSharing() {
    const peer = hostRef.current.peer;
    if (peer) {
      setConfirm({
        title: "Stop sharing?",
        message: `${peer} is connected and will be disconnected.`,
        confirmLabel: "Stop sharing",
        danger: true,
        onConfirm: () => {
          setConfirm(null);
          stopSharing();
        },
      });
    } else {
      stopSharing();
    }
  }

  function onAddDevice(label: string) {
    void startPairing(label).catch((e: unknown) => console.warn(e));
  }

  function onRevoke(peer: PairedPeer) {
    setConfirm({
      title: `Revoke ${peer.label}?`,
      message: "It won't be able to connect until you pair it again.",
      confirmLabel: "Revoke",
      danger: true,
      onConfirm: () => {
        setConfirm(null);
        void revokePeer(peer.fingerprint).catch((e: unknown) => console.warn(e));
      },
    });
  }

  return {
    host,
    hostRef,
    handleEvent,
    handleExit,
    startSharing,
    onStopSharing,
    onAddDevice,
    onRevoke,
  };
}
