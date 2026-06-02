import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

import {
  type EngineEvent,
  type EngineExited,
  type PairedPeer,
  type SavedHost,
  listKnownHosts,
  renameKnownHost,
  forgetKnownHost,
  connectClient,
  disconnectClient,
  startHost,
  stopHost,
  startPairing,
  revokePeer,
  listPeers,
  hideWindow,
  showWindow,
  getPrefs,
  checkForUpdates,
  installUpdate,
  friendlyError,
} from "./ipc";
import {
  type ClientState,
  type ConnectVia,
  type HostState,
  type RowError,
  initialHostState,
} from "./state";
import { ConnectView } from "./components/ConnectView";
import { AddComputerSheet } from "./components/AddComputerSheet";
import { SharingSheet } from "./components/SharingSheet";
import { ConfirmDialog } from "./components/ConfirmDialog";

type Sheet = "none" | "add" | "sharing";
type Confirm = {
  title: string;
  message: string;
  confirmLabel: string;
  danger?: boolean;
  onConfirm: () => void;
};

function App() {
  const [hosts, setHosts] = useState<SavedHost[]>([]);
  const [client, setClient] = useState<ClientState>({ kind: "idle" });
  const [host, setHost] = useState<HostState>(initialHostState);
  const [rowErrors, setRowErrors] = useState<Record<string, RowError>>({});
  const [sheet, setSheet] = useState<Sheet>("none");
  const [addPrefill, setAddPrefill] = useState<string | undefined>(undefined);
  const [confirm, setConfirm] = useState<Confirm | null>(null);

  // Listeners are registered once; they read current state through refs.
  const clientRef = useRef(client);
  clientRef.current = client;
  const hostsRef = useRef(hosts);
  hostsRef.current = hosts;
  const hostRef = useRef(host);
  hostRef.current = host;
  // Tracks whether we hid the window for a live session, so we restore it on
  // *any* terminal transition (clean disconnect, error, or crash-exit) without
  // re-showing — and stealing focus — when it was never hidden.
  const windowHidden = useRef(false);

  function restoreWindow() {
    if (windowHidden.current) {
      showWindow();
      windowHidden.current = false;
    }
  }

  const refreshHosts = () =>
    listKnownHosts()
      .then(setHosts)
      .catch((e) => console.warn("list_known_hosts failed", e));

  const labelFor = (addr: string) =>
    hostsRef.current.find((h) => h.addr === addr)?.label ?? addr;

  // --- Engine event plumbing -------------------------------------------------

  useEffect(() => {
    refreshHosts();

    const unstatus = listen<EngineEvent>("engine-status", ({ payload }) => {
      if (payload.role === "client") handleClientEvent(payload);
      else handleHostEvent(payload);
    });
    const unexit = listen<EngineExited>("engine-exited", ({ payload }) => {
      if (payload.role === "client") {
        const prev = clientRef.current;
        // The session ended (clean or crash); bring the chrome back if we hid
        // it. Don't clobber an error the Add sheet is showing; otherwise stop a
        // stuck spinner and re-home to the address book.
        restoreWindow();
        if (prev.kind !== "error") setClient({ kind: "idle" });
      } else {
        setHost(initialHostState);
      }
    });
    // The tray's "Pair a device…" shows the window and asks us to open Sharing.
    const unsharing = listen("open-sharing", () => {
      setSheet("sharing");
      if (hostRef.current.running) listPeers().catch((e) => console.warn(e));
    });
    // The tray asks us to connect through the window (only when a session is
    // already live, so the switch-computer confirm runs).
    const unconnect = listen<string>("request-connect", ({ payload }) => {
      onConnect(payload);
    });

    // Re-apply the persisted sharing posture: if hosting was last left on,
    // start it again. Wait until both listeners are live so we don't miss the
    // host's `listening` event (the toggle would otherwise show off while the
    // engine is actually up). startHost re-persists the posture, harmlessly.
    Promise.all([unstatus, unexit])
      .then(() => getPrefs())
      .then((prefs) => {
        if (prefs.sharing_enabled) {
          // Surface a failure into host state so the Sharing sheet shows why
          // hosting didn't come back (e.g. engine binary missing), rather than
          // a silently-off toggle.
          startHost().catch((e) => {
            console.warn("auto-start host failed", e);
            setHost((h) => ({ ...h, error: String(e) }));
          });
        }
      })
      .catch((e) => console.warn("get_prefs failed", e));

    return () => {
      unstatus.then((f) => f());
      unexit.then((f) => f());
      unsharing.then((f) => f());
      unconnect.then((f) => f());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleClientEvent(p: EngineEvent) {
    const prev = clientRef.current;
    switch (p.event) {
      case "connecting":
        // We set "connecting" optimistically at initiate-time (so we know the
        // addr + via); only adopt the event if we somehow missed that.
        if (prev.kind === "idle") {
          setClient({ kind: "connecting", addr: p.host ?? "", via: "saved" });
        }
        break;
      case "connected": {
        const addr =
          prev.kind === "connecting" || prev.kind === "connected"
            ? prev.addr
            : p.host ?? "";
        setClient({ kind: "connected", addr, profile: p.profile ?? "" });
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
        const addr = prev.kind !== "idle" ? prev.addr : p.host ?? "";
        const via: ConnectVia =
          prev.kind === "connecting" || prev.kind === "error" ? prev.via : "saved";
        const fe = friendlyError(p.message ?? "Something went wrong.", labelFor(addr));
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

  function handleHostEvent(p: EngineEvent) {
    switch (p.event) {
      case "listening":
        setHost((h) => ({
          ...h,
          running: true,
          addr: p.addr ?? "",
          error: null,
        }));
        listPeers().catch((e) => console.warn("list_peers failed", e));
        break;
      case "peer_connected":
        setHost((h) => ({ ...h, peer: p.peer ?? "a client" }));
        break;
      case "peer_disconnected":
        setHost((h) => ({ ...h, peer: null }));
        break;
      case "pairing_pin":
        setHost((h) => ({ ...h, pin: p.pin ?? null, secondsLeft: p.expires_in_secs ?? 0 }));
        break;
      case "paired":
        setHost((h) => ({ ...h, pin: null, pendingPeer: null }));
        break;
      case "pairing_required":
        setHost((h) => ({ ...h, pendingPeer: p.peer ?? "a device" }));
        break;
      case "peer_list":
        setHost((h) => ({ ...h, peers: p.peers ?? [] }));
        break;
      case "error":
        setHost((h) => ({ ...h, running: false, error: p.message ?? "unknown error" }));
        break;
    }
  }

  // Native feel: suppress the webview's right-click menu, except on inputs
  // where copy/paste is expected.
  useEffect(() => {
    const onContextMenu = (e: MouseEvent) => {
      const t = e.target as HTMLElement;
      const editable =
        t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable;
      if (!editable) e.preventDefault();
    };
    document.addEventListener("contextmenu", onContextMenu);
    return () => document.removeEventListener("contextmenu", onContextMenu);
  }, []);

  // Tick the host pairing-PIN countdown; clear when it elapses.
  useEffect(() => {
    if (host.pin === null) return;
    if (host.secondsLeft <= 0) {
      setHost((h) => ({ ...h, pin: null }));
      return;
    }
    const t = setTimeout(
      () => setHost((h) => ({ ...h, secondsLeft: h.secondsLeft - 1 })),
      1000,
    );
    return () => clearTimeout(t);
  }, [host.pin, host.secondsLeft]);

  // --- Client actions --------------------------------------------------------

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

  // --- Host actions ----------------------------------------------------------

  // The supervisor's *explicit* stop path emits no `engine-exited` event (it
  // removes the handle before the stdout reader hits EOF, to avoid a double
  // reap). So we reset host state optimistically here — otherwise the toggle
  // would never flip back to off. stop_engine is reliable (it always tears the
  // engine down), so showing "off" immediately is correct.
  function stopSharing() {
    stopHost().catch((e) => console.warn(e));
    setHost(initialHostState);
  }

  function onStopSharing() {
    if (host.peer) {
      setConfirm({
        title: "Stop sharing?",
        message: `${host.peer} is connected and will be disconnected.`,
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

  function onRevoke(peer: PairedPeer) {
    setConfirm({
      title: `Revoke ${peer.label}?`,
      message: "It won't be able to connect until you pair it again.",
      confirmLabel: "Revoke",
      danger: true,
      onConfirm: () => {
        setConfirm(null);
        revokePeer(peer.fingerprint).catch((e) => console.warn(e));
      },
    });
  }

  // --- Misc ------------------------------------------------------------------

  function copy(text: string) {
    navigator.clipboard?.writeText(text).catch((e) => console.warn("copy failed", e));
  }

  function openAdd(prefill?: string) {
    setAddPrefill(prefill);
    // Clear a stale Add-sheet error from a prior attempt.
    if (clientRef.current.kind === "error") setClient({ kind: "idle" });
    setSheet("add");
  }

  const addStatus: "idle" | "connecting" | "error" =
    client.kind === "connecting" && client.via === "add"
      ? "connecting"
      : client.kind === "error" && client.via === "add"
        ? "error"
        : "idle";
  const addError = client.kind === "error" && client.via === "add" ? client.message : undefined;

  return (
    <main className="container">
      <UpdateBanner />
      <ConnectView
        hosts={hosts}
        client={client}
        rowErrors={rowErrors}
        onConnect={onConnect}
        onDisconnect={() => {
          // Like stopSharing(): the explicit stop emits no engine-exited, so
          // reset optimistically. The window is already visible (the user
          // clicked Disconnect in it), so there's nothing to restore.
          disconnectClient().catch((e) => console.warn(e));
          windowHidden.current = false;
          setClient({ kind: "idle" });
        }}
        onRename={onRename}
        onForget={onForget}
        onCopyAddress={copy}
        onPairAgain={(addr) => openAdd(addr)}
        onAdd={() => openAdd()}
        onOpenSharing={() => {
          setSheet("sharing");
          if (host.running) listPeers().catch((e) => console.warn(e));
        }}
      />

      {sheet === "add" && (
        <AddComputerSheet
          prefillAddr={addPrefill}
          status={addStatus}
          errorMessage={addError}
          onSubmit={(addr, pin, label) => beginConnect(addr, "add", pin, label || null)}
          onClose={() => setSheet("none")}
        />
      )}

      {sheet === "sharing" && (
        <SharingSheet
          host={host}
          onStart={() => startHost().catch((e) => console.warn(e))}
          onStop={onStopSharing}
          onAddDevice={(label) => startPairing(label).catch((e) => console.warn(e))}
          onRevoke={onRevoke}
          onCopy={copy}
          onClose={() => setSheet("none")}
        />
      )}

      {confirm && (
        <ConfirmDialog
          title={confirm.title}
          message={confirm.message}
          confirmLabel={confirm.confirmLabel}
          danger={confirm.danger}
          onConfirm={confirm.onConfirm}
          onCancel={() => setConfirm(null)}
        />
      )}
    </main>
  );
}

// Asks the backend for an available update on mount; if there is one, shows a
// banner whose button downloads + installs the signed bundle and restarts into
// it — so a successful install never returns here.
function UpdateBanner() {
  const [version, setVersion] = useState<string | null>(null);
  const [installing, setInstalling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkForUpdates()
      .then(setVersion)
      .catch((e) => console.warn("update check failed", e));
  }, []);

  if (!version) return null;

  async function install() {
    setError(null);
    setInstalling(true);
    try {
      await installUpdate();
    } catch (e) {
      setError(String(e));
      setInstalling(false);
    }
  }

  return (
    <div className="update-banner">
      <span>
        Update <strong>{version}</strong> available
      </span>
      <button className="btn sm" onClick={install} disabled={installing}>
        {installing ? "Installing…" : "Install & restart"}
      </button>
      {error && <span className="form-error">{error}</span>}
    </div>
  );
}

export default App;
