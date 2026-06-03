import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

import {
  type EngineEvent,
  type EngineExited,
  type SavedHost,
  listKnownHosts,
  listPeers,
  showWindow,
  getPrefs,
} from "./ipc";
import { type Confirm, type SheetName } from "./state";
import { useClientSession } from "./hooks/useClientSession";
import { useHostSharing } from "./hooks/useHostSharing";
import { ConnectView } from "./components/ConnectView";
import { AddComputerSheet } from "./components/AddComputerSheet";
import { SharingSheet } from "./components/SharingSheet";
import { ConfirmDialog } from "./components/ConfirmDialog";
import { UpdateBanner } from "./components/UpdateBanner";

// The control-plane shell: an address book of saved computers (the client
// role) plus a sharing-settings sheet (the host role). App owns the shared
// pieces — the address-book data, window-visibility lifecycle, and the modal
// shell (sheets + confirms) — and wires the engine events to two state-machine
// hooks (`useClientSession`, `useHostSharing`). No video renders here.
function App() {
  const [hosts, setHosts] = useState<SavedHost[]>([]);
  const [sheet, setSheet] = useState<SheetName>("none");
  const [addPrefill, setAddPrefill] = useState<string | undefined>(undefined);
  const [confirm, setConfirm] = useState<Confirm | null>(null);

  // The address book mirrored into a ref so the once-registered engine
  // listeners (and label lookups inside the hooks) read the current list.
  const hostsRef = useRef(hosts);
  hostsRef.current = hosts;
  // Tracks whether we hid the window for a live session, so we restore it on
  // *any* terminal transition (clean disconnect, error, or crash-exit) without
  // re-showing — and stealing focus — when it was never hidden.
  const windowHidden = useRef(false);
  // Guards the launch-time sharing auto-restore so it fires exactly once.
  // React StrictMode (dev) double-invokes the mount effect setup→cleanup→setup,
  // and the cleanup can't cancel the already-dispatched `getPrefs()` chain, so
  // without this guard both invocations call `startSharing()` and we spawn the
  // host engine twice. The ref persists across the StrictMode re-invoke (same
  // fiber), so the second chain sees it already set and skips.
  const didAutoRestore = useRef(false);

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

  const session = useClientSession({
    windowHidden,
    restoreWindow,
    refreshHosts,
    labelFor,
    setSheet,
    setConfirm,
  });
  const sharing = useHostSharing({ setConfirm });

  // Open the Sharing sheet and refresh the paired-device list if hosting is
  // live. Shared by the gear button and the tray's "Pair a device…" trigger.
  // NB: the tray path captures this once (first render) in the "open-sharing"
  // listener below, so it must only ever read through stable refs/setters —
  // `sharing.hostRef.current` is live; a direct `sharing.host` read would go
  // stale.
  function openSharing() {
    setSheet("sharing");
    if (sharing.hostRef.current.running) listPeers().catch((e) => console.warn(e));
  }

  function openAdd(prefill?: string) {
    setAddPrefill(prefill);
    session.clearAddError();
    setSheet("add");
  }

  function copy(text: string) {
    navigator.clipboard?.writeText(text).catch((e) => console.warn("copy failed", e));
  }

  // --- Engine event wiring ---------------------------------------------------

  useEffect(() => {
    refreshHosts();

    const unstatus = listen<EngineEvent>("engine-status", ({ payload }) => {
      if (payload.role === "client") session.handleEvent(payload);
      else sharing.handleEvent(payload);
    });
    const unexit = listen<EngineExited>("engine-exited", ({ payload }) => {
      if (payload.role === "client") session.handleExit();
      else sharing.handleExit();
    });
    // The tray's "Pair a device…" shows the window and asks us to open Sharing.
    const unsharing = listen("open-sharing", () => openSharing());
    // The tray asks us to connect through the window (only when a session is
    // already live, so the switch-computer confirm runs).
    const unconnect = listen<string>("request-connect", ({ payload }) =>
      session.onConnect(payload),
    );

    // Re-apply the persisted sharing posture: if hosting was last left on,
    // start it again. Wait until both listeners are live so we don't miss the
    // host's `listening` event (the toggle would otherwise show off while the
    // engine is actually up). startSharing re-persists the posture, harmlessly.
    Promise.all([unstatus, unexit])
      .then(() => getPrefs())
      .then((prefs) => {
        // Once-only: the check-and-set is synchronous (no await between), so
        // the StrictMode-doubled chains can't both pass it.
        if (prefs.sharing_enabled && !didAutoRestore.current) {
          didAutoRestore.current = true;
          sharing.startSharing();
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

  return (
    <main className="container">
      <UpdateBanner />
      <ConnectView
        hosts={hosts}
        client={session.client}
        rowErrors={session.rowErrors}
        onConnect={session.onConnect}
        onDisconnect={session.onDisconnect}
        onRename={session.onRename}
        onForget={session.onForget}
        onCopyAddress={copy}
        onPairAgain={(addr) => openAdd(addr)}
        onAdd={() => openAdd()}
        onOpenSharing={openSharing}
      />

      {sheet === "add" && (
        <AddComputerSheet
          prefillAddr={addPrefill}
          status={session.addStatus}
          errorMessage={session.addError}
          onSubmit={(addr, pin, label) => session.beginConnect(addr, "add", pin, label || null)}
          onClose={() => setSheet("none")}
        />
      )}

      {sheet === "sharing" && (
        <SharingSheet
          host={sharing.host}
          onStart={sharing.startSharing}
          onStop={sharing.onStopSharing}
          onAddDevice={sharing.onAddDevice}
          onRevoke={sharing.onRevoke}
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

export default App;
