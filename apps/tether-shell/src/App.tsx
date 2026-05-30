import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

// One paired device, mirroring tether_ipc::PairedPeer.
type PairedPeer = {
  fingerprint: string;
  label: string;
  paired_at_unix: number;
};

// Mirrors tether_ipc::EngineEvent (flattened) plus the `role` the
// supervisor tags each line with. Only the fields the UI reads are typed.
type StatusEvent = {
  role: "host" | "client";
  event:
    | "listening"
    | "peer_connected"
    | "peer_disconnected"
    | "connecting"
    | "connected"
    | "disconnected"
    | "error"
    | "pairing_pin"
    | "paired"
    | "pairing_required"
    | "peer_list";
  addr?: string;
  fingerprint?: string;
  peer?: string;
  host?: string;
  profile?: string;
  reason?: string;
  message?: string;
  pin?: string;
  expires_in_secs?: number;
  label?: string;
  peers?: PairedPeer[];
};

type ExitedEvent = { role: "host" | "client" };

function HostPanel() {
  const [running, setRunning] = useState(false);
  // Default off: real screen capture negotiates HEVC, which works on the
  // verified Windows/QSV path. Test pattern forces the H.264 floor, whose
  // only Windows encoder (h264_mf) fails the self-decodable-IDR check — a
  // black window. Keep the toggle for cross-platform dev.
  const [testPattern, setTestPattern] = useState(false);
  const [addr, setAddr] = useState("");
  const [fingerprint, setFingerprint] = useState("");
  const [peer, setPeer] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Pairing window: the PIN to read out and a live countdown.
  const [newLabel, setNewLabel] = useState("");
  const [pin, setPin] = useState<string | null>(null);
  const [secondsLeft, setSecondsLeft] = useState(0);
  // A device tried to connect while no window was open.
  const [pendingPeer, setPendingPeer] = useState<string | null>(null);
  const [peers, setPeers] = useState<PairedPeer[]>([]);

  useEffect(() => {
    const unstatus = listen<StatusEvent>("engine-status", ({ payload }) => {
      if (payload.role !== "host") return;
      switch (payload.event) {
        case "listening":
          setAddr(payload.addr ?? "");
          setFingerprint(payload.fingerprint ?? "");
          setError(null);
          // Populate the paired-devices list once we're hosting. Best-effort:
          // the host also pushes peer_list after every pair/revoke.
          invoke("list_peers").catch((e) => console.warn("list_peers failed", e));
          break;
        case "peer_connected":
          setPeer(payload.peer ?? "a client");
          break;
        case "peer_disconnected":
          setPeer(null);
          break;
        case "pairing_pin":
          setPin(payload.pin ?? null);
          setSecondsLeft(payload.expires_in_secs ?? 0);
          break;
        case "paired":
          // Window consumed; clear the PIN. A peer_list refresh follows.
          setPin(null);
          setPendingPeer(null);
          break;
        case "pairing_required":
          setPendingPeer(payload.peer ?? "a device");
          break;
        case "peer_list":
          setPeers(payload.peers ?? []);
          break;
        case "error":
          // A host error (e.g. bind failure) means it isn't hosting; the
          // engine exits right after, but reset now so the UI doesn't
          // strand the "Stop hosting" button alongside the error.
          setRunning(false);
          setError(payload.message ?? "unknown error");
          break;
      }
    });
    const unexit = listen<ExitedEvent>("engine-exited", ({ payload }) => {
      if (payload.role !== "host") return;
      setRunning(false);
      setAddr("");
      setFingerprint("");
      setPeer(null);
      setPin(null);
      setPendingPeer(null);
      setPeers([]);
    });
    return () => {
      unstatus.then((f) => f());
      unexit.then((f) => f());
    };
  }, []);

  // Tick the PIN countdown once a second; clear the PIN when it elapses
  // (the host closed the window on its side too).
  useEffect(() => {
    if (pin === null) return;
    if (secondsLeft <= 0) {
      setPin(null);
      return;
    }
    const t = setTimeout(() => setSecondsLeft((s) => s - 1), 1000);
    return () => clearTimeout(t);
  }, [pin, secondsLeft]);

  async function start() {
    setError(null);
    try {
      await invoke("start_host", { testPattern });
      setRunning(true);
    } catch (e) {
      setError(String(e));
    }
  }

  async function stop() {
    await invoke("stop_engine", { role: "host" });
    setRunning(false);
  }

  async function addDevice() {
    setError(null);
    try {
      await invoke("start_pairing", { label: newLabel || "New device" });
      setNewLabel("");
    } catch (e) {
      setError(String(e));
    }
  }

  async function revoke(fp: string) {
    try {
      await invoke("revoke_peer", { fingerprint: fp });
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <section className="panel">
      <h2>Host this machine</h2>
      {!running ? (
        <>
          <label className="check">
            <input
              type="checkbox"
              checked={testPattern}
              onChange={(e) => setTestPattern(e.currentTarget.checked)}
            />
            Test pattern (no screen capture)
          </label>
          <button onClick={start}>Start hosting</button>
        </>
      ) : (
        <>
          <div className="status">
            <span className={peer ? "dot on" : "dot wait"} />
            {peer ? `Client connected: ${peer}` : "Waiting for a client…"}
          </div>
          {addr && (
            <dl className="kv">
              <dt>Address</dt>
              <dd className="mono">{addr}</dd>
              <dt>Fingerprint</dt>
              <dd className="mono break">{fingerprint}</dd>
            </dl>
          )}

          <div className="pairing">
            <h3>Add a device</h3>
            {pin ? (
              <div className="pin-box">
                <span className="pin">{pin}</span>
                <span className="pin-hint">
                  Enter this PIN on the new device — expires in {secondsLeft}s
                </span>
              </div>
            ) : (
              <div className="add-device">
                <input
                  placeholder="Device name (e.g. my laptop)"
                  value={newLabel}
                  onChange={(e) => setNewLabel(e.currentTarget.value)}
                />
                <button onClick={addDevice}>Add a device</button>
              </div>
            )}
            {pendingPeer && !pin && (
              <p className="hint">
                {pendingPeer} tried to connect but isn’t paired. Click “Add a
                device”, then enter the PIN on it.
              </p>
            )}
          </div>

          {peers.length > 0 && (
            <div className="devices">
              <h3>Paired devices</h3>
              <ul>
                {peers.map((p) => (
                  <li key={p.fingerprint}>
                    <span className="device-label">{p.label}</span>
                    <span className="mono break device-fp">{p.fingerprint}</span>
                    <button className="link danger" onClick={() => revoke(p.fingerprint)}>
                      Revoke
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <button className="secondary" onClick={stop}>
            Stop hosting
          </button>
        </>
      )}
      {error && <p className="error">{error}</p>}
    </section>
  );
}

function ClientPanel() {
  const [addr, setAddr] = useState("");
  const [pin, setPin] = useState("");
  const [label, setLabel] = useState("");
  const [state, setState] = useState<
    "idle" | "connecting" | "connected" | "error"
  >("idle");
  const [detail, setDetail] = useState<string | null>(null);

  useEffect(() => {
    const unstatus = listen<StatusEvent>("engine-status", ({ payload }) => {
      if (payload.role !== "client") return;
      switch (payload.event) {
        case "connecting":
          setState("connecting");
          setDetail(`Connecting to ${payload.host}…`);
          break;
        case "connected":
          setState("connected");
          setDetail(`Streaming ${payload.profile ?? ""}`.trim());
          break;
        case "disconnected":
          setState("idle");
          setDetail(payload.reason ? `Disconnected: ${payload.reason}` : null);
          break;
        case "error":
          setState("error");
          setDetail(payload.message ?? "unknown error");
          break;
      }
    });
    const unexit = listen<ExitedEvent>("engine-exited", ({ payload }) => {
      if (payload.role !== "client") return;
      setState((s) => (s === "error" ? s : "idle"));
    });
    return () => {
      unstatus.then((f) => f());
      unexit.then((f) => f());
    };
    // Register once on mount: re-running on every `addr` keystroke would
    // churn the listeners and (under StrictMode) double-fire events.
  }, []);

  async function connect() {
    setDetail(null);
    try {
      // A PIN means first-contact pairing; without one the client reconnects
      // using the host fingerprint it pinned last time (known-hosts).
      await invoke("connect_client", {
        addr,
        pin: pin.trim() || null,
        label: label.trim() || null,
      });
    } catch (e) {
      setState("error");
      setDetail(String(e));
    }
  }

  async function disconnect() {
    await invoke("stop_engine", { role: "client" });
    setState("idle");
  }

  const busy = state === "connecting" || state === "connected";

  return (
    <section className="panel">
      <h2>Connect to a host</h2>
      <label>
        Address
        <input
          className="mono"
          placeholder="127.0.0.1:7654"
          value={addr}
          disabled={busy}
          onChange={(e) => setAddr(e.currentTarget.value)}
        />
      </label>
      <label>
        PIN <span className="optional">(only for a new host)</span>
        <input
          className="mono"
          placeholder="leave blank to reconnect"
          value={pin}
          disabled={busy}
          onChange={(e) => setPin(e.currentTarget.value)}
        />
      </label>
      {pin.trim() && (
        <label>
          Name for this host <span className="optional">(optional)</span>
          <input
            placeholder="e.g. office desktop"
            value={label}
            disabled={busy}
            onChange={(e) => setLabel(e.currentTarget.value)}
          />
        </label>
      )}
      {!busy ? (
        <button onClick={connect} disabled={!addr}>
          {pin.trim() ? "Pair & connect" : "Connect"}
        </button>
      ) : (
        <button className="secondary" onClick={disconnect}>
          Disconnect
        </button>
      )}
      {detail && (
        <div className="status">
          <span
            className={
              "dot " +
              (state === "connected"
                ? "on"
                : state === "error"
                  ? "err"
                  : "wait")
            }
          />
          {detail}
        </div>
      )}
    </section>
  );
}

function App() {
  return (
    <main className="container">
      <h1>Tether</h1>
      <HostPanel />
      <ClientPanel />
    </main>
  );
}

export default App;
