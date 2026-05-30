import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

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
    | "error";
  addr?: string;
  fingerprint?: string;
  peer?: string;
  host?: string;
  profile?: string;
  reason?: string;
  message?: string;
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

  useEffect(() => {
    const unstatus = listen<StatusEvent>("engine-status", ({ payload }) => {
      if (payload.role !== "host") return;
      switch (payload.event) {
        case "listening":
          setAddr(payload.addr ?? "");
          setFingerprint(payload.fingerprint ?? "");
          setError(null);
          break;
        case "peer_connected":
          setPeer(payload.peer ?? "a client");
          break;
        case "peer_disconnected":
          setPeer(null);
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
    });
    return () => {
      unstatus.then((f) => f());
      unexit.then((f) => f());
    };
  }, []);

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
  const [fingerprint, setFingerprint] = useState("");
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
      await invoke("connect_client", { addr, fingerprint });
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
        Fingerprint
        <input
          className="mono"
          placeholder="64 hex chars"
          value={fingerprint}
          disabled={busy}
          onChange={(e) => setFingerprint(e.currentTarget.value)}
        />
      </label>
      {!busy ? (
        <button onClick={connect} disabled={!addr || !fingerprint}>
          Connect
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
