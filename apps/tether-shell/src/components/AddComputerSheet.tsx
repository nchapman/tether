import { useState } from "react";
import { Sheet } from "./Sheet";
import { PinInput } from "./PinInput";
import { ArrowRightIcon } from "../icons";
import { DEFAULT_PORT, normalizeSocketAddr } from "../socketAddress";

// First-contact pairing. The highest-stakes, rarest action, so it gets its own
// focused surface. The copy is deliberately directional — the #1 confusion is
// *which* machine generates the PIN (it's the other one).
export function AddComputerSheet({
  prefillAddr,
  status,
  errorMessage,
  onSubmit,
  onClose,
}: {
  prefillAddr?: string;
  status: "idle" | "connecting" | "error";
  errorMessage?: string;
  onSubmit: (addr: string, pin: string, label: string) => void;
  onClose: () => void;
}) {
  const [addr, setAddr] = useState(prefillAddr ?? "");
  const [pin, setPin] = useState("");
  const [label, setLabel] = useState("");

  const busy = status === "connecting";
  const normalizedAddr = normalizeSocketAddr(addr.trim());
  const addrOk = normalizedAddr !== null;
  const pinOk = pin.length === 8;
  const canSubmit = addrOk && pinOk && !busy;

  function submit() {
    if (busy || pin.length !== 8 || normalizedAddr === null) return;
    onSubmit(normalizedAddr, pin, label.trim());
  }

  return (
    <Sheet title="Add a computer" onClose={onClose}>
      <form
        className="form"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <label className="field">
          <span className="field-label">Address</span>
          <input
            className="mono"
            placeholder="192.168.1.10"
            value={addr}
            disabled={busy}
            autoFocus
            onChange={(e) => setAddr(e.currentTarget.value)}
          />
          <span className="field-hint">
            Use an IP address. Port {DEFAULT_PORT} is added automatically.
          </span>
        </label>

        <div className="field">
          <span className="field-label">Pairing PIN</span>
          <PinInput value={pin} onChange={setPin} onComplete={submit} disabled={busy} />
          <span className="field-hint">
            Open Tether on the computer you want to reach, turn on sharing, and
            choose “Pair a device” — enter the PIN it shows here.
          </span>
        </div>

        <label className="field">
          <span className="field-label">
            Name <span className="optional">optional</span>
          </span>
          <input
            placeholder="Office desktop"
            value={label}
            disabled={busy}
            onChange={(e) => setLabel(e.currentTarget.value)}
          />
        </label>

        {status === "error" && errorMessage && (
          <p className="form-error" role="alert">
            {errorMessage}
          </p>
        )}

        <div className="form-actions">
          <button type="button" className="btn ghost" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="submit" className="btn primary" disabled={!canSubmit}>
            {busy ? "Pairing…" : "Pair"}
            {!busy && <ArrowRightIcon size={15} />}
          </button>
        </div>
      </form>
    </Sheet>
  );
}
