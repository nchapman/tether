import { useState } from "react";
import { Sheet } from "./Sheet";
import { PinInput } from "./PinInput";
import { ArrowRightIcon } from "../icons";

const DEFAULT_PORT = "7654";

// Append the default port unless the address already has one. Handles IPv6:
// a bare literal (`fe80::1`) is bracketed (`[fe80::1]:7654`) so it parses as a
// SocketAddr, and an already-bracketed address keeps its port if present.
function withDefaultPort(addr: string): string {
  if (addr.startsWith("[")) {
    // Bracketed IPv6, with or without a trailing `:port`.
    return addr.includes("]:") ? addr : `${addr}:${DEFAULT_PORT}`;
  }
  const colons = (addr.match(/:/g) ?? []).length;
  if (colons === 1) return addr; // host:port or IPv4:port
  if (colons > 1) return `[${addr}]:${DEFAULT_PORT}`; // bare IPv6 literal
  return `${addr}:${DEFAULT_PORT}`; // bare host / IPv4
}

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
  const addrOk = addr.trim().length > 0;
  const pinOk = pin.length === 8;
  const canSubmit = addrOk && pinOk && !busy;

  function submit() {
    if (!canSubmit) return;
    onSubmit(withDefaultPort(addr.trim()), pin, label.trim());
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
          <span className="field-hint">Port {DEFAULT_PORT} is added automatically.</span>
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
