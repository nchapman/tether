import { useState } from "react";
import { Sheet } from "./Sheet";
import { PinInput } from "./PinInput";
import { ArrowRightIcon } from "../icons";

const DEFAULT_PORT = "7654";
const PORT_RE = /^\d{1,5}$/;
const IPV4_PART_RE = /^\d{1,3}$/;
const BRACKETED_IPV6_SOCKET_RE = /^\[([0-9a-fA-F:.]+)\]:(\d{1,5})$/;
const IPV4_SOCKET_RE = /^(.+):(\d{1,5})$/;

// Append the default port unless the address already has one. Handles IPv6:
// a bare literal (`fe80::1`) is bracketed (`[fe80::1]:7654`) so it parses as a
// SocketAddr, and an already-bracketed address keeps its port if present.
function withDefaultPort(addr: string): string {
  if (addr.startsWith("[")) {
    // Bracketed IPv6, with or without a trailing `:port`.
    return addr.includes("]:") ? addr : `${addr}:${DEFAULT_PORT}`;
  }
  let colons = 0;
  for (const char of addr) {
    if (char === ":") colons += 1;
  }
  if (colons === 1) return addr; // host:port or IPv4:port
  if (colons > 1) return `[${addr}]:${DEFAULT_PORT}`; // bare IPv6 literal
  return `${addr}:${DEFAULT_PORT}`; // bare IPv4 or unsupported hostname
}

function validPort(raw: string): boolean {
  if (PORT_RE.exec(raw) === null) return false;
  const port = Number(raw);
  return port > 0 && port <= 65535;
}

function validIpv4(raw: string): boolean {
  const parts = raw.split(".");
  return (
    parts.length === 4 &&
    parts.every((part) => IPV4_PART_RE.exec(part) !== null && Number(part) <= 255)
  );
}

function validBracketedIpv6Socket(raw: string): boolean {
  const match = BRACKETED_IPV6_SOCKET_RE.exec(raw);
  return match !== null && match[1].includes(":") && validPort(match[2]);
}

function normalizeSocketAddr(raw: string): string | null {
  const candidate = withDefaultPort(raw);
  const ipv4 = IPV4_SOCKET_RE.exec(candidate);
  if (ipv4 !== null && validIpv4(ipv4[1]) && validPort(ipv4[2])) {
    return candidate;
  }
  if (validBracketedIpv6Socket(candidate)) {
    return candidate;
  }
  return null;
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
