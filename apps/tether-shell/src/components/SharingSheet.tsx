import { useState } from "react";
import type { PairedPeer } from "../ipc";
import type { HostState } from "../state";
import { Sheet } from "./Sheet";
import { OverflowMenu } from "./OverflowMenu";
import { CopyIcon } from "../icons";

// Sharing (the host role) as a settings sheet. Reads like a system preference:
// one switch, the address to hand out, and who's paired. The PIN flow handles
// the trust ceremony, so there's nothing else to configure here.
export function SharingSheet({
  host,
  onStart,
  onStop,
  onAddDevice,
  onRevoke,
  onCopy,
  onClose,
}: {
  host: HostState;
  onStart: () => void;
  onStop: () => void;
  onAddDevice: (label: string) => void;
  onRevoke: (peer: PairedPeer) => void;
  onCopy: (text: string) => void;
  onClose: () => void;
}) {
  const [newLabel, setNewLabel] = useState("");

  return (
    <Sheet title="Sharing" onClose={onClose}>
      <div className="sharing">
        <SharingToggle running={host.running} onStart={onStart} onStop={onStop} />

        {host.error && <p className="form-error">{host.error}</p>}

        {host.running && (
          <>
            <HostAddress addr={host.addr} onCopy={onCopy} />
            <HostStatus peer={host.peer} />
            <PairingPanel
              pin={host.pin}
              secondsLeft={host.secondsLeft}
              pendingPeer={host.pendingPeer}
              newLabel={newLabel}
              onLabelChange={setNewLabel}
              onAddDevice={(label) => {
                onAddDevice(label);
                setNewLabel("");
              }}
            />
            <PairedDevices peers={host.peers} onRevoke={onRevoke} />
          </>
        )}
      </div>
    </Sheet>
  );
}

function SharingToggle({
  running,
  onStart,
  onStop,
}: {
  running: boolean;
  onStart: () => void;
  onStop: () => void;
}) {
  return (
    <div className="switch-row">
      <span className="field-label">Allow remote connections</span>
      <button
        role="switch"
        aria-checked={running}
        className={"switch" + (running ? " on" : "")}
        onClick={() => (running ? onStop() : onStart())}
      >
        <span className="switch-knob" />
      </button>
    </div>
  );
}

function HostAddress({
  addr,
  onCopy,
}: {
  addr: string;
  onCopy: (text: string) => void;
}) {
  return (
    <div className="kv">
      <span className="field-label">Address</span>
      <div className="copy-row">
        <span className="mono">{addr}</span>
        <button className="icon-btn" aria-label="Copy address" onClick={() => onCopy(addr)}>
          <CopyIcon />
        </button>
      </div>
    </div>
  );
}

function HostStatus({ peer }: { peer: string | null }) {
  return (
    <div className="status">
      <span className={"dot " + (peer ? "on" : "wait")} />
      {peer ? `${peer} connected` : "Waiting for someone to connect…"}
    </div>
  );
}

function PairingPanel({
  pin,
  secondsLeft,
  pendingPeer,
  newLabel,
  onLabelChange,
  onAddDevice,
}: {
  pin: string | null;
  secondsLeft: number;
  pendingPeer: string | null;
  newLabel: string;
  onLabelChange: (label: string) => void;
  onAddDevice: (label: string) => void;
}) {
  return (
    <div className="pairing">
      {pin ? (
        <PinDisplay pin={pin} secondsLeft={secondsLeft} />
      ) : (
        <AddDeviceControl
          value={newLabel}
          onChange={onLabelChange}
          onAdd={() => onAddDevice(newLabel.trim() || "New device")}
        />
      )}
      {pendingPeer && !pin && (
        <p className="field-hint">
          {pendingPeer} tried to connect but isn't paired. Pair a device, then
          give them the PIN it shows.
        </p>
      )}
    </div>
  );
}

function PinDisplay({ pin, secondsLeft }: { pin: string; secondsLeft: number }) {
  return (
    <div className="pin-display">
      <span className="pin mono">{formatPin(pin)}</span>
      <span className="field-hint">
        Enter this PIN on the other computer — expires in {secondsLeft}s
      </span>
    </div>
  );
}

function AddDeviceControl({
  value,
  onChange,
  onAdd,
}: {
  value: string;
  onChange: (label: string) => void;
  onAdd: () => void;
}) {
  return (
    <div className="add-device">
      <input
        placeholder="Device name (e.g. my laptop)"
        value={value}
        onChange={(e) => onChange(e.currentTarget.value)}
      />
      <button className="btn" onClick={onAdd}>
        Pair a device
      </button>
    </div>
  );
}

function PairedDevices({
  peers,
  onRevoke,
}: {
  peers: PairedPeer[];
  onRevoke: (peer: PairedPeer) => void;
}) {
  if (peers.length === 0) return null;

  return (
    <div className="devices">
      <span className="section-label">Paired devices</span>
      <ul>
        {peers.map((p) => (
          <li key={p.fingerprint} className="device-row">
            <span className="row-name">{p.label}</span>
            <OverflowMenu
              label={`Actions for ${p.label}`}
              actions={[{ label: "Revoke", onSelect: () => onRevoke(p), danger: true }]}
            />
          </li>
        ))}
      </ul>
    </div>
  );
}

function formatPin(pin: string) {
  return pin.length === 8 ? `${pin.slice(0, 4)}-${pin.slice(4)}` : pin;
}
