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
        <div className="switch-row">
          <span className="field-label">Allow remote connections</span>
          <button
            role="switch"
            aria-checked={host.running}
            className={"switch" + (host.running ? " on" : "")}
            onClick={() => (host.running ? onStop() : onStart())}
          >
            <span className="switch-knob" />
          </button>
        </div>

        {host.error && <p className="form-error">{host.error}</p>}

        {host.running && (
          <>
            <div className="kv">
              <span className="field-label">Address</span>
              <div className="copy-row">
                <span className="mono">{host.addr}</span>
                <button
                  className="icon-btn"
                  aria-label="Copy address"
                  onClick={() => onCopy(host.addr)}
                >
                  <CopyIcon />
                </button>
              </div>
            </div>

            <div className="status">
              <span className={"dot " + (host.peer ? "on" : "wait")} />
              {host.peer ? `${host.peer} connected` : "Waiting for someone to connect…"}
            </div>

            <div className="pairing">
              {host.pin ? (
                <div className="pin-box">
                  <span className="pin mono">{host.pin}</span>
                  <span className="field-hint">
                    Enter this PIN on the new computer — expires in {host.secondsLeft}s
                  </span>
                </div>
              ) : (
                <div className="add-device">
                  <input
                    placeholder="Device name (e.g. my laptop)"
                    value={newLabel}
                    onChange={(e) => setNewLabel(e.currentTarget.value)}
                  />
                  <button
                    className="btn"
                    onClick={() => {
                      onAddDevice(newLabel.trim() || "New device");
                      setNewLabel("");
                    }}
                  >
                    Add a device
                  </button>
                </div>
              )}
              {host.pendingPeer && !host.pin && (
                <p className="field-hint">
                  {host.pendingPeer} tried to connect but isn't paired. Add a
                  device, then enter the PIN on it.
                </p>
              )}
            </div>

            {host.peers.length > 0 && (
              <div className="devices">
                <span className="section-label">Paired devices</span>
                <ul>
                  {host.peers.map((p) => (
                    <li key={p.fingerprint} className="device-row">
                      <span className="row-name">{p.label}</span>
                      <OverflowMenu
                        label={`Actions for ${p.label}`}
                        actions={[
                          { label: "Revoke", onSelect: () => onRevoke(p), danger: true },
                        ]}
                      />
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </div>
    </Sheet>
  );
}
