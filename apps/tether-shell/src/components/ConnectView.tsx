import { useState } from "react";
import type { SavedHost } from "../ipc";
import { recencyLabel } from "../ipc";
import type { ClientState } from "../state";
import { OverflowMenu } from "./OverflowMenu";
import { PlusIcon, SettingsIcon, MonitorIcon } from "../icons";

// The address book: the Connect window's home screen. Single-click a row's
// name region to reconnect (pinned-cert resume); Rename/Forget/Copy live in a
// separate ⋯ target so they never launch a connection.
export function ConnectView({
  hosts,
  client,
  rowErrors,
  onConnect,
  onDisconnect,
  onRename,
  onForget,
  onCopyAddress,
  onAdd,
  onOpenSharing,
}: {
  hosts: SavedHost[];
  client: ClientState;
  rowErrors: Record<string, string>;
  onConnect: (addr: string) => void;
  onDisconnect: () => void;
  onRename: (addr: string, label: string) => void;
  onForget: (host: SavedHost) => void;
  onCopyAddress: (addr: string) => void;
  onAdd: () => void;
  onOpenSharing: () => void;
}) {
  const [editingAddr, setEditingAddr] = useState<string | null>(null);

  const connectingAddr =
    client.kind === "connecting" ? client.addr : null;
  const connectedAddr = client.kind === "connected" ? client.addr : null;
  const busy = client.kind === "connecting" || client.kind === "connected";

  return (
    <div className="connect-view">
      <header className="titlebar">
        <h1>Tether</h1>
        <div className="titlebar-actions">
          <button className="icon-btn" aria-label="Sharing settings" onClick={onOpenSharing}>
            <SettingsIcon />
          </button>
          <button className="btn primary sm" onClick={onAdd}>
            <PlusIcon size={15} /> Add
          </button>
        </div>
      </header>

      {hosts.length === 0 ? (
        <div className="empty">
          <MonitorIcon size={40} className="empty-glyph" />
          <p className="empty-title">No computers yet.</p>
          <p className="empty-sub">Add one to connect.</p>
          <button className="btn primary" onClick={onAdd}>
            <PlusIcon size={15} /> Add a computer
          </button>
          <p className="empty-hint">
            You'll need its address and a pairing PIN from that computer.
          </p>
        </div>
      ) : (
        <ul className="host-list">
          {hosts.map((host, i) => (
            <HostRow
              key={host.addr}
              host={host}
              index={i}
              editing={editingAddr === host.addr}
              connecting={connectingAddr === host.addr}
              connected={connectedAddr === host.addr}
              disabled={busy && connectingAddr !== host.addr && connectedAddr !== host.addr}
              error={rowErrors[host.addr]}
              onConnect={() => onConnect(host.addr)}
              onStartRename={() => setEditingAddr(host.addr)}
              onCancelRename={() => setEditingAddr(null)}
              onCommitRename={(label) => {
                setEditingAddr(null);
                if (label && label !== host.label) onRename(host.addr, label);
              }}
              onForget={() => onForget(host)}
              onCopyAddress={() => onCopyAddress(host.addr)}
            />
          ))}
        </ul>
      )}

      {client.kind === "connected" && (
        <div className="session-bar">
          <span className="dot on" />
          <span className="session-text">
            Streaming{client.profile ? ` ${client.profile}` : ""} ·{" "}
            {hostLabel(hosts, client.addr)}
          </span>
          <button className="btn ghost sm" onClick={onDisconnect}>
            Disconnect
          </button>
        </div>
      )}
    </div>
  );
}

function hostLabel(hosts: SavedHost[], addr: string): string {
  return hosts.find((h) => h.addr === addr)?.label ?? addr;
}

function HostRow({
  host,
  index,
  editing,
  connecting,
  connected,
  disabled,
  error,
  onConnect,
  onStartRename,
  onCancelRename,
  onCommitRename,
  onForget,
  onCopyAddress,
}: {
  host: SavedHost;
  index: number;
  editing: boolean;
  connecting: boolean;
  connected: boolean;
  disabled: boolean;
  error?: string;
  onConnect: () => void;
  onStartRename: () => void;
  onCancelRename: () => void;
  onCommitRename: (label: string) => void;
  onForget: () => void;
  onCopyAddress: () => void;
}) {
  // Seeded once from the prop; rows are keyed by addr so this instance is
  // stable across refreshes. (If a background refresh while editing is ever
  // added, reset the draft when host.label changes.)
  const [draft, setDraft] = useState(host.label);

  if (editing) {
    return (
      <li className="host-row editing">
        <input
          className="rename-input"
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") onCommitRename(draft.trim());
            if (e.key === "Escape") onCancelRename();
          }}
          onBlur={() => onCommitRename(draft.trim())}
        />
      </li>
    );
  }

  return (
    <li className={"host-row" + (connected ? " active" : "")}>
      <button
        className="host-main"
        disabled={disabled}
        onClick={onConnect}
        title={`Connect to ${host.label}`}
      >
        <span className={"row-dot" + (connecting ? " connecting" : connected ? " on" : "")} />
        <span className="row-text">
          <span className="row-name">{host.label}</span>
          <span className="row-sub mono">
            {host.addr} · {connecting ? "connecting…" : recencyLabel(host)}
          </span>
          {error && <span className="row-error">{error}</span>}
        </span>
        {index < 9 && <kbd className="row-kbd">⌘{index + 1}</kbd>}
      </button>
      <OverflowMenu
        label={`Actions for ${host.label}`}
        actions={[
          { label: "Rename", onSelect: onStartRename },
          { label: "Copy address", onSelect: onCopyAddress },
          { label: "Forget", onSelect: onForget, danger: true },
        ]}
      />
    </li>
  );
}
