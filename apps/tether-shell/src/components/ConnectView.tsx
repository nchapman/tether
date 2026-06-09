import { useState } from "react";
import type { SavedHost } from "../ipc";
import type { ClientState, RowError } from "../state";
import { APP_TITLE } from "../appChannel";
import { PlusIcon, SettingsIcon, MonitorIcon } from "../icons";
import { HostRow } from "./HostRow";

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
  onPairAgain,
  onAdd,
  onOpenSharing,
}: {
  hosts: SavedHost[];
  client: ClientState;
  rowErrors: Record<string, RowError>;
  onConnect: (addr: string) => void;
  onDisconnect: () => void;
  onRename: (addr: string, label: string) => void;
  onForget: (host: SavedHost) => void;
  onCopyAddress: (addr: string) => void;
  onPairAgain: (addr: string) => void;
  onAdd: () => void;
  onOpenSharing: () => void;
}) {
  const [editingAddr, setEditingAddr] = useState<string | null>(null);

  // ↑/↓ move focus between rows (native list nav); Enter activates the focused
  // row's button. Disabled rows are skipped.
  function onListKeyDown(e: React.KeyboardEvent<HTMLUListElement>) {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    const items = Array.from(
      e.currentTarget.querySelectorAll<HTMLButtonElement>("button.host-main:not(:disabled)"),
    );
    if (items.length === 0) return;
    e.preventDefault();
    const active = document.activeElement as HTMLElement | null;
    let current = items.indexOf(active as HTMLButtonElement);
    // Focus may be on a secondary button inside a row (e.g. "Pair again");
    // anchor navigation to that row rather than jumping to an endpoint.
    if (current === -1 && active) {
      const row = active.closest("li.host-row");
      if (row) current = items.findIndex((btn) => row.contains(btn));
    }
    let next: number;
    if (current === -1) next = e.key === "ArrowDown" ? 0 : items.length - 1;
    else if (e.key === "ArrowDown") next = Math.min(current + 1, items.length - 1);
    else next = Math.max(current - 1, 0);
    items[next].focus();
  }

  const connectingAddr =
    client.kind === "connecting" ? client.addr : null;
  const connectedAddr = client.kind === "connected" ? client.addr : null;
  const busy = client.kind === "connecting" || client.kind === "connected";

  return (
    <div className="connect-view">
      <header className="titlebar">
        <h1>{APP_TITLE}</h1>
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
        <EmptyHosts onAdd={onAdd} />
      ) : (
        <ul className="host-list" onKeyDown={onListKeyDown}>
          {hosts.map((host) => (
            <HostRow
              key={host.addr}
              host={host}
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
              onPairAgain={() => onPairAgain(host.addr)}
            />
          ))}
        </ul>
      )}

      {client.kind === "connected" && (
        <SessionBar
          profile={client.profile}
          hostLabel={hostLabel(hosts, client.addr)}
          onDisconnect={onDisconnect}
        />
      )}
    </div>
  );
}

function EmptyHosts({ onAdd }: { onAdd: () => void }) {
  return (
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
  );
}

function SessionBar({
  profile,
  hostLabel,
  onDisconnect,
}: {
  profile: string;
  hostLabel: string;
  onDisconnect: () => void;
}) {
  return (
    <div className="session-bar">
      <span className="dot on" />
      <span className="session-text">
        Streaming{profile ? ` ${profile}` : ""} · {hostLabel}
      </span>
      <button className="btn ghost sm" onClick={onDisconnect}>
        Disconnect
      </button>
    </div>
  );
}

function hostLabel(hosts: SavedHost[], addr: string): string {
  return hosts.find((h) => h.addr === addr)?.label ?? addr;
}
