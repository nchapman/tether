import { useState } from "react";

import type { SavedHost } from "../ipc";
import { recencyLabel } from "../ipc";
import type { RowError } from "../state";
import { OverflowMenu } from "./OverflowMenu";

export type HostRowProps = {
  host: SavedHost;
  editing: boolean;
  connecting: boolean;
  connected: boolean;
  disabled: boolean;
  error?: RowError;
  onConnect: () => void;
  onStartRename: () => void;
  onCancelRename: () => void;
  onCommitRename: (label: string) => void;
  onForget: () => void;
  onCopyAddress: () => void;
  onPairAgain: () => void;
};

export function HostRow({
  host,
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
  onPairAgain,
}: HostRowProps) {
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
      <div className="host-row-line">
        <button
          className="host-main"
          disabled={disabled}
          onClick={onConnect}
          title={`Connect to ${host.label}`}
        >
          <span className={"row-dot" + rowDotStatus(connecting, connected)} />
          <span className="row-text">
            <span className="row-name">{host.label}</span>
            <span className="row-sub mono">
              {host.addr} · {connecting ? "connecting…" : recencyLabel(host)}
            </span>
          </span>
        </button>
        <OverflowMenu
          label={`Actions for ${host.label}`}
          actions={[
            { label: "Rename", onSelect: onStartRename },
            { label: "Copy address", onSelect: onCopyAddress },
            { label: "Forget", onSelect: onForget, danger: true },
          ]}
        />
      </div>
      {error && (
        <p className="row-error">
          <span>{error.message}</span>
          {error.pairAgain && (
            <button className="link-btn" onClick={onPairAgain}>
              Pair again
            </button>
          )}
        </p>
      )}
    </li>
  );
}

function rowDotStatus(connecting: boolean, connected: boolean) {
  if (connecting) return " connecting";
  if (connected) return " on";
  return "";
}
