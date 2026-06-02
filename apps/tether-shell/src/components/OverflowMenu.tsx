import { useEffect, useRef, useState } from "react";
import { MoreIcon } from "../icons";

export type MenuAction = {
  label: string;
  onSelect: () => void;
  danger?: boolean;
};

// A quiet "⋯" button that opens a small popover of actions. A separate target
// from the row's primary click, so Rename/Forget never mis-fire a connection.
// Closes on outside-click, Escape, or after selecting an item.
export function OverflowMenu({ actions, label = "More actions" }: {
  actions: MenuAction[];
  label?: string;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="overflow" ref={rootRef}>
      <button
        className="icon-btn"
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
      >
        <MoreIcon />
      </button>
      {open && (
        <div className="menu" role="menu">
          {actions.map((a) => (
            <button
              key={a.label}
              role="menuitem"
              className={"menu-item" + (a.danger ? " danger" : "")}
              onClick={(e) => {
                e.stopPropagation();
                setOpen(false);
                a.onSelect();
              }}
            >
              {a.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
