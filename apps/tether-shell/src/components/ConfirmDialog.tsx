import { useEffect, useRef } from "react";

// A small centered confirm dialog. We roll our own rather than use
// window.confirm(), which renders the browser's chrome and instantly reads as
// "this is a webpage." Esc cancels, Enter confirms, scrim-click cancels.
export function ConfirmDialog({
  title,
  message,
  confirmLabel,
  danger,
  onConfirm,
  onCancel,
}: {
  title: string;
  message: string;
  confirmLabel: string;
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const confirmRef = useRef<HTMLButtonElement>(null);
  // Refs so the focus-and-listen effect runs once; an inline onCancel/onConfirm
  // would re-run it every parent render (e.g. a PIN countdown ticking) and yank
  // focus back to the confirm button each time.
  const handlers = useRef({ onConfirm, onCancel });
  handlers.current = { onConfirm, onCancel };

  useEffect(() => {
    confirmRef.current?.focus();
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") handlers.current.onCancel();
      if (e.key === "Enter") handlers.current.onConfirm();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="scrim center" onMouseDown={onCancel}>
      <div
        className="dialog"
        role="alertdialog"
        aria-modal="true"
        aria-label={title}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h3>{title}</h3>
        <p>{message}</p>
        <div className="form-actions">
          <button className="btn ghost" onClick={onCancel}>
            Cancel
          </button>
          <button
            ref={confirmRef}
            className={"btn " + (danger ? "danger" : "primary")}
            onClick={onConfirm}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
