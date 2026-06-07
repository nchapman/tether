import { useEffect, useRef } from "react";
import { CloseIcon } from "../icons";

// A modal sheet that slides down from the top of the window. Used for both
// "Add a computer" and "Sharing". Native-feel details: Esc closes, focus moves
// into the sheet on open and returns on close, clicking the scrim closes, and
// the panel animates in (respecting prefers-reduced-motion via CSS).
export function Sheet({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  // Hold onClose in a ref so the open/close effect runs exactly once — an
  // inline onClose prop changes identity every parent render (e.g. the host
  // PIN countdown ticking), which would otherwise re-add the key listener and,
  // worse, yank focus back to the first control mid-typing.
  const onCloseRef = useRef(onClose);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    // Focus the first focusable control so keyboard users land inside.
    panelRef.current
      ?.querySelector<HTMLElement>(
        "input, button, [tabindex]:not([tabindex='-1'])",
      )
      ?.focus();

    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.stopPropagation();
        onCloseRef.current();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      previouslyFocused?.focus();
    };
  }, []);

  return (
    <div className="scrim" onMouseDown={onClose}>
      <div
        ref={panelRef}
        className="sheet"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        // Don't let clicks inside the panel reach the scrim's close handler.
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="sheet-head">
          <h2>{title}</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            <CloseIcon />
          </button>
        </header>
        <div className="sheet-body">{children}</div>
      </div>
    </div>
  );
}
