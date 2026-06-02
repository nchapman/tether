import { useRef } from "react";

// Segmented PIN entry: one box per digit, OTP-style. Signals "one-time code"
// the way a flat text field can't, and the auto-advance / paste-distribute
// behavior makes typing an 8-digit code painless. `value` is the joined
// string; the parent owns it.
//
// PIN_DIGITS is 8 (see tether_pairing). Only digits are accepted.
const DIGITS = 8;

export function PinInput({
  value,
  onChange,
  onComplete,
  disabled,
}: {
  value: string;
  onChange: (pin: string) => void;
  onComplete?: () => void;
  disabled?: boolean;
}) {
  const refs = useRef<(HTMLInputElement | null)[]>([]);

  // Rebuild the whole string from one box's edit, then refocus appropriately.
  function setDigit(index: number, digit: string) {
    const chars = value.padEnd(DIGITS, " ").split("");
    chars[index] = digit || " ";
    const next = chars.join("").replace(/ /g, "");
    onChange(next);
    if (digit && index < DIGITS - 1) refs.current[index + 1]?.focus();
    if (next.length === DIGITS) onComplete?.();
  }

  function onKeyDown(index: number, e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Backspace" && !e.currentTarget.value && index > 0) {
      refs.current[index - 1]?.focus();
    } else if (e.key === "ArrowLeft" && index > 0) {
      refs.current[index - 1]?.focus();
    } else if (e.key === "ArrowRight" && index < DIGITS - 1) {
      refs.current[index + 1]?.focus();
    }
  }

  // Paste distributes across the boxes from wherever it lands.
  function onPaste(index: number, e: React.ClipboardEvent) {
    e.preventDefault();
    const pasted = e.clipboardData.getData("text").replace(/\D/g, "");
    if (!pasted) return;
    const chars = value.padEnd(DIGITS, " ").split("");
    for (let i = 0; i < pasted.length && index + i < DIGITS; i++) {
      chars[index + i] = pasted[i];
    }
    const next = chars.join("").replace(/ /g, "");
    onChange(next);
    const landed = Math.min(index + pasted.length, DIGITS - 1);
    refs.current[landed]?.focus();
    if (next.length === DIGITS) onComplete?.();
  }

  return (
    <div className="pin-input" role="group" aria-label="Pairing PIN">
      {Array.from({ length: DIGITS }, (_, i) => (
        <input
          key={i}
          ref={(el) => {
            refs.current[i] = el;
          }}
          className="pin-box mono"
          inputMode="numeric"
          maxLength={1}
          disabled={disabled}
          value={value[i] ?? ""}
          aria-label={`Digit ${i + 1}`}
          onChange={(e) => setDigit(i, e.currentTarget.value.replace(/\D/g, "").slice(-1))}
          onKeyDown={(e) => onKeyDown(i, e)}
          onPaste={(e) => onPaste(i, e)}
          // A separator after the 4th digit reads as two groups of four.
          data-group={i === 4 ? "second" : undefined}
        />
      ))}
    </div>
  );
}
