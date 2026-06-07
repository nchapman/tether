#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STRICT=0
SUMMARY_ONLY=0
if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    case "$arg" in
      --strict)
        STRICT=1
        ;;
      --summary-only)
        SUMMARY_ONLY=1
        ;;
      --help|-h)
        cat <<HELP
Usage: scripts/check_hardware_test_metadata.sh [--strict] [--summary-only]

Audits #[ignore] tests in workspace Rust files.

Default:
  - prints a machine-readable and summary report.
  - always exits 0.

--strict:
  - exits non-zero if any ignore annotation is missing
    a "requires" or "diagnostic" marker.

--summary-only:
  - prints only summary counts and issue list.
HELP
        exit 0
        ;;
      *)
        echo "Unknown option: $arg" >&2
        echo "Use --help for usage." >&2
        exit 2
        ;;
    esac
  done
fi

TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT

# Emit: file<TAB>line<TAB>test<TAB>ignore_msg
while IFS= read -r file; do
  # rg emits backslash separators on Windows. Normalize to forward slashes so
  # awk's `-v` assignment doesn't escape-process them (\t -> tab, etc.) and so
  # the family `case "$file"` patterns below still match.
  file="${file//\\//}"
  awk -v f="$file" '
    BEGIN {
      pending_msg = "";
      pending_line = 0;
    }

    function clear_pending() {
      pending_msg = "";
      pending_line = 0;
    }

    function test_name_from(line, name) {
      if (match(line, /fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(/) == 0) {
        return "";
      }
      name = substr(line, RSTART, RLENGTH);
      sub(/^fn[[:space:]]+/, "", name);
      sub(/[[:space:]]*\($/, "", name);
      return name;
    }

    function emit_if_pending_for(line, name) {
      if (pending_msg == "") {
        return 0;
      }
      name = test_name_from(line);
      if (name == "") {
        return 0;
      }
      printf "%s\t%d\t%s\t%s\n", f, pending_line, name, pending_msg;
      clear_pending();
      return 1;
    }

    /^[[:space:]]*#\[ignore[[:space:]]*\]$/ {
      pending_msg = "<missing ignore message>";
      pending_line = NR;
      next;
    }

    /^[[:space:]]*#\[ignore[[:space:]]*=[[:space:]]*"/ {
      pending_msg = $0;
      sub(/^[[:space:]]*#\[ignore[[:space:]]*=[[:space:]]*"/, "", pending_msg);
      sub(/"[[:space:]]*\][[:space:]]*$/, "", pending_msg);
      pending_line = NR;
      next;
    }

    pending_msg != "" {
      if (emit_if_pending_for($0)) {
        next;
      }

      # Attribute stacks and blank/comment lines may sit between #[ignore]
      # and the test function. Any other item boundary means the annotation is
      # not attached to a function Cargo can run.
      if ($0 ~ /^[[:space:]]*($|#\[|\/\/|\/\*\*|\*)/) {
        next;
      }
      if ($0 ~ /^[[:space:]]*(pub[[:space:]]+)?(mod|struct|enum|impl|trait|type|const|static)[[:space:]]+/) {
        printf "%s\t%d\t%s\t%s\n", f, pending_line, "<unassociated>", pending_msg;
        clear_pending();
        next;
      }
    }

    END {
      if (pending_msg != "") {
        printf "%s\t%d\t%s\t%s\n", f, pending_line, "<unassociated>", pending_msg;
      }
    }
  ' "$file"
done < <(rg --files crates apps -g '*.rs') > "$TMPFILE"

total=$(wc -l < "$TMPFILE")
missing_require=0
missing_run_hint=0
codec_count=0
render_count=0
gpuconvert_count=0
scaler_count=0
audio_loopback_count=0
other_count=0
bench_like=0
diagnostic_like=0
unknown_marker=()

while IFS=$'\t' read -r file line test msg; do
  [[ -z "$file" ]] && continue

  family="other"
  case "$file" in
    */tether-codec/*|*/tether-probe/*)
      family="codec"
      ;;
    */tether-render/*)
      family="render"
      ;;
    */tether-gpuconvert/*)
      family="gpuconvert"
      ;;
    */tether-scaler/*)
      family="scaler"
      ;;
    */tether-audio/*)
      family="audio_loopback"
      ;;
  esac
  case "$family" in
    codec)
      ((codec_count += 1))
      ;;
    render)
      ((render_count += 1))
      ;;
    gpuconvert)
      ((gpuconvert_count += 1))
      ;;
    scaler)
      ((scaler_count += 1))
      ;;
    audio_loopback)
      ((audio_loopback_count += 1))
      ;;
    other)
      ((other_count += 1))
      ;;
  esac

  if [[ "$msg" == diagnostic:* ]]; then
    ((diagnostic_like += 1))
  elif [[ "$msg" == perf:* || "$msg" == *benchmark* ]]; then
    ((bench_like += 1))
  fi

  if [[ "$msg" != requires\ * && "$msg" != diagnostic:* && "$msg" != perf:* ]]; then
    ((missing_require += 1))
    unknown_marker+=("$file:$line $test -> $msg")
  fi

  if [[ "$msg" == *"requires"* && "$msg" != *"run with:"* && "$msg" != *"run on"* ]]; then
    ((missing_run_hint += 1))
  fi

done < "$TMPFILE"

# Summary first.
echo "Hardware test ignore metadata"
echo "- total ignored tests: $total"
echo "- benchmark-like entries: $bench_like"
echo "- diagnostic marker entries: $diagnostic_like"
echo "- missing requires/diagnostic marker: $missing_require"
echo "- missing explicit command hint (advisory): $missing_run_hint"
echo
echo "Family summary:"
echo "- codec-family: $codec_count"
echo "- render-family: $render_count"
echo "- gpuconvert-family: $gpuconvert_count"
echo "- scaler-family: $scaler_count"
echo "- audio-loopback-family: $audio_loopback_count"
echo "- other-family: $other_count"

if [[ "$SUMMARY_ONLY" -eq 1 ]]; then
  true
else
  echo
  echo "Top-level inventory (file:line | test | message):"
  if [[ "$total" -eq 0 ]]; then
    echo "(no #[ignore = \"...\"] tests found)"
  else
    sort "$TMPFILE" | awk -F $'\t' '{printf "%s:%s | %s | %s\n", $1, $2, $3, $4}'
  fi
fi

if [[ ${#unknown_marker[@]} -gt 0 ]]; then
  echo
  echo "Non-standard ignore messages:"
  for item in "${unknown_marker[@]}"; do
    echo "- $item"
  done
fi

if [[ "$STRICT" -eq 1 && "$missing_require" -gt 0 ]]; then
  echo "strict mode: failing because of non-standard markers." >&2
  exit 1
fi

echo
echo "check_hardware_test_metadata: ok"
