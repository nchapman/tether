#!/usr/bin/env bash
# Stage the host/client engine binaries as Tauri sidecars.
#
# Tauri's `externalBin` bundles a binary named `<name>-<target-triple>` from
# the configured directory and installs it next to the app binary as `<name>`.
# This copies the release engines into apps/tether-shell/src-tauri/binaries/
# with that triple suffix so `tauri build` can pick them up.
#
# Run after `cargo build --release -p tether-host -p tether-client`. The shell
# resolves the installed sidecars at runtime via current_exe() — see
# apps/tether-shell/src-tauri/src/supervisor.rs.
#
# Usage: scripts/stage-engine-binaries.sh [target-dir]
#   target-dir defaults to target/release.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-${REPO_ROOT}/target/release}"
DEST_DIR="${REPO_ROOT}/apps/tether-shell/src-tauri/binaries"

log() { printf '\033[36m[stage-engines]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31m[stage-engines] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Host target triple, e.g. x86_64-unknown-linux-gnu / aarch64-apple-darwin.
TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
[[ -n "${TRIPLE}" ]] || die "could not determine host target triple from rustc -vV"

# Windows executables carry a .exe suffix; Unix engines have none.
case "${TRIPLE}" in
  *windows*) EXE=".exe" ;;
  *)         EXE="" ;;
esac

mkdir -p "${DEST_DIR}"
for engine in tether-host tether-client; do
  src="${TARGET_DIR}/${engine}${EXE}"
  dst="${DEST_DIR}/${engine}-${TRIPLE}${EXE}"
  [[ -f "${src}" ]] || die "missing ${src} — build it with \`cargo build --release -p ${engine}\`"
  cp "${src}" "${dst}"
  log "staged ${engine}${EXE} -> binaries/${engine}-${TRIPLE}${EXE}"
done

log "done: engines staged for Tauri externalBin"
