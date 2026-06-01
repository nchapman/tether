#!/usr/bin/env bash
# Stage the Visual C++ runtime DLLs the Windows engines link against, so the
# Tauri bundler can ship them next to the app (app-local deployment).
#
# Our static FFmpeg is built with -MD (dynamic CRT) — see scripts/flags.sh in
# nchapman/tether-ffmpeg — and Rust's msvc target uses the dynamic CRT too, so
# tether-host/client/shell import msvcp140.dll, vcruntime140.dll, and
# vcruntime140_1.dll. Those are NOT part of Windows (they ship in the "Visual
# C++ 2015-2022 Redistributable"), so a clean machine may lack them. Microsoft
# permits app-local deployment of these exact files, so we copy them from the
# MSVC redist into binaries/ and reference them from tauri.windows.conf.json's
# bundle.resources (which Tauri auto-merges on Windows only).
#
# Run after building the engines, before `tauri build`. No-op off Windows.
#
# Usage: scripts/stage-windows-runtime.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/apps/tether-shell/src-tauri/binaries"

log() { printf '\033[36m[stage-winrt]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31m[stage-winrt] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Only Windows links the VC runtime; elsewhere this is a no-op so callers
# (Makefile, release.yml) can invoke it unconditionally.
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) ;;
  *)
    log "not Windows ($(uname -s)) — nothing to stage"
    exit 0
    ;;
esac

DLLS=(msvcp140.dll vcruntime140.dll vcruntime140_1.dll)

# Locate the MSVC redist via vswhere (ships with every VS 2017+ / Build Tools).
VSWHERE="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
[[ -x "${VSWHERE}" ]] || die "vswhere not found at ${VSWHERE}"
VS_PATH="$("${VSWHERE}" -latest -products '*' -property installationPath | tr -d '\r')"
[[ -n "${VS_PATH}" ]] || die "vswhere returned no VS installation"
# vswhere prints a Windows path; translate the drive letter for bash.
VS_ROOT="/$(printf '%s' "${VS_PATH}" | sed 's/\\/\//g; s/^\([A-Za-z]\):/\L\1/')"
log "VS install: ${VS_ROOT}"

# Pick the highest-versioned x64 CRT redist dir (Microsoft.VC<N>.CRT). The redist
# tree is the license-clean source for app-local copies.
mapfile -t CRT_DIRS < <(
  find "${VS_ROOT}/VC/Redist/MSVC" -type d -path '*/x64/Microsoft.VC*.CRT' 2>/dev/null | sort -V
)
[[ ${#CRT_DIRS[@]} -gt 0 ]] || die "no x64 Microsoft.VC*.CRT redist dir under ${VS_ROOT}/VC/Redist/MSVC"
CRT_DIR="${CRT_DIRS[-1]}"
log "CRT redist: ${CRT_DIR}"

mkdir -p "${DEST_DIR}"
for dll in "${DLLS[@]}"; do
  src="${CRT_DIR}/${dll}"
  [[ -f "${src}" ]] || die "missing ${dll} in ${CRT_DIR}"
  cp "${src}" "${DEST_DIR}/${dll}"
  log "staged ${dll} -> binaries/${dll}"
done

log "done: VC runtime DLLs staged for Tauri bundle.resources"
