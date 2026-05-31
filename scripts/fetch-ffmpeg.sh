#!/usr/bin/env bash
# Download the prebuilt static LGPL FFmpeg for this host from the
# tether-ffmpeg releases and stage it under vendor/ffmpeg/prefix.
#
# rusty_ffmpeg (under rsmpeg, in tether-codec) links against this prefix via
# the committed `.cargo/ffmpeg-env.toml` (FFMPEG_PKG_CONFIG_PATH +
# FFMPEG_LINK_MODE=static). Run this once after cloning, and again whenever
# scripts/ffmpeg-version changes. Idempotent: a matching staged prefix is left
# untouched unless --force is passed.
#
# Why download instead of building from source: the LGPL/static configure
# matrix (vaapi, videotoolbox, nvenc/amf/qsv, mediafoundation) lives in the
# nchapman/tether-ffmpeg repo and is built+attested by CI there. Dev and CI
# here just consume the pinned artifact, so no FFmpeg build toolchain is needed.
#
# Usage: scripts/fetch-ffmpeg.sh [--force]
set -euo pipefail

REPO_SLUG="nchapman/tether-ffmpeg"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${REPO_ROOT}/vendor/ffmpeg"
PREFIX_DIR="${VENDOR_DIR}/prefix"
STAMP_FILE="${VENDOR_DIR}/.stamp"

log()  { printf '\033[36m[fetch-ffmpeg]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[fetch-ffmpeg] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

VERSION="$(tr -d '[:space:]' < "${REPO_ROOT}/scripts/ffmpeg-version")"
[[ -n "${VERSION}" ]] || die "scripts/ffmpeg-version is empty"

# --- Map the host to a release artifact (os, arch) --------------------------
case "$(uname -s)" in
  Linux)  OS="linux" ;;
  Darwin) OS="macos" ;;
  # Windows links FFmpeg through a different rusty_ffmpeg path (FFMPEG_LIBS_DIR,
  # not pkg-config); that wiring doesn't exist yet, so don't pretend to stage it.
  MINGW*|MSYS*|CYGWIN*) die "Windows is not wired for static FFmpeg yet (Linux/macOS only)" ;;
  *) die "unsupported OS: $(uname -s)" ;;
esac
case "$(uname -m)" in
  x86_64|amd64)  ARCH="x86_64" ;;
  arm64|aarch64) ARCH="arm64" ;;
  *) die "unsupported arch: $(uname -m)" ;;
esac

TARGET="${OS}-${ARCH}"
ASSET="tether-ffmpeg-${VERSION}-${TARGET}-lgpl-static.tar.xz"
WANT_STAMP="${VERSION} ${TARGET}"

# --- Idempotency: skip if the staged prefix already matches -----------------
if [[ "${FORCE}" -eq 0 && -f "${STAMP_FILE}" && -d "${PREFIX_DIR}/lib/pkgconfig" ]]; then
  if [[ "$(cat "${STAMP_FILE}")" == "${WANT_STAMP}" ]]; then
    log "up to date: ${ASSET} already staged (pass --force to re-fetch)"
    exit 0
  fi
fi

# --- Download + checksum-verify into a scratch dir --------------------------
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

log "fetching ${ASSET}"
if command -v gh >/dev/null 2>&1; then
  gh release download "v${VERSION}" -R "${REPO_SLUG}" \
    -p "${ASSET}" -p "${ASSET}.sha256" -D "${TMP_DIR}" \
    || die "gh release download failed for tag v${VERSION}"
else
  BASE_URL="https://github.com/${REPO_SLUG}/releases/download/v${VERSION}"
  for f in "${ASSET}" "${ASSET}.sha256"; do
    curl -fsSL "${BASE_URL}/${f}" -o "${TMP_DIR}/${f}" \
      || die "curl failed for ${BASE_URL}/${f}"
  done
fi

log "verifying checksum"
( cd "${TMP_DIR}"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c "${ASSET}.sha256"
  else
    shasum -a 256 -c "${ASSET}.sha256"   # macOS has no sha256sum
  fi
) || die "checksum mismatch for ${ASSET}"

# --- Stage into vendor/ffmpeg/prefix ----------------------------------------
log "extracting into ${PREFIX_DIR#"${REPO_ROOT}/"}"
rm -rf "${PREFIX_DIR}"
mkdir -p "${VENDOR_DIR}"
# The tarball's top-level dir is `prefix/`, so this lands at vendor/ffmpeg/prefix.
tar -C "${VENDOR_DIR}" -xJf "${TMP_DIR}/${ASSET}"
[[ -d "${PREFIX_DIR}/lib/pkgconfig" ]] || die "extracted tree missing lib/pkgconfig"

# --- Relocate the pkg-config prefix -----------------------------------------
# The .pc files bake the CI runner's absolute prefix into prefix=/libdir=/
# includedir= (libdir/includedir are literal paths, not ${prefix}-derived, so
# pkgconf --define-prefix alone wouldn't fix them). Rewrite the baked prefix
# string to this machine's staged prefix across every .pc.
BAKED_PREFIX="$(sed -n 's/^prefix=//p' "${PREFIX_DIR}/lib/pkgconfig/libavutil.pc" | head -n1)"
[[ -n "${BAKED_PREFIX}" ]] || die "could not read baked prefix from libavutil.pc"
log "relocating pkg-config prefix -> ${PREFIX_DIR}"
baked="${BAKED_PREFIX}" real="${PREFIX_DIR}" \
  perl -i -pe 's{\Q$ENV{baked}\E}{$ENV{real}}g' "${PREFIX_DIR}"/lib/pkgconfig/*.pc

echo "${WANT_STAMP}" > "${STAMP_FILE}"
log "done: ${ASSET} staged for static linking"
