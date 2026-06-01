#!/usr/bin/env bash
# Set the project version across every manifest that carries one, so a `v<ver>`
# release tag matches the bundle version Tauri stamps into installers.
#
# Updates:
#   - Cargo.toml                                  [workspace.package] version
#   - apps/tether-shell/package.json              "version"
#   - apps/tether-shell/src-tauri/Cargo.toml      [package] version
#   - apps/tether-shell/src-tauri/tauri.conf.json "version"
#
# After running, `cargo build` to refresh Cargo.lock, then commit and tag.
# See docs/RELEASING.md.
#
# Usage: scripts/sync-version.sh <version>   e.g. scripts/sync-version.sh 0.1.0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { printf '\033[31m[sync-version] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }
log() { printf '\033[36m[sync-version]\033[0m %s\n' "$*" >&2; }

VERSION="${1:-}"
[[ -n "${VERSION}" ]] || die "usage: scripts/sync-version.sh <version>"
# Plain semver, optional pre-release suffix (e.g. 0.1.0, 1.2.3-rc1).
[[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]] \
  || die "not a semver version: ${VERSION}"

# `^version = ` matches only the [workspace.package]/[package] line — inline
# dependency `{ version = "..." }` entries are mid-line, never at column 0.
v="${VERSION}" perl -i -pe 's/^version = ".*"/version = "$ENV{v}"/' \
  "${REPO_ROOT}/Cargo.toml" \
  "${REPO_ROOT}/apps/tether-shell/src-tauri/Cargo.toml"

# The top-level "version" key sits at 2-space indent in these formatted JSON
# files; anchoring to `^  "version"` (column 0 + two spaces) replaces only it,
# never a deeper-nested "version" a future edit might add.
v="${VERSION}" perl -i -pe 's/^  "version": "[^"]*"/  "version": "$ENV{v}"/' \
  "${REPO_ROOT}/apps/tether-shell/package.json" \
  "${REPO_ROOT}/apps/tether-shell/src-tauri/tauri.conf.json"

log "set version to ${VERSION} across workspace + shell manifests"
log "next: \`cargo build\` to refresh Cargo.lock, then commit and \`git tag v${VERSION}\`"
