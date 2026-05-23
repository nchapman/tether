#!/usr/bin/env bash
# Fail if any `[dependencies]` or `[target.*.dependencies]` table in a
# workspace Cargo.toml enables a `*test-support*` feature.
#
# The `test-support` feature on each crate gates fakes meant only for
# downstream integration tests. Enabling it from a non-`[dev-dependencies]`
# slot would compile the fakes into release builds — every cargo build
# from anywhere in the dep graph would pull them in via feature unification.
#
# This script is intentionally simple: a structural awk pass over each
# Cargo.toml. It does not try to parse TOML, just tracks the most-recently-
# seen `[section]` header. False positives are vanishingly rare because
# `*test-support*` is namespaced to this project's convention.

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

# Discover every Cargo.toml in the workspace (members + apps).
mapfile -t tomls < <(
  find . -name Cargo.toml \
    -not -path './target/*' \
    -not -path './.git/*' \
  | sort
)

bad=()
for f in "${tomls[@]}"; do
  # awk script: enter dep-table mode whenever we see a [dependencies] or
  # a [target.'cfg(...)'.dependencies] (and any build-dependencies),
  # but NOT [dev-dependencies]. In that mode, flag any line whose
  # value references "test-support" in a `features = [...]` array.
  matches=$(awk '
    /^\[/                                                    { in_bad = 0 }
    /^\[(dependencies|build-dependencies)\]/                  { in_bad = 1; next }
    /^\[target\.[^\]]+\.(dependencies|build-dependencies)\]/  { in_bad = 1; next }
    /^\[dev-dependencies\]/                                  { in_bad = 0; next }
    /^\[/                                                    { in_bad = 0 }
    in_bad && /features[[:space:]]*=[[:space:]]*\[[^\]]*"[^"]*test-support[^"]*"/ {
      print FILENAME ":" NR ": " $0
    }
  ' "$f") || true
  if [[ -n "$matches" ]]; then
    bad+=("$matches")
  fi
done

if [[ ${#bad[@]} -gt 0 ]]; then
  printf 'ERROR: `test-support` features must only be enabled under [dev-dependencies].\n' >&2
  printf 'Enabling them elsewhere bakes test fakes into release builds via feature unification.\n' >&2
  printf 'Offending lines:\n' >&2
  printf '%s\n' "${bad[@]}" >&2
  exit 1
fi

printf 'check_test_support_gating: ok (%d Cargo.toml files scanned)\n' "${#tomls[@]}"
