# Tether — common dev workflows.
#
# `make help` (or just `make`) lists every target with a one-line
# description. Targets are documentation; the real cost-of-truth is
# the cargo invocation each one wraps.

.PHONY: help build release test test-hw test-correctness bench probe clippy fmt check clean \
        ffmpeg ffmpeg-clean engines shell shell-install shell-check

help:
	@awk 'BEGIN { FS = ":.*##"; printf "\nTether dev targets:\n\n" } \
	      /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' \
	      $(MAKEFILE_LIST)

# --- FFmpeg --------------------------------------------------------------
# tether-codec statically links the prebuilt LGPL FFmpeg from the
# nchapman/tether-ffmpeg releases (pinned in scripts/ffmpeg-version). The
# fetch is idempotent — a no-op once the matching artifact is staged — so it's
# cheap to list as a prerequisite of every compiling target.

ffmpeg: ## Download + stage the pinned static FFmpeg for this host (idempotent)
	@scripts/fetch-ffmpeg.sh

ffmpeg-clean: ## Remove the staged FFmpeg so the next build re-fetches
	rm -rf vendor/ffmpeg

build: ffmpeg ## Dev build of the full workspace
	cargo build --workspace

release: ffmpeg ## Release build of the host + client binaries
	cargo build --release -p tether-host -p tether-client

test: ffmpeg ## Fast unit tests (no hardware required)
	cargo test --workspace --lib

check: ## Static workspace hygiene checks (no compile / no codegen)
	@scripts/check_test_support_gating.sh

test-hw: test-correctness bench ## All hardware-backed tests + benchmarks

test-correctness: ffmpeg ## Hardware-backed correctness tests (VAAPI codec, render dmabuf, gpuconvert; skips bench)
	cargo test -p tether-codec --lib -- --ignored --skip bench
	cargo test -p tether-render --lib -- --ignored
	cargo test -p tether-gpuconvert --lib -- --ignored

bench: ffmpeg ## VAAPI benchmark matrix (codec x resolution x layer; prints p50/p99/max ms)
	cargo test -p tether-codec --lib bench -- --ignored --nocapture --test-threads=1

probe: ffmpeg ## Print which VAAPI codecs are buildable on this host
	cargo test -p tether-codec --lib probe_encoder_kind_smoke -- --ignored --nocapture

clippy: ffmpeg ## Lint the workspace (advisory — some pre-existing warnings exist)
	cargo clippy --workspace --all-targets

fmt: ## Format the workspace
	cargo fmt --all

clean: ## Cargo clean
	cargo clean

# --- Tauri shell (control-plane UI) ---------------------------------------
# The shell is excluded from the cargo workspace and builds via pnpm + the
# Tauri CLI. It spawns the host/client binaries from target/debug, so those
# must be built first (the `shell` target does this for you).

engines: ffmpeg ## Build the host + client binaries the shell spawns (target/debug)
	cargo build -p tether-host -p tether-client

shell-install: ## Install the shell's frontend dependencies (run once, or after dep changes)
	pnpm --dir apps/tether-shell install

shell: engines ## Run the Tauri shell in dev mode (builds the engines first)
	pnpm --dir apps/tether-shell tauri dev

shell-check: ## Typecheck the shell without running it (TypeScript + src-tauri Rust)
	pnpm --dir apps/tether-shell exec tsc --noEmit
	cargo check --manifest-path apps/tether-shell/src-tauri/Cargo.toml
