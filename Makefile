# Tether — common dev workflows.
#
# `make help` (or just `make`) lists every target with a one-line
# description. Targets are documentation; the real cost-of-truth is
# the cargo invocation each one wraps.

.PHONY: help build release test test-hw test-correctness bench probe clippy fmt check clean

help:
	@awk 'BEGIN { FS = ":.*##"; printf "\nTether dev targets:\n\n" } \
	      /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' \
	      $(MAKEFILE_LIST)

build: ## Dev build of the full workspace
	cargo build --workspace

release: ## Release build of the host + client binaries
	cargo build --release -p tether-host -p tether-client

test: ## Fast unit tests (no hardware required)
	cargo test --workspace --lib

check: ## Static workspace hygiene checks (no compile / no codegen)
	@scripts/check_test_support_gating.sh

test-hw: test-correctness bench ## All hardware-backed tests + benchmarks

test-correctness: ## Hardware-backed correctness tests (VAAPI codec, render dmabuf, gpuconvert; skips bench)
	cargo test -p tether-codec --lib -- --ignored --skip bench
	cargo test -p tether-render --lib -- --ignored
	cargo test -p tether-gpuconvert --lib -- --ignored

bench: ## VAAPI benchmark matrix (codec x resolution x layer; prints p50/p99/max ms)
	cargo test -p tether-codec --lib bench -- --ignored --nocapture --test-threads=1

probe: ## Print which VAAPI codecs are buildable on this host
	cargo test -p tether-codec --lib probe_encoder_kind_smoke -- --ignored --nocapture

clippy: ## Lint the workspace (advisory — some pre-existing warnings exist)
	cargo clippy --workspace --all-targets

fmt: ## Format the workspace
	cargo fmt --all

clean: ## Cargo clean
	cargo clean
