# Releasing Tether

Tether ships as a single **Tether shell** installer per platform, built with
Tauri's bundler. The `tether-host` / `tether-client` engines are bundled inside
the installer as Tauri sidecars (`externalBin`), and the shell resolves them at
runtime next to its own binary. Auto-update is wired through Tauri's updater
plugin against GitHub Releases.

## CI overview

- **`.github/workflows/ci.yml`** — on every push/PR: rustfmt, the test-support
  gating check, a warning-free workspace build, no-hardware unit tests, the
  shell typecheck + backend tests, and advisory clippy. Runs on Linux, macOS,
  and Windows. Hardware tests (VAAPI/Vulkan/Metal) are **not** run in CI — they
  need a real GPU; run `make test-hw` locally.
- **`.github/workflows/release.yml`** — on a `v*` tag: builds the engines,
  stages them as sidecars, and runs `tauri-action` to bundle, sign the updater
  manifest, and publish a **draft** GitHub Release with installers + `latest.json`.
- **`.github/actions/setup`** — shared setup (mise toolchains, Linux system
  deps, static FFmpeg fetch + cache, cargo cache).
- **Local pre-commit hook** — `make hooks` installs `.githooks/pre-commit`,
  which runs rustfmt (workspace + shell) and the test-support gating check
  before each commit so the tree can't drift out of format. Run `make ci` to
  reproduce the full no-hardware gate locally.

## One-time setup (required before the first release)

The updater public key is committed in `tauri.conf.json`
(`plugins.updater.pubkey`). The matching **private key** must be added as repo
secrets so CI can sign updates:

| Secret | Value |
| --- | --- |
| `TAURI_SIGNING_PRIVATE_KEY` | Contents of the minisign private key (the `*.key` file). |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | The key password (empty string if the key has none). |

To rotate or regenerate the keypair:

```sh
pnpm --dir apps/tether-shell tauri signer generate -w /path/outside/repo/tether-updater.key
```

Then put the printed public key into `tauri.conf.json` and the private key into
the secret above. **Never commit the private key.**

> OS code signing (Apple notarization, Windows Authenticode) is **not** set up
> yet — installers are unsigned, so macOS shows a Gatekeeper warning and Windows
> a SmartScreen prompt on first run. The updater manifest is still signed, so
> auto-update integrity is protected regardless.

## Cutting a release

1. Bump the version everywhere it appears:
   ```sh
   scripts/sync-version.sh 0.1.0
   cargo build          # refresh Cargo.lock
   ```
2. Commit and tag:
   ```sh
   git commit -am "Release v0.1.0"
   git tag v0.1.0
   git push origin main --tags
   ```
3. The tag triggers `release.yml`. When it finishes, a **draft** release holds
   the installers + `latest.json`. Review the artifacts, then **publish** the
   release. The updater endpoint (`releases/latest/download/latest.json`) only
   resolves once the release is published and non-prerelease.

## Runtime dependencies (not bundled)

FFmpeg is statically linked into the engines, but platform GPU/capture stacks
are dynamic system libraries the installer does **not** ship:

- **Linux:** a VAAPI driver, a Vulkan ICD, and a running PipeWire + portal stack.
- **macOS:** none beyond the OS frameworks.
- **Windows:** none beyond the OS (D3D11 / Media Foundation are system components).

## Local packaging

`make package` runs the same bundle steps locally (unsigned updater artifacts
unless `TAURI_SIGNING_PRIVATE_KEY` is exported). Output lands under
`apps/tether-shell/src-tauri/target/release/bundle/`.
