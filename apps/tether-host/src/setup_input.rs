//! Linux input permission: granting the Tether host access to
//! `/dev/uinput` for portal-free input injection.
//!
//! Access is gated by a udev rule (`60-tether-uinput.rules`). There are
//! three ways it gets installed, in order of preference:
//!
//!   1. **deb/rpm install** ships the rule system-wide — nothing to do at
//!      runtime (see `packaging/linux/udev/`).
//!   2. **First connection** (AppImage / `cargo run` / any host without
//!      the rule): when the injector finds `/dev/uinput` unavailable, the
//!      host requests the grant through a GUI PolicyKit dialog and retries
//!      — the same "permission prompt on first use" model as the
//!      screen-capture portal. This is [`linux_injector`].
//!   3. **`--setup-input`**: an explicit, scriptable pre-provision step
//!      (handy for headless setup or to grant ahead of time). This is
//!      [`run`].
//!
//! All three install the identical rule bytes; the privileged write goes
//! through `pkexec` so the user approves one PolicyKit dialog rather than
//! editing files as root.

use std::process::Command;

use tether_input::inject::{InjectError, Injector, NoopInjector, UinputInjector};

/// Where the rule lands when we install it at runtime. `/etc/udev/rules.d`
/// is the local-admin directory (packages use `/usr/lib/udev/rules.d`);
/// 60- orders it after the system defaults but before higher-numbered
/// local overrides.
const RULE_PATH: &str = "/etc/udev/rules.d/60-tether-uinput.rules";

/// The rule content, kept in lockstep with the packaged copy so the
/// installer and the package ship identical bytes.
const RULE: &str = include_str!("../../../packaging/linux/udev/60-tether-uinput.rules");

/// Install the udev rule via `pkexec`, reload udev, load the uinput
/// module, and settle so the new permissions apply *now* (the caller can
/// retry opening `/dev/uinput` immediately on return). Returns `Ok(())` on
/// success, or a human-readable error describing the failure.
fn install_udev_rule() -> Result<(), String> {
    // Privileged script: write the rule from argv (no shell escaping of
    // the rule body needed), then make the new permissions apply *now*.
    // Order matters:
    //   1. write the rule, then
    //   2. `--reload-rules` (synchronous over the control socket) so the
    //      new rule is live *before* any uevent that should match it,
    //   3. `modprobe` to load the module / create the node — the `add`
    //      uevent it emits is now evaluated against the new rule, and
    //   4. an explicit `trigger` (`change`) re-runs the `uaccess` builtin
    //      as belt-and-suspenders if the module was already loaded.
    // Reloading before the module load is the fix for the otherwise-racy
    // ordering where the `add` event could be processed under the old
    // ruleset. `settle` blocks until the triggered events finish, so the
    // caller's immediate re-open sees the applied ACL. `$1` is the rule
    // body.
    const SCRIPT: &str = "printf '%s' \"$1\" > /etc/udev/rules.d/60-tether-uinput.rules \
         && udevadm control --reload-rules \
         && { modprobe uinput 2>/dev/null || true; } \
         && { udevadm trigger --subsystem-match=misc --sysname-match=uinput 2>/dev/null \
              || udevadm trigger /dev/uinput 2>/dev/null || udevadm trigger; } \
         && { udevadm settle --timeout=5 2>/dev/null || true; }";

    let status = Command::new("pkexec")
        .arg("/bin/sh")
        .arg("-c")
        .arg(SCRIPT)
        .arg("sh") // $0 for the inner shell
        .arg(RULE) // $1 — the rule body
        .status();

    match status {
        Ok(s) if s.success() => Ok(()),
        // pkexec exits 126 when the user dismisses/declines the dialog,
        // 127 when authentication can't be obtained. Either way we just
        // report it; the caller decides how loud to be.
        Ok(s) => Err(format!("pkexec exited with {s}")),
        Err(e) => Err(format!("could not run pkexec ({e})")),
    }
}

/// The manual, run-it-as-root fallback shown when `pkexec` is unavailable
/// or declined. Kept in one place so the CLI and the lazy path print the
/// same instructions. Embeds the rule body (via a quoted heredoc, so the
/// backticks in its comments stay literal) rather than pointing at a
/// source-tree path — installed/AppImage users have no checkout on disk.
fn manual_install_hint() -> String {
    format!(
        "Grant it manually as root:\n\
         \x20   sudo tee {RULE_PATH} > /dev/null <<'EOF'\n\
         {RULE}EOF\n\
         \x20   sudo udevadm control --reload-rules \
         && sudo modprobe uinput && sudo udevadm trigger"
    )
}

/// `--setup-input`: install the rule up front and exit. Returns a process
/// exit code (0 = success) so `main` can exit directly without spinning up
/// the server. Idempotent — re-running just rewrites the same rule.
pub fn run() -> i32 {
    println!("Installing {RULE_PATH} (you'll be asked to authenticate)...");
    match install_udev_rule() {
        Ok(()) => {
            println!(
                "Input permissions configured.\n\
                 \n\
                 If the pointer/keyboard still don't respond, your host has no active\n\
                 local seat (e.g. headless): add your user to the `input` group and log\n\
                 back in:\n    sudo usermod -aG input $USER"
            );
            0
        }
        Err(e) => {
            eprintln!("Setup failed ({e}).\n\n{}", manual_install_hint());
            1
        }
    }
}

/// Build the Linux input injector, requesting the `/dev/uinput` permission
/// grant through a GUI prompt on first use if it isn't already granted.
///
/// This is the host's per-connection injector factory. The flow mirrors
/// the screen-capture portal: try to use the device; if it's unavailable
/// (no udev rule yet), surface a one-time system authorization dialog,
/// then retry. A declined or failed grant degrades to [`NoopInjector`]
/// (video keeps working, input is dropped with a logged hint) — never a
/// fatal error, and the next connection will ask again.
pub async fn linux_injector() -> Box<dyn Injector> {
    match UinputInjector::connect().await {
        Ok(inj) => return Box::new(inj),
        Err(InjectError::DeviceUnavailable(reason)) => {
            tracing::info!(
                %reason,
                "input device not yet authorized; requesting one-time access via PolicyKit"
            );
            // pkexec blocks on the PolicyKit dialog — keep it off the async
            // runtime so we don't stall a worker while the user decides.
            match tokio::task::spawn_blocking(install_udev_rule).await {
                Ok(Ok(())) => match connect_after_grant().await {
                    Some(inj) => {
                        tracing::info!("input access granted; uinput backend active");
                        return inj;
                    }
                    None => tracing::warn!(
                        "input access was granted but /dev/uinput still won't open \
                         (udev may still be applying it); input disabled for this \
                         session — the next connection will pick it up"
                    ),
                },
                Ok(Err(e)) => tracing::warn!(
                    error = %e,
                    "input authorization was not completed; input disabled for this session. {}",
                    manual_install_hint()
                ),
                Err(join) => tracing::warn!(
                    error = %join,
                    "input authorization task failed to run; input disabled for this session"
                ),
            }
        }
        Err(e) => tracing::warn!(
            error = %e,
            "linux input backend unavailable; input disabled for this session"
        ),
    }
    Box::new(NoopInjector)
}

/// Open the freshly granted device, retrying briefly: the `uaccess` ACL
/// write can land a beat after `udevadm settle` returns — and `settle`
/// itself may have timed out (we don't treat that as fatal). A few short
/// retries absorb that race so a slow udev doesn't cost the whole session;
/// when it works the user never notices. Returns `None` if the device
/// never became usable, or on any non-permission failure.
async fn connect_after_grant() -> Option<Box<dyn Injector>> {
    const ATTEMPTS: usize = 4;
    const DELAY: std::time::Duration = std::time::Duration::from_millis(150);
    for attempt in 0..ATTEMPTS {
        match UinputInjector::connect().await {
            Ok(inj) => return Some(Box::new(inj)),
            // Still being applied — wait and retry (but not after the last try).
            Err(InjectError::DeviceUnavailable(_)) if attempt + 1 < ATTEMPTS => {
                tokio::time::sleep(DELAY).await;
            }
            Err(e) => {
                tracing::debug!(error = %e, "post-grant uinput open failed");
                return None;
            }
        }
    }
    None
}
