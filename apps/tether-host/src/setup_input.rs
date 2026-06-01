//! `--setup-input`: one-time install of the udev rule that grants the
//! Tether host access to `/dev/uinput` for portal-free input injection.
//!
//! Runs the privileged step through `pkexec`, so the user approves a
//! single PolicyKit dialog rather than editing files as root. After this,
//! input injection works with no per-session prompt (see
//! `tether-input::inject::uinput`). Idempotent — re-running just rewrites
//! the same rule.

use std::process::Command;

/// Where the rule lands. 60- orders it after the system defaults but
/// before higher-numbered local overrides.
const RULE_PATH: &str = "/etc/udev/rules.d/60-tether-uinput.rules";

/// The rule content, kept in lockstep with the packaged copy so the
/// installer and the package ship identical bytes.
const RULE: &str = include_str!("../../../packaging/linux/udev/60-tether-uinput.rules");

/// Install the udev rule via `pkexec`, reload udev, and load the uinput
/// module. Returns a process exit code (0 = success) so `main` can exit
/// directly without spinning up the server.
pub fn run() -> i32 {
    // Privileged script: write the rule from argv (no shell escaping of
    // the rule body needed), then make the new permissions apply *now*.
    // Order matters — load the uinput module FIRST so the node exists,
    // then reload rules and trigger so the rule's GROUP/MODE/uaccess are
    // applied to it. Triggering before the node exists would no-op and
    // leave /dev/uinput unreadable until the next boot. `$1` is the rule
    // body, passed as a single argument below.
    const SCRIPT: &str = "printf '%s' \"$1\" > /etc/udev/rules.d/60-tether-uinput.rules \
         && { modprobe uinput 2>/dev/null || true; } \
         && udevadm control --reload-rules \
         && { udevadm trigger /dev/uinput 2>/dev/null || udevadm trigger; }";

    println!("Installing {RULE_PATH} (you'll be asked to authenticate)...");

    let status = Command::new("pkexec")
        .arg("/bin/sh")
        .arg("-c")
        .arg(SCRIPT)
        .arg("sh") // $0 for the inner shell
        .arg(RULE) // $1 — the rule body
        .status();

    match status {
        Ok(s) if s.success() => {
            println!(
                "Input permissions configured.\n\
                 \n\
                 If the pointer/keyboard still don't respond, your host has no active\n\
                 local seat (e.g. headless): add your user to the `input` group and log\n\
                 back in:\n    sudo usermod -aG input $USER"
            );
            0
        }
        Ok(s) => {
            eprintln!(
                "Setup failed (pkexec exited with {s}).\n\
                 \n\
                 Run the equivalent manually as root:\n\
                 \x20   sudo install -m 0644 packaging/linux/udev/60-tether-uinput.rules {RULE_PATH}\n\
                 \x20   sudo udevadm control --reload-rules && sudo udevadm trigger\n\
                 \x20   sudo modprobe uinput"
            );
            1
        }
        Err(e) => {
            eprintln!(
                "Could not run pkexec ({e}).\n\
                 \n\
                 Install PolicyKit, or run the equivalent manually as root:\n\
                 \x20   sudo install -m 0644 packaging/linux/udev/60-tether-uinput.rules {RULE_PATH}\n\
                 \x20   sudo udevadm control --reload-rules && sudo udevadm trigger\n\
                 \x20   sudo modprobe uinput"
            );
            1
        }
    }
}
