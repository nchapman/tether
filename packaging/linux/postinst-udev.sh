#!/bin/sh
# Post-install hook for the Tether deb/rpm packages.
#
# The package ships 60-tether-uinput.rules into /usr/lib/udev/rules.d so
# the host can inject input via /dev/uinput without a runtime permission
# prompt. Reload udev so the new rule is live, load the uinput module so
# its `add` uevent is evaluated against that rule (creating the node for
# the current boot — the rule's static_node handles subsequent boots),
# then trigger to (re)apply the rule to an already-present node.
#
# Best-effort throughout: every command is guarded and the script always
# exits 0, so a failure here can never abort package installation (a
# non-zero deb postinst / rpm %post would mark the install as failed).
# Runs unconditionally — ignoring the install-vs-upgrade arg ($1) is safe
# because reload/modprobe/trigger are idempotent on both deb and rpm.

if command -v udevadm >/dev/null 2>&1; then
    udevadm control --reload-rules || true
fi

modprobe uinput 2>/dev/null || true

if command -v udevadm >/dev/null 2>&1; then
    udevadm trigger --subsystem-match=misc --sysname-match=uinput 2>/dev/null \
        || udevadm trigger || true
fi

exit 0
