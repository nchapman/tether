fn main() {
    // VAAPI is Linux-only. Short-circuit on other platforms so a
    // workspace-wide `cargo check` on macOS/Windows doesn't fail at
    // pkg-config when no one's actually going to link the symbols.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }
    // 1.10 introduced vaExportSurfaceHandle. Probe enforces it instead of
    // trusting a comment that will rot.
    pkg_config::Config::new()
        .atleast_version("1.10")
        .probe("libva")
        .expect("libva >= 1.10 not found via pkg-config; install libva-dev / libva-devel");
}
