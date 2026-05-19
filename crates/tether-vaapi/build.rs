fn main() {
    // 1.10 introduced vaExportSurfaceHandle. Probe enforces it instead of
    // trusting a comment that will rot.
    pkg_config::Config::new()
        .atleast_version("1.10")
        .probe("libva")
        .expect("libva >= 1.10 not found via pkg-config; install libva-dev / libva-devel");
}
