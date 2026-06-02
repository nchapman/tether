//! Windows-only link wiring for the statically linked FFmpeg.
//!
//! Identical to `crates/tether-codec/build.rs` — both crates statically link
//! the staged LGPL FFmpeg via rsmpeg, and each produces its own test binary, so
//! each needs to name the Win32 import libs the static archives reference. On
//! Linux/macOS rusty_ffmpeg's pkg-config `.statik(true)` path supplies these
//! from the `.pc` `Libs:` lines; this build script is inert there. Keep the
//! `SYSTEM_LIBS` set in sync with tether-codec's if the upstream FFmpeg
//! configure changes.
fn main() {
    println!("cargo:rerun-if-env-changed=FFMPEG_LIBS_DIR");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let Ok(libs_dir) = std::env::var("FFMPEG_LIBS_DIR") else {
        return;
    };

    // `vpl.lib` (bundled oneVPL dispatcher) distinguishes our staged static
    // prefix from a dynamic/system FFmpeg, where the DLLs already carry these
    // deps and emitting them would fail the link.
    if !std::path::Path::new(&libs_dir).join("vpl.lib").exists() {
        return;
    }

    println!("cargo:rustc-link-search=native={libs_dir}");
    println!("cargo:rustc-link-lib=static=vpl");

    const SYSTEM_LIBS: &[&str] = &[
        // libavutil / libavcodec / libavfilter
        "mfuuid", "ole32", "strmiids", "user32", "advapi32", "bcrypt",
        // libavdevice (vfw/dshow capture, GDI)
        "psapi", "uuid", "oleaut32", "shlwapi", "gdi32", "vfw32",
        // libavformat (TLS / sockets)
        "secur32", "ncrypt", "crypt32", "ws2_32",
    ];
    for lib in SYSTEM_LIBS {
        println!("cargo:rustc-link-lib={lib}");
    }
}
