//! Windows-only link wiring for the statically linked FFmpeg.
//!
//! On Linux/macOS rusty_ffmpeg probes our staged prefix with pkg-config
//! (`.statik(true)`), so the transitive system libs the static archives need
//! come straight from the `.pc` `Libs:` lines. On Windows rusty_ffmpeg ignores
//! pkg-config entirely: it links the 7 FFmpeg libs from `FFMPEG_LIBS_DIR` and
//! nothing else. The static `.lib` archives reference Win32 APIs and the bundled
//! oneVPL (`vpl.lib`) that the import-lib (dynamic) path used to resolve inside
//! the FFmpeg DLLs — with static archives those externals are unresolved unless
//! we name them here. The set is the union of every `Libs:` line in the staged
//! `lib/pkgconfig/*.pc`; regenerate it if the upstream FFmpeg configure changes.
fn main() {
    println!("cargo:rerun-if-env-changed=FFMPEG_LIBS_DIR");

    // Only relevant for the staged static-prefix link on Windows. On Unix the
    // pkg-config path supplies these; bail so we never perturb that link.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let Ok(libs_dir) = std::env::var("FFMPEG_LIBS_DIR") else {
        return;
    };

    // `vpl.lib` (the bundled oneVPL dispatcher) is unique to our staged static
    // prefix. Its presence distinguishes "static link against vendor/ffmpeg"
    // from a dynamic/system FFmpeg (e.g. a BtbN shared build pointed at by a
    // local mise override), where the FFmpeg DLLs already carry these deps and
    // emitting them — `static=vpl` especially — would fail the link.
    if !std::path::Path::new(&libs_dir).join("vpl.lib").exists() {
        return;
    }

    // oneVPL dispatcher ships as a static lib in the staged prefix alongside the
    // FFmpeg .lib files; the rest are Windows SDK import libs found on PATH.
    println!("cargo:rustc-link-search=native={libs_dir}");
    println!("cargo:rustc-link-lib=static=vpl");

    // Win32 import libs the FFmpeg static archives reference, grouped by the
    // .pc that pulls them in (libavutil/libavcodec/libavfilter, libavdevice,
    // libavformat). Duplicates across .pc files are collapsed.
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
