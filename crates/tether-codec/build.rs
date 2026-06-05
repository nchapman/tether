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

    let target_os = std::env::var("CARGO_CFG_TARGET_OS");

    // macOS: rusty_ffmpeg's pkg-config link supplies `-lopus`, and a debug
    // build links fine. But under the release profile (`lto = "thin"`,
    // `codegen-units = 1`) the static `libopus.a` ends up scanned before
    // `libavcodec.a` on the ld64 command line, so opus members aren't pulled
    // and libavcodec's in-tree libopus wrapper (`--enable-libopus`) is left
    // with undefined `_opus_*` symbols — the release *test* binaries for this
    // crate and its dependents (tether-render) fail to link. The host/client
    // *binaries* escape it because they also pull in tether-audio, whose
    // build.rs links opus; the test binaries have no such second source.
    // Re-emit opus from this crate (which depends on rusty_ffmpeg) so the
    // archive is listed again, after libavcodec, resolving the references.
    // Listing a static archive twice on the link line is a standard, harmless
    // idiom — ld pulls each member once.
    if target_os.as_deref() == Ok("macos") {
        match std::env::var("FFMPEG_LIBS_DIR") {
            Ok(libs_dir) => {
                println!("cargo:rustc-link-search=native={libs_dir}");
                println!("cargo:rustc-link-lib=static=opus");
            }
            // `.cargo/ffmpeg-env.toml` always sets FFMPEG_LIBS_DIR, so this is
            // effectively unreachable — but warn rather than silently skip, since
            // the failure it guards (undefined `_opus_*` in release test binaries)
            // is silent and profile-dependent.
            Err(_) => println!(
                "cargo:warning=FFMPEG_LIBS_DIR unset; skipping macOS -lopus re-emit. \
                 Run `mise run ffmpeg` if release test binaries fail to link."
            ),
        }
        return;
    }

    // Only relevant for the staged static-prefix link on Windows. On Unix the
    // pkg-config path supplies these; bail so we never perturb that link.
    if target_os.as_deref() != Ok("windows") {
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

    // oneVPL dispatcher and libopus both ship as static libs in the staged
    // prefix alongside the FFmpeg .lib files; the rest are Windows SDK import
    // libs found on PATH. `opus.lib` resolves the libopus wrapper objects in
    // libavcodec (`--enable-libopus`): `libavcodec.pc`'s `Libs:` line names it.
    // Without it the tether-codec test binary fails to link
    // (`unresolved external symbol opus_strerror` / `opus_multistream_*`).
    // `tether-audio/build.rs` links the *same* staged `opus.lib` for its own
    // direct libopus binding; the host/client binaries (which pull in both
    // crates) get a single copy of the archive. But this crate owns the
    // FFmpeg link, so it must name opus itself — a binary using tether-codec
    // without tether-audio (e.g. this crate's own test binary) has no other
    // source for the symbol.
    println!("cargo:rustc-link-search=native={libs_dir}");
    println!("cargo:rustc-link-lib=static=vpl");
    println!("cargo:rustc-link-lib=static=opus");

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
