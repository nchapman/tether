//! Link wiring for the audio codec's direct libopus binding.
//!
//! `tether-audio` calls libopus directly (see `src/opus_sys.rs`) rather than
//! going through FFmpeg/rsmpeg as the video codec does, so it links the
//! standalone `libopus` archive the static FFmpeg package already stages — the
//! same one FFmpeg's `libavcodec.a` resolves its `opus_*` references from, so no
//! second copy of the library enters the binary.
//!
//! The staged prefix's lib dir is named by `FFMPEG_LIBS_DIR` (set for every
//! target in `.cargo/ffmpeg-env.toml`). We gate on the opus archive actually
//! being present so a non-staged / system FFmpeg build doesn't mis-link here.
fn main() {
    println!("cargo:rerun-if-env-changed=FFMPEG_LIBS_DIR");

    let Ok(libs_dir) = std::env::var("FFMPEG_LIBS_DIR") else {
        return;
    };
    let dir = std::path::Path::new(&libs_dir);

    // Static opus archive name: MSVC `opus.lib`, otherwise `libopus.a`.
    let is_msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    let archive = if is_msvc { "opus.lib" } else { "libopus.a" };
    if !dir.join(archive).exists() {
        return;
    }

    println!("cargo:rustc-link-search=native={libs_dir}");
    println!("cargo:rustc-link-lib=static=opus");

    // libopus references libm; FFmpeg (which would otherwise pull it in) is no
    // longer linked by this crate, so the standalone test binary needs it
    // explicitly. No-op where libm is part of the C runtime (macOS, MSVC).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=m");
    }
}
