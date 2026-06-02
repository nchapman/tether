//! macOS hardware round-trip: ScreenCaptureKit system audio → Opus encode →
//! Opus decode → cpal playback.
//!
//! `#[ignore]` per the repo convention — it needs a real audio output device
//! and Screen Recording permission (SCK audio is gated behind it). Run with:
//!
//! ```text
//! cargo test -p tether-audio --test macos_audio_roundtrip -- --ignored --nocapture
//! ```
//!
//! Play something audible on the host while it runs to exercise non-silent
//! capture; the assertions only require that frames flow through the whole
//! chain, since CI machines are silent.
#![cfg(target_os = "macos")]

use std::time::{Duration, Instant};

use tether_audio::{capture, AudioPlayer, OpusConfig, OpusDecoder, OpusEncoder};

#[test]
#[ignore = "requires macOS audio output device + Screen Recording permission"]
fn macos_capture_encode_decode_playback_roundtrip() {
    let cfg = OpusConfig::default();

    let handle = capture::start(cfg).expect("start macOS system-audio capture");
    let (_player, sink) = AudioPlayer::with_defaults(cfg).expect("open default audio output");
    let mut enc = OpusEncoder::new(cfg).expect("build opus encoder");
    let mut dec = OpusDecoder::new(cfg).expect("build opus decoder");

    let mut captured = 0usize;
    let mut played = 0usize;
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        let Ok(frame) = handle.rx.recv_timeout(Duration::from_millis(200)) else {
            continue;
        };
        assert_eq!(frame.channels, cfg.channels, "capture channel count");
        captured += 1;
        for pkt in enc.encode(&frame.samples).expect("opus encode") {
            let pcm = dec.decode(&pkt).expect("opus decode");
            assert_eq!(pcm.channels, cfg.channels);
            sink.submit(&pcm);
            played += 1;
        }
    }
    handle.stop();

    let (underruns, overruns, _buffered) = sink.stats();
    eprintln!("captured={captured} played={played} underruns={underruns} overruns={overruns}");
    assert!(captured > 0, "captured no audio frames from SCK in 2s");
    assert!(played > 0, "no frames survived encode->decode to playback");
}
