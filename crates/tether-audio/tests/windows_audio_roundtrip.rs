//! Windows hardware round-trip: WASAPI loopback system audio → Opus encode →
//! Opus decode → cpal playback.
//!
//! `#[ignore]` per the repo convention — it needs a real audio output device.
//! Unlike the macOS (ScreenCaptureKit) and Linux (PipeWire monitor) backends,
//! WASAPI loopback delivers **no buffers while the endpoint is digitally idle**,
//! so audio must actually be playing on the default render device for capture to
//! produce frames. Run with something audible playing:
//!
//! ```text
//! cargo test -p tether-audio --test windows_audio_roundtrip -- --ignored --nocapture
//! ```
#![cfg(target_os = "windows")]

use std::time::{Duration, Instant};

use tether_audio::{capture, AudioPlayer, OpusConfig, OpusDecoder, OpusEncoder};

#[test]
#[ignore = "requires Windows audio output device with audio actively playing (WASAPI loopback is silent-gated); run with: cargo test -p tether-audio -- --ignored"]
fn windows_capture_encode_decode_playback_roundtrip() {
    let cfg = OpusConfig::default();

    let handle = capture::start(cfg).expect("start Windows WASAPI loopback capture");
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
        // The converter always emits at the configured rate/channels regardless
        // of the device mix format, so frames match what the encoder expects.
        assert_eq!(frame.channels, cfg.channels, "capture channel count");
        assert_eq!(frame.sample_rate, cfg.sample_rate, "capture sample rate");
        captured += 1;
        for pkt in enc.encode(&frame.samples).expect("opus encode") {
            let pcm = dec.decode(&pkt).expect("opus decode");
            assert_eq!(pcm.channels, cfg.channels);
            sink.submit(&pcm);
            played += 1;
        }
    }
    handle.stop();

    let (underruns, dropped_samples, _buffered) = sink.stats();
    eprintln!(
        "captured={captured} played={played} underruns={underruns} dropped_samples={dropped_samples}"
    );
    assert!(
        captured > 0,
        "captured no audio frames from WASAPI loopback in 2s (is audio playing?)"
    );
    assert!(played > 0, "no frames survived encode->decode to playback");
}
