//! Linux hardware round-trip: PipeWire sink-monitor capture → Opus encode →
//! Opus decode → cpal playback.
//!
//! `#[ignore]` per the repo convention — it needs a running PipeWire daemon
//! with an active (non-suspended) default sink and an audio output device. Run
//! with:
//!
//! ```text
//! cargo test -p tether-audio --test linux_audio_roundtrip -- --ignored --nocapture
//! ```
//!
//! Play something audible on the host while it runs so the sink stays active
//! and capture is non-silent; the assertions only require that frames flow
//! through the whole chain, since CI machines are silent and may suspend the
//! sink.
#![cfg(target_os = "linux")]

use std::time::{Duration, Instant};

use tether_audio::{capture, AudioPlayer, OpusConfig, OpusDecoder, OpusEncoder};

#[test]
#[ignore = "requires a running PipeWire daemon with an active default sink + audio output device"]
fn linux_capture_encode_decode_playback_roundtrip() {
    let cfg = OpusConfig::default();

    let handle = capture::start(cfg).expect("start Linux system-audio capture");
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

    let (underruns, dropped_samples, _buffered) = sink.stats();
    eprintln!(
        "captured={captured} played={played} underruns={underruns} dropped_samples={dropped_samples}"
    );
    assert!(captured > 0, "captured no audio frames from PipeWire in 2s");
    assert!(played > 0, "no frames survived encode->decode to playback");
}
