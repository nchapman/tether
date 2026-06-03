//! Windows system-audio capture via WASAPI loopback.
//!
//! WASAPI is the native path to system *output* audio on Windows: opening the
//! default render endpoint in shared mode with `AUDCLNT_STREAMFLAGS_LOOPBACK`
//! turns an `IAudioCaptureClient` into a tap on exactly "what's playing" — the
//! mirror of the Linux PipeWire sink monitor and the macOS ScreenCaptureKit
//! audio path. cpal (our playback abstraction) doesn't expose loopback, so we
//! bind the Core Audio interfaces directly with the same `windows` crate the
//! host video path already uses.
//!
//! A dedicated thread runs the capture loop. Unlike PipeWire/SCK — where the
//! framework hands us f32 at the rate/channels we ask for — a loopback client
//! is *pinned to the render endpoint's mix format* (passing any other format to
//! `Initialize` fails). The mix format is virtually always 48 kHz stereo
//! 32-bit-float, but consumer hardware (USB DACs, 44.1 kHz endpoints, surround
//! receivers) varies, so we convert the captured PCM to the Opus encoder's
//! configured f32/rate/channels before emitting frames — see [`FormatConverter`].
//!
//! Loopback delivers no buffers while the endpoint is digitally idle; that's
//! fine, the client's jitter buffer conceals the gap. We poll (rather than run
//! event-driven) because event-driven loopback doesn't signal during silence.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::{bounded, Sender, TrySendError};
use windows::core::GUID;
use windows::Win32::Media::Audio::{
    eConsole, eRender, IAudioCaptureClient, IAudioClient, IMMDeviceEnumerator, MMDeviceEnumerator,
    AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK, WAVEFORMATEX, WAVEFORMATEXTENSIBLE,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL,
    COINIT_MULTITHREADED,
};

use crate::capture::{AudioCaptureHandle, CaptureError};
use crate::{AudioFrame, OpusConfig};

/// ~80 ms of 10 ms frames. Capture should always be drained promptly; a short
/// queue keeps capture-side latency tight and drops the newest on overflow.
const CAPTURE_CHANNEL_DEPTH: usize = 8;

/// REFERENCE_TIME is in 100 ns units; one millisecond is 10 000 of them.
const REFTIMES_PER_MILLISEC: i64 = 10_000;

/// Shared-mode loopback buffer length. The capture period is fixed by the
/// engine; this only sizes the ring we read from, so a comfortable margin keeps
/// us from overrunning if a poll is late without adding steady-state latency.
const BUFFER_DURATION_MS: i64 = 200;

/// How often we poll for captured packets when the endpoint is idle. Matches the
/// 10 ms Opus frame so a busy stream is read about a frame at a time.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// `pdwFlags` bit set when the engine dropped frames before this buffer (the
/// captured data is valid but discontinuous with the previous buffer).
const AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY: u32 = 0x1;

/// `pdwFlags` bit set when the engine wrote a silent buffer; the sample memory
/// is then undefined, so we synthesize zeros instead of reading it.
const AUDCLNT_BUFFERFLAGS_SILENT: u32 = 0x2;

/// Mix-format `wFormatTag`s / EXTENSIBLE subformats we can read.
const WAVE_FORMAT_PCM: u16 = 0x0001;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;
const SUBTYPE_PCM: GUID = GUID::from_u128(0x0000_0001_0000_0010_8000_00aa_0038_9b71);
const SUBTYPE_IEEE_FLOAT: GUID = GUID::from_u128(0x0000_0003_0000_0010_8000_00aa_0038_9b71);

/// Start a WASAPI loopback capture on a dedicated thread.
pub fn start(cfg: OpusConfig) -> Result<AudioCaptureHandle, CaptureError> {
    let (tx, rx) = bounded::<AudioFrame>(CAPTURE_CHANNEL_DEPTH);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = Arc::clone(&stop);

    let thread = thread::Builder::new()
        .name("tether-audio-capture".into())
        .spawn(move || {
            if let Err(e) = run(cfg, &tx, &stop_thread) {
                tracing::error!(error = %e, "Windows audio capture ended with error");
            }
        })
        .map_err(|e| CaptureError::Backend(format!("spawn capture thread: {e}")))?;

    Ok(AudioCaptureHandle::from_parts(rx, stop, thread))
}

/// Wrap a `windows::core::Error` in our backend error with call-site context.
fn winerr(ctx: &str, e: windows::core::Error) -> CaptureError {
    CaptureError::Backend(format!("{ctx}: {e}"))
}

fn run(cfg: OpusConfig, tx: &Sender<AudioFrame>, stop: &AtomicBool) -> Result<(), CaptureError> {
    // The MMDevice/IAudioClient APIs require COM on this thread. CoInitializeEx
    // returns S_OK (first init on this thread) or S_FALSE (already initialized
    // with the same model) — both increment the COM refcount and both require a
    // paired CoUninitialize, so `is_ok()` (true for either) is exactly the
    // "should I uninit?" predicate. A hard error (e.g. RPC_E_CHANGED_MODE) leaves
    // the refcount untouched, so we skip the uninit below.
    // SAFETY: first COM call on this freshly spawned thread; paired with the
    // CoUninitialize below when we initialized it.
    let com_initialized = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }.is_ok();

    let result = run_capture(cfg, tx, stop);

    if com_initialized {
        // SAFETY: balances the successful CoInitializeEx above; every COM object
        // created in `run_capture` has been dropped by now.
        unsafe { CoUninitialize() };
    }
    result
}

fn run_capture(
    cfg: OpusConfig,
    tx: &Sender<AudioFrame>,
    stop: &AtomicBool,
) -> Result<(), CaptureError> {
    // SAFETY: standard Core Audio activation sequence. Each call's success is
    // checked before the returned interface is used; the interfaces live only on
    // this thread for the duration of the capture loop.
    let (audio_client, capture_client, mix) = unsafe {
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
                .map_err(|e| winerr("create device enumerator", e))?;
        // Loopback taps the *render* endpoint (eRender); eConsole is the default
        // device games/media use, i.e. what the user actually hears.
        let device = enumerator
            .GetDefaultAudioEndpoint(eRender, eConsole)
            .map_err(|e| winerr("get default render endpoint", e))?;
        let audio_client: IAudioClient = device
            .Activate(CLSCTX_ALL, None)
            .map_err(|e| winerr("activate audio client", e))?;

        let pwfx = audio_client
            .GetMixFormat()
            .map_err(|e| winerr("get mix format", e))?;
        // Parse before the side-effecting Initialize: an unsupported/unparseable
        // mix format should bail before we initialize a client we'd immediately
        // discard. `parse_mix_format` only reads `pwfx`; free it on the error
        // path so the bail doesn't leak the CoTaskMemAlloc'd format.
        let mix = match parse_mix_format(pwfx) {
            Ok(mix) => mix,
            Err(e) => {
                CoTaskMemFree(Some(pwfx.cast::<c_void>()));
                return Err(e);
            }
        };
        // Loopback can only be initialized with the endpoint's own mix format, so
        // hand `pwfx` straight to Initialize. It must stay live across that call —
        // free only afterwards.
        let init_result = audio_client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK,
            BUFFER_DURATION_MS * REFTIMES_PER_MILLISEC,
            0,
            pwfx,
            None,
        );
        // GetMixFormat allocates with CoTaskMemAlloc; free it now that Initialize
        // is done reading it.
        CoTaskMemFree(Some(pwfx.cast::<c_void>()));
        init_result.map_err(|e| winerr("initialize loopback client", e))?;
        let capture_client: IAudioCaptureClient = audio_client
            .GetService()
            .map_err(|e| winerr("get capture client", e))?;
        audio_client
            .Start()
            .map_err(|e| winerr("start audio client", e))?;
        (audio_client, capture_client, mix)
    };

    let mut converter = FormatConverter::new(&mix, cfg.channels, cfg.sample_rate);
    tracing::info!(
        device_rate = mix.sample_rate,
        device_channels = mix.channels,
        device_format = ?mix.sample_format,
        out_rate = cfg.sample_rate,
        out_channels = cfg.channels,
        "Windows system audio capture started (WASAPI loopback)"
    );

    let result = capture_loop(&capture_client, &mut converter, &mix, cfg, tx, stop);

    // SAFETY: `audio_client` was successfully Start()ed above; stopping a started
    // client is always valid. Best-effort on shutdown.
    if let Err(e) = unsafe { audio_client.Stop() } {
        tracing::warn!(error = %e, "IAudioClient::Stop failed during shutdown");
    }
    result
}

/// Drain captured packets until told to stop or the consumer goes away.
fn capture_loop(
    capture_client: &IAudioCaptureClient,
    converter: &mut FormatConverter,
    mix: &MixFormat,
    cfg: OpusConfig,
    tx: &Sender<AudioFrame>,
    stop: &AtomicBool,
) -> Result<(), CaptureError> {
    let mut dropped = false;
    let mut logged_discontinuity = false;
    while !stop.load(Ordering::Relaxed) {
        let mut got_packet = false;
        loop {
            // SAFETY: capture_client is a live started loopback service.
            let packet_frames = unsafe { capture_client.GetNextPacketSize() }
                .map_err(|e| winerr("get next packet size", e))?;
            if packet_frames == 0 {
                break;
            }
            got_packet = true;

            let mut data: *mut u8 = std::ptr::null_mut();
            let mut frames: u32 = 0;
            let mut flags: u32 = 0;
            // SAFETY: out-params are valid locals; on success WASAPI fills `data`
            // with `frames * nBlockAlign` readable bytes that stay valid until
            // ReleaseBuffer below.
            unsafe {
                capture_client
                    .GetBuffer(&mut data, &mut frames, &mut flags, None, None)
                    .map_err(|e| winerr("get capture buffer", e))?;
            }

            // Engine dropped frames before this buffer: data is still valid, but
            // there's now a gap in the timeline. We can't backfill it, so just
            // surface it once for diagnosis (mirrors the Linux format-divergence log).
            if flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY != 0 && !logged_discontinuity {
                logged_discontinuity = true;
                tracing::warn!("WASAPI reported an audio capture glitch (data discontinuity)");
            }

            let frame_count = frames as usize;
            let silent = flags & AUDCLNT_BUFFERFLAGS_SILENT != 0 || data.is_null();
            let interleaved = if silent {
                // Silent buffer memory is undefined; emit zeros of the right span
                // so the playback clock keeps advancing without reading garbage.
                vec![0.0f32; frame_count * mix.channels]
            } else {
                // SAFETY: `data` points to `frames * block_align` valid bytes per
                // the GetBuffer contract; we read exactly that span.
                let bytes =
                    unsafe { std::slice::from_raw_parts(data, frame_count * mix.block_align) };
                decode_to_f32(bytes, mix.sample_format, mix.bytes_per_sample)
            };

            // SAFETY: pairs with the GetBuffer above; `frames` is the count it gave us.
            unsafe {
                capture_client
                    .ReleaseBuffer(frames)
                    .map_err(|e| winerr("release capture buffer", e))?;
            }

            let samples = converter.process(interleaved);
            if samples.is_empty() {
                continue;
            }
            let frame = AudioFrame::new(cfg.sample_rate, cfg.channels, samples);
            match tx.try_send(frame) {
                Ok(()) => dropped = false,
                Err(TrySendError::Full(_)) => {
                    if !dropped {
                        dropped = true;
                        tracing::warn!(
                            "audio capture channel full; dropping frames (consumer behind)"
                        );
                    }
                }
                Err(TrySendError::Disconnected(_)) => {
                    tracing::info!("audio capture receiver dropped; stopping loopback capture");
                    return Ok(());
                }
            }
        }
        if !got_packet {
            // Endpoint idle (no audio playing) or caught up; back off briefly.
            thread::sleep(POLL_INTERVAL);
        }
    }
    Ok(())
}

/// The decoded shape of the render endpoint's mix format.
#[derive(Clone, Copy, Debug)]
struct MixFormat {
    sample_format: SampleFormat,
    channels: usize,
    sample_rate: u32,
    /// Bytes per single sample (e.g. 4 for f32).
    bytes_per_sample: usize,
    /// Bytes per interleaved frame (`bytes_per_sample * channels`).
    block_align: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleFormat {
    F32,
    I16,
    I32,
}

/// Read the relevant fields out of the engine's `WAVEFORMATEX`, resolving the
/// `WAVE_FORMAT_EXTENSIBLE` indirection that shared-mode mix formats use.
///
/// Everything the driver tells us is validated here (the boundary), so the hot
/// path can trust the returned [`MixFormat`]: the format tag/bit-depth maps to a
/// `SampleFormat` we can actually decode, the EXTENSIBLE struct is large enough
/// to dereference, the container holds exactly the valid bits we read, and
/// `block_align == channels * bytes_per_sample` so frame framing is exact.
fn parse_mix_format(pwfx: *const WAVEFORMATEX) -> Result<MixFormat, CaptureError> {
    // SAFETY: `pwfx` is the engine-allocated mix format from GetMixFormat, which
    // is always at least a full `WAVEFORMATEX`. WAVEFORMATEX is `packed(1)`, so
    // copy each field out by value here — `wfx.field` by reference (e.g. inside
    // `format!`) would form an unaligned reference and fail to compile.
    let wfx = unsafe { *pwfx };
    let format_tag = wfx.wFormatTag;
    let bits = wfx.wBitsPerSample;
    let channels = wfx.nChannels as usize;
    let sample_rate = wfx.nSamplesPerSec;
    let block_align = wfx.nBlockAlign as usize;
    let cb_size = wfx.cbSize;

    let sample_format = if format_tag == WAVE_FORMAT_EXTENSIBLE {
        // `cbSize` counts the bytes after the WAVEFORMATEX header; EXTENSIBLE
        // adds a 22-byte tail. Guard before the wider cast so a malformed driver
        // format (tag says EXTENSIBLE, allocation too small) can't over-read.
        if cb_size < 22 {
            return Err(CaptureError::Backend(format!(
                "WAVE_FORMAT_EXTENSIBLE with cbSize={cb_size} (< 22); allocation too small"
            )));
        }
        // SAFETY: tag is EXTENSIBLE and cbSize >= 22, so the allocation is a full
        // WAVEFORMATEXTENSIBLE. The struct is `packed(1)`, so read its fields via
        // `addr_of!` + `read_unaligned` rather than forming references to them.
        // `Samples` is a union; `wValidBitsPerSample` is the member for a
        // PCM/float subformat.
        let ext = pwfx as *const WAVEFORMATEXTENSIBLE;
        let sub_format = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!((*ext).SubFormat)) };
        let valid_bits = unsafe {
            std::ptr::read_unaligned(std::ptr::addr_of!((*ext).Samples.wValidBitsPerSample))
        };
        // Match the *valid* bits against the container (`bits`): an exact pair is
        // the only thing `decode_to_f32` reads correctly. 24-in-32 PCM (valid 24,
        // container 32) is left out deliberately — decoding it as I32 would be
        // 256x too loud, so we reject rather than corrupt. (Shared-mode mix
        // formats are float in practice, so this path is defensive.)
        match (sub_format, valid_bits, bits) {
            (SUBTYPE_IEEE_FLOAT, 32, 32) => SampleFormat::F32,
            (SUBTYPE_PCM, 16, 16) => SampleFormat::I16,
            (SUBTYPE_PCM, 32, 32) => SampleFormat::I32,
            _ => return Err(unsupported_format(valid_bits)),
        }
    } else {
        match (format_tag, bits) {
            (WAVE_FORMAT_IEEE_FLOAT, 32) => SampleFormat::F32,
            (WAVE_FORMAT_PCM, 16) => SampleFormat::I16,
            (WAVE_FORMAT_PCM, 32) => SampleFormat::I32,
            _ => return Err(unsupported_format(bits)),
        }
    };

    if channels == 0 {
        return Err(CaptureError::Backend(
            "mix format reports 0 channels".into(),
        ));
    }
    if sample_rate == 0 {
        return Err(CaptureError::Backend(
            "mix format reports 0 Hz sample rate".into(),
        ));
    }
    let bytes_per_sample = bits as usize / 8;
    // The hot path reads `frame_count * block_align` bytes and re-derives the
    // sample count via `bytes_per_sample`; if the driver's stride doesn't equal
    // `channels * bytes_per_sample`, those two disagree and frames misalign.
    // Reject up front rather than silently corrupt (also rules out `block_align == 0`).
    let expected_block_align = channels * bytes_per_sample;
    if block_align != expected_block_align {
        return Err(CaptureError::Backend(format!(
            "mix format block_align={block_align} != channels*bytes_per_sample={expected_block_align}"
        )));
    }
    Ok(MixFormat {
        sample_format,
        channels,
        sample_rate,
        bytes_per_sample,
        block_align,
    })
}

fn unsupported_format(bits: u16) -> CaptureError {
    CaptureError::Backend(format!(
        "unsupported render mix format ({bits}-bit; expected 32-bit float, 16- or 32-bit PCM)"
    ))
}

/// Decode a tightly packed interleaved sample buffer in the endpoint's native
/// format into interleaved `f32`. Integer PCM is normalized to `[-1.0, 1.0)`.
#[allow(
    clippy::cast_precision_loss,
    reason = "int-PCM->f32 normalization is inherently lossy and intended"
)]
fn decode_to_f32(bytes: &[u8], format: SampleFormat, bytes_per_sample: usize) -> Vec<f32> {
    bytes
        .chunks_exact(bytes_per_sample)
        .map(|s| match format {
            SampleFormat::F32 => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
            SampleFormat::I16 => f32::from(i16::from_le_bytes([s[0], s[1]])) / 32_768.0,
            SampleFormat::I32 => {
                i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0
            }
        })
        .collect()
}

/// Converts interleaved f32 from the endpoint's channel count + rate to the Opus
/// encoder's. Channel remix is stateless; resampling carries state across calls.
struct FormatConverter {
    in_channels: usize,
    out_channels: usize,
    resampler: Option<LinearResampler>,
}

impl FormatConverter {
    fn new(mix: &MixFormat, out_channels: u8, out_rate: u32) -> Self {
        let out_channels = out_channels as usize;
        // Skip the resampler entirely when rates already match (the common
        // 48 kHz case), so that path stays a verbatim copy with no startup skew.
        let resampler = (mix.sample_rate != out_rate)
            .then(|| LinearResampler::new(out_channels, mix.sample_rate, out_rate));
        Self {
            in_channels: mix.channels,
            out_channels,
            resampler,
        }
    }

    fn process(&mut self, interleaved: Vec<f32>) -> Vec<f32> {
        // Take the decoded buffer by value so the common 48 kHz/stereo path
        // (no remix, no resample) moves it straight through instead of copying.
        let remixed = if self.in_channels == self.out_channels {
            interleaved
        } else {
            remix(&interleaved, self.in_channels, self.out_channels)
        };
        let mut out = match &mut self.resampler {
            Some(r) => r.process(&remixed),
            None => remixed,
        };
        // The encoder (and OS mixers downstream of the client) expect normalized
        // PCM, but WASAPI's float mix can exceed ±1.0 (apps may output >0 dBFS),
        // and a surround downmix sums channels. Clamp so Opus never sees
        // out-of-range input — mirrors the decoder's clamp on the far side.
        for s in &mut out {
            *s = s.clamp(-1.0, 1.0);
        }
        out
    }
}

/// -3 dB (≈ 1/√2), the ITU-R BS.775 attenuation for folding the center and
/// surround channels into a stereo downmix.
const DOWNMIX_GAIN: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Remix interleaved f32 from `in_channels` to `out_channels`. v1 only ever
/// targets mono or stereo. Stereo: a stereo source passes through, mono
/// duplicates, and a standard 5.1/7.1 source is downmixed per BS.775 (front L/R
/// at unity, center and surrounds at −3 dB) so dialogue in the center channel
/// isn't lost. Mono: a stereo source averages.
fn remix(interleaved: &[f32], in_ch: usize, out_ch: usize) -> Vec<f32> {
    if in_ch == out_ch {
        return interleaved.to_vec();
    }
    let frames = interleaved.len() / in_ch.max(1);
    let mut out = Vec::with_capacity(frames * out_ch);
    for f in 0..frames {
        let src = &interleaved[f * in_ch..f * in_ch + in_ch];
        match out_ch {
            1 => out.push(if in_ch == 1 {
                src[0]
            } else {
                0.5 * (src[0] + src[1])
            }),
            2 => {
                let (l, r) = downmix_stereo(src);
                out.push(l);
                out.push(r);
            }
            _ => {
                // Not reachable in v1 (encoder caps channels at 2); copy what we
                // can and zero-fill so the frame stays well-formed.
                for c in 0..out_ch {
                    out.push(src.get(c).copied().unwrap_or(0.0));
                }
            }
        }
    }
    out
}

/// Downmix one source frame to a stereo (L, R) pair.
///
/// We assume WASAPI's canonical interleave order (FL, FR, FC, LFE, BL, BR, SL,
/// SR) rather than parsing `dwChannelMask`, which holds for every standard
/// `KSAUDIO_SPEAKER_*` layout. Only the two ubiquitous surround layouts (5.1 =
/// 6ch, 7.1 = 8ch) get the full BS.775 fold-in; any other channel count we don't
/// recognize falls back to the front L/R pair (FL/FR are always indices 0/1) so
/// an exotic layout degrades to plausible stereo rather than a wrong mapping.
fn downmix_stereo(src: &[f32]) -> (f32, f32) {
    match src.len() {
        1 => (src[0], src[0]),
        // 5.1: FL FR FC LFE BL BR (LFE dropped).
        6 => (
            src[0] + DOWNMIX_GAIN * (src[2] + src[4]),
            src[1] + DOWNMIX_GAIN * (src[2] + src[5]),
        ),
        // 7.1: FL FR FC LFE BL BR SL SR (LFE dropped).
        8 => (
            src[0] + DOWNMIX_GAIN * (src[2] + src[4] + src[6]),
            src[1] + DOWNMIX_GAIN * (src[2] + src[5] + src[7]),
        ),
        // Stereo, or an unrecognized layout: front L/R.
        _ => (src[0], src[1]),
    }
}

/// Streaming linear resampler over interleaved f32. Carries the previous call's
/// last frame so interpolation is seamless across WASAPI buffer boundaries.
///
/// Linear interpolation is modest quality, but the rate-conversion path is only
/// hit on non-48 kHz endpoints (uncommon); the 48 kHz fast path bypasses it
/// entirely. A polyphase resampler would be the upgrade if quality complaints
/// surface on 44.1 kHz hardware.
struct LinearResampler {
    channels: usize,
    /// Input frames advanced per output frame (`in_rate / out_rate`).
    step: f64,
    /// Fractional read position; `0` = the carried `prev` frame, `1` = the first
    /// frame of the current input buffer.
    pos: f64,
    /// Last input frame from the previous call (the sample just before `pos = 1`).
    prev: Vec<f32>,
}

impl LinearResampler {
    fn new(channels: usize, in_rate: u32, out_rate: u32) -> Self {
        Self {
            channels,
            step: f64::from(in_rate) / f64::from(out_rate),
            // Start at the first real frame; the synthetic `prev` (zeros) is never
            // read on the first call because `pos` begins at 1.
            pos: 1.0,
            prev: vec![0.0; channels],
        }
    }

    // `pos` is bounded to `[0, n)` by the loop guard, so `floor()` is a
    // non-negative index below `n` and `frac` is in `[0, 1)` — both casts are
    // exact for the ranges they ever see.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "resampler position is bounded to [0, n) and frac to [0, 1)"
    )]
    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let ch = self.channels;
        let n = input.len() / ch;
        if n == 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        // Virtual stream E indexed 0..=n: E[0] = prev, E[k] = input frame k-1.
        // Emit while a right neighbour exists inside the current buffer.
        while self.pos < n as f64 {
            let i = self.pos.floor() as usize; // 0..=n-1
            let frac = (self.pos - i as f64) as f32;
            for c in 0..ch {
                let left = if i == 0 {
                    self.prev[c]
                } else {
                    input[(i - 1) * ch + c]
                };
                let right = input[i * ch + c];
                out.push(left + (right - left) * frac);
            }
            self.pos += self.step;
        }
        // Rebase position onto the next buffer and carry this buffer's last frame.
        self.pos -= n as f64;
        let last = (n - 1) * ch;
        self.prev.clear();
        self.prev.extend_from_slice(&input[last..last + ch]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_f32_round_trips_samples() {
        let mut bytes = Vec::new();
        for v in [0.25f32, -0.5, 1.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = decode_to_f32(&bytes, SampleFormat::F32, 4);
        assert_eq!(out, vec![0.25, -0.5, 1.0]);
    }

    #[test]
    fn decode_i16_normalizes_to_unit_range() {
        let mut bytes = Vec::new();
        for v in [i16::MAX, i16::MIN, 0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = decode_to_f32(&bytes, SampleFormat::I16, 2);
        assert!((out[0] - 0.999_97).abs() < 1e-3);
        assert_eq!(out[1], -1.0);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn remix_passthrough_when_equal() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(remix(&input, 2, 2), input);
    }

    #[test]
    fn remix_mono_to_stereo_duplicates() {
        let input = vec![0.3, -0.7];
        assert_eq!(remix(&input, 1, 2), vec![0.3, 0.3, -0.7, -0.7]);
    }

    #[test]
    fn remix_5_1_to_stereo_folds_center_and_rears() {
        // One 5.1 frame (FL, FR, FC, LFE, BL, BR). LFE is dropped; FC and the
        // back pair fold into L/R at -3 dB (BS.775).
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let out = remix(&input, 6, 2);
        let l = 0.1 + DOWNMIX_GAIN * (0.3 + 0.5);
        let r = 0.2 + DOWNMIX_GAIN * (0.3 + 0.6);
        assert!((out[0] - l).abs() < 1e-6, "L={}, want {l}", out[0]);
        assert!((out[1] - r).abs() < 1e-6, "R={}, want {r}", out[1]);
    }

    #[test]
    fn remix_7_1_to_stereo_folds_center_rears_and_sides() {
        // 7.1 frame (FL, FR, FC, LFE, BL, BR, SL, SR).
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let out = remix(&input, 8, 2);
        let l = 0.1 + DOWNMIX_GAIN * (0.3 + 0.5 + 0.7);
        let r = 0.2 + DOWNMIX_GAIN * (0.3 + 0.6 + 0.8);
        assert!((out[0] - l).abs() < 1e-6);
        assert!((out[1] - r).abs() < 1e-6);
    }

    #[test]
    fn remix_unrecognized_layout_falls_back_to_front_pair() {
        // A 3-channel source we don't have a BS.775 mapping for: take FL/FR.
        let input = vec![0.1, 0.2, 0.3];
        assert_eq!(remix(&input, 3, 2), vec![0.1, 0.2]);
    }

    #[test]
    fn remix_stereo_to_mono_averages() {
        let input = vec![0.4, 0.6];
        assert_eq!(remix(&input, 2, 1), vec![0.5]);
    }

    #[test]
    fn resampler_doubles_frame_count_at_2x_in_steady_state() {
        // 24 kHz -> 48 kHz: step 0.5, so ~2 output frames per input frame.
        let mut r = LinearResampler::new(1, 24_000, 48_000);
        // Prime, then measure a steady-state buffer (first call drops a leading
        // partial frame by design).
        let _ = r.process(&[0.0, 1.0, 2.0, 3.0]);
        let out = r.process(&[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(out.len(), 8, "4 input frames at 2x -> 8 output frames");
        // A monotonically rising ramp stays monotonic through interpolation.
        for w in out.windows(2) {
            assert!(w[1] >= w[0], "ramp must stay monotonic: {out:?}");
        }
    }

    #[test]
    fn resampler_halves_frame_count_at_half_rate() {
        // 48 kHz -> 24 kHz: step 2.0, so ~half as many output frames.
        let mut r = LinearResampler::new(1, 48_000, 24_000);
        let _ = r.process(&[0.0, 1.0, 2.0, 3.0]);
        let out = r.process(&[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(out.len(), 2, "4 input frames at 0.5x -> 2 output frames");
    }

    #[test]
    fn resampler_preserves_stereo_interleaving() {
        let mut r = LinearResampler::new(2, 24_000, 48_000);
        // Two stereo frames: L rising, R falling.
        let out = r.process(&[0.0, 1.0, 1.0, 0.0]);
        assert!(out.len() % 2 == 0, "output stays frame-aligned");
    }

    /// Downsampling (`step > 1`) across many *small* buffers must keep `pos`
    /// bounded (it's a phase accumulator, not a running offset) and converge to
    /// the correct long-run output rate — the case a single-buffer test misses.
    #[test]
    fn resampler_downsampling_stays_bounded_over_many_small_buffers() {
        // 96 kHz -> 48 kHz, step = 2.0, fed one frame at a time.
        let mut r = LinearResampler::new(1, 96_000, 48_000);
        let mut total_out = 0usize;
        let buffers = 10_000;
        for i in 0..buffers {
            let out = r.process(&[i as f32]);
            total_out += out.len();
            // The invariant the index math relies on: `pos < 1` would be ideal,
            // but the contract that prevents drift/OOB is simply that `pos` never
            // runs away. With step=2 and 1-frame buffers it stays well under 3.
            assert!(
                r.pos.is_finite() && (0.0..3.0).contains(&r.pos),
                "pos drifted out of bounds: {}",
                r.pos
            );
        }
        // ~half as many output frames as input (one buffer of startup slack).
        let expected = buffers / 2;
        assert!(
            total_out.abs_diff(expected) <= 1,
            "downsampled output rate drifted: got {total_out}, want ~{expected}"
        );
    }

    /// Interpolation must be seamless across buffer boundaries: resampling a ramp
    /// split into two calls equals resampling the joined ramp in one call (modulo
    /// the documented one-frame startup), proving the carried `prev` frame stitches
    /// the seam correctly rather than restarting at each buffer.
    #[test]
    fn resampler_is_continuous_across_buffer_boundaries() {
        let ramp: Vec<f32> = (0..16).map(|i| i as f32).collect();

        let mut whole = LinearResampler::new(1, 24_000, 48_000);
        let out_whole = whole.process(&ramp);

        let mut split = LinearResampler::new(1, 24_000, 48_000);
        let mut out_split = split.process(&ramp[..8]);
        out_split.extend(split.process(&ramp[8..]));

        assert_eq!(
            out_whole.len(),
            out_split.len(),
            "same input must yield the same number of output frames"
        );
        for (a, b) in out_whole.iter().zip(&out_split) {
            assert!((a - b).abs() < 1e-6, "seam diverged: {a} vs {b}");
        }
    }

    /// Out-of-range float (WASAPI's mix can exceed ±1.0) is clamped before it
    /// reaches the encoder.
    #[test]
    fn converter_clamps_out_of_range_samples() {
        let mix = MixFormat {
            sample_format: SampleFormat::F32,
            channels: 2,
            sample_rate: 48_000,
            bytes_per_sample: 4,
            block_align: 8,
        };
        let mut conv = FormatConverter::new(&mix, 2, 48_000);
        let out = conv.process(vec![2.5, -3.0, 0.5, -0.5]);
        assert_eq!(out, vec![1.0, -1.0, 0.5, -0.5]);
    }
}
