# Investigation handoff: Linux client freezes during macOS → Linux LAN demo

**Status:** RESOLVED. Kept for the design rationale that produced the
hardening in commits `ec29aea`, `61d9616`, `de7c811`, `f8a7a55`, and
`1778e4e`. All three hypotheses below turned out to contribute; fixes
landed in roughly that order. Post-fix measurement showed zero
`decode_errs` / `idr_reqs` / `render_drops` on a 50 s Mac→Linux
session at 13 Mbps peak. The current robustness posture is summarised
in `docs/ARCHITECTURE.md#robustness`.

**Original status (kept verbatim below):** open, needs Linux-side investigation
**Symptom:** Mid-session freezes of both video AND mouse input simultaneously
**Environment:** macOS Apple Silicon host → Linux Intel Arc client, fast LAN (LAN ping <2ms, WiFi for at least one end)
**Confidence:** Symptom is real and reproducible; root cause uncertain.

---

## What we observed

Mac→Linux LAN demo works end-to-end (HEVC encode on macOS via VideoToolbox, HEVC decode on Linux via VAAPI, wgpu/Vulkan render). Encode/decode times are excellent (~0.7–1.5 ms each). RTT 5–15 ms typical.

Mid-session the picture freezes for several seconds at a time. Mouse cursor input from the client to the host also stops being applied during these freeze windows. They appear correlated — both come back at roughly the same time.

Sample client-side log slice (from a representative session at `~/Code/tether/client.log` on the mind machine):

```
17.560  ERROR ffmpeg: [hevc] Could not find ref with POC 169
17.611  ERROR ffmpeg: [hevc] Could not find ref with POC 172
17.671  ERROR ffmpeg: [hevc] Could not find ref with POC 1     ← GOP reset (IDR landed)
17.677  ERROR ffmpeg: [hevc] Could not find ref with POC 2     ← but the IDR's own fragments
...                                                                were dropped too — every
                                                                  subsequent P-frame fails
```

The recovery loop (auto-IDR on detected decode failure) IS firing — `decode_errs=26 idr_reqs=1` in the same stats window. Host responds with keyframe bursts (`kf_per_s=0.48`). But the IDR keyframes themselves are also losing fragments on the wire, so the decoder stays stuck.

Full client log: `mind:~/Code/tether/client.log`. Host log: `~/Drive/Code/tether/host.log` (on the Mac).

---

## Why this matters

Fragment loss this frequent on a quiet LAN is not expected. The application-level recovery (auto-IDR) is doing what it's designed to do, but it's papering over a real underlying issue. Before adding more bandaids (next-step plan is to carry IDR fragments on a reliable QUIC stream — see "Followup work" below), we should confirm whether the loss is genuine wire loss or whether something on the client is dropping packets before QUIC ever sees them.

The "video and mouse freeze together" symptom is the most diagnostic clue. Video flows over **unreliable QUIC datagrams**; mouse flows over a **reliable QUIC stream**. Different transport primitives. If both freeze simultaneously, the cause is either:
- **Below QUIC**: the actual network is dropping packets / WiFi is unhealthy / the OS UDP socket queue is overflowing
- **Above QUIC**: the client's tokio runtime is starving because some task is blocking and not yielding (other tasks, including the mouse-send task, can't make progress)

It is **not** likely to be a normal "single channel went silent" problem; that would freeze video OR mouse, not both.

---

## Hypotheses, ordered by likelihood

### 1. OS-level UDP receive buffer overflow on the Linux client *(most likely)*

Linux's default `SO_RCVBUF` for UDP sockets is ~256 KB. A H.265 keyframe at the bursty bitrate spikes we see (5–8 Mbps) can be several tens of KB; the receiver has to drain the socket buffer faster than the sender fills it or the kernel silently drops packets *before* quinn ever sees them. quinn's own datagram receive buffer (`TransportConfig::datagram_receive_buffer_size`, set to 8 MiB in `tether-transport/src/server.rs`) is downstream of the OS UDP queue and doesn't help if the kernel already dropped the packet.

If the client's recv task is briefly slow — for example during a decode-error storm where the new `av_log` → tracing bridge synchronously formats + writes a log line per ffmpeg message (see `crates/tether-codec/src/av_log.rs`) — the kernel UDP queue can overflow within milliseconds and we start losing fragments. This compounds: lost fragments cause more decode errors, which cause more log lines, which slow the recv task further.

**How to test (cheap, no code changes):**
```bash
# Before starting a session, snapshot the baseline:
cat /proc/net/snmp | grep -E '^Udp:'

# Run a session for ~30s, freeze a few times, close the client.

# Snapshot after:
cat /proc/net/snmp | grep -E '^Udp:'
```

Look at `RcvbufErrors` and `InErrors`. If either climbs significantly (more than a handful) during the session, the kernel is dropping packets at the UDP socket queue and this is the cause.

Also useful:
```bash
ss -unmp 'sport = :7654 or dport = :7654'   # while session is live
```
Look at the `Recv-Q` column — non-zero means the userspace is behind on draining.

**If confirmed, the fix has two layers:**
1. Bump `SO_RCVBUF` on the UDP socket. quinn uses tokio's `UdpSocket` underneath; the right place is at `Endpoint` construction in `tether-transport/src/client.rs`. Use `socket2::Socket::set_recv_buffer_size()` on the raw fd before handing to quinn, or set `EndpointConfig::default()` with explicit socket options. Aim for 8–16 MiB. The kernel ceiling is `/proc/sys/net/core/rmem_max` — may need raising via `sysctl -w net.core.rmem_max=33554432`.
2. Reduce per-packet work on the client recv path so the queue drains faster. Most plausibly: move the `av_log` → tracing formatting off the decoder thread. See "Followup work" item 2 below.

### 2. WiFi reliability *(plausible if either end is on WiFi)*

WiFi has bursty packet loss under contention (other networks, interference, BSS-load). A clean ping doesn't always show it because pings are tiny; video keyframes are big.

**How to test:**
```bash
# During a streaming session, run from mind:
ping -i 0.01 -c 3000 <host-mac-ip>
```
Look at the loss percentage and the latency-jitter pattern. Anything over 0.5% sustained loss is enough to explain the freezes.

**Cross-check:**
- Wire one or both machines (Ethernet) and repeat. If the freezes go away, WiFi is the cause.
- Run `iw dev <iface> link` on mind during a session and watch the bitrate / signal numbers.

**If confirmed:** the application-level fix is to make recovery from packet loss faster + cheaper. That's the reliable-IDR work in "Followup work" item 1.

### 3. Tokio runtime starvation on the client *(plausible, code-side)*

The client's recv loop in `apps/tether-client/src/main.rs` (the long `loop { match conn_recv.recv_datagram().await { ... } }` around line 295) processes each video fragment synchronously: defragment → `decoder.submit()` → `decoder.next_frame()` → render-channel send. All in one `.await`. tokio is cooperative — if this task takes >a few ms without yielding, other tasks on the same runtime don't run.

The new `av_log` bridge (`crates/tether-codec/src/av_log.rs`) installed an FFmpeg log callback that synchronously formats each libavcodec message and routes it through `tracing`. During a decode-error storm (the "Could not find ref" lines), we're doing the format + `tracing::error!` call N times per submitted packet, on the decoder's thread, which is the recv loop's thread. With a file-writing tracing subscriber, each call is a sync write.

That would explain: (a) recv loop falls behind → hypothesis 1 cascades; (b) the input-send task doesn't get scheduled → mouse stops moving.

**How to test:**
```bash
# Quiet down libavcodec entirely for a session and see if the freezes happen:
RUST_LOG=tether=info,ffmpeg=off ./target/release/tether-client ...
```
If the freezes don't happen with ffmpeg logging suppressed, the av_log bridge is on the critical path.

(`RUST_LOG=...ffmpeg=off` suppresses the `target: "ffmpeg"` records the av_log callback emits, but the callback itself still runs and bumps the counter — so auto-IDR still works. The cost saved is just the per-record formatting + write.)

**If confirmed:** the fix is to move the av_log callback's work off the decoder thread. The cheapest version: callback bumps the atomic counter (already done) and pushes the pre-formatted line into a small `mpsc::TrySend` channel; a separate thread drains the channel into tracing. If the channel is full, drop the line (atomic counter still bumps). The decoder never blocks on log writes.

---

## What's already been ruled out / confirmed

- **Cert / fingerprint mismatch** — not it. Cert persistence works; same fingerprint across runs.
- **Color spec dispatch** — fully working. Renderer picks the sRGB EOTF path correctly. Colors look right.
- **Auto-IDR signal path** — working. `decode_errs` increments, IDR requests fire, host responds.
- **Decoder API errors** — none. `decoder.submit()` / `next_frame()` always return `Ok`. The freezes happen entirely through the libavcodec-internal-skip-NALU path that the av_log bridge exists to surface.
- **HEVC interop** — solid. Apple `hevc_videotoolbox` → Intel `hevc_vaapi` round-trips clean when fragments aren't lost.

---

## Followup work (separate from this investigation)

These are real improvements regardless of what we find above, but should be informed by what we learn:

1. **Carry IDR fragments on a reliable QUIC stream.** Today every video fragment rides an unreliable datagram. If an IDR loses one fragment, the entire frame is dropped and the decoder stays stuck until another IDR arrives. Marking the keyframes as reliable would make loss-recovery deterministic (one round trip) instead of stochastic. Sketch: `VideoPacket::First` for a keyframe gets sent via `conn.send_uni` on a fresh stream; P-frames stay on datagrams. Receiver-side reassembly already keys on `frame_seq`. ~100 lines split between `tether-transport` and the host's fragmenter.

2. **Move av_log formatting off the decoder thread** (mentioned in hypothesis 3). Right thing to do regardless — synchronous tracing in an FFmpeg callback is a latency footgun.

3. **OS-level UDP buffer tuning at startup.** Even if the immediate freeze isn't caused by buffer overflow, raising `SO_RCVBUF` to ~8 MiB defensively is cheap and gives headroom for future higher-bitrate workloads (4K, HDR).

---

## Files to look at

- `crates/tether-transport/src/server.rs` — `transport_config()`, where `datagram_receive_buffer_size` is set
- `crates/tether-transport/src/client.rs` — endpoint construction (where to add `SO_RCVBUF` tuning)
- `crates/tether-codec/src/av_log.rs` — the FFmpeg log → tracing bridge (hypothesis 3)
- `apps/tether-client/src/main.rs` — the recv loop (around line 295, search for `recv_datagram`)
- `host.log` (Mac repo root) and `mind:~/Code/tether/client.log` — captured session output

## Reproduction

1. On Mac: `make release && ./target/release/tether-host 0.0.0.0:7654`
2. On Linux: `git pull && make release && ./target/release/tether-client <mac-ip>:7654 <fingerprint>`
3. Move the mouse around, type in some windows on the Mac to drive bitrate up. Freezes typically appear within ~30s.

The freeze windows are 1–10 seconds long. They correlate with bitrate spikes in the host's `send stats` log lines (`kbps_out=7000+`).
