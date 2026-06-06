# Decode-probe fixtures

Single-IDR bitstreams used by `tether_codec::profile_probe` to verify
that this platform's hardware decoder can actually consume each profile
we might negotiate. Each fixture is 256×256 grey, one frame; tiny on
purpose so they ride in the binary without bloating it.

**Why 256×256 and not smaller.** NVIDIA's NVDEC has a per-codec minimum
coded size — HEVC is 144×144. A sub-minimum HEVC fixture (the original
128×128 set) makes NVDEC's `send_packet` fail with a bare `-1`, so an
NVIDIA host silently drops HEVC to the H.264 floor (H.264's NVDEC minimum
is far smaller, so its fixture decoded fine and masked the problem). 256
clears every hardware decoder's minimum with margin and matches the NVENC
probe canvas. Keep new fixtures at 256×256.

## Regenerating

Requires ffmpeg with libx264 + libx265 + libaom-av1 (libx265 must be
built with `--enable-rext`; Homebrew's `ffmpeg` formula is). From this
directory:

```sh
ffmpeg -y -hide_banner -loglevel warning \
  -f lavfi -i color=c=gray:s=256x256:r=30 -frames:v 1 \
  -c:v libx264 -profile:v baseline -preset ultrafast -pix_fmt yuv420p \
  -bsf:v h264_metadata -f h264 h264_yuv420_8bit.idr

ffmpeg -y -hide_banner -loglevel warning \
  -f lavfi -i color=c=gray:s=256x256:r=30 -frames:v 1 \
  -c:v libx265 -preset ultrafast -x265-params "log-level=error" \
  -pix_fmt yuv420p -f hevc hevc_yuv420_8bit.idr

ffmpeg -y -hide_banner -loglevel warning \
  -f lavfi -i color=c=gray:s=256x256:r=30 -frames:v 1 \
  -c:v libx265 -preset ultrafast -x265-params "log-level=error" \
  -pix_fmt yuv444p -f hevc hevc_yuv444_8bit.idr

ffmpeg -y -hide_banner -loglevel warning \
  -f lavfi -i color=c=gray:s=256x256:r=30 -frames:v 1 \
  -c:v libx265 -preset ultrafast -x265-params "log-level=error" \
  -pix_fmt yuv420p10le -f hevc hevc_yuv420_10bit.idr

ffmpeg -y -hide_banner -loglevel warning \
  -f lavfi -i color=c=gray:s=256x256:r=30 -frames:v 1 \
  -c:v libx265 -preset ultrafast -x265-params "log-level=error" \
  -pix_fmt yuv422p -f hevc hevc_yuv422_8bit.idr

ffmpeg -y -hide_banner -loglevel warning \
  -f lavfi -i color=c=gray:s=256x256:r=30 -frames:v 1 \
  -c:v libx265 -preset ultrafast -x265-params "log-level=error" \
  -pix_fmt yuv422p10le -f hevc hevc_yuv422_10bit.idr

ffmpeg -y -hide_banner -loglevel warning \
  -f lavfi -i color=c=gray:s=256x256:r=30 -frames:v 1 \
  -c:v libx265 -preset ultrafast -x265-params "log-level=error" \
  -pix_fmt yuv444p10le -f hevc hevc_yuv444_10bit.idr

ffmpeg -y -hide_banner -loglevel warning \
  -f lavfi -i color=c=gray:s=256x256:r=30 -frames:v 1 \
  -c:v libaom-av1 -crf 50 -b:v 0 -cpu-used 8 \
  -pix_fmt yuv420p -f obu av1_yuv420_8bit.idr

ffmpeg -y -hide_banner -loglevel warning \
  -f lavfi -i color=c=gray:s=256x256:r=30 -frames:v 1 \
  -c:v libaom-av1 -crf 50 -b:v 0 -cpu-used 8 \
  -pix_fmt yuv420p10le -f obu av1_yuv420_10bit.idr
```

Verify with `ffprobe`:

```sh
for f in *.idr; do
  echo "=== $f ==="
  ffprobe -hide_banner -v error -select_streams v:0 \
    -show_entries stream=codec_name,profile,pix_fmt,width,height "$f"
done
```

Expected:

- `h264_yuv420_8bit.idr` — `h264`, profile `Constrained Baseline`, `yuv420p`, 256×256
- `hevc_yuv420_8bit.idr` — `hevc`, profile `Main`, `yuv420p`, 256×256
- `hevc_yuv420_10bit.idr` — `hevc`, profile `Main 10`, `yuv420p10le`, 256×256
- `hevc_yuv422_8bit.idr` — `hevc`, profile `Rext`, `yuv422p`, 256×256
- `hevc_yuv422_10bit.idr` — `hevc`, profile `Rext`, `yuv422p10le`, 256×256
- `hevc_yuv444_8bit.idr` — `hevc`, profile `Rext`, `yuv444p`, 256×256
- `hevc_yuv444_10bit.idr` — `hevc`, profile `Rext`, `yuv444p10le`, 256×256
- `av1_yuv420_8bit.idr` — `av1`, profile `Main`, `yuv420p`, 256×256
- `av1_yuv420_10bit.idr` — `av1`, profile `Main`, `yuv420p10le`, 256×256

If the HEVC 4:4:4 fixture comes out as `Main` instead of `Rext`, the
linked libx265 wasn't built with `--enable-rext`. On macOS rebuild via
`brew reinstall ffmpeg` (the bottle includes Rext); on Linux check
`x265 --help | grep -i rext` and rebuild with the option enabled.

## Why the bitstream format matters

The probe submits the fixture to a fresh hardware decoder and demands
back a `Frame::Gpu` (i.e. a `VAAPI surface` on Linux or a
VideoToolbox-backed `CVPixelBuffer` on macOS). If the decoder falls
back to software for that profile, the result is `Frame::Cpu` and we
classify the profile as "not decodable" — even if the codec ID is
nominally supported. This is exactly the failure mode the old
construction-only probe couldn't detect.

Adding a new profile (10-bit, AV1, H.264 4:4:4, …) means generating a
fixture here and extending `fixture_for` in
`crates/tether-codec/src/profile_probe.rs`.
