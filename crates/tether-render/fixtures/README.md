# Render-test fixtures

## `colorbars_hevc_yuv444_*.idr` — colour-decode validation

Single-IDR HEVC 4:4:4 bitstreams of a four-bar **red / green / blue /
white** pattern, used by `iosurface_test.rs` (and, as they land, the
Linux/Windows render tests) to assert the decode → import → shader path
reproduces **colour** — not just that a frame renders. The white bar is
the regression guard for hue casts on neutrals (the "colors are wrong"
class of bug); red↔blue catch a Cb/Cr swap; green catches a dropped or
inverted chroma channel.

These are the **shared cross-platform fixture pattern**: every
platform's colour test decodes the same bitstream and runs the same
`assert_colorbars` check, so coverage is uniform.

Encoded off-platform with libx265 (VideoToolbox can't encode Main444),
BT.709 limited range to match the renderer's hard-pinned colour spec.

### Regenerating

Needs ffmpeg with libx265 built `--enable-rext` (Homebrew's bottle is;
`ffprobe` should report `profile=Rext`). From this directory:

```sh
gen() {  # $1 = pix_fmt, $2 = output, $3 = bar width, $4 = height
  ffmpeg -y -hide_banner -loglevel error \
    -f lavfi -i color=c=red:s=$3x$4:r=30 \
    -f lavfi -i color=c=lime:s=$3x$4:r=30 \
    -f lavfi -i color=c=blue:s=$3x$4:r=30 \
    -f lavfi -i color=c=white:s=$3x$4:r=30 \
    -filter_complex "[0:v][1:v][2:v][3:v]hstack=inputs=4,scale=out_color_matrix=bt709:out_range=tv,format=$1" \
    -frames:v 1 -c:v libx265 -preset ultrafast -x265-params "log-level=error" \
    -pix_fmt "$1" -f hevc "$2"
}
gen yuv444p     colorbars_hevc_yuv444_8bit.idr            32 128
gen yuv444p10le colorbars_hevc_yuv444_10bit.idr           32 128
gen yuv444p10le colorbars_hevc_yuv444_10bit_1920x1200.idr 480 1200
```

The 1920×1200 cell exists because 128×128 is too small to expose a
chroma-plane stride / row-pitch alignment bug — those only mis-read at
widths that aren't a clean multiple of the hardware alignment.

## `test-pattern-3360x2100.png`

Photographic + geometric pattern used by the Linux dma-buf round-trip
harness (`dmabuf_test.rs`) for SSIM / geometric-residual checks.
