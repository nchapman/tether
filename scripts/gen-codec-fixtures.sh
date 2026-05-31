#!/usr/bin/env bash
# Regenerate the committed H.264 test fixtures under crates/tether-codec/testdata.
#
# Why a fixture instead of encoding on the fly: tether links its own LGPL
# FFmpeg (no libx264/libx265), so there is no software H.264 encoder available
# to tests. The decoder paths (native H.264 decode, VAAPI decode) are still
# present, so we feed them a pre-encoded Annex-B bitstream instead.
#
# This script needs a *system* ffmpeg built with libx264 (any recent build).
# It is a maintenance tool, not part of the build — run it only when a fixture
# needs to change, then commit the regenerated .bin.
#
# Fixture format: a sequence of access units, each prefixed with a little-endian
# u32 byte length. Each AU is self-contained (AUD + SPS + PPS + IDR, via
# x264 repeat-headers), so tests can submit them one-by-one to a parser-less
# decoder and every AU decodes on its own.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/crates/tether-codec/testdata"
RAW="$(mktemp --suffix=.h264)"
trap 'rm -f "${RAW}"' EXIT

command -v ffmpeg >/dev/null || { echo "need a system ffmpeg with libx264" >&2; exit 1; }
ffmpeg -hide_banner -h encoder=libx264 >/dev/null 2>&1 || { echo "system ffmpeg lacks libx264" >&2; exit 1; }

# 320x240 / 4 all-intra frames / no B-frames / zerolatency. AUDs delimit access
# units; repeat-headers makes each AU independently decodable.
ffmpeg -hide_banner -y -f lavfi -i "testsrc=size=320x240:rate=30:duration=1" \
  -frames:v 4 -c:v libx264 -pix_fmt yuv420p -g 1 -bf 0 -tune zerolatency \
  -x264-params "aud=1:repeat-headers=1:annexb=1" -f h264 "${RAW}"

python3 - "${RAW}" "${OUT_DIR}/h264_320x240_intra.bin" <<'PY'
import re, struct, sys
raw, dst = sys.argv[1], sys.argv[2]
data = open(raw, 'rb').read()
starts = [m.start() for m in re.finditer(b'\x00\x00\x00\x01\x09', data)]  # AUD NAL
assert starts and starts[0] == 0, f"unexpected AUD layout: {starts[:4]}"
bounds = starts + [len(data)]
out = bytearray()
n = 0
for i in range(len(starts)):
    au = data[bounds[i]:bounds[i+1]]
    out += struct.pack('<I', len(au)) + au
    n += 1
open(dst, 'wb').write(out)
print(f"wrote {n} access units, {len(out)} bytes -> {dst}")
PY
