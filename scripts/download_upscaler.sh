#!/usr/bin/env bash
# Downloads Real-ESRGAN's compact x4 model (realesr-general-x4v3, BSD-3-Clause,
# ~5 MB) and converts it to safetensors with scripts/convert_pth.py (no PyTorch
# needed, only numpy).
#   scripts/download_upscaler.sh [dir]
set -euo pipefail
DIR="${1:-data/upscale}"
mkdir -p "$DIR"
OUT="$DIR/realesr-general-x4v3.safetensors"
if [ -s "$OUT" ]; then echo "already have $OUT"; exit 0; fi
TMP="$(mktemp -d)"
curl -fL --retry 3 -o "$TMP/realesr-general-x4v3.pth" https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
python3 -I "$(dirname "$0")/convert_pth.py" "$TMP/realesr-general-x4v3.pth" "$OUT"
rm -rf "$TMP"
echo "Ready: $OUT. Try: ./build_and_run.sh upscale $OUT photo.png photo_x4.png"
