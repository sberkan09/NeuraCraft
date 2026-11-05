#!/usr/bin/env bash
# Downloads RIFE v4 (ECCV 2022 IFNet; MIT, Megvii) as safetensors
# (TensorForger/RIFE-safetensors, ~12 MB) for `interpolate` / `video --interpolate`.
#   scripts/download_rife.sh [dir]
set -euo pipefail
DIR="${1:-data/interpolate}"
mkdir -p "$DIR"
OUT="$DIR/rife-v4-flownet.safetensors"
if [ -s "$OUT" ]; then echo "already have $OUT"; exit 0; fi
curl -fL --retry 3 -o "$OUT.part" https://huggingface.co/TensorForger/RIFE-safetensors/resolve/main/flownet.safetensors
mv "$OUT.part" "$OUT"
echo "Ready: $OUT. Try: ./build_and_run.sh interpolate $OUT frames/ smooth.gif --passes 2"
