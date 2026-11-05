#!/usr/bin/env bash
# Downloads Depth Anything V2 Small (depth-anything/Depth-Anything-V2-Small-hf,
# Apache-2.0, ~100 MB) for `depth`, `parallax` and depth ControlNet inputs.
#   scripts/download_depth_anything.sh [dir]
set -euo pipefail
DIR="${1:-data/depth-anything-v2-small}"
mkdir -p "$DIR"
for f in config.json model.safetensors; do
    if [ -s "$DIR/$f" ]; then echo "already have $DIR/$f"; continue; fi
    curl -fL --retry 3 -o "$DIR/$f.part" "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/$f"
    mv "$DIR/$f.part" "$DIR/$f"
done
echo "Ready in $DIR. Try: ./build_and_run.sh parallax $DIR photo.png photo3d.gif"
