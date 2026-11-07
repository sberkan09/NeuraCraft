#!/usr/bin/env bash
# Downloads a ControlNet 1.1 model for Stable Diffusion 1.5 in diffusers layout
# (lllyasviel/control_v11p_sd15_<kind>, OpenRAIL; fp16, ~720 MB). Default: canny.
#   scripts/download_controlnet.sh [dir] [kind]     e.g. kind = canny, scribble, lineart, softedge, depth (f1p for depth)
set -euo pipefail
KIND="${2:-canny}"
DIR="${1:-data/controlnet-$KIND}"
REPO="lllyasviel/control_v11p_sd15_$KIND"
[ "$KIND" = depth ] && REPO="lllyasviel/control_v11f1p_sd15_depth"
mkdir -p "$DIR"
for f in config.json diffusion_pytorch_model.fp16.safetensors; do
    if [ -s "$DIR/$f" ]; then echo "already have $DIR/$f"; continue; fi
    curl -fL --retry 3 -o "$DIR/$f.part" "https://huggingface.co/$REPO/resolve/main/$f"
    mv "$DIR/$f.part" "$DIR/$f"
done
echo "Ready in $DIR. Try: ./build_and_run.sh txt2img data/animatediff \"a cabin in summer\" out.png --controlnet $DIR --control photo.png --canny  (depth: --depth-model data/depth-anything-v2-small)"
