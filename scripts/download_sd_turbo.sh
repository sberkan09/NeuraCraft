#!/usr/bin/env bash
# Downloads Stable Diffusion Turbo (stabilityai/sd-turbo; Stability AI
# Non-Commercial Research Community License) in diffusers layout, fp16 weights.
#   scripts/download_sd_turbo.sh [dir] [components...]
# Components: vae (default; ~170 MB). More follow as NeuraCraft runs them.
set -euo pipefail

DIR="${1:-data/sd-turbo}"
shift || true
COMPONENTS=("${@:-vae}")
BASE="${SD_TURBO_BASE_URL:-https://huggingface.co/stabilityai/sd-turbo/resolve/main}"

files=(model_index.json)
for c in "${COMPONENTS[@]}"; do
    case "$c" in
    vae) files+=(vae/config.json vae/diffusion_pytorch_model.fp16.safetensors) ;;
    *) echo "unknown component: $c" >&2; exit 1 ;;
    esac
done

for f in "${files[@]}"; do
    mkdir -p "$DIR/$(dirname "$f")"
    if [ -s "$DIR/$f" ]; then
        echo "already have $DIR/$f"
        continue
    fi
    echo "downloading $f ..."
    curl -fL --retry 3 -o "$DIR/$f.part" "$BASE/$f"
    mv "$DIR/$f.part" "$DIR/$f"
done
echo "SD-Turbo (${COMPONENTS[*]}) ready in $DIR. Try: ./build_and_run.sh vae $DIR"
