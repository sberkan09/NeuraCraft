#!/usr/bin/env bash
# Downloads text-to-video weights for `NeuraCraft video DIR t2v`:
#   - a Stable Diffusion 1.5 model in diffusers layout (default
#     emilianJR/epiCRealism, CreativeML OpenRAIL-M; ~4.3 GB: UNet fp32, VAE,
#     CLIP ViT-L text encoder, tokenizer, scheduler), into DIR, and
#   - the AnimateDiff-Lightning motion module (ByteDance, CreativeML
#     OpenRAIL++-M; ~0.9 GB, distilled for 1, 2, 4 or 8 steps), into DIR/motion,
#   - TAESD (madebyollin/taesd, MIT, ~10 MB) into DIR/taesd for fast decoding.
#   scripts/download_animatediff.sh [dir] [steps]
# Other SD 1.5 models work too: SD15_BASE_URL=https://huggingface.co/<repo>/resolve/main
set -euo pipefail

DIR="${1:-data/animatediff}"
STEPS="${2:-4}"
case "$STEPS" in 1|2|4|8) ;; *) echo "steps must be 1, 2, 4 or 8" >&2; exit 1 ;; esac
BASE="${SD15_BASE_URL:-https://huggingface.co/emilianJR/epiCRealism/resolve/main}"
MOTION="${MOTION_BASE_URL:-https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/main}"

get() {  # url, destination
    if [ -s "$2" ]; then echo "already have $2"; return; fi
    mkdir -p "$(dirname "$2")"
    echo "downloading $2 ..."
    curl -fL --retry 3 -o "$2.part" "$1"
    mv "$2.part" "$2"
}

for f in model_index.json scheduler/scheduler_config.json tokenizer/vocab.json tokenizer/merges.txt \
         tokenizer/tokenizer_config.json tokenizer/special_tokens_map.json text_encoder/config.json \
         text_encoder/model.safetensors vae/config.json vae/diffusion_pytorch_model.safetensors \
         unet/config.json unet/diffusion_pytorch_model.safetensors; do
    get "$BASE/$f" "$DIR/$f"
done
get https://huggingface.co/madebyollin/taesd/resolve/main/config.json "$DIR/taesd/config.json"
get https://huggingface.co/madebyollin/taesd/resolve/main/diffusion_pytorch_model.safetensors "$DIR/taesd/diffusion_pytorch_model.safetensors"
get "$MOTION/animatediff_lightning_${STEPS}step_diffusers.safetensors" "$DIR/motion/animatediff_lightning_${STEPS}step_diffusers.safetensors"
echo "Ready in $DIR. Try: ./build_and_run.sh video $DIR t2v \"a corgi running on the beach, golden hour\" --steps $STEPS"
