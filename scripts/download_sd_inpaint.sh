#!/usr/bin/env bash
# Downloads Stable Diffusion 1.5 inpainting (stable-diffusion-v1-5/stable-diffusion-inpainting,
# CreativeML OpenRAIL-M; fp16, ~2.1 GB) plus TAESD, for `NeuraCraft inpaint` and
# `video DIR outpaint`. Other SD 1.5 inpainting models in diffusers layout work too:
# SD15_INPAINT_BASE_URL=https://huggingface.co/<repo>/resolve/main
#   scripts/download_sd_inpaint.sh [dir]
set -euo pipefail
DIR="${1:-data/sd-inpaint}"
BASE="${SD15_INPAINT_BASE_URL:-https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/resolve/main}"
get() {
    if [ -s "$2" ]; then echo "already have $2"; return; fi
    mkdir -p "$(dirname "$2")"
    echo "downloading $2 ..."
    curl -fL --retry 3 -o "$2.part" "$1"
    mv "$2.part" "$2"
}
for f in scheduler/scheduler_config.json tokenizer/vocab.json tokenizer/merges.txt tokenizer/tokenizer_config.json \
         tokenizer/special_tokens_map.json text_encoder/config.json text_encoder/model.fp16.safetensors \
         vae/config.json vae/diffusion_pytorch_model.fp16.safetensors unet/config.json unet/diffusion_pytorch_model.fp16.safetensors; do
    get "$BASE/$f" "$DIR/$f"
done
get https://huggingface.co/madebyollin/taesd/resolve/main/config.json "$DIR/taesd/config.json"
get https://huggingface.co/madebyollin/taesd/resolve/main/diffusion_pytorch_model.safetensors "$DIR/taesd/diffusion_pytorch_model.safetensors"
echo "Ready in $DIR. Try: ./build_and_run.sh inpaint $DIR photo.png mask.png \"a vase of flowers\""
