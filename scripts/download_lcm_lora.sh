#!/usr/bin/env bash
# Downloads LCM-LoRA for Stable Diffusion 1.5 (latent-consistency/lcm-lora-sdv1-5,
# OpenRAIL++, ~135 MB): with --sampler lcm any SD 1.5 model draws in 4 steps.
#   scripts/download_lcm_lora.sh [dir]
set -euo pipefail
DIR="${1:-data/loras}"
mkdir -p "$DIR"
OUT="$DIR/lcm-lora-sdv1-5.safetensors"
if [ -s "$OUT" ]; then echo "already have $OUT"; exit 0; fi
curl -fL --retry 3 -o "$OUT.part" https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors
mv "$OUT.part" "$OUT"
echo "Ready: $OUT. Try: ./build_and_run.sh txt2img data/animatediff \"a red fox in the snow\" fox.jpg --steps 4 --sampler lcm --lora $OUT"
