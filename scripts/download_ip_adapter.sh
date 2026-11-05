#!/usr/bin/env bash
# Downloads IP-Adapter for SD 1.5 (h94/IP-Adapter, Apache-2.0): the adapter
# (~45 MB) and its CLIP ViT-H image encoder (~2.5 GB) into DIR and DIR/image_encoder.
#   scripts/download_ip_adapter.sh [dir]
set -euo pipefail
DIR="${1:-data/ip-adapter}"
BASE=https://huggingface.co/h94/IP-Adapter/resolve/main/models
get() {
    if [ -s "$2" ]; then echo "already have $2"; return; fi
    mkdir -p "$(dirname "$2")"
    curl -fL --retry 3 -o "$2.part" "$1"
    mv "$2.part" "$2"
}
get "$BASE/ip-adapter_sd15.safetensors" "$DIR/ip-adapter_sd15.safetensors"
get "$BASE/image_encoder/config.json" "$DIR/image_encoder/config.json"
get "$BASE/image_encoder/model.safetensors" "$DIR/image_encoder/model.safetensors"
echo "Ready in $DIR. Try: ./build_and_run.sh txt2img data/animatediff \"a cat\" out.png --ip-adapter $DIR/ip-adapter_sd15.safetensors --image-prompt style.png"
