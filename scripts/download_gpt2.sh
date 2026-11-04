#!/usr/bin/env bash
# Downloads GPT-2 small (124M, MIT license) from Hugging Face: weights in
# safetensors format plus the tokenizer files (~550 MB in total).
#   scripts/download_gpt2.sh [dir]
set -euo pipefail

DIR="${1:-data/gpt2}"
BASE="${GPT2_BASE_URL:-https://huggingface.co/openai-community/gpt2/resolve/main}"
mkdir -p "$DIR"
for f in config.json vocab.json merges.txt model.safetensors; do
    if [ -s "$DIR/$f" ]; then
        echo "already have $DIR/$f"
        continue
    fi
    echo "downloading $f ..."
    curl -fL --retry 3 -o "$DIR/$f.part" "$BASE/$f"
    mv "$DIR/$f.part" "$DIR/$f"
done
echo "GPT-2 ready in $DIR. Try: ./build_and_run.sh gpt2 $DIR \"The meaning of life is\""
