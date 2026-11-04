#!/usr/bin/env bash
# Downloads Llama-3.2-1B-Instruct (Llama 3.2 Community License) from the
# ungated unsloth mirror on Hugging Face: bf16 safetensors plus the tokenizer
# (~2.5 GB). Set LLAMA32_BASE_URL to use another mirror, e.g. meta-llama's own
# repository with an access token.
#   scripts/download_llama32.sh [dir]
set -euo pipefail

DIR="${1:-data/llama3}"
BASE="${LLAMA32_BASE_URL:-https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main}"
mkdir -p "$DIR"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors; do
    if [ -s "$DIR/$f" ]; then
        echo "already have $DIR/$f"
        continue
    fi
    echo "downloading $f ..."
    curl -fL --retry 3 -o "$DIR/$f.part" "$BASE/$f"
    mv "$DIR/$f.part" "$DIR/$f"
done
echo "Llama 3.2 ready in $DIR. Try: ./build_and_run.sh llama $DIR \"What is the capital of Turkey?\" --chat"
