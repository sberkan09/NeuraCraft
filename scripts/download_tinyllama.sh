#!/usr/bin/env bash
# Downloads TinyLlama-1.1B-Chat-v1.0 (Apache 2.0) from Hugging Face:
# bf16 safetensors weights plus the tokenizer (~2.2 GB in total).
#   scripts/download_tinyllama.sh [dir]
set -euo pipefail

DIR="${1:-data/tinyllama}"
BASE="${TINYLLAMA_BASE_URL:-https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main}"
mkdir -p "$DIR"
for f in config.json tokenizer.json tokenizer_config.json model.safetensors; do
    if [ -s "$DIR/$f" ]; then
        echo "already have $DIR/$f"
        continue
    fi
    echo "downloading $f ..."
    curl -fL --retry 3 -o "$DIR/$f.part" "$BASE/$f"
    mv "$DIR/$f.part" "$DIR/$f"
done
echo "TinyLlama ready in $DIR. Try: ./build_and_run.sh llama $DIR \"What is the capital of Turkey?\" --chat"
