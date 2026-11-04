#!/usr/bin/env bash
# Downloads Tiny Shakespeare (~1.1 MB, public domain) for the GPT demo.
#   scripts/download_shakespeare.sh [dir]
set -euo pipefail

DIR="${1:-data}"
URL="${SHAKESPEARE_URL:-https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt}"
mkdir -p "$DIR"
if [ ! -f "$DIR/shakespeare.txt" ]; then
    curl -fL --retry 3 -o "$DIR/shakespeare.txt" "$URL"
fi
echo "Ready: $DIR/shakespeare.txt. Train with: ./build_and_run.sh gpt $DIR/shakespeare.txt 2"
