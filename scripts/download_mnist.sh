#!/usr/bin/env bash
# Downloads the MNIST IDX files into a directory (default: data/mnist).
#   scripts/download_mnist.sh [dir]
set -euo pipefail

DIR="${1:-data/mnist}"
BASE_URL="${MNIST_BASE_URL:-https://storage.googleapis.com/cvdf-datasets/mnist}"
FILES="train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"

mkdir -p "$DIR"
for f in $FILES; do
    if [ -f "$DIR/$f" ]; then
        echo "already have $DIR/$f"
        continue
    fi
    echo "downloading $f ..."
    curl -fL --retry 3 -o "$DIR/$f.gz" "$BASE_URL/$f.gz"
    gunzip -f "$DIR/$f.gz"
done
echo "MNIST ready in $DIR. Train with: ./build_and_run.sh mnist $DIR 5"
