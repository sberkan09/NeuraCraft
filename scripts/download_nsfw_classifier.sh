#!/usr/bin/env bash
# Downloads Falconsai's ViT NSFW classifier (Apache-2.0, ~343 MB, the ONNX export by onnx-community)
# and writes its manifest, for the workflow "classify" step: labels normal / nsfw.
#   scripts/download_nsfw_classifier.sh [dir]
set -euo pipefail
DIR="${1:-data/onnx}"
mkdir -p "$DIR"
OUT="$DIR/nsfw.onnx"
cat > "$DIR/nsfw.json" <<'JSON'
{"input": {"size": [224, 224], "range": [0, 1], "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
 "output": {"labels": ["normal", "nsfw"], "activation": "softmax"}}
JSON
if [ -s "$OUT" ]; then echo "already have $OUT"; exit 0; fi
curl -fL --retry 3 -o "$OUT.part" https://huggingface.co/onnx-community/nsfw_image_detection-ONNX/resolve/main/onnx/model.onnx
mv "$OUT.part" "$OUT"
echo "Ready: $OUT (+ nsfw.json). In a workflow: {\"op\": \"classify\", \"model\": \"$OUT\", \"in\": \"pic\", \"path\": \"out/moderation.json\"}"
