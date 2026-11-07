#!/usr/bin/env bash
# Builds the fuzz targets (Clang, ASan + UBSan) and runs each for SECONDS (default 60) on seeds made
# from the test files. Crashes land in fuzz/out/<target>-crash-*; rerun one with
#   build-fuzz/fuzz_<target> fuzz/out/<target>-crash-...
set -euo pipefail
cd "$(dirname "$0")/.."
SECONDS_EACH=${1:-60}
cmake -S . -B build-fuzz -DCMAKE_CXX_COMPILER=clang++ -DNEURACRAFT_FUZZ=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
cmake --build build-fuzz --parallel >/dev/null
C=fuzz/out/corpus
mkdir -p "$C"/{json,workflow,png,safetensors,onnx,tensorproto,mesh}
cp docs/workflows/*.json "$C/json/"; cp docs/workflows/*.json "$C/workflow/"
i=0; for f in tests/onnx/cases/*/model.onnx tests/onnx/pipeline/*.onnx; do cp "$f" "$C/onnx/m$i.onnx"; i=$((i+1)); done
i=0; for f in tests/onnx/cases/*/*.pb; do [ "$(wc -c <"$f")" -lt 4096 ] && cp "$f" "$C/tensorproto/t$i.pb"; i=$((i+1)); done
cp fuzz/seeds/*.png fuzz/seeds/*.ppm "$C/png/"; cp fuzz/seeds/*.safetensors "$C/safetensors/"; cp fuzz/seeds/*.glb fuzz/seeds/*.obj fuzz/seeds/*.ply fuzz/seeds/*.bvh "$C/mesh/"
status=0
for t in json workflow png safetensors onnx tensorproto mesh; do
    echo "== fuzz_$t (${SECONDS_EACH} s)"
    if ! ./build-fuzz/fuzz_$t -max_total_time="$SECONDS_EACH" -rss_limit_mb=2048 -malloc_limit_mb=1024 -timeout=10 -max_len=65536 \
        -artifact_prefix=fuzz/out/$t- "$C/$t" 2>fuzz/out/$t.log; then
        status=1
        grep -E "ERROR|SUMMARY" fuzz/out/$t.log | head -5
    fi
    grep -E "^Done" fuzz/out/$t.log || true
done
exit $status
