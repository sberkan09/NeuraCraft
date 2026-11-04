#!/usr/bin/env bash
# A short, gentle benchmark for Apple silicon (or any machine): builds
# NeuraCraft, then times SD-Turbo text to image on the CPU (Accelerate) and
# the GPU (Metal), int8 and float32, plus a short animation. Pauses between
# runs so the machine stays cool; about 5-10 minutes in all. Results go to
# benchmark_results.txt (paste that file back).
#   scripts/mac_benchmark.sh [sd-turbo dir]
set -uo pipefail
cd "$(dirname "$0")/.."
DIR="${1:-data/sd-turbo}"
OUT=benchmark_results.txt
COOL="${COOL_DOWN_SECONDS:-20}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNEURACRAFT_NATIVE=ON >/dev/null
cmake --build build --parallel 4 >/dev/null || { echo "build failed"; exit 1; }
[ -s "$DIR/unet/config.json" ] || scripts/download_sd_turbo.sh "$DIR"

{
    echo "== NeuraCraft benchmark, $(date)"
    uname -a
    sysctl -n machdep.cpu.brand_string 2>/dev/null
    sysctl -n hw.memsize 2>/dev/null | awk '{printf "memory: %.0f GB\n", $1 / 1073741824}'
    git log --oneline -1
} > "$OUT"

run() {  # label, command...
    local label="$1"; shift
    echo "== $label" | tee -a "$OUT"
    local start=$(date +%s)
    "$@" >> "$OUT" 2>&1
    local status=$?
    echo "   exit $status, wall $(( $(date +%s) - start )) s" | tee -a "$OUT"
    sleep "$COOL"
}

P="A cinematic photo of a red fox sitting in the snow, golden hour"
run "GEMM benchmark (quick)" ./bin/NeuraCraft bench --quick
run "txt2img 512, int8, TAESD, CPU" ./bin/NeuraCraft txt2img "$DIR" "$P" /tmp/nc_cpu_int8.png --tiny-vae "$DIR/taesd"
run "txt2img 512, fp32, TAESD, CPU" ./bin/NeuraCraft txt2img "$DIR" "$P" /tmp/nc_cpu_fp32.png --fp32 --tiny-vae "$DIR/taesd"
run "txt2img 512, fp32, TAESD, GPU" ./bin/NeuraCraft --gpu txt2img "$DIR" "$P" /tmp/nc_gpu_fp32.png --fp32 --tiny-vae "$DIR/taesd"
run "txt2img 512, int8, full VAE, GPU" ./bin/NeuraCraft --gpu txt2img "$DIR" "$P" /tmp/nc_gpu_vae.png
run "morph animation 4 frames, 512, GPU fp32" ./bin/NeuraCraft --gpu video "$DIR" morph "a red fox" "a white owl" --frames 2 --fp32 -o /tmp/nc_morph.gif
echo "done: $OUT"
