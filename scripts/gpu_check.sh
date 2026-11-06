#!/usr/bin/env bash
# What a GPU machine checks (scripts/gpu_pod.sh runs it): build with CUDA, the test suite with the
# GPU (exact FP32, then TF32), the GEMM benchmark, and SD-Turbo on the GPU against the CPU.
set -uo pipefail
cd "$(dirname "$0")/.."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
t0=$(date +%s)
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DNEURACRAFT_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DNEURACRAFT_NATIVE=ON > /tmp/cmake.log 2>&1 || { tail -30 /tmp/cmake.log; exit 1; }
cmake --build build-cuda --parallel "$(nproc)" > /tmp/build.log 2>&1 || { grep -E "error" -A3 /tmp/build.log | head -60; exit 1; }
echo "build: $(( $(date +%s) - t0 )) s"
NC=bin/NeuraCraft
echo "--- tests (FP32) ---"
NEURACRAFT_CUDA_TF32=0 ./build-cuda/neuracraft_tests 2>&1 | grep -E "GPU:|FAIL|passed|failed|max diff|attention|differ|mismatch" | head -40
echo "--- bench ---"
$NC bench 2>&1 | tail -12
echo "--- sd-turbo ---"
M=/data/sd-turbo
mkdir -p /data
[ -s $M/unet/diffusion_pytorch_model.fp16.safetensors ] || bash scripts/download_sd_turbo.sh $M vae text_encoder unet > /tmp/dl.log 2>&1 || tail -5 /tmp/dl.log
for dev in gpu gpu32 cpu; do
    flag=""; [ $dev != cpu ] && flag=--gpu
    tf=1; [ $dev = gpu32 ] && tf=0
    s=$(date +%s.%N)
    NEURACRAFT_CUDA_TF32=$tf $NC $flag txt2img $M "a red fox in the snow, photo" /tmp/fox_$dev.png --steps 1 --seed 7 --size 512 > /tmp/sd_$dev.log 2>&1 || tail -5 /tmp/sd_$dev.log
    echo "$dev: $(echo "$(date +%s.%N) - $s" | bc 2>/dev/null || python3 -c "print(round($(date +%s.%N)-$s,2))") s"
    grep -iE "unet|denois|step|vae|decode|text" /tmp/sd_$dev.log | tail -4
done
python3 - <<'PY'
import numpy as np, base64, io
from PIL import Image
b = np.asarray(Image.open("/tmp/fox_cpu.png").convert("RGB")).astype(np.float64)
for name in ["gpu", "gpu32"]:
    a = np.asarray(Image.open("/tmp/fox_%s.png" % name).convert("RGB")).astype(np.float64)
    mse = ((a - b) ** 2).mean()
    print("%s vs cpu: mean abs %.3f, PSNR %.1f dB" % (name, np.abs(a - b).mean(), 10 * np.log10(255 ** 2 / max(mse, 1e-9))))
buf = io.BytesIO(); Image.open("/tmp/fox_gpu.png").convert("RGB").resize((128, 128)).save(buf, "JPEG", quality=80)
print("thumb:" + base64.b64encode(buf.getvalue()).decode())
PY
