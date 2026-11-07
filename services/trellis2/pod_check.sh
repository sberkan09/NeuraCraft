#!/usr/bin/env bash
# The 3D service checked on a rented GPU (Runpod, image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04,
# env HF_TOKEN with access to facebook/dinov3-vitl16-pretrain-lvd1689m, TOKEN for downloads, port 8000/http):
# installs TRELLIS.2, starts the service, builds the engine, and runs image_to_3d workflows on
# TRELLIS.2's example pictures. Results: /files/<name> with the X-Token header.
#   bash scripts... services/trellis2/pod_check.sh BRANCH
set -uo pipefail
BRANCH=${1:-main}
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake build-essential libgl1 libglib2.0-0 python3.11-venv python3.11-dev > /dev/null
export PYTHON=python3.11
[ -d /nc ] || git clone -q -b "$BRANCH" https://github.com/sberkan09/NeuraCraft /nc
cd /nc && git pull -q
T0=$(date +%s)
TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1) \
    bash services/trellis2/install.sh /opt/trellis2 > /tmp/install.log 2>&1 || { tail -40 /tmp/install.log; sleep infinity; }
echo "install: $(( $(date +%s) - T0 )) s"; tail -1 /tmp/install.log
# (HF_TOKEN in the environment gives the service access to the gated DINOv3 weights)
mkdir -p /results
( cd /opt/trellis2 && . venv/bin/activate && TRELLIS_DIR=/opt/trellis2/TRELLIS.2 HOST=0.0.0.0 PORT=8000 FILES_DIR=/results \
    python server.py > /tmp/server.log 2>&1 ) &
T0=$(date +%s)
cmake -S /nc -B /nc/build -DCMAKE_BUILD_TYPE=Release > /dev/null && cmake --build /nc/build --parallel "$(nproc)" --target NeuraCraft > /tmp/build.log 2>&1 \
    || { tail -30 /tmp/build.log; sleep infinity; }
echo "engine build: $(( $(date +%s) - T0 )) s"
until curl -s localhost:8000/health | grep -q '"ok": true'; do
    sleep 10
    if ! pgrep -f server.py > /dev/null; then echo "service died:"; tail -40 /tmp/server.log; sleep infinity; fi
done
grep "loaded in" /tmp/server.log
export NEURACRAFT_3D_URL=http://127.0.0.1:8000
cd /results
python3 - <<'PY'
from PIL import Image
import glob
names = sorted(glob.glob("/opt/trellis2/TRELLIS.2/assets/example_image/*"))
for i, n in enumerate([names[0], names[11], names[30]]):
    Image.open(n).convert("RGBA").save(f"/results/in{i}.png")
PY
run() {  # picture index, quality
    cat > wf.json <<WF
{"steps":[{"op":"load","path":"in$1.png","out":"pic"},
 {"op":"image_to_3d","in":"pic","quality":"$2","seed":42,"texture":2048,"faces":300000,"out":"model"},
 {"op":"render","in":"model","width":512,"height":512,"yaw":30,"pitch":15,"out":"view"},
 {"op":"render","in":"model","width":320,"height":320,"frames":24,"out":"turn"},
 {"op":"save","in":"model","path":"m$1_$2.glb"},{"op":"save","in":"view","path":"m$1_$2.png"},
 {"op":"save","in":"turn","path":"m$1_$2.gif","fps":12}]}
WF
    local s=$(date +%s)
    /nc/bin/NeuraCraft workflow wf.json > "log$1_$2.txt" 2>&1 && echo "picture $1 ($2): $(( $(date +%s) - s )) s, $(du -h m$1_$2.glb | cut -f1)" \
        || { echo "picture $1 ($2) failed:"; tail -5 "log$1_$2.txt"; }
}
run 0 standard
run 1 standard
run 2 standard
run 0 fast
run 0 high
grep "^\[3d\] .*faces" /tmp/server.log
echo "=== pod check done ==="
sleep infinity
