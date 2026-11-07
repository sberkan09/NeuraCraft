#!/usr/bin/env bash
# The whole 3D chain checked on one rented GPU: a picture -> image_to_3d (TRELLIS.2) -> rig (SkinTokens)
# -> animate (CMU motion capture) -> an animated, textured .glb and a rendered clip. Runpod image
# runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04, 48 GB GPU, env HF_TOKEN (DINOv3 access),
# TOKEN for downloads, port 8000/http. Pictures: docs/images/chain_*.png. Results: /files/<name>.
#   bash services/pod_chain.sh BRANCH
set -uo pipefail
BRANCH=${1:-main}
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake build-essential curl unzip python3.11-venv python3.11-dev libgl1 libegl1 libglib2.0-0 \
    libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 libsm6 libusb-1.0-0 libgomp1 > /dev/null
export PYTHON=python3.11
[ -d /nc ] || git clone -q -b "$BRANCH" https://github.com/sberkan09/NeuraCraft /nc
cd /nc && git pull -q
mkdir -p /results
T0=$(date +%s)
TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1) \
    bash services/trellis2/install.sh /opt/trellis2 > /results/install_trellis2.log 2>&1 || { tail -40 /results/install_trellis2.log; sleep infinity; }
bash services/rigging/install.sh /opt/skintokens > /results/install_rigging.log 2>&1 || { tail -40 /results/install_rigging.log; sleep infinity; }
echo "install: $(( $(date +%s) - T0 )) s"
( cd /opt/trellis2 && . venv/bin/activate && TRELLIS_DIR=/opt/trellis2/TRELLIS.2 HOST=0.0.0.0 PORT=8000 FILES_DIR=/results \
    python server.py > /results/trellis2.log 2>&1 ) &
( cd /opt/skintokens && . venv/bin/activate && SKINTOKENS_DIR=/opt/skintokens/SkinTokens HOST=127.0.0.1 PORT=8001 \
    python server.py > /results/rigging.log 2>&1 ) &
cmake -S /nc -B /nc/build -DCMAKE_BUILD_TYPE=Release > /dev/null && cmake --build /nc/build --parallel "$(nproc)" --target NeuraCraft > /results/build.log 2>&1 \
    || { tail -30 /results/build.log; sleep infinity; }
bash scripts/download_cmu_motions.sh /opt/motions > /dev/null
for port in 8000 8001; do
    until curl -s localhost:$port/health | grep -q '"ok": true'; do
        sleep 10
        if curl -s localhost:$port/health | grep -q '"error"'; then echo "service on $port failed"; curl -s localhost:$port/health; sleep infinity; fi
    done
done
echo "services ready: $(( $(date +%s) - T0 )) s"
export NEURACRAFT_3D_URL=http://127.0.0.1:8000 NEURACRAFT_RIG_URL=http://127.0.0.1:8001
cd /results
for pic in /nc/docs/images/chain_*.png; do
    n=$(basename "$pic" .png); n=${n#chain_}
    cat > wf_$n.json <<WF
{"steps":[{"op":"load","path":"$pic","out":"pic"},
 {"op":"image_to_3d","in":"pic","quality":"standard","faces":200000,"out":"model"},
 {"op":"rig","in":"model","out":"rigged"},
 {"op":"animate","in":"rigged","capture":"/opt/motions/walk.bvh","seconds":4,"fps":24,"out":"walk"},
 {"op":"animate","in":"rigged","capture":"/opt/motions/dance.bvh","seconds":6,"fps":24,"out":"dance"},
 {"op":"render","in":"walk","width":384,"height":384,"yaw":-35,"out":"walk_clip"},
 {"op":"render","in":"dance","width":384,"height":384,"yaw":-20,"out":"dance_clip"},
 {"op":"save","in":"model","path":"${n}_model.glb"},{"op":"save","in":"rigged","path":"${n}_rigged.glb"},
 {"op":"save","in":"walk","path":"${n}_walk.glb"},{"op":"save","in":"dance","path":"${n}_dance.glb"},
 {"op":"save","in":"walk_clip","path":"${n}_walk.gif","fps":24},{"op":"save","in":"dance_clip","path":"${n}_dance.gif","fps":24}]}
WF
    s=$(date +%s)
    /nc/bin/NeuraCraft workflow wf_$n.json > log_$n.txt 2>&1 && echo "$n: $(( $(date +%s) - s )) s" || { echo "$n failed:"; tail -6 log_$n.txt; }
    grep "animate:" log_$n.txt | head -6
done
echo "=== chain check done ==="
sleep infinity
