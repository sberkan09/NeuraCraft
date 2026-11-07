#!/usr/bin/env bash
# The rigging service checked on a rented GPU (Runpod, image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04,
# env TOKEN for downloads, optional SAMPLES="url ..." of more .glb models, port 8000/http): installs
# SkinTokens, starts the service, builds the engine and rigs SkinTokens' example and the samples with
# the engine's rig step. Results: /files/<name> with the X-Token header.
#   bash services/rigging/pod_check.sh BRANCH
set -uo pipefail
BRANCH=${1:-main}
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake build-essential curl python3.11-venv python3.11-dev \
    libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 libsm6 libgl1 libegl1 > /dev/null
export PYTHON=python3.11
[ -d /nc ] || git clone -q -b "$BRANCH" https://github.com/sberkan09/NeuraCraft /nc
cd /nc && git pull -q
T0=$(date +%s)
bash services/rigging/install.sh /opt/skintokens > /tmp/install.log 2>&1 || { tail -40 /tmp/install.log; sleep infinity; }
echo "install: $(( $(date +%s) - T0 )) s"; tail -1 /tmp/install.log
mkdir -p /results
( cd /opt/skintokens && . venv/bin/activate && SKINTOKENS_DIR=/opt/skintokens/SkinTokens HOST=0.0.0.0 PORT=8000 FILES_DIR=/results \
    python server.py > /tmp/server.log 2>&1 ) &
T0=$(date +%s)
cmake -S /nc -B /nc/build -DCMAKE_BUILD_TYPE=Release > /dev/null && cmake --build /nc/build --parallel "$(nproc)" --target NeuraCraft > /tmp/build.log 2>&1 \
    || { tail -30 /tmp/build.log; sleep infinity; }
echo "engine build: $(( $(date +%s) - T0 )) s"
until curl -s localhost:8000/health | grep -q '"ok": true'; do
    sleep 10
    if ! pgrep -f server.py > /dev/null || curl -s localhost:8000/health | grep -q '"error"'; then
        echo "service failed:"; tail -60 /tmp/server.log; cp /tmp/server.log /results/server.log; sleep infinity
    fi
done
grep "loaded in" /tmp/server.log
export NEURACRAFT_RIG_URL=http://127.0.0.1:8000
cd /results
cp /opt/skintokens/SkinTokens/examples/giraffe.glb in_giraffe.glb
for url in ${SAMPLES:-}; do curl -sfL -o "in_$(basename "$url")" "$url" || echo "could not fetch $url"; done
run() {  # input name, seed
    cat > wf.json <<WF
{"steps":[{"op":"load","path":"in_$1.glb","out":"m"},{"op":"rig","in":"m","seed":$2,"out":"r"},
          {"op":"render","in":"r","width":512,"height":512,"yaw":-30,"out":"v"},
          {"op":"save","in":"r","path":"rig_$1_$2.glb"},{"op":"save","in":"v","path":"rig_$1_$2.png"}]}
WF
    local s=$(date +%s)
    /nc/bin/NeuraCraft workflow wf.json > "log_$1_$2.txt" 2>&1 && echo "$1 (seed $2): $(( $(date +%s) - s )) s, $(du -h rig_$1_$2.glb | cut -f1)" \
        || { echo "$1 (seed $2) failed:"; tail -5 "log_$1_$2.txt"; }
}
for f in in_*.glb; do n=${f#in_}; run "${n%.glb}" 42; done
for f in in_*.glb; do n=${f#in_}; [ "${n%.glb}" = giraffe ] || run "${n%.glb}" 7; done
grep "^\[rig\] seed" /tmp/server.log
cp /tmp/server.log /results/server.log
echo "=== pod check done ==="
sleep infinity
