#!/usr/bin/env bash
# Runs on a rented GPU machine (a Runpod pod's start command): builds the branch with CUDA, runs
# scripts/gpu_check.sh, then watches the branch and runs again on every new commit. Everything goes
# to stdout (the pod's log). Start it with:
#   apt-get update && apt-get install -y git && git clone -b BRANCH https://github.com/sberkan09/NeuraCraft /nc && bash /nc/scripts/gpu_pod.sh BRANCH
set -uo pipefail
BRANCH=${1:-main}
IDLE_MINUTES=${IDLE_MINUTES:-120}
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake g++ make curl python3 python3-numpy python3-pil > /dev/null
cd /nc
last=""
idle=0
while true; do
    git fetch -q origin "$BRANCH" && git reset -q --hard "origin/$BRANCH"
    head=$(git rev-parse --short HEAD)
    if [ "$head" != "$last" ]; then
        echo "=== gpu-pod: $head $(git log -1 --format=%s) ==="
        bash scripts/gpu_check.sh 2>&1
        echo "=== gpu-pod: done $head ==="
        last=$head
        idle=0
    fi
    sleep 30
    idle=$((idle + 1))
    if [ $idle -gt $((IDLE_MINUTES * 2)) ]; then echo "=== gpu-pod: idle for $IDLE_MINUTES min, stopping ==="; sleep infinity; fi
done
