#!/usr/bin/env bash
# Installs SkinTokens / TokenRig and the rigging service in a Python 3.11 virtual environment, on Linux
# with an NVIDIA GPU (14 GB or more). Needs python3.11 with venv, git, and the system libraries Blender's
# bpy module loads (libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 libsm6 libgl1).
#   services/rigging/install.sh [DIR]      (default /opt/skintokens; then: DIR/run.sh)
set -euo pipefail
DIR=${1:-/opt/skintokens}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3.11}
mkdir -p "$DIR" && cd "$DIR"
[ -d SkinTokens ] || git clone -q https://github.com/VAST-AI-Research/SkinTokens.git
[ -d venv ] || "$PY" -m venv venv
. venv/bin/activate
pip install -q --upgrade pip wheel setuptools
pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -q -r SkinTokens/requirements.txt requests
# FlashAttention 2 (required by TokenRig): a prebuilt wheel for torch 2.7 / CUDA 12 / Python 3.11.
FA=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abi
pip install -q "${FA}TRUE-cp311-cp311-linux_x86_64.whl" && python -c "import flash_attn_2_cuda" 2> /dev/null \
    || { pip install -q --force-reinstall --no-deps "${FA}FALSE-cp311-cp311-linux_x86_64.whl" && python -c "import flash_attn_2_cuda"; }
(cd SkinTokens && python download.py --model)
cp "$HERE/server.py" "$DIR/server.py"
cat > run.sh <<RUN
#!/usr/bin/env bash
cd "$DIR" && . venv/bin/activate && SKINTOKENS_DIR="$DIR/SkinTokens" exec python server.py
RUN
chmod +x run.sh
cd SkinTokens && python -c "import torch, bpy, flash_attn, demo; print('skintokens ready, torch', torch.__version__, 'bpy', bpy.app.version_string)"
