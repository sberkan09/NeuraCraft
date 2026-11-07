#!/usr/bin/env bash
# Installs TRELLIS.2 and the 3D service in a Python virtual environment, on Linux with an NVIDIA GPU
# (24 GB or more) and the CUDA 12.4 toolkit (nvcc). About 20-30 minutes: five CUDA extensions are built.
#   services/trellis2/install.sh [DIR]      (default /opt/trellis2; then: DIR/run.sh)
set -euo pipefail
DIR=${1:-/opt/trellis2}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;8.6;8.9;9.0"}
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
mkdir -p "$DIR" && cd "$DIR"
[ -d TRELLIS.2 ] || git clone -q --recursive -b main https://github.com/microsoft/TRELLIS.2.git
[ -d venv ] || "$PY" -m venv venv
. venv/bin/activate
pip install -q --upgrade pip wheel setuptools
pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -q imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers pandas zstandard kornia timm pillow
pip install -q git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
# Attention: FlashAttention (a prebuilt wheel; building it from source would take an hour), else xFormers.
timeout 900 pip install -q flash-attn==2.7.4.post1 --no-build-isolation \
    || pip install -q xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124
mkdir -p ext
build() {  # name, git url, ref
    [ -d "ext/$1" ] || git clone -q --recursive ${3:+-b "$3"} "$2" "ext/$1"
    pip install -q "ext/$1" --no-build-isolation
}
build nvdiffrast https://github.com/NVlabs/nvdiffrast.git v0.4.0
build nvdiffrec https://github.com/JeffreyXiang/nvdiffrec.git renderutils
build CuMesh https://github.com/JeffreyXiang/CuMesh.git
build FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git
pip install -q TRELLIS.2/o-voxel --no-build-isolation
cp "$HERE/server.py" "$DIR/server.py"
cat > run.sh <<RUN
#!/usr/bin/env bash
cd "$DIR" && . venv/bin/activate && TRELLIS_DIR="$DIR/TRELLIS.2" exec python server.py
RUN
chmod +x run.sh
python -c "import torch, o_voxel, cumesh, flex_gemm, nvdiffrast.torch; print('trellis2 ready, torch', torch.__version__, 'cuda', torch.version.cuda)"
