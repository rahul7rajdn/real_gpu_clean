#!/bin/bash


set -e  # stop if any command fails

pwd

echo $USER

cd /tmp
pip install --user virtualenv

export PATH=$PATH:/home/hice1/$USER/.local/bin/

virtualenv pytorch_cuda_env
source /tmp/pytorch_cuda_env/bin/activate

export TMPDIR=/tmp
export PIP_CACHE_DIR=/tmp/pipcache
export PIP_NO_CACHE_DIR=1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib seaborn numpy pandas

python - << 'EOF'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU device_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
EOF
