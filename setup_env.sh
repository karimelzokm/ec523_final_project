#!/usr/bin/env bash
# setup_env.sh – rebuild the grpo conda env from scratch with pinned versions
# Run once on a login node: bash setup_env.sh
set -euo pipefail

ENV_PATH="/projectnb/vkolagrp/kelzokm/conda_envs/grpo"
CONDA_SH="/usr4/ugrad/kelzokm/miniconda3/etc/profile.d/conda.sh"

source "$CONDA_SH"

# remove existing env if it exists
if conda env list | grep -q "$ENV_PATH"; then
    echo "Removing existing env at $ENV_PATH ..."
    conda env remove -p "$ENV_PATH" -y
fi

# create fresh env
echo "Creating fresh env (Python 3.11) ..."
conda create -y -p "$ENV_PATH" python=3.11

conda activate "$ENV_PATH"
export PYTHONNOUSERSITE=1   # keep ~/.local from interfering

# pytorch first (pinned to avoid version conflicts)
echo "Installing PyTorch 2.5.1+cu121 ..."
pip install "torch==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-deps          # avoid pulling torchvision/torchaudio

# Install torch deps that we actually need
pip install filelock jinja2 networkx sympy typing-extensions

# huggingface stack
echo "Installing HuggingFace stack ..."
pip install \
    "transformers==4.46.3" \
    "peft==0.13.2" \
    "trl==0.13.2" \
    "accelerate==0.34.2" \
    "datasets==3.2.0" \
    "tokenizers==0.20.0" \
    "bitsandbytes==0.45.0" \
    "pyarrow>=14.0" \
    "numpy>=1.24" \
    "scipy>=1.11" \
    "tqdm>=4.66"
# torchvision intentionally NOT installed (not needed; causes breakage with mismatched torch)

# check everything installed ok
echo ""
echo "Verifying installs ..."
python - <<'EOF'
import torch, transformers, peft, datasets, bitsandbytes, accelerate
print(f"  torch          {torch.__version__}   cuda={torch.cuda.is_available()}")
print(f"  transformers   {transformers.__version__}")
print(f"  peft           {peft.__version__}")
print(f"  datasets       {datasets.__version__}")
print(f"  bitsandbytes   {bitsandbytes.__version__}")
print(f"  accelerate     {accelerate.__version__}")
EOF

echo ""
echo "Done. Activate with:"
echo "  conda activate $ENV_PATH"
