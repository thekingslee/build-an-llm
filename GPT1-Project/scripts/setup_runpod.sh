#!/usr/bin/env bash
# =============================================================================
# setup_runpod.sh — One-shot pod setup script for RunPod
#
# Run this immediately after spinning up a new pod:
#   bash /workspace/build-an-llm/GPT1-Project/scripts/setup_runpod.sh
#
# Prerequisites:
#   - A Network Volume mounted at /runpod-volume
#   - Your WANDB_API_KEY exported or set below
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Config — edit these before running
# ---------------------------------------------------------------------------
REPO_URL="https://github.com/thekingslee/build-an-llm.git"   # your repo URL
REPO_BRANCH="feat/30m-param-model"
WORKSPACE="/workspace/build-an-llm"
WANDB_KEY="${WANDB_API_KEY:-}"   # export WANDB_API_KEY=<key> before running, or set here

# ---------------------------------------------------------------------------
# 1. Verify GPU
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " GPU Check"
echo "============================================================"
nvidia-smi
echo ""

# ---------------------------------------------------------------------------
# 2. Create Network Volume directories
# ---------------------------------------------------------------------------
echo "Creating Network Volume directories..."
mkdir -p /runpod-volume/checkpoints
mkdir -p /runpod-volume/datasets
echo "  /runpod-volume/checkpoints  ✅"
echo "  /runpod-volume/datasets     ✅"

# ---------------------------------------------------------------------------
# 3. Clone repo (skip if already present)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Repository Setup"
echo "============================================================"
if [ -d "$WORKSPACE/.git" ]; then
    echo "Repo already cloned. Pulling latest..."
    git -C "$WORKSPACE" fetch origin
    git -C "$WORKSPACE" checkout "$REPO_BRANCH"
    git -C "$WORKSPACE" pull origin "$REPO_BRANCH"
else
    echo "Cloning $REPO_URL..."
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORKSPACE"
fi
echo "Repo ready at $WORKSPACE  ✅"

# ---------------------------------------------------------------------------
# 3.5. Install System Utilities (tmux)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " System Utilities"
echo "============================================================"
if ! command -v tmux &>/dev/null; then
    echo "Installing tmux..."
    apt-get update -y && apt-get install -y tmux
fi
echo "tmux installed  ✅"

# ---------------------------------------------------------------------------
# 4. Install uv + sync dependencies
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Python Dependencies"
echo "============================================================"
# Ensure paths are set even if uv was installed in a previous session
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

cd "$WORKSPACE"
uv sync
echo "Dependencies installed  ✅"

# ---------------------------------------------------------------------------
# 5. Verify PyTorch + CUDA
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " PyTorch + CUDA Verification"
echo "============================================================"
uv run python - <<'PYEOF'
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PYEOF

# ---------------------------------------------------------------------------
# 6. Set up W&B
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Weights & Biases"
echo "============================================================"
if [ -n "$WANDB_KEY" ]; then
    uv run wandb login "$WANDB_KEY"
    echo "W&B logged in  ✅"
else
    echo "⚠️  WANDB_API_KEY not set. Export it before training:"
    echo "    export WANDB_API_KEY=<your_key>"
fi

# ---------------------------------------------------------------------------
# 7. Print config env detection
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Config Check"
echo "============================================================"
cd "$WORKSPACE"
uv run python -c "
import sys
import os
sys.path.insert(0, os.path.join('$WORKSPACE', 'GPT1-Project'))
from src.utils.config import CONFIG
print(f'CHECKPOINT_DIR : {CONFIG.CHECKPOINT_DIR}')
print(f'DATASET_DIR    : {CONFIG.DATASET_DIR}')
print(f'BATCH_SIZE     : {CONFIG.BATCH_SIZE}')
print(f'LEARNING_RATE  : {CONFIG.LEARNING_RATE}')
print(f'USE_AMP        : {CONFIG.USE_AMP}')
print(f'NUM_WORKERS    : {CONFIG.NUM_WORKERS}')
" 2>/dev/null || echo "(Config check skipped — run manually after cd-ing into $WORKSPACE)"

# ---------------------------------------------------------------------------
# 8. Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Setup complete! Launch training with:"
echo ""
echo "   cd $WORKSPACE"
echo "   uv run python GPT1-Project/scripts/run_training.py"
echo "============================================================"
