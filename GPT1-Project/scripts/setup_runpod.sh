#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/thekingslee/build-an-llm.git"
REPO_BRANCH="feat/continued-pretraining"
WORKSPACE="/workspace/build-an-llm"

# --- Optional Weights & Biases Configuration ---
# Paste your W&B API Key below (e.g. WANDB_API_KEY="your_api_key_here")
# If left blank, you can also uncomment WANDB_MODE="disabled" to turn it off completely.
export WANDB_API_KEY=""
export WANDB_MODE="online" # Set to "disabled" or "offline" if you don't want W&B logging

# 1. Load github repo
if [ -d "$WORKSPACE/.git" ]; then
    git -C "$WORKSPACE" fetch origin && git -C "$WORKSPACE" checkout "$REPO_BRANCH" && git -C "$WORKSPACE" pull origin "$REPO_BRANCH"
else
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORKSPACE"
fi

cd "$WORKSPACE"

# 2. Setup & install project dependencies
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
uv sync

if [ -n "${WANDB_API_KEY:-}" ]; then
    uv run wandb login "$WANDB_API_KEY"
fi

# Create needed directories silently
mkdir -p "$WORKSPACE/checkpoints" "$WORKSPACE/datasets" "$WORKSPACE/hf_cache" "$WORKSPACE/wandb" "$WORKSPACE/tmp"

# 3. Install tmux
if ! command -v tmux &>/dev/null; then
    apt-get update -y && apt-get install -y tmux
fi

# 4 & 5. Start tmux and run the training
tmux new-session -d -s training "cd \$WORKSPACE && export HF_HOME=\$WORKSPACE/hf_cache WANDB_DIR=\$WORKSPACE/wandb TMPDIR=\$WORKSPACE/tmp WANDB_API_KEY='$WANDB_API_KEY' WANDB_MODE='$WANDB_MODE' && uv run python GPT1-Project/scripts/run_training.py"
echo "Setup complete! Training has started in the background."
echo "To view training progress, run: tmux attach -t training"
