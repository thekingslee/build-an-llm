#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/thekingslee/build-an-llm.git"
REPO_BRANCH="feat/SFT"
WORKSPACE="/workspace/build-an-llm"

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
mkdir -p "$WORKSPACE/checkpoints/sft" "$WORKSPACE/datasets" "$WORKSPACE/hf_cache" "$WORKSPACE/wandb"

# Verify pretrained checkpoint exists
if [ ! -f "$WORKSPACE/checkpoints/best_model.pt" ]; then
    echo "⚠️  Note: Pretrained checkpoint '$WORKSPACE/checkpoints/best_model.pt' not found."
    echo "   Ensure your pretrained base model weights are in '$WORKSPACE/checkpoints/' before SFT begins."
fi

# 3. Install tmux
if ! command -v tmux &>/dev/null; then
    apt-get update -y && apt-get install -y tmux
fi

# 4 & 5. Start tmux and run SFT training
tmux new-session -d -s sft_training "cd $WORKSPACE && export HF_HOME=$WORKSPACE/hf_cache WANDB_DIR=$WORKSPACE/wandb && uv run python GPT1-Project/scripts/run_sft.py"
echo "Setup complete! SFT Training has started in the background."
echo "To view SFT training progress, run: tmux attach -t sft_training"
