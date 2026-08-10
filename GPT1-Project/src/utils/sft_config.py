import os
import torch
from dataclasses import dataclass, field

from src.utils.config import _IS_RUNPOD, _IS_COLAB, _CHECKPOINT_DIR

_SFT_CHECKPOINT_DIR = os.path.join(_CHECKPOINT_DIR, "sft")

# Absolute path to the sample data file — works regardless of working directory
_SFT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "sft_data.jsonl"
)


@dataclass
class SFTConfig:
    # -------- Source checkpoint --------
    # Path to the pretrained best_model.pt to fine-tune from.
    PRETRAINED_CHECKPOINT: str = field(
        default_factory=lambda: os.path.join(_CHECKPOINT_DIR, "best_model.pt")
    )

    # -------- SFT dataset --------
    # JSONL or JSON array with {"instruction", "input"?, "output"} per example.
    # Override via --sft-data-path CLI arg in run_sft.py.
    SFT_DATA_PATH: str = field(default_factory=lambda: _SFT_DATA_PATH)
    VAL_SPLIT: float = 0.1          # fraction held out for validation

    # -------- Training --------
    # Batch sizes: RunPod 128, Colab 16, local 4
    BATCH_SIZE: int = 128 if _IS_RUNPOD else (16 if _IS_COLAB else 4)
    EPOCHS: int = 3
    # LR is sqrt-scaled from the local baseline (batch=4, LR=2e-5):
    #   RunPod  (128): 2e-5 × sqrt(128/4) = 2e-5 × 5.66 ≈ 1.1e-4  — but capped lower to preserve weights
    #   Colab   (16):  2e-5 × sqrt(16/4)  = 2e-5 × 2.0  = 4e-5
    #   Local   (4):   2e-5 (baseline)
    LEARNING_RATE: float = (
        5.7e-5 if _IS_RUNPOD else   # sqrt(128/16) × 2e-5; stays conservative for SFT
        4e-5   if _IS_COLAB  else
        2e-5
    )
    WARMUP_STEPS: int = 200         # scaled up from 100 to account for larger batch steps
    GRAD_CLIP: float = 1.0

    # -------- Architecture (must match pretrained checkpoint exactly) --------
    EMBED_SIZE: int = 256
    NUM_LAYERS: int = 4
    HEADS: int = 4
    MAX_LEN: int = 512

    # -------- Layer freezing --------
    # Freeze token + position embeddings during fine-tuning.
    # Recommended when SFT dataset is small (< ~5k examples).
    FREEZE_EMBEDDINGS: bool = False

    # -------- Checkpointing --------
    CHECKPOINT_DIR: str = field(default_factory=lambda: _SFT_CHECKPOINT_DIR)
    KEEP_LAST_N_CHECKPOINTS: int = 3

    # -------- Early stopping --------
    EARLY_STOPPING_PATIENCE: int = 2
    EARLY_STOPPING_MIN_DELTA: float = 1e-4

    # -------- DataLoader --------
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True

    # -------- Mixed precision --------
    USE_AMP: bool = True

    # -------- Device (auto-detected) --------
    DEVICE: str = field(
        default_factory=lambda: (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    )


SFT_CONFIG = SFTConfig()
