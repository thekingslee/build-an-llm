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
    BATCH_SIZE: int = 16 if (_IS_RUNPOD or _IS_COLAB) else 4
    EPOCHS: int = 3
    LEARNING_RATE: float = 2e-5     # much lower than pretraining to preserve weights
    WARMUP_STEPS: int = 100
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
