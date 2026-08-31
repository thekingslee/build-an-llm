import os
import torch
from dataclasses import dataclass, field

from src.utils.config import _IS_RUNPOD, _IS_COLAB, _CHECKPOINT_DIR, _DATASET_DIR

_SFT_CHECKPOINT_DIR = os.path.join(_CHECKPOINT_DIR, "sft")

_SFT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "sft_data.jsonl"
)


@dataclass
class SFTConfig:
    PRETRAINED_CHECKPOINT: str = field(
        default_factory=lambda: os.path.join(_CHECKPOINT_DIR, "best_model.pt")
    )

    DATASET_NAMES: list = field(
        default_factory=lambda: [
            "sentiment",
            "extraction",
            "formatting",
            "conversation",
            # "summarization",
        ]
    )
    LOCAL_DATA_PATH: str = field(default_factory=lambda: _SFT_DATA_PATH)
    MAX_SAMPLES_PER_DATASET: int | None = 10_000  # Number of samples to slice per dataset source (e.g. 10k each)
    VAL_SPLIT: float = 0.1
    TEST_SPLIT: float = 0.05
    FILTER_OVERLENGTH: bool = True  # Discard examples exceeding MAX_LEN instead of truncating

    BATCH_SIZE: int = 128 
    EPOCHS: int = 3
    LEARNING_RATE: float = 5.7e-5
    WARMUP_STEPS: int = 200
    GRAD_CLIP: float = 1.0

    EMBED_SIZE: int = 256
    NUM_LAYERS: int = 4
    HEADS: int = 4
    MAX_LEN: int = 512

    FREEZE_EMBEDDINGS: bool = False

    CHECKPOINT_DIR: str = field(default_factory=lambda: _SFT_CHECKPOINT_DIR)
    KEEP_LAST_N_CHECKPOINTS: int = 3

    EARLY_STOPPING_PATIENCE: int = 2
    EARLY_STOPPING_MIN_DELTA: float = 1e-4

    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True

    USE_AMP: bool = True

    DEVICE: str = field(
        default_factory=lambda: (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    SFT_DATA_PATH: str = field(default_factory=lambda: _SFT_DATA_PATH)


SFT_CONFIG = SFTConfig()
