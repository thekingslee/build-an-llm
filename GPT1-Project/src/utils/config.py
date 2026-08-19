import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Environment detection — drives all path decisions automatically.
# RunPod mounts the Network Volume at /runpod-volume by default.
# ---------------------------------------------------------------------------
_IS_RUNPOD = os.path.exists("/runpod-volume")
_IS_COLAB  = os.path.exists("/content/drive")

if _IS_RUNPOD:
    _CHECKPOINT_DIR = "/workspace/build-an-llm/checkpoints"
    _DATASET_DIR    = "/workspace/build-an-llm/datasets"
elif _IS_COLAB:
    _CHECKPOINT_DIR = "/content/drive/MyDrive/GPT1_Checkpoints"
    _DATASET_DIR    = "/content/drive/MyDrive/GPT1_Checkpoints/GPT1_Datasets"
else:
    _CHECKPOINT_DIR = "./checkpoints"
    _DATASET_DIR    = "./data"


@dataclass
class Config:
    # ---------------- Data & Training ----------------
    BATCH_SIZE: int   = 128 if (_IS_RUNPOD or _IS_COLAB) else 16      # Lowered to 16 locally to prevent MPS OOM (12.27GB buffer error)
    SEQ_LEN:    int   = 512      # Context length for training sequences
    STRIDE:     int   = 320        # ~37% overlap, clearly uneven, 1.6x dataset
    EPOCHS:     int   = 20       # Chinchilla-optimal is ~2 epochs; early stopping fires around epoch 4-7
    LEARNING_RATE: float = 4.2e-4  # Adjusted from 6e-4 for batch size 128 (sqrt scaling rule)

    # ---------------- Early Stopping ----------------
    TARGET_LOSS:               float = 1.5
    EARLY_STOPPING_PATIENCE:   int   = 3
    EARLY_STOPPING_MIN_DELTA:  float = 3e-4  # Adjusted for batch 128 (less smooth than 256, but smoother than 64)

    # ------------- Model Architecture -------------
    # FOR A 30M parameter model
    EMBED_SIZE: int = 256 #768
    NUM_LAYERS: int = 4 #12
    HEADS: int = 4 #12
    MAX_LEN: int = 512

    # --------- Scheduler & Checkpointing ---------
    TOTAL_TRAINING_STEPS:     int = 100_000
    WARMUP_STEPS:             int = 500    # Token-scaled from 2000: keeps warmup ~65M tokens seen
    SAVE_EVERY:               int = 1000    # Step checkpoint frequency (matches new steps/epoch scale)
    KEEP_LAST_N_CHECKPOINTS:  int = 5      # Rotate step checkpoints; keep only last N to save disk

    # ---------------- Paths (env-aware) ----------------
    CHECKPOINT_DIR: str = field(default_factory=lambda: _CHECKPOINT_DIR)
    DATASET_DIR:    str = field(default_factory=lambda: _DATASET_DIR)

    # ---------------- DataLoader ----------------
    NUM_WORKERS: int  = 4     # Parallel data workers — RunPod pods have 8-16 CPUs, use them
    PIN_MEMORY:  bool = True  # Faster CPU->GPU transfer (set False if running on CPU-only)

    # ---------------- Mixed Precision ----------------
    USE_AMP: bool = True  # torch.cuda.amp autocast + GradScaler — free 1.5-2x speedup on 4090

    # ------------------- Device -------------------
    DEVICE: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"


CONFIG = Config()
