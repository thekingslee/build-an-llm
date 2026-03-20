from dataclasses import dataclass

@dataclass
class Config:
    # ---------------- Data & Training ----------------
    BATCH_SIZE: int = 16
    SEQ_LEN: int = 128
    EPOCHS: int = 10
    LEARNING_RATE: float = 3e-4
    TARGET_LOSS: float = 1.5

    # ------------- Model Architecture -------------
    EMBED_SIZE: int = 768
    NUM_LAYERS: int = 12
    HEADS: int = 12
    MAX_LEN: int = 512

    # --------- Scheduler & Checkpointing ---------
    TOTAL_TRAINING_STEPS: int = 100000
    WARMUP_STEPS: int = 2000
    SAVE_EVERY: int = 500
    CHECKPOINT_DIR: str = "./checkpoints"

    # ------------------- Device -------------------
    DEVICE: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# This instance is what you import
CONFIG = Config()
