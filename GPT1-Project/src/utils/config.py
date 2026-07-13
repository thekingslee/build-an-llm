from dataclasses import dataclass

@dataclass
class Config:
    # ---------------- Data & Training ----------------
    BATCH_SIZE: int = 8  # Scaled down from GPT-1's 64 to fit 8GB unified memory (M1)
    SEQ_LEN: int = 128  # Scaled down from 512 for the same reason
    EPOCHS: int = 100  # GPT-1 original: 100 epochs
    LEARNING_RATE: float = 3e-4  # GPT-1 original: 2.5e-4 max learning rate
    TARGET_LOSS: float = 1.5
    EARLY_STOPPING_PATIENCE: int = 3
    EARLY_STOPPING_MIN_DELTA: float = 1e-4

    # ------------- Model Architecture -------------
    # 30M-param config (4/4/256) from the README's table, instead of the
    # 110M config (12/12/768) which doesn't fit in 8GB unified memory.
    EMBED_SIZE: int = 256
    NUM_LAYERS: int = 4
    HEADS: int = 4
    MAX_LEN: int = 128

    # --------- Scheduler & Checkpointing ---------
    TOTAL_TRAINING_STEPS: int = 100000
    WARMUP_STEPS: int = 2000
    SAVE_EVERY: int = 2000
    CHECKPOINT_DIR: str = "./checkpoints"
    # CHECKPOINT_DIR: str = "/content/drive/MyDrive/GPT1_Checkpoints" # Use this path instead when running in Colab

    # ------------------- Device -------------------
    DEVICE: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# This instance is what you import
CONFIG = Config()
