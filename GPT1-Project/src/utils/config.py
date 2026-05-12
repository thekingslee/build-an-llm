from dataclasses import dataclass

# 

@dataclass
class Config:
    # ---------------- Data & Training ----------------
    BATCH_SIZE: int = 160 #64  # GPT-1 original: 64 minibatch size
    SEQ_LEN: int = 512  # Context length for training sequences: 512 tokens
    EPOCHS: int = 100  # GPT-1 original: 100 epochs
    LEARNING_RATE: float = 3e-4  # GPT-1 original: 2.5e-4 max learning rate
    TARGET_LOSS: float = 1.5
    EARLY_STOPPING_PATIENCE: int = 3
    EARLY_STOPPING_MIN_DELTA: float = 1e-4

    # ------------- Model Architecture -------------
    # FOR A 30M parameter model
    EMBED_SIZE: int = 256 #768
    NUM_LAYERS: int = 4 #12
    HEADS: int = 4 #12
    MAX_LEN: int = 512

    # --------- Scheduler & Checkpointing ---------
    TOTAL_TRAINING_STEPS: int = 100000
    WARMUP_STEPS: int = 2000
    SAVE_EVERY: int = 2000
    # CHECKPOINT_DIR: str = "./checkpoints"
    CHECKPOINT_DIR: str = "/content/drive/MyDrive/GPT1_Checkpoints" # Adjusted for Google Drive in Colab
    DATASET_DIR: str = "/content/drive/MyDrive/GPT1_Checkpoints/GPT1_Datasets" # Adjusted for Google Drive in Colab

    # ------------------- Device -------------------
    DEVICE: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# This instance is what you import
CONFIG = Config()
