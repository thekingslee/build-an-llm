from torch.utils.data import Dataset, DataLoader
import os
import torch
from src.utils.config import CONFIG

def load_tokens_from_drive(dataset_name):
    """
    Load tokenized train/test splits from saved cache in Drive.
    """
    saved_path = os.path.join(CONFIG.CHECKPOINT_DIR, dataset_name)

    if not os.path.exists(saved_path):
        raise FileNotFoundError(
            f"Required cached tokenized splits not found at {saved_path}. "
        )

    try:
        cached = torch.load(saved_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Could not load cached tokenized splits from {saved_path}: {e}") from e

    train_tokens = cached.get("train_tokens")
    val_tokens = cached.get("val_tokens")
    test_tokens = cached.get("test_tokens")

    if train_tokens is None or test_tokens is None:
        raise KeyError(
            f"Cached split file at {saved_path} is missing required keys "
            "'train_tokens' and/or 'test_tokens'."
        )

    print(f"Loaded cached tokenized splits from {saved_path}")
    return train_tokens, val_tokens, test_tokens
