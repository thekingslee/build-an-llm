# scripts/run_sft.py

import os
import sys
import argparse

# Ensure project root is in sys.path so src modules resolve correctly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.sft_config import SFT_CONFIG
from src.training.sft_train import sft_train


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for GPT-1 30M")
    parser.add_argument(
        "--sft-data-path",
        type=str,
        default=None,
        help="Path to JSONL or JSON SFT dataset (instruction/input/output format). "
             "Defaults to SFTConfig.SFT_DATA_PATH.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Path to pretrained best_model.pt checkpoint. "
             "Defaults to SFTConfig.PRETRAINED_CHECKPOINT.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of SFT epochs. Defaults to SFTConfig.EPOCHS.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Defaults to SFTConfig.LEARNING_RATE.",
    )
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        default=None,
        help="Freeze token and position embeddings during fine-tuning.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = SFT_CONFIG

    # Apply any CLI overrides
    if args.sft_data_path:
        cfg.SFT_DATA_PATH = args.sft_data_path
    if args.pretrained_checkpoint:
        cfg.PRETRAINED_CHECKPOINT = args.pretrained_checkpoint
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.lr is not None:
        cfg.LEARNING_RATE = args.lr
    if args.freeze_embeddings:
        cfg.FREEZE_EMBEDDINGS = True

    sft_train(cfg)


if __name__ == "__main__":
    main()
