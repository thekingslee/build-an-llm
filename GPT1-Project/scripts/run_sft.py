# scripts/run_sft.py

import os
import sys
import argparse
import json

# Ensure project root is in sys.path so src modules resolve correctly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.sft_config import SFT_CONFIG
from src.training.sft_train import sft_train


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for GPT-1 30M")

    # ---- Data source (pick one) ----
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--sft-data-path",
        type=str,
        default=None,
        help="Path to a local JSONL or JSON SFT dataset (instruction/input/output format).",
    )
    data_group.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name to load, e.g. 'tatsu-lab/alpaca'. "
             "The loader auto-detects columns. Use --column-mapping to override.",
    )

    # ---- HuggingFace dataset options ----
    parser.add_argument(
        "--hf-subset",
        type=str,
        default=None,
        help="HuggingFace dataset subset/config name if required.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="HuggingFace dataset split to use (default: train). "
             "Supports slice notation, e.g. 'train[:5000]'.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of examples loaded from HuggingFace (useful for quick tests).",
    )
    parser.add_argument(
        "--column-mapping",
        type=str,
        default=None,
        help='Manual column mapping as JSON, e.g. \'{"instruction":"prompt","output":"completion"}\'. '
             "Only needed when auto-detection fails.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print a preview of the HuggingFace dataset columns and sample rows, then exit. "
             "Use this to figure out column names before running full SFT.",
    )

    # ---- Training overrides ----
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Path to pretrained best_model.pt. Defaults to SFTConfig.PRETRAINED_CHECKPOINT.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of SFT epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate.",
    )
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Freeze token and position embeddings during fine-tuning.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = SFT_CONFIG

    # Apply training overrides
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

    # Parse column mapping JSON if provided
    column_mapping = None
    if args.column_mapping:
        column_mapping = json.loads(args.column_mapping)

    # --inspect: preview the dataset then exit
    if args.inspect:
        if not args.hf_dataset:
            print("--inspect requires --hf-dataset")
            sys.exit(1)
        from src.data.hf_sft_loader import inspect_dataset
        inspect_dataset(args.hf_dataset, subset=args.hf_subset, split=args.hf_split)
        sys.exit(0)

    sft_train(
        cfg=cfg,
        hf_dataset=args.hf_dataset,
        hf_subset=args.hf_subset,
        hf_split=args.hf_split,
        max_samples=args.max_samples,
        column_mapping=column_mapping,
    )


if __name__ == "__main__":
    main()
