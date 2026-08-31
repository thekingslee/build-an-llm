import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.sft_dataset import available_datasets
from src.utils.sft_config import SFT_CONFIG
from src.training.sft_train import sft_train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning for GPT-1 30M",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "One or more dataset names to load and combine at training time.\n"
            f"Available: {available_datasets()}\n"
            "Examples:\n"
            "  --datasets extraction formatting conversation\n"
            "  --datasets extraction\n"
            "  --datasets local --local-data-path data/my_sft.jsonl\n"
            "  --datasets anabury/extraction-ner-sft-100k"
        ),
    )
    parser.add_argument(
        "--local-data-path",
        type=str,
        default=None,
        help="Path to a local JSONL/JSON file. Required when 'local' is in --datasets.",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Cap examples per dataset source.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Path to pretrained best_model.pt.",
    )
    parser.add_argument("--epochs", type=int,   default=None)
    parser.add_argument("--lr",     type=float, default=None)
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Freeze token and position embeddings during fine-tuning.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = SFT_CONFIG

    if args.datasets:
        cfg.DATASET_NAMES = args.datasets
    if args.local_data_path:
        cfg.LOCAL_DATA_PATH = args.local_data_path
    if args.max_samples_per_dataset is not None:
        cfg.MAX_SAMPLES_PER_DATASET = args.max_samples_per_dataset
    if args.pretrained_checkpoint:
        cfg.PRETRAINED_CHECKPOINT = args.pretrained_checkpoint
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.lr is not None:
        cfg.LEARNING_RATE = args.lr
    if args.freeze_embeddings:
        cfg.FREEZE_EMBEDDINGS = True

    sft_train(cfg=cfg)


if __name__ == "__main__":
    main()
