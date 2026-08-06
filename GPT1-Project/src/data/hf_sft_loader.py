# src/data/hf_sft_loader.py
#
# Flexible HuggingFace dataset loader for SFT.
# Handles any dataset regardless of column names or structure.
# Outputs a list of {"instruction", "input", "output"} dicts
# that feed directly into SFTDataset.

from datasets import load_dataset, DatasetDict, Dataset
from typing import Optional
import json
import os
import random

# ---------------------------------------------------------------------------
# Known column aliases — add more as you encounter new datasets
# ---------------------------------------------------------------------------
_INSTRUCTION_ALIASES = [
    "instruction", "prompt", "question", "query", "system",
    "human", "user", "task", "context", "input",  # "input" last — low priority to avoid Alpaca collision
]
_OUTPUT_ALIASES = [
    "output", "response", "answer", "completion", "assistant",
    "gpt", "target", "label", "text",
]
_INPUT_ALIASES = [
    "input", "context", "passage", "document", "article", "background",
]


def _detect_columns(columns: list[str]) -> dict[str, Optional[str]]:
    """
    Heuristically map dataset columns to instruction / input / output.
    Returns a dict with keys: instruction, input, output (values may be None).
    """
    cols_lower = {c.lower(): c for c in columns}

    def find(aliases):
        for alias in aliases:
            if alias in cols_lower:
                return cols_lower[alias]
        return None

    instruction_col = find(_INSTRUCTION_ALIASES)
    output_col = find(_OUTPUT_ALIASES)

    # Avoid mapping the same column to both instruction and input
    input_col = None
    for alias in _INPUT_ALIASES:
        candidate = cols_lower.get(alias)
        if candidate and candidate != instruction_col:
            input_col = candidate
            break

    return {"instruction": instruction_col, "input": input_col, "output": output_col}


def _row_to_example(row: dict, mapping: dict[str, Optional[str]]) -> Optional[dict]:
    """Convert one dataset row to {instruction, input, output} using detected mapping."""
    instruction = ""
    inp = ""
    output = ""

    if mapping["instruction"]:
        instruction = str(row.get(mapping["instruction"], "") or "").strip()
    if mapping["input"]:
        inp = str(row.get(mapping["input"], "") or "").strip()
        # If input and instruction map to the same column, clear input
        if mapping["input"] == mapping["instruction"]:
            inp = ""
    if mapping["output"]:
        output = str(row.get(mapping["output"], "") or "").strip()

    # Must have at least an instruction and an output to be usable
    if not instruction or not output:
        return None

    return {"instruction": instruction, "input": inp, "output": output}


def load_from_huggingface(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    column_mapping: Optional[dict[str, str]] = None,
    save_path: Optional[str] = None,
) -> list[dict]:
    """
    Load any HuggingFace dataset and convert it to SFT format.

    Args:
        dataset_name:   HuggingFace dataset name, e.g. "tatsu-lab/alpaca"
        subset:         Dataset config/subset name if required, e.g. "default"
        split:          Which split to use, e.g. "train", "train[:5000]"
        max_samples:    Cap the number of examples (useful for quick tests)
        column_mapping: Manual override — e.g. {"instruction": "prompt", "output": "completion"}
                        Any key not provided is still auto-detected.
        save_path:      If given, saves the converted examples as a JSONL file
                        so future runs skip the download (e.g. "data/sft_data.jsonl")

    Returns:
        List of {"instruction", "input", "output"} dicts ready for SFTDataset.

    Examples:
        # Alpaca-style dataset
        load_from_huggingface("tatsu-lab/alpaca")

        # Custom dataset with non-standard columns
        load_from_huggingface(
            "my-org/my-dataset",
            column_mapping={"instruction": "prompt", "output": "response"}
        )

        # Load only 1000 examples from a large dataset
        load_from_huggingface("Open-Orca/OpenOrca", max_samples=1000)

        # Load and cache to disk for faster future runs
        load_from_huggingface(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft",
            save_path="data/sft_data.jsonl"
        )
    """
    print(f"Loading dataset: {dataset_name}" + (f" ({subset})" if subset else ""))

    raw = load_dataset(dataset_name, subset, split=split) if subset else \
          load_dataset(dataset_name, split=split)

    # load_dataset with a split returns a Dataset, not DatasetDict
    if isinstance(raw, DatasetDict):
        available = list(raw.keys())
        raw = raw[available[0]]
        print(f"  DatasetDict detected — using split '{available[0]}'. "
              f"Available: {available}")

    if not isinstance(raw, Dataset):
        raise TypeError(
            f"Expected a Dataset after loading split '{split}', got {type(raw)}. "
            "Try specifying a split explicitly, e.g. split='train'."
        )
    columns = raw.column_names
    print(f"  Columns found: {columns}")

    # Auto-detect mapping, then apply any manual overrides
    mapping = _detect_columns(columns)
    if column_mapping:
        mapping.update(column_mapping)

    print(f"  Column mapping: {mapping}")

    missing = [k for k, v in mapping.items() if k in ("instruction", "output") and v is None]
    if missing:
        print(
            f"\n  Could not auto-detect columns for: {missing}\n"
            f"  Available columns: {columns}\n"
            f"  Pass column_mapping={{'instruction': '<col>', 'output': '<col>'}} to fix this.\n"
        )
        raise ValueError(f"Cannot map required columns: {missing}")

    # Convert rows
    if max_samples:
        raw = raw.select(range(min(max_samples, len(raw))))

    examples = []
    skipped = 0
    for row in raw:
        ex = _row_to_example(dict(row), mapping)
        if ex:
            examples.append(ex)
        else:
            skipped += 1

    print(f"  Converted: {len(examples)} examples, {skipped} skipped (missing instruction or output)")

    # Shuffle before saving so the cached file matches what's used in training
    random.Random(42).shuffle(examples)

    # Optionally persist to disk so future runs are instant
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Saved to {save_path}")

    return examples


def inspect_dataset(dataset_name: str, subset: Optional[str] = None, split: str = "train", n: int = 3):
    """
    Print a quick summary of a HuggingFace dataset before loading it for SFT.
    Useful for figuring out column names and data structure.

    Usage:
        from src.data.hf_sft_loader import inspect_dataset
        inspect_dataset("tatsu-lab/alpaca")
    """
    raw = load_dataset(dataset_name, subset, split=f"{split}[:{n}]") if subset else \
          load_dataset(dataset_name, split=f"{split}[:{n}]")

    if isinstance(raw, DatasetDict):
        raw = raw[list(raw.keys())[0]]

    print(f"\nDataset: {dataset_name}")
    print(f"Columns: {raw.column_names}")
    print(f"Sample rows ({n}):")
    for i, row in enumerate(raw):
        print(f"\n--- Row {i} ---")
        for col, val in row.items():
            preview = str(val)[:120].replace("\n", " ")
            print(f"  {col}: {preview}")
    print()
