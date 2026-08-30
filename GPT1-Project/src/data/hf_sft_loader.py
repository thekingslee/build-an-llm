from datasets import load_dataset, DatasetDict, Dataset
from typing import Optional
import json, os, random

_INSTRUCTION_ALIASES = [
    "instruction", "prompt", "question", "query", "system",
    "human", "user", "task", "context", "input",
]
_OUTPUT_ALIASES = [
    "output", "response", "answer", "completion", "assistant",
    "gpt", "target", "label", "text",
]
_INPUT_ALIASES = [
    "input", "context", "passage", "document", "article", "background",
]


def _detect_columns(columns: list[str]) -> dict[str, Optional[str]]:
    cols_lower = {c.lower(): c for c in columns}

    def find(aliases):
        for alias in aliases:
            if alias in cols_lower:
                return cols_lower[alias]
        return None

    instruction_col = find(_INSTRUCTION_ALIASES)
    output_col      = find(_OUTPUT_ALIASES)

    input_col = None
    for alias in _INPUT_ALIASES:
        candidate = cols_lower.get(alias)
        if candidate and candidate != instruction_col:
            input_col = candidate
            break

    return {"instruction": instruction_col, "input": input_col, "output": output_col}


def _row_to_example(row: dict, mapping: dict) -> Optional[dict]:
    instruction = str(row.get(mapping["instruction"], "") or "").strip() if mapping["instruction"] else ""
    inp         = str(row.get(mapping["input"],        "") or "").strip() if mapping["input"]        else ""
    output      = str(row.get(mapping["output"],       "") or "").strip() if mapping["output"]       else ""

    if mapping.get("input") and mapping["input"] == mapping["instruction"]:
        inp = ""

    if not instruction or not output:
        return None

    return {"instruction": instruction, "input": inp, "output": output}


def load_from_huggingface(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    column_mapping: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> list[dict]:
    print(f"Loading dataset: {dataset_name}" + (f" ({subset})" if subset else ""))

    raw = (load_dataset(dataset_name, subset, split=split) if subset
           else load_dataset(dataset_name, split=split))

    if isinstance(raw, DatasetDict):
        available = list(raw.keys())
        raw = raw[available[0]]
        print(f"  DatasetDict detected — using split '{available[0]}'. Available: {available}")

    if not isinstance(raw, Dataset):
        raise TypeError(
            f"Expected a Dataset after loading split '{split}', got {type(raw)}. "
            "Try specifying a split explicitly, e.g. split='train'."
        )

    print(f"  Columns found: {raw.column_names}")

    mapping = _detect_columns(raw.column_names)
    if column_mapping:
        mapping.update(column_mapping)
    print(f"  Column mapping: {mapping}")

    missing = [k for k, v in mapping.items() if k in ("instruction", "output") and v is None]
    if missing:
        raise ValueError(
            f"Cannot map required columns: {missing}. Available: {raw.column_names}\n"
            f"Pass column_mapping={{'instruction': '<col>', 'output': '<col>'}} to fix this."
        )

    if max_samples:
        raw = raw.select(range(min(max_samples, len(raw))))

    examples, skipped = [], 0
    for row in raw:
        ex = _row_to_example(dict(row), mapping)
        if ex:
            examples.append(ex)
        else:
            skipped += 1

    print(f"  Converted: {len(examples)} examples, {skipped} skipped")

    random.Random(42).shuffle(examples)

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Saved to {save_path}")

    return examples


def inspect_dataset(dataset_name: str, subset: Optional[str] = None, split: str = "train", n: int = 3):
    raw = (load_dataset(dataset_name, subset, split=f"{split}[:{n}]") if subset
           else load_dataset(dataset_name, split=f"{split}[:{n}]"))

    if isinstance(raw, DatasetDict):
        raw = raw[list(raw.keys())[0]]

    print(f"\nDataset: {dataset_name}")
    print(f"Columns: {raw.column_names}")
    for i, row in enumerate(raw):
        print(f"\n--- Row {i} ---")
        for col, val in row.items():
            print(f"  {col}: {str(val)[:120].replace(chr(10), ' ')}")


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Prepare a HuggingFace dataset as a local JSONL file for SFT training."
    )
    parser.add_argument("--dataset",        required=True)
    parser.add_argument("--subset",         default=None)
    parser.add_argument("--split",          default="train")
    parser.add_argument("--max-samples",    type=int, default=None)
    parser.add_argument("--column-mapping", default=None)
    parser.add_argument("--save-path",      default=None)
    parser.add_argument("--inspect",        action="store_true")
    parser.add_argument("--inspect-n",      type=int, default=3)

    args    = parser.parse_args()
    col_map = json.loads(args.column_mapping) if args.column_mapping else None

    if args.inspect:
        inspect_dataset(args.dataset, subset=args.subset, split=args.split, n=args.inspect_n)
        sys.exit(0)

    examples = load_from_huggingface(
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        max_samples=args.max_samples,
        column_mapping=col_map,
        save_path=args.save_path,
    )

    if not args.save_path:
        for ex in examples[:3]:
            print(json.dumps(ex, ensure_ascii=False, indent=2))
        print(f"\nTotal: {len(examples)} examples. Pass --save-path to write to disk.")
