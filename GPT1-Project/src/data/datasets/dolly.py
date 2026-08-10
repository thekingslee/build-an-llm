# src/data/datasets/dolly.py
# Dataset: databricks/databricks-dolly-15k
# 15k high-quality human-written instruction examples across 8 task categories.
# Columns: instruction, context, response, category

from datasets import load_dataset


def load(max_samples=None) -> list[dict]:
    raw = load_dataset("databricks/databricks-dolly-15k", split="train")
    if max_samples:
        raw = raw.select(range(min(max_samples, len(raw))))

    examples = []
    for row in raw:
        instruction = (row.get("instruction") or "").strip()
        output = (row.get("response") or "").strip()
        if not instruction or not output:
            continue
        examples.append({
            "instruction": instruction,
            "input": (row.get("context") or "").strip(),  # context → input
            "output": output,
        })

    return examples
