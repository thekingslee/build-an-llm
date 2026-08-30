import json
from pathlib import Path


def load(path: str, max_samples=None) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Local SFT dataset not found: {path}\n"
            "Either provide a valid --local-data-path or choose a HuggingFace dataset "
            "with --datasets alpaca / dolly / openorca."
        )

    with open(p, encoding="utf-8") as f:
        if p.suffix == ".jsonl":
            rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {path}, got {type(rows)}")

    examples = []
    for row in rows:
        instruction = (row.get("instruction") or "").strip()
        output      = (row.get("output") or "").strip()
        if not instruction or not output:
            continue
        examples.append({
            "instruction": instruction,
            "input": (row.get("input") or "").strip(),
            "output": output,
        })

    if max_samples:
        examples = examples[:max_samples]

    return examples
