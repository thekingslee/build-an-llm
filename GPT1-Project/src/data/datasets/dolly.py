from datasets import load_dataset

def load(max_samples=None) -> list[dict]:
    raw = load_dataset("databricks/databricks-dolly-15k", split="train")
    if max_samples:
        raw = raw.select(range(min(max_samples, len(raw))))

    examples = []
    for row in raw:
        instruction = (row.get("instruction") or "").strip()
        output      = (row.get("response") or "").strip()
        if not instruction or not output:
            continue
        examples.append({
            "instruction": instruction,
            "input": (row.get("context") or "").strip(),
            "output": output,
        })

    return examples
