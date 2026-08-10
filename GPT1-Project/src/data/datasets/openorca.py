from datasets import load_dataset


def load(max_samples=None) -> list[dict]:
    raw = load_dataset("Open-Orca/OpenOrca", split="train")
    if max_samples:
        raw = raw.select(range(min(max_samples, len(raw))))

    examples = []
    for row in raw:
        system_prompt = (row.get("system_prompt") or "").strip()
        question      = (row.get("question") or "").strip()
        output        = (row.get("response") or "").strip()

        if not question or not output:
            continue

        instruction = f"{system_prompt}\n\n{question}" if system_prompt else question

        examples.append({
            "instruction": instruction,
            "input": "",
            "output": output,
        })

    return examples
