# src/data/datasets/openorca.py
# Dataset: Open-Orca/OpenOrca
# ~1M GPT-4/3.5 generated reasoning traces aligned to FLAN collection.
# Columns: id, system_prompt, question, response
# system_prompt is prepended to question to form the full instruction.

from datasets import load_dataset


def load(max_samples=None) -> list[dict]:
    raw = load_dataset("Open-Orca/OpenOrca", split="train")
    if max_samples:
        raw = raw.select(range(min(max_samples, len(raw))))

    examples = []
    for row in raw:
        system_prompt = (row.get("system_prompt") or "").strip()
        question = (row.get("question") or "").strip()
        output = (row.get("response") or "").strip()

        if not question or not output:
            continue

        # Combine system prompt + question into a single instruction
        instruction = f"{system_prompt}\n\n{question}" if system_prompt else question

        examples.append({
            "instruction": instruction,
            "input": "",
            "output": output,
        })

    return examples
