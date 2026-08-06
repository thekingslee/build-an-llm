import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Alpaca-style prompt template.
# When `input` is empty we collapse to the two-section variant.
# ---------------------------------------------------------------------------
_PROMPT_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)
_PROMPT_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def _build_prompt(example: dict) -> str:
    inp = example.get("input", "").strip()
    if inp:
        return _PROMPT_WITH_INPUT.format(
            instruction=example["instruction"], input=inp
        )
    return _PROMPT_NO_INPUT.format(instruction=example["instruction"])


class SFTDataset(Dataset):
    """
    Each example is an (input_ids, labels) pair where labels mirrors
    input_ids but has -100 on every prompt token and every padding position.
    CrossEntropyLoss(ignore_index=-100) then only trains on the response.
    """

    def __init__(self, examples: list[dict], tokenizer, max_len: int):
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        skipped = 0

        for ex in examples:
            prompt = _build_prompt(ex)
            response = ex.get("output", "").strip()
            if not response:
                skipped += 1
                continue

            full_text = prompt + response + tokenizer.eos_token

            # Tokenize separately so we know the exact prompt length in tokens
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)

            # Need at least one response token after the prompt
            if len(full_ids) <= len(prompt_ids):
                skipped += 1
                continue

            # Truncate to max_len + 1 so we can produce a length-max_len shift pair
            full_ids = full_ids[: max_len + 1]
            prompt_len = min(len(prompt_ids), len(full_ids) - 1)

            # Autoregressive shift: x predicts y
            x = full_ids[:-1]  # length <= max_len
            y = full_ids[1:]   # length <= max_len, shifted by 1

            # Mask prompt positions in y with -100 so loss is response-only.
            # At position i in x we predict y[i] = full_ids[i+1].
            # Response tokens first appear in y at index (prompt_len - 1).
            response_start = max(prompt_len - 1, 0)
            y = [-100] * response_start + y[response_start:]

            # Right-pad both sequences to max_len
            seq_len = len(x)
            pad_len = max_len - seq_len
            x = x + [tokenizer.pad_token_id] * pad_len
            y = y + [-100] * pad_len

            self.samples.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
            ))

        print(f"SFTDataset: {len(self.samples)} examples loaded, {skipped} skipped.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_sft_data(path: str) -> list[dict]:
    """Load a JSONL or JSON-array file of {instruction, input?, output} dicts."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        if p.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data)}")
    return data


def prepare_sft_dataloader(examples: list[dict], tokenizer, config, split: str) -> DataLoader:
    dataset = SFTDataset(examples, tokenizer, config.MAX_LEN)
    is_train = split.lower() == "train"
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(config.NUM_WORKERS > 0),
    )
