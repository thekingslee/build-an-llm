import json
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Registry of SFT datasets (all hosted in Alpaca format)
HF_SFT_DATASETS = {
    "extraction":    "anabury/extraction-ner-sft-100k",
    "formatting":    "anabury/text-formatting-sft-100k",
    "conversation":  "anabury/conversation-sft-100k",
    "sentiment":     "Asharox/sentiment-sft-100k",
    "summarization": "Asharox/summarization-sft-100k",
}

_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)

_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def available_datasets() -> list[str]:
    """Returns list of registered dataset aliases."""
    return list(HF_SFT_DATASETS.keys()) + ["local"]


def load_alpaca_format(source: str, split: str = "train", max_samples: int = None) -> list[dict]:
    """
    Loads an Alpaca-formatted dataset from HuggingFace or a local JSON/JSONL file.
    Expects rows to have 'instruction', 'output', and optional 'input'.
    """
    if os.path.exists(source):
        # Load local JSONL / JSON file
        raw_rows = []
        with open(source, "r", encoding="utf-8") as f:
            if source.endswith(".jsonl"):
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if isinstance(item, list):
                                raw_rows.extend(item)
                            else:
                                raw_rows.append(item)
                        except Exception:
                            continue
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, list):
                            raw_rows.extend(item)
                        else:
                            raw_rows.append(item)
                elif isinstance(data, dict):
                    for key in ["data", "rows", "examples", "train", "samples"]:
                        if key in data and isinstance(data[key], list):
                            raw_rows = data[key]
                            break
                    else:
                        raw_rows = [data]
                else:
                    raw_rows = []
    else:
        # Load from Hugging Face
        try:
            raw = load_dataset(source, split=split)
        except Exception:
            raw = load_dataset(source)

        # Handle DatasetDict (when load_dataset returns dict of splits)
        if hasattr(raw, "keys") and not hasattr(raw, "select"):
            if split and split in raw:
                raw = raw[split]
            elif "train" in raw:
                raw = raw["train"]
            else:
                first_key = list(raw.keys())[0]
                raw = raw[first_key]

        if max_samples and hasattr(raw, "select"):
            raw = raw.select(range(min(max_samples, len(raw))))
        raw_rows = raw

    examples = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue

        instruction = str(row.get("instruction") or "").strip()
        output      = str(row.get("output") or "").strip()
        inp         = str(row.get("input") or "").strip()

        if not instruction or not output:
            continue

        examples.append({
            "instruction": instruction,
            "input": inp,
            "output": output,
        })

        if max_samples and len(examples) >= max_samples:
            break

    return examples


def load_sft_datasets(
    names: list[str],
    max_samples_per_dataset: int = None,
    local_path: str = None,
) -> list[dict]:
    """
    Loads and combines one or more datasets by name/alias or Hugging Face ID.
    Cuts up to max_samples_per_dataset from each specified dataset.
    """
    all_examples = []

    for name in names:
        if name == "local":
            if not local_path:
                raise ValueError("Dataset 'local' was specified but no local_path was provided.")
            source = local_path
        elif name in HF_SFT_DATASETS:
            source = HF_SFT_DATASETS[name]
        elif "/" in name:
            source = name
        else:
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {available_datasets()} "
                "or pass a direct HuggingFace repo path (e.g. 'org/dataset-name')."
            )

        print(f"\n[SFT Data] Loading '{name}' from: {source} (max {max_samples_per_dataset or 'all'} samples)...")
        examples = load_alpaca_format(source=source, max_samples=max_samples_per_dataset)
        print(f"[SFT Data] Loaded {len(examples):,} raw examples from '{name}'")
        all_examples.extend(examples)

    print(f"\n[SFT Data] Total combined raw examples: {len(all_examples):,} across {len(names)} dataset(s)")
    return all_examples


class SFTDataset(Dataset):
    def __init__(
        self,
        examples: list[dict],
        tokenizer,
        max_len: int = 512,
        filter_overlength: bool = True,
    ):
        self.samples = []
        self.num_discarded = 0
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        self.total_tokens = 0

        for ex in (examples or []):
            instruction = ex["instruction"]
            inp         = ex.get("input", "")
            output      = ex["output"]

            prompt = (
                _PROMPT_WITH_INPUT.format(instruction=instruction, input=inp)
                if inp.strip()
                else _PROMPT_NO_INPUT.format(instruction=instruction)
            )

            prompt_ids   = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(output)

            total_len = len(prompt_ids) + len(response_ids)
            # Post-processing: dispose examples exceeding context length limit
            if filter_overlength and total_len > max_len:
                self.num_discarded += 1
                continue

            full_ids = prompt_ids + response_ids
            if len(full_ids) > max_len + 1:
                full_ids = full_ids[: max_len + 1]

            x = full_ids[:-1]
            y = full_ids[1:]

            prompt_len     = min(len(prompt_ids), len(x))
            response_start = max(prompt_len - 1, 0)
            target_tokens  = len(y[response_start:])
            y = [-100] * response_start + y[response_start:]

            self.total_prompt_tokens += response_start
            self.total_response_tokens += target_tokens
            self.total_tokens += len(x)

            pad_len = max_len - len(x)
            x = x + [tokenizer.pad_token_id] * pad_len
            y = y + [-100] * pad_len

            self.samples.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_stats(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_response_tokens": self.total_response_tokens,
            "num_discarded": self.num_discarded,
            "num_samples": len(self.samples),
        }


def load_or_create_sft_splits(tokenizer, cfg):
    """
    Loads raw examples from specified datasets, tokenizes in-memory, and splits into train/val/test.
    Returns (train_ds, val_ds, test_ds, total_raw_count).
    """
    # Load from raw Hugging Face datasets / local files
    all_examples = load_sft_datasets(
        names=cfg.DATASET_NAMES,
        local_path=cfg.LOCAL_DATA_PATH,
        max_samples_per_dataset=cfg.MAX_SAMPLES_PER_DATASET,
    )
    random.Random(42).shuffle(all_examples)

    val_ratio  = cfg.VAL_SPLIT
    test_ratio = getattr(cfg, "TEST_SPLIT", 0.0)

    val_size   = int(len(all_examples) * val_ratio)
    test_size  = int(len(all_examples) * test_ratio)
    train_size = len(all_examples) - val_size - test_size

    train_examples = all_examples[:train_size]
    val_examples   = all_examples[train_size:train_size + val_size]
    test_examples  = all_examples[train_size + val_size:]

    filter_overlength = getattr(cfg, "FILTER_OVERLENGTH", True)

    print("\n[SFT Data] Tokenizing and formatting splits in memory...")
    train_ds = SFTDataset(train_examples, tokenizer, cfg.MAX_LEN, filter_overlength=filter_overlength)
    val_ds   = SFTDataset(val_examples,   tokenizer, cfg.MAX_LEN, filter_overlength=filter_overlength)
    test_ds  = (
        SFTDataset(test_examples, tokenizer, cfg.MAX_LEN, filter_overlength=filter_overlength)
        if test_size > 0 else None
    )

    return train_ds, val_ds, test_ds, len(all_examples)


def create_sft_dataloader(dataset: SFTDataset, cfg, is_train: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=is_train,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY and cfg.DEVICE == "cuda",
    )
