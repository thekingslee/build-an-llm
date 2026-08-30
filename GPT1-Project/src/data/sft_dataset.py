from torch.utils.data import Dataset, DataLoader
import torch

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

class SFTDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.samples = []
        for ex in examples:
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

            full_ids = prompt_ids + response_ids
            if len(full_ids) > max_len + 1:
                full_ids = full_ids[: max_len + 1]

            x = full_ids[:-1]
            y = full_ids[1:]

            prompt_len     = min(len(prompt_ids), len(x))
            response_start = max(prompt_len - 1, 0)
            y = [-100] * response_start + y[response_start:]

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

def prepare_sft_dataloader(examples, tokenizer, cfg, split="train"):
    dataset  = SFTDataset(examples, tokenizer, cfg.MAX_LEN)
    is_train = split == "train"
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=is_train,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY and cfg.DEVICE == "cuda",
    )
