# src/data/nigerian_corpus.py

from torch.utils.data import Dataset, DataLoader
import torch
from transformers import GPT2Tokenizer

class GPTDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.data = torch.tensor(tokens)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

def load_tokens(file_path, tokenizer):
    """
    Load the Nigerian Book Corpus, tokenize, and return train/test tokens
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # For simplicity, use 90% train, 10% test
    split_idx = int(0.9 * len(lines))
    train_texts = lines[:split_idx]
    test_texts = lines[split_idx:]

    # Tokenize
    train_tokens = []
    for text in train_texts:
        train_tokens.extend(tokenizer.encode(text))

    test_tokens = []
    for text in test_texts:
        test_tokens.extend(tokenizer.encode(text))

    return train_tokens, test_tokens

def prepare_dataloader(tokens, config):
    """
    Wrap GPTDataset in a DataLoader
    """
    dataset = GPTDataset(tokens, config.SEQ_LEN)
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
