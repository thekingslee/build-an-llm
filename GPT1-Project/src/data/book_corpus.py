# src/data/book_corpus.py

from torch.utils.data import Dataset, DataLoader
import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset
from src.data.ablation_dataset import load_tokens_from_drive
from src.utils.config import CONFIG
import random

class GPTDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.data = torch.tensor(tokens)
        self.seq_len = seq_len

    def __len__(self):
        # Ensure we return at least 0, even if we don't have enough tokens
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

def load_local_corpus_data(data_dir):
    """
    Load all .txt files from the data directory
    Returns combined lines from all text files found
    """
    import os
    import glob
    
    try:
        # Get all .txt files in the data directory
        txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
        
        if not txt_files:
            return []
        
        all_lines = []
        for txt_file in txt_files:
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                # Clean and filter lines
                clean_lines = [line.strip() for line in lines if line.strip()]
                all_lines.extend(clean_lines)
                                
            except Exception as e:
                print(f"  ⚠️  Error loading {txt_file}: {e}")
                continue
        
        return all_lines
        
    except Exception as e:
        print(f"⚠️  Error accessing directory {data_dir}: {e}")
        return []

def load_huggingface_corpus_data(source):
    """
    Load BookCorpus from HuggingFace
    """
    try:
        dataset = load_dataset(source, split="train")
        texts = [item['text'] for item in dataset if item['text'].strip()]
        return texts
    except Exception as e:
        print(f"⚠️  Error loading HuggingFace corpus: {e}")
        return []

def load_tokens(tokenizer, train_split=0.8, val_split=0.1):
    """
    Load andd combine multiple data sources, then tokenize
    
    Args:
        tokenizer: GPT2 tokenizer
        train_split: Fraction of data to use for training (default 0.9)
    
    Returns:
        train_tokens, val_tokens, test_tokens: Tokenized and split data
    """ 


    all_texts = []
    dataset_name = "bookcorpus_tokenized_splits.pt"



    # ------------------- Get Saved Datasets from HuggingFace -------------------
    try:
        train_tokens, val_tokens, test_tokens = load_tokens_from_drive(dataset_name)
        print("✅ Using cached tokenized/split data from Drive")
        return train_tokens, val_tokens, test_tokens

    except Exception as e:
        print(f"⚠️  Could not use cached tokenized splits ({e}). Falling back to HuggingFace load.")

    

    # ------------------- Load HuggingFace BookCorpus - When absent on Drive -------------------
    bookcorpus_texts = load_huggingface_corpus_data("rojagtap/bookcorpus")
    if bookcorpus_texts:
        all_texts.extend(bookcorpus_texts)
        print(f"✅ Added {len(bookcorpus_texts)} texts from HuggingFace corpus")
    
    print(f"Total texts loaded: {len(all_texts)}")

    # Shuffle the data before splits
    rng = random.Random(42)
    rng.shuffle(all_texts)

    # Split combined data into train/val/test
    train_end = int(train_split * len(all_texts))
    val_end = train_end + int(val_split * len(all_texts))

    train_texts = all_texts[:train_end]
    val_texts = all_texts[train_end:val_end]
    test_texts = all_texts[val_end:]
    
    print(f"Data split: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")


    
    # ------------------- Tokenized Loaded HuggingFace BookCorpus -------------------
    # Tokenize train data
    train_tokens = []
    for text in train_texts:
        if text.strip():  # Skip empty texts
            train_tokens.extend(tokenizer.encode(text, max_length=CONFIG.MAX_LEN, truncation=True))
    
    # Tokenize validation data
    val_tokens = []
    for text in val_texts:
        if text.strip():  # Skip empty texts
            val_tokens.extend(tokenizer.encode(text, max_length=CONFIG.MAX_LEN, truncation=True))

    # Tokenize test data
    test_tokens = []
    for text in test_texts:
        if text.strip():  # Skip empty texts
            test_tokens.extend(tokenizer.encode(text, max_length=CONFIG.MAX_LEN, truncation=True))
    
    print(f"✅ Tokenization complete!")



    # -------------------Save tokenized splits so ablation runs can reuse the exact same data. -------------------    
    try:
        import os

        save_dir = CONFIG.DATASET_DIR
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "bookcorpus_tokenized_splits.pt")

        torch.save(
            {
                "train_tokens": train_tokens,
                "val_tokens": val_tokens,
                "test_tokens": test_tokens,
                "train_split": train_split,
                "val_split": val_split,
                "num_texts": len(all_texts),
                "seed": 42,
            },
            save_path,
        )
        print(f"✅ Saved tokenized shuffled splits to {save_path}")
    except Exception as e:
        print(f"⚠️  Could not save tokenized splits: {e}")


    return train_tokens, val_tokens, test_tokens

def prepare_dataloader(tokens, config, split):
    """
    Wrap GPTDataset in a DataLoader with validation
    """
    # Validate we have enough tokens
    if len(tokens) < config.SEQ_LEN:
        print(f"⚠️  Warning: Only {len(tokens)} tokens available, but SEQ_LEN={config.SEQ_LEN}")
        
        # For testing purposes, use a smaller sequence length
        effective_seq_len = min(config.SEQ_LEN, max(1, len(tokens) // 2))
        print(f"   Using effective SEQ_LEN={effective_seq_len} for this run")
    else:
        effective_seq_len = config.SEQ_LEN
    
    dataset = GPTDataset(tokens, effective_seq_len)

    # Validate dataset has samples
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty! Need at least {effective_seq_len + 1} tokens, got {len(tokens)}")
    
    print(f"   Created dataset tokens for {split} with {len(dataset)} samples (seq_len={effective_seq_len})")
    
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
