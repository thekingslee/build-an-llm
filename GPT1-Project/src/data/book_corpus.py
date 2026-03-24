# src/data/book_corpus.py

from torch.utils.data import Dataset, DataLoader
import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset

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
        dataset = load_dataset(source, split="train[:1%]")
        texts = [item['text'] for item in dataset if item['text'].strip()]
        return texts
    except Exception as e:
        print(f"⚠️  Error loading HuggingFace corpus: {e}")
        return []

def load_tokens(tokenizer, train_split=0.9):
    """
    Load andd combine multiple data sources, then tokenize
    
    Args:
        tokenizer: GPT2 tokenizer
        train_split: Fraction of data to use for training (default 0.9)
    
    Returns:
        train_tokens, test_tokens: Tokenized and split data
    """ 
    all_texts = []
    
    # Load local Nigerian corpus
    local_texts = load_local_corpus_data("data/")
    if local_texts:
        all_texts.extend(local_texts)
        print(f"✅ Added {len(local_texts)} texts from local corpus")
    
    # Load HuggingFace BookCorpus
    # bookcorpus_texts = load_huggingface_corpus_data("rojagtap/bookcorpus")
    # if bookcorpus_texts:
    #     all_texts.extend(bookcorpus_texts)
    #     print(f"✅ Added {len(bookcorpus_texts)} texts from HuggingFace corpus")
    
    # if not all_texts:
    #     raise RuntimeError("❌ No corpus data could be loaded! Check your data sources.")
    
    # print(f"Total texts loaded: {len(all_texts)}")
    
    # Split combined data into train/test
    split_idx = int(train_split * len(all_texts))
    train_texts = all_texts[:split_idx]
    test_texts = all_texts[split_idx:]
    
    print(f"Data split: {len(train_texts)} train, {len(test_texts)} test")
    
    # Tokenize train data
    train_tokens = []
    for text in train_texts:
        if text.strip():  # Skip empty texts
            train_tokens.extend(tokenizer.encode(text))
    
    # Tokenize test data
    test_tokens = []
    for text in test_texts:
        if text.strip():  # Skip empty texts
            test_tokens.extend(tokenizer.encode(text))
    
    print(f"✅ Tokenization complete!")
    print(f"   Train tokens: {len(train_tokens):,}")
    print(f"   Test tokens: {len(test_tokens):,}")
    
    return train_tokens, test_tokens

def prepare_dataloader(tokens, config):
    """
    Wrap GPTDataset in a DataLoader with validation
    """
    # Validate we have enough tokens
    if len(tokens) < config.SEQ_LEN:
        print(f"⚠️  Warning: Only {len(tokens)} tokens available, but SEQ_LEN={config.SEQ_LEN}")
        print(f"   Dataset will be empty. Consider:")
        print(f"   1. Adding more training data")
        print(f"   2. Reducing SEQ_LEN in config")
        
        # For testing purposes, use a smaller sequence length
        effective_seq_len = min(config.SEQ_LEN, max(1, len(tokens) // 2))
        print(f"   Using effective SEQ_LEN={effective_seq_len} for this run")
    else:
        effective_seq_len = config.SEQ_LEN
    
    dataset = GPTDataset(tokens, effective_seq_len)
    
    # Validate dataset has samples
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty! Need at least {effective_seq_len + 1} tokens, got {len(tokens)}")
    
    print(f"   Created dataset with {len(dataset)} samples (seq_len={effective_seq_len})")
    
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
