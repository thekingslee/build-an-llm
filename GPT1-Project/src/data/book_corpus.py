# src/data/book_corpus.py

from torch.utils.data import Dataset, DataLoader
import torch
from datasets import load_dataset
from src.data.ablation_dataset import load_tokens_from_drive
from src.utils.config import CONFIG
import random
from tqdm import tqdm

class GPTDataset(Dataset):
    def __init__(self, tokens, seq_len, stride: int = 1):
        """
        Args:
            tokens:   flat list of token IDs
            seq_len:  length of each training window
            stride:   how far the window advances between examples.
                      stride=1  → full sliding window (every position, max overlap)
                      stride=seq_len → non-overlapping chunks (modern practice for large datasets)
        """
        self.data = torch.tensor(tokens)
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self):
        # Number of valid starting positions given the stride
        return max(0, (len(self.data) - self.seq_len) // self.stride)

    def __getitem__(self, index):
        start = index * self.stride
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
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

def load_huggingface_corpus_data(source, name=None, split="100%"):
    """
    Load BookCorpus from HuggingFace
    """
    try:
        if name:
            dataset = load_dataset(source, name, split=f"train[:{split}]")
        else:
            dataset = load_dataset(source, split=f"train[:{split}]")
        texts = [item['text'] for item in dataset if item['text'].strip()]  # type: ignore
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


    dataset_name = "600k_token_tokenized_splits.pt"
    all_texts = []



    # ------------------- Get Saved Datasets from Storage -------------------
    try:
        train_tokens, val_tokens, test_tokens = load_tokens_from_drive(dataset_name)
        print("✅ Using cached tokenized/split data from Network Volume")

        return train_tokens, val_tokens, test_tokens

    except Exception as e:
        print(f"⚠️  Could not use cached tokenized splits ({e}). Falling back to HuggingFace load.")

    

    # ------------------- Load HuggingFace Naija Bookcorpus - When absent on Drive -------------------
    naijacorpus_texts = load_huggingface_corpus_data("thekingslee/9ja-bookcorpus")
    if naijacorpus_texts:
        all_texts.extend(naijacorpus_texts)
        print(f"✅ Added 9ja-bookcorpus of {len(naijacorpus_texts)} texts from HuggingFace")

    educorpus_texts = load_huggingface_corpus_data("theKingslee/fineweb-1b-tokens")  
    if educorpus_texts:
        all_texts.extend(educorpus_texts)
        print(f"✅ Added fineweb-edu of {len(educorpus_texts)} texts from HuggingFace")
    
    print(f"Total texts loaded: {len(all_texts)}")


    # ------------------- Clean Raw Texts (NEW STEP) -------------------
    cleaned_texts = clean_data(all_texts)
    all_texts = cleaned_texts

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
    print(f"Starting fast parallel tokenization for {len(all_texts)} texts...")
    
    def tokenize_split_in_batches(texts, batch_size=50000):
        token_ids = []
        # Process the strings in large blocks to maximize multi-threading efficiency
        for i in tqdm(range(0, len(texts), batch_size), desc = "Tokenizing batches"):
            batch = texts[i : i + batch_size]
            
            # batch_encode_plus releases the GIL and processes rows in parallel using Rust
            encodings = tokenizer(
                batch,
                add_special_tokens=False, # Standard for GPT pre-training
                truncation=True,
                max_length=CONFIG.MAX_LEN
            )
            
            # Flatten the list of lists into a single continuous token stream
            for encoding in encodings["input_ids"]:
                token_ids.extend(encoding)
                
            if i % (batch_size * 4) == 0 and i > 0:
                print(f"Processed {i}/{len(texts)} texts...")
                
        return token_ids
    

    # Tokenize train data
    print("Tokenizing train split...")
    train_tokens = tokenize_split_in_batches(train_texts)

    print("Tokenizing validation split...")
    val_tokens = tokenize_split_in_batches(val_texts)

    print("Tokenizing test split...")
    test_tokens = tokenize_split_in_batches(test_texts)
    
    print(f"Total Token split: {len(train_tokens)} train, {len(val_tokens)} val, {len(test_tokens)} test")
    print(f"✅ Tokenization complete!")



    # -------------------Save tokenized splits so ablation runs can reuse the exact same data. -------------------    
    try:
        import os

        save_dir = CONFIG.DATASET_DIR
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, dataset_name)

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
    Wrap GPTDataset in a DataLoader with validation.

    Uses stride=1 (full sliding window) by default, consistent with GPT-1 pretraining.
    To reduce epoch length on large datasets, set a STRIDE in config (e.g. STRIDE=512
    gives non-overlapping chunks and ~500x fewer batches per epoch).
    """
    # Validate we have enough tokens
    if len(tokens) < config.SEQ_LEN:
        print(f"⚠️  Warning: Only {len(tokens)} tokens available, but SEQ_LEN={config.SEQ_LEN}")

        # For testing purposes, use a smaller sequence length
        effective_seq_len = min(config.SEQ_LEN, max(1, len(tokens) // 2))
        print(f"   Using effective SEQ_LEN={effective_seq_len} for this run")
    else:
        effective_seq_len = config.SEQ_LEN

    STRIDE = config.STRIDE
    dataset = GPTDataset(tokens, effective_seq_len, stride=STRIDE)

    # Validate dataset has samples
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty! Need at least {effective_seq_len + 1} tokens, got {len(tokens)}")

    print(f"   Created {split} dataset: {len(dataset):,} samples "
          f"(seq_len={effective_seq_len}, stride={STRIDE})")

    # Only shuffle and drop incomplete batches for the training split
    is_train = (split.lower() == "train")
    num_workers = getattr(config, "NUM_WORKERS", 0)

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=is_train,  # True for train, False for val/test
        drop_last=True if is_train else False,
        num_workers=num_workers,
        pin_memory=getattr(config, "PIN_MEMORY", False),
        persistent_workers=(num_workers > 0),  # Keep workers alive between epochs
    )

def clean_data(dataset):
    print("Cleaning datasets (removing standalone punctuation and empty lines)...")
    cleaned_texts = []

    for text in dataset:
        stripped_text = text.strip()
        
        # 1. Skip if empty
        if not stripped_text:
            continue
            
        # 2. Skip if it is just a standalone period or punctuation mark (length <= 2)
        if len(stripped_text) <= 2:
            continue
            
        # 3. Robust Check: Skip if the line contains absolutely no alphanumeric letters/numbers
        if not any(char.isalnum() for char in stripped_text):
            continue
            
        cleaned_texts.append(text)
        
    print(f"Cleaning complete. Dropped {len(dataset) - len(cleaned_texts)} junk lines.")
    print(f"Remaining clean texts: {len(cleaned_texts)}")

    # import pickle
    # import os

    # local_cache_dir = "./local_cache"
    # os.makedirs(local_cache_dir, exist_ok=True)
    # local_cache_path = os.path.join(local_cache_dir, "cleaned_text_strings.pkl")

    # try:
    #     print(f"Caching {len(cleaned_texts)} cleaned text strings locally...")
    #     with open(local_cache_path, "wb") as f:
    #         pickle.dump(cleaned_texts, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     print(f"✅ Cleaned dataset cached locally at: {local_cache_path}")
    # except Exception as e:
    #     print(f"⚠️  Failed to save local cache (but continuing pipeline): {e}")

    return cleaned_texts
