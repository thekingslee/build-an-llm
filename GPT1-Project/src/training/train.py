# src/training/train.py

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, GPT2Tokenizer
import wandb
import os

# Project modules
from src.models.gpt1 import GPT1
from src.data.nigerian_corpus import load_tokens, prepare_dataloader
from src.utils.config import CONFIG

def train():
    # ------------------- Device -------------------
    device = CONFIG.DEVICE
    print(f"Using device: {device}")

    # ------------------- Tokenizer -------------------
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # ------------------- Load Dataset -------------------
    train_tokens, test_tokens = load_tokens("data/nigerian_books.txt", tokenizer)
    train_loader = prepare_dataloader(train_tokens, CONFIG)
    test_loader = prepare_dataloader(test_tokens, CONFIG)

    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # ------------------- Model -------------------
    model = GPT1(
        vocab_size=tokenizer.vocab_size,
        embed_size=CONFIG.EMBED_SIZE,
        num_layers=CONFIG.NUM_LAYERS,
        heads=CONFIG.HEADS,
        max_len=CONFIG.MAX_LEN
    ).to(device)

    # ------------------- Optimizer & Loss -------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # ------------------- Scheduler -------------------
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG.WARMUP_STEPS,
        num_training_steps=CONFIG.TOTAL_TRAINING_STEPS
    )

    # ------------------- W&B Logging -------------------
    wandb.init(
        project="GPT1-Nigerian-Book",
        name="gpt1_run",
        config=vars(CONFIG)
    )

    # ------------------- Training Loop -------------------
    for epoch in range(CONFIG.EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # Log every 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                wandb.log({"batch_loss": loss.item()})

        print(f"Epoch {epoch+1} completed, Avg Loss: {total_loss/len(train_loader):.4f}")
        wandb.log({"epoch_loss": total_loss/len(train_loader)})

        # ------------------- Optional Checkpoint -------------------
        checkpoint_dir = CONFIG.CHECKPOINT_DIR
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Training complete!")
