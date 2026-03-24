# src/training/train.py

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, GPT2Tokenizer
import wandb
import tqdm
import os

# Project modules
from src.models.gpt1 import GPT1
from src.data.book_corpus import load_tokens, prepare_dataloader
from src.utils.config import CONFIG

def train():
    # ------------------- Device -------------------
    device = CONFIG.DEVICE
    print(f"Using device: {device}")

    # ------------------- Tokenizer -------------------
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # ------------------- Load Dataset -------------------
    train_tokens, val_tokens, test_tokens = load_tokens(tokenizer)
    train_loader = prepare_dataloader(train_tokens, CONFIG)
    val_loader = prepare_dataloader(val_tokens, CONFIG)
    test_loader = prepare_dataloader(test_tokens, CONFIG)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

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
    batches_per_epoch = len(train_loader)
    actual_total_steps = batches_per_epoch * CONFIG.EPOCHS
    print(f"[Scheduler Setup] Actual total training steps (batches_per_epoch * epochs): {actual_total_steps}")
    print(f"[Scheduler Setup] Scheduler will use min({CONFIG.WARMUP_STEPS}, {actual_total_steps // 10}) warmup steps.")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(CONFIG.WARMUP_STEPS, actual_total_steps // 10),
        num_training_steps=actual_total_steps
    )

    # ------------------- W&B Logging -------------------
    # Clean previous W&B run before starting new one
    if wandb.run is not None:
        wandb.finish()

    wandb.init(
        project="GPT1-Nigerian-Book",
        name="gpt1_run",
        config={
            "model": "GPT-1",
            "seq_len": CONFIG.SEQ_LEN,
            "batch_size": CONFIG.BATCH_SIZE,
            "learning_rate": CONFIG.LEARNING_RATE,
            "optimizer": "AdamW",
            "epochs": CONFIG.EPOCHS,
        }
    )

    # Track gradients and parameters
    wandb.watch(model, log="all", log_freq=100)

    # ----------------- Checkpoint Settings -----------------
    checkpoint_dir = CONFIG.CHECKPOINT_DIR
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_ckpt = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

    # ----------------- Resume from checkpoint if exists -----------------
    global_step = 0
    start_epoch = 0

    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        global_step = checkpoint["step"]
        start_epoch = checkpoint["epoch"]
        print(f"Resuming from checkpoint: step {global_step}, epoch {start_epoch}")

    # ----------------- Early Stopping State -----------------
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # ------------------- Training Loop -------------------
    for epoch in range(start_epoch, CONFIG.EPOCHS):
        model.train()
        total_loss = 0 # Training loss

        for batch_idx, (x, y) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            total_loss += loss.item()

            # Checkpoint
            if global_step % CONFIG.SAVE_EVERY == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"gpt1_step_{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss.item(),
                }, checkpoint_path)

                # Save as latest checkpoint for automatic resume
                torch.save({
                    "step": global_step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss.item(),
                }, latest_ckpt)

                wandb.save(checkpoint_path)
                print(f"Checkpoint saved at step {global_step}")

            # Log every 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                # W&B logging
                wandb.log({
                    "batch_loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "step": global_step
                })

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed, Avg Train Loss: {avg_train_loss:.4f}")
        wandb.log({"epoch_loss": avg_train_loss})

        # Evaluation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xInput, yTarget in val_loader:
                xInput, yTarget = xInput.to(device), yTarget.to(device)
                predictions = model(xInput).view(-1, tokenizer.vocab_size)
                yTarget = yTarget.view(-1)
                val_loss += criterion(predictions, yTarget).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} | Avg Val Loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch})

        # Early 
        # If no meaningful improvement in the val_loss after 3 consecutive epoch, we terminate.
        if avg_val_loss < (best_val_loss - CONFIG.EARLY_STOPPING_MIN_DELTA):
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(
                f"No val improvement for {epochs_without_improvement} epoch(s). "
                f"Best val loss: {best_val_loss:.4f}"
            )
            if epochs_without_improvement >= CONFIG.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at step {global_step}.")
                break
 
    print("Training complete!")
