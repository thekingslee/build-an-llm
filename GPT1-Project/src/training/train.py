# src/training/train.py

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup, GPT2Tokenizer
import wandb
import tqdm
import os
import glob
from datetime import datetime

# Project modules
from src.models.gpt1 import GPT1
from src.data.book_corpus import load_tokens, prepare_dataloader
from src.utils.config import CONFIG
from src.utils.helpers import rotate_step_checkpoints

def train():
    # ------------------- Device -------------------
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available(): # For Apple Silicon GPUs
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    print(f"Using the following hyper-parameters: {CONFIG}")

    # ------------------- Tokenizer -------------------
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # ------------------- Load Dataset -------------------
    train_tokens, val_tokens, test_tokens = load_tokens(tokenizer)
    train_loader = prepare_dataloader(train_tokens, CONFIG, "train")
    val_loader = prepare_dataloader(val_tokens, CONFIG, "validate")
    test_loader = prepare_dataloader(test_tokens, CONFIG, "test")

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
    print(f"[Scheduler Setup] Total training steps per epoch(batches_per_epoch): {batches_per_epoch}")
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
        project="30M-GPT1-Model",
        name=f"30m_gpt1_runpod_{datetime.now()}",
        config={
            "model": "GPT-1",
            "seq_len": CONFIG.SEQ_LEN,
            "batch_size": CONFIG.BATCH_SIZE,
            "learning_rate": CONFIG.LEARNING_RATE,
            "optimizer": "AdamW",
            "epochs": CONFIG.EPOCHS,
            "use_amp": CONFIG.USE_AMP,
            "warmup_steps": CONFIG.WARMUP_STEPS,
        }
    )

    # Track gradients and parameters
    wandb.watch(model, log="all", log_freq=100)

    # ----------------- Checkpoint Settings -----------------
    checkpoint_dir = CONFIG.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_ckpt   = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    best_ckpt     = os.path.join(checkpoint_dir, "best_model.pt")

    # ----------------- AMP Scaler -----------------
    # GradScaler keeps fp16 gradients from underflowing during backward.
    # It's a no-op when USE_AMP=False (enabled=False), so no conditional logic needed below.
    scaler = GradScaler(enabled=CONFIG.USE_AMP)

    # ----------------- Resume from checkpoint if exists -----------------
    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"]) # For backward compatibility with older checkpoints
        global_step = checkpoint["step"]
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resuming from checkpoint: step {global_step}, epoch {start_epoch}")


    # ----------------- Early Stopping State -----------------
    epochs_without_improvement = 0
    step_ckpt_paths = []  # Tracks saved step checkpoints for rotation


    # ------------------- Training Loop -------------------
    for epoch in range(start_epoch, CONFIG.EPOCHS):
        model.train()
        total_loss = 0 # Training loss

        for batch_idx, (x, y) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # ---------- Forward + Loss (AMP autocast) ----------
            with autocast(enabled=CONFIG.USE_AMP):
                outputs = model(x)
                loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))

            # ---------- Backward (scaled for AMP) ----------
            scaled_loss = scaler.scale(loss)
            assert isinstance(scaled_loss, torch.Tensor)
            scaled_loss.backward()

            # Unscale before grad norm + clip so the norm is in true fp32 scale
            scaler.unscale_(optimizer)

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            total_loss += loss.item()

            # ---------- Step Checkpoint ----------
            if global_step % CONFIG.SAVE_EVERY == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"gpt1_step_{global_step}.pt")
                ckpt_data = {
                    "step": global_step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": loss.item(),
                    "best_val_loss": best_val_loss,
                }
                torch.save(ckpt_data, checkpoint_path)
                torch.save(ckpt_data, latest_ckpt)  # Always keep latest for resume
                rotate_step_checkpoints(step_ckpt_paths, checkpoint_path)
                wandb.save(checkpoint_path)
                print(f"Checkpoint saved at step {global_step}")

            # ---------- Batch Logging ----------
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Grad Norm: {total_norm:.6f}")
                
                wandb.log({
                    "batch_loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": total_norm,
                    "step": global_step,
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

        # ---------- Save checkpoint at the end of each epoch ----------
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f"gpt1_epoch_{epoch+1}.pt")
        epoch_ckpt_data = {
            "step": global_step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": avg_val_loss,
            "best_val_loss": best_val_loss,
        }
        torch.save(epoch_ckpt_data, epoch_checkpoint_path)
        torch.save(epoch_ckpt_data, latest_ckpt)
        wandb.save(epoch_checkpoint_path)
        print(f"Epoch checkpoint saved: {epoch_checkpoint_path}")

        # ---------- Best Model ----------
        if avg_val_loss < best_val_loss:
            torch.save(epoch_ckpt_data, best_ckpt)
            print(f"   ✅ New best model saved (val_loss={avg_val_loss:.4f})") 

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
