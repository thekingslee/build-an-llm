import os
import re
import glob
import random
import torch
import wandb
import tqdm
from torch import nn
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup, GPT2Tokenizer
from datetime import datetime

from src.models.gpt1 import GPT1
from src.data.sft_dataset import load_or_create_sft_splits, create_sft_dataloader
from src.utils.sft_config import SFT_CONFIG


def _load_pretrained(model: GPT1, checkpoint_path: str, device: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}\n"
            "Run pretraining first (scripts/run_training.py) before SFT."
        )
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    print(f"Loaded pretrained weights from {checkpoint_path}")


def _get_highest_saved_epoch(checkpoint_dir: str) -> int:
    highest = 0
    if not os.path.exists(checkpoint_dir):
        return 0
    for p in glob.glob(os.path.join(checkpoint_dir, "sft_epoch_*.pt")):
        match = re.search(r"sft_epoch_(\d+)\.pt", p)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest


def sft_train(cfg=None):
    if cfg is None:
        cfg = SFT_CONFIG

    device  = cfg.DEVICE
    use_amp = cfg.USE_AMP and device == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"SFT config: {cfg}")

    tokenizer           = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds, test_ds, total_raw_count = load_or_create_sft_splits(tokenizer, cfg)

    train_loader = create_sft_dataloader(train_ds, cfg, is_train=True)
    val_loader   = create_sft_dataloader(val_ds,   cfg, is_train=False)
    test_loader  = create_sft_dataloader(test_ds,  cfg, is_train=False) if test_ds else None

    total_tokens_cum  = train_ds.total_tokens + val_ds.total_tokens + (test_ds.total_tokens if test_ds else 0)
    total_resp_tokens = train_ds.total_response_tokens + val_ds.total_response_tokens + (test_ds.total_response_tokens if test_ds else 0)
    total_discarded   = train_ds.num_discarded + val_ds.num_discarded + (test_ds.num_discarded if test_ds else 0)
    total_kept        = len(train_ds) + len(val_ds) + (len(test_ds) if test_ds else 0)

    print("\n" + "=" * 65)
    print("SFT DATASET & TOKEN SUMMARY")
    print("=" * 65)
    print(f"Total Raw Samples Loaded:       {total_raw_count:,}")
    print(f"Discarded (> {cfg.MAX_LEN} context length): {total_discarded:,} ({total_discarded / max(1, total_raw_count) * 100:.2f}%)")
    print(f"Valid Samples Kept:             {total_kept:,}")
    print("-" * 65)
    print(f"Train Split:  {len(train_ds):>7,} samples | {train_ds.total_tokens:>10,} tokens ({train_ds.total_response_tokens:>10,} target tokens)")
    print(f"Val Split:    {len(val_ds):>7,} samples | {val_ds.total_tokens:>10,} tokens ({val_ds.total_response_tokens:>10,} target tokens)")
    if test_ds:
        print(f"Test Split:   {len(test_ds):>7,} samples | {test_ds.total_tokens:>10,} tokens ({test_ds.total_response_tokens:>10,} target tokens)")
    print("-" * 65)
    print(f"Cumulative Total Tokens:        {total_tokens_cum:,} tokens")
    print(f"Cumulative Target/Loss Tokens*: {total_resp_tokens:,} tokens")
    print("=" * 65 + "\n")

    model = GPT1(
        vocab_size=tokenizer.vocab_size,
        embed_size=cfg.EMBED_SIZE,
        num_layers=cfg.NUM_LAYERS,
        heads=cfg.HEADS,
        max_len=cfg.MAX_LEN,
    ).to(device)

    if cfg.FREEZE_EMBEDDINGS:
        for p in model.token_embedding.parameters():
            p.requires_grad = False
        for p in model.position_embedding.parameters():
            p.requires_grad = False
        print("Embeddings frozen.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LEARNING_RATE,
        weight_decay=0.01,
    )

    scaler = GradScaler(device=device, enabled=use_amp)

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    latest_ckpt     = os.path.join(cfg.CHECKPOINT_DIR, "sft_latest.pt")
    best_ckpt       = os.path.join(cfg.CHECKPOINT_DIR, "sft_best.pt")

    global_step       = 0
    start_epoch       = 0
    start_batch_idx   = 0
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    highest_saved_epoch = _get_highest_saved_epoch(cfg.CHECKPOINT_DIR)
    highest_epoch_ckpt  = os.path.join(cfg.CHECKPOINT_DIR, f"sft_epoch_{highest_saved_epoch}.pt") if highest_saved_epoch > 0 else None

    # Determine initial checkpoint source
    if os.path.exists(latest_ckpt):
        resume_ckpt_path = latest_ckpt
    elif highest_epoch_ckpt and os.path.exists(highest_epoch_ckpt):
        resume_ckpt_path = highest_epoch_ckpt
    else:
        resume_ckpt_path = None

    if resume_ckpt_path:
        try:
            ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as e:
                    print(f"Note: Starting with fresh optimizer for new stage ({e}).")
            if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"]:
                try:
                    scaler.load_state_dict(ckpt["scaler_state_dict"])
                except Exception:
                    pass

            global_step       = ckpt.get("step", 0)
            start_epoch       = ckpt.get("epoch", highest_saved_epoch)
            start_batch_idx   = ckpt.get("batch_idx", 0)
            if ckpt.get("epoch_completed", False):
                start_epoch += 1
                start_batch_idx = 0
            best_val_loss     = ckpt.get("best_val_loss", float("inf"))
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)
            print(f"✅ Successfully resumed weights from: {resume_ckpt_path} (step {global_step}, next epoch {start_epoch + 1})")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {resume_ckpt_path} ({e}). Using base model.")
            _load_pretrained(model, cfg.PRETRAINED_CHECKPOINT, device)
            start_epoch = highest_saved_epoch
    else:
        _load_pretrained(model, cfg.PRETRAINED_CHECKPOINT, device)
        start_epoch = highest_saved_epoch

    target_total_epochs = start_epoch + cfg.EPOCHS
    remaining_epochs    = target_total_epochs - start_epoch
    total_training_steps = len(train_loader) * remaining_epochs
    warmup_steps        = min(cfg.WARMUP_STEPS, max(1, total_training_steps // 10))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    print(f"Training plan: Epoch {start_epoch + 1} -> Epoch {target_total_epochs} ({remaining_epochs} epochs | {total_training_steps} steps | {warmup_steps} warmup)")

    if wandb.run is not None:
        wandb.finish()

    wandb.login(anonymous="allow")
    wandb.init(
        project="30M-GPT1-SFT",
        name=f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "pretrained_checkpoint": cfg.PRETRAINED_CHECKPOINT,
            "datasets":              cfg.DATASET_NAMES,
            "epochs":                cfg.EPOCHS,
            "start_epoch":           start_epoch,
            "target_total_epochs":   target_total_epochs,
            "batch_size":            cfg.BATCH_SIZE,
            "learning_rate":         cfg.LEARNING_RATE,
            "warmup_steps":          warmup_steps,
            "freeze_embeddings":     cfg.FREEZE_EMBEDDINGS,
            "use_amp":               use_amp,
            "max_len":               cfg.MAX_LEN,
            "train_samples":         len(train_ds),
            "val_samples":           len(val_ds),
            "test_samples":          len(test_ds) if test_ds else 0,
            "train_tokens":          train_ds.total_tokens,
            "val_tokens":            val_ds.total_tokens,
            "test_tokens":           test_ds.total_tokens if test_ds else 0,
            "cumulative_tokens":     total_tokens_cum,
            "cumulative_loss_tokens": total_resp_tokens,
            "discarded_samples":     total_discarded,
        },
    )
    wandb.watch(model, log="gradients", log_freq=50)

    for epoch in range(start_epoch, target_total_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (x, y) in enumerate(
            tqdm.tqdm(train_loader, desc=f"SFT Epoch {epoch + 1}/{target_total_epochs}")
        ):
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device if device != "mps" else "cpu", enabled=use_amp):
                logits = model(x)
                loss   = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            epoch_loss  += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"  Epoch {epoch+1} | step {global_step} | "
                    f"loss {loss.item():.4f} | grad_norm {float(grad_norm):.4f}"
                )
                wandb.log({
                    "sft/batch_loss": loss.item(),
                    "sft/lr":         scheduler.get_last_lr()[0],
                    "sft/grad_norm":  float(grad_norm),
                    "step":           global_step,
                })

            if global_step % getattr(cfg, "SAVE_EVERY_STEPS", 50) == 0:
                torch.save({
                    "step":                 global_step,
                    "epoch":                epoch,
                    "batch_idx":            batch_idx + 1,
                    "epoch_completed":      False,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict":    scaler.state_dict(),
                    "best_val_loss":        best_val_loss,
                    "epochs_no_improve":    epochs_no_improve,
                }, latest_ckpt)

        avg_train_loss = epoch_loss / max(1, (len(train_loader) - (start_batch_idx if epoch == start_epoch else 0)))
        print(f"Epoch {epoch+1} avg train loss: {avg_train_loss:.4f}")
        wandb.log({"sft/epoch_train_loss": avg_train_loss, "epoch": epoch + 1})

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                with autocast(device_type=device if device != "mps" else "cpu", enabled=use_amp):
                    logits    = model(x_val)
                    val_loss += criterion(
                        logits.view(-1, logits.size(-1)), y_val.view(-1)
                    ).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} avg val loss: {avg_val_loss:.4f}")
        wandb.log({"sft/epoch_val_loss": avg_val_loss, "epoch": epoch + 1})

        ckpt_data = {
            "step":                 global_step,
            "epoch":                epoch,
            "batch_idx":            0,
            "epoch_completed":      True,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "val_loss":             avg_val_loss,
            "best_val_loss":        best_val_loss,
            "epochs_no_improve":    epochs_no_improve,
        }
        epoch_path = os.path.join(cfg.CHECKPOINT_DIR, f"sft_epoch_{epoch+1}.pt")
        torch.save(ckpt_data, epoch_path)
        torch.save(ckpt_data, latest_ckpt)
        wandb.save(epoch_path)
        print(f"Checkpoint saved: {epoch_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss     = avg_val_loss
            epochs_no_improve = 0
            ckpt_data["best_val_loss"] = best_val_loss
            torch.save(ckpt_data, best_ckpt)
            wandb.save(best_ckpt)
            print(f"   ✅ New best SFT model saved: {best_ckpt} (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(
                f"   No improvement for {epochs_no_improve} epoch(s). "
                f"Best val loss: {best_val_loss:.4f}"
            )
            if epochs_no_improve >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at step {global_step}.")
                break

    wandb.finish()
    print(f"SFT complete. Best checkpoint: {best_ckpt}")
