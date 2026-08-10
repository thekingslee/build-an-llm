import os
import random
import torch
import wandb
import tqdm
from torch import nn
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup, GPT2Tokenizer
from datetime import datetime

from src.models.gpt1 import GPT1
from src.data.sft_dataset import prepare_sft_dataloader
from src.data.datasets import load_datasets
from src.utils.sft_config import SFT_CONFIG


def _load_pretrained(model: GPT1, checkpoint_path: str, device: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}\n"
            "Run pretraining first (scripts/run_training.py) before SFT."
        )
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    print(f"Loaded pretrained weights from {checkpoint_path}")


def _rotate_checkpoints(paths: list[str], new_path: str, keep: int):
    paths.append(new_path)
    while len(paths) > keep:
        oldest = paths.pop(0)
        if os.path.exists(oldest):
            os.remove(oldest)
            print(f"   Rotated out: {os.path.basename(oldest)}")


def sft_train(cfg=None):
    if cfg is None:
        cfg = SFT_CONFIG

    device  = cfg.DEVICE
    use_amp = cfg.USE_AMP and device == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"SFT config: {cfg}")

    tokenizer           = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    all_examples = load_datasets(
        names=cfg.DATASET_NAMES,
        local_path=cfg.LOCAL_DATA_PATH,
        max_samples_per_dataset=cfg.MAX_SAMPLES_PER_DATASET,
    )
    random.Random(42).shuffle(all_examples)

    split_idx      = int(len(all_examples) * (1 - cfg.VAL_SPLIT))
    train_examples = all_examples[:split_idx]
    val_examples   = all_examples[split_idx:]
    print(f"SFT split: {len(train_examples)} train / {len(val_examples)} val")

    train_loader = prepare_sft_dataloader(train_examples, tokenizer, cfg, "train")
    val_loader   = prepare_sft_dataloader(val_examples,   tokenizer, cfg, "val")
    print(f"Batches — train: {len(train_loader)}, val: {len(val_loader)}")

    model = GPT1(
        vocab_size=tokenizer.vocab_size,
        embed_size=cfg.EMBED_SIZE,
        num_layers=cfg.NUM_LAYERS,
        heads=cfg.HEADS,
        max_len=cfg.MAX_LEN,
    ).to(device)

    _load_pretrained(model, cfg.PRETRAINED_CHECKPOINT, device)

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

    total_steps  = len(train_loader) * cfg.EPOCHS
    warmup_steps = min(cfg.WARMUP_STEPS, total_steps // 10)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"Scheduler: {warmup_steps} warmup / {total_steps} total steps")

    scaler = GradScaler(device=device, enabled=use_amp)

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    latest_ckpt     = os.path.join(cfg.CHECKPOINT_DIR, "sft_latest.pt")
    best_ckpt       = os.path.join(cfg.CHECKPOINT_DIR, "sft_best.pt")
    step_ckpt_paths: list[str] = []

    global_step       = 0
    start_epoch       = 0
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step       = ckpt["step"]
        start_epoch       = ckpt["epoch"] + 1
        best_val_loss     = ckpt.get("best_val_loss", float("inf"))
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        print(f"Resumed SFT from step {global_step}, epoch {start_epoch}")

    if wandb.run is not None:
        wandb.finish()

    wandb.login(anonymous="allow")
    wandb.init(
        project="30M-GPT1-SFT",
        name=f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "pretrained_checkpoint": cfg.PRETRAINED_CHECKPOINT,
            "sft_data":              cfg.SFT_DATA_PATH,
            "epochs":                cfg.EPOCHS,
            "batch_size":            cfg.BATCH_SIZE,
            "learning_rate":         cfg.LEARNING_RATE,
            "warmup_steps":          warmup_steps,
            "freeze_embeddings":     cfg.FREEZE_EMBEDDINGS,
            "use_amp":               use_amp,
        },
    )
    wandb.watch(model, log="gradients", log_freq=50)

    for epoch in range(start_epoch, cfg.EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (x, y) in enumerate(
            tqdm.tqdm(train_loader, desc=f"SFT Epoch {epoch + 1}/{cfg.EPOCHS}")
        ):
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

        avg_train_loss = epoch_loss / len(train_loader)
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
        _rotate_checkpoints(step_ckpt_paths, epoch_path, cfg.KEEP_LAST_N_CHECKPOINTS)
        wandb.save(epoch_path)
        print(f"Checkpoint saved: {epoch_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss     = avg_val_loss
            epochs_no_improve = 0
            torch.save(ckpt_data, best_ckpt)
            print(f"   New best SFT model (val_loss={best_val_loss:.4f})")
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
