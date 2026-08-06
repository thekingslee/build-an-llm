# Changelog
All notable changes to this project will be documented is this file.

## [Unreleased]

## [1.1.0] - 2026-08-06
### Added
- Supervised Fine-Tuning (SFT) pipeline for GPT-1 30M model
  - `GPT1-Project/src/data/sft_dataset.py` — Alpaca-format dataset with prompt loss masking
  - `GPT1-Project/src/data/hf_sft_loader.py` — flexible HuggingFace dataset loader with auto column detection
  - `GPT1-Project/src/utils/sft_config.py` — SFT hyperparameter config (env-aware)
  - `GPT1-Project/src/training/sft_train.py` — full SFT training loop with AMP, W&B, checkpointing
  - `GPT1-Project/scripts/run_sft.py` — CLI entry point with --hf-dataset, --inspect, and override flags
  - `GPT1-Project/data/sft_data.jsonl` — 10-example Alpaca-format sample dataset

### Fixed
- Replaced deprecated `torch.cuda.amp` imports with `torch.amp` in `train.py` and `sft_train.py`
- AMP now correctly disabled on MPS devices (Apple Silicon)
- Added `wandb.login(anonymous="allow")` before `wandb.init()` in both training scripts

## [1.0.0] - 2026-05-25
### Added
- GPT-1 Project/
- pdf_scrape/
- web_scrape/
- README.md
- pyproject.toml
- uv.lock
