# GPT-1 30M Project

Python environment for this repo lives at the **monorepo root**.

---

## Pretraining

```bash
uv sync
uv run python GPT1-Project/scripts/run_training.py
```

Alternatively, activate the root `.venv` (or select it in your IDE) and run `python scripts/run_training.py` from inside `GPT1-Project/`.

Checkpoints are saved to `checkpoints/` (RunPod: `/workspace/build-an-llm/checkpoints`). The best model is saved as `checkpoints/best_model.pt`.

---

## Supervised Fine-Tuning (SFT)

> **Prerequisite:** You must complete pretraining first. SFT loads `checkpoints/best_model.pt` as its starting point.

### 1. Prepare your SFT dataset

Create a JSONL file where each line is a JSON object with `instruction`, optional `input`, and `output` fields (Alpaca format):

```jsonl
{"instruction": "Summarise this text.", "input": "The cat sat on the mat.", "output": "A cat sat on a mat."}
{"instruction": "What is 2 + 2?", "input": "", "output": "4"}
```

A 10-example sample is provided at `GPT1-Project/data/sft_data.jsonl` to verify the pipeline works end to end.

### 2. Run SFT

```bash
uv run python GPT1-Project/scripts/run_sft.py
```

**With overrides:**

```bash
uv run python GPT1-Project/scripts/run_sft.py \
  --sft-data-path /path/to/your/data.jsonl \
  --pretrained-checkpoint checkpoints/best_model.pt \
  --epochs 3 \
  --lr 2e-5 \
  --freeze-embeddings
```

### 3. SFT outputs

| File | Description |
|---|---|
| `checkpoints/sft/sft_best.pt` | Best SFT checkpoint (lowest val loss) |
| `checkpoints/sft/sft_latest.pt` | Latest checkpoint (for resuming) |
| `checkpoints/sft/sft_epoch_N.pt` | Per-epoch checkpoints (last 3 kept) |

### Key differences from pretraining

| | Pretraining | SFT |
|---|---|---|
| Data | Raw text token stream | Instruction / response pairs |
| Loss | All tokens | Response tokens only (prompt masked with `-100`) |
| Learning rate | 4.2e-4 | 2e-5 |
| Epochs | 10 | 3 |
| Starting weights | Random | `best_model.pt` |

---

## File structure

```
GPT1-Project/
  data/
    sft_data.jsonl          sample SFT dataset (10 examples)
  scripts/
    run_training.py         pretraining entry point
    run_sft.py              SFT entry point
  src/
    data/
      book_corpus.py        pretraining dataset
      sft_dataset.py        SFT dataset with prompt masking
    models/
      gpt1.py               model architecture
    training/
      train.py              pretraining loop
      sft_train.py          SFT training loop
    utils/
      config.py             pretraining hyperparameters
      sft_config.py         SFT hyperparameters
```
