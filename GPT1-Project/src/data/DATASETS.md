# SFT Datasets

All datasets are hosted on HuggingFace in Alpaca format (100,000 examples each).

| Domain | HuggingFace Link |
|---|---|
| Extraction (NER) | https://huggingface.co/datasets/anabury/extraction-ner-sft-100k |
| Text Formatting | https://huggingface.co/datasets/anabury/text-formatting-sft-100k |
| Conversation | https://huggingface.co/datasets/anabury/conversation-sft-100k |

## Load in Python

from datasets import load_dataset

extraction   = load_dataset("anabury/extraction-ner-sft-100k")
formatting   = load_dataset("anabury/text-formatting-sft-100k")
conversation = load_dataset("anabury/conversation-sft-100k")
