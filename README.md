# Build an LLM

## Description
Large Language Model (LLM) from the ground up, based on the GPT-1 implementation. This model will be trained on Nigerian text data extracted from various sources.
The final mined data in the course of this project is at https://huggingface.co/datasets/theKingslee/9ja-bookcorpus

## Setup Instructions
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/thekingslee/build-an-llm.git
   cd build-an-llm
   ```
   
2. Python environment for this repo lives at the monorepo root. From the repository root directory:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   uv sync
   ```
   Alternatively, activate the root .venv (or select it in your IDE) and run python scripts/run_training.py from inside GPT1-Project/.

## Usage
To scrape a text source on the web, run the below on terminal:
```bash
python3 web_scrape/scrape_webpage.py 'source link'
```
To extract text data from a files: NB: Always remember the trailing '/'
```bash
python3 pdf_scrape/scrape_local_pdf.py "file's directory/"
```

## Directory Structure

- `GPT1-Project/`
  - `scripts/`
    - `run_training.py`
  - `src/`
    - `data/`
      - `book_corpus.py`
      - `nigerian_corpus.py`
    - `models/`
      - `gpt1.py`
    - `training/`
      - `train.py`
    - `utils/`
      - `config.py`
  - `README.md`
- `docs/`
  - `evals/`
    - `README.md` (Evaluation Hub)
    - `pretraining_report.md` (Pretraining Benchmarks)
    - `figures/`
    - `raw_results/`
- `pdf_scrape/`
  - `scrape_local_pdf.py`
  - `scraped_pdf.py`
  - `notebooks`
- `web_scrape/`
  - `scrape_webpage.py`
- `pyproject.toml`
- `.gitignore`
- `uv.lock`
- `README.md`

## 📊 Evaluations & Benchmarking
Detailed performance metrics, loss curves, perplexity scores, and hardware benchmarks are available in the **[Evaluation Hub](docs/evals/README.md)**.
- 🚀 **[Pretraining Evaluation Report](docs/evals/pretraining_report.md)**: Full analysis of pre-training dynamics on the 9ja-bookcorpus dataset.

## Model Config
| Model | Params | n_layer | n_head | n_embd |
|-------|--------|---------|--------|--------|
| 30M model | ~30M | 4 | 4 | 256 |
| 110M model | ~110M | 12 | 12 | 768 |

## Project Video Map and Receipts
- [Project inception](https://x.com/theKingslee/status/2020561221460607079?s=20)
- [Team selection](https://x.com/theKingslee/status/2029879926405435507?s=20)
- [Basic terms and processes definitions](https://x.com/theKingslee/status/2030269155039985886?s=20)
- [First interaction with model after pre-training](https://x.com/theKingslee/status/2057103886633037833?s=20)

## Contribution
We welcome contributions! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/YourFeature
   ```
5. Open a pull request.


## Acknowledgements
- Special thanks to the authors whose works inspired this project, Kingsley Nworie https://thekingslee.com, Abdullahi Mujaheed Aliyu https://github.com/MLHermit and Ayeni Oluwatosin Olawale https://github.com/AyeniOluwatosinOlawale
