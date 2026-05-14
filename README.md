# Build an LLM

## Description
Large Language Model (LLM) from the ground up, based on the GPT-1 implementation. This model will be trained on Nigerian text data extracted from various sources.
The final mined data in the course of this project is at https://huggingface.co/datasets/theKingslee/9ja-bookcorpus

## Installation Instructions
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/thekingslee/build-an-llm.git
   cd build-an-llm
   ```
   
2. Python environment for this repo lives at the monorepo roo and installing the required libraries and dependencies. From the repository root directory:
   ```bash
   uv sync
   pip install .
   uv run python GPT1-Project/scripts/run_training.py
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
  - `scr/`
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
- `pdf_scrape/`
  - `scrape_local_pdf.py`
  - `scraped_pdf.py`
  - `notebooks`
- `web_scrape/`
  - `scrape_webpage.py`
- pyproject.toml
- .gitignore
- uv.lock
- `README.md`



## Contributing
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
- Special thanks to the authors whose works inspired this project, https://thekingslee, https://github.com/MLHermit and https://github.com/AyeniOluwatosinOlawale
