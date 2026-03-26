# GPT1 project

This project implements a GPT-1 like model for text generation.

### Setup

1.  **Clone the monorepo:**

    ```bash
    git clone https://github.com/your-repo/build-an-llm.git
    cd build-an-llm
    ```

2.  **Create a virtual environment and install dependencies (from the monorepo root):**

    ```bash
    python -m venv venv
    source venv/bin/activate
    uv sync
    ```

### Usage

To train the model from the `GPT1-Project` directory, navigate to it and run:

```bash
cd GPT1-Project
python src/training/train.py
```

### Configuration

Model and training parameters can be adjusted in `src/utils/config.py`.

