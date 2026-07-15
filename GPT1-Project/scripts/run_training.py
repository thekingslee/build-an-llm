# scripts/run_training.py

import os
import sys

# Ensure project root is in sys.path so src modules work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Activate virtual environment automatically (optional): GPT1-Project/.venv or repo root .venv
_repo_root = os.path.dirname(project_root)
_venv_sub = (
    (".venv", "Scripts", "activate_this.py")
    if sys.platform == "win32"
    else (".venv", "bin", "activate_this.py")
)
for _base in (project_root, _repo_root):
    _venv_path = os.path.join(_base, *_venv_sub)
    if os.path.exists(_venv_path):
        exec(open(_venv_path).read(), {"__file__": _venv_path})
        break

# Import the modular training function
from src.training.train import train

# Run training
if __name__ == "__main__":
    train()
