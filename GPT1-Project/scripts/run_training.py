# scripts/run_training.py

import os
import sys

# 1️⃣ Ensure project root is in sys.path so src modules work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 2️⃣ Activate UV virtual environment automatically (optional)
venv_path = os.path.join(project_root, ".venv", "bin", "activate_this.py")
if os.path.exists(venv_path):
    exec(open(venv_path).read(), {"__file__": venv_path})

# 3️⃣ Import the modular training function
from src.training.train import train

# 4️⃣ Run training
if __name__ == "__main__":
    train()
