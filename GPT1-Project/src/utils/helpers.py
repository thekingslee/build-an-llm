import os
from src.utils.config import CONFIG

 
# Helper: rotate step checkpoints, keeping only the last N
def rotate_step_checkpoints(step_ckpt_paths, new_path):
    step_ckpt_paths.append(new_path)
    while len(step_ckpt_paths) > CONFIG.KEEP_LAST_N_CHECKPOINTS:
        oldest = step_ckpt_paths.pop(0)
        if os.path.exists(oldest):
            os.remove(oldest)
            print(f"   Rotated out old checkpoint: {os.path.basename(oldest)}")