# scripts/run_training.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.pipelines.train_pipeline import run_training_pipeline


if __name__ == "__main__":

    config = load_config("configs/train.yaml")
    run_training_pipeline(config)