# scripts/run_reprocess_all.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.pipelines.reprocess_pipeline import run_full_reprocessing


if __name__ == "__main__":

    config = load_config("configs/inference.yaml")

    run_full_reprocessing(config)