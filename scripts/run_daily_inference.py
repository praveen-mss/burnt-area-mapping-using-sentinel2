# scripts/run_daily_inference.py

from src.utils.config import load_config
from src.pipelines.daily_inference_pipeline import run_daily_inference


if __name__ == "__main__":

    config = load_config("configs/inference.yaml")
    run_daily_inference(config)