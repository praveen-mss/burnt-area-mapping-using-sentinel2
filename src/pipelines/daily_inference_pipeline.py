import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

from src.inference.predict_tile import predict_tile
from src.utils.logger import get_logger

logger = get_logger()


def _use_registered_threshold(config):

    threshold_source = config.get("threshold_source")

    if threshold_source is not None:
        return threshold_source == "registered"

    return config.get("use_registered_threshold", False)


def run_daily_inference(config):

    today = datetime.now().strftime("%d-%b-%Y")
    data_root = config["data_root"]
    output_root = config["output_root"]
    registry = config["model_registry"]
    threshold = config["threshold"]
    use_registered_threshold = _use_registered_threshold(config)
    max_workers = config.get("max_workers", 2)

    safe_paths = []

    for tile in os.listdir(data_root):

        tile_path = os.path.join(data_root, tile)

        if not os.path.isdir(tile_path):
            continue

        date_folder = os.path.join(tile_path, today)

        if os.path.exists(date_folder):
            for file in os.listdir(date_folder):
                if file.endswith(".SAFE"):
                    safe_paths.append(os.path.join(date_folder, file))

    logger.info(f"Found {len(safe_paths)} SAFE tiles")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(
                predict_tile,
                safe,
                registry,
                threshold,
                os.path.join(output_root, os.path.basename(safe).split("_")[5]),
                use_registered_threshold
            )
            for safe in safe_paths
        ]

        for f in futures:
            f.result()

    logger.info("Daily inference completed.")
