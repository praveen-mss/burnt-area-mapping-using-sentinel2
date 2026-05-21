# src/pipelines/reprocess_pipeline.py

import os
from concurrent.futures import ProcessPoolExecutor

from src.inference.predict_tile import predict_tile
from src.utils.logger import get_logger

logger = get_logger()


def _use_registered_threshold(config):

    threshold_source = config.get("threshold_source")

    if threshold_source is not None:
        return threshold_source == "registered"

    return config.get("use_registered_threshold", False)


def run_full_reprocessing(config):

    data_root = config["data_root"]
    output_root = config["output_root"]
    registry = config["model_registry"]
    threshold = config["threshold"]
    use_registered_threshold = _use_registered_threshold(config)
    max_workers = config.get("max_workers", 4)

    safe_paths = []

    logger.info("Scanning all tiles and dates...")

    for tile in os.listdir(data_root):

        tile_path = os.path.join(data_root, tile)

        if not os.path.isdir(tile_path):
            continue

        for date_folder in os.listdir(tile_path):

            date_path = os.path.join(tile_path, date_folder)

            if not os.path.isdir(date_path):
                continue

            for file in os.listdir(date_path):

                if file.endswith(".SAFE"):
                    safe_paths.append(os.path.join(date_path, file))

    logger.info(f"Found {len(safe_paths)} SAFE scenes")

    if len(safe_paths) == 0:
        logger.warning("No SAFE scenes found.")
        return

    logger.info("Starting parallel processing...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        futures = []

        for safe in safe_paths:

            tile = os.path.basename(safe).split("_")[5]

            output_dir = os.path.join(output_root, tile)

            futures.append(
                executor.submit(
                    predict_tile,
                    safe,
                    registry,
                    threshold,
                    output_dir,
                    use_registered_threshold
                )
            )

        for f in futures:
            try:
                f.result()
            except Exception as e:
                print("Tile processing failed:", e)

    logger.info("Full historical reprocessing completed.")
