import sys
import os

# add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.postprocessing.aggregate_tile_outputs import aggregate_tile

# path to config yaml
config_path = "configs/inference.yaml"

config = load_config(config_path)

processed_root = config["output_root"]
final_root     = config["output_root_final"]

tiles = os.listdir(processed_root)

for tile in tiles:

    if os.path.isdir(os.path.join(processed_root,tile)):

        print("Aggregating:",tile)

        aggregate_tile(
            tile,
            processed_root,
            final_root,
            config
        )