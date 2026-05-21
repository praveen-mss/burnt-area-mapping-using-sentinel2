import os
import glob
import numpy as np
import rasterio
from datetime import datetime

from src.postprocessing.mmu import apply_mmu
from src.postprocessing.aux_mask import apply_aux_masks
from src.postprocessing.temporal_filter import temporal_consistency


def aggregate_tile(tile, processed_root, final_root, config):

    tile_dir = os.path.join(processed_root, tile)

    prob_files = sorted(glob.glob(os.path.join(tile_dir, "*_BurntProb.tif")))
    mask_files = sorted(glob.glob(os.path.join(tile_dir, "*_BurntMask.tif")))
    fcc_files  = sorted(glob.glob(os.path.join(tile_dir, "*_FCC.tif")))

    if len(prob_files) == 0:
        return

    # ------------------------------------------------
    # Read metadata
    # ------------------------------------------------

    with rasterio.open(prob_files[0]) as src:
        meta = src.meta.copy()
        H = src.height
        W = src.width

    prob_stack = []
    mask_stack = []
    doy_stack  = []

    # ------------------------------------------------
    # Build stacks
    # ------------------------------------------------

    mask_by_date = {
        os.path.basename(path).split("_")[1]: path
        for path in mask_files
    }

    for prob_file in prob_files:

        date_str = os.path.basename(prob_file).split("_")[1]
        mask_file = mask_by_date.get(date_str)

        if mask_file is None:
            raise FileNotFoundError(f"Missing burn mask for probability file: {prob_file}")

        doy = datetime.strptime(date_str, "%Y%m%d").timetuple().tm_yday

        with rasterio.open(prob_file) as src:
            prob = src.read(1)

        with rasterio.open(mask_file) as src:
            mask = src.read(1)

        prob_stack.append(prob)
        mask_stack.append(mask)

        doy_map = np.where(mask == 1, doy, 0)

        doy_stack.append(doy_map)

    prob_stack = np.stack(prob_stack)
    mask_stack = np.stack(mask_stack)
    doy_stack  = np.stack(doy_stack)

    # ------------------------------------------------
    # Aggregation
    # ------------------------------------------------

    config_post = config["postprocessing"]

    # Temporal consistency filter
    if config_post["temporal_consistency"]["enabled"]:

        final_mask = temporal_consistency(
            mask_stack,
            config_post["temporal_consistency"]["window"]
        )

    else:

        final_mask = np.max(mask_stack, axis=0)

    # Maximum probability
    max_prob = np.max(prob_stack, axis=0)

    # First burn DOY
    first_doy = np.zeros((H, W))

    for i in range(len(doy_stack)):
        cond = (first_doy == 0) & (doy_stack[i] > 0)
        first_doy[cond] = doy_stack[i][cond]

    # ------------------------------------------------
    # Apply MMU
    # ------------------------------------------------

    if config_post["mmu"]["enabled"]:

        final_mask = apply_mmu(
            final_mask,
            config_post["mmu"]["pixels"]
        )

    # ------------------------------------------------
    # Apply auxiliary masks
    # (returns mask of burnable area)
    # ------------------------------------------------

    final_mask, valid_mask = apply_aux_masks(
        final_mask,
        tile,
        config,
        meta,
        return_valid_mask=True
    )

    # Apply same mask to all outputs
    max_prob[~valid_mask]  = 0
    first_doy[~valid_mask] = 0
    first_doy[final_mask == 0] = 0

    # ------------------------------------------------
    # Final output directory
    # ------------------------------------------------

    tile_out = os.path.join(final_root, tile)

    os.makedirs(tile_out, exist_ok=True)

    # ------------------------------------------------
    # Metadata
    # ------------------------------------------------

    meta_mask = meta.copy()
    meta_mask.update(dtype="uint8", count=1, compress="LZW")

    meta_prob = meta.copy()
    meta_prob.update(dtype="float32", count=1, compress="LZW")

    # ------------------------------------------------
    # Save outputs
    # ------------------------------------------------

    mask_out = os.path.join(tile_out, f"{tile}_FinalBurnMask.tif")
    prob_out = os.path.join(tile_out, f"{tile}_MaxBurnProb.tif")
    doy_out  = os.path.join(tile_out, f"{tile}_FirstBurnDOY.tif")

    with rasterio.open(mask_out, "w", **meta_mask) as dst:
        dst.write(final_mask.astype("uint8"), 1)

    with rasterio.open(prob_out, "w", **meta_prob) as dst:
        dst.write(max_prob.astype("float32"), 1)

    with rasterio.open(doy_out, "w", **meta_prob) as dst:
        dst.write(first_doy.astype("float32"), 1)

    # ------------------------------------------------
    # Latest FCC
    # ------------------------------------------------

    if len(fcc_files) > 0:

        latest_fcc = fcc_files[-1]

        fcc_out = os.path.join(tile_out, f"{tile}_LatestFCC.tif")

        with rasterio.open(latest_fcc) as src:
            fcc_data = src.read()
            meta_fcc = src.meta.copy()

        # Optional masking of FCC visualization
        fcc_data[:, ~valid_mask] = 0

        with rasterio.open(fcc_out, "w", **meta_fcc) as dst:
            dst.write(fcc_data)
