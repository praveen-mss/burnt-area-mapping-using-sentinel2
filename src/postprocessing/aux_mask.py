import os
import rasterio
import numpy as np

from rasterio.warp import reproject, Resampling


def align_mask(mask_path, ref_meta):

    with rasterio.open(mask_path) as src:

        src_data = src.read(1)

        aligned = np.zeros(
            (ref_meta["height"], ref_meta["width"]),
            dtype=np.uint8
        )

        reproject(
            source=src_data,
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            resampling=Resampling.nearest
        )

    return aligned


def apply_aux_masks(mask, tile, config, ref_meta, return_valid_mask=False):

    aux_root = config["postprocessing"]["masks"]["mask_path"]

    tile_dir = os.path.join(aux_root, tile)

    if not os.path.exists(tile_dir):

        if return_valid_mask:
            valid = np.ones_like(mask, dtype=bool)
            return mask, valid

        return mask

    masks_cfg = config["postprocessing"]["masks"]

    water = None
    urban = None
    forest = None
    agriculture = None

    def load(name):

        path = os.path.join(tile_dir, f"{name}.tif")

        if os.path.exists(path):

            return align_mask(path, ref_meta)

        return None

    if masks_cfg["use_water"]:
        water = load("water")

    if masks_cfg["use_urban"]:
        urban = load("urban")

    if masks_cfg["use_forest"]:
        forest = load("forest")

    if masks_cfg["use_agriculture"]:
        agriculture = load("agriculture")

    valid = np.ones_like(mask, dtype=bool)

    # remove water
    if water is not None:
        valid &= (water == 0)

    # remove urban
    if urban is not None:
        valid &= (urban == 0)

    # retain forest or agriculture if provided
    if forest is not None or agriculture is not None:

        keep = np.zeros_like(mask, dtype=bool)

        if forest is not None:
            keep |= (forest == 1)

        if agriculture is not None:
            keep |= (agriculture == 1)

        valid &= keep

    masked = mask.copy()

    masked[~valid] = 0

    if return_valid_mask:
        return masked, valid

    return masked