# src/preprocessing/cloud_mask.py

import numpy as np


SCL_MASK_DEFAULT = {
    "saturated": [1],
    "cloud_shadow": [3],
    "clouds": [7, 8, 9, 10],
    "snow": [11],
    "water": [6],
    "terrain_shadow": [2]
}


def generate_mask(
    scl: np.ndarray,
    mask_water: bool = True,
    mask_terrain_shadow: bool = False
) -> np.ndarray:

    invalid_classes = []

    invalid_classes += SCL_MASK_DEFAULT["saturated"]
    invalid_classes += SCL_MASK_DEFAULT["cloud_shadow"]
    invalid_classes += SCL_MASK_DEFAULT["clouds"]
    invalid_classes += SCL_MASK_DEFAULT["snow"]

    if mask_water:
        invalid_classes += SCL_MASK_DEFAULT["water"]

    if mask_terrain_shadow:
        invalid_classes += SCL_MASK_DEFAULT["terrain_shadow"]

    mask = ~np.isin(scl, invalid_classes)

    return mask


def apply_mask(image_stack: np.ndarray, mask: np.ndarray) -> np.ndarray:

    masked = image_stack.copy()

    for i in range(masked.shape[0]):
        masked[i, ~mask] = np.nan

    return masked