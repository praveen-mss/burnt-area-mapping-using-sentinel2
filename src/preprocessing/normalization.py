# src/preprocessing/normalization.py

import numpy as np


def minmax_scale(image: np.ndarray) -> np.ndarray:
    scaled = image.copy()
    for i in range(scaled.shape[0]):
        band = scaled[i]
        min_val = np.nanmin(band)
        max_val = np.nanmax(band)
        scaled[i] = (band - min_val) / (max_val - min_val + 1e-6)
    return scaled


def standardize(image: np.ndarray) -> np.ndarray:
    standardized = image.copy()
    for i in range(standardized.shape[0]):
        band = standardized[i]
        mean = np.nanmean(band)
        std = np.nanstd(band)
        standardized[i] = (band - mean) / (std + 1e-6)
    return standardized