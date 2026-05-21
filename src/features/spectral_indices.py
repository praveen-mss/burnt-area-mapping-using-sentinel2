# src/features/spectral_indices.py

import numpy as np


EPS = 1e-6


def _safe_div(numerator, denominator):
    return numerator / (denominator + EPS)


# ---------------------------------------------------------
# Vegetation Indices
# ---------------------------------------------------------

def compute_ndvi(b8, b4):
    """
    NDVI = (NIR - RED) / (NIR + RED)
    """
    return _safe_div((b8 - b4), (b8 + b4))


def compute_savi(b8, b4, L=0.5):
    """
    SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)
    """
    return ((b8 - b4) / (b8 + b4 + L + EPS)) * (1 + L)


# ---------------------------------------------------------
# Burn Indices
# ---------------------------------------------------------

def compute_nbr(b8, b12):
    """
    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    Critical for burn severity
    """
    return _safe_div((b8 - b12), (b8 + b12))


def compute_nbr2(b11, b12):
    """
    NBR2 = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
    Sensitive to moisture and char
    """
    return _safe_div((b11 - b12), (b11 + b12))


def compute_bai(b8, b4):
    """
    Burned Area Index (BAI)
    Highlights char and ash.
    """
    return 1 / (((0.1 - b4) ** 2) + ((0.06 - b8) ** 2) + EPS)


# ---------------------------------------------------------
# Moisture Indices
# ---------------------------------------------------------

def compute_ndmi(b8, b11):
    """
    NDMI = (NIR - SWIR1) / (NIR + SWIR1)
    """
    return _safe_div((b8 - b11), (b8 + b11))


def compute_msi(b11, b8):
    """
    MSI = SWIR1 / NIR
    Moisture stress index
    """
    return _safe_div(b11, b8)


# ---------------------------------------------------------
# Urban / Soil Confusion Reduction
# ---------------------------------------------------------

def compute_ndbi(b11, b8):
    """
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    Helps separate built-up from burn scars
    """
    return _safe_div((b11 - b8), (b11 + b8))


# ---------------------------------------------------------
# MASTER FEATURE BUILDER
# ---------------------------------------------------------
# src/features/spectral_indices.py

def build_feature_stack(bands):

    B02 = bands["B02"]
    B03 = bands["B03"]
    B04 = bands["B04"]
    B08 = bands["B08"]
    B11 = bands["B11"]
    B12 = bands["B12"]

    eps = 1e-6

    # Spectral indices
    NDVI = (B08 - B04) / (B08 + B04 + eps)

    NDWI = (B03 - B08) / (B03 + B08 + eps)

    NDMI = (B08 - B11) / (B08 + B11 + eps)

    NBR1 = (B08 - B12) / (B08 + B12 + eps)

    NBR2 = (B11 - B12) / (B11 + B12 + eps)

    SAVI = 1.5 * (B08 - B04) / (B08 + B04 + 0.5 + eps)

    MIRBI = 10 * B12 - 9.8 * B11 + 2

    GEMI = (
        (2 * (B08**2 - B04**2) + 1.5 * B08 + 0.5 * B04)
        / (B08 + B04 + 0.5 + eps)
    )

    BAI = 1 / ((0.1 - B04) ** 2 + (0.06 - B08) ** 2 + eps)

    features = [
        B11,
        B12,
        B02,
        B03,
        B04,
        B08,
        BAI,
        GEMI,
        MIRBI,
        NBR1,
        NBR2,
        NDMI,
        NDVI,
        NDWI,
        SAVI,
    ]

    names = [
        "B11",
        "B12",
        "B02",
        "B03",
        "B04",
        "B08",
        "BAI",
        "GEMI",
        "MIRBI",
        "NBR1",
        "NBR2",
        "NDMI",
        "NDVI",
        "NDWI",
        "SAVI",
    ]

    feature_stack = np.stack(features)

    return feature_stack, names