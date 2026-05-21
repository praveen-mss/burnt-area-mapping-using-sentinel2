import os
import pyproj

os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()

import rasterio
import joblib
import numpy as np
import pandas as pd
import rasterio

from src.ingestion.sentinel_loader import Sentinel2L2ALoader
from src.features.spectral_indices import build_feature_stack
from src.registry.best_model_selector import select_best_model_info
from src.utils.logger import get_logger

logger = get_logger()


def _parse_safe_name(safe):

    base = os.path.basename(safe)

    tile = base.split("_")[5]
    date = base.split("_")[2][:8]

    return tile, date


def predict_tile(
    safe_path,
    registry_dir,
    threshold,
    output_dir,
    use_registered_threshold=False
):

    tile, date = _parse_safe_name(safe_path)

    logger.info(f"Processing {tile} {date}")

    os.makedirs(output_dir, exist_ok=True)

    model_path, metadata = select_best_model_info(registry_dir)
    model = joblib.load(model_path)

    metadata_threshold = metadata.get("metrics", {}).get("optimal_threshold")
    if use_registered_threshold and metadata_threshold is not None:
        threshold = float(metadata_threshold)
        logger.info(f"Using registered optimal threshold: {threshold:.3f}")
    else:
        threshold = float(threshold)
        logger.info(f"Using user-configured threshold: {threshold:.3f}")

    loader = Sentinel2L2ALoader(safe_path)

    bands = ["B02","B03","B04","B08","B11","B12"]

    # loader already applies SCL masking
    stack, meta, scl_mask = loader.load_bands(bands)

    band_dict = {
        "B02":stack[0],
        "B03":stack[1],
        "B04":stack[2],
        "B08":stack[3],
        "B11":stack[4],
        "B12":stack[5],
    }

    # --------------------------------------------
    # Build feature stack
    # --------------------------------------------

    feature_stack, feature_names = build_feature_stack(band_dict)

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        expected = metadata.get("features", [])

    if not expected:
        raise ValueError("Selected model does not expose feature names and metadata has no feature list.")

    missing_features = [f for f in expected if f not in feature_names]
    if missing_features:
        raise ValueError(f"Model expects features not produced by inference: {missing_features}")

    idx = [feature_names.index(f) for f in expected]

    feature_stack = feature_stack[idx]

    H, W = feature_stack.shape[1:]

    X = feature_stack.reshape(len(expected), -1).T

    X = pd.DataFrame(X, columns=expected)

    valid = ~X.isna().any(axis=1)

    probs = np.zeros(len(X))
    uncertainty = np.zeros(len(X))

    X_valid = X[valid]
    predict_with_feature_names = hasattr(model, "feature_names_in_")

    batch = 500000

    valid_index = np.where(valid)[0]

    for i in range(0, len(X_valid), batch):

        X_batch = X_valid.iloc[i:i+batch]
        if not predict_with_feature_names:
            X_batch = X_batch.to_numpy()

        part = model.predict_proba(X_batch)

        idx_slice = valid_index[i:i+batch]

        probs[idx_slice] = part[:,1]

        uncertainty[idx_slice] = 1 - np.max(part,axis=1)

    prob = probs.reshape(H, W)

    burn_mask = (prob >= threshold).astype(np.uint8)

    uncert = uncertainty.reshape(H, W)

    # --------------------------------------------
    # Metadata
    # --------------------------------------------

    meta_prob = meta.copy()
    meta_prob.update(
        count=1,
        dtype="float32",
        compress="LZW"
    )

    meta_mask = meta.copy()
    meta_mask.update(
        count=1,
        dtype="uint8",
        compress="LZW"
    )

    meta_fcc = meta.copy()
    meta_fcc.update(
        count=3,
        dtype="float32",
        compress="LZW"
    )

    # --------------------------------------------
    # Output paths
    # --------------------------------------------

    prob_file = os.path.join(output_dir,f"{tile}_{date}_BurntProb.tif")
    mask_file = os.path.join(output_dir,f"{tile}_{date}_BurntMask.tif")
    unc_file  = os.path.join(output_dir,f"{tile}_{date}_Uncertainty.tif")
    fcc_file  = os.path.join(output_dir,f"{tile}_{date}_FCC.tif")

    # --------------------------------------------
    # Save probability
    # --------------------------------------------

    with rasterio.open(prob_file,"w",**meta_prob) as dst:
        dst.write(prob.astype("float32"),1)

    # --------------------------------------------
    # Save burn mask
    # --------------------------------------------

    with rasterio.open(mask_file,"w",**meta_mask) as dst:
        dst.write(burn_mask,1)

    # --------------------------------------------
    # Save uncertainty
    # --------------------------------------------

    with rasterio.open(unc_file,"w",**meta_prob) as dst:
        dst.write(uncert.astype("float32"),1)

    # --------------------------------------------
    # Save FCC (B12,B8,B3)
    # --------------------------------------------

    fcc = np.stack([
        band_dict["B12"],
        band_dict["B08"],
        band_dict["B03"]
    ])

    with rasterio.open(fcc_file,"w",**meta_fcc) as dst:
        dst.write(fcc.astype("float32"))

    logger.info(f"Completed {tile} \n")
    logger.info(f"============================================================= \n")

    return tile
