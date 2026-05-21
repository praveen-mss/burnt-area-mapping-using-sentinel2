# src/registry/model_registry.py

import os
import json
import joblib
import numpy as np
from datetime import datetime


def _convert_to_serializable(obj):
    """
    Recursively convert numpy objects to Python native types.
    """

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]

    return obj


def register_model(model, metrics, feature_list, registry_dir="models/registry"):

    os.makedirs(registry_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = os.path.join(registry_dir, f"model_{timestamp}.pkl")
    metadata_path = os.path.join(registry_dir, f"model_{timestamp}.json")

    # Save model
    joblib.dump(model, model_path)

    # Convert metrics safely
    clean_metrics = _convert_to_serializable(metrics)

    metadata = {
        "timestamp": timestamp,
        "features": feature_list,
        "metrics": clean_metrics
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("Model registered:", model_path)
    print("Metadata saved:", metadata_path)

    return model_path, metadata_path