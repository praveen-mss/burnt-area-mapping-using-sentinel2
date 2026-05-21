# src/registry/best_model_selector.py

import os
import json


def select_best_model_info(registry_dir="models/registry", metric="roc_auc"):

    if not os.path.isdir(registry_dir):
        raise FileNotFoundError(f"Model registry directory not found: {registry_dir}")

    best_score = -1
    best_model_path = None
    best_metadata = None

    for file in os.listdir(registry_dir):

        if not file.endswith(".json"):
            continue

        path = os.path.join(registry_dir, file)

        try:
            with open(path) as f:
                metadata = json.load(f)

        except Exception as e:
            print(f"Skipping corrupted file: {file}")
            continue

        score = metadata.get("metrics", {}).get(metric, -1)

        if score is None:
            continue

        if score > best_score:
            best_score = score
            best_model_path = file.replace(".json", ".pkl")
            best_metadata = metadata

    if best_model_path is None:
        raise FileNotFoundError(f"No valid model metadata found in: {registry_dir}")

    model_path = os.path.join(registry_dir, best_model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model metadata points to missing file: {model_path}")

    print(f"\nBest model based on {metric}: {best_model_path}")
    print(f"Score: {best_score:.4f}")

    return model_path, best_metadata


def select_best_model(registry_dir="models/registry", metric="roc_auc"):

    model_path, _ = select_best_model_info(registry_dir, metric)

    return model_path
