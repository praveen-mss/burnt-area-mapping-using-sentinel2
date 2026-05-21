import os
import json
import pandas as pd


def summarize_experiments(registry_dir="models/registry"):

    records = []

    for file in os.listdir(registry_dir):
        if file.endswith(".json"):

            path = os.path.join(registry_dir, file)

            try:
                with open(path) as f:
                    metadata = json.load(f)

                record = metadata["metrics"].copy()
                record["timestamp"] = metadata["timestamp"]
                records.append(record)

            except Exception as e:
                print(f"Skipping corrupted file: {file}")
                continue

    if not records:
        print("No valid experiment files found.")
        return None

    df = pd.DataFrame(records)

    if "roc_auc" in df.columns:
        df = df.sort_values("roc_auc", ascending=False)

    print(df)

    return df

def clean_registry(registry_dir="models/registry"):

    for file in os.listdir(registry_dir):
        if file.endswith(".json"):
            path = os.path.join(registry_dir, file)
            try:
                with open(path) as f:
                    json.load(f)
            except:
                print("Removing corrupted file:", file)
                os.remove(path)