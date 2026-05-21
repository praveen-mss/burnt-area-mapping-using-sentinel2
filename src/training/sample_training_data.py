import pandas as pd
import numpy as np


def sample_training_data(df, label_col, max_samples=500000, random_state=42):

    if len(df) <= max_samples:
        return df

    print(f"Downsampling training data to {max_samples} samples")

    class_counts = df[label_col].value_counts()
    raw_targets = class_counts / len(df) * max_samples
    sample_counts = np.floor(raw_targets).astype(int)
    sample_counts = sample_counts.clip(lower=1)

    remainder = max_samples - int(sample_counts.sum())
    if remainder > 0:
        for label in (raw_targets - sample_counts).sort_values(ascending=False).index[:remainder]:
            sample_counts[label] += 1
    elif remainder < 0:
        for label in sample_counts.sort_values(ascending=False).index[:abs(remainder)]:
            if sample_counts[label] > 1:
                sample_counts[label] -= 1

    sampled = []
    for label, group in df.groupby(label_col):
        sampled.append(
            group.sample(
                min(sample_counts[label], len(group)),
                random_state=random_state
            )
        )

    return pd.concat(sampled).reset_index(drop=True)
