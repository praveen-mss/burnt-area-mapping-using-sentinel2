import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_feature_importance(model, feature_names, save_path=None):

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features[::-1], sorted_importances[::-1])
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

    return pd.DataFrame({
        "feature": sorted_features,
        "importance": sorted_importances
    })