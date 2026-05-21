import numpy as np
from sklearn.metrics import f1_score


def optimize_threshold(model, X_val, y_val):

    y_proba = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 50)
    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_val, y_pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = t

    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Best F1: {best_f1:.3f}")

    return best_threshold
