from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import numpy as np


def cross_validate_model(model, X, y, n_splits=5):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_scores = []
    f1_scores = []
    kappa_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        kappa_scores.append(cohen_kappa_score(y_test, y_pred))

        print(f"Fold {fold+1}: "
              f"ACC={acc_scores[-1]:.3f}, "
              f"F1={f1_scores[-1]:.3f}, "
              f"Kappa={kappa_scores[-1]:.3f}")

    print("\nMean CV Accuracy:", np.mean(acc_scores))
    print("Mean CV F1:", np.mean(f1_scores))
    print("Mean CV Kappa:", np.mean(kappa_scores))

    return {
        "accuracy": acc_scores,
        "f1": f1_scores,
        "kappa": kappa_scores
    }