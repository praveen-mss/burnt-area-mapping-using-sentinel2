import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score


def evaluate_model(model, X_test, y_test, labels=(0, 1), positive_label=1):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred, labels=list(labels))

    tn, fp, fn, tp = cm.ravel()

    overall_accuracy = accuracy_score(y_test, y_pred)

    producers_accuracy = np.divide(
        cm.diagonal(),
        cm.sum(axis=1),
        out=np.zeros(len(labels), dtype=float),
        where=cm.sum(axis=1) != 0
    )
    users_accuracy = np.divide(
        cm.diagonal(),
        cm.sum(axis=0),
        out=np.zeros(len(labels), dtype=float),
        where=cm.sum(axis=0) != 0
    )

    positive_index = list(labels).index(positive_label)
    burned_pa = producers_accuracy[positive_index]
    burned_ua = users_accuracy[positive_index]

    f1 = f1_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
    if len(np.unique(y_test)) < 2:
        roc_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_test, y_prob)

    results = {
        "overall_accuracy": float(overall_accuracy),
        "f1_score": float(f1),
        "roc_auc": None if np.isnan(roc_auc) else float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "producers_accuracy": producers_accuracy.tolist(),
        "users_accuracy": users_accuracy.tolist(),
        "burnt_class_producers_accuracy": float(burned_pa),
        "burnt_class_users_accuracy": float(burned_ua)
    }

    print("\n===== MODEL ACCURACY METRICS =====")

    print(f"Overall Accuracy : {overall_accuracy:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    if np.isnan(roc_auc):
        print("ROC AUC          : undefined (single class in y_test)")
    else:
        print(f"ROC AUC          : {roc_auc:.4f}")

    print("\nProducer's Accuracy:")
    print(f"  Unburned : {producers_accuracy[0]:.4f}")
    print(f"  Burned   : {producers_accuracy[1]:.4f}")

    print("\nUser's Accuracy:")
    print(f"  Unburned : {users_accuracy[0]:.4f}")
    print(f"  Burned   : {users_accuracy[1]:.4f}")

    print("\nBurnt Class Accuracy:")
    print(f"  Producer's Accuracy : {burned_pa:.4f}")
    print(f"  User's Accuracy     : {burned_ua:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return results
