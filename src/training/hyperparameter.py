from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def tune_random_forest(X_train, y_train):

    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    rf = RandomForestClassifier(
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    search = RandomizedSearchCV(
        rf,
        param_grid,
        n_iter=20,
        cv=3,
        scoring="f1",
        verbose=1,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_