import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from src.training.sample_training_data import sample_training_data
from src.training.split import stratified_split
from src.training.hyperparameter import tune_random_forest
from src.training.class_weights import compute_scale_pos_weight
from src.evaluation.metrics import evaluate_model


def train_model_from_csv(
    csv_path,
    feature_list,
    label_col,
    model_output_path,
    tune=True,
    use_xgb=True,
    use_ensemble=True
):

    print("Reading training data...")

    df = pd.read_csv(csv_path)

    # =====================================================
    # Standardize training band names
    # =====================================================
    rename_map = {
        "BLUE": "B02",
        "GREEN": "B03",
        "RED": "B04",
        "NIR": "B08",
        "SWIR1": "B11",
        "SWIR2": "B12"
    }

    df = df.rename(columns=rename_map)

    print("Available columns:")
    print(df.columns.tolist())

    # =====================================================
    # Check missing features
    # =====================================================
    missing_features = [
        f for f in feature_list if f not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"\nMissing required features: {missing_features}\n"
            f"\nAvailable columns:\n{df.columns.tolist()}"
        )

    # =====================================================
    # Keep required features only
    # =====================================================
    df = df[feature_list + [label_col]].dropna()

    print(f"Samples after dropna: {len(df)}")

    # =====================================================
    # Sample data
    # =====================================================
    df = sample_training_data(
        df,
        label_col,
        max_samples=11350430 ## Full dataset size
    )

    # =====================================================
    # Train-test split
    # =====================================================
    X_train, X_test, y_train, y_test = stratified_split(
        df,
        label_col
    )

    X_train_model = X_train.to_numpy()
    X_test_model = X_test.to_numpy()

    # =====================================================
    # Random Forest
    # =====================================================
    if tune:

        print("Tuning Random Forest...")

        rf_model, best_params = tune_random_forest(
            X_train_model,
            y_train
        )

        print("Best params:", best_params)

    else:

        print("Training Random Forest with class weighting...")

        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(X_train_model, y_train)

    model = rf_model

    # =====================================================
    # Optional XGBoost
    # =====================================================
    if use_xgb or use_ensemble:

        print("Training XGBoost model with imbalance handling...")
        from xgboost import XGBClassifier

        scale_weight = compute_scale_pos_weight(y_train)

        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            eval_metric="logloss",
            scale_pos_weight=scale_weight
        )

        xgb.fit(X_train_model, y_train)

        if use_ensemble:
            print("Building soft-voting RF + XGBoost ensemble...")

            model = VotingClassifier(
                estimators=[
                    ("rf", rf_model),
                    ("xgb", xgb),
                ],
                voting="soft",
                n_jobs=-1
            )
            model.fit(X_train_model, y_train)
        elif use_xgb:
            model = xgb

    # =====================================================
    # Evaluation
    # =====================================================
    print("Evaluating model...")

    results = evaluate_model(
        model,
        X_test_model,
        y_test
    )

    # =====================================================
    # Save model
    # =====================================================
    joblib.dump(model, model_output_path)

    print(f"Model saved to: {model_output_path}")

    return model, results, X_train_model, X_test_model, y_train, y_test
