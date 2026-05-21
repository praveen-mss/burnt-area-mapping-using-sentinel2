# src/training/calibration.py

from sklearn.calibration import CalibratedClassifierCV
import sklearn


def calibrate_model(model, X_train, y_train, method="isotonic"):

    print(f"Using sklearn version: {sklearn.__version__}")

    try:
        # For sklearn >= 1.4
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv=3
        )
    except TypeError:
        # For older sklearn versions
        calibrated = CalibratedClassifierCV(
            base_estimator=model,
            method=method,
            cv=3
        )

    calibrated.fit(X_train, y_train)

    print("Model calibrated successfully.")

    return calibrated