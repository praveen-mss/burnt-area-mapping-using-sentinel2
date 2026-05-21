import os
import json

from src.training.train import train_model_from_csv
from src.evaluation.threshold_optimization import optimize_threshold
from src.registry.model_registry import register_model
from src.registry.mlflow_registry import log_experiment
from src.utils.logger import get_logger

logger = get_logger()


def run_training_pipeline(config):

    logger.info("Starting training pipeline")

    model, results, X_train, X_test, y_train, y_test = train_model_from_csv(
        csv_path=config["csv_path"],
        feature_list=config["features"],
        label_col=config["label_col"],
        model_output_path=config["model_output_path"],
        tune=config.get("tune", True),
        use_xgb=config.get("use_xgb", False),
        use_ensemble=config.get("use_ensemble", False)
    )

    logger.info("Optimizing burn probability threshold")

    threshold = optimize_threshold(model, X_test, y_test)
    results["optimal_threshold"] = float(threshold)

    # Save metrics JSON
    metrics_dir = "models/metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_path = os.path.join(metrics_dir, "training_metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Accuracy metrics saved to {metrics_path}")

    # Register model
    register_model(model, results, config["features"])

    # MLflow logging
    log_experiment(model, results, config["features"])

    logger.info("Training pipeline completed")

    return model, results
