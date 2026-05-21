import mlflow
import mlflow.sklearn


def log_experiment(model, metrics, feature_list):

    mlflow.set_experiment("Burnt_Area_RF")

    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("features", ",".join(feature_list))

        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print("Experiment logged in MLflow.")