import mlflow
import mlflow.sklearn
import os


EXPERIMENT_NAME = "churn-prediction"


def setup_mlflow():
    """Initialize MLflow experiment."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_training_run(metrics: dict, feature_importance: dict, model, scaler, params: dict = None) -> str:
    """Log a training run to MLflow."""
    setup_mlflow()

    with mlflow.start_run() as run:
        # Log parameters
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_split": 10,
            "model_type": "RandomForest"
        }
        if params:
            default_params.update(params)

        mlflow.log_params(default_params)

        # Log metrics
        for k, v in metrics.items():
            if k != "confusion_matrix":
                mlflow.log_metric(k, v)

        # Log feature importance
        for feature, importance in feature_importance.items():
            mlflow.log_metric(f"importance_{feature}", importance)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        print(f"✅ MLflow run logged: {run_id}")
        return run_id


def get_all_runs() -> list:
    """Retrieve all past experiment runs."""
    setup_mlflow()

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if not experiment:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    results = []
    for run in runs:
        results.append({
            "run_id": run.info.run_id[:8],
            "start_time": run.info.start_time,
            "accuracy": run.data.metrics.get("accuracy", 0),
            "precision": run.data.metrics.get("precision", 0),
            "recall": run.data.metrics.get("recall", 0),
            "f1_score": run.data.metrics.get("f1_score", 0),
            "roc_auc": run.data.metrics.get("roc_auc", 0),
            "status": run.info.status
        })

    return results


if __name__ == "__main__":
    from train import train

    print("Training model and logging to MLflow...")
    result = train(save=True)

    run_id = log_training_run(
        metrics=result["metrics"],
        feature_importance=result["feature_importance"],
        model=result["model"],
        scaler=result["scaler"]
    )

    print("\n📋 All Runs:")
    runs = get_all_runs()
    for r in runs:
        print(f"  Run {r['run_id']} | AUC: {r['roc_auc']} | F1: {r['f1_score']} | Status: {r['status']}")