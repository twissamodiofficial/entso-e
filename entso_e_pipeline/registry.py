import os

import dagshub
import mlflow

from . import config

EXPERIMENT_NAME = "entso-e-load-forecast"
DAGSHUB_REPO_OWNER = "twissamodiofficial"
DAGSHUB_REPO_NAME = "entso-e"


def _setup():
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_run(metrics: dict, params: dict = None):
    _setup()
    with mlflow.start_run():
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(config.MODELS_DIR, artifact_path="model")
        run_id = mlflow.active_run().info.run_id
    print(f"Logged MLflow run {run_id}")
    return run_id


def pull_latest_artifacts():
    _setup()
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No runs found in experiment '{EXPERIMENT_NAME}'")

    run_id = runs[0].info.run_id
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="model", dst_path=os.path.dirname(config.MODELS_DIR) or "."
    )
    print(f"Pulled artifacts from run {run_id} into {config.MODELS_DIR}")
    return run_id