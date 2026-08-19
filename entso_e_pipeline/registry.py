import os

import dagshub
import mlflow

from . import config

EXPERIMENT_NAME = "entso-e-load-forecast"
DAGSHUB_REPO_OWNER = "twissamodiofficial"
DAGSHUB_REPO_NAME = "entso-e"

# use the same name as config.MODELS_DIR itself, so the uploaded artifact
# folder and the downloaded local folder always match exactly
ARTIFACT_PATH = os.path.basename(config.MODELS_DIR.rstrip("/"))


def _setup():
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_run(metrics: dict, params: dict = None):
    """Log everything currently in config.MODELS_DIR (models + transformers
    + cqr_q, whatever save_models() just wrote) as one MLflow run's
    artifacts, plus the val metrics for tracking over time."""
    _setup()
    with mlflow.start_run():
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(config.MODELS_DIR, artifact_path=ARTIFACT_PATH)
        run_id = mlflow.active_run().info.run_id
    print(f"Logged MLflow run {run_id}")
    return run_id


def pull_latest_artifacts():
    """Download the most recent run's artifacts into config.MODELS_DIR,
    so point.load()/quantile.load_all()/pipeline.load_transformers() work
    unchanged — this is what CI calls before predicting, since it starts
    with no local models/ directory."""
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
    dst_parent = os.path.dirname(config.MODELS_DIR.rstrip("/")) or "."
    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=ARTIFACT_PATH, dst_path=dst_parent
    )
    print(f"Pulled artifacts from run {run_id} into {downloaded_path}")
    print("Contents:", os.listdir(downloaded_path))
    return run_id