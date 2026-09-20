"""Load and publish the point/quantile forecast model bundle."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..modeling import point, quantile


def _configure_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "DagsHub model serving requires mlflow; install the project dependencies."
        ) from exc

    repository = os.environ.get("DAGSHUB_REPO", "twissamodiofficial/entso-e")
    token = (
        os.environ.get("DAGSHUB_TOKEN")
        or os.environ.get("DAGSHUB_USER_TOKEN")
    )
    if token:
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", token)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        try:
            import dagshub

            owner, name = repository.split("/", 1)
            dagshub.init(
                repo_owner=owner,
                repo_name=name,
                mlflow=True,
                dvc=False,
            )
            tracking_uri = f"https://dagshub.com/{repository}.mlflow"
        except (ImportError, ValueError) as exc:
            raise RuntimeError(
                "DAGSHUB_REPO must use owner/name and the dagshub package "
                "must be installed when MLFLOW_TRACKING_URI is not set."
            ) from exc
    mlflow.set_tracking_uri(tracking_uri)
    os.environ.setdefault(
        "MLFLOW_TRACKING_USERNAME",
        os.environ.get("DAGSHUB_USERNAME", "token"),
    )
    return mlflow


@dataclass
class ForecastModel:
    """Loaded model bundle, including the frozen conformal adjustment."""

    point_model: object
    quantile_models: dict
    interval_adjustment_mw: float
    model_version: str
    artifact_uri: str | None = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        quantile_predictions = quantile.predict_all(self.quantile_models, features)
        return pd.DataFrame(
            {
                "point_forecast_mw": point.predict(self.point_model, features),
                "q10_forecast_mw": (
                    quantile_predictions[0.1] - self.interval_adjustment_mw
                ),
                "q50_forecast_mw": quantile_predictions[0.5],
                "q90_forecast_mw": (
                    quantile_predictions[0.9] + self.interval_adjustment_mw
                ),
            },
            index=features.index,
        )


def _from_directory(directory: str | Path, model_version: str, artifact_uri=None):
    directory = Path(directory)
    calibration_path = directory / "calibration.json"
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Model artifact is missing {calibration_path.name}; retrain and publish it."
        )
    calibration = json.loads(calibration_path.read_text())
    return ForecastModel(
        point_model=point.load(str(directory / "lightgbm_v1.txt")),
        quantile_models=quantile.load_all(str(directory)),
        interval_adjustment_mw=float(calibration["adjustment_mw"]),
        model_version=model_version,
        artifact_uri=artifact_uri,
    )


def load_local_model(models_dir: str = "models") -> ForecastModel:
    return _from_directory(models_dir, model_version="local")


def load_dagshub_model(store) -> ForecastModel:
    mlflow = _configure_mlflow()
    version = store.latest_model_version()
    if not version or not version.get("artifact_uri"):
        raise RuntimeError("No published DagsHub model artifact is registered in Supabase.")
    directory = mlflow.artifacts.download_artifacts(
        artifact_uri=version["artifact_uri"]
    )
    return _from_directory(
        directory,
        model_version=version["model_version"],
        artifact_uri=version["artifact_uri"],
    )


def publish_dagshub_model(
    models_dir: str,
    model_version: str,
    *,
    training_rounds: dict | None = None,
    training_metrics: dict | None = None,
    store=None,
) -> dict[str, str]:
    """Log the complete model bundle to DagsHub MLflow and register its URI."""

    mlflow = _configure_mlflow()
    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "entso-e-load")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=model_version) as run:
        if training_rounds:
            mlflow.log_params({
                f"rounds_{kind}": value
                for kind, value in training_rounds.items()
                if not isinstance(value, dict)
            })
        mlflow.log_artifacts(models_dir, artifact_path="model")
        if training_metrics:
            flattened = {}
            for split, values in training_metrics.items():
                if isinstance(values, dict):
                    for name, value in values.items():
                        if isinstance(value, (int, float)):
                            flattened[f"{split}_{name}"] = float(value)
            if flattened:
                mlflow.log_metrics(flattened)
        artifact_uri = f"runs:/{run.info.run_id}/model"

    if store is not None:
        store.upsert_model_version(
            model_version,
            artifact_uri=artifact_uri,
            training_rounds=training_rounds,
            training_metrics=training_metrics,
        )
    return {"model_version": model_version, "artifact_uri": artifact_uri}
