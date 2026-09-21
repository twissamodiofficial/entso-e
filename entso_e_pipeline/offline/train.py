"""Build validated historical splits and train the forecasting models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .. import config
from ..modeling import point, quantile
from ..modeling import metrics
from ..modeling.schema import split_target, validate_temporal_splits
from ..storage.supabase import SupabaseRawStore


def _window(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    start = pd.Timestamp(start, tz=config.TIMEZONE)
    end = pd.Timestamp(end, tz=config.TIMEZONE)
    return frame.loc[(frame.index >= start) & (frame.index < end)]


def build_splits(
    store: SupabaseRawStore | None = None,
    feature_version: str = config.FEATURE_VERSION,
    split_definition: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Read versioned model features from the canonical Supabase store."""

    store = store or SupabaseRawStore()
    frame = store.features(feature_version)
    if frame.empty:
        raise ValueError(
            "No materialized Supabase features found. "
            "Run offline.materialize first."
        )
    return _validated_splits(frame, split_definition or config.SPLITS)


def _validated_splits(
    frame: pd.DataFrame,
    split_definition: dict,
) -> dict[str, pd.DataFrame]:
    splits = {
        name: pd.concat([
            _window(frame, start, end)
            for start, end in windows
        ])
        for name, windows in split_definition.items()
    }
    validate_temporal_splits(splits)
    for name, split in splits.items():
        split_target(split, name)
    return splits


def train_models(
    selection_train: pd.DataFrame,
    selection_validation: pd.DataFrame,
    final_train: pd.DataFrame,
) -> dict[str, object]:
    """Select rounds, then refit fresh models on the final training window."""

    selected_point = point.train(selection_train, selection_validation)
    selected_quantiles = quantile.train_all(selection_train, selection_validation)
    rounds = {
        "point": selected_point.best_iteration,
        "quantile": {
            quantile_level: model.best_iteration
            for quantile_level, model in selected_quantiles.items()
        },
    }
    final = {
        "point": point.train(final_train, num_boost_round=rounds["point"]),
        "quantiles": quantile.train_all(
            final_train,
            num_boost_round=rounds["quantile"],
        ),
    }
    return {
        "selected_point": selected_point,
        "selected_quantiles": selected_quantiles,
        **final,
        "rounds": rounds,
    }


def evaluate_models(
    models: dict[str, object],
    frame: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Evaluate point and quantile forecasts without changing the models."""

    features, actual = split_target(frame, "Evaluation data")
    result = {
        "point": metrics.point_metrics(actual, point.predict(models["point"], features))
    }
    result["quantile"] = {
        str(quantile_level): metrics.quantile_metrics(
            actual,
            prediction,
            quantile_level,
        )
        for quantile_level, prediction in quantile.predict_all(
            models["quantiles"], features
        ).items()
    }
    return result


def calibrate_interval(
    models: dict[str, object],
    frame: pd.DataFrame,
    coverage: float = config.TARGET_COVERAGE,
) -> dict[str, float]:
    """Fit and report one conformal interval adjustment on calibration data."""

    features, actual = split_target(frame, "Calibration data")
    predictions = quantile.predict_all(models["quantiles"], features)
    lower = predictions[0.1]
    upper = predictions[0.9]
    adjustment = metrics.conformal_interval_adjustment(
        actual, lower, upper, coverage
    )
    return {
        "target_coverage": coverage,
        "adjustment_mw": adjustment,
        "coverage_before": metrics.interval_coverage(actual, lower, upper),
        "coverage_after": metrics.interval_coverage(
            actual, lower - adjustment, upper + adjustment
        ),
    }


def evaluate_calibrated_interval(
    models: dict[str, object],
    frame: pd.DataFrame,
    adjustment: float,
    coverage: float = config.TARGET_COVERAGE,
) -> dict[str, float]:
    """Evaluate a frozen calibration adjustment on a later split."""

    features, actual = split_target(frame, "Test data")
    predictions = quantile.predict_all(models["quantiles"], features)
    return {
        "target_coverage": coverage,
        "coverage": metrics.interval_coverage(
            actual,
            predictions[0.1] - adjustment,
            predictions[0.9] + adjustment,
        ),
        "adjustment_mw": adjustment,
    }


def save_models(
    point_model,
    quantile_models: dict[float, object],
    models_dir: str | Path = config.MODELS_DIR,
) -> None:
    """Save the final point and quantile models."""

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    point.save(point_model, str(Path(models_dir) / "lightgbm_v1.txt"))
    quantile.save(quantile_models, str(models_dir))


def save_calibration(
    interval_calibration: dict[str, float],
    models_dir: str | Path = config.MODELS_DIR,
) -> None:
    """Save the frozen interval adjustment alongside the model artifacts."""

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    (Path(models_dir) / "calibration.json").write_text(
        json.dumps(interval_calibration, indent=2) + "\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("historical", "production"),
        default="historical",
        help="Historical evaluation workflow or production refit through June 30.",
    )
    parser.add_argument(
        "--models-dir",
        help="Output directory; production defaults to models/production.",
    )
    parser.add_argument(
        "--publish-dagshub",
        action="store_true",
        help="Publish the model bundle to DagsHub MLflow and register it in Supabase.",
    )
    parser.add_argument("--model-version")
    args = parser.parse_args(argv)

    historical_splits = build_splits()
    if args.profile == "production":
        splits = build_splits(split_definition=config.PRODUCTION_SPLITS)
        models = train_models(
            historical_splits["train"],
            historical_splits["val"],
            splits["train"],
        )
        interval_calibration = calibrate_interval(models, splits["calibration"])
        metrics_report = {
            "validation_selection": evaluate_models(
                {"point": models["selected_point"], "quantiles": models["selected_quantiles"]},
                historical_splits["val"],
            ),
            "production_calibration": {
                **evaluate_models(models, splits["calibration"]),
                "interval": interval_calibration,
            },
        }
        training_window = {"start": "2019-01-01", "end": "2026-07-01"}
        calibration_window = {"start": "2026-07-01", "end": "2026-09-16"}
        default_models_dir = str(Path(config.MODELS_DIR) / "production")
    else:
        splits = historical_splits
        models = train_models(
            splits["train"],
            splits["val"],
            pd.concat([splits["train"], splits["val"]]),
        )
        interval_calibration = calibrate_interval(models, splits["calibration"])
        metrics_report = {
            "validation_selection": evaluate_models(
                {"point": models["selected_point"], "quantiles": models["selected_quantiles"]},
                splits["val"],
            ),
            "calibration": {
                **evaluate_models(models, splits["calibration"]),
                "interval": interval_calibration,
            },
            "test": {
                **evaluate_models(models, splits["test"]),
                "interval": evaluate_calibrated_interval(
                    models,
                    splits["test"],
                    interval_calibration["adjustment_mw"],
                ),
            },
        }
        training_window = {"start": "2019-01-01", "end": "2026-04-01"}
        calibration_window = {"start": "2026-04-01", "end": "2026-07-01"}
        default_models_dir = config.MODELS_DIR

    models_dir = args.models_dir or default_models_dir
    save_models(models["point"], models["quantiles"], models_dir)
    save_calibration(interval_calibration, models_dir)
    report = {
        "profile": args.profile,
        "models_dir": models_dir,
        "training_window": training_window,
        "calibration_window": calibration_window,
        "rounds": models["rounds"],
        "metrics": metrics_report,
    }
    if args.publish_dagshub:
        from ..serving.model import publish_dagshub_model

        model_version = args.model_version or (
            f"load-forecast-{args.profile}-"
            + pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        )
        report["model"] = publish_dagshub_model(
            models_dir,
            model_version,
            training_rounds=models["rounds"],
            training_metrics=report["metrics"],
            store=SupabaseRawStore(),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
