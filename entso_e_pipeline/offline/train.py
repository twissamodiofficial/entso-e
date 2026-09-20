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
) -> dict[str, pd.DataFrame]:
    """Read versioned model features from the canonical Supabase store."""

    store = store or SupabaseRawStore()
    frame = store.features(feature_version)
    if frame.empty:
        raise ValueError(
            "No materialized Supabase features found. "
            "Run offline.materialize first."
        )
    return _validated_splits(frame)


def _validated_splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits = {
        name: pd.concat([
            _window(frame, start, end)
            for start, end in windows
        ])
        for name, windows in config.SPLITS.items()
    }
    validate_temporal_splits(splits)
    for name, split in splits.items():
        split_target(split, name)
    return splits


def train_historical_models(
    splits: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """Select rounds on train/validation and refit on their combined rows."""

    train = splits["train"]
    validation = splits["val"]
    selected_point = point.train(train, validation)
    selected_quantiles = quantile.train_all(train, validation)

    rounds = {
        "point": selected_point.best_iteration,
        "quantile": {
            quantile_level: model.best_iteration
            for quantile_level, model in selected_quantiles.items()
        },
    }
    combined = pd.concat([train, validation])
    point_model = point.train(combined, num_boost_round=rounds["point"])
    quantile_models = quantile.train_all(
        combined,
        num_boost_round=rounds["quantile"],
    )
    return {
        "selected_point": selected_point,
        "selected_quantiles": selected_quantiles,
        "point": point_model,
        "quantiles": quantile_models,
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
    parser.add_argument("--models-dir", default=config.MODELS_DIR)
    parser.add_argument(
        "--publish-dagshub",
        action="store_true",
        help="Publish the model bundle to DagsHub MLflow and register it in Supabase.",
    )
    parser.add_argument("--model-version")
    args = parser.parse_args(argv)

    splits = build_splits()
    models = train_historical_models(splits)
    selected = {
        "point": models["selected_point"],
        "quantiles": models["selected_quantiles"],
    }
    final = {
        "point": models["point"],
        "quantiles": models["quantiles"],
    }
    interval_calibration = calibrate_interval(final, splits["calibration"])
    save_models(final["point"], final["quantiles"], args.models_dir)
    save_calibration(interval_calibration, args.models_dir)
    report = {
        "models_dir": args.models_dir,
        "rounds": models["rounds"],
        "metrics": {
            "validation_selection": evaluate_models(selected, splits["val"]),
            "calibration": {
                **evaluate_models(final, splits["calibration"]),
                "interval": interval_calibration,
            },
            "test": {
                **evaluate_models(final, splits["test"]),
                "interval": evaluate_calibrated_interval(
                    final,
                    splits["test"],
                    interval_calibration["adjustment_mw"],
                ),
            },
        },
    }
    if args.publish_dagshub:
        from ..serving.model import publish_dagshub_model

        model_version = args.model_version or (
            "load-forecast-"
            + pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        )
        report["model"] = publish_dagshub_model(
            args.models_dir,
            model_version,
            training_rounds=models["rounds"],
            training_metrics=report["metrics"],
            store=SupabaseRawStore(),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
