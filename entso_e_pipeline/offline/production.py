"""Refit and publish the production model using saved offline settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .. import config
from ..storage.supabase import SupabaseRawStore
from .train import (
    build_splits,
    calibrate_interval,
    evaluate_models,
    fit_models,
    load_training_config,
    save_calibration,
    save_models,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-config",
        default=str(Path(config.MODELS_DIR) / "training_config.json"),
        help="Offline training configuration containing selected rounds.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(Path(config.MODELS_DIR) / "production"),
        help="Output directory for the production model bundle.",
    )
    parser.add_argument(
        "--publish-dagshub",
        action="store_true",
        help="Publish the production bundle and register it in Supabase.",
    )
    parser.add_argument("--model-version")
    args = parser.parse_args(argv)

    training_config = load_training_config(args.training_config)
    splits = build_splits(
        feature_version=training_config["feature_version"],
        split_definition=config.PRODUCTION_SPLITS,
    )
    models = fit_models(splits["train"], training_config["rounds"])
    interval_calibration = calibrate_interval(models, splits["calibration"])
    metrics_report = {
        "production_calibration": {
            **evaluate_models(models, splits["calibration"]),
            "interval": interval_calibration,
        },
    }
    save_models(models["point"], models["quantiles"], args.models_dir)
    save_calibration(interval_calibration, args.models_dir)
    report = {
        "profile": "production",
        "models_dir": args.models_dir,
        "training_config": args.training_config,
        "training_window": {"start": "2019-01-01", "end": "2026-07-01"},
        "calibration_window": {"start": "2026-07-01", "end": "2026-09-16"},
        "rounds": training_config["rounds"],
        "metrics": metrics_report,
    }
    if args.publish_dagshub:
        from ..serving.model import publish_dagshub_model

        model_version = args.model_version or (
            "load-forecast-production-"
            + pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        )
        report["model"] = publish_dagshub_model(
            args.models_dir,
            model_version,
            training_rounds=training_config["rounds"],
            training_metrics=report["metrics"],
            store=SupabaseRawStore(),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
