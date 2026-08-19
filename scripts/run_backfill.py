import sys

from entso_e_pipeline import config, registry
from entso_e_pipeline.pipeline import ForecastPipeline
from entso_e_pipeline.evaluation import metrics

pipeline = ForecastPipeline()

if "--skip-ingest" not in sys.argv:
    pipeline.ingest()

pipeline.build_features()
pipeline.train()
pipeline.calibrate()
pipeline.save_models()

val_df = pipeline.features["val"]
preds = pipeline.predict(val_df)

overall = metrics.evaluate(val_df[config.TARGET], preds["point_pred"], label="point_val")
print("\nPoint forecast (val)")
metrics.print_metrics(overall)

cov = ((val_df[config.TARGET] >= preds["q10_calibrated"]) & (val_df[config.TARGET] <= preds["q90_calibrated"])).mean()
print(f"\nCalibrated [q10, q90] coverage (val): {cov*100:.1f}%")

registry.log_run(
    metrics={
        "val_mae": overall["mae"],
        "val_rmse": overall["rmse"],
        "val_mape": overall["mape"],
        "val_coverage": cov,
    },
    params={
        "categorical_features": ",".join(config.CATEGORICAL_FEATURES),
        "weather_features": ",".join(config.WEATHER_FEATURES),
        "warmup_hours": config.WARMUP_HOURS,
    },
)