import pandas as pd

from entso_e_pipeline import config
from entso_e_pipeline.pipeline import ForecastPipeline
from entso_e_pipeline.evaluation import metrics
from entso_e_pipeline.modeling import calibration

pipeline = ForecastPipeline()
pipeline.load_features()
pipeline.load_models()
pipeline.cqr_q = pipeline.calibrate()

test_df = pipeline.features["test"]
preds = pipeline.predict(test_df)

print("Point forecast (test)")
overall = metrics.evaluate(test_df[config.TARGET], preds["point_pred"], label="point_test")
metrics.print_metrics(overall)

print("\nBy hour of day:")
print(metrics.evaluate_by_group(test_df[config.TARGET], preds["point_pred"], test_df["hour"], "hour").to_string(index=False))

print("\nBy weekday/weekend:")
print(metrics.evaluate_by_group(test_df[config.TARGET], preds["point_pred"], test_df["weekend"], "weekend").to_string(index=False))

cov_raw = calibration.coverage(test_df[config.TARGET], preds["q10"], preds["q90"])
cov_calibrated = calibration.coverage(test_df[config.TARGET], preds["q10_calibrated"], preds["q90_calibrated"])
width_calibrated = calibration.mean_interval_width(preds["q10_calibrated"], preds["q90_calibrated"])

print(f"\nRaw [q10, q90] coverage (test): {cov_raw*100:.1f}%")
print(f"Calibrated [q10, q90] coverage (test): {cov_calibrated*100:.1f}% (target {config.TARGET_COVERAGE*100:.0f}%, width {width_calibrated:.0f} MW)")

out = pd.concat([test_df[[config.TARGET]].rename(columns={config.TARGET: "actual"}), preds], axis=1)
out.to_parquet(f"{config.PROCESSED_DIR}/test_predictions_final.parquet")
