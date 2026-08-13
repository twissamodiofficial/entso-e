import lightgbm as lgb
import numpy as np
import pandas as pd

from baseline_eval import evaluate, evaluate_by_group
from cqr import calibrate, coverage, mean_interval_width
from train_quantile import pinball_loss

TARGET = "Actual Load"
DROP_COLS = [TARGET]
QUANTILES = [0.1, 0.5, 0.9]

if __name__ == "__main__":
    test = pd.read_parquet("data/processed/features_weather_test.parquet")
    feature_cols = [c for c in test.columns if c not in DROP_COLS]
    X_test, y_test = test[feature_cols], test[TARGET]

    # --- point forecast ---
    point_model = lgb.Booster(model_file="models/lightgbm_v1.txt")
    y_pred_point = pd.Series(point_model.predict(X_test), index=X_test.index)

    print("Point forecast (test)")
    overall = evaluate(y_test, y_pred_point, label="lightgbm_test")
    for k, v in overall.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nBy hour of day:")
    print(evaluate_by_group(y_test, y_pred_point, test["hour"], "hour").to_string(index=False))

    print("\nBy weekday/weekend:")
    print(evaluate_by_group(y_test, y_pred_point, test["weekend"], "weekend").to_string(index=False))

    print("\nQuantile models (test)")
    preds = {}
    for q in QUANTILES:
        model = lgb.Booster(model_file=f"models/lightgbm_q{q}.txt")
        preds[q] = pd.Series(model.predict(X_test), index=X_test.index)
        loss = pinball_loss(y_test, preds[q], q)
        print(f"  q{q} pinball loss: {loss:.3f}")

    crossing_low_mid = (preds[0.1] > preds[0.5]).sum()
    crossing_mid_high = (preds[0.5] > preds[0.9]).sum()
    print(f"\nQuantile crossing: q0.1 > q0.5 in {crossing_low_mid} rows, "
          f"q0.5 > q0.9 in {crossing_mid_high} rows (out of {len(y_test)})")

    median_mae = np.mean(np.abs(y_test - preds[0.5]))
    print(f"\nMedian (q0.5) MAE on test: {median_mae:.3f} "
          f"(compare to point-forecast test MAE: {overall['mae']:.3f})")

    # raw (uncalibrated) coverage on test
    cov_raw = coverage(y_test, preds[0.1], preds[0.9])
    width_raw = mean_interval_width(preds[0.1], preds[0.9])
    print(f"\nRAW [q0.1, q0.9] coverage on test: {cov_raw*100:.1f}% "
          f"(mean width {width_raw:.0f} MW)")

    # CQR: recompute Q from val (both halves this time — we're done
    # tuning, this is the final Q), then APPLY (never refit) to test
    val_preds = pd.read_parquet("data/processed/val_quantile_predictions.parquet")
    Q = calibrate(val_preds["actual"], val_preds["q10"], val_preds["q90"])
    print(f"\nCQR adjustment Q (fit on val, applied to test): {Q:.1f} MW")

    test_lower = preds[0.1] - Q
    test_upper = preds[0.9] + Q
    cov_calibrated = coverage(y_test, test_lower, test_upper)
    width_calibrated = mean_interval_width(test_lower, test_upper)
    print(f"\nCALIBRATED [q0.1, q0.9] coverage on test: {cov_calibrated*100:.1f}% "
          f"(target 80%, mean width {width_calibrated:.0f} MW)")

    out = pd.DataFrame({
        "actual": y_test,
        "point_pred": y_pred_point,
        "q10": preds[0.1],
        "q50": preds[0.5],
        "q90": preds[0.9],
        "q10_calibrated": test_lower,
        "q90_calibrated": test_upper,
    })