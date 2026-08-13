import numpy as np
import pandas as pd

TARGET_COVERAGE = 0.80


def conformity_scores(y_true, q_lo, q_hi):
    return np.maximum(q_lo - y_true, y_true - q_hi)


def calibrate(y_cal, q_lo_cal, q_hi_cal, target_coverage=TARGET_COVERAGE):
    scores = conformity_scores(y_cal, q_lo_cal, q_hi_cal)
    n = len(scores)
    level = np.ceil((n + 1) * target_coverage) / n
    level = min(level, 1.0)
    Q = np.quantile(scores, level)
    return Q


def apply_calibration(q_lo, q_hi, Q):
    return q_lo - Q, q_hi + Q


def coverage(y_true, lower, upper):
    inside = (y_true >= lower) & (y_true <= upper)
    return inside.mean()


def mean_interval_width(lower, upper):
    return (upper - lower).mean()


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/val_quantile_predictions.parquet")
    df = df.sort_index()

    split_point = len(df) // 2
    cal, eval_ = df.iloc[:split_point], df.iloc[split_point:]

    print(f"Calibration set: {len(cal)} rows ({cal.index.min()} to {cal.index.max()})")
    print(f"Eval set:        {len(eval_)} rows ({eval_.index.min()} to {eval_.index.max()})")

    cov_before = coverage(eval_["actual"], eval_["q10"], eval_["q90"])
    width_before = mean_interval_width(eval_["q10"], eval_["q90"])
    print(f"\nBEFORE calibration (eval half): coverage={cov_before*100:.1f}%, "
          f"mean interval width={width_before:.0f} MW")

    Q = calibrate(cal["actual"], cal["q10"], cal["q90"])
    print(f"\nCalibration adjustment Q: {Q:.1f} MW (each bound widened by this)")

    eval_lower, eval_upper = apply_calibration(eval_["q10"], eval_["q90"], Q)
    cov_after = coverage(eval_["actual"], eval_lower, eval_upper)
    width_after = mean_interval_width(eval_lower, eval_upper)
    print(f"\nAFTER calibration (eval half):  coverage={cov_after*100:.1f}%, "
          f"mean interval width={width_after:.0f} MW")

    print(f"\nTarget coverage: {TARGET_COVERAGE*100:.0f}%")
    print(f"Width cost of calibration: +{width_after - width_before:.0f} MW "
          f"({(width_after/width_before - 1)*100:.1f}% wider)")