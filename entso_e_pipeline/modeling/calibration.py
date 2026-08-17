import numpy as np
import pandas as pd

from .. import config


def conformity_scores(y_true, q_lo, q_hi):
    return np.maximum(q_lo - y_true, y_true - q_hi)


def calibrate(y_cal, q_lo_cal, q_hi_cal, target_coverage=config.TARGET_COVERAGE) -> float:
    scores = conformity_scores(y_cal, q_lo_cal, q_hi_cal)
    n = len(scores)
    level = min(np.ceil((n + 1) * target_coverage) / n, 1.0)
    return np.quantile(scores, level)


def apply(q_lo, q_hi, Q: float):
    return q_lo - Q, q_hi + Q


def coverage(y_true, lower, upper) -> float:
    return ((y_true >= lower) & (y_true <= upper)).mean()


def mean_interval_width(lower, upper) -> float:
    return (upper - lower).mean()
