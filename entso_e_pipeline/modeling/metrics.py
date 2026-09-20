"""Evaluation metrics for validated load forecasts."""

from __future__ import annotations

import numpy as np


def mean_absolute_error(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def mean_absolute_percentage_error(actual, predicted) -> float:
    """Return MAPE as a percentage; load actuals must be nonzero."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if np.any(actual == 0):
        raise ValueError("MAPE is undefined when actual load contains zero values.")
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def pinball_loss(actual, predicted, quantile: float) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    difference = actual - predicted
    return float(np.mean(np.maximum(quantile * difference, (quantile - 1) * difference)))


def point_metrics(actual, predicted) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(actual, predicted),
        "mape_percent": mean_absolute_percentage_error(actual, predicted),
    }


def quantile_metrics(actual, predicted, quantile: float) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(actual, predicted),
        "mape_percent": mean_absolute_percentage_error(actual, predicted),
        "pinball_loss": pinball_loss(actual, predicted, quantile),
    }


def interval_coverage(actual, lower, upper) -> float:
    """Return the fraction of actuals inside a predictive interval."""

    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (actual.shape == lower.shape == upper.shape):
        raise ValueError("Interval arrays must have matching shapes.")
    return float(np.mean((actual >= lower) & (actual <= upper)))


def conformal_interval_adjustment(
    actual,
    lower,
    upper,
    coverage: float,
) -> float:
    """Fit one split-conformal widening value on a calibration period."""

    if not 0 < coverage <= 1:
        raise ValueError("Interval coverage must be between zero and one.")
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (actual.shape == lower.shape == upper.shape):
        raise ValueError("Interval arrays must have matching shapes.")
    if actual.size == 0:
        raise ValueError("Cannot calibrate an empty interval.")
    scores = np.maximum.reduce([lower - actual, actual - upper, np.zeros_like(actual)])
    rank = min(int(np.ceil((actual.size + 1) * coverage)), actual.size)
    return float(np.sort(scores)[rank - 1])
