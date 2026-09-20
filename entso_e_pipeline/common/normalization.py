"""Timestamp decoding and final numeric-value validation."""

import numpy as np
import pandas as pd

from .. import config
from ..time_utils import as_local_index


def to_local_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Decode timestamps without cleaning, filling, or dropping source rows."""

    frame = weather.copy()
    timestamps = frame.pop("time") if "time" in frame.columns else frame.index
    frame.index = as_local_index(timestamps, config.TIMEZONE)
    frame.index.name = "time"
    return frame


def validate_numeric_values(frame: pd.DataFrame, label: str) -> None:
    """Reject missing, nonnumeric and infinite values in final model inputs."""

    for column in frame.columns:
        values = frame[column]
        if values.isna().any():
            raise ValueError(f"{label} contains missing values in {column!r}.")
        if (
            not pd.api.types.is_numeric_dtype(values.dtype)
            or pd.api.types.is_complex_dtype(values.dtype)
        ):
            raise TypeError(f"{label} column {column!r} must be real numeric data.")
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains non-finite values in {column!r}.")
