"""Clean source readings before feature engineering; never interpolate gaps."""

import numpy as np
import pandas as pd

from . import config
from .time_utils import as_local, as_local_index, hourly_index


def _clean(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Sort, reconcile identical duplicates, and mark invalid readings missing."""

    if not frame.columns.is_unique:
        raise ValueError("Source data contains duplicate column names.")
    frame = frame.loc[:, columns].copy()
    frame.index = as_local_index(frame.index, config.TIMEZONE)
    if frame.index.hasnans:
        raise ValueError("Source data contains missing timestamps.")
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        conflicts = frame.groupby(level=0).nunique(dropna=False).gt(1).any(axis=1)
        if conflicts.any():
            raise ValueError("Source data contains conflicting duplicate timestamps.")
        frame = frame.loc[~frame.index.duplicated()]
    return frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def prepare_load(raw_load: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Average available readings within each hour, leaving empty hours missing.

    Pass both bounds for an exact half-open hourly output window. Neither
    partial hours nor entirely missing hours are rejected during preprocessing.
    No reading outside an hour contributes to its value.
    """

    load = _clean(raw_load, [config.TARGET])
    start = as_local(start, config.TIMEZONE) if start is not None else None
    end = as_local(end, config.TIMEZONE) if end is not None else None
    for bound in (start, end):
        if bound is not None and bound != bound.tz_convert("UTC").floor("h"):
            raise ValueError("Preprocessing bounds must align to whole hours.")
    if start is not None and end is not None and start >= end:
        raise ValueError("Preprocessing start must precede end.")
    if start is not None:
        load = load.loc[load.index >= start]
    if end is not None:
        load = load.loc[load.index < end]
    hourly = load.resample("h").mean()
    if start is not None and end is not None:
        hourly = hourly.reindex(hourly_index(start, end))
    return hourly


def prepare_weather(raw_weather: pd.DataFrame) -> pd.DataFrame:
    """Clean already-hourly forecasts without filling from other hours or runs."""

    return _clean(raw_weather, config.WEATHER_FEATURES)
