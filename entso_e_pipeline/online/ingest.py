"""Ingest newly available observations into the shared raw store."""

from __future__ import annotations

import pandas as pd

from .. import config, preprocessing
from ..ingestion.load import fetch_load
from ..ingestion.weather import pull_forecast
from ..storage.supabase import SupabaseRawStore
from ..time_utils import as_local


def ingest_load_window(start, end, store: SupabaseRawStore | None = None) -> int:
    """Fetch source load readings and upsert them without preprocessing."""

    store = store or SupabaseRawStore()
    return store.upsert_load(fetch_load(start, end))


def ingest_previous_day_load(
    reference_date=None,
    store: SupabaseRawStore | None = None,
) -> int:
    """Ensure the seven-day history before a forecast date exists."""

    return ingest_load_catchup(reference_date, store)


def _missing_history_days(
    store: SupabaseRawStore,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Return local days whose processed hourly load history is incomplete."""

    raw_load = store.load(start, end)
    hourly = preprocessing.prepare_load(raw_load, start=start, end=end)
    missing = []
    for day in pd.date_range(start, end, freq="D", inclusive="left"):
        expected = pd.date_range(
            day,
            day + pd.DateOffset(days=1),
            freq="h",
            inclusive="left",
        )
        if hourly.reindex(expected)[config.TARGET].isna().any():
            missing.append(day)
    return missing


def ingest_load_catchup(
    reference_date=None,
    store: SupabaseRawStore | None = None,
) -> int:
    """Catch up all missing load days needed before a forecast date.

    Supabase is the availability watermark. The source API is queried from the
    first missing Amsterdam day through the cutoff, not just for the previous
    day. A forecast cannot proceed until every required hourly history value is
    present after preprocessing.
    """

    reference = as_local(
        reference_date or pd.Timestamp.now(tz=config.TIMEZONE),
        config.TIMEZONE,
    ).normalize()
    history_start = reference - pd.DateOffset(days=config.LOOKBACK_DAYS)
    store = store or SupabaseRawStore()
    missing = _missing_history_days(store, history_start, reference)
    if not missing:
        return 0

    fetched_rows = ingest_load_window(missing[0], reference, store)
    remaining = _missing_history_days(store, history_start, reference)
    if remaining:
        dates = ", ".join(day.date().isoformat() for day in remaining)
        raise RuntimeError(
            "Required load history is still incomplete after catch-up; "
            f"missing Amsterdam days: {dates}"
        )
    return fetched_rows


def ingest_weather_window(start, end, store: SupabaseRawStore | None = None) -> int:
    """Fetch forecast weather and upsert it without feature engineering."""

    store = store or SupabaseRawStore()
    return store.upsert_weather(pull_forecast(start, end))
