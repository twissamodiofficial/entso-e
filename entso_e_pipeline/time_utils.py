"""Timezone and local-calendar helpers shared by pipeline stages."""

import pandas as pd


def as_local(value, timezone: str) -> pd.Timestamp:
    """Return ``value`` as a timezone-aware timestamp in ``timezone``."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def as_local_index(
    values,
    timezone: str,
    source_timezone: str = "UTC",
) -> pd.DatetimeIndex:
    """Return timestamp values as an aware index in ``timezone``.

    Naive values are interpreted in ``source_timezone``. Aware values retain
    their instant while changing representation to ``timezone``.
    """

    if source_timezone == "UTC":
        # ``utc=True`` also handles mixed fixed offsets on an autumn DST day.
        index = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    else:
        index = pd.DatetimeIndex(values)
        if index.tz is None:
            index = index.tz_localize(source_timezone)
    return index.tz_convert(timezone)


def local_day_start(value, timezone: str) -> pd.Timestamp:
    """Return the local midnight containing ``value``."""

    return as_local(value, timezone).normalize()


def hourly_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return hourly instants in the half-open interval ``[start, end)``."""

    return pd.date_range(start=start, end=end, inclusive="left", freq="h")


def local_day_hours(value, timezone: str) -> pd.DatetimeIndex:
    """Return every hourly instant in the local calendar day containing value."""

    start = local_day_start(value, timezone)
    end = start + pd.DateOffset(days=1)
    return hourly_index(start, end)


def validate_hourly_index(index: pd.DatetimeIndex, label: str) -> None:
    """Require ordered, unique, complete hourly instants, including DST days."""

    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError(f"{label} must have a timezone-aware DatetimeIndex.")
    if index.empty or index.hasnans:
        raise ValueError(f"{label} timestamps must be nonempty and contain no NaT.")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{label} timestamps must be sorted and unique.")
    utc = index.tz_convert("UTC")
    if not utc.equals(utc.floor("h")):
        raise ValueError(f"{label} timestamps must align to whole hours.")
    if ((index[1:] - index[:-1]) != pd.Timedelta(hours=1)).any():
        raise ValueError(f"{label} must be contiguous hourly data.")
