"""JMA GSM weather forecasts with one lead-time policy for history and live use."""

import pandas as pd
import requests

from .. import config
from ..common.normalization import to_local_weather
from ..time_utils import as_local, validate_hourly_index

FORECAST_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
# Keep requests small enough for multi-year backfills and the free API tier.
_REQUEST_DAYS = 14


def _bounds(start, end):
    """Return local and UTC forms of a complete hourly half-open window."""

    start_local = as_local(start, config.TIMEZONE)
    end_local = as_local(end, config.TIMEZONE)
    for bound in (start_local, end_local):
        validate_hourly_index(pd.DatetimeIndex([bound]), "Weather bound")
    if start_local >= end_local:
        raise ValueError("Weather start must be before end.")
    return start_local, end_local, start_local.tz_convert("UTC"), end_local.tz_convert("UTC")


def _pull_hourly(start_utc, end_utc, columns: dict[str, str]) -> pd.DataFrame:
    """Fetch one UTC chunk and retain exactly its requested hourly instants."""

    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "models": config.WEATHER_MODEL,
        "hourly": ",".join(columns),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "timeformat": "iso8601",
        "start_date": start_utc.date().isoformat(),
        # API date bounds are inclusive; our end bound is exclusive.
        "end_date": (end_utc - pd.Timedelta(hours=1)).date().isoformat(),
    }
    response = requests.get(FORECAST_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload.get("hourly"), dict):
        raise ValueError("Weather response does not contain an hourly payload.")

    frame = pd.DataFrame(payload["hourly"])
    missing = [column for column in ("time", *columns) if column not in frame.columns]
    if missing:
        raise ValueError(f"Weather response is missing forecast columns: {missing}")
    units = payload.get("hourly_units", {})
    if any(units.get(column) != "°C" for column in columns):
        raise ValueError("Weather forecast temperatures must be reported in degrees Celsius.")

    # Select forecast fields explicitly: never substitute observed weather or
    # unsuffixed latest-run values when historical forecast data is absent.
    weather = to_local_weather(frame[["time", *columns]]).rename(columns=columns)
    weather = weather.loc[
        (weather.index >= start_utc) & (weather.index < end_utc),
        config.WEATHER_FEATURES,
    ]
    return weather


def pull_forecast(start, end) -> pd.DataFrame:
    """Fetch historical or upcoming JMA forecasts for local ``[start, end)``.

    Both uses request the same Previous Runs fields, with a fixed two-day
    (48 elapsed hour) lead offset. This deliberately uses older forecasts in
    live operation so the input policy matches training. The offset leaves a
    buffer before each target day's midnight, including 23/25-hour DST days.
    It is not a reconstruction of one particular midnight-issued model run.

    Long windows are fetched in UTC date chunks. Missing readings are preserved
    for preprocessing and final validation, without substituting other sources.
    """

    _, _, start_utc, end_utc = _bounds(start, end)
    previous_days = config.WEATHER_FORECAST_PREVIOUS_DAYS
    if type(previous_days) is not int or not 2 <= previous_days <= 7:
        raise ValueError("Weather forecast lead offset must be an integer from 2 to 7 days.")
    columns = {
        f"{feature}_previous_day{previous_days}": feature
        for feature in config.WEATHER_FEATURES
    }

    chunks = []
    chunk_start = start_utc
    while chunk_start < end_utc:
        chunk_end = min(
            chunk_start.normalize() + pd.Timedelta(days=_REQUEST_DAYS), end_utc
        )
        chunks.append(_pull_hourly(chunk_start, chunk_end, columns))
        chunk_start = chunk_end
    weather = pd.concat(chunks)
    weather.attrs.update(
        weather_source="open_meteo_previous_runs",
        weather_model=config.WEATHER_MODEL,
        weather_forecast_previous_days=previous_days,
    )
    return weather


def pull_historical_weather(start, end) -> pd.DataFrame:
    """Fetch past weather forecasts through the same path used for live inputs."""

    return pull_forecast(start, end)


__all__ = ["pull_forecast", "pull_historical_weather"]
