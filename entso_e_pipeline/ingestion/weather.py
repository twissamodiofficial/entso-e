import os
import time

import pandas as pd
import requests

from .. import config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_SPLITS = {
    "train": ("2019-01-01", "2025-12-31"),
    "val": ("2026-01-01", "2026-03-31"),
    "test": ("2026-04-01", "2026-06-30"),
}


def pull_weather(start_date: str, end_date: str, variables: list) -> pd.DataFrame:
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": config.TIMEZONE,
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def pull_forecast(days_ahead: int = 2) -> pd.DataFrame:
    """ used at prediction time"""
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": ",".join(config.WEATHER_FEATURES),
        "timezone": config.TIMEZONE,
        "forecast_days": days_ahead,
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index = df.index.tz_localize(
        config.TIMEZONE, nonexistent="shift_forward", ambiguous="NaT"
    )
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="first")]
    return df


def ingest_all(out_dir: str = config.RAW_DIR):
    os.makedirs(out_dir, exist_ok=True)
    for split_name, (start, end) in WEATHER_SPLITS.items():
        print(f"\n{split_name.upper()}: pulling {start} to {end}...")
        df = pull_weather(start, end, variables=config.WEATHER_FEATURES)
        out_path = f"{out_dir}/weather_{split_name}.csv"
        df.to_csv(out_path)
        print(f"  {len(df)} rows -> {out_path}")
        time.sleep(1)