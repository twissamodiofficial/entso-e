import os
import time

import pandas as pd
import requests

OUT_DIR = "data/raw"
os.makedirs(OUT_DIR, exist_ok=True)

# Amsterdam coordinates
LATITUDE = 52.37
LONGITUDE = 4.90

# most probable weather variables to pull from Open-Meteo
HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "surface_pressure",
]

SPLITS = {
    "train": ("2019-01-01", "2025-12-31"),
    "val": ("2026-01-01", "2026-03-31"),
    "test": ("2026-04-01", "2026-06-30"),
}

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def pull_weather(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Europe/Amsterdam",
    }
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    return df


if __name__ == "__main__":
    for split_name, (start, end) in SPLITS.items():
        print(f"\n{split_name.upper()}: pulling {start} to {end}...")
        df = pull_weather(start, end)
        out_path = f"{OUT_DIR}/weather_{split_name}.csv"
        df.to_csv(out_path)
        print(f"  {len(df)} rows -> {out_path}")
        time.sleep(1)