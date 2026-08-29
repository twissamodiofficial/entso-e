"""
bootstrap_live_store.py

Cold-starts (or catches up) Supabase's live store (load_actuals, weather)
so the daily loop (run_daily_ingest.py -> run_predict.py) has enough
history to compute lag features on its first run.

Watermark-driven, not date-driven: reads the latest datetime already in
Supabase and backfills from there (or from the end of config.SPLITS["test"]
if Supabase is empty) through yesterday. This makes it idempotent and safe
to rerun - if it fails partway, or you run it twice by accident, it just
picks up wherever it left off instead of re-pulling months of data or
requiring you to remember not to run it again.

Run this once before the daily GitHub Actions workflows are enabled, or
any time the live store falls behind (e.g. after the daily cron was paused).
"""
import sys

import pandas as pd

from entso_e_pipeline import config, storage
from entso_e_pipeline.ingestion.load import _client as entsoe_client
from entso_e_pipeline.ingestion.weather import pull_weather

# Fallback start if Supabase has no data yet - the day after the test
# split ends, so live history picks up exactly where the frozen
# train/val/test dataset (run_batch_backfill.py) leaves off.
_, TEST_END = config.SPLITS["test"][0]
FALLBACK_START = pd.Timestamp(TEST_END, tz=config.TIMEZONE).strftime("%Y-%m-%d")


def backfill_load(start: pd.Timestamp, end: pd.Timestamp):
    if start >= end:
        print(f"load: already up to date (latest={start.date()}), skipping")
        return
    client = entsoe_client()
    raw = client.query_load(config.COUNTRY_CODE, start=start, end=end)
    hourly = raw.resample("h").mean()
    storage.upsert_load_actuals(hourly)
    print(f"load: {len(hourly)} rows -> Supabase load_actuals ({start.date()} to {end.date()})")


def backfill_weather(start: pd.Timestamp, end: pd.Timestamp):
    if start >= end:
        print(f"weather: already up to date (latest={start.date()}), skipping")
        return
    df = pull_weather(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), variables=config.WEATHER_FEATURES)
    storage.upsert_weather(df, source="actual")
    print(f"weather: {len(df)} rows -> Supabase weather (source=actual) ({start.date()} to {end.date()})")


if __name__ == "__main__":
    try:
        today = pd.Timestamp.now(tz=config.TIMEZONE).normalize()
        yesterday = today - pd.Timedelta(days=1)

        load_watermark = storage.latest_load_actuals_datetime()
        load_start = (load_watermark + pd.Timedelta(hours=1)) if load_watermark is not None \
            else pd.Timestamp(FALLBACK_START, tz=config.TIMEZONE)
        print(f"Backfilling load from {load_start.date()} through {yesterday.date()}")
        backfill_load(load_start, today)  # exclusive end, covers through yesterday

        weather_watermark = storage.latest_weather_datetime(source="actual")
        weather_start = (weather_watermark + pd.Timedelta(hours=1)) if weather_watermark is not None \
            else pd.Timestamp(FALLBACK_START, tz=config.TIMEZONE)
        print(f"Backfilling weather from {weather_start.date()} through {yesterday.date()}")
        backfill_weather(weather_start, yesterday)
    except Exception as e:
        print(f"BOOTSTRAP FAILED: {e}")
        sys.exit(1)

    print("Bootstrap OK: live store caught up through yesterday")
