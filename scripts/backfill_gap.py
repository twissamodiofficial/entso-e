import pandas as pd

from entso_e_pipeline import config, storage
from entso_e_pipeline.ingestion.load import _client as entsoe_client
from entso_e_pipeline.ingestion.weather import pull_weather

GAP_START = "2026-07-01"


def backfill_load(start: str, end: str):
    client = entsoe_client()
    raw = client.query_load(
        config.COUNTRY_CODE,
        start=pd.Timestamp(start, tz=config.TIMEZONE),
        end=pd.Timestamp(end, tz=config.TIMEZONE),
    )
    hourly = raw.resample("h").mean()
    storage.upsert_load_actuals(hourly)
    print(f"load: {len(hourly)} rows -> Supabase load_actuals")


def backfill_weather(start: str, end: str):
    df = pull_weather(start, end, variables=config.WEATHER_FEATURES)
    storage.upsert_weather(df, source="actual")
    print(f"weather: {len(df)} rows -> Supabase weather (source=actual)")


if __name__ == "__main__":
    today = pd.Timestamp.now(tz=config.TIMEZONE).normalize()
    yesterday = today - pd.Timedelta(days=1)

    load_end = today.strftime("%Y-%m-%d")           # exclusive end, covers through yesterday
    weather_end = yesterday.strftime("%Y-%m-%d")

    print(f"Backfilling load {GAP_START} to {load_end} (exclusive end, covers through yesterday)")
    backfill_load(GAP_START, load_end)

    print(f"Backfilling weather {GAP_START} to {weather_end}")
    backfill_weather(GAP_START, weather_end)