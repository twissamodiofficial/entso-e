import os

import pandas as pd
from supabase import create_client

from . import config


def _client():
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def upsert_load_actuals(load: pd.DataFrame):
    client = _client()
    rows = load.reset_index().rename(columns={"index": "datetime", "Actual Load": "actual_load"})
    rows["datetime"] = rows["datetime"].astype(str)
    client.table("load_actuals").upsert(rows.to_dict(orient="records")).execute()


def fetch_load_actuals(start: str = None, end: str = None) -> pd.DataFrame:
    client = _client()
    query = client.table("load_actuals").select("*")
    if start:
        query = query.gte("datetime", start)
    if end:
        query = query.lte("datetime", end)
    result = query.execute()
    df = pd.DataFrame(result.data)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_convert(config.TIMEZONE)
        df = df.set_index("datetime").sort_index()
        df = df.rename(columns={"actual_load": "Actual Load"})
    return df


def upsert_weather(weather: pd.DataFrame, source: str):
    client = _client()
    rows = weather.reset_index().rename(columns={"index": "datetime", "time": "datetime"})
    rows["datetime"] = rows["datetime"].astype(str)
    rows["source"] = source
    client.table("weather").upsert(rows.to_dict(orient="records")).execute()


def fetch_weather(source: str, start: str = None, end: str = None) -> pd.DataFrame:
    client = _client()
    query = client.table("weather").select("*").eq("source", source)
    if start:
        query = query.gte("datetime", start)
    if end:
        query = query.lte("datetime", end)
    result = query.execute()
    df = pd.DataFrame(result.data)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_convert(config.TIMEZONE)
        df = df.set_index("datetime").sort_index().drop(columns=["source"])
    return df


def upsert_forecasts(preds: pd.DataFrame, forecast_made_at: pd.Timestamp):
    client = _client()
    rows = preds.reset_index().rename(columns={"index": "datetime"})
    rows["datetime"] = rows["datetime"].astype(str)
    rows["forecast_made_at"] = str(forecast_made_at)
    client.table("forecasts").upsert(rows.to_dict(orient="records")).execute()


def fetch_forecasts(start: str = None, end: str = None) -> pd.DataFrame:
    client = _client()
    query = client.table("forecasts").select("*")
    if start:
        query = query.gte("datetime", start)
    if end:
        query = query.lte("datetime", end)
    result = query.execute()
    df = pd.DataFrame(result.data)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_convert(config.TIMEZONE)
        df = df.set_index("datetime").sort_index()
    return df