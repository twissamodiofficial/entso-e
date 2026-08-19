import os

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

from . import config

load_dotenv()

PAGE_SIZE = 1000


def _client():
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def _paginated_fetch(query_builder):
    client = _client()
    all_rows = []
    offset = 0
    while True:
        query = query_builder(client).range(offset, offset + PAGE_SIZE - 1)
        result = query.execute()
        rows = result.data
        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return all_rows


def upsert_load_actuals(load: pd.DataFrame):
    client = _client()
    rows = load.reset_index().rename(columns={"index": "datetime", "Actual Load": "actual_load"})
    rows["datetime"] = rows["datetime"].astype(str)
    client.table("load_actuals").upsert(rows.to_dict(orient="records")).execute()


def fetch_load_actuals(start: str = None, end: str = None) -> pd.DataFrame:
    def build(client):
        q = client.table("load_actuals").select("*").order("datetime")
        if start:
            q = q.gte("datetime", start)
        if end:
            q = q.lte("datetime", end)
        return q

    data = _paginated_fetch(build)
    df = pd.DataFrame(data)
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
    def build(client):
        q = client.table("weather").select("*").eq("source", source).order("datetime")
        if start:
            q = q.gte("datetime", start)
        if end:
            q = q.lte("datetime", end)
        return q

    data = _paginated_fetch(build)
    df = pd.DataFrame(data)
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
    def build(client):
        q = client.table("forecasts").select("*").order("datetime")
        if start:
            q = q.gte("datetime", start)
        if end:
            q = q.lte("datetime", end)
        return q

    data = _paginated_fetch(build)
    df = pd.DataFrame(data)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_convert(config.TIMEZONE)
        df = df.set_index("datetime").sort_index()
    return df


def upsert_daily_metrics(row: dict):
    client = _client()
    client.table("daily_metrics").upsert(row).execute()


def fetch_daily_metrics(start: str = None, end: str = None) -> pd.DataFrame:
    def build(client):
        q = client.table("daily_metrics").select("*").order("date")
        if start:
            q = q.gte("date", start)
        if end:
            q = q.lte("date", end)
        return q

    data = _paginated_fetch(build)
    df = pd.DataFrame(data)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    return df