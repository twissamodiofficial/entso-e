"""Run one live Amsterdam-day forecast from Supabase-backed inputs."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .. import config, preprocessing
from ..features import engineering
from ..storage.supabase import SupabaseRawStore
from ..time_utils import as_local, local_day_hours
from ..serving.model import load_dagshub_model, load_local_model
from .ingest import ensure_forecast_history, ingest_weather_window


def run_forecast(
    forecast_date=None,
    *,
    store: SupabaseRawStore | None = None,
    models_dir: str | None = None,
) -> dict:
    """Ingest, materialize, predict, and persist one forecast day."""

    forecast_start = as_local(
        forecast_date or pd.Timestamp.now(tz=config.TIMEZONE),
        config.TIMEZONE,
    ).normalize()
    horizon = local_day_hours(forecast_start, config.TIMEZONE)
    horizon_end = forecast_start + pd.DateOffset(days=1)
    history_start = forecast_start - pd.DateOffset(days=config.LOOKBACK_DAYS)
    store = store or SupabaseRawStore()

    load_rows = ensure_forecast_history(forecast_start, store)
    weather_rows = ingest_weather_window(forecast_start, horizon_end, store)

    load = preprocessing.prepare_load(
        store.load(history_start, forecast_start),
        start=history_start,
        end=forecast_start,
    )
    weather = preprocessing.prepare_weather(
        store.weather(forecast_start, horizon_end)
    )
    store.upsert_hourly_load(load)
    store.upsert_hourly_weather(weather)

    load_transformer, calendar_transformer = engineering.make_transformers()
    features = engineering.transform(
        load,
        horizon,
        load_transformer,
        calendar_transformer,
        weather,
    )
    features[config.TARGET] = float("nan")
    store.upsert_features(features)

    model = (
        load_local_model(models_dir)
        if models_dir
        else load_dagshub_model(store)
    )
    forecasts = model.predict(features.drop(columns=[config.TARGET]))
    existing_run = store.latest_forecast_run(
        model.model_version,
        forecast_start.date(),
    )
    run_id = (
        existing_run["run_id"]
        if existing_run
        else store.create_forecast_run(
            model.model_version,
            forecast_start.date(),
            horizon[0],
            horizon_end,
        )
    )
    forecast_rows = store.upsert_forecasts(run_id, forecasts)
    return {
        "forecast_date": forecast_start.date().isoformat(),
        "run_id": run_id,
        "model_version": model.model_version,
        "load_rows_fetched": load_rows,
        "weather_rows_fetched": weather_rows,
        "forecast_rows": forecast_rows,
        "interval_adjustment_mw": model.interval_adjustment_mw,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-date")
    parser.add_argument(
        "--models-dir",
        help="Use local model files; omit to load the registered DagsHub model.",
    )
    args = parser.parse_args(argv)
    print(json.dumps(run_forecast(args.forecast_date, models_dir=args.models_dir), indent=2))


if __name__ == "__main__":
    main()
