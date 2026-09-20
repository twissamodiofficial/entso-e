"""Attach completed actuals and daily metrics to stored forecast runs."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .. import config, preprocessing
from ..modeling import metrics
from ..storage.supabase import SupabaseRawStore
from ..time_utils import as_local, local_day_hours
from .ingest import ingest_load_catchup


def reconcile_forecast(forecast_date=None, *, store=None, model_version=None) -> dict:
    forecast_day = as_local(
        forecast_date or pd.Timestamp.now(tz=config.TIMEZONE) - pd.DateOffset(days=1),
        config.TIMEZONE,
    ).normalize()
    store = store or SupabaseRawStore()
    run = store.forecast_run_for_date(forecast_day.date(), model_version)
    if not run:
        raise ValueError(f"No forecast run found for {forecast_day.date()}.")

    actual_end = forecast_day + pd.DateOffset(days=1)
    ingest_load_catchup(actual_end, store)
    actual = preprocessing.prepare_load(
        store.load(forecast_day, actual_end),
        start=forecast_day,
        end=actual_end,
    )[config.TARGET]
    expected = local_day_hours(forecast_day, config.TIMEZONE)
    actual = actual.reindex(expected)
    if actual.isna().any():
        raise RuntimeError("Actual load is still incomplete for the forecast day.")

    forecasts = store.forecast_values(run["run_id"])
    if forecasts.empty:
        raise RuntimeError(f"Forecast run {run['run_id']} has no forecast values.")
    forecasts["actual_load_mw"] = actual.reindex(forecasts.index)
    forecasts["actual_observed_at"] = forecasts.index.where(
        forecasts["actual_load_mw"].notna(), pd.NaT
    )
    store.upsert_forecasts(run["run_id"], forecasts)

    evaluated = forecasts.dropna(subset=["actual_load_mw"])
    daily = pd.DataFrame(index=[forecast_day.date()])
    daily.index.name = "valid_date"
    daily["evaluated_rows"] = len(evaluated)
    actual_values = evaluated["actual_load_mw"]
    daily["point_mae_mw"] = metrics.mean_absolute_error(
        actual_values, evaluated["point_forecast_mw"]
    )
    daily["point_mape_percent"] = metrics.mean_absolute_percentage_error(
        actual_values, evaluated["point_forecast_mw"]
    )
    for level, column in ((0.1, "q10_forecast_mw"), (0.5, "q50_forecast_mw"), (0.9, "q90_forecast_mw")):
        daily[f"q{int(level * 100)}_pinball_loss"] = metrics.pinball_loss(
            actual_values, evaluated[column], level
        )
    daily["interval_coverage"] = metrics.interval_coverage(
        actual_values,
        evaluated["q10_forecast_mw"],
        evaluated["q90_forecast_mw"],
    )
    store.upsert_daily_metrics(run["run_id"], daily)
    return {
        "run_id": run["run_id"],
        "forecast_date": forecast_day.date().isoformat(),
        "evaluated_rows": len(evaluated),
        "point_mae_mw": float(daily.iloc[0]["point_mae_mw"]),
        "point_mape_percent": float(daily.iloc[0]["point_mape_percent"]),
        "interval_coverage": float(daily.iloc[0]["interval_coverage"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-date")
    parser.add_argument("--model-version")
    args = parser.parse_args(argv)
    print(json.dumps(reconcile_forecast(
        args.forecast_date,
        model_version=args.model_version,
    ), indent=2))


if __name__ == "__main__":
    main()
