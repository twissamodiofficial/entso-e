import sys

import pandas as pd

from entso_e_pipeline import config, storage
from entso_e_pipeline.ingestion.load import _client as entsoe_client
from entso_e_pipeline.ingestion.weather import pull_weather
from entso_e_pipeline.evaluation import metrics
from entso_e_pipeline.modeling import calibration


def fetch_yesterday_load() -> pd.DataFrame:
    client = entsoe_client()
    end = pd.Timestamp.now(tz=config.TIMEZONE).normalize()
    start = end - pd.Timedelta(days=1)
    raw = client.query_load(config.COUNTRY_CODE, start=start, end=end)
    return raw.resample("h").mean()


def fetch_yesterday_weather() -> pd.DataFrame:
    end = pd.Timestamp.now(tz=config.TIMEZONE).normalize()
    start = end - pd.Timedelta(days=1)
    return pull_weather(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), variables=config.WEATHER_FEATURES)


def score_yesterday(yesterday_load: pd.DataFrame):
    day = yesterday_load.index[0].date()
    start = str(yesterday_load.index.min())
    end = str(yesterday_load.index.max())

    yesterday_forecast = storage.fetch_forecasts(start=start, end=end)
    if yesterday_forecast.empty:
        print(f"No forecast found for {day}, skipping scoring")
        return

    joined = yesterday_forecast.join(yesterday_load[config.TARGET], how="inner")
    if joined.empty:
        print(f"Forecast/actual join empty for {day}, skipping scoring")
        return

    y_true = joined[config.TARGET]
    row = {
        "date": str(day),
        "period_start": start,
        "period_end": end,
        "mae": float(metrics.mae(y_true, joined["point_pred"])),
        "rmse": float(metrics.rmse(y_true, joined["point_pred"])),
        "mape": float(metrics.mape(y_true, joined["point_pred"])),
        "coverage": float(calibration.coverage(y_true, joined["q10_calibrated"], joined["q90_calibrated"])),
        "n_hours": len(joined),
    }
    storage.upsert_daily_metrics(row)
    print(f"Scored {day}: MAPE={row['mape']:.2f}%  coverage={row['coverage']*100:.1f}%")


if __name__ == "__main__":
    try:
        yesterday_load = fetch_yesterday_load()
        yesterday_weather = fetch_yesterday_weather()

        storage.upsert_load_actuals(yesterday_load)
        storage.upsert_weather(yesterday_weather, source="actual")
        score_yesterday(yesterday_load)
    except Exception as e:
        print(f"INGEST FAILED: {e}")
        sys.exit(1)

    print(f"Ingest OK: {len(yesterday_load)} load rows, {len(yesterday_weather)} weather rows")