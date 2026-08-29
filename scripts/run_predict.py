import sys

import pandas as pd

from entso_e_pipeline import config, storage, registry
from entso_e_pipeline.features import engineering
from entso_e_pipeline.ingestion.weather import pull_forecast
from entso_e_pipeline.pipeline import ForecastPipeline


def build_prediction_features(load_history: pd.DataFrame, weather_forecast: pd.DataFrame,
                               ws, dtf) -> pd.DataFrame:
    origin = load_history.index.max()
    target_start = origin + pd.Timedelta(hours=1)
    target_hours = pd.date_range(target_start, periods=24, freq="h", tz=config.TIMEZONE)

    features = engineering.transform_future(load_history, target_hours, ws, dtf, weather_forecast)

    missing_hours = target_hours.difference(features.index)
    if len(missing_hours) > 0:
        print(f"WARNING: dropping {len(missing_hours)} hour(s) with missing weather forecast: "
              f"{list(missing_hours)}")

    return features


if __name__ == "__main__":
    try:
        registry.pull_latest_artifacts()

        window_start = pd.Timestamp.now(tz=config.TIMEZONE) - pd.Timedelta(hours=config.WARMUP_HOURS + 24)
        load_history = storage.fetch_load_actuals(start=str(window_start))
        if load_history.empty:
            raise RuntimeError("No actuals in Supabase - has run_daily_ingest.py run at least once?")

        weather_forecast = pull_forecast(days_ahead=2)

        pipeline = ForecastPipeline()
        pipeline.load_models()

        features = build_prediction_features(load_history, weather_forecast, pipeline.ws, pipeline.dtf)
        if features.empty:
            raise RuntimeError("No usable prediction rows - check weather forecast coverage")

        preds = pipeline.predict(features)

        forecast_made_at = pd.Timestamp.now(tz=config.TIMEZONE)
        storage.upsert_forecasts(preds, forecast_made_at)
        storage.upsert_weather(weather_forecast.reindex(features.index), source="forecast")
    except Exception as e:
        print(f"PREDICT FAILED: {e}")
        sys.exit(1)

    print(f"Predict OK: wrote {len(preds)} forecast rows")