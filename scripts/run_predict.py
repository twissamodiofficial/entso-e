import sys

import pandas as pd

from entso_e_pipeline import config, storage, registry
from entso_e_pipeline.ingestion.weather import pull_forecast
from entso_e_pipeline.pipeline import ForecastPipeline


def build_prediction_features(load_history: pd.DataFrame, weather_forecast: pd.DataFrame,
                               ws, dtf) -> pd.DataFrame:
    origin = load_history.index.max()
    target_start = origin + pd.Timedelta(hours=1)
    target_hours = pd.date_range(target_start, periods=24, freq="h", tz=config.TIMEZONE)

    future = pd.DataFrame(index=target_hours, columns=[config.TARGET], dtype=float)
    stitched = pd.concat([load_history.tail(config.WARMUP_HOURS), future])

    df_ws = ws.transform(stitched)
    df_dtf = dtf.transform(df_ws)
    features = pd.concat([df_dtf, stitched], axis=1)
    features = features.reindex(target_hours)

    if features.isna().any().any():
        print("WARNING: missing values before holiday/weather join:")
        print(features.isna().sum()[features.isna().sum() > 0])

    features["is_holiday"] = [int(d in config.COUNTRY_HOLIDAYS) for d in features.index.date]
    features["holiday_name"] = [config.COUNTRY_HOLIDAYS.get(d, "none") for d in features.index.date]
    for col in config.CATEGORICAL_FEATURES:
        features[col] = features[col].astype("category")

    weather_target = weather_forecast.reindex(target_hours)
    features = pd.concat([features.drop(columns=[config.TARGET]), weather_target], axis=1)

    if features.isna().any().any():
        print("WARNING: dropping rows with missing values (likely weather forecast gaps):")
        print(features[features.isna().any(axis=1)])

    return features.dropna()


if __name__ == "__main__":
    try:
        registry.pull_latest_artifacts()

        window_start = (pd.Timestamp.now(tz=config.TIMEZONE) - pd.Timedelta(hours=config.WARMUP_HOURS + 24)).strftime("%Y-%m-%d")
        load_history = storage.fetch_load_actuals(start=window_start)
        if load_history.empty:
            raise RuntimeError("No actuals in Supabase - has run_ingest.py run at least once?")

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