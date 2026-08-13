"""
Reusable data-loading and feature-engineering pipeline for the ENTSO-E NL
load forecasting project.

ASSUMPTIONS:
  - Forecast origin = midnight. I.e. when generating a forecast, the
    previous day's data (through 23:00) is assumed fully available.
    This is what makes `lag_24` valid across an entire next-day horizon.
  - Resolution: hourly (raw ENTSO-E pull is 15-min, resampled to hourly
    mean here).
  - Country: NL only for now.
"""

import pandas as pd
import holidays
from sktime.transformations.summarize import WindowSummarizer
from feature_engine.datetime import DatetimeFeatures

COUNTRY_HOLIDAYS = holidays.Netherlands(years=range(2019, 2028))

CATEGORICAL_FEATURES = ["hour", "day_of_week", "month", "weekend", "is_holiday", "holiday_name"]

WEATHER_FEATURES = ["apparent_temperature", "dew_point_2m"]

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["Unnamed: 0"], utc=True).dt.tz_convert("Europe/Amsterdam")
    df = df.drop(columns=["Unnamed: 0"]).set_index("DateTime")
    load = df.resample("h").mean()
    return load

def load_weather(path: str) -> pd.DataFrame:
    """Read a raw Open-Meteo weather CSV and localize its tz-naive index
    to Europe/Amsterdam, handling both DST edge cases:
      - spring-forward gap (a hour that never happened, e.g. 2019-03-31
        02:00) -> shifted forward to the next valid time
      - fall-back ambiguity (an hour that happened twice, e.g.
        2019-10-27 02:00) -> marked NaT and dropped, rather than guessed
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = df.index.tz_localize(
        "Europe/Amsterdam", nonexistent="shift_forward", ambiguous="NaT"
    )
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="first")]
    return df[WEATHER_FEATURES]


def make_transformers():
    ws = WindowSummarizer(
        target_cols=["Actual Load"],
        lag_feature={"lag": [24, 168], "mean": [[24, 24]]},
        n_jobs=1,
    )
    dtf = DatetimeFeatures(
        variables="index",
        features_to_extract=["hour", "day_of_week", "month", "weekend"],
    )
    return ws, dtf


def _add_holiday_and_cast(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features["is_holiday"] = [int(d in COUNTRY_HOLIDAYS) for d in features.index.date]
    features["holiday_name"] = [COUNTRY_HOLIDAYS.get(d, "none") for d in features.index.date]
    for col in CATEGORICAL_FEATURES:
        features[col] = features[col].astype("category")
    return features.dropna()


def fit_transform_train(load_train: pd.DataFrame, weather_train: pd.DataFrame = None) -> tuple[pd.DataFrame, WindowSummarizer, DatetimeFeatures]:
    ws, dtf = make_transformers()
    df_ws = ws.fit_transform(load_train)
    df_dtf = dtf.fit_transform(df_ws)
    features = pd.concat([df_dtf, load_train], axis=1)
    features = _add_holiday_and_cast(features)
    if weather_train is not None:
        features = pd.concat([features, weather_train], axis=1).dropna()

    return features, ws, dtf



WARMUP_HOURS = 168


def transform(load_split: pd.DataFrame, ws: WindowSummarizer, dtf: DatetimeFeatures,
              load_prior: pd.DataFrame = None, weather_split: pd.DataFrame = None) -> pd.DataFrame:
    if load_prior is not None:
        stitched = pd.concat([load_prior.tail(WARMUP_HOURS), load_split])
    else:
        stitched = load_split

    df_ws = ws.transform(stitched)
    df_dtf = dtf.transform(df_ws)
    features = pd.concat([df_dtf, stitched], axis=1)
    features = _add_holiday_and_cast(features)

    # trim back to only this split's own timestamps
    features = features.loc[features.index >= load_split.index.min()]
    if weather_split is not None:
        features = pd.concat([features, weather_split], axis=1).dropna()
    return features


if __name__ == "__main__":
    load_train = load_and_clean("data/raw/load_train.csv")
    load_val = load_and_clean("data/raw/load_val.csv")
    load_test = load_and_clean("data/raw/load_test.csv")

    weather_train = load_weather("data/raw/weather_train.csv")
    weather_val = load_weather("data/raw/weather_val.csv")
    weather_test = load_weather("data/raw/weather_test.csv")

    features_train, ws, dtf = fit_transform_train(load_train, weather_train)
    features_val = transform(load_val, ws, dtf, load_prior=load_train, weather_split=weather_val)
    features_test = transform(load_test, ws, dtf, load_prior=load_val, weather_split=weather_test)

    for name, df in [("train", features_train), ("val", features_val), ("test", features_test)]:
        df.to_parquet(f"data/processed/features_weather_{name}.parquet")
        print(f"{name}: {df.shape} -> data/processed/features_weather_{name}.parquet")