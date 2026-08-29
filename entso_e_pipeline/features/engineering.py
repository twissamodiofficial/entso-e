import pandas as pd
from feature_engine.datetime import DatetimeFeatures
from sktime.transformations.summarize import WindowSummarizer

from .. import config


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["Unnamed: 0"], utc=True).dt.tz_convert(config.TIMEZONE)
    df = df.drop(columns=["Unnamed: 0"]).set_index("DateTime")
    return df.resample("h").mean()


def load_weather(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = df.index.tz_localize(
        config.TIMEZONE, nonexistent="shift_forward", ambiguous="NaT"
    )
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="first")]
    return df[config.WEATHER_FEATURES]


def make_transformers():
    ws = WindowSummarizer(
        target_cols=[config.TARGET],
        lag_feature={"lag": [24, 168], "mean": [[24, 24]]},
        n_jobs=1,
    )
    dtf = DatetimeFeatures(
        variables="index",
        features_to_extract=["hour", "day_of_week", "month", "weekend"],
    )
    return ws, dtf

def _category_dtypes() -> dict:
    dtypes = {col: pd.CategoricalDtype(categories=cats) for col, cats in config.FIXED_CATEGORIES.items()}
    holiday_names = sorted(set(config.COUNTRY_HOLIDAYS.values()) | {"none"})
    dtypes["holiday_name"] = pd.CategoricalDtype(categories=holiday_names)
    return dtypes

def _add_holiday_and_cast(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features["is_holiday"] = [int(d in config.COUNTRY_HOLIDAYS) for d in features.index.date]
    features["holiday_name"] = [config.COUNTRY_HOLIDAYS.get(d, "none") for d in features.index.date]

    for col, dtype in _category_dtypes().items():
        features[col] = features[col].where(features[col].isin(dtype.categories), "none" if col == "holiday_name" else dtype.categories[0])
        features[col] = features[col].astype(dtype)

    return features

def _build_feature_frame(stitched: pd.DataFrame, ws: WindowSummarizer, dtf: DatetimeFeatures) -> pd.DataFrame:
    """Core feature construction shared by every caller (train/val/test/live).
    No assumption about whether TARGET is real or a placeholder - callers
    decide what to do with that column and whether/how to drop NaNs."""
    df_ws = ws.transform(stitched)
    df_dtf = dtf.transform(df_ws)
    features = pd.concat([df_dtf, stitched], axis=1)
    return _add_holiday_and_cast(features)


def fit_transform_train(load_train: pd.DataFrame, weather_train: pd.DataFrame = None):
    ws, dtf = make_transformers()
    df_ws = ws.fit_transform(load_train)
    df_dtf = dtf.fit_transform(df_ws)
    features = pd.concat([df_dtf, load_train], axis=1)
    features = _add_holiday_and_cast(features)
    if weather_train is not None:
        features = pd.concat([features, weather_train], axis=1).dropna()
    return features, ws, dtf


def transform(load_split: pd.DataFrame, ws: WindowSummarizer, dtf: DatetimeFeatures,
               load_prior: pd.DataFrame = None,
               weather_split: pd.DataFrame = None) -> pd.DataFrame:
    """train/val/test: TARGET is known and real. A missing TARGET here means
    an actual gap in the data, so dropna() after the weather join is correct -
    it drops rows that genuinely can't be scored/trained on."""
    if load_prior is not None:
        stitched = pd.concat([load_prior.tail(config.WARMUP_HOURS), load_split])
    else:
        stitched = load_split

    features = _build_feature_frame(stitched, ws, dtf)
    features = features.loc[features.index >= load_split.index.min()]

    if weather_split is not None:
        features = pd.concat([features, weather_split], axis=1).dropna()
    return features


def transform_future(load_history: pd.DataFrame, horizon: pd.DatetimeIndex,
                      ws: WindowSummarizer, dtf: DatetimeFeatures,
                      weather_forecast: pd.DataFrame) -> pd.DataFrame:
    """Live prediction: TARGET doesn't exist yet for `horizon`, by definition -
    there is nothing to fabricate and nothing to drop it for. Lag/rolling
    features only ever look backward (they pull from load_history), so the
    NaN TARGET placeholder for the horizon rows is never used as an input;
    it just needs to be there so `stitched` has the right shape to run
    through the same _build_feature_frame() every other split uses -
    guaranteeing identical dtypes/categories/columns as train/val/test."""
    if len(load_history) < config.WARMUP_HOURS:
        raise ValueError(
            f"Need at least {config.WARMUP_HOURS} hours of load history to build "
            f"lag features, got {len(load_history)}."
        )

    placeholder = pd.DataFrame(index=horizon, columns=[config.TARGET], dtype=float)
    stitched = pd.concat([load_history.tail(config.WARMUP_HOURS), placeholder])

    features = _build_feature_frame(stitched, ws, dtf)
    features = features.loc[horizon].drop(columns=[config.TARGET])

    weather_target = weather_forecast.reindex(horizon)
    features = pd.concat([features, weather_target], axis=1)
    return features.dropna()
