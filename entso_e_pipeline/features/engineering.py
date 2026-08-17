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
    if load_prior is not None:
        stitched = pd.concat([load_prior.tail(config.WARMUP_HOURS), load_split])
    else:
        stitched = load_split

    df_ws = ws.transform(stitched)
    df_dtf = dtf.transform(df_ws)
    features = pd.concat([df_dtf, stitched], axis=1)
    features = _add_holiday_and_cast(features)
    features = features.loc[features.index >= load_split.index.min()]

    if weather_split is not None:
        features = pd.concat([features, weather_split], axis=1).dropna()
    return features
